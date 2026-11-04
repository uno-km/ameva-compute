/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ameva_sd_api.h
 * @brief ameva-android v0.4: minimal in-process "load a diffusion model, generate an image" C API.
 *
 * Wraps stable-diffusion.cpp (third_party/stable-diffusion.cpp, pinned commit 50d6405, with its
 * own pinned ggml fork 8e800cef -- not llama.cpp's or whisper.cpp's ggml; this one requires
 * GGML_MAX_NAME=160, an ABI-level difference, confirmed by its own static_assert) + the
 * ggml-ameva GPU backend (adapters/llama/ggml-ameva -- the SAME backend AmevaLlm loads, already
 * implementing GGML_OP_CONV_2D/GROUP_NORM/UPSCALE/TIMESTEP_EMBEDDING), loaded as a dynamic plugin
 * exactly like the llama path (ggml_backend_load against a path, not a static link).
 *
 * stable-diffusion.cpp resolves its `backend` ctx-param by NAME against whatever devices are
 * registered in ggml's backend registry at the time new_sd_ctx() runs (src/core/
 * ggml_extend_backend.cpp's sd_backend_resolve_name/init_named_backend) -- so loading the AMEVA
 * plugin ourselves before new_sd_ctx() and passing backend="AMEVA0" is enough; no sd.cpp source
 * patch is needed for backend selection itself.
 */

#ifndef AMEVA_SD_API_H
#define AMEVA_SD_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AmevaSd AmevaSd;

typedef enum {
    AMEVA_SD_OK = 0,
    AMEVA_SD_ERR_BACKEND_LOAD_FAILED,
    AMEVA_SD_ERR_MODEL_LOAD_FAILED,
    AMEVA_SD_ERR_GENERATE_FAILED,
    AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL,
    AMEVA_SD_ERR_INVALID_ARGUMENT,
    /** AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY requested, but the model's file size (or device memory)
     *  could not be fit within the budget. Refused outright, before new_sd_ctx() runs --
     *  GPU_ONLY never spills onto CPU, silently or otherwise. */
    AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED,
    /** ameva_sd_cancel() was called while ameva_sd_generate() was running. No image is written. */
    AMEVA_SD_ERR_CANCELLED,
    /** An options struct carried a struct_size this library does not know (caller built against a
     *  different header version). Refused instead of reading past the caller's struct. */
    AMEVA_SD_ERR_ABI_MISMATCH
} AmevaSdStatus;

/**
 * @brief How ameva_sd_load() sizes the GPU (ggml-ameva/"AMEVA0") placement against currently free
 *        device memory. Opt-in only: a NULL AmevaSdGpuBudgetOptions* (or AMEVA_SD_GPU_BUDGET_MODE_
 *        DEFAULT) keeps the previous fixed behavior -- the whole model unconditionally placed on
 *        "AMEVA0", which can OOM-kill the whole process if it does not fit (same risk pattern
 *        confirmed on the LLM path's unconditional n_gpu_layers=-1; see ameva_llm_api.h's
 *        AmevaGpuBudgetOptions, which this mirrors).
 *
 * GPU_THEN_CPU is implemented by translating budget_fraction into stable-diffusion.cpp's own
 * sd_ctx_params_t.auto_fit + max_vram reserve-GiB mechanism (third_party/stable-diffusion.cpp's
 * sd::backend_fit, already used by the CLI's --auto-fit) -- its own component-level (DiT/VAE/
 * conditioner) placement decides the split, and it logs its plan (LOG_INFO "auto-fit plan...").
 * GPU_ONLY is a coarser, file-size-based pre-check (this wrapper does not call backend_fit's
 * per-component estimator directly): if the model file is larger than the budget, the load is
 * refused before new_sd_ctx() runs at all.
 */
typedef enum {
    AMEVA_SD_GPU_BUDGET_MODE_DEFAULT = 0,   // no budget-fit: unconditional "AMEVA0" (previous behavior)
    AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY,      // fit the whole model in budget, or fail -- never spills to CPU
    AMEVA_SD_GPU_BUDGET_MODE_GPU_THEN_CPU,  // sd.cpp's own auto-fit component placement; logged, never silent
    AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY       // run entirely on CPU; the AMEVA plugin is not placed at all
} AmevaSdGpuBudgetMode;

typedef struct {
    AmevaSdGpuBudgetMode mode;
    /** Fraction of the device's currently free memory the budget may use, e.g. 0.95 for 95%.
     *  0.0f means "use the default" (0.95). Ignored when mode is DEFAULT or CPU_ONLY. */
    float budget_fraction;
} AmevaSdGpuBudgetOptions;

/**
 * @param model_path Filesystem path to a single-file GGUF/safetensors/ckpt diffusion checkpoint
 *        (not bundled with this library).
 * @param backend_plugin_path Filesystem path to libggml-ameva-sd.so.
 * @param cache_dir Per-app-writable directory (Android: Context.filesDir) used to derive the
 *        AMEVA_PIPELINE_CACHE / AMEVA_QMV_TUNE_FILE paths ggml-ameva otherwise falls back to
 *        $HOME for (same gap AmevaLlm.load() already closes -- AmevaDiffusion never had it in
 *        v0.3.0, which is most of why its first generation took minutes: shader pipeline compiles
 *        and GEMV layout auto-tuning for every one of the UNet's distinct weight shapes happened
 *        from scratch, with nowhere to persist the result for the next call). NULL/empty leaves
 *        both off.
 * @param n_threads CPU thread count for whatever this model leaves on CPU (text encoder tokenize,
 *        any op ggml-ameva declines); 0 uses stable-diffusion.cpp's own default.
 * @param gpu_budget_opts NULL: previous fixed behavior (unconditional "AMEVA0", no budget-fit).
 *        Set to size the GPU placement against free device memory instead -- see
 *        AmevaSdGpuBudgetOptions.
 */
AmevaSdStatus ameva_sd_load(
    const char* model_path,
    const char* backend_plugin_path,
    const char* cache_dir,
    int n_threads,
    const AmevaSdGpuBudgetOptions* gpu_budget_opts,
    AmevaSd** out_sd,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief Run one txt2img generation and write the first resulting image as packed RGB8.
 *
 * @param out_rgb Destination buffer; must be at least width*height*3 bytes. Written only on
 *        AMEVA_SD_OK -- a failed or too-small-buffer call leaves it untouched.
 * @param out_rgb_size Size of out_rgb; AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL if it does not match
 *        width*height*3 exactly (no silent crop/stride mismatch).
 */
AmevaSdStatus ameva_sd_txt2img(
    AmevaSd* sd,
    const char* prompt,
    int width,
    int height,
    int steps,
    int64_t seed,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief ameva_sd_txt2img with an explicit classifier-free-guidance scale.
 *
 * ameva_sd_txt2img leaves guidance at stable-diffusion.cpp's default (7.0). Guidance-distilled
 * models (SDXS, SD-Turbo, LCM and similar one- to four-step models) are trained for a guidance
 * scale of about 1.0; at 7.0 they produce burnt, over-saturated images -- not a backend defect.
 *
 * @param cfg_scale Guidance scale. A value <= 0 keeps stable-diffusion.cpp's default (7.0); a
 *        non-finite value is rejected with AMEVA_SD_ERR_INVALID_ARGUMENT. All other parameters
 *        are as for ameva_sd_txt2img.
 */
AmevaSdStatus ameva_sd_txt2img_ex(
    AmevaSd* sd,
    const char* prompt,
    int width,
    int height,
    int steps,
    float cfg_scale,
    int64_t seed,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
);

/* ------------------------------------------------------------------------------------------------
 * Extended API: multi-file models, component placement, img2img, progress, cancel, duty cycle.
 * The functions above are kept unchanged for existing callers.
 * ---------------------------------------------------------------------------------------------- */

/** Where a model component runs. Ignored when budget_mode is GPU_THEN_CPU (sd.cpp's auto-fit
 *  decides and logs the placement) or CPU_ONLY (everything runs on the CPU). */
typedef enum {
    AMEVA_SD_PLACEMENT_GPU = 0,
    AMEVA_SD_PLACEMENT_CPU = 1
} AmevaSdPlacement;

/** Where the diffusion model's weights live between uses. Ignored under GPU_THEN_CPU (auto-fit
 *  picks the residency itself and switches to on-disk residency when the model does not fit). */
typedef enum {
    /** Weights stay allocated for the context's lifetime (fastest repeat generations). */
    AMEVA_SD_PARAMS_RESIDENT = 0,
    /** Weights are re-read from the model file for each use and released afterwards
     *  (sd.cpp params backend "disk"). Lowest steady-state memory, slower per generation. */
    AMEVA_SD_PARAMS_DISK = 1
} AmevaSdParamsResidency;

typedef struct {
    /** Must be sizeof(AmevaSdLoadOptions); set by ameva_sd_load_options_init(). */
    uint32_t struct_size;
    /** Single-file checkpoint (SD 1.x / 2.x / SDXL / SD-Turbo / SDXS). NULL when the model is
     *  split into components (set diffusion_model_path instead). */
    const char* model_path;
    /** Standalone diffusion weights (Z-Image, Flux, ...). */
    const char* diffusion_model_path;
    /** LLM text encoder (Z-Image: Qwen3). */
    const char* llm_path;
    const char* clip_l_path;
    const char* t5xxl_path;
    const char* vae_path;
    /** Tiny autoencoder used instead of the full VAE for decoding (faster, lower memory). */
    const char* taesd_path;
    /** libggml-ameva-sd.so. Required unless budget_mode is CPU_ONLY. */
    const char* backend_plugin_path;
    /** See ameva_sd_load(). NULL/empty disables the pipeline/tuning caches. */
    const char* cache_dir;
    /** 0: stable-diffusion.cpp default. */
    int n_threads;
    AmevaSdGpuBudgetMode budget_mode;
    /** See AmevaSdGpuBudgetOptions.budget_fraction. */
    float budget_fraction;
    AmevaSdPlacement text_encoder_placement;
    AmevaSdPlacement vae_placement;
    AmevaSdParamsResidency diffusion_params;
    /** Non-zero: memory-map the model files (lower peak RSS during load). Default 1. */
    int enable_mmap;
    /** CLIP-Vision (ViT-H/14) image encoder; required with ip_adapter_path. */
    const char* clip_vision_path;
    /** IP-Adapter weights (SD 1.5 / SDXL, classic or Plus). Enables AmevaSdGenerateParams.ip_adapter_*. */
    const char* ip_adapter_path;
    /** ControlNet weights. Enables AmevaSdGenerateParams.control_*. */
    const char* control_net_path;
    /** AnimateDiff motion module (SD 1.5). Enables ameva_sd_generate_video(). */
    const char* motion_module_path;
    /** PhotoMaker weights (SDXL). */
    const char* photo_maker_path;
} AmevaSdLoadOptions;

/** Fills defaults: struct_size, budget_mode DEFAULT, budget_fraction 0.95, GPU placements,
 *  resident params, enable_mmap 1. Everything else zero/NULL. */
void ameva_sd_load_options_init(AmevaSdLoadOptions* opts);

/**
 * @brief Load a (possibly multi-file) diffusion model. At least one of model_path or
 *        diffusion_model_path must be set. GPU_ONLY sums the sizes of every given file for its
 *        budget check.
 */
AmevaSdStatus ameva_sd_load_ex(
    const AmevaSdLoadOptions* opts,
    AmevaSd** out_sd,
    char* err_buf,
    size_t err_buf_size
);

/** Called once per sampling step from the generating thread. [seconds_for_step] is the measured
 *  wall time of that step, excluding any duty-cycle idle time. */
typedef void (*AmevaSdProgressFn)(int step, int steps, float seconds_for_step, void* user_data);

typedef struct {
    /** Must be sizeof(AmevaSdGenerateParams); set by ameva_sd_generate_params_init(). */
    uint32_t struct_size;
    const char* prompt;
    /** NULL or "" for none. */
    const char* negative_prompt;
    int width;
    int height;
    /** <= 0: 20. */
    int steps;
    /** <= 0: stable-diffusion.cpp default (7.0). Non-finite: AMEVA_SD_ERR_INVALID_ARGUMENT. */
    float cfg_scale;
    /** <= 0: model default. Flow models (Z-Image, SD3, Flux) only. */
    float flow_shift;
    /** stable-diffusion.cpp sampler name (euler, euler_a, dpm++2m, lcm, res_multistep, ...).
     *  NULL/"": model default. Unknown name: AMEVA_SD_ERR_INVALID_ARGUMENT. */
    const char* sampler;
    /** stable-diffusion.cpp scheduler name (discrete, karras, simple, sgm_uniform, ...).
     *  NULL/"": model default. Unknown name: AMEVA_SD_ERR_INVALID_ARGUMENT. */
    const char* scheduler;
    /** -1: random. */
    int64_t seed;
    /** img2img source, packed RGB8 of exactly width*height*3 bytes. NULL: txt2img. */
    const uint8_t* init_rgb;
    size_t init_rgb_size;
    /** img2img denoising strength in (0, 1]; ignored for txt2img. */
    float strength;
    /** Non-zero: decode the latent in tiles (bounded VAE memory for large images). */
    int vae_tiling;
    /** 1..100. Below 100 the generating thread idles after each sampling step for
     *  (100 - d) / d of that step's measured time, so the GPU is busy at most d percent of the
     *  sampling phase. Keeps the display compositor and thermals responsive. 100: no idling. */
    int duty_cycle_percent;
    AmevaSdProgressFn progress;
    void* progress_user_data;
    /** LoRA files and their multipliers (lora_count entries each). Applied for this call only. */
    const char* const* lora_paths;
    const float* lora_multipliers;
    int lora_count;
    /** <= 0: model default. 2 = skip the last CLIP layer (common for anime SD 1.5 models). */
    int clip_skip;
    /** ControlNet hint, packed RGB8 of exactly width*height*3 bytes. Needs control_net_path at load. */
    const uint8_t* control_rgb;
    size_t control_rgb_size;
    /** <= 0: 0.9. */
    float control_strength;
    /** Non-zero: run Canny edge detection on control_rgb first (for canny ControlNets). */
    int control_canny;
    /** IP-Adapter reference image, packed RGB8 of ip_adapter_width*ip_adapter_height*3 bytes (any size).
     *  Needs ip_adapter_path and clip_vision_path at load. */
    const uint8_t* ip_adapter_rgb;
    int ip_adapter_width;
    int ip_adapter_height;
    /** <= 0: 1.0. 0.6-0.8 lets the text prompt keep more control. */
    float ip_adapter_strength;
    /** Hires fix (second pass at a larger size). <= 1: off. Output size is width*scale x height*scale
     *  (rounded down to a multiple of 8); out_rgb must match that size. */
    float hires_scale;
    /** <= 0: same as steps. */
    int hires_steps;
    /** <= 0: 0.5. */
    float hires_denoise;
    /** sd.cpp hires upscaler name (latent, latent_nearest, latent_bicubic, lanczos, ...); NULL/"": latent. */
    const char* hires_upscaler;
} AmevaSdGenerateParams;

/** Fills defaults: struct_size, 512x512, 20 steps, seed -1, strength 0.75, duty 100. */
void ameva_sd_generate_params_init(AmevaSdGenerateParams* params);

/**
 * @brief Generate one image (txt2img, or img2img when init_rgb is set) as packed RGB8.
 *        Only one generation runs per process at a time (stable-diffusion.cpp's progress hook is
 *        process-global); a concurrent call blocks until the running one returns.
 */
AmevaSdStatus ameva_sd_generate(
    AmevaSd* sd,
    const AmevaSdGenerateParams* params,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief AnimateDiff video: [frames] frames of width*height packed RGB8, written back to back into
 *        out_rgb (exactly frames*width*height*3 bytes). Needs motion_module_path at load. Uses the
 *        same prompt/sampler/LoRA/progress/duty fields as ameva_sd_generate (init/control/IP-Adapter
 *        fields are not used). *out_frames receives the number of frames written.
 */
AmevaSdStatus ameva_sd_generate_video(
    AmevaSd* sd,
    const AmevaSdGenerateParams* params,
    int frames,
    int fps,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    int* out_frames,
    char* err_buf,
    size_t err_buf_size
);

/** ESRGAN upscaler (RealESRGAN .pth/.safetensors/.gguf). */
typedef struct AmevaSdUpscaler AmevaSdUpscaler;

/** backend_plugin_path NULL/empty: CPU. tile_size <= 0: 128. */
AmevaSdStatus ameva_sd_upscaler_load(
    const char* model_path,
    const char* backend_plugin_path,
    int n_threads,
    int tile_size,
    AmevaSdUpscaler** out_upscaler,
    char* err_buf,
    size_t err_buf_size
);

/** Upscale factor of the loaded model (e.g. 4), or 0 for NULL. */
int ameva_sd_upscaler_factor(const AmevaSdUpscaler* upscaler);

/** out_rgb must be exactly (width*f)*(height*f)*3 bytes where f = ameva_sd_upscaler_factor(). */
AmevaSdStatus ameva_sd_upscale(
    AmevaSdUpscaler* upscaler,
    const uint8_t* rgb,
    int width,
    int height,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
);

void ameva_sd_upscaler_free(AmevaSdUpscaler* upscaler);

/** Where the loaded model's weights live, as stable-diffusion.cpp reported it while loading, e.g.
 *  "total params memory size = 1720.26MB (VRAM 159.68MB, RAM 1560.57MB): text_encoders 125.20MB(RAM),
 *  diffusion_model 1435.37MB(RAM), vae 159.68MB(VRAM), ...". A module marked (RAM) runs on the CPU even when
 *  the GPU was requested (GPU_THEN_CPU placement); callers should show that instead of reporting a GPU run.
 *  Owned by [sd]; valid until ameva_sd_free(). Empty when the line was not printed. */
const char* ameva_sd_placement(const AmevaSd* sd);

/** Request that the running ameva_sd_generate() on [sd] stop as soon as possible; it then returns
 *  AMEVA_SD_ERR_CANCELLED. Safe to call from any thread, and when nothing is running (no effect). */
void ameva_sd_cancel(AmevaSd* sd);

/** @brief Release the context, model, and backend registration. Safe to call with NULL. */
void ameva_sd_free(AmevaSd* sd);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_SD_API_H */
