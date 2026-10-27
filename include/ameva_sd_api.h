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
    AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED
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

/** @brief Release the context, model, and backend registration. Safe to call with NULL. */
void ameva_sd_free(AmevaSd* sd);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_SD_API_H */
