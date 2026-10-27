/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ameva_llm_api.h
 * @brief ameva-android v0.2: minimal in-process "load model, generate text" C API.
 *
 * Wraps llama.cpp (third_party/llama.cpp, pinned tag b8471) + the ggml-ameva GPU backend
 * (adapters/llama/ggml-ameva), loaded as a dynamic plugin exactly as it is in the Termux
 * deployment (ggml_backend_load against a path, not a static link). No upstream ggml-vulkan
 * backend is ever loaded here -- only ggml-ameva and llama.cpp's always-present CPU backend --
 * so an op ggml-ameva declines falls back to CPU, never to the known-corrupt upstream Vulkan
 * flash-attention path (see docs/handover notes on llama-bundle-vulkan-defects). Flash attention
 * is also explicitly disabled in the context params as defense in depth.
 *
 * Every failure is reported with a distinct code and a real message -- no silent CPU fallback
 * if the GPU backend plugin fails to load, no placeholder output.
 */

#ifndef AMEVA_LLM_API_H
#define AMEVA_LLM_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AmevaLlm AmevaLlm;

typedef enum {
    AMEVA_LLM_OK = 0,
    AMEVA_LLM_ERR_BACKEND_LOAD_FAILED,
    AMEVA_LLM_ERR_BACKEND_NO_DEVICE,
    AMEVA_LLM_ERR_MODEL_LOAD_FAILED,
    AMEVA_LLM_ERR_CONTEXT_CREATE_FAILED,
    AMEVA_LLM_ERR_TOKENIZE_FAILED,
    AMEVA_LLM_ERR_DECODE_FAILED,
    AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL,
    AMEVA_LLM_ERR_INVALID_ARGUMENT,
    AMEVA_LLM_ERR_VISION_LOAD_FAILED,
    AMEVA_LLM_ERR_VISION_NOT_LOADED,
    AMEVA_LLM_ERR_IMAGE_LOAD_FAILED,
    AMEVA_LLM_ERR_VISION_TOKENIZE_FAILED,
    AMEVA_LLM_ERR_VISION_ENCODE_FAILED,
    /** GPU-only backend ("vulkan"/"opencl", or the default) requested, but the model requires an
     *  op ggml-ameva does not implement (e.g. M-RoPE/vision rope for Qwen2-VL). Refused outright
     *  instead of silently handing that op to the CPU backend. Use backend="auto" to allow a
     *  visible, logged per-op CPU fallback, or backend="cpu" to run entirely on CPU. */
    AMEVA_LLM_ERR_GPU_ONLY_UNSUPPORTED_OP
} AmevaLlmStatus;

/**
 * @brief Load the ggml-ameva backend plugin and a GGUF model, and create a context.
 *
 * @param model_path Filesystem path to a .gguf model (not bundled with this library).
 * @param backend_plugin_path Filesystem path to libggml-ameva.so (the app's own native library
 *        directory at runtime -- Context.applicationInfo.nativeLibraryDir on the Kotlin side).
 * @param n_ctx Context size in tokens (0: use the model's own default).
 * @param out_llm Output handle, guaranteed NULL on failure.
 * @param err_buf Destination for a human-readable failure reason; always NUL-terminated.
 * @param err_buf_size Size of err_buf.
 * @return AMEVA_LLM_OK on success, an error code otherwise.
 */
/**
 * @param cache_dir Per-app-writable directory (Android: Context.filesDir) used to derive the
 *        AMEVA_PIPELINE_CACHE and AMEVA_QMV_TUNE_FILE paths ggml-ameva otherwise falls back to
 *        $HOME for -- which does not exist in an app process, silently turning both off (pipeline
 *        recompilation every cold start, GEMV auto-tuning measurements never kept). NULL/empty
 *        leaves both off, matching the previous behavior.
 * @param backend "cpu", "vulkan", "opencl", or "auto" (NULL/empty: "vulkan", matching the
 *        previous fixed behavior). "auto" runs ameva_backend_bench() against all three and
 *        loads whichever is actually fastest on this device -- not a fixed cpu-then-gpu rule.
 *        Benchmarking "vulkan"/"opencl" happens in short-lived forked helper processes (see
 *        ameva_backend_bench.h); the real model load below always happens in this process.
 */
AmevaLlmStatus ameva_llm_load(
    const char* model_path,
    const char* backend_plugin_path,
    const char* cache_dir,
    const char* backend,
    uint32_t n_ctx,
    AmevaLlm** out_llm,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief Tokenize prompt, run the decode loop, and sample text with the given parameters.
 *
 * @param temperature Sampling temperature (0.0f: effectively greedy via llama_sampler_init_temp).
 * @param seed RNG seed for the distribution sampler -- deterministic for a fixed seed.
 * @param out_buf Destination for the generated text (NUL-terminated on success).
 * @param out_buf_size Size of out_buf; AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL if generation would
 *        not fit (nothing partial is written in that case).
 * @return AMEVA_LLM_OK on success, an error code otherwise.
 */
AmevaLlmStatus ameva_llm_generate(
    AmevaLlm* llm,
    const char* prompt,
    int32_t max_tokens,
    float temperature,
    uint64_t seed,
    char* out_buf,
    size_t out_buf_size,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief Clear the KV cache and reset the decode position to 0, keeping the loaded model and
 *        backend. Use this between independent conversations/summaries instead of
 *        ameva_llm_free() + ameva_llm_load() -- the model weights (and, on a GPU backend,
 *        pipeline/tuning state) stay resident, so the next ameva_llm_generate() call starts a
 *        fresh context without the multi-hundred-ms-to-seconds reload cost.
 *
 * @return AMEVA_LLM_OK on success, AMEVA_LLM_ERR_INVALID_ARGUMENT if llm is NULL.
 */
AmevaLlmStatus ameva_llm_reset(
    AmevaLlm* llm,
    char* err_buf,
    size_t err_buf_size
);

/** @brief Release the context, model, and backend registration. Safe to call with NULL. */
void ameva_llm_free(AmevaLlm* llm);

/**
 * @brief Attach a CLIP/vision projector (mmproj GGUF, third_party/llama.cpp/tools/mtmd) to an
 *        already-loaded text model, enabling ameva_llm_generate_with_image(). The vision encoder
 *        runs on the same device list ameva_llm_load() chose (ggml-ameva GPU backend or CPU) --
 *        it is picked up through the process-wide ggml backend registry ggml_backend_load()
 *        already populated, not re-selected here.
 *
 * May be called at most once per AmevaLlm handle; call again after ameva_llm_free() +
 * ameva_llm_load() to switch mmproj files.
 *
 * @param mmproj_path Filesystem path to the mmproj GGUF matching the loaded text model's family
 *        (e.g. mmproj-Qwen2-VL-2B-Instruct-f16.gguf for a Qwen2-VL-2B-Instruct text model).
 * @return AMEVA_LLM_OK on success, AMEVA_LLM_ERR_VISION_LOAD_FAILED if the mmproj file is missing,
 *         malformed, or does not match the text model's vision projector type.
 */
AmevaLlmStatus ameva_llm_load_mmproj(
    AmevaLlm* llm,
    const char* mmproj_path,
    char* err_buf,
    size_t err_buf_size
);

/** @brief Whether ameva_llm_load_mmproj() has successfully attached a vision encoder. */
bool ameva_llm_supports_vision(AmevaLlm* llm);

/**
 * @brief Encode one image and generate text conditioned on it and a text prompt. Requires a
 *        prior successful ameva_llm_load_mmproj() call -- AMEVA_LLM_ERR_VISION_NOT_LOADED
 *        otherwise, never a silent text-only fallback.
 *
 * @param image_path Filesystem path to a JPEG/PNG/BMP/GIF/static-WebP file (stb_image, bundled in
 *        mtmd). Decoded and preprocessed (resize/patchify) entirely inside mtmd/clip.cpp.
 * @param prompt Text prompt. If it contains mtmd's media marker ("<__media__>"), the image is
 *        substituted there; otherwise the marker is prepended automatically, so a plain
 *        "describe this image" prompt works unmodified.
 * @param max_tokens, temperature, seed, out_buf, out_buf_size: same contract as
 *        ameva_llm_generate().
 * @return AMEVA_LLM_OK on success. AMEVA_LLM_ERR_IMAGE_LOAD_FAILED if image_path cannot be
 *         decoded, AMEVA_LLM_ERR_VISION_TOKENIZE_FAILED/AMEVA_LLM_ERR_VISION_ENCODE_FAILED for a
 *         tokenize/vision-encode failure, or any ameva_llm_generate() code for the text decode
 *         that follows.
 */
AmevaLlmStatus ameva_llm_generate_with_image(
    AmevaLlm* llm,
    const char* image_path,
    const char* prompt,
    int32_t max_tokens,
    float temperature,
    uint64_t seed,
    char* out_buf,
    size_t out_buf_size,
    char* err_buf,
    size_t err_buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_LLM_API_H */
