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
    AMEVA_LLM_ERR_INVALID_ARGUMENT
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
AmevaLlmStatus ameva_llm_load(
    const char* model_path,
    const char* backend_plugin_path,
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

/** @brief Release the context, model, and backend registration. Safe to call with NULL. */
void ameva_llm_free(AmevaLlm* llm);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_LLM_API_H */
