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
    AMEVA_SD_ERR_INVALID_ARGUMENT
} AmevaSdStatus;

/**
 * @param model_path Filesystem path to a single-file GGUF/safetensors/ckpt diffusion checkpoint
 *        (not bundled with this library).
 * @param backend_plugin_path Filesystem path to libggml-ameva.so.
 * @param n_threads CPU thread count for whatever this model leaves on CPU (text encoder tokenize,
 *        any op ggml-ameva declines); 0 uses stable-diffusion.cpp's own default.
 */
AmevaSdStatus ameva_sd_load(
    const char* model_path,
    const char* backend_plugin_path,
    int n_threads,
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

/** @brief Release the context, model, and backend registration. Safe to call with NULL. */
void ameva_sd_free(AmevaSd* sd);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_SD_API_H */
