/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ameva_stt_api.h
 * @brief ameva-android v0.3: minimal in-process "load Whisper model, transcribe PCM" C API.
 *
 * Wraps whisper.cpp (third_party/whisper.cpp, pinned tag v1.8.6 + two upstream fixes, commit
 * 610e664b) + the formal AMEVA Vulkan STT backend (adapters/whisper/ggml-backend-ameva.cpp),
 * statically linked (unlike ameva_llm_api.h's llama.cpp path, which loads ggml-ameva as a
 * dynamic plugin -- whisper.cpp's AMEVA hook self-registers via ameva_backend_register() inside
 * whisper_backend_init(), compiled directly into this .so).
 *
 * Ground truth (2026-10-16): this backend currently accelerates exactly two encoder matmul
 * shapes (f16 and q5_0 weights, src/shaders/stt/matmul_{f16,q5_0}_f32.comp) via two
 * SHA-256-pinned SPIR-V binaries that must be extracted to app-writable storage and pointed to
 * by spirv_f16_path / spirv_q5_0_path -- it is not a full whisper-encoder GPU port. Every other
 * op in the graph runs on whichever backend ggml's scheduler picks (CPU, if AMEVA declines the
 * op) -- that is normal multi-backend scheduling, not a silent fallback: ameva_backend_register()
 * either provides a real device or does not register at all (see ggml-backend-ameva.cpp's
 * fail_kernel path), there is no partial/fake registration.
 */

#ifndef AMEVA_STT_API_H
#define AMEVA_STT_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AmevaStt AmevaStt;

typedef enum {
    AMEVA_STT_OK = 0,
    AMEVA_STT_ERR_MODEL_LOAD_FAILED,
    AMEVA_STT_ERR_TRANSCRIBE_FAILED,
    AMEVA_STT_ERR_OUTPUT_BUFFER_TOO_SMALL,
    AMEVA_STT_ERR_INVALID_ARGUMENT
} AmevaSttStatus;

/**
 * @brief Load a GGML Whisper model (e.g. ggml-base.en.bin, ggml-small-q5_0.bin) and register the
 *        AMEVA Vulkan STT backend.
 *
 * @param model_path Filesystem path to a whisper.cpp GGML model file (not bundled with this
 *        library -- same caller-supplies-the-model convention as ameva_llm_api.h).
 * @param spirv_f16_path Filesystem path to matmul_f16_f32.spv (SDK asset, extracted by the
 *        caller to app-writable storage; see AmevaSpeech.kt). NULL/empty: the f16 encoder
 *        matmul runs on CPU (ameva_backend_register() declines that kernel, not a crash).
 * @param spirv_q5_0_path Same, for matmul_q5_0_f32.spv / q5_0-quantized models. NULL/empty:
 *        that kernel runs on CPU.
 * @param language BCP-47-ish whisper language code (e.g. "ko", "en"), NULL/""/"auto" for
 *        auto-detection.
 * @param out_stt Output handle, guaranteed NULL on failure.
 * @param err_buf Destination for a human-readable failure reason; always NUL-terminated.
 * @param err_buf_size Size of err_buf.
 * @return AMEVA_STT_OK on success, an error code otherwise.
 */
AmevaSttStatus ameva_stt_load(
    const char* model_path,
    const char* spirv_f16_path,
    const char* spirv_q5_0_path,
    const char* language,
    AmevaStt** out_stt,
    char* err_buf,
    size_t err_buf_size
);

/**
 * @brief Transcribe mono 16 kHz float32 PCM samples (whisper.cpp's only supported input format;
 *        caller converts from int16 mic PCM, see AmevaSpeech.kt).
 *
 * @param pcmf32 Pointer to n_samples float32 samples in [-1.0, 1.0], 16000 Hz, mono.
 * @param translate If true, translate non-English speech to English instead of transcribing
 *        in the source language.
 * @param out_buf Destination for the transcript (NUL-terminated on success, plain text with no
 *        timestamps -- segments are joined in order).
 * @param out_buf_size Size of out_buf; AMEVA_STT_ERR_OUTPUT_BUFFER_TOO_SMALL if the transcript
 *        would not fit (nothing partial is written in that case).
 * @return AMEVA_STT_OK on success, an error code otherwise.
 */
AmevaSttStatus ameva_stt_transcribe(
    AmevaStt* stt,
    const float* pcmf32,
    int32_t n_samples,
    int32_t translate,
    char* out_buf,
    size_t out_buf_size,
    char* err_buf,
    size_t err_buf_size
);

/** @brief Release the whisper context and AMEVA backend. Safe to call with NULL. */
void ameva_stt_free(AmevaStt* stt);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_STT_API_H */
