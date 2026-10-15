/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_stt_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "whisper.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct AmevaStt {
    whisper_context* ctx = nullptr;
    std::string language; // whisper_full_params.language points at this for the object's lifetime
};

namespace {

void setMsg(char* buf, size_t size, const char* fmt, const char* detail) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, fmt, detail != nullptr ? detail : "(no detail)");
}

#if defined(__ANDROID__)
// Same rationale as ameva_llm_api.cpp: whisper.cpp logs via plain fprintf(stderr, ...) by
// default, invisible on Android with no attached console. Route it to logcat instead.
void androidLogCallback(ggml_log_level level, const char* text, void*) {
    int prio = ANDROID_LOG_INFO;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN; break;
        case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default: prio = ANDROID_LOG_INFO; break;
    }
    __android_log_write(prio, "ameva_stt", text);
}

void installLogCallbackOnce() {
    static bool installed = false;
    if (!installed) {
        whisper_log_set(androidLogCallback, nullptr);
        installed = true;
    }
}
#else
void installLogCallbackOnce() {}
#endif

} // namespace

AmevaSttStatus ameva_stt_load(
    const char* model_path,
    const char* spirv_f16_path,
    const char* spirv_q5_0_path,
    const char* language,
    AmevaStt** out_stt,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_stt != nullptr) *out_stt = nullptr;
    if (model_path == nullptr || out_stt == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_stt_load: invalid argument (%s)", "NULL pointer");
        return AMEVA_STT_ERR_INVALID_ARGUMENT;
    }

    installLogCallbackOnce();

    // ameva_backend_register() (called from whisper_backend_init(), itself called from
    // whisper_init_from_file_with_params() below) reads these at registration time -- they must
    // be set before that call. NULL/empty leaves ggml-backend-ameva.cpp's own hardcoded Termux
    // path, which does not exist in an app process, so load_spirv_binary() fails and that one
    // kernel runs on CPU (ggml-backend-ameva.cpp's fail_kernel path -- not a crash, not a silent
    // partial registration).
    if (spirv_f16_path != nullptr && spirv_f16_path[0] != '\0') {
        setenv("AMEVA_SPIRV_F16_PATH", spirv_f16_path, 1);
    }
    if (spirv_q5_0_path != nullptr && spirv_q5_0_path[0] != '\0') {
        setenv("AMEVA_SPIRV_Q5_0_PATH", spirv_q5_0_path, 1);
    }

    auto* stt = new AmevaStt();
    if (language != nullptr && language[0] != '\0') {
        stt->language = language;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.flash_attn = false; // same defense-in-depth as ameva_llm_api.cpp: no upstream Vulkan FA

    stt->ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (stt->ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "whisper_init_from_file_with_params failed for %s", model_path);
        delete stt;
        return AMEVA_STT_ERR_MODEL_LOAD_FAILED;
    }

    *out_stt = stt;
    return AMEVA_STT_OK;
}

AmevaSttStatus ameva_stt_transcribe(
    AmevaStt* stt,
    const float* pcmf32,
    int32_t n_samples,
    int32_t translate,
    char* out_buf,
    size_t out_buf_size,
    char* err_buf,
    size_t err_buf_size
) {
    if (stt == nullptr || stt->ctx == nullptr || pcmf32 == nullptr || n_samples <= 0 ||
        out_buf == nullptr || out_buf_size == 0) {
        setMsg(err_buf, err_buf_size, "ameva_stt_transcribe: invalid argument (%s)", "NULL pointer or n_samples <= 0");
        return AMEVA_STT_ERR_INVALID_ARGUMENT;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.translate         = translate != 0;
    wparams.no_timestamps     = true; // plain transcript text, not an SRT-style dump
    wparams.print_special     = false;
    wparams.print_progress    = false;
    wparams.print_realtime    = false;
    wparams.print_timestamps  = false;
    wparams.single_segment    = false;
    wparams.language          = stt->language.empty() ? "auto" : stt->language.c_str();

    if (whisper_full(stt->ctx, wparams, pcmf32, n_samples) != 0) {
        setMsg(err_buf, err_buf_size, "whisper_full failed (%s)", "decode error");
        return AMEVA_STT_ERR_TRANSCRIBE_FAILED;
    }

    std::string result;
    const int n_segments = whisper_full_n_segments(stt->ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(stt->ctx, i);
        if (text == nullptr) continue;
        if (!result.empty()) result += ' ';
        result += text;
    }

    if (result.size() >= out_buf_size) {
        setMsg(err_buf, err_buf_size, "transcript exceeds out_buf_size (%s)", "truncation refused");
        return AMEVA_STT_ERR_OUTPUT_BUFFER_TOO_SMALL;
    }
    std::memcpy(out_buf, result.data(), result.size());
    out_buf[result.size()] = '\0';
    return AMEVA_STT_OK;
}

void ameva_stt_free(AmevaStt* stt) {
    if (stt == nullptr) return;
    if (stt->ctx != nullptr) whisper_free(stt->ctx);
    delete stt;
}
