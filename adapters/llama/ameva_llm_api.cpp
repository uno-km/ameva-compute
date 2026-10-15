/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_llm_api.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct AmevaLlm {
    ggml_backend_reg_t backend_reg = nullptr;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
};

namespace {

void setMsg(char* buf, size_t size, const char* fmt, const char* detail) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, fmt, detail != nullptr ? detail : "(no detail)");
}

#if defined(__ANDROID__)
// llama.cpp/ggml otherwise log via plain fprintf(stderr, ...), which is not visible anywhere on
// Android (no console attached, no logcat redirection) -- every past silent "it just failed" here
// was actually a real, specific diagnostic message with nowhere to go. Route it to logcat instead.
void androidLogCallback(ggml_log_level level, const char* text, void*) {
    int prio = ANDROID_LOG_INFO;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN; break;
        case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default: prio = ANDROID_LOG_INFO; break;
    }
    __android_log_write(prio, "ameva_llm", text);
}

void installLogCallbackOnce() {
    static bool installed = false;
    if (!installed) {
        llama_log_set(androidLogCallback, nullptr);
        installed = true;
    }
}
#else
void installLogCallbackOnce() {}
#endif

// max_tokens can cut generation off mid-codepoint: BPE token boundaries don't align to UTF-8
// character boundaries, so the last token appended to `result` may be an incomplete trailing
// multi-byte sequence. JNI's NewStringUTF aborts the whole process on malformed input (confirmed:
// "JNI DETECTED ERROR ... illegal continuation byte" SIGABRT) rather than failing softly, so this
// must never be left for the JNI boundary to discover -- trim it here so out_buf is always
// complete, well-formed UTF-8.
void trimIncompleteUtf8Tail(std::string& s) {
    if (s.empty()) return;
    for (int back = 1; back <= 4 && back <= (int)s.size(); ++back) {
        unsigned char c = (unsigned char)s[s.size() - back];
        if ((c & 0xC0) == 0x80) continue; // continuation byte, keep scanning backward
        int expectedLen;
        if ((c & 0x80) == 0x00) expectedLen = 1;
        else if ((c & 0xE0) == 0xC0) expectedLen = 2;
        else if ((c & 0xF0) == 0xE0) expectedLen = 3;
        else if ((c & 0xF8) == 0xF0) expectedLen = 4;
        else { s.erase(s.size() - back); return; } // not a valid lead byte either; drop it
        if (expectedLen > back) s.erase(s.size() - back); // incomplete sequence
        return;
    }
}

void freeLlm(AmevaLlm* llm) {
    if (llm == nullptr) return;
    if (llm->ctx != nullptr) llama_free(llm->ctx);
    if (llm->model != nullptr) llama_model_free(llm->model);
    // ggml_backend_reg_t has no public unload call in this ggml version; the plugin .so stays
    // mapped for the process lifetime, matching llama.cpp's own backend-load lifecycle.
    delete llm;
}

} // namespace

AmevaLlmStatus ameva_llm_load(
    const char* model_path,
    const char* backend_plugin_path,
    const char* cache_dir,
    uint32_t n_ctx,
    AmevaLlm** out_llm,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_llm != nullptr) *out_llm = nullptr;
    if (model_path == nullptr || backend_plugin_path == nullptr || out_llm == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_load: invalid argument (%s)", "NULL pointer");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }

    // ggml-ameva falls back to $HOME for both caches, which does not exist in an app process --
    // point it at the caller's own writable directory instead (Android: Context.filesDir).
    if (cache_dir != nullptr && cache_dir[0] != '\0') {
        std::string pipelineCache = std::string(cache_dir) + "/ggml-ameva-pipelines.bin";
        std::string qmvTune = std::string(cache_dir) + "/ggml-ameva-qmv-tune.txt";
        setenv("AMEVA_PIPELINE_CACHE", pipelineCache.c_str(), 1);
        setenv("AMEVA_QMV_TUNE_FILE", qmvTune.c_str(), 1);
    }

    installLogCallbackOnce();
    llama_backend_init();

    auto* llm = new AmevaLlm();

    llm->backend_reg = ggml_backend_load(backend_plugin_path);
    if (llm->backend_reg == nullptr) {
        setMsg(err_buf, err_buf_size, "ggml_backend_load failed for %s", backend_plugin_path);
        freeLlm(llm);
        return AMEVA_LLM_ERR_BACKEND_LOAD_FAILED;
    }

    size_t dev_count = ggml_backend_reg_dev_count(llm->backend_reg);
    if (dev_count == 0) {
        setMsg(err_buf, err_buf_size, "ggml-ameva plugin %s registered zero devices",
               backend_plugin_path);
        freeLlm(llm);
        return AMEVA_LLM_ERR_BACKEND_NO_DEVICE;
    }
    ggml_backend_dev_t ameva_dev = ggml_backend_reg_dev_get(llm->backend_reg, 0);

    // NULL-terminated device list: only ggml-ameva + llama.cpp's own CPU backend run anything.
    // Upstream ggml-vulkan is never loaded in this process, so an op ggml-ameva declines falls
    // back to CPU, not to the known-corrupt upstream Vulkan flash-attention path.
    static ggml_backend_dev_t devices[2];
    devices[0] = ameva_dev;
    devices[1] = nullptr;

    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices;
    mparams.n_gpu_layers = -1; // offload every layer the backend accepts

    llm->model = llama_model_load_from_file(model_path, mparams);
    if (llm->model == nullptr) {
        setMsg(err_buf, err_buf_size, "llama_model_load_from_file failed for %s", model_path);
        freeLlm(llm);
        return AMEVA_LLM_ERR_MODEL_LOAD_FAILED;
    }
    llm->vocab = llama_model_get_vocab(llm->model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx; // 0 keeps the model's own default
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llm->ctx = llama_init_from_model(llm->model, cparams);
    if (llm->ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "llama_init_from_model failed for %s", model_path);
        freeLlm(llm);
        return AMEVA_LLM_ERR_CONTEXT_CREATE_FAILED;
    }

    *out_llm = llm;
    return AMEVA_LLM_OK;
}

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
) {
    if (llm == nullptr || prompt == nullptr || out_buf == nullptr || out_buf_size == 0) {
        setMsg(err_buf, err_buf_size, "ameva_llm_generate: invalid argument (%s)", "NULL pointer");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }

    const int32_t prompt_len = (int32_t)std::strlen(prompt);
    int32_t n_prompt_tokens = -llama_tokenize(llm->vocab, prompt, prompt_len, nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        setMsg(err_buf, err_buf_size, "llama_tokenize sizing failed for prompt (%s)", prompt);
        return AMEVA_LLM_ERR_TOKENIZE_FAILED;
    }
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(llm->vocab, prompt, prompt_len, prompt_tokens.data(), n_prompt_tokens,
                        true, true) < 0) {
        setMsg(err_buf, err_buf_size, "llama_tokenize failed for prompt (%s)", prompt);
        return AMEVA_LLM_ERR_TOKENIZE_FAILED;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    if (llama_decode(llm->ctx, batch) != 0) {
        setMsg(err_buf, err_buf_size, "llama_decode failed on prompt (%s)", prompt);
        return AMEVA_LLM_ERR_DECODE_FAILED;
    }

    llama_sampler_chain_params scparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(scparams);
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t)seed));

    std::string result;
    char piece[256];
    AmevaLlmStatus status = AMEVA_LLM_OK;

    for (int32_t i = 0; i < max_tokens; ++i) {
        llama_token tok = llama_sampler_sample(chain, llm->ctx, -1);
        llama_sampler_accept(chain, tok);

        if (llama_vocab_is_eog(llm->vocab, tok)) break;

        int32_t piece_len = llama_token_to_piece(llm->vocab, tok, piece, sizeof(piece), 0, true);
        if (piece_len < 0) {
            setMsg(err_buf, err_buf_size, "llama_token_to_piece failed (%s)", "overflowed scratch buffer");
            status = AMEVA_LLM_ERR_DECODE_FAILED;
            break;
        }
        if (result.size() + (size_t)piece_len >= out_buf_size) {
            setMsg(err_buf, err_buf_size, "generated text exceeds out_buf_size (%s)", "truncation refused");
            status = AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL;
            break;
        }
        result.append(piece, piece_len);

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(llm->ctx, next) != 0) {
            setMsg(err_buf, err_buf_size, "llama_decode failed mid-generation (%s)", prompt);
            status = AMEVA_LLM_ERR_DECODE_FAILED;
            break;
        }
    }

    llama_sampler_free(chain);

    if (status != AMEVA_LLM_OK) {
        return status;
    }

    trimIncompleteUtf8Tail(result);
    std::memcpy(out_buf, result.data(), result.size());
    out_buf[result.size()] = '\0';
    return AMEVA_LLM_OK;
}

void ameva_llm_free(AmevaLlm* llm) {
    freeLlm(llm);
}
