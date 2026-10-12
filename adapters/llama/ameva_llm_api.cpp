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

    std::memcpy(out_buf, result.data(), result.size());
    out_buf[result.size()] = '\0';
    return AMEVA_LLM_OK;
}

void ameva_llm_free(AmevaLlm* llm) {
    freeLlm(llm);
}
