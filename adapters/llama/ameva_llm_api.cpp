/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_llm_api.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ameva_backend_bench.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct AmevaLlm {
    ggml_backend_reg_t backend_reg = nullptr;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    mtmd_context* mtmd = nullptr; // set by ameva_llm_load_mmproj(); NULL = text-only
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

// Fleet observation (2026-10-27, S21 Qwen2.5-VL-7B): mparams.n_gpu_layers=-1 (offload every layer)
// crashed the whole Termux session mid-load on a 7 GiB-RAM UMA device once the model no longer
// fits in free memory -- ggml-ameva's device memory report is real (/proc/meminfo MemAvailable),
// but nothing used it to size the offload before this. Mirrors sd::backend_fit's budget-then-split
// approach (see third_party/stable-diffusion.cpp/src/core/backend_fit.cpp) at layer granularity
// instead of pipeline-component granularity: peek the GGUF tensor list (no tensor data read) for
// llama.cpp's universal "blk.N.*" naming to estimate a per-layer byte cost, then offload only as
// many layers as the budget allows. Never a silent full-GPU attempt -- a tighter fit than requested
// is always logged.
struct LayerBudgetEstimate {
    int32_t n_layers = 0;
    int64_t avg_layer_bytes = 0;
    int64_t non_layer_bytes = 0; // embeddings/output/norms: always resident wherever any layer runs
    bool ok = false;
};

LayerBudgetEstimate estimateLayerBudget(const char* model_path) {
    LayerBudgetEstimate est;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context* gguf = gguf_init_from_file(model_path, params);
    if (gguf == nullptr) return est;

    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    int64_t layer_bytes_sum = 0;
    int32_t max_layer_idx = -1;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf, i);
        const int64_t sz = (int64_t)gguf_get_tensor_size(gguf, i);
        int layer_idx = -1;
        if (name != nullptr && std::sscanf(name, "blk.%d.", &layer_idx) == 1) {
            layer_bytes_sum += sz;
            if (layer_idx > max_layer_idx) max_layer_idx = layer_idx;
        } else {
            est.non_layer_bytes += sz;
        }
    }
    gguf_free(gguf);
    if (max_layer_idx < 0) return est; // no "blk.N." tensors recognized; caller keeps its own default
    est.n_layers = max_layer_idx + 1;
    est.avg_layer_bytes = layer_bytes_sum / est.n_layers;
    est.ok = true;
    return est;
}

struct GpuBudgetResult {
    bool ok = true;              // false only for AMEVA_GPU_BUDGET_MODE_GPU_ONLY that does not fit
    int32_t n_gpu_layers = -1;
    std::string fail_reason;     // set only when !ok
};

void logBudgetWarn(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
#if defined(__ANDROID__)
    __android_log_write(ANDROID_LOG_WARN, "ameva_llm", msg);
#endif
}

// Opt-in only: opts == NULL or mode == AMEVA_GPU_BUDGET_MODE_DEFAULT reproduces the previous fixed
// behavior (-1, offload everything, no estimation). Every other mode sizes the offload against
// ggml-ameva's real free-memory report (/proc/meminfo MemAvailable, this device is UMA) and the
// GGUF's own "blk.N.*" tensor layout -- never silently falls back to "offload everything" once the
// caller has opted in, since that is exactly the fixed behavior that OOM-killed the whole process
// in the first place.
GpuBudgetResult fitGpuLayerBudget(ggml_backend_dev_t dev, const char* model_path,
                                   const AmevaGpuBudgetOptions* opts) {
    GpuBudgetResult result;
    if (opts == nullptr || opts->mode == AMEVA_GPU_BUDGET_MODE_DEFAULT) {
        result.n_gpu_layers = -1;
        return result;
    }
    if (opts->mode == AMEVA_GPU_BUDGET_MODE_CPU_ONLY) {
        result.n_gpu_layers = 0;
        return result;
    }

    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    const LayerBudgetEstimate est = estimateLayerBudget(model_path);
    const float fraction = (opts->budget_fraction > 0.f) ? opts->budget_fraction : 0.95f;
    const bool gpuOnly = (opts->mode == AMEVA_GPU_BUDGET_MODE_GPU_ONLY);

    if (free_bytes == 0 || !est.ok || est.avg_layer_bytes <= 0) {
        if (gpuOnly) {
            result.ok = false;
            result.fail_reason = "AMEVA_GPU_BUDGET_MODE_GPU_ONLY: device memory or model layer size "
                                  "could not be estimated; refusing an unmeasured full-GPU attempt";
            return result;
        }
        result.n_gpu_layers = -1; // GPU_THEN_CPU without a usable estimate: previous fixed behavior
        return result;
    }

    constexpr int64_t kSafetyReserveBytes = 1536LL * 1024 * 1024; // KV cache + compute-buffer headroom
    const int64_t budget = (int64_t)((double)free_bytes * (double)fraction)
                          - kSafetyReserveBytes - est.non_layer_bytes;
    int32_t fit_layers = 0;
    if (budget > 0) {
        const int64_t max_fit = budget / est.avg_layer_bytes;
        fit_layers = (int32_t)std::min<int64_t>(est.n_layers, max_fit);
    }

    if (fit_layers < est.n_layers) {
        if (gpuOnly) {
            result.ok = false;
            char reason[320];
            std::snprintf(reason, sizeof(reason),
                "AMEVA_GPU_BUDGET_MODE_GPU_ONLY: only %d/%d transformer layers fit the budget "
                "(free=%lld MiB, fraction=%.2f, reserve=%lld MiB); refusing to spill %d layer(s) to CPU",
                fit_layers, est.n_layers, (long long)(free_bytes / (1024 * 1024)), fraction,
                (long long)(kSafetyReserveBytes / (1024 * 1024)), est.n_layers - fit_layers);
            result.fail_reason = reason;
            return result;
        }
        char msg[320];
        std::snprintf(msg, sizeof(msg),
            "ameva_llm_load [budget-fit WARN]: only %d/%d transformer layers fit the ggml-ameva "
            "memory budget (free=%lld MiB, fraction=%.2f) -- %d layer(s) run on CPU instead of a "
            "silent full-GPU attempt that would risk an OOM kill",
            fit_layers, est.n_layers, (long long)(free_bytes / (1024 * 1024)), fraction,
            est.n_layers - fit_layers);
        logBudgetWarn(msg);
    }
    result.n_gpu_layers = fit_layers;
    return result;
}

void freeLlm(AmevaLlm* llm) {
    if (llm == nullptr) return;
    if (llm->mtmd != nullptr) mtmd_free(llm->mtmd);
    if (llm->ctx != nullptr) llama_free(llm->ctx);
    if (llm->model != nullptr) llama_model_free(llm->model);
    // ggml_backend_reg_t has no public unload call in this ggml version; the plugin .so stays
    // mapped for the process lifetime, matching llama.cpp's own backend-load lifecycle.
    delete llm;
}

// Shared by ameva_llm_generate() and ameva_llm_generate_with_image(): sample up to max_tokens
// from whatever is already decoded at the context's current position, appending text to `out`.
// Assumes the prompt (text-only or mtmd image+text chunks) was already decoded by the caller.
AmevaLlmStatus sampleLoop(
    AmevaLlm* llm,
    int32_t max_tokens,
    float temperature,
    uint64_t seed,
    std::string& out,
    size_t out_cap,
    char* err_buf,
    size_t err_buf_size
) {
    llama_sampler_chain_params scparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(scparams);
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t)seed));

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
        if (out.size() + (size_t)piece_len >= out_cap) {
            setMsg(err_buf, err_buf_size, "generated text exceeds out_buf_size (%s)", "truncation refused");
            status = AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL;
            break;
        }
        out.append(piece, piece_len);

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(llm->ctx, next) != 0) {
            setMsg(err_buf, err_buf_size, "llama_decode failed mid-generation (%s)", "see log");
            status = AMEVA_LLM_ERR_DECODE_FAILED;
            break;
        }
    }

    llama_sampler_free(chain);
    return status;
}

// "auto": benchmark cpu/vulkan/opencl (ameva_backend_bench.h) and return whichever is actually
// fastest on this device. Never silently prefers one on tie-ish grounds without a measurement --
// if a candidate's benchmark reports it unavailable, it is excluded, not guessed to be slow.
// Returns empty on total failure (should not happen: cpu always succeeds).
std::string resolveAutoBackend(const char* backend_plugin_path, char* err_buf, size_t err_buf_size) {
    struct Candidate { const char* name; double ms; bool ok; };
    Candidate candidates[3] = { {"cpu", -1.0, false}, {"vulkan", -1.0, false}, {"opencl", -1.0, false} };

    for (auto& c : candidates) {
        char benchErr[128] = {0};
        AmevaBenchStatus st = ameva_backend_bench(c.name, backend_plugin_path, &c.ms, benchErr, sizeof(benchErr));
        c.ok = (st == AMEVA_BENCH_OK);
#if defined(__ANDROID__)
        if (c.ok) {
            __android_log_print(ANDROID_LOG_INFO, "ameva_llm", "auto-bench %s: %.2f ms/op", c.name, c.ms);
        } else {
            __android_log_print(ANDROID_LOG_INFO, "ameva_llm", "auto-bench %s: unavailable (%s)", c.name, benchErr);
        }
#endif
    }

    const Candidate* best = nullptr;
    for (const auto& c : candidates) {
        if (c.ok && (best == nullptr || c.ms < best->ms)) best = &c;
    }
    if (best == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_load: auto backend selection found %s",
               "no usable backend (not even cpu) -- refusing to guess");
        return std::string();
    }
    return std::string(best->name);
}

} // namespace

AmevaLlmStatus ameva_llm_load(
    const char* model_path,
    const char* backend_plugin_path,
    const char* cache_dir,
    const char* backend,
    uint32_t n_ctx,
    const AmevaGpuBudgetOptions* gpu_budget_opts,
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

    // "auto" is the only caller intent that tolerates a per-op CPU fallback (with a visible log
    // line, never silent). An explicit "vulkan"/"opencl" -- or the default when backend is
    // NULL/empty -- is a GPU-only commitment: an op ggml-ameva cannot run must fail the load, not
    // quietly hand that op to the CPU backend llama_init_from_model always keeps available.
    const bool wasAutoRequested = (backend != nullptr && std::string(backend) == "auto");
    std::string resolvedBackend = (backend != nullptr && backend[0] != '\0') ? backend : "vulkan";
    if (resolvedBackend == "auto") {
        resolvedBackend = resolveAutoBackend(backend_plugin_path, err_buf, err_buf_size);
        if (resolvedBackend.empty()) {
            return AMEVA_LLM_ERR_BACKEND_NO_DEVICE;
        }
    }
    const bool gpuOnlyCommitment = !wasAutoRequested && (resolvedBackend == "vulkan" || resolvedBackend == "opencl");

    auto* llm = new AmevaLlm();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1; // offload every layer the backend accepts

    static ggml_backend_dev_t devices[2];
    if (resolvedBackend == "cpu") {
        // mparams.devices stays NULL: llama's own default device list (CPU only). The ggml-ameva
        // plugin is never loaded in this process at all -- not "loaded but unused", genuinely
        // absent, so there is no question of which backend a CPU-selected load actually used.
    } else if (resolvedBackend == "vulkan" || resolvedBackend == "opencl") {
        setenv("AMEVA_BACKEND", resolvedBackend.c_str(), 1);

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

        const GpuBudgetResult budget_result = fitGpuLayerBudget(ameva_dev, model_path, gpu_budget_opts);
        if (!budget_result.ok) {
            setMsg(err_buf, err_buf_size, "%s", budget_result.fail_reason.c_str());
            freeLlm(llm);
            return AMEVA_LLM_ERR_GPU_BUDGET_EXCEEDED;
        }
        mparams.n_gpu_layers = budget_result.n_gpu_layers;

        // NULL-terminated device list: only ggml-ameva + llama.cpp's own CPU backend run
        // anything. Upstream ggml-vulkan is never loaded in this process, so an op ggml-ameva
        // declines falls back to CPU, not to the known-corrupt upstream Vulkan FA path.
        devices[0] = ameva_dev;
        devices[1] = nullptr;
        mparams.devices = devices;
    } else {
        setMsg(err_buf, err_buf_size, "ameva_llm_load: unknown backend \"%s\"", resolvedBackend.c_str());
        freeLlm(llm);
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }

    llm->model = llama_model_load_from_file(model_path, mparams);
    if (llm->model == nullptr) {
        setMsg(err_buf, err_buf_size, "llama_model_load_from_file failed for %s", model_path);
        freeLlm(llm);
        return AMEVA_LLM_ERR_MODEL_LOAD_FAILED;
    }
    llm->vocab = llama_model_get_vocab(llm->model);

    // M-RoPE (Qwen2-VL and similar) and GGML_ROPE_TYPE_VISION are both rejected by
    // ggml-ameva's compute_support (see ggml-ameva.cpp GGML_OP_ROPE: only NORM/NEOX accepted).
    // Under a GPU-only commitment that must be a hard failure here, not a per-op CPU hand-off
    // discovered only once decode runs the rope node.
    const llama_rope_type ropeType = llama_model_rope_type(llm->model);
    const bool ropeUnsupportedByAmeva =
        ropeType == LLAMA_ROPE_TYPE_MROPE || ropeType == LLAMA_ROPE_TYPE_IMROPE ||
        ropeType == LLAMA_ROPE_TYPE_VISION;
    if (ropeUnsupportedByAmeva) {
        if (gpuOnlyCommitment) {
            setMsg(err_buf, err_buf_size,
                   "AMEVA_ERR_ROPE_MROPE_NOT_SUPPORTED: model rope type requires M-RoPE/vision "
                   "rope, which ggml-ameva does not implement; refusing silent CPU fallback under "
                   "GPU-only backend \"%s\"", resolvedBackend.c_str());
            freeLlm(llm);
            return AMEVA_LLM_ERR_GPU_ONLY_UNSUPPORTED_OP;
        }
        // "auto" resolved to a GPU backend but this model's rope op cannot run there: visible,
        // not silent. The rope nodes will land on the CPU backend llama_init_from_model always
        // keeps available; everything else still runs on ggml-ameva. If auto instead resolved to
        // "cpu" outright there is no fallback to report -- the whole model already runs on CPU.
        if (resolvedBackend == "vulkan" || resolvedBackend == "opencl") {
            char warnMsg[256];
            std::snprintf(warnMsg, sizeof(warnMsg),
                "ameva_llm_load [auto-fallback WARN]: model rope type (M-RoPE/vision) unsupported "
                "by ggml-ameva -- rope ops will run on CPU while backend=\"%s\" handles the rest",
                resolvedBackend.c_str());
            std::fprintf(stderr, "%s\n", warnMsg);
#if defined(__ANDROID__)
            __android_log_write(ANDROID_LOG_WARN, "ameva_llm", warnMsg);
#endif
        }
    }

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

    std::string result;
    AmevaLlmStatus status = sampleLoop(llm, max_tokens, temperature, seed, result, out_buf_size,
                                        err_buf, err_buf_size);
    if (status != AMEVA_LLM_OK) {
        return status;
    }

    trimIncompleteUtf8Tail(result);
    std::memcpy(out_buf, result.data(), result.size());
    out_buf[result.size()] = '\0';
    return AMEVA_LLM_OK;
}

AmevaLlmStatus ameva_llm_reset(
    AmevaLlm* llm,
    char* err_buf,
    size_t err_buf_size
) {
    if (llm == nullptr || llm->ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_reset: invalid argument (%s)", "NULL handle");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }
    llama_memory_clear(llama_get_memory(llm->ctx), /*data=*/true);
    return AMEVA_LLM_OK;
}

void ameva_llm_free(AmevaLlm* llm) {
    freeLlm(llm);
}

AmevaLlmStatus ameva_llm_load_mmproj(
    AmevaLlm* llm,
    const char* mmproj_path,
    char* err_buf,
    size_t err_buf_size
) {
    if (llm == nullptr || llm->model == nullptr || mmproj_path == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_load_mmproj: invalid argument (%s)", "NULL pointer");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }
    if (llm->mtmd != nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_load_mmproj: %s",
               "a vision encoder is already loaded on this handle -- ameva_llm_free() + "
               "ameva_llm_load() to switch mmproj files");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    // Picked up through the process-wide ggml backend registry ggml_backend_load() already
    // populated in ameva_llm_load() -- the same ggml-ameva device, not a second selection here.
    mparams.use_gpu = true;
    mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // same defense-in-depth as the text path
    mparams.print_timings = false;
    mparams.n_threads = 4;
    mparams.warmup = false; // the caller's first ameva_llm_generate_with_image() IS the warmup

    llm->mtmd = mtmd_init_from_file(mmproj_path, llm->model, mparams);
    if (llm->mtmd == nullptr) {
        setMsg(err_buf, err_buf_size, "mtmd_init_from_file failed for %s", mmproj_path);
        return AMEVA_LLM_ERR_VISION_LOAD_FAILED;
    }
    return AMEVA_LLM_OK;
}

bool ameva_llm_supports_vision(AmevaLlm* llm) {
    return llm != nullptr && llm->mtmd != nullptr && mtmd_support_vision(llm->mtmd);
}

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
) {
    if (llm == nullptr || image_path == nullptr || prompt == nullptr || out_buf == nullptr ||
        out_buf_size == 0) {
        setMsg(err_buf, err_buf_size, "ameva_llm_generate_with_image: invalid argument (%s)",
               "NULL pointer");
        return AMEVA_LLM_ERR_INVALID_ARGUMENT;
    }
    if (llm->mtmd == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_llm_generate_with_image: %s",
               "no vision encoder loaded -- call ameva_llm_load_mmproj() first");
        return AMEVA_LLM_ERR_VISION_NOT_LOADED;
    }

    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(llm->mtmd, image_path);
    if (bitmap == nullptr) {
        setMsg(err_buf, err_buf_size, "mtmd_helper_bitmap_init_from_file failed for %s", image_path);
        return AMEVA_LLM_ERR_IMAGE_LOAD_FAILED;
    }

    // The media marker may already be in the caller's prompt (multi-image placement); otherwise
    // prepend it so a plain "describe this image" prompt works unmodified.
    std::string markedPrompt(prompt);
    const char* marker = mtmd_default_marker();
    if (markedPrompt.find(marker) == std::string::npos) {
        markedPrompt = std::string(marker) + "\n" + markedPrompt;
    }

    mtmd_input_text text{markedPrompt.c_str(), /*add_special=*/true, /*parse_special=*/true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {bitmap};
    int32_t tok_rc = mtmd_tokenize(llm->mtmd, chunks, &text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tok_rc != 0) {
        mtmd_input_chunks_free(chunks);
        setMsg(err_buf, err_buf_size, "mtmd_tokenize failed (%s)",
               tok_rc == 1 ? "bitmap/marker count mismatch" : "image preprocessing error");
        return AMEVA_LLM_ERR_VISION_TOKENIZE_FAILED;
    }

    llama_pos new_n_past = 0;
    int32_t eval_rc = mtmd_helper_eval_chunks(llm->mtmd, llm->ctx, chunks, /*n_past=*/0,
                                              /*seq_id=*/0, /*n_batch=*/llama_n_batch(llm->ctx),
                                              /*logits_last=*/true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (eval_rc != 0) {
        setMsg(err_buf, err_buf_size, "mtmd_helper_eval_chunks failed (%s)",
               "vision encode or decode of the image+text prompt");
        return AMEVA_LLM_ERR_VISION_ENCODE_FAILED;
    }

    std::string result;
    AmevaLlmStatus status = sampleLoop(llm, max_tokens, temperature, seed, result, out_buf_size,
                                        err_buf, err_buf_size);
    if (status != AMEVA_LLM_OK) {
        return status;
    }

    trimIncompleteUtf8Tail(result);
    std::memcpy(out_buf, result.data(), result.size());
    out_buf[result.size()] = '\0';
    return AMEVA_LLM_OK;
}
