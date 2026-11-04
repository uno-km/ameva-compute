/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_sd_api.h"

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "stable-diffusion.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct AmevaSd {
    ggml_backend_reg_t backend_reg = nullptr;
    sd_ctx_t* ctx = nullptr;
    std::atomic<bool> cancel_requested{false};
    std::string placement; // see ameva_sd_placement()
};

namespace {

// stable-diffusion.cpp's progress hook is a process-global; one generation at a time.
std::mutex g_generate_mutex;

void setMsg(char* buf, size_t size, const char* fmt, const char* detail) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, fmt, detail != nullptr ? detail : "(no detail)");
}

void setRaw(char* buf, size_t size, const char* text) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, "%s", text != nullptr ? text : "(no detail)");
}

// The memory line stable-diffusion.cpp prints while loading ("total params memory size = ... (VRAM x, RAM y):
// text_encoders ...(RAM), diffusion_model ...(VRAM), ..."), captured for ameva_sd_placement(). Loads are
// serialized by the caller's load path; the mutex only guards the string.
std::mutex g_placement_mutex;
std::string g_last_memory_line;

void capturePlacementLine(const char* text) {
    if (text == nullptr) return;
    const char* hit = std::strstr(text, "total params memory size");
    if (hit == nullptr) return;
    std::string line(hit);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    std::lock_guard<std::mutex> lock(g_placement_mutex);
    g_last_memory_line = line;
}

#if defined(__ANDROID__)
void androidSdLogCallback(enum sd_log_level_t level, const char* text, void*) {
    capturePlacementLine(text);
    int prio = ANDROID_LOG_INFO;
    if (level == SD_LOG_ERROR) prio = ANDROID_LOG_ERROR;
    else if (level == SD_LOG_WARN) prio = ANDROID_LOG_WARN;
    else if (level == SD_LOG_DEBUG) prio = ANDROID_LOG_DEBUG;
    __android_log_write(prio, "ameva_sd", text != nullptr ? text : "");
}

void installLogCallbackOnce() {
    static bool installed = false;
    if (!installed) {
        sd_set_log_callback(androidSdLogCallback, nullptr);
        installed = true;
    }
}
#else
void hostSdLogCallback(enum sd_log_level_t, const char* text, void*) {
    capturePlacementLine(text);
    if (text != nullptr) std::fputs(text, stderr);
}

void installLogCallbackOnce() {
    static bool installed = false;
    if (!installed) {
        sd_set_log_callback(hostSdLogCallback, nullptr);
        installed = true;
    }
}
#endif

void freeSd(AmevaSd* sd) {
    if (sd == nullptr) return;
    if (sd->ctx != nullptr) free_sd_ctx(sd->ctx);
    // Same lifecycle note as ameva_llm_api.cpp: the plugin .so stays mapped for the process
    // lifetime, there is no public ggml_backend_reg_t unload call in this ggml version.
    delete sd;
}

bool nonEmpty(const char* s) { return s != nullptr && s[0] != '\0'; }

/**
 * ggml_backend_load() registers the plugin in ggml's process-global backend registry on every
 * call, and this ggml version has no unregister. Loading it again for each model would append a
 * duplicate AMEVA0 device per load, so the registration is done once per process and reused.
 * A different path after the first load is refused rather than silently ignored.
 */
ggml_backend_reg_t loadPluginOnce(const char* path, char* err_buf, size_t err_buf_size) {
    static std::mutex m;
    static std::string loadedPath;
    static ggml_backend_reg_t reg = nullptr;
    std::lock_guard<std::mutex> lock(m);
    if (reg != nullptr) {
        if (loadedPath == path) return reg;
        setMsg(err_buf, err_buf_size, "ggml-ameva plugin already loaded from a different path (%s)", loadedPath.c_str());
        return nullptr;
    }
    reg = ggml_backend_load(path);
    if (reg == nullptr) {
        setMsg(err_buf, err_buf_size, "ggml_backend_load failed for %s", path);
        return nullptr;
    }
    loadedPath = path;
    return reg;
}

/** Sum of the sizes of every non-empty path; false (with the failing path) if one cannot be stat'ed. */
bool sumFileSizes(const std::vector<const char*>& paths, int64_t* total, const char** failed) {
    *total = 0;
    for (const char* p : paths) {
        if (!nonEmpty(p)) continue;
        struct stat st;
        if (stat(p, &st) != 0) {
            *failed = p;
            return false;
        }
        *total += (int64_t)st.st_size;
    }
    return true;
}

struct ProgressTrampoline {
    AmevaSdProgressFn user_fn = nullptr;
    void* user_data = nullptr;
    int duty_percent = 100;
    std::atomic<bool>* cancel = nullptr;
};

/**
 * Receives every stable-diffusion.cpp progress report during ameva_sd_generate(): sampling steps,
 * VAE tiles, and lazily loaded tensor batches. [seconds] is sd.cpp's own measured time for the unit
 * just finished. The duty cycle idles for (100-d)/d of that time after each unit. A single idle is
 * capped at 5 s so a pathological measurement cannot stall the generation.
 */
void progressTrampoline(int step, int steps, float seconds, void* data) {
    auto* t = static_cast<ProgressTrampoline*>(data);
    if (t == nullptr) return;
    if (t->user_fn != nullptr) t->user_fn(step, steps, seconds, t->user_data);
    if (t->duty_percent >= 100 || step <= 0 || !(seconds > 0.f) || !std::isfinite(seconds)) return;
    if (t->cancel != nullptr && t->cancel->load()) return;
    const double idle = std::min(5.0, (double)seconds * (100.0 - t->duty_percent) / t->duty_percent);
    std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(idle * 1e6)));
}

} // namespace

void ameva_sd_load_options_init(AmevaSdLoadOptions* opts) {
    if (opts == nullptr) return;
    *opts = AmevaSdLoadOptions{};
    opts->struct_size = (uint32_t)sizeof(AmevaSdLoadOptions);
    opts->budget_mode = AMEVA_SD_GPU_BUDGET_MODE_DEFAULT;
    opts->budget_fraction = 0.95f;
    opts->text_encoder_placement = AMEVA_SD_PLACEMENT_GPU;
    opts->vae_placement = AMEVA_SD_PLACEMENT_GPU;
    opts->diffusion_params = AMEVA_SD_PARAMS_RESIDENT;
    opts->enable_mmap = 1;
}

// Bytes of a single-file checkpoint per stable-diffusion module, read from the GGUF tensor table (no data loaded).
// Names are classified as sd.cpp's backend_fit does. false for a non-GGUF file or an unreadable header.
// AMEVA_PROFILE set: ggml-ameva's totals after every generation. The plugin prints them at process exit, which an
// Android app process never reaches through atexit.
static void reportGpuProfile(const AmevaSd* sd) {
    if (sd == nullptr || sd->backend_reg == nullptr || getenv("AMEVA_PROFILE") == nullptr) return;
    using ReportFn = void (*)(void);
    auto fn = (ReportFn)ggml_backend_reg_get_proc_address(sd->backend_reg, "ggml_backend_ameva_profile_report");
    if (fn != nullptr) fn();
}

struct CheckpointParts { int64_t diffusion = 0, text_encoder = 0, vae = 0; };
static bool ggufParts(const char* path, CheckpointParts* out) {
    if (!nonEmpty(path)) return false;
    const size_t n = strlen(path);
    if (n < 5 || strcmp(path + n - 5, ".gguf") != 0) return false;
    gguf_init_params gp = {true, nullptr};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (g == nullptr) return false;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        const int64_t bytes = (int64_t)gguf_get_tensor_size(g, i);
        auto has = [&](const char* k) { return name.find(k) != std::string::npos; };
        if (has("model.diffusion_model.") || has("unet.")) out->diffusion += bytes;
        else if (has("first_stage_model.") || name.rfind("vae.", 0) == 0 || name.rfind("tae.", 0) == 0) out->vae += bytes;
        else out->text_encoder += bytes; // cond_stage_model, text_encoders, conditioner, ...
    }
    gguf_free(g);
    return true;
}

// Whether the device runs GGML_OP_FLASH_ATTN_EXT for the head sizes of the supported diffusion models
// (SD1.x 40/80/160, SDXL 64, DiT 128), with f16 K/V as stable-diffusion.cpp passes them, self- and cross-attention.
static bool deviceRunsFlashAttn(ggml_backend_dev_t dev) {
    ggml_init_params ip = {64 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* c = ggml_init(ip);
    if (c == nullptr) return false;
    bool ok = true;
    for (int d : {40, 64, 80, 128, 160}) {
        for (int kv : {77, 1024}) {
            ggml_tensor* q = ggml_new_tensor_4d(c, GGML_TYPE_F32, d, 1024, 8, 1);
            ggml_tensor* k = ggml_new_tensor_4d(c, GGML_TYPE_F16, d, kv, 8, 1);
            ggml_tensor* v = ggml_new_tensor_4d(c, GGML_TYPE_F16, d, kv, 8, 1);
            ggml_tensor* y = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / std::sqrt((float)d), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
            ok = ok && ggml_backend_dev_supports_op(dev, y);
        }
    }
    ggml_free(c);
    return ok;
}

// Whether the device computes GGML_OP_CONV_2D itself (ggml-ameva's supports_op checks op, types and shapes only,
// so a small no-alloc node answers it). ggml-ameva's OpenCL backend has no convolution kernel: with conv-direct
// there, every CONV_2D runs on the CPU and the graph is split around it, so conv-direct is enabled only where
// the device accepts the op.
static bool deviceRunsConv2d(ggml_backend_dev_t dev) {
    ggml_init_params ip = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* c = ggml_init(ip);
    if (c == nullptr) return false;
    ggml_tensor* w = ggml_new_tensor_4d(c, GGML_TYPE_F16, 3, 3, 8, 8);
    ggml_tensor* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, 16, 16, 8, 1);
    ggml_tensor* y = ggml_conv_2d_direct(c, w, x, 1, 1, 1, 1, 1, 1);
    const bool ok = ggml_backend_dev_supports_op(dev, y);
    ggml_free(c);
    return ok;
}

AmevaSdStatus ameva_sd_load_ex(
    const AmevaSdLoadOptions* opts,
    AmevaSd** out_sd,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_sd != nullptr) *out_sd = nullptr;
    if (opts == nullptr || out_sd == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: invalid argument (%s)", "NULL pointer");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (opts->struct_size != sizeof(AmevaSdLoadOptions)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: %s",
               "AmevaSdLoadOptions.struct_size mismatch (call ameva_sd_load_options_init)");
        return AMEVA_SD_ERR_ABI_MISMATCH;
    }
    if (!nonEmpty(opts->model_path) && !nonEmpty(opts->diffusion_model_path)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: invalid argument (%s)",
               "model_path or diffusion_model_path is required");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    const bool cpuOnly = (opts->budget_mode == AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY);
    if (!cpuOnly && !nonEmpty(opts->backend_plugin_path)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: invalid argument (%s)",
               "backend_plugin_path is required unless budget_mode is CPU_ONLY");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (opts->budget_fraction < 0.f || opts->budget_fraction > 1.f || !std::isfinite(opts->budget_fraction)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: invalid argument (%s)",
               "budget_fraction must be within [0, 1]");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }

    // Must be set before ggml_backend_load() below: ggml-ameva reads these once, at first use,
    // into a function-local static (setting them after the plugin is loaded has no effect).
    if (nonEmpty(opts->cache_dir)) {
        std::string pipelineCache = std::string(opts->cache_dir) + "/ggml-ameva-pipelines.bin";
        std::string qmvTune = std::string(opts->cache_dir) + "/ggml-ameva-qmv-tune.txt";
        setenv("AMEVA_PIPELINE_CACHE", pipelineCache.c_str(), 1);
        setenv("AMEVA_QMV_TUNE_FILE", qmvTune.c_str(), 1);
    }

    installLogCallbackOnce();

    auto* sd = new AmevaSd();

    if (!cpuOnly) {
        sd->backend_reg = loadPluginOnce(opts->backend_plugin_path, err_buf, err_buf_size);
        if (sd->backend_reg == nullptr) {
            freeSd(sd);
            return AMEVA_SD_ERR_BACKEND_LOAD_FAILED;
        }
        if (ggml_backend_reg_dev_count(sd->backend_reg) == 0) {
            setMsg(err_buf, err_buf_size, "ggml-ameva plugin %s registered zero devices", opts->backend_plugin_path);
            freeSd(sd);
            return AMEVA_SD_ERR_BACKEND_LOAD_FAILED;
        }
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = nonEmpty(opts->model_path) ? opts->model_path : nullptr;
    params.diffusion_model_path = nonEmpty(opts->diffusion_model_path) ? opts->diffusion_model_path : nullptr;
    params.llm_path = nonEmpty(opts->llm_path) ? opts->llm_path : nullptr;
    params.clip_l_path = nonEmpty(opts->clip_l_path) ? opts->clip_l_path : nullptr;
    params.t5xxl_path = nonEmpty(opts->t5xxl_path) ? opts->t5xxl_path : nullptr;
    params.vae_path = nonEmpty(opts->vae_path) ? opts->vae_path : nullptr;
    params.taesd_path = nonEmpty(opts->taesd_path) ? opts->taesd_path : nullptr;
    params.clip_vision_path = nonEmpty(opts->clip_vision_path) ? opts->clip_vision_path : nullptr;
    params.ip_adapter_path = nonEmpty(opts->ip_adapter_path) ? opts->ip_adapter_path : nullptr;
    params.control_net_path = nonEmpty(opts->control_net_path) ? opts->control_net_path : nullptr;
    params.motion_module_path = nonEmpty(opts->motion_module_path) ? opts->motion_module_path : nullptr;
    params.photo_maker_path = nonEmpty(opts->photo_maker_path) ? opts->photo_maker_path : nullptr;
    if (nonEmpty(opts->ip_adapter_path) && !nonEmpty(opts->clip_vision_path)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load_ex: invalid argument (%s)",
               "ip_adapter_path requires clip_vision_path (CLIP-Vision ViT-H/14)");
        freeSd(sd);
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    params.n_threads = opts->n_threads > 0 ? opts->n_threads : -1; // -1: stable-diffusion.cpp default
    params.flash_attn = false; // same defense-in-depth as ameva_llm_api.cpp/ameva_stt_api.cpp
    params.enable_mmap = opts->enable_mmap != 0;

    // Explicit per-module placement. The spec strings must outlive new_sd_ctx() (raw pointers).
    std::string backendSpec;
    std::string paramsSpec;
    std::string maxVramSpec;
    bool streamedDiffusion = false; // GPU_THEN_CPU stream tier: diffusion weights in RAM, computed on the GPU
    if (cpuOnly) {
        params.backend = "CPU";
        params.diffusion_conv_direct = true;
        params.vae_conv_direct = true;
    } else {
        const bool teCpu = (opts->text_encoder_placement == AMEVA_SD_PLACEMENT_CPU);
        const bool vaeCpu = (opts->vae_placement == AMEVA_SD_PLACEMENT_CPU);
        if (!teCpu && !vaeCpu) {
            backendSpec = "AMEVA0";
        } else {
            backendSpec = std::string("diffusion=AMEVA0,te=") + (teCpu ? "CPU" : "AMEVA0") +
                          ",vae=" + (vaeCpu ? "CPU" : "AMEVA0");
        }
        params.backend = backendSpec.c_str();
        // ggml-ameva runs CONV_2D natively. Without conv-direct, stable-diffusion.cpp lowers every
        // convolution to IM2COL + MUL_MAT; at 512x512 those IM2COL tensors exceed a storage-buffer
        // binding, are declined, and the graph is split across GPU and CPU (measured on S25 with SDXS:
        // sampling 26.5 s and TAESD decode 16.3 s, against 1.9 s and 0.76 s with conv-direct).
        // Asked of the device: ggml-ameva's OpenCL backend declines CONV_2D, and the IM2COL path is used there.
        const bool convOnDevice = deviceRunsConv2d(ggml_backend_reg_dev_get(sd->backend_reg, 0));
        params.diffusion_conv_direct = convOnDevice;
        // Flash attention in the diffusion model: without it a 512x512 SD1.5 self-attention node needs a ~1 GiB
        // score buffer that graph-cut cannot split (S20: allocated once, then "failed to allocate" on every later
        // generation). Enabled only when the device accepts the op for every probed head size.
        const bool faOnDevice = deviceRunsFlashAttn(ggml_backend_reg_dev_get(sd->backend_reg, 0));
        params.diffusion_flash_attn = faOnDevice;
        params.vae_conv_direct = convOnDevice && !vaeCpu;
#if defined(__ANDROID__)
        __android_log_print(ANDROID_LOG_INFO, "ameva_sd", "conv-direct %s, diffusion flash-attention %s",
                            convOnDevice ? "on" : "off (device declines CONV_2D)",
                            faOnDevice ? "on" : "off (device declines FLASH_ATTN_EXT)");
#endif
        if (opts->diffusion_params == AMEVA_SD_PARAMS_DISK) {
            paramsSpec = "diffusion=disk";
            params.params_backend = paramsSpec.c_str();
        }

        if (opts->budget_mode == AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY ||
            opts->budget_mode == AMEVA_SD_GPU_BUDGET_MODE_GPU_THEN_CPU) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_t ameva_dev = ggml_backend_reg_dev_get(sd->backend_reg, 0);
            ggml_backend_dev_memory(ameva_dev, &free_bytes, &total_bytes);
            const float fraction = (opts->budget_fraction > 0.f) ? opts->budget_fraction : 0.95f;
            const bool gpuOnly = (opts->budget_mode == AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY);
            constexpr int64_t kSafetyReserveBytes = 512LL * 1024 * 1024; // matches backend_fit's own default reserve

            if (free_bytes == 0) {
                if (gpuOnly) {
                    setMsg(err_buf, err_buf_size,
                           "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: device memory could not be estimated "
                           "(%s); refusing an unmeasured full-GPU attempt", "/proc/meminfo unreadable");
                    freeSd(sd);
                    return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                }
                // GPU_THEN_CPU without a usable estimate: keep the explicit placement above.
                // Logged so the decision is visible.
#if defined(__ANDROID__)
                __android_log_write(ANDROID_LOG_WARN, "ameva_sd",
                    "GPU_THEN_CPU: free device memory unknown; auto-fit not applied, explicit placement used");
#endif
            } else {
                const int64_t budget = (int64_t)((double)free_bytes * (double)fraction) - kSafetyReserveBytes;
                if (gpuOnly) {
                    int64_t total = 0;
                    const char* failed = nullptr;
                    if (!sumFileSizes({opts->model_path, opts->diffusion_model_path, opts->llm_path,
                                       opts->clip_l_path, opts->t5xxl_path, opts->vae_path,
                                       opts->taesd_path, opts->clip_vision_path, opts->ip_adapter_path,
                                       opts->control_net_path, opts->motion_module_path,
                                       opts->photo_maker_path}, &total, &failed)) {
                        setMsg(err_buf, err_buf_size,
                               "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: could not stat model file (%s)", failed);
                        freeSd(sd);
                        return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                    }
                    if (total > budget) {
                        char reason[320];
                        std::snprintf(reason, sizeof(reason),
                            "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: model files %lld MiB exceed the %lld MiB "
                            "budget (free=%lld MiB, fraction=%.2f); refusing a partial/CPU spill",
                            (long long)(total / (1024 * 1024)), (long long)(budget / (1024 * 1024)),
                            (long long)(free_bytes / (1024 * 1024)), fraction);
                        setRaw(err_buf, err_buf_size, reason);
                        freeSd(sd);
                        return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                    }
                } else {
                    // GPU_THEN_CPU: stable-diffusion.cpp's component-level auto-fit (sd::backend_fit).
                    // Negative max_vram = "free minus N GiB". It logs its plan ("auto-fit plan ...")
                    // and switches heavy components to on-disk residency when they do not fit.
                    const double reserveGib =
                        (double)(free_bytes - (uint64_t)std::max<int64_t>(budget, 0)) / (1024.0 * 1024.0 * 1024.0);
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "-%.3f", reserveGib);
                    maxVramSpec = buf;
                    params.max_vram = maxVramSpec.c_str(); // graph-cut keeps each compute graph under the budget
                    // sd.cpp's auto-fit reserves a fixed 2 GiB of compute memory per DiT/conditioner (sized for
                    // Flux-class models), so on phones with a 1.5-2 GiB budget it moved even SDXS's 488 MB UNet to
                    // the CPU (A35/A53/S21: "auto-fit: --backend diffusion=cpu"). When every file plus a compute
                    // reserve fits, the explicit all-GPU placement above is kept and graph-cut bounds compute;
                    // auto-fit only plans models that do not fit.
                    // Headroom for buffers outside the cut graphs (graph-cut keeps each compute graph under max_vram):
                    // SDXS needs 64 MB (UNet) and 224 MB (TAESD) of compute at 512x512.
                    constexpr int64_t kComputeReserveBytes = 384LL * 1024 * 1024;
                    int64_t files = 0;
                    const char* failed = nullptr;
                    const bool sized = sumFileSizes({opts->model_path, opts->diffusion_model_path, opts->llm_path,
                                                     opts->clip_l_path, opts->t5xxl_path, opts->vae_path,
                                                     opts->taesd_path, opts->clip_vision_path, opts->ip_adapter_path,
                                                     opts->control_net_path, opts->motion_module_path,
                                                     opts->photo_maker_path}, &files, &failed);
                    const bool fits = sized && files + kComputeReserveBytes <= budget;
                    // Not all at once: time-share on the GPU. Every module's weights are loaded for its phase and
                    // freed after (sd.cpp params backend "disk"), so the peak is the largest phase: the diffusion
                    // model with what runs inside it (ControlNet, IP-Adapter, motion module), the text encoder(s),
                    // CLIP-Vision or the VAE. A single-file checkpoint counts whole (it also holds TE and VAE).
                    bool timeShare = false;
                    bool streamDiffusion = false;
                    std::string streamDetail;
                    int64_t peak = 0;
                    if (sized && !fits) {
                        auto sz = [](const char* p) -> int64_t {
                            struct stat st;
                            return (nonEmpty(p) && stat(p, &st) == 0) ? (int64_t)st.st_size : 0;
                        };
                        const int64_t inDiffusion = sz(opts->control_net_path) + sz(opts->ip_adapter_path) +
                                                    sz(opts->motion_module_path);
                        CheckpointParts parts;
                        if (!ggufParts(opts->model_path, &parts)) parts.diffusion = sz(opts->model_path); // counted whole
                        peak = std::max({parts.diffusion + inDiffusion, parts.text_encoder, parts.vae,
                                         sz(opts->diffusion_model_path) + inDiffusion,
                                         sz(opts->llm_path), sz(opts->clip_l_path) + sz(opts->t5xxl_path),
                                         sz(opts->clip_vision_path), sz(opts->vae_path), sz(opts->photo_maker_path)});
                        // Time-share keeps one module resident at a time; graph-cut sizes each segment by its
                        // params + compute + cut inputs/outputs against max_vram, so a smaller reserve suffices here
                        // (A35/A53 with Anything q8 + IP-Adapter: phase 1491 MiB, budget ~1830 MiB).
                        constexpr int64_t kTimeShareReserveBytes = 256LL * 1024 * 1024;
                        timeShare = peak + kTimeShareReserveBytes <= budget;
                        if (timeShare) {
                            paramsSpec = "diffusion=disk,te=disk,vae=disk";
                            if (nonEmpty(opts->clip_vision_path)) paramsSpec += ",clip_vision=disk";
                            if (nonEmpty(opts->control_net_path)) paramsSpec += ",controlnet=disk";
                            if (nonEmpty(opts->photo_maker_path)) paramsSpec += ",photomaker=disk";
                            params.params_backend = paramsSpec.c_str();
                        }
                        // The diffusion phase does not fit: stream it. Its weights stay in RAM (CPU params backend,
                        // mmap'd), the UNet still computes on the GPU, and sd.cpp's layer streaming keeps as many
                        // graph-cut segments resident as the budget allows and uploads the others per evaluation.
                        // The other phases time-share the GPU as above; one that does not fit by itself (A35/A53:
                        // CLIP-Vision ViT-H 1206 MiB) runs on the CPU. Before this tier, sd.cpp auto-fit put the whole
                        // SD1.5 UNet on the CPU there (Anything q8 + IP-Adapter: phase 1491 MiB, budget 1372-1456 MiB).
                        // Needs the stream budget (minus sd.cpp's 512 MiB residency safety) to hold more than one
                        // segment; below kMinStreamBudget the auto-fit plan is kept.
                        constexpr int64_t kMinStreamBudget = 1024LL * 1024 * 1024;
                        const bool unetModel = nonEmpty(opts->model_path) && parts.diffusion > 0 &&
                                               !nonEmpty(opts->diffusion_model_path) && !nonEmpty(opts->motion_module_path);
                        if (!timeShare && unetModel && budget >= kMinStreamBudget) {
                            auto fitsAlone = [&](int64_t bytes) { return bytes + kTimeShareReserveBytes <= budget; };
                            const bool teGpu = !teCpu && fitsAlone(parts.text_encoder);
                            const bool vaeGpu = !vaeCpu && fitsAlone(parts.vae) && fitsAlone(sz(opts->vae_path));
                            const bool visionGpu = fitsAlone(sz(opts->clip_vision_path));
                            const bool controlGpu = fitsAlone(sz(opts->control_net_path));
                            streamDiffusion = true;
                            backendSpec = std::string("AMEVA0,te=") + (teGpu ? "AMEVA0" : "CPU") +
                                          ",vae=" + (vaeGpu ? "AMEVA0" : "CPU");
                            paramsSpec = std::string("diffusion=CPU") + (teGpu ? ",te=disk" : "") + (vaeGpu ? ",vae=disk" : "");
                            if (nonEmpty(opts->clip_vision_path)) {
                                backendSpec += visionGpu ? ",clip_vision=AMEVA0" : ",clip_vision=CPU";
                                if (visionGpu) paramsSpec += ",clip_vision=disk";
                            }
                            if (nonEmpty(opts->control_net_path)) {
                                backendSpec += controlGpu ? ",controlnet=AMEVA0" : ",controlnet=CPU";
                                if (controlGpu) paramsSpec += ",controlnet=disk";
                            }
                            if (nonEmpty(opts->photo_maker_path)) backendSpec += ",photomaker=CPU";
                            params.backend = backendSpec.c_str();
                            params.params_backend = paramsSpec.c_str();
                            params.stream_layers = true;
                            streamedDiffusion = true;
                            params.vae_conv_direct = vaeGpu ? convOnDevice : true;
                            streamDetail = std::string("te ") + (teGpu ? "GPU" : "CPU") + ", vae " + (vaeGpu ? "GPU" : "CPU") +
                                           (nonEmpty(opts->clip_vision_path) ? std::string(", clip_vision ") + (visionGpu ? "GPU" : "CPU") : "") +
                                           (nonEmpty(opts->control_net_path) ? std::string(", controlnet ") + (controlGpu ? "GPU" : "CPU") : "");
                        }
                    }
                    params.auto_fit = !(fits || timeShare || streamDiffusion);
                    if (fits || timeShare || streamDiffusion) {
                        // Explicit placement: graph-cut budget as a positive size, capped at 1.5 GiB per cut graph
                        // (segment params + compute). S20 (Adreno 650, 5 GiB budget): uncut, the SD1.5 UNet asked
                        // for one 1.07 GiB compute buffer; the first generation got it, every later one failed to
                        // allocate it with 3.7 GB still available. Smaller cut graphs keep allocations modest.
                        constexpr double kMaxCutGraphGib = 1.5;
                        const double budgetGib = (double)budget / (1024.0 * 1024.0 * 1024.0);
                        char cap[64];
                        std::snprintf(cap, sizeof(cap), "%.3f", std::min(budgetGib, kMaxCutGraphGib));
                        maxVramSpec = cap;
                        params.max_vram = maxVramSpec.c_str();
                    }
#if defined(__ANDROID__)
                    __android_log_print(ANDROID_LOG_INFO, "ameva_sd",
                        "GPU_THEN_CPU: files %lld MiB (+%lld MiB reserve), largest phase %lld MiB (+256 MiB), budget %lld MiB"
                        " -> %s", (long long)(files / (1024 * 1024)), (long long)(kComputeReserveBytes / (1024 * 1024)),
                        (long long)(peak / (1024 * 1024)), (long long)(budget / (1024 * 1024)),
                        fits ? "all on the GPU, graph-cut"
                             : timeShare ? "GPU time-share (weights per phase), graph-cut"
                             : streamDiffusion ? "diffusion streamed to the GPU (weights in RAM), graph-cut"
                                               : "sd.cpp auto-fit plan");
                    if (streamDiffusion)
                        __android_log_print(ANDROID_LOG_INFO, "ameva_sd", "GPU_THEN_CPU stream: %s", streamDetail.c_str());
#endif
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_placement_mutex);
        g_last_memory_line.clear();
    }
    sd->ctx = new_sd_ctx(&params);
    {
        std::lock_guard<std::mutex> lock(g_placement_mutex);
        // A streamed diffusion model reports its weights as RAM (params backend) although it computes on the GPU.
        sd->placement = cpuOnly ? std::string("CPU_ONLY; ") + g_last_memory_line
                        : streamedDiffusion ? std::string("STREAM diffusion=GPU; ") + g_last_memory_line
                                            : g_last_memory_line;
    }
    if (sd->ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "new_sd_ctx failed for %s",
               nonEmpty(opts->model_path) ? opts->model_path : opts->diffusion_model_path);
        freeSd(sd);
        return AMEVA_SD_ERR_MODEL_LOAD_FAILED;
    }

    *out_sd = sd;
    return AMEVA_SD_OK;
}

AmevaSdStatus ameva_sd_load(
    const char* model_path,
    const char* backend_plugin_path,
    const char* cache_dir,
    int n_threads,
    const AmevaSdGpuBudgetOptions* gpu_budget_opts,
    AmevaSd** out_sd,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_sd != nullptr) *out_sd = nullptr;
    if (model_path == nullptr || backend_plugin_path == nullptr || out_sd == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_sd_load: invalid argument (%s)", "NULL pointer");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    AmevaSdLoadOptions opts;
    ameva_sd_load_options_init(&opts);
    opts.model_path = model_path;
    opts.backend_plugin_path = backend_plugin_path;
    opts.cache_dir = cache_dir;
    opts.n_threads = n_threads;
    if (gpu_budget_opts != nullptr) {
        opts.budget_mode = gpu_budget_opts->mode;
        opts.budget_fraction = gpu_budget_opts->budget_fraction;
    }
    return ameva_sd_load_ex(&opts, out_sd, err_buf, err_buf_size);
}

void ameva_sd_generate_params_init(AmevaSdGenerateParams* params) {
    if (params == nullptr) return;
    *params = AmevaSdGenerateParams{};
    params->struct_size = (uint32_t)sizeof(AmevaSdGenerateParams);
    params->width = 512;
    params->height = 512;
    params->steps = 20;
    params->seed = -1;
    params->strength = 0.75f;
    params->duty_cycle_percent = 100;
}

namespace {

/** Owns the buffers sd_img_gen_params_t points into for the duration of one call. */
struct ExtraInputs {
    std::vector<sd_lora_t> loras;
    std::vector<uint8_t> control;
};

/** Validates and fills LoRA / CLIP skip / ControlNet / IP-Adapter fields. */
AmevaSdStatus applyExtras(const AmevaSdGenerateParams* gp, sd_img_gen_params_t* params, ExtraInputs* keep,
                          char* err_buf, size_t err_buf_size) {
    if (gp->lora_count < 0 || (gp->lora_count > 0 && (gp->lora_paths == nullptr || gp->lora_multipliers == nullptr))) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "lora_paths/lora_multipliers/lora_count");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    for (int i = 0; i < gp->lora_count; ++i) {
        struct stat st;
        if (!nonEmpty(gp->lora_paths[i]) || stat(gp->lora_paths[i], &st) != 0) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: LoRA file not found (%s)", gp->lora_paths[i]);
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        if (!std::isfinite(gp->lora_multipliers[i])) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "LoRA multiplier not finite");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        sd_lora_t l = {};
        l.is_high_noise = false;
        l.multiplier = gp->lora_multipliers[i];
        l.path = gp->lora_paths[i];
        keep->loras.push_back(l);
    }
    if (!keep->loras.empty()) {
        params->loras = keep->loras.data();
        params->lora_count = (uint32_t)keep->loras.size();
    }
    if (gp->clip_skip > 0) params->clip_skip = gp->clip_skip;

    const size_t expected = (size_t)gp->width * (size_t)gp->height * 3;
    if (gp->control_rgb != nullptr) {
        if (gp->control_rgb_size != expected) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "control_rgb_size must be width*height*3");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        keep->control.assign(gp->control_rgb, gp->control_rgb + expected);
        params->control_image.width = (uint32_t)gp->width;
        params->control_image.height = (uint32_t)gp->height;
        params->control_image.channel = 3;
        params->control_image.data = keep->control.data();
        params->control_strength = gp->control_strength > 0.f ? gp->control_strength : 0.9f;
        if (gp->control_canny) {
            // sd-cli defaults: high 0.08, low 0.08, weak 0.8, strong 1.0, not inverted.
            if (!preprocess_canny(params->control_image, 0.08f, 0.08f, 0.8f, 1.0f, false)) {
                setMsg(err_buf, err_buf_size, "ameva_sd_generate: %s", "canny preprocessing failed");
                return AMEVA_SD_ERR_GENERATE_FAILED;
            }
        }
    }
    if (gp->ip_adapter_rgb != nullptr) {
        if (gp->ip_adapter_width <= 0 || gp->ip_adapter_height <= 0) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "ip_adapter_width/height");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        params->ip_adapter_image.width = (uint32_t)gp->ip_adapter_width;
        params->ip_adapter_image.height = (uint32_t)gp->ip_adapter_height;
        params->ip_adapter_image.channel = 3;
        params->ip_adapter_image.data = const_cast<uint8_t*>(gp->ip_adapter_rgb); // read-only in sd.cpp
        params->ip_adapter_strength = gp->ip_adapter_strength > 0.f ? gp->ip_adapter_strength : 1.0f;
    }
    if (gp->hires_scale > 1.f) {
        if (!std::isfinite(gp->hires_scale) || gp->hires_scale > 4.f) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "hires_scale must be in (1, 4]");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        params->hires.enabled = true;
        params->hires.scale = gp->hires_scale;
        params->hires.target_width = ((int)(gp->width * gp->hires_scale) / 8) * 8;
        params->hires.target_height = ((int)(gp->height * gp->hires_scale) / 8) * 8;
        params->hires.steps = gp->hires_steps > 0 ? gp->hires_steps : 0;
        params->hires.denoising_strength = gp->hires_denoise > 0.f ? gp->hires_denoise : 0.5f;
        if (nonEmpty(gp->hires_upscaler)) {
            enum sd_hires_upscaler_t u = str_to_sd_hires_upscaler(gp->hires_upscaler);
            if (u == SD_HIRES_UPSCALER_COUNT || u == SD_HIRES_UPSCALER_NONE) {
                setMsg(err_buf, err_buf_size, "ameva_sd_generate: unknown hires upscaler (%s)", gp->hires_upscaler);
                return AMEVA_SD_ERR_INVALID_ARGUMENT;
            }
            params->hires.upscaler = u;
        } else {
            params->hires.upscaler = SD_HIRES_UPSCALER_LATENT;
        }
    }
    return AMEVA_SD_OK;
}

} // namespace

AmevaSdStatus ameva_sd_generate(
    AmevaSd* sd,
    const AmevaSdGenerateParams* gp,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
) {
    if (sd == nullptr || sd->ctx == nullptr || gp == nullptr || out_rgb == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "NULL pointer");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (gp->struct_size != sizeof(AmevaSdGenerateParams)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: %s",
               "AmevaSdGenerateParams.struct_size mismatch (call ameva_sd_generate_params_init)");
        return AMEVA_SD_ERR_ABI_MISMATCH;
    }
    if (gp->prompt == nullptr || gp->width <= 0 || gp->height <= 0 || gp->width % 8 != 0 || gp->height % 8 != 0) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)",
               "prompt is NULL or width/height is not a positive multiple of 8");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(gp->cfg_scale) || !std::isfinite(gp->flow_shift)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "cfg_scale/flow_shift not finite");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (gp->duty_cycle_percent < 1 || gp->duty_cycle_percent > 100) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "duty_cycle_percent must be 1..100");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    const bool hires = gp->hires_scale > 1.f;
    const int outW = hires ? ((int)(gp->width * gp->hires_scale) / 8) * 8 : gp->width;
    const int outH = hires ? ((int)(gp->height * gp->hires_scale) / 8) * 8 : gp->height;
    const size_t expectedSize = (size_t)outW * (size_t)outH * 3;
    const size_t inputSize = (size_t)gp->width * (size_t)gp->height * 3;
    if (out_rgb_size != expectedSize) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: out_rgb_size must be outW*outH*3 (%s)",
               "no silent crop/stride mismatch");
        return AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL;
    }
    const bool img2img = gp->init_rgb != nullptr;
    if (img2img) {
        if (gp->init_rgb_size != inputSize) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)",
                   "init_rgb_size must be width*height*3");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        if (!(gp->strength > 0.f && gp->strength <= 1.f)) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: invalid argument (%s)", "strength must be in (0, 1]");
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
    }

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = gp->prompt;
    params.negative_prompt = nonEmpty(gp->negative_prompt) ? gp->negative_prompt : "";
    params.width = gp->width;
    params.height = gp->height;
    params.sample_params.sample_steps = gp->steps > 0 ? gp->steps : 20;
    if (gp->cfg_scale > 0.0f) params.sample_params.guidance.txt_cfg = gp->cfg_scale;
    if (gp->flow_shift > 0.0f) params.sample_params.flow_shift = gp->flow_shift;
    if (nonEmpty(gp->sampler)) {
        enum sample_method_t m = str_to_sample_method(gp->sampler);
        if (m == SAMPLE_METHOD_COUNT) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: unknown sampler (%s)", gp->sampler);
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        params.sample_params.sample_method = m;
    }
    if (nonEmpty(gp->scheduler)) {
        enum scheduler_t s = str_to_scheduler(gp->scheduler);
        if (s == SCHEDULER_COUNT) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate: unknown scheduler (%s)", gp->scheduler);
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        params.sample_params.scheduler = s;
    }
    params.seed = gp->seed;
    params.batch_count = 1;
    params.vae_tiling_params.enabled = gp->vae_tiling != 0;
    if (img2img) {
        params.init_image.width = (uint32_t)gp->width;
        params.init_image.height = (uint32_t)gp->height;
        params.init_image.channel = 3;
        params.init_image.data = const_cast<uint8_t*>(gp->init_rgb); // read-only inside sd.cpp
        params.strength = gp->strength;
        // sd.cpp starts img2img at step floor(steps x strength). With a one-step model (SDXS) and any strength below
        // 1 that is step 0 of [sigma_max, 0]: the whole step from sigma_max, the source image is drowned in noise and
        // the result ignores it (S21, SDXS strength 0.55: unrelated pictures for every seed). There the strength is
        // read as the noise level and the one step starts at that sigma instead.
        const int steps = params.sample_params.sample_steps;
        if (gp->strength < 1.f && (int)((float)steps * gp->strength) == 0) {
            params.sample_params.extra_sample_args = "strength_as_noise_level=true,force_first_sigma=true";
        }
    }
    ExtraInputs extras;
    {
        AmevaSdStatus st = applyExtras(gp, &params, &extras, err_buf, err_buf_size);
        if (st != AMEVA_SD_OK) return st;
    }

    std::lock_guard<std::mutex> lock(g_generate_mutex);

    sd->cancel_requested.store(false);
    sd_cancel_generation(sd->ctx, SD_CANCEL_RESET);

    ProgressTrampoline tramp;
    tramp.user_fn = gp->progress;
    tramp.user_data = gp->progress_user_data;
    tramp.duty_percent = gp->duty_cycle_percent;
    tramp.cancel = &sd->cancel_requested;
    sd_set_progress_callback(progressTrampoline, &tramp);

    sd_image_t* images = nullptr;
    int numImages = 0;
    const bool ok = generate_image(sd->ctx, &params, &images, &numImages);

    sd_set_progress_callback(nullptr, nullptr);
    reportGpuProfile(sd);

    if (sd->cancel_requested.load()) {
        if (images != nullptr) free_sd_images(images, numImages);
        setMsg(err_buf, err_buf_size, "ameva_sd_generate: %s", "cancelled");
        return AMEVA_SD_ERR_CANCELLED;
    }
    if (!ok || images == nullptr || numImages <= 0) {
        if (images != nullptr) free_sd_images(images, numImages);
        setMsg(err_buf, err_buf_size, "generate_image failed (%s)", "see logcat tag ameva_sd");
        return AMEVA_SD_ERR_GENERATE_FAILED;
    }

    const sd_image_t& img = images[0];
    AmevaSdStatus status = AMEVA_SD_OK;
    if (img.width != (uint32_t)outW || img.height != (uint32_t)outH || img.channel != 3 || img.data == nullptr) {
        setMsg(err_buf, err_buf_size, "generate_image returned unexpected shape (%s)", "width/height/channel mismatch");
        status = AMEVA_SD_ERR_GENERATE_FAILED;
    } else {
        std::memcpy(out_rgb, img.data, expectedSize);
    }
    free_sd_images(images, numImages);
    return status;
}

AmevaSdStatus ameva_sd_generate_video(
    AmevaSd* sd,
    const AmevaSdGenerateParams* gp,
    int frames,
    int fps,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    int* out_frames,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_frames != nullptr) *out_frames = 0;
    if (sd == nullptr || sd->ctx == nullptr || gp == nullptr || out_rgb == nullptr || out_frames == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: invalid argument (%s)", "NULL pointer");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (gp->struct_size != sizeof(AmevaSdGenerateParams)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: %s", "AmevaSdGenerateParams.struct_size mismatch");
        return AMEVA_SD_ERR_ABI_MISMATCH;
    }
    if (gp->prompt == nullptr || gp->width <= 0 || gp->height <= 0 || gp->width % 8 != 0 || gp->height % 8 != 0 ||
        frames < 1 || frames > 64 || fps < 1 || fps > 60) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: invalid argument (%s)",
               "prompt, size (multiple of 8), frames 1..64 or fps 1..60");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (gp->duty_cycle_percent < 1 || gp->duty_cycle_percent > 100 || !std::isfinite(gp->cfg_scale)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: invalid argument (%s)", "duty_cycle_percent or cfg_scale");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    const size_t frameSize = (size_t)gp->width * (size_t)gp->height * 3;
    if (out_rgb_size != frameSize * (size_t)frames) {
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: %s", "out_rgb_size must be frames*width*height*3");
        return AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL;
    }

    sd_vid_gen_params_t vp;
    sd_vid_gen_params_init(&vp);
    vp.prompt = gp->prompt;
    vp.negative_prompt = nonEmpty(gp->negative_prompt) ? gp->negative_prompt : "";
    vp.width = gp->width;
    vp.height = gp->height;
    vp.video_frames = frames;
    vp.fps = fps;
    vp.seed = gp->seed;
    vp.sample_params.sample_steps = gp->steps > 0 ? gp->steps : 20;
    if (gp->cfg_scale > 0.f) vp.sample_params.guidance.txt_cfg = gp->cfg_scale;
    if (nonEmpty(gp->sampler)) {
        enum sample_method_t m = str_to_sample_method(gp->sampler);
        if (m == SAMPLE_METHOD_COUNT) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: unknown sampler (%s)", gp->sampler);
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        vp.sample_params.sample_method = m;
    }
    if (nonEmpty(gp->scheduler)) {
        enum scheduler_t sc = str_to_scheduler(gp->scheduler);
        if (sc == SCHEDULER_COUNT) {
            setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: unknown scheduler (%s)", gp->scheduler);
            return AMEVA_SD_ERR_INVALID_ARGUMENT;
        }
        vp.sample_params.scheduler = sc;
    }
    vp.vae_tiling_params.enabled = gp->vae_tiling != 0;
    std::vector<sd_lora_t> loras;
    for (int i = 0; i < gp->lora_count; ++i) {
        sd_lora_t l = {};
        l.multiplier = gp->lora_multipliers[i];
        l.path = gp->lora_paths[i];
        loras.push_back(l);
    }
    if (!loras.empty()) { vp.loras = loras.data(); vp.lora_count = (uint32_t)loras.size(); }
    if (gp->clip_skip > 0) vp.clip_skip = gp->clip_skip;

    std::lock_guard<std::mutex> lock(g_generate_mutex);
    sd->cancel_requested.store(false);
    sd_cancel_generation(sd->ctx, SD_CANCEL_RESET);
    ProgressTrampoline tramp;
    tramp.user_fn = gp->progress;
    tramp.user_data = gp->progress_user_data;
    tramp.duty_percent = gp->duty_cycle_percent;
    tramp.cancel = &sd->cancel_requested;
    sd_set_progress_callback(progressTrampoline, &tramp);

    sd_image_t* out = nullptr;
    int n = 0;
    sd_audio_t* audio = nullptr;
    const bool ok = generate_video(sd->ctx, &vp, &out, &n, &audio);
    sd_set_progress_callback(nullptr, nullptr);
    if (audio != nullptr) free_sd_audio(audio);

    if (sd->cancel_requested.load()) {
        if (out != nullptr) free_sd_images(out, n);
        setMsg(err_buf, err_buf_size, "ameva_sd_generate_video: %s", "cancelled");
        return AMEVA_SD_ERR_CANCELLED;
    }
    if (!ok || out == nullptr || n <= 0) {
        if (out != nullptr) free_sd_images(out, n);
        setMsg(err_buf, err_buf_size, "generate_video failed (%s)", "see logcat tag ameva_sd; motion module loaded?");
        return AMEVA_SD_ERR_GENERATE_FAILED;
    }
    const int count = std::min(n, frames);
    AmevaSdStatus status = AMEVA_SD_OK;
    for (int i = 0; i < count; ++i) {
        const sd_image_t& f = out[i];
        if (f.width != (uint32_t)gp->width || f.height != (uint32_t)gp->height || f.channel != 3 || f.data == nullptr) {
            setMsg(err_buf, err_buf_size, "generate_video returned unexpected shape (%s)", "width/height/channel mismatch");
            status = AMEVA_SD_ERR_GENERATE_FAILED;
            break;
        }
        std::memcpy(out_rgb + frameSize * (size_t)i, f.data, frameSize);
    }
    free_sd_images(out, n);
    if (status == AMEVA_SD_OK) *out_frames = count;
    return status;
}

struct AmevaSdUpscaler {
    upscaler_ctx_t* ctx = nullptr;
    int factor = 0;
};

AmevaSdStatus ameva_sd_upscaler_load(
    const char* model_path,
    const char* backend_plugin_path,
    int n_threads,
    int tile_size,
    AmevaSdUpscaler** out_upscaler,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_upscaler != nullptr) *out_upscaler = nullptr;
    if (!nonEmpty(model_path) || out_upscaler == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_sd_upscaler_load: invalid argument (%s)", "model_path");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    installLogCallbackOnce();
    const char* backend = "CPU";
    if (nonEmpty(backend_plugin_path)) {
        if (loadPluginOnce(backend_plugin_path, err_buf, err_buf_size) == nullptr) return AMEVA_SD_ERR_BACKEND_LOAD_FAILED;
        backend = "AMEVA0";
    }
    upscaler_ctx_t* ctx = new_upscaler_ctx(model_path, false, n_threads > 0 ? n_threads : -1,
                                           tile_size > 0 ? tile_size : 128, backend, nullptr);
    if (ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "new_upscaler_ctx failed for %s", model_path);
        return AMEVA_SD_ERR_MODEL_LOAD_FAILED;
    }
    auto* u = new AmevaSdUpscaler();
    u->ctx = ctx;
    u->factor = get_upscale_factor(ctx);
    if (u->factor <= 0) {
        setMsg(err_buf, err_buf_size, "upscaler reported factor <= 0 for %s", model_path);
        free_upscaler_ctx(ctx);
        delete u;
        return AMEVA_SD_ERR_MODEL_LOAD_FAILED;
    }
    *out_upscaler = u;
    return AMEVA_SD_OK;
}

int ameva_sd_upscaler_factor(const AmevaSdUpscaler* upscaler) {
    return upscaler != nullptr ? upscaler->factor : 0;
}

AmevaSdStatus ameva_sd_upscale(
    AmevaSdUpscaler* upscaler,
    const uint8_t* rgb,
    int width,
    int height,
    uint8_t* out_rgb,
    size_t out_rgb_size,
    char* err_buf,
    size_t err_buf_size
) {
    if (upscaler == nullptr || rgb == nullptr || out_rgb == nullptr || width <= 0 || height <= 0) {
        setMsg(err_buf, err_buf_size, "ameva_sd_upscale: invalid argument (%s)", "NULL pointer or size");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    const int f = upscaler->factor;
    const size_t expected = (size_t)width * f * (size_t)height * f * 3;
    if (out_rgb_size != expected) {
        setMsg(err_buf, err_buf_size, "ameva_sd_upscale: %s", "out_rgb_size must be (w*f)*(h*f)*3");
        return AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL;
    }
    sd_image_t in = {(uint32_t)width, (uint32_t)height, 3, const_cast<uint8_t*>(rgb)};
    sd_image_t* out = nullptr;
    int n = 0;
    std::lock_guard<std::mutex> lock(g_generate_mutex);
    if (!upscale(upscaler->ctx, in, (uint32_t)f, &out, &n) || out == nullptr || n <= 0) {
        if (out != nullptr) free_sd_images(out, n);
        setMsg(err_buf, err_buf_size, "upscale failed (%s)", "see logcat tag ameva_sd");
        return AMEVA_SD_ERR_GENERATE_FAILED;
    }
    AmevaSdStatus status = AMEVA_SD_OK;
    if (out[0].width != (uint32_t)(width * f) || out[0].height != (uint32_t)(height * f) || out[0].channel != 3) {
        setMsg(err_buf, err_buf_size, "upscale returned unexpected shape (%s)", "width/height/channel");
        status = AMEVA_SD_ERR_GENERATE_FAILED;
    } else {
        std::memcpy(out_rgb, out[0].data, expected);
    }
    free_sd_images(out, n);
    return status;
}

void ameva_sd_upscaler_free(AmevaSdUpscaler* upscaler) {
    if (upscaler == nullptr) return;
    if (upscaler->ctx != nullptr) free_upscaler_ctx(upscaler->ctx);
    delete upscaler;
}

const char* ameva_sd_placement(const AmevaSd* sd) {
    return (sd != nullptr) ? sd->placement.c_str() : "";
}

void ameva_sd_cancel(AmevaSd* sd) {
    if (sd == nullptr || sd->ctx == nullptr) return;
    sd->cancel_requested.store(true);
    sd_cancel_generation(sd->ctx, SD_CANCEL_ALL);
}

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
) {
    return ameva_sd_txt2img_ex(sd, prompt, width, height, steps, 0.0f, seed, out_rgb, out_rgb_size,
                               err_buf, err_buf_size);
}

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
) {
    if (sd == nullptr || sd->ctx == nullptr || prompt == nullptr || out_rgb == nullptr ||
        width <= 0 || height <= 0) {
        setMsg(err_buf, err_buf_size, "ameva_sd_txt2img: invalid argument (%s)", "NULL pointer or non-positive size");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(cfg_scale)) {
        setMsg(err_buf, err_buf_size, "ameva_sd_txt2img: invalid argument (%s)", "cfg_scale is not finite");
        return AMEVA_SD_ERR_INVALID_ARGUMENT;
    }
    const size_t expectedSize = (size_t)width * (size_t)height * 3;
    if (out_rgb_size != expectedSize) {
        setMsg(err_buf, err_buf_size, "ameva_sd_txt2img: out_rgb_size must be width*height*3 (%s)",
               "no silent crop/stride mismatch");
        return AMEVA_SD_ERR_OUTPUT_BUFFER_TOO_SMALL;
    }

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt;
    params.width = width;
    params.height = height;
    params.sample_params.sample_steps = steps > 0 ? steps : 20;
    if (cfg_scale > 0.0f) {
        params.sample_params.guidance.txt_cfg = cfg_scale;
    }
    params.seed = seed;
    params.batch_count = 1;

    std::lock_guard<std::mutex> lock(g_generate_mutex);

    sd_image_t* images = nullptr;
    int numImages = 0;
    if (!generate_image(sd->ctx, &params, &images, &numImages) || images == nullptr || numImages <= 0) {
        setMsg(err_buf, err_buf_size, "generate_image failed for prompt (%s)", prompt);
        return AMEVA_SD_ERR_GENERATE_FAILED;
    }

    const sd_image_t& img = images[0];
    AmevaSdStatus status = AMEVA_SD_OK;
    if (img.width != (uint32_t)width || img.height != (uint32_t)height || img.channel != 3 || img.data == nullptr) {
        setMsg(err_buf, err_buf_size, "generate_image returned unexpected shape (%s)", "width/height/channel mismatch");
        status = AMEVA_SD_ERR_GENERATE_FAILED;
    } else {
        std::memcpy(out_rgb, img.data, expectedSize);
    }
    free_sd_images(images, numImages);
    return status;
}

void ameva_sd_free(AmevaSd* sd) {
    freeSd(sd);
}
