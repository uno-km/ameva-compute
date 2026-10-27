/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_sd_api.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "stable-diffusion.h"
#include "ggml-backend.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct AmevaSd {
    ggml_backend_reg_t backend_reg = nullptr;
    sd_ctx_t* ctx = nullptr;
};

namespace {

void setMsg(char* buf, size_t size, const char* fmt, const char* detail) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, fmt, detail != nullptr ? detail : "(no detail)");
}

#if defined(__ANDROID__)
void androidSdLogCallback(enum sd_log_level_t level, const char* text, void*) {
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
void installLogCallbackOnce() {}
#endif

void freeSd(AmevaSd* sd) {
    if (sd == nullptr) return;
    if (sd->ctx != nullptr) free_sd_ctx(sd->ctx);
    // Same lifecycle note as ameva_llm_api.cpp: the plugin .so stays mapped for the process
    // lifetime, there is no public ggml_backend_reg_t unload call in this ggml version.
    delete sd;
}

} // namespace

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

    // Must be set before ggml_backend_load() below: ggml-ameva reads these once, at first use,
    // into a function-local static (same reasoning as AMEVA_BACKEND in ggml-ameva.cpp -- setting
    // it after the plugin is already loaded has no effect for the rest of this process).
    if (cache_dir != nullptr && cache_dir[0] != '\0') {
        std::string pipelineCache = std::string(cache_dir) + "/ggml-ameva-pipelines.bin";
        std::string qmvTune = std::string(cache_dir) + "/ggml-ameva-qmv-tune.txt";
        setenv("AMEVA_PIPELINE_CACHE", pipelineCache.c_str(), 1);
        setenv("AMEVA_QMV_TUNE_FILE", qmvTune.c_str(), 1);
    }

    installLogCallbackOnce();

    auto* sd = new AmevaSd();

    sd->backend_reg = ggml_backend_load(backend_plugin_path);
    if (sd->backend_reg == nullptr) {
        setMsg(err_buf, err_buf_size, "ggml_backend_load failed for %s", backend_plugin_path);
        freeSd(sd);
        return AMEVA_SD_ERR_BACKEND_LOAD_FAILED;
    }
    if (ggml_backend_reg_dev_count(sd->backend_reg) == 0) {
        setMsg(err_buf, err_buf_size, "ggml-ameva plugin %s registered zero devices", backend_plugin_path);
        freeSd(sd);
        return AMEVA_SD_ERR_BACKEND_LOAD_FAILED;
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = model_path;
    params.backend = "AMEVA0"; // stable-diffusion.cpp resolves this by name against the ggml
                                // backend registry -- ggml_backend_load() above already put
                                // AMEVA0 in it, matching ggml_backend_ameva_device()'s GGML_AMEVA_DEVICE_NAME.
    params.n_threads = n_threads > 0 ? n_threads : -1; // -1: stable-diffusion.cpp's own default
    params.flash_attn = false; // same defense-in-depth as ameva_llm_api.cpp/ameva_stt_api.cpp

    // Opt-in only: gpu_budget_opts == NULL (or mode == DEFAULT) keeps params.backend == "AMEVA0"
    // exactly as set above -- the previous fixed, unconditional-full-GPU behavior. maxVramSpec
    // must stay alive through new_sd_ctx() below, since params.max_vram is a raw pointer into it.
    std::string maxVramSpec;
    if (gpu_budget_opts != nullptr && gpu_budget_opts->mode != AMEVA_SD_GPU_BUDGET_MODE_DEFAULT) {
        if (gpu_budget_opts->mode == AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY) {
            params.backend = "CPU";
        } else {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_t ameva_dev = ggml_backend_reg_dev_get(sd->backend_reg, 0);
            ggml_backend_dev_memory(ameva_dev, &free_bytes, &total_bytes);
            const float fraction = (gpu_budget_opts->budget_fraction > 0.f)
                                        ? gpu_budget_opts->budget_fraction : 0.95f;
            const bool gpuOnly = (gpu_budget_opts->mode == AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY);
            constexpr int64_t kSafetyReserveBytes = 512LL * 1024 * 1024; // matches backend_fit's own default reserve

            if (free_bytes == 0) {
                if (gpuOnly) {
                    setMsg(err_buf, err_buf_size,
                           "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: device memory could not be estimated "
                           "(%s); refusing an unmeasured full-GPU attempt", "/proc/meminfo unreadable");
                    freeSd(sd);
                    return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                }
                // GPU_THEN_CPU without a usable estimate: leave params.backend == "AMEVA0" (DEFAULT)
            } else {
                const int64_t budget = (int64_t)((double)free_bytes * (double)fraction) - kSafetyReserveBytes;
                if (gpuOnly) {
                    // Coarser than GPU_THEN_CPU's component-level auto-fit below: a single whole-file
                    // size check, since GPU_ONLY must refuse before new_sd_ctx() runs at all rather
                    // than discover a per-component shortfall partway through model loading.
                    struct stat st;
                    if (stat(model_path, &st) != 0) {
                        setMsg(err_buf, err_buf_size,
                               "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: could not stat model file (%s)",
                               model_path);
                        freeSd(sd);
                        return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                    }
                    if ((int64_t)st.st_size > budget) {
                        char reason[320];
                        std::snprintf(reason, sizeof(reason),
                            "AMEVA_SD_GPU_BUDGET_MODE_GPU_ONLY: model file %lld MiB exceeds the %lld MiB "
                            "budget (free=%lld MiB, fraction=%.2f); refusing a partial/CPU spill",
                            (long long)(st.st_size / (1024 * 1024)), (long long)(budget / (1024 * 1024)),
                            (long long)(free_bytes / (1024 * 1024)), fraction);
                        setMsg(err_buf, err_buf_size, "%s", reason);
                        freeSd(sd);
                        return AMEVA_SD_ERR_GPU_BUDGET_EXCEEDED;
                    }
                    // Fits: proceed unconditionally on "AMEVA0", same as DEFAULT.
                } else {
                    // GPU_THEN_CPU: hand off to stable-diffusion.cpp's own component-level auto-fit
                    // (third_party/stable-diffusion.cpp's sd::backend_fit, already used by the CLI's
                    // --auto-fit) by translating the fraction into its reserve-GiB syntax (negative
                    // value = "free minus N GiB", see ggml_graph_cut.cpp) -- more precise than this
                    // wrapper's own file-size estimate, and it logs its plan (LOG_INFO "auto-fit
                    // plan..."), never acting silently.
                    const double reserveGib =
                        (double)(free_bytes - (uint64_t)std::max<int64_t>(budget, 0)) / (1024.0 * 1024.0 * 1024.0);
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "-%.3f", reserveGib);
                    maxVramSpec = buf;
                    params.auto_fit = true;
                    params.max_vram = maxVramSpec.c_str();
                }
            }
        }
    }

    sd->ctx = new_sd_ctx(&params);
    if (sd->ctx == nullptr) {
        setMsg(err_buf, err_buf_size, "new_sd_ctx failed for %s", model_path);
        freeSd(sd);
        return AMEVA_SD_ERR_MODEL_LOAD_FAILED;
    }

    *out_sd = sd;
    return AMEVA_SD_OK;
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

    for (int i = 0; i < numImages; ++i) {
        free(images[i].data);
    }
    free(images);

    return status;
}

void ameva_sd_free(AmevaSd* sd) {
    freeSd(sd);
}
