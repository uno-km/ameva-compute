/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_sd_api.h"

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
    if (sd == nullptr || sd->ctx == nullptr || prompt == nullptr || out_rgb == nullptr ||
        width <= 0 || height <= 0) {
        setMsg(err_buf, err_buf_size, "ameva_sd_txt2img: invalid argument (%s)", "NULL pointer or non-positive size");
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
