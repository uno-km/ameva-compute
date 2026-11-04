/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * JNI binding for ameva_sd_api.h. Throws com.ameva.amuda.AmudaException on every
 * AmevaSdStatus != AMEVA_SD_OK -- same convention as ameva_llm_jni.cpp/ameva_stt_jni.cpp.
 */

#include <jni.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ameva_sd_api.h"

namespace {

void throwAmudaException(JNIEnv* env, AmevaSdStatus code, const char* detail) {
    jclass cls = env->FindClass("com/ameva/amuda/AmudaException");
    if (cls == nullptr) return; /* FindClass already threw NoClassDefFoundError */
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
    jstring jmsg = env->NewStringUTF(detail != nullptr ? detail : "(no detail)");
    jthrowable ex = (jthrowable)env->NewObject(cls, ctor, (jint)code, jmsg);
    env->Throw(ex);
}

std::string jstringToUtf8(JNIEnv* env, jstring s) {
    if (s == nullptr) return std::string();
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

std::vector<uint8_t> bytes(JNIEnv* env, jbyteArray a) {
    std::vector<uint8_t> v;
    if (a == nullptr) return v;
    jsize n = env->GetArrayLength(a);
    v.resize((size_t)n);
    env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte*>(v.data()));
    return v;
}

struct JniProgress {
    JNIEnv* env = nullptr;          // valid only on the generating thread, which is the caller's
    jobject listener = nullptr;     // local ref owned by the caller frame
    jmethodID onProgress = nullptr;
    AmevaSd* sd = nullptr;
    jthrowable pending = nullptr;   // global ref: listener exception, rethrown after generate returns
};

void jniProgressFn(int step, int steps, float seconds, void* data) {
    auto* p = static_cast<JniProgress*>(data);
    if (p == nullptr || p->listener == nullptr || p->pending != nullptr) return;
    p->env->CallVoidMethod(p->listener, p->onProgress, (jint)step, (jint)steps, (jfloat)seconds);
    if (p->env->ExceptionCheck()) {
        // Not swallowed: kept, the generation is cancelled, and it is rethrown to the caller.
        jthrowable ex = p->env->ExceptionOccurred();
        p->env->ExceptionClear();
        p->pending = (jthrowable)p->env->NewGlobalRef(ex);
        p->env->DeleteLocalRef(ex);
        ameva_sd_cancel(p->sd);
    }
}

/**
 * Reads com.ameva.amuda.diffusion.AmevaDiffusion$GenerateParams (Kotlin `var` properties compile to
 * same-named fields) into an AmevaSdGenerateParams whose pointers stay valid while this object lives.
 */
struct GenInputs {
    std::string prompt, negative, sampler, scheduler, hiresUpscaler;
    int outW = 0, outH = 0;
    std::vector<uint8_t> init, control, ip;
    std::vector<std::string> loraPathStore;
    std::vector<const char*> loraPaths;
    std::vector<float> loraMults;
    AmevaSdGenerateParams gp;

    bool read(JNIEnv* env, jobject p) {
        ameva_sd_generate_params_init(&gp);
        jclass c = env->GetObjectClass(p);
        auto fI = [&](const char* n) { return env->GetIntField(p, env->GetFieldID(c, n, "I")); };
        auto fF = [&](const char* n) { return env->GetFloatField(p, env->GetFieldID(c, n, "F")); };
        auto fZ = [&](const char* n) { return env->GetBooleanField(p, env->GetFieldID(c, n, "Z")) == JNI_TRUE; };
        auto fS = [&](const char* n) { return jstringToUtf8(env, (jstring)env->GetObjectField(p, env->GetFieldID(c, n, "Ljava/lang/String;"))); };
        auto fB = [&](const char* n) { return (jbyteArray)env->GetObjectField(p, env->GetFieldID(c, n, "[B")); };
        prompt = fS("prompt");
        negative = fS("negativePrompt");
        sampler = fS("sampler");
        scheduler = fS("scheduler");
        if (env->ExceptionCheck()) return false;
        gp.prompt = prompt.c_str();
        gp.negative_prompt = negative.c_str();
        gp.sampler = sampler.c_str();
        gp.scheduler = scheduler.c_str();
        gp.width = fI("width");
        gp.height = fI("height");
        gp.steps = fI("steps");
        gp.cfg_scale = fF("cfgScale");
        gp.flow_shift = fF("flowShift");
        gp.seed = (int64_t)env->GetLongField(p, env->GetFieldID(c, "seed", "J"));
        jbyteArray initA = fB("initRgb");
        if (initA != nullptr) {
            init = bytes(env, initA);
            gp.init_rgb = init.data();
            gp.init_rgb_size = init.size();
            gp.strength = fF("strength");
        }
        gp.vae_tiling = fZ("vaeTiling") ? 1 : 0;
        gp.duty_cycle_percent = fI("dutyCyclePercent");
        gp.clip_skip = fI("clipSkip");
        jobjectArray lp = (jobjectArray)env->GetObjectField(p, env->GetFieldID(c, "loraPaths", "[Ljava/lang/String;"));
        jfloatArray lm = (jfloatArray)env->GetObjectField(p, env->GetFieldID(c, "loraMultipliers", "[F"));
        if (env->ExceptionCheck()) return false;
        if (lp != nullptr && lm != nullptr) {
            jsize n = env->GetArrayLength(lp);
            if (env->GetArrayLength(lm) != n) return false;
            loraMults.resize((size_t)n);
            env->GetFloatArrayRegion(lm, 0, n, loraMults.data());
            for (jsize i = 0; i < n; ++i) loraPathStore.push_back(jstringToUtf8(env, (jstring)env->GetObjectArrayElement(lp, i)));
            for (auto& s : loraPathStore) loraPaths.push_back(s.c_str());
            gp.lora_paths = loraPaths.data();
            gp.lora_multipliers = loraMults.data();
            gp.lora_count = (int)n;
        }
        jbyteArray ctlA = fB("controlRgb");
        if (ctlA != nullptr) {
            control = bytes(env, ctlA);
            gp.control_rgb = control.data();
            gp.control_rgb_size = control.size();
            gp.control_strength = fF("controlStrength");
            gp.control_canny = fZ("controlCanny") ? 1 : 0;
        }
        jbyteArray ipA = fB("ipAdapterRgb");
        if (ipA != nullptr) {
            ip = bytes(env, ipA);
            gp.ip_adapter_rgb = ip.data();
            gp.ip_adapter_width = fI("ipAdapterWidth");
            gp.ip_adapter_height = fI("ipAdapterHeight");
            gp.ip_adapter_strength = fF("ipAdapterStrength");
            if ((size_t)gp.ip_adapter_width * (size_t)gp.ip_adapter_height * 3 != ip.size()) return false;
        }
        gp.hires_scale = fF("hiresScale");
        gp.hires_steps = fI("hiresSteps");
        gp.hires_denoise = fF("hiresDenoise");
        hiresUpscaler = fS("hiresUpscaler");
        gp.hires_upscaler = hiresUpscaler.c_str();
        if (gp.hires_scale > 1.f) {
            outW = ((int)(gp.width * gp.hires_scale) / 8) * 8;
            outH = ((int)(gp.height * gp.hires_scale) / 8) * 8;
        } else {
            outW = gp.width;
            outH = gp.height;
        }
        env->DeleteLocalRef(c);
        return !env->ExceptionCheck();
    }
};

bool bindListener(JNIEnv* env, jobject listener, JniProgress* prog) {
    if (listener == nullptr) return true;
    jclass lcls = env->GetObjectClass(listener);
    prog->onProgress = env->GetMethodID(lcls, "onProgress", "(IIF)V");
    env->DeleteLocalRef(lcls);
    if (prog->onProgress == nullptr) return false; // NoSuchMethodError already pending
    prog->listener = listener;
    return true;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeLoad(JNIEnv* env, jclass, jstring modelPath,
                                                           jstring backendPluginPath, jstring cacheDir,
                                                           jint nThreads, jint gpuBudgetMode,
                                                           jfloat gpuBudgetFraction) {
    std::string model = jstringToUtf8(env, modelPath);
    std::string backendPlugin = jstringToUtf8(env, backendPluginPath);
    std::string cache = jstringToUtf8(env, cacheDir);

    // gpuBudgetMode mirrors AmevaSdGpuBudgetMode's declaration order (0=DEFAULT ... 3=CPU_ONLY).
    AmevaSdGpuBudgetOptions budgetOpts = {};
    budgetOpts.mode = (AmevaSdGpuBudgetMode)gpuBudgetMode;
    budgetOpts.budget_fraction = (float)gpuBudgetFraction;
    const AmevaSdGpuBudgetOptions* budgetOptsPtr =
        (budgetOpts.mode == AMEVA_SD_GPU_BUDGET_MODE_DEFAULT) ? nullptr : &budgetOpts;

    AmevaSd* sd = nullptr;
    char errBuf[256] = {0};
    AmevaSdStatus status = ameva_sd_load(model.c_str(), backendPlugin.c_str(),
                                          cache.empty() ? nullptr : cache.c_str(), (int)nThreads,
                                          budgetOptsPtr, &sd, errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)sd;
}

JNIEXPORT jbyteArray JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeTxt2Img(JNIEnv* env, jclass, jlong handle,
                                                              jstring prompt, jint width, jint height,
                                                              jint steps, jfloat cfgScale, jlong seed) {
    AmevaSd* sd = (AmevaSd*)(intptr_t)handle;
    std::string promptStr = jstringToUtf8(env, prompt);

    size_t rgbSize = (size_t)width * (size_t)height * 3;
    std::vector<uint8_t> rgb(rgbSize);
    char errBuf[256] = {0};
    AmevaSdStatus status = ameva_sd_txt2img_ex(
        sd, promptStr.c_str(), (int)width, (int)height, (int)steps, (float)cfgScale, (int64_t)seed,
        rgb.data(), rgb.size(), errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return nullptr;
    }

    jbyteArray result = env->NewByteArray((jsize)rgbSize);
    env->SetByteArrayRegion(result, 0, (jsize)rgbSize, reinterpret_cast<const jbyte*>(rgb.data()));
    return result;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeFree(JNIEnv*, jclass, jlong handle) {
    ameva_sd_free((AmevaSd*)(intptr_t)handle);
}

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeLoadEx(
    JNIEnv* env, jclass, jobjectArray paths, jstring backendPluginPath, jstring cacheDir,
    jint nThreads, jint budgetMode, jfloat budgetFraction, jint tePlacement, jint vaePlacement,
    jint diffusionParams, jboolean enableMmap) {
    // paths order (AmevaDiffusion.kt LoadOptions.pathArray): model, diffusion, llm, clipL, t5xxl, vae,
    // taesd, clipVision, ipAdapter, controlNet, motionModule, photoMaker.
    const int kPaths = 12;
    if (paths == nullptr || env->GetArrayLength(paths) != kPaths) {
        throwAmudaException(env, AMEVA_SD_ERR_INVALID_ARGUMENT, "nativeLoadEx: path array must have 12 entries");
        return 0;
    }
    std::string p[kPaths];
    for (int i = 0; i < kPaths; ++i) p[i] = jstringToUtf8(env, (jstring)env->GetObjectArrayElement(paths, i));
    std::string plugin = jstringToUtf8(env, backendPluginPath);
    std::string cache = jstringToUtf8(env, cacheDir);

    AmevaSdLoadOptions opts;
    ameva_sd_load_options_init(&opts);
    opts.model_path = p[0].c_str();
    opts.diffusion_model_path = p[1].c_str();
    opts.llm_path = p[2].c_str();
    opts.clip_l_path = p[3].c_str();
    opts.t5xxl_path = p[4].c_str();
    opts.vae_path = p[5].c_str();
    opts.taesd_path = p[6].c_str();
    opts.clip_vision_path = p[7].c_str();
    opts.ip_adapter_path = p[8].c_str();
    opts.control_net_path = p[9].c_str();
    opts.motion_module_path = p[10].c_str();
    opts.photo_maker_path = p[11].c_str();
    opts.backend_plugin_path = plugin.c_str();
    opts.cache_dir = cache.c_str();
    opts.n_threads = (int)nThreads;
    // Ordinals mirror the Kotlin enums' declaration order (GpuBudgetMode, Placement, ParamsResidency).
    opts.budget_mode = (AmevaSdGpuBudgetMode)budgetMode;
    opts.budget_fraction = (float)budgetFraction;
    opts.text_encoder_placement = (AmevaSdPlacement)tePlacement;
    opts.vae_placement = (AmevaSdPlacement)vaePlacement;
    opts.diffusion_params = (AmevaSdParamsResidency)diffusionParams;
    opts.enable_mmap = enableMmap ? 1 : 0;

    AmevaSd* sd = nullptr;
    char errBuf[512] = {0};
    AmevaSdStatus status = ameva_sd_load_ex(&opts, &sd, errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)sd;
}

JNIEXPORT jbyteArray JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeGenerate(
    JNIEnv* env, jclass, jlong handle, jobject params, jobject listener, jint videoFrames, jint fps) {
    AmevaSd* sd = (AmevaSd*)(intptr_t)handle;
    GenInputs in;
    if (!in.read(env, params)) {
        if (!env->ExceptionCheck()) {
            throwAmudaException(env, AMEVA_SD_ERR_INVALID_ARGUMENT,
                                "GenerateParams: loraPaths/loraMultipliers length or ipAdapter size mismatch");
        }
        return nullptr;
    }
    JniProgress prog;
    prog.env = env;
    prog.sd = sd;
    if (!bindListener(env, listener, &prog)) return nullptr;
    in.gp.progress = listener != nullptr ? jniProgressFn : nullptr;
    in.gp.progress_user_data = &prog;

    const bool video = videoFrames > 0;
    const size_t frameSize = (size_t)(video ? in.gp.width : in.outW) * (size_t)(video ? in.gp.height : in.outH) * 3;
    std::vector<uint8_t> rgb(frameSize * (size_t)(video ? videoFrames : 1));
    char errBuf[512] = {0};
    int framesOut = 1;
    AmevaSdStatus status = video
        ? ameva_sd_generate_video(sd, &in.gp, (int)videoFrames, (int)fps, rgb.data(), rgb.size(), &framesOut,
                                  errBuf, sizeof(errBuf))
        : ameva_sd_generate(sd, &in.gp, rgb.data(), rgb.size(), errBuf, sizeof(errBuf));

    if (prog.pending != nullptr) {
        env->Throw(prog.pending);
        env->DeleteGlobalRef(prog.pending);
        return nullptr;
    }
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return nullptr;
    }
    const size_t outSize = frameSize * (size_t)framesOut;
    jbyteArray result = env->NewByteArray((jsize)outSize);
    if (result == nullptr) return nullptr; // OutOfMemoryError pending
    env->SetByteArrayRegion(result, 0, (jsize)outSize, reinterpret_cast<const jbyte*>(rgb.data()));
    return result;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeCancel(JNIEnv*, jclass, jlong handle) {
    ameva_sd_cancel((AmevaSd*)(intptr_t)handle);
}

JNIEXPORT jstring JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativePlacement(JNIEnv* env, jclass, jlong handle) {
    return env->NewStringUTF(ameva_sd_placement((const AmevaSd*)(intptr_t)handle));
}

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_diffusion_AmevaUpscaler_nativeLoad(JNIEnv* env, jclass, jstring modelPath,
                                                          jstring backendPluginPath, jint nThreads, jint tile) {
    std::string model = jstringToUtf8(env, modelPath);
    std::string plugin = jstringToUtf8(env, backendPluginPath);
    AmevaSdUpscaler* u = nullptr;
    char errBuf[512] = {0};
    AmevaSdStatus status = ameva_sd_upscaler_load(model.c_str(), plugin.c_str(), (int)nThreads, (int)tile, &u,
                                                  errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)u;
}

JNIEXPORT jint JNICALL
Java_com_ameva_amuda_diffusion_AmevaUpscaler_nativeFactor(JNIEnv*, jclass, jlong handle) {
    return (jint)ameva_sd_upscaler_factor((AmevaSdUpscaler*)(intptr_t)handle);
}

JNIEXPORT jbyteArray JNICALL
Java_com_ameva_amuda_diffusion_AmevaUpscaler_nativeUpscale(JNIEnv* env, jclass, jlong handle, jbyteArray rgbIn,
                                                             jint width, jint height) {
    auto* u = (AmevaSdUpscaler*)(intptr_t)handle;
    std::vector<uint8_t> in = bytes(env, rgbIn);
    if (in.size() != (size_t)width * (size_t)height * 3) {
        throwAmudaException(env, AMEVA_SD_ERR_INVALID_ARGUMENT, "rgb size must be width*height*3");
        return nullptr;
    }
    const int f = ameva_sd_upscaler_factor(u);
    const size_t outSize = (size_t)width * f * (size_t)height * f * 3;
    std::vector<uint8_t> out(outSize);
    char errBuf[512] = {0};
    AmevaSdStatus status = ameva_sd_upscale(u, in.data(), (int)width, (int)height, out.data(), out.size(),
                                            errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return nullptr;
    }
    jbyteArray result = env->NewByteArray((jsize)outSize);
    if (result == nullptr) return nullptr;
    env->SetByteArrayRegion(result, 0, (jsize)outSize, reinterpret_cast<const jbyte*>(out.data()));
    return result;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_diffusion_AmevaUpscaler_nativeFree(JNIEnv*, jclass, jlong handle) {
    ameva_sd_upscaler_free((AmevaSdUpscaler*)(intptr_t)handle);
}

} // extern "C"
