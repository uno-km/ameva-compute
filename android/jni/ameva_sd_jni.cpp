/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * JNI binding for ameva_sd_api.h (v0.4). Throws com.ameva.amuda.AmudaException on every
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

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeLoad(JNIEnv* env, jclass, jstring modelPath,
                                                           jstring backendPluginPath, jint nThreads) {
    std::string model = jstringToUtf8(env, modelPath);
    std::string backendPlugin = jstringToUtf8(env, backendPluginPath);

    AmevaSd* sd = nullptr;
    char errBuf[256] = {0};
    AmevaSdStatus status = ameva_sd_load(model.c_str(), backendPlugin.c_str(), (int)nThreads,
                                          &sd, errBuf, sizeof(errBuf));
    if (status != AMEVA_SD_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)sd;
}

JNIEXPORT jbyteArray JNICALL
Java_com_ameva_amuda_diffusion_AmevaDiffusion_nativeTxt2Img(JNIEnv* env, jclass, jlong handle,
                                                              jstring prompt, jint width, jint height,
                                                              jint steps, jlong seed) {
    AmevaSd* sd = (AmevaSd*)(intptr_t)handle;
    std::string promptStr = jstringToUtf8(env, prompt);

    size_t rgbSize = (size_t)width * (size_t)height * 3;
    std::vector<uint8_t> rgb(rgbSize);
    char errBuf[256] = {0};
    AmevaSdStatus status = ameva_sd_txt2img(
        sd, promptStr.c_str(), (int)width, (int)height, (int)steps, (int64_t)seed,
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

} // extern "C"
