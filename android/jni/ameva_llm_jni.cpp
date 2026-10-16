/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * JNI binding for ameva_llm_api.h (v0.2). Throws com.ameva.amuda.AmudaException on every
 * AmevaLlmStatus != AMEVA_LLM_OK -- same error-propagation convention as v0.1's ameva_jni.cpp.
 */

#include <jni.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ameva_llm_api.h"

namespace {

void throwAmudaException(JNIEnv* env, AmevaLlmStatus code, const char* detail) {
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
Java_com_ameva_amuda_llm_AmevaLlm_nativeLoad(JNIEnv* env, jclass, jstring modelPath,
                                               jstring backendPluginPath, jstring cacheDir,
                                               jstring backendName, jint nCtx) {
    std::string model = jstringToUtf8(env, modelPath);
    std::string backendPlugin = jstringToUtf8(env, backendPluginPath);
    std::string cache = jstringToUtf8(env, cacheDir);
    std::string backend = jstringToUtf8(env, backendName);

    AmevaLlm* llm = nullptr;
    char errBuf[256] = {0};
    AmevaLlmStatus status = ameva_llm_load(model.c_str(), backendPlugin.c_str(), cache.c_str(),
                                            backend.empty() ? nullptr : backend.c_str(),
                                            (uint32_t)nCtx, &llm, errBuf, sizeof(errBuf));
    if (status != AMEVA_LLM_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)llm;
}

JNIEXPORT jstring JNICALL
Java_com_ameva_amuda_llm_AmevaLlm_nativeGenerate(JNIEnv* env, jclass, jlong handle, jstring prompt,
                                                   jint maxTokens, jfloat temperature, jlong seed,
                                                   jint outBufSize) {
    AmevaLlm* llm = (AmevaLlm*)(intptr_t)handle;
    std::string promptStr = jstringToUtf8(env, prompt);

    std::vector<char> outBuf((size_t)outBufSize);
    char errBuf[256] = {0};
    AmevaLlmStatus status = ameva_llm_generate(
        llm, promptStr.c_str(), (int32_t)maxTokens, (float)temperature, (uint64_t)seed,
        outBuf.data(), outBuf.size(), errBuf, sizeof(errBuf));
    if (status != AMEVA_LLM_OK) {
        throwAmudaException(env, status, errBuf);
        return nullptr;
    }
    return env->NewStringUTF(outBuf.data());
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_llm_AmevaLlm_nativeReset(JNIEnv* env, jclass, jlong handle) {
    AmevaLlm* llm = (AmevaLlm*)(intptr_t)handle;
    char errBuf[256] = {0};
    AmevaLlmStatus status = ameva_llm_reset(llm, errBuf, sizeof(errBuf));
    if (status != AMEVA_LLM_OK) {
        throwAmudaException(env, status, errBuf);
    }
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_llm_AmevaLlm_nativeFree(JNIEnv*, jclass, jlong handle) {
    ameva_llm_free((AmevaLlm*)(intptr_t)handle);
}

} // extern "C"
