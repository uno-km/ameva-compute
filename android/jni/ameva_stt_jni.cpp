/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * JNI binding for ameva_stt_api.h (v0.3). Throws com.ameva.amuda.AmudaException on every
 * AmevaSttStatus != AMEVA_STT_OK -- same convention as ameva_llm_jni.cpp.
 */

#include <jni.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ameva_stt_api.h"

namespace {

void throwAmudaException(JNIEnv* env, AmevaSttStatus code, const char* detail) {
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
Java_com_ameva_amuda_stt_AmevaSpeech_nativeLoad(JNIEnv* env, jclass, jstring modelPath,
                                                  jstring spirvF16Path, jstring spirvQ50Path,
                                                  jstring language) {
    std::string model = jstringToUtf8(env, modelPath);
    std::string f16 = jstringToUtf8(env, spirvF16Path);
    std::string q50 = jstringToUtf8(env, spirvQ50Path);
    std::string lang = jstringToUtf8(env, language);

    AmevaStt* stt = nullptr;
    char errBuf[256] = {0};
    AmevaSttStatus status = ameva_stt_load(
        model.c_str(), f16.empty() ? nullptr : f16.c_str(),
        q50.empty() ? nullptr : q50.c_str(), lang.empty() ? nullptr : lang.c_str(),
        &stt, errBuf, sizeof(errBuf));
    if (status != AMEVA_STT_OK) {
        throwAmudaException(env, status, errBuf);
        return 0;
    }
    return (jlong)(intptr_t)stt;
}

JNIEXPORT jstring JNICALL
Java_com_ameva_amuda_stt_AmevaSpeech_nativeTranscribe(JNIEnv* env, jclass, jlong handle,
                                                        jfloatArray pcmf32, jboolean translate,
                                                        jint outBufSize) {
    AmevaStt* stt = (AmevaStt*)(intptr_t)handle;
    jsize n = env->GetArrayLength(pcmf32);
    std::vector<float> samples((size_t)n);
    env->GetFloatArrayRegion(pcmf32, 0, n, samples.data());

    std::vector<char> outBuf((size_t)outBufSize);
    char errBuf[256] = {0};
    AmevaSttStatus status = ameva_stt_transcribe(
        stt, samples.data(), (int32_t)n, translate ? 1 : 0,
        outBuf.data(), outBuf.size(), errBuf, sizeof(errBuf));
    if (status != AMEVA_STT_OK) {
        throwAmudaException(env, status, errBuf);
        return nullptr;
    }
    return env->NewStringUTF(outBuf.data());
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_stt_AmevaSpeech_nativeFree(JNIEnv*, jclass, jlong handle) {
    ameva_stt_free((AmevaStt*)(intptr_t)handle);
}

} // extern "C"
