/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * ameva-android v0.1 JNI binding.
 *
 * Thin wrapper over ac_runtime.h only (ac_runtime_context_*, ac_runtime_buffer_*,
 * ac_runtime_kernel_*, ac_runtime_dispatch). Every AcResult != AC_SUCCESS is thrown to
 * Kotlin as a com.ameva.amuda.AmudaException carrying the raw numeric code and its
 * symbolic name -- no silent fallback, no swallowed error.
 *
 * Dispatch is always blocking (out_event == NULL): v0.1's purpose is proving a real
 * kernel ran and produced correct output from an app process, not async scheduling.
 */

#include <jni.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

#include "ac_runtime.h"
#include "ac_base.h"

namespace {

const char* ac_error_name(AcResult r) {
    switch (r) {
        case AC_SUCCESS: return "AC_SUCCESS";
        case AC_ERR_INVALID_CONTEXT: return "AC_ERR_INVALID_CONTEXT";
        case AC_ERR_DRIVER_NOT_FOUND: return "AC_ERR_DRIVER_NOT_FOUND";
        case AC_ERR_DRIVER_SYMBOL_MISSING: return "AC_ERR_DRIVER_SYMBOL_MISSING";
        case AC_ERR_DEVICE_NOT_SUPPORTED: return "AC_ERR_DEVICE_NOT_SUPPORTED";
        case AC_ERR_DRIVER_INIT_FAILED: return "AC_ERR_DRIVER_INIT_FAILED";
        case AC_ERR_INVALID_ARGUMENT: return "AC_ERR_INVALID_ARGUMENT";
        case AC_ERR_CONTEXT_MISMATCH: return "AC_ERR_CONTEXT_MISMATCH";
        case AC_ERR_FEATURE_NOT_SUPPORTED: return "AC_ERR_FEATURE_NOT_SUPPORTED";
        case AC_ERR_INVALID_SHAPE: return "AC_ERR_INVALID_SHAPE";
        case AC_ERR_ABI_MISMATCH: return "AC_ERR_ABI_MISMATCH";
        case AC_ERR_BACKEND_UNAVAILABLE: return "AC_ERR_BACKEND_UNAVAILABLE";
        case AC_ERR_OUT_OF_HOST_MEMORY: return "AC_ERR_OUT_OF_HOST_MEMORY";
        case AC_ERR_OUT_OF_DEVICE_MEMORY: return "AC_ERR_OUT_OF_DEVICE_MEMORY";
        case AC_ERR_BUFFER_ALIGNMENT_INVALID: return "AC_ERR_BUFFER_ALIGNMENT_INVALID";
        case AC_ERR_BUFFER_NOT_MAPPED: return "AC_ERR_BUFFER_NOT_MAPPED";
        case AC_ERR_NULL_POINTER: return "AC_ERR_NULL_POINTER";
        case AC_ERR_INVALID_BUFFER_SIZE: return "AC_ERR_INVALID_BUFFER_SIZE";
        case AC_ERR_KERNEL_BUILD_FAILED: return "AC_ERR_KERNEL_BUILD_FAILED";
        case AC_ERR_KERNEL_NOT_FOUND: return "AC_ERR_KERNEL_NOT_FOUND";
        case AC_ERR_KERNEL_ARG_INVALID: return "AC_ERR_KERNEL_ARG_INVALID";
        case AC_ERR_BINARY_CORRUPTED: return "AC_ERR_BINARY_CORRUPTED";
        case AC_ERR_INVALID_WORK_GROUP_SIZE: return "AC_ERR_INVALID_WORK_GROUP_SIZE";
        case AC_ERR_INVALID_WORK_ITEM_DIMS: return "AC_ERR_INVALID_WORK_ITEM_DIMS";
        case AC_ERR_MALI_INFINITE_LOOP_DETECTED: return "AC_ERR_MALI_INFINITE_LOOP_DETECTED";
        case AC_ERR_ADRENO_UNSAFE_SUBGROUP_OP: return "AC_ERR_ADRENO_UNSAFE_SUBGROUP_OP";
        case AC_ERR_LAUNCH_TIMEOUT: return "AC_ERR_LAUNCH_TIMEOUT";
        case AC_ERR_DEVICE_BUSY: return "AC_ERR_DEVICE_BUSY";
        case AC_ERR_RESOURCE_IN_USE: return "AC_ERR_RESOURCE_IN_USE";
        case AC_ERR_NAN_OR_INF_DETECTED: return "AC_ERR_NAN_OR_INF_DETECTED";
        case AC_ERR_NUMERICAL_OVERFLOW: return "AC_ERR_NUMERICAL_OVERFLOW";
        case AC_ERR_DRIVER_CRASH_OR_LOST: return "AC_ERR_DRIVER_CRASH_OR_LOST";
        case AC_ERR_OS_PROCESS_KILL_SIGNAL: return "AC_ERR_OS_PROCESS_KILL_SIGNAL";
        case AC_ERR_THERMAL_THROTTLING_TRIPPED: return "AC_ERR_THERMAL_THROTTLING_TRIPPED";
        default: return "AC_ERR_UNKNOWN";
    }
}

void throwAmudaException(JNIEnv* env, AcResult code, const char* where) {
    jclass cls = env->FindClass("com/ameva/amuda/AmudaException");
    if (cls == nullptr) {
        return; /* FindClass already threw NoClassDefFoundError */
    }
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
    char msg[256];
    snprintf(msg, sizeof(msg), "%s failed: %s (%d)", where, ac_error_name(code), (int)code);
    jstring jmsg = env->NewStringUTF(msg);
    jthrowable ex = (jthrowable)env->NewObject(cls, ctor, (jint)code, jmsg);
    env->Throw(ex);
}

#define CHECK_OR_THROW(call, where)              \
    do {                                          \
        AcResult _r = (call);                     \
        if (_r != AC_SUCCESS) {                    \
            throwAmudaException(env, _r, where);    \
            return 0;                                \
        }                                             \
    } while (0)

#define CHECK_OR_THROW_VOID(call, where)         \
    do {                                           \
        AcResult _r = (call);                       \
        if (_r != AC_SUCCESS) {                       \
            throwAmudaException(env, _r, where);        \
            return;                                       \
        }                                                  \
    } while (0)

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
Java_com_ameva_amuda_AmudaContext_nativeContextCreate(JNIEnv* env, jclass, jint backendKind) {
    AcContextDesc desc = {AC_ABI_VERSION, sizeof(AcContextDesc), (AcBackendKind)backendKind, 0};
    AcContext* ctx = nullptr;
    CHECK_OR_THROW(ac_runtime_context_create(&desc, &ctx), "ac_runtime_context_create");
    return (jlong)(intptr_t)ctx;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeContextDestroy(JNIEnv* env, jclass, jlong handle) {
    AcContext* ctx = (AcContext*)(intptr_t)handle;
    CHECK_OR_THROW_VOID(ac_runtime_context_destroy(ctx), "ac_runtime_context_destroy");
}

JNIEXPORT jstring JNICALL
Java_com_ameva_amuda_AmudaContext_nativeGetDeviceName(JNIEnv* env, jclass, jlong handle) {
    AcContext* ctx = (AcContext*)(intptr_t)handle;
    char name[256] = {0};
    AcResult r = ac_runtime_context_get_device_name(ctx, name, sizeof(name));
    if (r != AC_SUCCESS) {
        throwAmudaException(env, r, "ac_runtime_context_get_device_name");
        return nullptr;
    }
    return env->NewStringUTF(name);
}

JNIEXPORT jint JNICALL
Java_com_ameva_amuda_AmudaContext_nativeGetBackendKind(JNIEnv* env, jclass, jlong handle) {
    AcContext* ctx = (AcContext*)(intptr_t)handle;
    AcBackendKind kind = AC_BACKEND_UNKNOWN;
    AcResult r = ac_runtime_context_get_backend_kind(ctx, &kind);
    if (r != AC_SUCCESS) {
        throwAmudaException(env, r, "ac_runtime_context_get_backend_kind");
        return 0;
    }
    return (jint)kind;
}

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_AmudaContext_nativeBufferCreate(JNIEnv* env, jclass, jlong ctxHandle,
                                                       jlong sizeBytes, jint usage) {
    AcContext* ctx = (AcContext*)(intptr_t)ctxHandle;
    AcBufferDesc desc = {AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)sizeBytes,
                          (AcBufferUsageFlags)usage, 0};
    AcBuffer* buf = nullptr;
    CHECK_OR_THROW(ac_runtime_buffer_create(ctx, &desc, &buf), "ac_runtime_buffer_create");
    return (jlong)(intptr_t)buf;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeBufferDestroy(JNIEnv* env, jclass, jlong handle) {
    AcBuffer* buf = (AcBuffer*)(intptr_t)handle;
    CHECK_OR_THROW_VOID(ac_runtime_buffer_destroy(buf), "ac_runtime_buffer_destroy");
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeBufferWrite(JNIEnv* env, jclass, jlong handle,
                                                      jlong offset, jbyteArray data) {
    AcBuffer* buf = (AcBuffer*)(intptr_t)handle;
    jsize len = env->GetArrayLength(data);
    std::vector<uint8_t> tmp(len);
    env->GetByteArrayRegion(data, 0, len, (jbyte*)tmp.data());
    CHECK_OR_THROW_VOID(ac_runtime_buffer_write(buf, (size_t)offset, tmp.data(), (size_t)len),
                         "ac_runtime_buffer_write");
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeBufferRead(JNIEnv* env, jclass, jlong handle,
                                                     jlong offset, jbyteArray outData) {
    AcBuffer* buf = (AcBuffer*)(intptr_t)handle;
    jsize len = env->GetArrayLength(outData);
    std::vector<uint8_t> tmp(len);
    AcResult r = ac_runtime_buffer_read(buf, (size_t)offset, tmp.data(), (size_t)len);
    if (r != AC_SUCCESS) {
        throwAmudaException(env, r, "ac_runtime_buffer_read");
        return;
    }
    env->SetByteArrayRegion(outData, 0, len, (jbyte*)tmp.data());
}

/** bindingTypes[i]: AC_RUNTIME_BINDING_STORAGE_BUFFER (0) or AC_RUNTIME_BINDING_UNIFORM_BUFFER (1). */
static std::vector<AcRuntimeBindingDesc> buildBindings(JNIEnv* env, jintArray bindingTypes) {
    jsize n = env->GetArrayLength(bindingTypes);
    std::vector<jint> types(n);
    env->GetIntArrayRegion(bindingTypes, 0, n, types.data());
    std::vector<AcRuntimeBindingDesc> bindings(n);
    for (jsize i = 0; i < n; ++i) {
        bindings[i] = {(uint32_t)i, (AcRuntimeBindingType)types[i], AC_STAGE_COMPUTE};
    }
    return bindings;
}

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_AmudaContext_nativeKernelCreateSpirv(
        JNIEnv* env, jclass, jlong ctxHandle, jbyteArray spirv, jstring kernelName,
        jstring entryPoint, jintArray bindingTypes, jlong pushConstantSize,
        jint localX, jint localY, jint localZ, jstring expectedSha256Hex) {
    AcContext* ctx = (AcContext*)(intptr_t)ctxHandle;

    jsize spirvLen = env->GetArrayLength(spirv);
    std::vector<uint8_t> spirvBytes(spirvLen);
    env->GetByteArrayRegion(spirv, 0, spirvLen, (jbyte*)spirvBytes.data());

    std::string name = jstringToUtf8(env, kernelName);
    std::string entry = jstringToUtf8(env, entryPoint);
    std::string sha = jstringToUtf8(env, expectedSha256Hex);
    std::vector<AcRuntimeBindingDesc> bindings = buildBindings(env, bindingTypes);

    AcKernelRuntimeMetadata meta = {};
    meta.kernel_name = name.c_str();
    meta.entry_point = entry.c_str();
    meta.binding_count = (uint32_t)bindings.size();
    meta.bindings = bindings.data();
    meta.push_constant_size = (size_t)pushConstantSize;
    meta.local_size = {(uint32_t)localX, (uint32_t)localY, (uint32_t)localZ};
    meta.required_features = 0;
    meta.expected_spirv_sha256 = sha.empty() ? nullptr : sha.c_str();
    meta.specialization_schema_count = 0;
    meta.specialization_schemas = nullptr;

    AcKernelDesc desc = {};
    desc.abi_version = AC_ABI_VERSION;
    desc.struct_size = sizeof(AcKernelDesc);
    desc.spirv_code = reinterpret_cast<const uint32_t*>(spirvBytes.data());
    desc.code_size_bytes = (size_t)spirvLen;
    desc.metadata = &meta;
    desc.specialization_count = 0;
    desc.specializations = nullptr;

    AcKernel* kernel = nullptr;
    CHECK_OR_THROW(ac_runtime_kernel_create(ctx, &desc, &kernel), "ac_runtime_kernel_create");
    return (jlong)(intptr_t)kernel;
}

JNIEXPORT jlong JNICALL
Java_com_ameva_amuda_AmudaContext_nativeKernelCreateSource(
        JNIEnv* env, jclass, jlong ctxHandle, jstring source, jstring kernelName,
        jstring entryPoint, jintArray bindingTypes, jlong pushConstantSize,
        jint localX, jint localY, jint localZ) {
    AcContext* ctx = (AcContext*)(intptr_t)ctxHandle;

    std::string src = jstringToUtf8(env, source);
    std::string name = jstringToUtf8(env, kernelName);
    std::string entry = jstringToUtf8(env, entryPoint);
    std::vector<AcRuntimeBindingDesc> bindings = buildBindings(env, bindingTypes);

    AcKernelRuntimeMetadata meta = {};
    meta.kernel_name = name.c_str();
    meta.entry_point = entry.c_str();
    meta.binding_count = (uint32_t)bindings.size();
    meta.bindings = bindings.data();
    meta.push_constant_size = (size_t)pushConstantSize;
    meta.local_size = {(uint32_t)localX, (uint32_t)localY, (uint32_t)localZ};
    meta.required_features = 0;
    meta.expected_spirv_sha256 = nullptr;
    meta.specialization_schema_count = 0;
    meta.specialization_schemas = nullptr;

    AcKernelSourceDesc desc = {};
    desc.abi_version = AC_ABI_VERSION;
    desc.struct_size = sizeof(AcKernelSourceDesc);
    desc.source = src.c_str();
    desc.source_size = 0; /* NUL-terminated */
    desc.build_options = nullptr;
    desc.metadata = &meta;
    desc.flags = 0;

    AcKernel* kernel = nullptr;
    CHECK_OR_THROW(ac_runtime_kernel_create_from_source(ctx, &desc, &kernel),
                   "ac_runtime_kernel_create_from_source");
    return (jlong)(intptr_t)kernel;
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeKernelDestroy(JNIEnv* env, jclass, jlong handle) {
    AcKernel* kernel = (AcKernel*)(intptr_t)handle;
    CHECK_OR_THROW_VOID(ac_runtime_kernel_destroy(kernel), "ac_runtime_kernel_destroy");
}

JNIEXPORT void JNICALL
Java_com_ameva_amuda_AmudaContext_nativeDispatchBlocking(
        JNIEnv* env, jclass, jlong ctxHandle, jlong kernelHandle, jlongArray bufferHandles,
        jbyteArray pushConstants, jint gridX, jint gridY, jint gridZ) {
    AcContext* ctx = (AcContext*)(intptr_t)ctxHandle;
    AcKernel* kernel = (AcKernel*)(intptr_t)kernelHandle;

    jsize n = env->GetArrayLength(bufferHandles);
    std::vector<jlong> handles(n);
    env->GetLongArrayRegion(bufferHandles, 0, n, handles.data());
    std::vector<AcBuffer*> buffers(n);
    for (jsize i = 0; i < n; ++i) {
        buffers[i] = (AcBuffer*)(intptr_t)handles[i];
    }

    std::vector<uint8_t> pc;
    const void* pcPtr = nullptr;
    size_t pcSize = 0;
    if (pushConstants != nullptr) {
        jsize len = env->GetArrayLength(pushConstants);
        pc.resize(len);
        env->GetByteArrayRegion(pushConstants, 0, len, (jbyte*)pc.data());
        pcPtr = pc.data();
        pcSize = (size_t)len;
    }

    AcLaunchDesc desc = {};
    desc.abi_version = AC_ABI_VERSION;
    desc.struct_size = sizeof(AcLaunchDesc);
    desc.kernel = kernel;
    desc.buffers = buffers.data();
    desc.buffer_count = (uint32_t)buffers.size();
    desc.push_constants = pcPtr;
    desc.push_constants_size = pcSize;
    desc.grid_dim = {(uint32_t)gridX, (uint32_t)gridY, (uint32_t)gridZ};

    CHECK_OR_THROW_VOID(ac_runtime_dispatch(ctx, &desc, nullptr), "ac_runtime_dispatch");
}

} // extern "C"
