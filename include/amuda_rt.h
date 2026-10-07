/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file amuda_rt.h
 * @brief AMUDA CUDA-like runtime (POC v0.1)
 *
 * Kernels are written in a CUDA C subset (.acu) and compiled by `amudacc` into a header that
 * embeds a Vulkan (SPIR-V) image and an OpenCL C image. This runtime loads those images and
 * launches them with CUDA-style grid/block semantics.
 *
 *   amuContext* ctx;  amuInitFromEnv(&ctx);               // AMUDA_BACKEND=vulkan|opencl
 *   amuBuffer* a;     amuMalloc(ctx, n * sizeof(float), &a);
 *   float* pa = (float*)amuHostPtr(a);                     // UMA: same physical memory
 *   amuFunction* fn;  amu_load_vector_add(ctx, &fn);       // from amudacc header
 *   amu_launch_vector_add(fn, amuDim(blocks,1,1), amuDim(256,1,1), a, b, c, n);
 *   amuDeviceSynchronize(ctx);
 *
 * Rules (Zero-Silent-Fallback):
 *  - The requested backend is used or the call fails. There is no CPU fallback anywhere.
 *  - Only GPU devices are accepted (software rasterizers such as llvmpipe are rejected).
 *  - Launches on one context execute in issue order (like the CUDA default stream).
 *  - CPU writes made through amuHostPtr() before amuLaunchKernel() are visible to the kernel.
 *    Kernel writes are visible through amuHostPtr() after amuDeviceSynchronize().
 *    Do not touch a buffer from the CPU while a launch that uses it is in flight.
 *  - After a timeout or device loss the context is poisoned: every later call returns that error.
 */
#ifndef AMUDA_RT_H
#define AMUDA_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AMU_RT_VERSION "0.1.0"
#define AMU_IMAGE_ABI 1u

typedef enum {
    AMU_SUCCESS = 0,
    AMU_ERR_INVALID_ARGUMENT = 1,
    AMU_ERR_BACKEND_UNAVAILABLE = 2, /* requested backend could not be initialised (no fallback) */
    AMU_ERR_OUT_OF_MEMORY = 3,
    AMU_ERR_KERNEL_IMAGE = 4,        /* image lacks code for this backend, ABI mismatch, integrity */
    AMU_ERR_KERNEL_BUILD = 5,        /* pipeline / program creation failed on the device */
    AMU_ERR_LAUNCH = 6,              /* invalid launch configuration or dispatch failure */
    AMU_ERR_TIMEOUT = 7,
    AMU_ERR_DEVICE_LOST = 8,
    AMU_ERR_NOT_SUPPORTED = 9,
    AMU_ERR_INTERNAL = 10
} amuStatus;

typedef enum {
    AMU_BACKEND_VULKAN = 1,
    AMU_BACKEND_OPENCL = 2
} amuBackend;

typedef struct { uint32_t x, y, z; } amuDim3;

static inline amuDim3 amuDim(uint32_t x, uint32_t y, uint32_t z) {
    amuDim3 d;
    d.x = x; d.y = y; d.z = z;
    return d;
}

typedef struct amuContext amuContext;
typedef struct amuBuffer amuBuffer;
typedef struct amuFunction amuFunction;

/* ---- kernel image produced by amudacc ---- */
typedef enum { AMU_ARG_BUFFER = 1, AMU_ARG_SCALAR = 2 } amuArgKind;
typedef enum { AMU_T_F32 = 1, AMU_T_I32 = 2, AMU_T_U32 = 3 } amuScalarType;

typedef struct {
    const char* name;
    uint32_t kind;      /* amuArgKind */
    uint32_t type;      /* amuScalarType (element type for buffers) */
    uint32_t readonly;  /* buffers declared const */
    uint32_t slot;      /* buffer: binding index, scalar: push-constant byte offset */
} amuParamDesc;

typedef struct {
    uint32_t abi;                  /* AMU_IMAGE_ABI */
    const char* name;
    const char* source_file;
    const uint32_t* spirv;         /* NULL if compiled without the vulkan target */
    size_t spirv_size;             /* bytes */
    const char* spirv_sha256;      /* 64 hex chars */
    const char* opencl_source;     /* NULL if compiled without the opencl target */
    uint32_t param_count;
    const amuParamDesc* params;
    uint32_t buffer_count;
    uint32_t push_constant_size;
    uint32_t shared_bytes;         /* static __shared__ bytes per block (amudacc caps at 16384) */
} amuKernelImage;

/* ---- context ---- */
amuStatus   amuInit(amuBackend backend, amuContext** out_ctx);
amuStatus   amuInitFromEnv(amuContext** out_ctx); /* AMUDA_BACKEND=vulkan|opencl, default vulkan */
amuStatus   amuDestroy(amuContext* ctx); /* ctx is released even on error; the error names what leaked */
amuBackend  amuGetBackend(const amuContext* ctx);
const char* amuBackendName(amuBackend backend);
const char* amuGetDeviceName(const amuContext* ctx);

/* ---- memory (UMA, host-visible) ---- */
amuStatus amuMalloc(amuContext* ctx, size_t bytes, amuBuffer** out_buf);
amuStatus amuFree(amuBuffer* buf);
void*     amuHostPtr(amuBuffer* buf);
size_t    amuBufferSize(const amuBuffer* buf);
amuStatus amuMemcpyHtoD(amuBuffer* dst, size_t dst_offset, const void* src, size_t bytes);
amuStatus amuMemcpyDtoH(void* dst, amuBuffer* src, size_t src_offset, size_t bytes);
amuStatus amuMemset(amuBuffer* dst, int value, size_t bytes);

/* ---- kernels ---- */
amuStatus amuModuleLoad(amuContext* ctx, const amuKernelImage* image, amuFunction** out_fn);
amuStatus amuModuleUnload(amuFunction* fn);
/* args[i] points at the value of parameter i (amuBuffer* for buffers), like cuLaunchKernel. */
amuStatus amuLaunchKernel(amuFunction* fn, amuDim3 grid, amuDim3 block, void** args, uint32_t arg_count);
amuStatus amuDeviceSynchronize(amuContext* ctx);
/* GPU execution time of the last completed launch in ms, or a negative value if the device
 * does not expose timestamps for this backend. */
double    amuLastGpuTimeMs(const amuContext* ctx);

/* ---- errors ---- */
const char* amuGetErrorName(amuStatus status);
const char* amuGetLastErrorDetail(void); /* thread-local, describes the most recent failure */

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_RT_H */
