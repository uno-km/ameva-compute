/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file amuda.h
 * @brief AMUDA (Ameva Mobile Unified Device Architecture) Primary Header
 *
 * Provides CUDA-compatible kernel macros, host dispatch APIs,
 * UMA Zero-Copy memory abstractions, and strict fail-fast error handling.
 */

#ifndef AMUDA_H
#define AMUDA_H

#include "ac_types.h"

/* ========================================================================= */
/* 1. Device Kernel CUDA-Compatible Macros (OpenCL C / SPIR-V Interop)       */
/* ========================================================================= */
#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)

#define __global__      __kernel
#define __device__      static inline
#define __host__        
#define __constant__    __constant
#define __shared__      __local

/* 1D / Flat Index Accessors */
#define threadIdx_x     get_local_id(0)
#define threadIdx_y     get_local_id(1)
#define threadIdx_z     get_local_id(2)

#define blockIdx_x      get_group_id(0)
#define blockIdx_y      get_group_id(1)
#define blockIdx_z      get_group_id(2)

#define blockDim_x      get_local_size(0)
#define blockDim_y      get_local_size(1)
#define blockDim_z      get_local_size(2)

#define gridDim_x       get_num_groups(0)
#define gridDim_y       get_num_groups(1)
#define gridDim_z       get_num_groups(2)

/* 3D Struct Syntax Emulation */
typedef struct { uint x; uint y; uint z; } _AmudaDim3Accessor;
#define threadIdx       ((_AmudaDim3Accessor){ (uint)get_local_id(0), (uint)get_local_id(1), (uint)get_local_id(2) })
#define blockIdx        ((_AmudaDim3Accessor){ (uint)get_group_id(0), (uint)get_group_id(1), (uint)get_group_id(2) })
#define blockDim        ((_AmudaDim3Accessor){ (uint)get_local_size(0), (uint)get_local_size(1), (uint)get_local_size(2) })
#define gridDim         ((_AmudaDim3Accessor){ (uint)get_num_groups(0), (uint)get_num_groups(1), (uint)get_num_groups(2) })

/* Synchronization & Atomics */
#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

#define atomicAdd(p, v) atomic_add((volatile __global int*)(p), (int)(v))
#define atomicSub(p, v) atomic_sub((volatile __global int*)(p), (int)(v))
#define atomicMin(p, v) atomic_min((volatile __global int*)(p), (int)(v))
#define atomicMax(p, v) atomic_max((volatile __global int*)(p), (int)(v))

#endif /* __OPENCL_VERSION__ */

/* ========================================================================= */
/* 2. Host-Side Native C API (ac_ Prefix)                                    */
/* ========================================================================= */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the AMUDA GPGPU compute engine.
 *
 * Dynamically binds the underlying vendor driver (/vendor/lib64/libOpenCL.so or Vulkan ICD),
 * evaluates hardware quirks (Adreno/Mali/Xclipse), and creates the device context.
 */
AcResult ac_init(AcContext** out_ctx);

/**
 * @brief Release the context and all associated hardware queues.
 */
AcResult ac_destroy(AcContext* ctx);

/**
 * @brief Allocate a 64-byte aligned UMA Zero-Copy buffer.
 */
AcResult ac_buffer_alloc(AcContext* ctx, size_t size_bytes, AcBuffer** out_buf);

/**
 * @brief Wrap an existing aligned host memory pointer into a Zero-Copy buffer.
 */
AcResult ac_buffer_from_host(AcContext* ctx, void* aligned_ptr, size_t size_bytes, AcBuffer** out_buf);

/**
 * @brief Synchronize host cache line to device (Clean/Flush).
 */
AcResult ac_buffer_sync_to_device(AcContext* ctx, AcBuffer* buf);

/**
 * @brief Synchronize device cache line to host (Invalidate).
 */
AcResult ac_buffer_sync_to_host(AcContext* ctx, AcBuffer* buf);

/**
 * @brief Release memory buffer.
 */
AcResult ac_buffer_free(AcContext* ctx, AcBuffer* buf);

/**
 * @brief Load and compile OpenCL C / AMUDA kernel source code with SHA-256 caching.
 */
AcResult ac_kernel_load_source(AcContext* ctx, const char* source_code, const char* func_name, AcKernel** out_kernel);

/**
 * @brief Load precompiled SPIR-V binary kernel.
 */
AcResult ac_kernel_load_binary(AcContext* ctx, const void* binary_data, size_t binary_size, const char* func_name, AcKernel** out_kernel);

/**
 * @brief Release kernel resources.
 */
AcResult ac_kernel_free(AcContext* ctx, AcKernel* kernel);

/**
 * @brief Asynchronously dispatch kernel across 3D Grid and Block.
 */
AcResult ac_launch(
    AcContext* ctx,
    AcKernel*  kernel,
    AcDim3     grid,
    AcDim3     block,
    uint32_t   argc,
    void**     argv,
    size_t*    arg_sizes,
    AcEvent**  out_event
);

/**
 * @brief Wait for event completion (Blocking).
 */
AcResult ac_event_wait(AcContext* ctx, AcEvent* event);

/**
 * @brief Get kernel execution time in microseconds.
 */
uint64_t ac_event_get_duration_us(AcEvent* event);

/**
 * @brief Release event resources.
 */
AcResult ac_event_free(AcContext* ctx, AcEvent* event);

/**
 * @brief Dimension helper constructors
 */
static inline AcDim3 ac_dim3(uint32_t x, uint32_t y, uint32_t z) {
    AcDim3 d; d.x = x; d.y = y; d.z = z; return d;
}

static inline AcDim3 ac_dim1(uint32_t x) {
    return ac_dim3(x, 1, 1);
}

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_H */
