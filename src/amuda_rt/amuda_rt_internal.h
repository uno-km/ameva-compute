/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef AMUDA_RT_INTERNAL_H
#define AMUDA_RT_INTERNAL_H

#include "amuda_rt.h"

#define AMU_MAX_BUFFERS 16u
#define AMU_MAX_PUSH_BYTES 128u
#define AMU_MAX_BLOCK_DIM 1024u

typedef struct amuBackendOps amuBackendOps;

/* internal symbols are not exported from libamuda_rt.so */
#if defined(__GNUC__) || defined(__clang__)
#define AMU_INTERNAL __attribute__((visibility("hidden")))
#else
#define AMU_INTERNAL
#endif

struct amuContext {
    amuBackend backend;
    const amuBackendOps* ops;
    void* impl;
    char device_name[256];
    amuStatus sticky;            /* non-zero after timeout / device loss */
    int inflight;                /* one launch in flight at most (in-order stream) */
    amuBuffer* inflight_bufs[AMU_MAX_BUFFERS];
    uint32_t inflight_nbufs;
    double last_gpu_ms;
    /* AMUDA_TRACE=1: per-launch phase timing printed to stderr */
    int trace;
    const char* trace_kernel;
    double trace_flush_ms, trace_dispatch_ms, trace_prev_drain_ms;
    double trace_sub_ms[3]; /* backend wait split: fence / timing query / release */
};

struct amuBuffer {
    amuContext* ctx;
    void* impl;
    size_t size;
};

struct amuFunction {
    amuContext* ctx;
    const amuKernelImage* image;
    void* impl;
};

struct amuBackendOps {
    const char* name;
    amuStatus (*init)(amuContext* ctx);
    amuStatus (*destroy)(amuContext* ctx); /* always releases ctx->impl; non-success = something was leaked */
    amuStatus (*buf_alloc)(amuContext* ctx, amuBuffer* buf, size_t bytes);
    amuStatus (*buf_free)(amuBuffer* buf);
    void* (*buf_host)(amuBuffer* buf);
    amuStatus (*buf_to_device)(amuBuffer* buf); /* make host writes visible to the device */
    amuStatus (*buf_to_host)(amuBuffer* buf);   /* make device writes visible to the host */
    amuStatus (*fn_load)(amuFunction* fn);
    amuStatus (*fn_unload)(amuFunction* fn);
    /* args validated by the common layer; asynchronous */
    amuStatus (*launch)(amuFunction* fn, amuDim3 grid, amuDim3 block, void** args);
    /* wait for the in-flight launch; gpu_ms < 0 when timing is unavailable */
    amuStatus (*wait)(amuContext* ctx, double* gpu_ms);
};

AMU_INTERNAL amuStatus amu_fail(amuStatus status, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#if defined(AMU_WITH_VULKAN)
extern AMU_INTERNAL const amuBackendOps amu_vulkan_ops;
#endif
#if defined(AMU_WITH_OPENCL)
extern AMU_INTERNAL const amuBackendOps amu_opencl_ops;
#endif

#endif /* AMUDA_RT_INTERNAL_H */
