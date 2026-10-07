/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
/* AMUDA CUDA-like runtime: backend-independent layer (validation, ordering, memory visibility). */
#include "amuda_rt_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static _Thread_local char g_detail[2048];

amuStatus amu_fail(amuStatus status, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_detail, sizeof(g_detail), fmt, ap);
    va_end(ap);
    return status;
}

const char* amuGetLastErrorDetail(void) { return g_detail; }

const char* amuGetErrorName(amuStatus s) {
    switch (s) {
    case AMU_SUCCESS: return "AMU_SUCCESS";
    case AMU_ERR_INVALID_ARGUMENT: return "AMU_ERR_INVALID_ARGUMENT";
    case AMU_ERR_BACKEND_UNAVAILABLE: return "AMU_ERR_BACKEND_UNAVAILABLE";
    case AMU_ERR_OUT_OF_MEMORY: return "AMU_ERR_OUT_OF_MEMORY";
    case AMU_ERR_KERNEL_IMAGE: return "AMU_ERR_KERNEL_IMAGE";
    case AMU_ERR_KERNEL_BUILD: return "AMU_ERR_KERNEL_BUILD";
    case AMU_ERR_LAUNCH: return "AMU_ERR_LAUNCH";
    case AMU_ERR_TIMEOUT: return "AMU_ERR_TIMEOUT";
    case AMU_ERR_DEVICE_LOST: return "AMU_ERR_DEVICE_LOST";
    case AMU_ERR_NOT_SUPPORTED: return "AMU_ERR_NOT_SUPPORTED";
    case AMU_ERR_INTERNAL: return "AMU_ERR_INTERNAL";
    }
    return "AMU_ERR_UNKNOWN";
}

const char* amuBackendName(amuBackend b) {
    switch (b) {
    case AMU_BACKEND_VULKAN: return "vulkan";
    case AMU_BACKEND_OPENCL: return "opencl";
    }
    return "unknown";
}

static const amuBackendOps* ops_for(amuBackend b) {
    switch (b) {
#if defined(AMU_WITH_VULKAN)
    case AMU_BACKEND_VULKAN: return &amu_vulkan_ops;
#endif
#if defined(AMU_WITH_OPENCL)
    case AMU_BACKEND_OPENCL: return &amu_opencl_ops;
#endif
    default: return NULL;
    }
}

amuStatus amuInit(amuBackend backend, amuContext** out_ctx) {
    if (!out_ctx) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuInit: out_ctx is NULL");
    *out_ctx = NULL;
    const amuBackendOps* ops = ops_for(backend);
    if (!ops) {
        return amu_fail(AMU_ERR_BACKEND_UNAVAILABLE,
                        "backend '%s' was not compiled into this amuda_rt build", amuBackendName(backend));
    }
    amuContext* ctx = (amuContext*)calloc(1, sizeof(*ctx));
    if (!ctx) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "amuInit: host allocation failed");
    ctx->backend = backend;
    ctx->ops = ops;
    ctx->last_gpu_ms = -1.0;
    {
        const char* tr = getenv("AMUDA_TRACE");
        ctx->trace = (tr && tr[0] == '1') ? 1 : 0;
    }
    amuStatus s = ops->init(ctx);
    if (s != AMU_SUCCESS) {
        free(ctx);
        return s;
    }
    *out_ctx = ctx;
    return AMU_SUCCESS;
}

amuStatus amuInitFromEnv(amuContext** out_ctx) {
    const char* v = getenv("AMUDA_BACKEND");
    if (!v || !*v || strcmp(v, "vulkan") == 0) return amuInit(AMU_BACKEND_VULKAN, out_ctx);
    if (strcmp(v, "opencl") == 0) return amuInit(AMU_BACKEND_OPENCL, out_ctx);
    if (out_ctx) *out_ctx = NULL;
    return amu_fail(AMU_ERR_INVALID_ARGUMENT, "AMUDA_BACKEND='%s' is not one of: vulkan, opencl", v);
}

static amuStatus drain(amuContext* ctx) {
    if (ctx->sticky) return ctx->sticky;
    if (!ctx->inflight) return AMU_SUCCESS;
    double ms = -1.0;
    double t_w0 = ctx->trace ? now_ms() : 0.0;
    ctx->trace_sub_ms[0] = ctx->trace_sub_ms[1] = ctx->trace_sub_ms[2] = -1.0;
    amuStatus s = ctx->ops->wait(ctx, &ms);
    double t_w1 = ctx->trace ? now_ms() : 0.0;
    if (s != AMU_SUCCESS) {
        if (s == AMU_ERR_TIMEOUT || s == AMU_ERR_DEVICE_LOST) ctx->sticky = s;
        ctx->inflight = 0;
        ctx->inflight_nbufs = 0;
        return s;
    }
    ctx->last_gpu_ms = ms;
    for (uint32_t i = 0; i < ctx->inflight_nbufs; ++i) {
        amuStatus hs = ctx->ops->buf_to_host(ctx->inflight_bufs[i]);
        if (hs != AMU_SUCCESS) {
            ctx->inflight = 0;
            ctx->inflight_nbufs = 0;
            return hs;
        }
    }
    if (ctx->trace) {
        fprintf(stderr,
                "AMU_TRACE kernel=%s prev_drain_ms=%.3f flush_ms=%.3f dispatch_ms=%.3f wait_ms=%.3f "
                "(fence=%.3f timing=%.3f release=%.3f) to_host_ms=%.3f gpu_ms=%.3f\n",
                ctx->trace_kernel ? ctx->trace_kernel : "?", ctx->trace_prev_drain_ms, ctx->trace_flush_ms,
                ctx->trace_dispatch_ms, t_w1 - t_w0, ctx->trace_sub_ms[0], ctx->trace_sub_ms[1],
                ctx->trace_sub_ms[2], now_ms() - t_w1, ms);
    }
    ctx->inflight = 0;
    ctx->inflight_nbufs = 0;
    return AMU_SUCCESS;
}

amuStatus amuDestroy(amuContext* ctx) {
    if (!ctx) return AMU_SUCCESS;
    amuStatus ds = AMU_SUCCESS;
    if (ctx->inflight && !ctx->sticky) ds = drain(ctx);
    amuStatus s = ctx->ops->destroy(ctx);
    free(ctx);
    return ds != AMU_SUCCESS ? ds : s;
}

amuBackend amuGetBackend(const amuContext* ctx) { return ctx ? ctx->backend : (amuBackend)0; }
const char* amuGetDeviceName(const amuContext* ctx) { return ctx ? ctx->device_name : ""; }
double amuLastGpuTimeMs(const amuContext* ctx) { return ctx ? ctx->last_gpu_ms : -1.0; }

amuStatus amuDeviceSynchronize(amuContext* ctx) {
    if (!ctx) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuDeviceSynchronize: ctx is NULL");
    return drain(ctx);
}

static int buffer_inflight(const amuBuffer* b) {
    const amuContext* ctx = b->ctx;
    if (!ctx->inflight) return 0;
    for (uint32_t i = 0; i < ctx->inflight_nbufs; ++i)
        if (ctx->inflight_bufs[i] == b) return 1;
    return 0;
}

amuStatus amuMalloc(amuContext* ctx, size_t bytes, amuBuffer** out_buf) {
    if (!ctx || !out_buf) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMalloc: NULL argument");
    *out_buf = NULL;
    if (ctx->sticky) return ctx->sticky;
    if (bytes == 0) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMalloc: size 0");
    amuBuffer* b = (amuBuffer*)calloc(1, sizeof(*b));
    if (!b) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "amuMalloc: host allocation failed");
    b->ctx = ctx;
    b->size = bytes;
    amuStatus s = ctx->ops->buf_alloc(ctx, b, bytes);
    if (s != AMU_SUCCESS) {
        free(b);
        return s;
    }
    *out_buf = b;
    return AMU_SUCCESS;
}

amuStatus amuFree(amuBuffer* b) {
    if (!b) return AMU_SUCCESS;
    if (buffer_inflight(b)) {
        amuStatus s = drain(b->ctx);
        if (s != AMU_SUCCESS) return s;
    }
    amuStatus s = b->ctx->ops->buf_free(b);
    free(b);
    return s;
}

void* amuHostPtr(amuBuffer* b) { return b ? b->ctx->ops->buf_host(b) : NULL; }
size_t amuBufferSize(const amuBuffer* b) { return b ? b->size : 0; }

amuStatus amuMemcpyHtoD(amuBuffer* dst, size_t off, const void* src, size_t bytes) {
    if (!dst || (!src && bytes)) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemcpyHtoD: NULL argument");
    if (off > dst->size || bytes > dst->size - off)
        return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemcpyHtoD: range [%zu,+%zu) exceeds buffer size %zu",
                        off, bytes, dst->size);
    if (buffer_inflight(dst)) {
        amuStatus s = drain(dst->ctx);
        if (s != AMU_SUCCESS) return s;
    }
    if (dst->ctx->sticky) return dst->ctx->sticky;
    memcpy((char*)dst->ctx->ops->buf_host(dst) + off, src, bytes);
    return AMU_SUCCESS;
}

amuStatus amuMemcpyDtoH(void* dst, amuBuffer* src, size_t off, size_t bytes) {
    if (!src || (!dst && bytes)) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemcpyDtoH: NULL argument");
    if (off > src->size || bytes > src->size - off)
        return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemcpyDtoH: range [%zu,+%zu) exceeds buffer size %zu",
                        off, bytes, src->size);
    if (buffer_inflight(src)) {
        amuStatus s = drain(src->ctx);
        if (s != AMU_SUCCESS) return s;
    }
    if (src->ctx->sticky) return src->ctx->sticky;
    memcpy(dst, (const char*)src->ctx->ops->buf_host(src) + off, bytes);
    return AMU_SUCCESS;
}

amuStatus amuMemset(amuBuffer* dst, int value, size_t bytes) {
    if (!dst) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemset: NULL buffer");
    if (bytes > dst->size) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuMemset: %zu > buffer size %zu", bytes, dst->size);
    if (buffer_inflight(dst)) {
        amuStatus s = drain(dst->ctx);
        if (s != AMU_SUCCESS) return s;
    }
    if (dst->ctx->sticky) return dst->ctx->sticky;
    memset(dst->ctx->ops->buf_host(dst), value, bytes);
    return AMU_SUCCESS;
}

amuStatus amuModuleLoad(amuContext* ctx, const amuKernelImage* img, amuFunction** out_fn) {
    if (!ctx || !img || !out_fn) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuModuleLoad: NULL argument");
    *out_fn = NULL;
    if (ctx->sticky) return ctx->sticky;
    if (img->abi != AMU_IMAGE_ABI)
        return amu_fail(AMU_ERR_KERNEL_IMAGE, "kernel image ABI %u, runtime expects %u (rebuild with matching amudacc)",
                        img->abi, AMU_IMAGE_ABI);
    if (img->buffer_count == 0 || img->buffer_count > AMU_MAX_BUFFERS || img->push_constant_size > AMU_MAX_PUSH_BYTES ||
        img->param_count != img->buffer_count + img->push_constant_size / 4u)
        return amu_fail(AMU_ERR_KERNEL_IMAGE, "kernel image '%s' has inconsistent parameter metadata", img->name);
    amuFunction* fn = (amuFunction*)calloc(1, sizeof(*fn));
    if (!fn) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "amuModuleLoad: host allocation failed");
    fn->ctx = ctx;
    fn->image = img;
    amuStatus s = ctx->ops->fn_load(fn);
    if (s != AMU_SUCCESS) {
        free(fn);
        return s;
    }
    *out_fn = fn;
    return AMU_SUCCESS;
}

amuStatus amuModuleUnload(amuFunction* fn) {
    if (!fn) return AMU_SUCCESS;
    if (fn->ctx->inflight && !fn->ctx->sticky) {
        amuStatus s = drain(fn->ctx);
        if (s != AMU_SUCCESS) return s; /* the launch may still use the pipeline: keep fn alive */
    }
    amuStatus s = fn->ctx->ops->fn_unload(fn);
    free(fn);
    return s;
}

amuStatus amuLaunchKernel(amuFunction* fn, amuDim3 grid, amuDim3 block, void** args, uint32_t argc) {
    if (!fn || (!args && argc)) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "amuLaunchKernel: NULL argument");
    amuContext* ctx = fn->ctx;
    const amuKernelImage* img = fn->image;
    if (ctx->sticky) return ctx->sticky;
    if (argc != img->param_count)
        return amu_fail(AMU_ERR_INVALID_ARGUMENT, "kernel '%s' takes %u arguments, got %u", img->name, img->param_count, argc);
    if (!grid.x || !grid.y || !grid.z)
        return amu_fail(AMU_ERR_LAUNCH, "kernel '%s': grid (%u,%u,%u) has a zero dimension", img->name, grid.x, grid.y, grid.z);
    if (!block.x || !block.y || !block.z || block.x > AMU_MAX_BLOCK_DIM || block.y > AMU_MAX_BLOCK_DIM ||
        block.z > AMU_MAX_BLOCK_DIM || (uint64_t)block.x * block.y * block.z > AMU_MAX_BLOCK_DIM)
        return amu_fail(AMU_ERR_LAUNCH, "kernel '%s': block (%u,%u,%u) invalid (each dim 1..%u, product <= %u)",
                        img->name, block.x, block.y, block.z, AMU_MAX_BLOCK_DIM, AMU_MAX_BLOCK_DIM);

    amuBuffer* bufs[AMU_MAX_BUFFERS];
    uint32_t nb = 0;
    for (uint32_t i = 0; i < argc; ++i) {
        if (!args[i]) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "kernel '%s': argument %u (%s) is NULL", img->name, i, img->params[i].name);
        if (img->params[i].kind == AMU_ARG_BUFFER) {
            amuBuffer* b = *(amuBuffer**)args[i];
            if (!b) return amu_fail(AMU_ERR_INVALID_ARGUMENT, "kernel '%s': buffer argument '%s' is NULL", img->name, img->params[i].name);
            if (b->ctx != ctx)
                return amu_fail(AMU_ERR_INVALID_ARGUMENT, "kernel '%s': buffer '%s' belongs to another context", img->name, img->params[i].name);
            if (nb >= AMU_MAX_BUFFERS) return amu_fail(AMU_ERR_KERNEL_IMAGE, "too many buffers");
            bufs[nb++] = b;
        }
    }
    if (nb != img->buffer_count) return amu_fail(AMU_ERR_KERNEL_IMAGE, "kernel '%s': buffer count mismatch", img->name);

    /* in-order semantics: the previous launch must finish before this one starts */
    double t0 = ctx->trace ? now_ms() : 0.0;
    amuStatus s = drain(ctx);
    if (s != AMU_SUCCESS) return s;
    double t1 = ctx->trace ? now_ms() : 0.0;

    for (uint32_t i = 0; i < nb; ++i) {
        s = ctx->ops->buf_to_device(bufs[i]);
        if (s != AMU_SUCCESS) return s;
    }
    double t2 = ctx->trace ? now_ms() : 0.0;
    s = ctx->ops->launch(fn, grid, block, args);
    if (ctx->trace) {
        ctx->trace_kernel = img->name;
        ctx->trace_prev_drain_ms = t1 - t0;
        ctx->trace_flush_ms = t2 - t1;
        ctx->trace_dispatch_ms = now_ms() - t2;
    }
    if (s != AMU_SUCCESS) {
        /* the dispatch did not happen; hand the buffers back to the host */
        for (uint32_t i = 0; i < nb; ++i) (void)ctx->ops->buf_to_host(bufs[i]);
        return s;
    }
    ctx->inflight = 1;
    ctx->inflight_nbufs = nb;
    memcpy(ctx->inflight_bufs, bufs, nb * sizeof(bufs[0]));
    return AMU_SUCCESS;
}
