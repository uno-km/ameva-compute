/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
/* AMUDA CUDA-like runtime: Vulkan backend on top of the AMUDA C ABI (ac_runtime). */
#define AMU_WITH_VULKAN 1
#include "amuda_rt_internal.h"
#include "ac_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double vk_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#define AMU_VK_WAIT_TIMEOUT_NS (60ull * 1000ull * 1000ull * 1000ull)
#define AMU_VK_MAX_PIPES 8u

typedef struct {
    AcContext* ac;
    AcEvent* event;
} VkCtx;

typedef struct {
    AcBuffer* ac;
    void* mapped;
    int coherent;
} VkBuf;

typedef struct {
    struct { uint32_t x, y, z; AcKernel* k; } pipes[AMU_VK_MAX_PIPES];
    uint32_t n;
} VkFn;

static const char* ac_name(AcResult r) {
    switch ((int)r) {
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
    case AC_ERR_LAUNCH_TIMEOUT: return "AC_ERR_LAUNCH_TIMEOUT";
    case AC_ERR_DEVICE_BUSY: return "AC_ERR_DEVICE_BUSY";
    case AC_ERR_RESOURCE_IN_USE: return "AC_ERR_RESOURCE_IN_USE";
    case AC_ERR_DRIVER_CRASH_OR_LOST: return "AC_ERR_DRIVER_CRASH_OR_LOST";
    default: return "AC_ERR_?";
    }
}

static amuStatus vk_init(amuContext* ctx) {
    VkCtx* v = (VkCtx*)calloc(1, sizeof(*v));
    if (!v) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "vulkan: host allocation failed");
    AcContextDesc d;
    memset(&d, 0, sizeof(d));
    d.abi_version = AC_ABI_VERSION;
    d.struct_size = (uint32_t)sizeof(AcContextDesc);
    d.backend_kind = AC_BACKEND_VULKAN;
    AcResult r = ac_runtime_context_create(&d, &v->ac);
    if (r != AC_SUCCESS) {
        free(v);
        return amu_fail(AMU_ERR_BACKEND_UNAVAILABLE,
                        "vulkan: ac_runtime_context_create failed: %s (%d). No GPU-type Vulkan device or loader "
                        "unavailable; CPU/software devices are rejected by design.", ac_name(r), (int)r);
    }
    r = ac_runtime_context_get_device_name(v->ac, ctx->device_name, sizeof(ctx->device_name));
    if (r != AC_SUCCESS) {
        ac_runtime_context_destroy(v->ac);
        free(v);
        return amu_fail(AMU_ERR_INTERNAL, "vulkan: get_device_name failed: %s", ac_name(r));
    }
    ctx->impl = v;
    return AMU_SUCCESS;
}

static amuStatus vk_destroy(amuContext* ctx) {
    VkCtx* v = (VkCtx*)ctx->impl;
    if (!v) return AMU_SUCCESS;
    if (v->event) {
        (void)ac_runtime_event_wait(v->event, AMU_VK_WAIT_TIMEOUT_NS);
        AcResult er = ac_runtime_event_destroy(v->event);
        if (er != AC_SUCCESS) {
            /* the GPU still owns the command buffer: releasing the event or the device would be undefined
             * behaviour, so both are leaked on purpose and reported */
            free(v);
            ctx->impl = NULL;
            return amu_fail(AMU_ERR_DEVICE_LOST, "vulkan: amuDestroy with GPU work still pending (%s); Vulkan device "
                                                 "and event intentionally leaked", ac_name(er));
        }
    }
    AcResult cr = ac_runtime_context_destroy(v->ac);
    free(v);
    ctx->impl = NULL;
    if (cr != AC_SUCCESS)
        return amu_fail(AMU_ERR_INTERNAL, "vulkan: ac_runtime_context_destroy failed (%s); device leaked", ac_name(cr));
    return AMU_SUCCESS;
}

static amuStatus vk_buf_alloc(amuContext* ctx, amuBuffer* b, size_t bytes) {
    VkCtx* v = (VkCtx*)ctx->impl;
    VkBuf* vb = (VkBuf*)calloc(1, sizeof(*vb));
    if (!vb) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "vulkan: host allocation failed");
    AcBufferDesc d;
    memset(&d, 0, sizeof(d));
    d.abi_version = AC_ABI_VERSION;
    d.struct_size = (uint32_t)sizeof(AcBufferDesc);
    d.size_bytes = bytes;
    d.usage = AC_BUFFER_USAGE_STORAGE;
    d.force_non_coherent = 0;
    AcResult r = ac_runtime_buffer_create(v->ac, &d, &vb->ac);
    if (r != AC_SUCCESS) {
        free(vb);
        return amu_fail(r == AC_ERR_OUT_OF_DEVICE_MEMORY ? AMU_ERR_OUT_OF_MEMORY : AMU_ERR_INTERNAL,
                        "vulkan: buffer_create(%zu bytes) failed: %s", bytes, ac_name(r));
    }
    uint32_t coh = 0;
    r = ac_runtime_buffer_get_mapped_ptr(vb->ac, &vb->mapped);
    if (r == AC_SUCCESS) r = ac_runtime_buffer_is_coherent(vb->ac, &coh);
    if (r != AC_SUCCESS || !vb->mapped) {
        ac_runtime_buffer_destroy(vb->ac);
        free(vb);
        return amu_fail(AMU_ERR_INTERNAL, "vulkan: buffer is not host-mapped: %s", ac_name(r));
    }
    vb->coherent = coh ? 1 : 0;
    b->impl = vb;
    return AMU_SUCCESS;
}

static amuStatus vk_buf_free(amuBuffer* b) {
    VkBuf* vb = (VkBuf*)b->impl;
    if (!vb) return AMU_SUCCESS;
    AcResult r = ac_runtime_buffer_destroy(vb->ac);
    free(vb);
    b->impl = NULL;
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "vulkan: buffer_destroy failed: %s", ac_name(r));
}

static void* vk_buf_host(amuBuffer* b) { return ((VkBuf*)b->impl)->mapped; }

static amuStatus vk_buf_to_device(amuBuffer* b) {
    VkBuf* vb = (VkBuf*)b->impl;
    if (vb->coherent) return AMU_SUCCESS;
    AcResult r = ac_runtime_buffer_flush(vb->ac, 0, b->size);
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "vulkan: buffer_flush failed: %s", ac_name(r));
}

static amuStatus vk_buf_to_host(amuBuffer* b) {
    VkBuf* vb = (VkBuf*)b->impl;
    if (vb->coherent) return AMU_SUCCESS;
    AcResult r = ac_runtime_buffer_invalidate(vb->ac, 0, b->size);
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "vulkan: buffer_invalidate failed: %s", ac_name(r));
}

static amuStatus vk_fn_load(amuFunction* fn) {
    const amuKernelImage* img = fn->image;
    if (!img->spirv || !img->spirv_size || !img->spirv_sha256 || strlen(img->spirv_sha256) != 64)
        return amu_fail(AMU_ERR_KERNEL_IMAGE,
                        "kernel '%s' has no Vulkan image (compile with amudacc --targets vulkan,...)", img->name);
    if (img->shared_bytes > 16384u)
        return amu_fail(AMU_ERR_NOT_SUPPORTED,
                        "kernel '%s' uses %u B of __shared__ memory; above the 16384 B every Vulkan device guarantees",
                        img->name, img->shared_bytes);
    VkFn* vf = (VkFn*)calloc(1, sizeof(*vf));
    if (!vf) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "vulkan: host allocation failed");
    fn->impl = vf;
    return AMU_SUCCESS;
}

static amuStatus vk_fn_unload(amuFunction* fn) {
    VkFn* vf = (VkFn*)fn->impl;
    if (!vf) return AMU_SUCCESS;
    AcResult first = AC_SUCCESS;
    for (uint32_t i = 0; i < vf->n; ++i) {
        AcResult r = ac_runtime_kernel_destroy(vf->pipes[i].k);
        if (r != AC_SUCCESS && first == AC_SUCCESS) first = r;
    }
    free(vf);
    fn->impl = NULL;
    return first == AC_SUCCESS ? AMU_SUCCESS
                               : amu_fail(AMU_ERR_INTERNAL, "vulkan: kernel_destroy failed: %s", ac_name(first));
}

/* One pipeline per block shape: the block size is a specialization constant (local_size_x_id). */
static amuStatus vk_pipeline(amuFunction* fn, amuDim3 block, AcKernel** out) {
    VkFn* vf = (VkFn*)fn->impl;
    VkCtx* v = (VkCtx*)fn->ctx->impl;
    const amuKernelImage* img = fn->image;
    for (uint32_t i = 0; i < vf->n; ++i) {
        if (vf->pipes[i].x == block.x && vf->pipes[i].y == block.y && vf->pipes[i].z == block.z) {
            *out = vf->pipes[i].k;
            return AMU_SUCCESS;
        }
    }
    if (vf->n == AMU_VK_MAX_PIPES)
        return amu_fail(AMU_ERR_NOT_SUPPORTED, "kernel '%s': more than %u distinct block shapes in one module",
                        img->name, AMU_VK_MAX_PIPES);

    AcRuntimeBindingDesc binds[AMU_MAX_BUFFERS];
    for (uint32_t i = 0; i < img->buffer_count; ++i) {
        binds[i].binding_index = i;
        binds[i].binding_type = AC_RUNTIME_BINDING_STORAGE_BUFFER;
        binds[i].stage_flags = AC_STAGE_COMPUTE;
    }
    const uint32_t dims[3] = { block.x, block.y, block.z };
    AcSpecConstantSchemaEntry schema[3];
    AcSpecConstant spec[3];
    for (uint32_t a = 0; a < 3; ++a) {
        schema[a].constant_id = a;
        schema[a].type = AC_SPEC_CONSTANT_U32;
        schema[a].default_value_bits = dims[a];
        schema[a].min_value_bits = 1;
        schema[a].max_value_bits = AMU_MAX_BLOCK_DIM;
        schema[a].overrides_local_size_axis = a + 1;
        spec[a].constant_id = a;
        spec[a].type = AC_SPEC_CONSTANT_U32;
        spec[a].value_bits = dims[a];
    }
    AcKernelRuntimeMetadata md;
    memset(&md, 0, sizeof(md));
    md.kernel_name = img->name;
    md.entry_point = "main";
    md.binding_count = img->buffer_count;
    md.bindings = binds;
    md.push_constant_size = img->push_constant_size;
    md.local_size.x = block.x;
    md.local_size.y = block.y;
    md.local_size.z = block.z;
    md.required_features = 0;
    md.expected_spirv_sha256 = img->spirv_sha256;
    md.specialization_schema_count = 3;
    md.specialization_schemas = schema;

    AcKernelDesc kd;
    memset(&kd, 0, sizeof(kd));
    kd.abi_version = AC_ABI_VERSION;
    kd.struct_size = (uint32_t)sizeof(AcKernelDesc);
    kd.spirv_code = img->spirv;
    kd.code_size_bytes = img->spirv_size;
    kd.metadata = &md;
    kd.specialization_count = 3;
    kd.specializations = spec;

    AcKernel* k = NULL;
    AcResult r = ac_runtime_kernel_create(v->ac, &kd, &k);
    if (r != AC_SUCCESS)
        return amu_fail(r == AC_ERR_INVALID_WORK_GROUP_SIZE || r == AC_ERR_INVALID_WORK_ITEM_DIMS ? AMU_ERR_LAUNCH
                                                                                                   : AMU_ERR_KERNEL_BUILD,
                        "vulkan: kernel '%s' pipeline for block (%u,%u,%u) failed: %s (%d)", img->name, block.x,
                        block.y, block.z, ac_name(r), (int)r);
    vf->pipes[vf->n].x = block.x;
    vf->pipes[vf->n].y = block.y;
    vf->pipes[vf->n].z = block.z;
    vf->pipes[vf->n].k = k;
    vf->n++;
    *out = k;
    return AMU_SUCCESS;
}

static amuStatus vk_launch(amuFunction* fn, amuDim3 grid, amuDim3 block, void** args) {
    VkCtx* v = (VkCtx*)fn->ctx->impl;
    const amuKernelImage* img = fn->image;
    AcKernel* k = NULL;
    amuStatus s = vk_pipeline(fn, block, &k);
    if (s != AMU_SUCCESS) return s;

    AcBuffer* bufs[AMU_MAX_BUFFERS];
    uint32_t pc[AMU_MAX_PUSH_BYTES / 4];
    memset(pc, 0, sizeof(pc));
    for (uint32_t i = 0; i < img->param_count; ++i) {
        const amuParamDesc* p = &img->params[i];
        if (p->kind == AMU_ARG_BUFFER) {
            if (p->slot >= img->buffer_count) return amu_fail(AMU_ERR_KERNEL_IMAGE, "bad binding slot");
            bufs[p->slot] = ((VkBuf*)(*(amuBuffer**)args[i])->impl)->ac;
        } else {
            if (p->slot + 4 > img->push_constant_size) return amu_fail(AMU_ERR_KERNEL_IMAGE, "bad push-constant slot");
            memcpy((char*)pc + p->slot, args[i], 4);
        }
    }
    AcLaunchDesc ld;
    memset(&ld, 0, sizeof(ld));
    ld.abi_version = AC_ABI_VERSION;
    ld.struct_size = (uint32_t)sizeof(AcLaunchDesc);
    ld.kernel = k;
    ld.buffers = bufs;
    ld.buffer_count = img->buffer_count;
    ld.push_constants = img->push_constant_size ? pc : NULL;
    ld.push_constants_size = img->push_constant_size;
    ld.grid_dim.x = grid.x;
    ld.grid_dim.y = grid.y;
    ld.grid_dim.z = grid.z;
    AcEvent* ev = NULL;
    AcResult r = ac_runtime_dispatch(v->ac, &ld, &ev);
    if (r != AC_SUCCESS)
        return amu_fail(AMU_ERR_LAUNCH, "vulkan: dispatch of '%s' grid (%u,%u,%u) block (%u,%u,%u) failed: %s (%d)",
                        img->name, grid.x, grid.y, grid.z, block.x, block.y, block.z, ac_name(r), (int)r);
    v->event = ev;
    return AMU_SUCCESS;
}

static amuStatus vk_wait(amuContext* ctx, double* gpu_ms) {
    VkCtx* v = (VkCtx*)ctx->impl;
    *gpu_ms = -1.0;
    if (!v->event) return amu_fail(AMU_ERR_INTERNAL, "vulkan: no event for in-flight launch");
    double t0 = vk_now_ms();
    AcResult r = ac_runtime_event_wait(v->event, AMU_VK_WAIT_TIMEOUT_NS);
    double t1 = vk_now_ms();
    if (r == AC_ERR_LAUNCH_TIMEOUT)
        return amu_fail(AMU_ERR_TIMEOUT, "vulkan: kernel did not finish within 60 s (context poisoned)");
    if (r != AC_SUCCESS) {
        amuStatus st = (r == AC_ERR_DRIVER_CRASH_OR_LOST) ? AMU_ERR_DEVICE_LOST : AMU_ERR_INTERNAL;
        return amu_fail(st, "vulkan: event_wait failed: %s (%d)", ac_name(r), (int)r);
    }
    uint64_t wall_ns = 0, gpu_ns = 0;
    r = ac_runtime_event_get_timing(v->event, &wall_ns, &gpu_ns);
    if (r == AC_SUCCESS && gpu_ns > 0) *gpu_ms = (double)gpu_ns / 1.0e6;
    double t2 = vk_now_ms();
    r = ac_runtime_event_destroy(v->event);
    v->event = NULL;
    ctx->trace_sub_ms[0] = t1 - t0;
    ctx->trace_sub_ms[1] = t2 - t1;
    ctx->trace_sub_ms[2] = vk_now_ms() - t2;
    if (r != AC_SUCCESS) return amu_fail(AMU_ERR_INTERNAL, "vulkan: event_destroy failed: %s", ac_name(r));
    return AMU_SUCCESS;
}

const amuBackendOps amu_vulkan_ops = {
    "vulkan", vk_init, vk_destroy, vk_buf_alloc, vk_buf_free, vk_buf_host, vk_buf_to_device, vk_buf_to_host,
    vk_fn_load, vk_fn_unload, vk_launch, vk_wait,
};
