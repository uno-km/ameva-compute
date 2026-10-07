/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
/* AMUDA CUDA-like runtime: OpenCL backend on top of the legacy AMUDA OpenCL engine (libamuda). */
#define AMU_WITH_OPENCL 1
#include "amuda_rt_internal.h"
#include "amuda.h" /* legacy engine: ac_init / ac_kernel_load_source / ac_launch (OpenCL, dlopen'd driver) */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same candidate order as the legacy engine (src/amuda/amuda_context.c). Used ONLY to explain a failed
 * ac_init: which library the dynamic linker refused, which one actually opened, how many platforms it has. */
static const char* const k_cl_candidates[] = {
    "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/vendor/lib64/egl/libGLES_mali.so",
    "/vendor/lib64/libmali.so", "/system/lib64/libOpenCL.so", "libOpenCL.so", NULL};

typedef int32_t (*amu_clGetPlatformIDs_fn)(uint32_t, void**, uint32_t*);

/* append with truncation; n never exceeds cap-1 */
static void cl_append(char* out, size_t cap, size_t* n, const char* piece) {
    if (*n + 1 >= cap) return;
    int w = snprintf(out + *n, cap - *n, "%s", piece);
    if (w < 0) return;
    *n += (size_t)w;
    if (*n >= cap) *n = cap - 1;
}

static void cl_probe(char* out, size_t cap) {
    size_t n = 0;
    out[0] = 0;
    for (int i = 0; k_cl_candidates[i]; ++i) {
        void* h = dlopen(k_cl_candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            const char* e = dlerror();
            const char* brief = e ? e : "?";
            if (e && strstr(e, "not accessible for the namespace")) brief = "blocked by linker namespace";
            else if (e && (strstr(e, "not found") || strstr(e, "No such file"))) brief = "not found";
            char piece[192];
            snprintf(piece, sizeof(piece), " [%s: %.80s]", k_cl_candidates[i], brief);
            cl_append(out, cap, &n, piece);
            continue;
        }
        /* ISO C has no object->function pointer cast; POSIX guarantees the representation, so copy it */
        void* sym = dlsym(h, "clGetPlatformIDs");
        amu_clGetPlatformIDs_fn gp = NULL;
        if (sym) memcpy(&gp, &sym, sizeof(gp));
        Dl_info info;
        const char* real = (sym && dladdr(sym, &info) && info.dli_fname) ? info.dli_fname : k_cl_candidates[i];
        uint32_t np = 0;
        int32_t rc = gp ? gp(0, NULL, &np) : -1;
        char piece[320];
        snprintf(piece, sizeof(piece), " [%s: opened %s, clGetPlatformIDs rc=%d platforms=%u]", k_cl_candidates[i],
                 real, (int)rc, (unsigned)np);
        cl_append(out, cap, &n, piece);
        dlclose(h);
        break; /* the legacy engine stops at the first library that opens */
    }
}

typedef struct {
    AcContext* ac;
    AcEvent* event;
} ClCtx;

static amuStatus cl_init(amuContext* ctx) {
    ClCtx* c = (ClCtx*)calloc(1, sizeof(*c));
    if (!c) return amu_fail(AMU_ERR_OUT_OF_MEMORY, "opencl: host allocation failed");
    AcResult r = ac_init(&c->ac);
    if (r != AC_SUCCESS || !c->ac) {
        free(c);
        char probe[1024];
        cl_probe(probe, sizeof(probe));
        return amu_fail(AMU_ERR_BACKEND_UNAVAILABLE,
                        "opencl: ac_init failed (%d). probe:%s. On Android 14+ the app linker namespace blocks "
                        "/vendor libraries unless LD_LIBRARY_PATH includes their directory; use "
                        "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/vendor/lib64/egl (system first, so system "
                        "libraries keep binding to system libbase/libcutils).", (int)r, probe);
    }
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s (%s)", c->ac->device_name, c->ac->vendor_name);
    ctx->impl = c;
    return AMU_SUCCESS;
}

static amuStatus cl_destroy(amuContext* ctx) {
    ClCtx* c = (ClCtx*)ctx->impl;
    if (!c) return AMU_SUCCESS;
    AcResult first = AC_SUCCESS;
    if (c->event) {
        AcResult r = ac_event_wait(c->ac, c->event);
        if (r != AC_SUCCESS) first = r;
        r = ac_event_free(c->ac, c->event);
        if (r != AC_SUCCESS && first == AC_SUCCESS) first = r;
    }
    AcResult r = ac_destroy(c->ac);
    if (r != AC_SUCCESS && first == AC_SUCCESS) first = r;
    free(c);
    ctx->impl = NULL;
    return first == AC_SUCCESS ? AMU_SUCCESS
                               : amu_fail(AMU_ERR_INTERNAL, "opencl: context teardown failed (%d)", (int)first);
}

static amuStatus cl_buf_alloc(amuContext* ctx, amuBuffer* b, size_t bytes) {
    ClCtx* c = (ClCtx*)ctx->impl;
    AcBuffer* ab = NULL;
    AcResult r = ac_buffer_alloc(c->ac, bytes, &ab);
    if (r != AC_SUCCESS || !ab)
        return amu_fail(r == AC_ERR_OUT_OF_DEVICE_MEMORY || r == AC_ERR_OUT_OF_HOST_MEMORY ? AMU_ERR_OUT_OF_MEMORY
                                                                                            : AMU_ERR_INTERNAL,
                        "opencl: ac_buffer_alloc(%zu) failed (%d)", bytes, (int)r);
    b->impl = ab; /* the legacy engine leaves new buffers mapped for host access */
    return AMU_SUCCESS;
}

static amuStatus cl_buf_free(amuBuffer* b) {
    ClCtx* c = (ClCtx*)b->ctx->impl;
    if (!b->impl) return AMU_SUCCESS;
    AcResult r = ac_buffer_free(c->ac, (AcBuffer*)b->impl);
    b->impl = NULL;
    if (r == AC_SUCCESS) return AMU_SUCCESS;
    if (r == AC_ERR_RESOURCE_IN_USE)
        return amu_fail(AMU_ERR_DEVICE_LOST, "opencl: clFinish failed while freeing a buffer; its host memory was "
                                             "intentionally leaked because the driver may still access it");
    return amu_fail(AMU_ERR_INTERNAL, "opencl: ac_buffer_free failed (%d)", (int)r);
}

static void* cl_buf_host(amuBuffer* b) { return ((AcBuffer*)b->impl)->host_ptr; }

static amuStatus cl_buf_to_device(amuBuffer* b) {
    ClCtx* c = (ClCtx*)b->ctx->impl;
    AcResult r = ac_buffer_sync_to_device(c->ac, (AcBuffer*)b->impl);
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "opencl: unmap (sync_to_device) failed (%d)", (int)r);
}

static amuStatus cl_buf_to_host(amuBuffer* b) {
    ClCtx* c = (ClCtx*)b->ctx->impl;
    AcResult r = ac_buffer_sync_to_host(c->ac, (AcBuffer*)b->impl);
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "opencl: map (sync_to_host) failed (%d)", (int)r);
}

static amuStatus cl_fn_load(amuFunction* fn) {
    ClCtx* c = (ClCtx*)fn->ctx->impl;
    const amuKernelImage* img = fn->image;
    if (!img->opencl_source)
        return amu_fail(AMU_ERR_KERNEL_IMAGE,
                        "kernel '%s' has no OpenCL image (compile with amudacc --targets ...,opencl)", img->name);
    if (c->ac->local_mem_size > 0 && img->shared_bytes > c->ac->local_mem_size)
        return amu_fail(AMU_ERR_NOT_SUPPORTED, "opencl: kernel '%s' needs %u B __shared__, device local memory is %llu B",
                        img->name, img->shared_bytes, (unsigned long long)c->ac->local_mem_size);
    AcKernel* k = NULL;
    AcResult r = ac_kernel_load_source(c->ac, img->opencl_source, img->name, &k);
    if (r != AC_SUCCESS || !k)
        return amu_fail(AMU_ERR_KERNEL_BUILD, "opencl: building kernel '%s' failed (%d); see [AMUDA_BUILD_ERROR] above",
                        img->name, (int)r);
    fn->impl = k;
    return AMU_SUCCESS;
}

static amuStatus cl_fn_unload(amuFunction* fn) {
    ClCtx* c = (ClCtx*)fn->ctx->impl;
    if (!fn->impl) return AMU_SUCCESS;
    AcResult r = ac_kernel_free(c->ac, (AcKernel*)fn->impl);
    fn->impl = NULL;
    return r == AC_SUCCESS ? AMU_SUCCESS : amu_fail(AMU_ERR_INTERNAL, "opencl: ac_kernel_free failed (%d)", (int)r);
}

static amuStatus cl_launch(amuFunction* fn, amuDim3 grid, amuDim3 block, void** args) {
    ClCtx* c = (ClCtx*)fn->ctx->impl;
    const amuKernelImage* img = fn->image;
    void* argv[AMU_MAX_BUFFERS + AMU_MAX_PUSH_BYTES / 4];
    size_t sizes[AMU_MAX_BUFFERS + AMU_MAX_PUSH_BYTES / 4];
    for (uint32_t i = 0; i < img->param_count; ++i) {
        if (img->params[i].kind == AMU_ARG_BUFFER) {
            AcBuffer* ab = (AcBuffer*)(*(amuBuffer**)args[i])->impl;
            argv[i] = &ab->native_mem;
            sizes[i] = sizeof(void*);
        } else {
            argv[i] = args[i];
            sizes[i] = 4;
        }
    }
    AcEvent* ev = NULL;
    AcResult r = ac_launch(c->ac, (AcKernel*)fn->impl, ac_dim3(grid.x, grid.y, grid.z),
                           ac_dim3(block.x, block.y, block.z), img->param_count, argv, sizes, &ev);
    if (r != AC_SUCCESS || !ev)
        return amu_fail(AMU_ERR_LAUNCH, "opencl: launch of '%s' grid (%u,%u,%u) block (%u,%u,%u) failed (%d)%s",
                        img->name, grid.x, grid.y, grid.z, block.x, block.y, block.z, (int)r,
                        r == AC_ERR_MALI_INFINITE_LOOP_DETECTED
                            ? " - legacy Mali guard rejects 1D blocks smaller than 32 threads"
                            : "");
    c->event = ev;
    return AMU_SUCCESS;
}

static amuStatus cl_wait(amuContext* ctx, double* gpu_ms) {
    ClCtx* c = (ClCtx*)ctx->impl;
    *gpu_ms = -1.0;
    if (!c->event) return amu_fail(AMU_ERR_INTERNAL, "opencl: no event for in-flight launch");
    AcResult r = ac_event_wait(c->ac, c->event);
    if (r != AC_SUCCESS) {
        (void)ac_event_free(c->ac, c->event);
        c->event = NULL;
        return amu_fail(AMU_ERR_DEVICE_LOST, "opencl: clWaitForEvents failed (%d) (context poisoned)", (int)r);
    }
    uint64_t us = ac_event_get_duration_us(c->event);
    if (us > 0) *gpu_ms = (double)us / 1000.0;
    (void)ac_event_free(c->ac, c->event);
    c->event = NULL;
    return AMU_SUCCESS;
}

const amuBackendOps amu_opencl_ops = {
    "opencl", cl_init, cl_destroy, cl_buf_alloc, cl_buf_free, cl_buf_host, cl_buf_to_device, cl_buf_to_host,
    cl_fn_load, cl_fn_unload, cl_launch, cl_wait,
};
