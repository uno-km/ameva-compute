/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// SPDX-License-Identifier: Apache-2.0
// ac_cl_kat: known-answer test of the OpenCL backend of ac_runtime (AC_BACKEND_OPENCL) through the public C ABI, and
// of the backend-neutral host access calls (ac_runtime_buffer_write / read / fill / get_address) on both backends.
//
//   1. context: backend kind, device name, limits
//   2. host access: write / read / fill round trips, offsets, out-of-range sizes rejected; get_mapped_ptr, flush,
//      invalidate and is_coherent rejected with AC_ERR_BUFFER_NOT_MAPPED on OpenCL buffers
//   3. kernels from OpenCL C: a kernel with two bindings and a push-constant struct; argument-count mismatch and build
//      failures rejected; SPIR-V creation rejected on an OpenCL context
//   4. a timestamped batch of three dispatches with binding offsets, a barrier and a flush: exact results, dispatch
//      times, stats; a host write made after recording and before submit is seen by the kernel (nothing runs before
//      flush / submit)
//   5. the program binary cache: data from one context merged into a second, the kernel created again from it; data
//      of a damaged header rejected
//   6. lifetime: context destruction refused while children are alive
//   7. Vulkan (when a Vulkan device exists): the same host access calls on a Vulkan buffer
// Prints AC_CL_KAT=PASS (n checks) or AC_CL_KAT=FAIL with the first failing check. Exit status 0 only on PASS.
// Build: clang -O2 -I include -o ac_cl_kat tests/ac_cl_kat.c <dir>/libac_runtime.so.2 -Wl,-rpath,<dir>
// Run: ./ac_cl_kat (ac_runtime opens the vendor OpenCL driver itself; no LD_LIBRARY_PATH for it)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_runtime.h"

static int g_checks = 0;

#define CHECK(cond, ...)                                                                                              \
    do {                                                                                                              \
        ++g_checks;                                                                                                   \
        if (!(cond)) {                                                                                                \
            printf("AC_CL_KAT=FAIL (check %d, line %d: ", g_checks, __LINE__);                                        \
            printf(__VA_ARGS__);                                                                                      \
            printf(")\n");                                                                                            \
            exit(1);                                                                                                  \
        }                                                                                                             \
    } while (0)

// y[i] = a * x[i] + y[i] + b for i < n, x and y from their binding starts
static const char* kAxpy =
    "typedef struct { uint n; float a; float b; } P;\n"
    "__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n"
    "void axpy(__global const float* x, __global float* y, P pc) {\n"
    "    const uint i = get_global_id(0);\n"
    "    if (i < pc.n) y[i] = pc.a * x[i] + y[i] + pc.b;\n"
    "}\n";

// the same with the binding ranges: y[i] += the number of floats bound to x plus the bytes bound to y
static const char* kRanges =
    "typedef struct { uint n; float a; float b; } P;\n"
    "typedef struct { uint bytes[2]; } R;\n"
    "__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n"
    "void axpy(__global const float* x, __global float* y, P pc, R ranges) {\n"
    "    const uint i = get_global_id(0);\n"
    "    if (i < pc.n) y[i] = (float)(ranges.bytes[0] / 4u) + (float)ranges.bytes[1];\n"
    "}\n";

static const AcRuntimeBindingDesc kTwo[2] = { { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
                                              { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE } };

static AcKernelRuntimeMetadata axpy_meta(uint32_t bindings, size_t push) {
    AcKernelRuntimeMetadata m;
    memset(&m, 0, sizeof(m));
    m.kernel_name = "axpy";
    m.entry_point = "axpy";
    m.binding_count = bindings;
    m.bindings = kTwo;
    m.push_constant_size = push;
    m.local_size.x = 64;
    m.local_size.y = 1;
    m.local_size.z = 1;
    return m;
}

static AcResult make_kernel_flags(AcContext* ctx, const char* src, const AcKernelRuntimeMetadata* m, uint32_t flags,
                                  AcKernel** k) {
    AcKernelSourceDesc d = { AC_ABI_VERSION, (uint32_t)sizeof(AcKernelSourceDesc), src, 0, "-cl-fast-relaxed-math", m,
                             flags };
    return ac_runtime_kernel_create_from_source(ctx, &d, k);
}

static AcResult make_kernel(AcContext* ctx, const char* src, const AcKernelRuntimeMetadata* m, AcKernel** k) {
    return make_kernel_flags(ctx, src, m, 0, k);
}

static AcBuffer* make_buffer(AcContext* ctx, size_t bytes) {
    AcBufferDesc bd = { AC_ABI_VERSION, (uint32_t)sizeof(AcBufferDesc), bytes,
                        AC_BUFFER_USAGE_STORAGE | AC_BUFFER_USAGE_TRANSFER_SRC | AC_BUFFER_USAGE_TRANSFER_DST, 0 };
    AcBuffer* b = NULL;
    const AcResult r = ac_runtime_buffer_create(ctx, &bd, &b);
    CHECK(r == AC_SUCCESS && b, "buffer_create(%zu) = %d", bytes, (int)r);
    return b;
}

// write / read / fill round trips on any backend
static void host_access(AcContext* ctx, const char* tag) {
    const size_t n = 1 << 18; // 1 MiB of floats
    AcBuffer* b = make_buffer(ctx, n * 4);
    float* h = (float*)malloc(n * 4);
    float* r = (float*)malloc(n * 4);
    for (size_t i = 0; i < n; ++i) h[i] = (float)i * 0.25f - 7.0f;
    CHECK(ac_runtime_buffer_write(b, 0, h, n * 4) == AC_SUCCESS, "%s write", tag);
    memset(r, 0, n * 4);
    CHECK(ac_runtime_buffer_read(b, 0, r, n * 4) == AC_SUCCESS, "%s read", tag);
    CHECK(memcmp(h, r, n * 4) == 0, "%s read back differs", tag);
    CHECK(ac_runtime_buffer_fill(b, 4096, 1000, 0x7F) == AC_SUCCESS, "%s fill", tag);
    unsigned char* rb = (unsigned char*)r;
    CHECK(ac_runtime_buffer_read(b, 4000, rb, 1200) == AC_SUCCESS, "%s read after fill", tag);
    int bad = -1;
    for (int i = 0; i < 1200 && bad < 0; ++i) {
        const int off = 4000 + i;
        const unsigned char want = (off >= 4096 && off < 5096) ? 0x7F : ((const unsigned char*)h)[off];
        if (rb[i] != want) bad = off;
    }
    CHECK(bad < 0, "%s fill: byte %d differs", tag, bad);
    const float v[3] = { 1.5f, -2.5f, 3.75f };
    CHECK(ac_runtime_buffer_write(b, 12, v, sizeof(v)) == AC_SUCCESS, "%s unaligned write", tag);
    float w[3] = { 0, 0, 0 };
    CHECK(ac_runtime_buffer_read(b, 12, w, sizeof(w)) == AC_SUCCESS && memcmp(v, w, sizeof(v)) == 0,
          "%s unaligned read back", tag);
    CHECK(ac_runtime_buffer_write(b, n * 4 - 4, v, 8) == AC_ERR_INVALID_BUFFER_SIZE, "%s write past the end", tag);
    CHECK(ac_runtime_buffer_read(b, n * 4 + 1, w, 0) == AC_ERR_INVALID_BUFFER_SIZE, "%s read offset past the end", tag);
    void* addr = NULL;
    CHECK(ac_runtime_buffer_get_address(b, &addr) == AC_SUCCESS && addr, "%s get_address", tag);
    CHECK(ac_runtime_buffer_destroy(b) == AC_SUCCESS, "%s buffer_destroy", tag);
    free(h);
    free(r);
}

int main(void) {
    // 1. context
    AcContextDesc cd = { AC_ABI_VERSION, (uint32_t)sizeof(AcContextDesc), AC_BACKEND_OPENCL, 0 };
    AcContext* ctx = NULL;
    AcResult r = ac_runtime_context_create(&cd, &ctx);
    CHECK(r == AC_SUCCESS && ctx, "context_create(AC_BACKEND_OPENCL) = %d", (int)r);
    AcBackendKind kind = AC_BACKEND_UNKNOWN;
    CHECK(ac_runtime_context_get_backend_kind(ctx, &kind) == AC_SUCCESS && kind == AC_BACKEND_OPENCL, "backend kind %u",
          (unsigned)kind);
    char name[256] = { 0 };
    CHECK(ac_runtime_context_get_device_name(ctx, name, sizeof(name)) == AC_SUCCESS && name[0], "device name");
    AcDeviceLimits lim;
    memset(&lim, 0, sizeof(lim));
    lim.abi_version = AC_ABI_VERSION;
    lim.struct_size = (uint32_t)sizeof(lim);
    CHECK(ac_runtime_context_get_limits(ctx, &lim) == AC_SUCCESS, "get_limits");
    printf("device %s: binding range %llu, offset alignment %llu, max allocation %llu, workgroup %u (%u,%u,%u), "
           "local memory %u, subgroup %u, timestamps %u, int8 dot %u (accelerated flag %u)\n",
           name, (unsigned long long)lim.max_storage_buffer_range,
           (unsigned long long)lim.min_storage_buffer_offset_alignment,
           (unsigned long long)lim.max_memory_allocation_size, lim.max_compute_workgroup_invocations,
           lim.max_compute_workgroup_size[0], lim.max_compute_workgroup_size[1], lim.max_compute_workgroup_size[2],
           lim.max_compute_shared_memory_size, lim.subgroup_size, lim.timestamps_supported, lim.integer_dot_product,
           lim.integer_dot_product_accelerated);
    CHECK(lim.min_storage_buffer_offset_alignment >= 16 && lim.max_storage_buffer_range > 0, "limits plausible");

    // 2. host access, and the mapped-memory calls refused on SVM buffers
    host_access(ctx, "opencl");
    {
        AcBuffer* b = make_buffer(ctx, 4096);
        void* p = (void*)1;
        uint32_t coh = 7;
        CHECK(ac_runtime_buffer_get_mapped_ptr(b, &p) == AC_ERR_BUFFER_NOT_MAPPED && p == NULL, "get_mapped_ptr refused");
        CHECK(ac_runtime_buffer_flush(b, 0, 64) == AC_ERR_BUFFER_NOT_MAPPED, "flush refused");
        CHECK(ac_runtime_buffer_invalidate(b, 0, 64) == AC_ERR_BUFFER_NOT_MAPPED, "invalidate refused");
        CHECK(ac_runtime_buffer_is_coherent(b, &coh) == AC_ERR_BUFFER_NOT_MAPPED, "is_coherent refused");
        CHECK(ac_runtime_buffer_destroy(b) == AC_SUCCESS, "destroy");
    }

    // 3. kernels
    const AcKernelRuntimeMetadata m2 = axpy_meta(2, 12);
    AcKernel* k = NULL;
    r = make_kernel(ctx, kAxpy, &m2, &k);
    CHECK(r == AC_SUCCESS && k, "kernel_create_from_source(axpy) = %d", (int)r);
    char ident[65] = { 0 };
    CHECK(ac_runtime_kernel_get_pipeline_identity(k, ident, sizeof(ident)) == AC_SUCCESS && strlen(ident) == 64, "identity");
    size_t ps = 0;
    CHECK(ac_runtime_kernel_get_push_constant_size(k, &ps) == AC_SUCCESS && ps == 12, "push size %zu", ps);
    AcDim3 ls = { 0, 0, 0 };
    CHECK(ac_runtime_kernel_get_effective_local_size(k, &ls) == AC_SUCCESS && ls.x == 64 && ls.y == 1, "local size");
    {
        const AcKernelRuntimeMetadata m1 = axpy_meta(1, 12); // the source takes three arguments
        AcKernel* bad = NULL;
        CHECK(make_kernel(ctx, kAxpy, &m1, &bad) == AC_ERR_KERNEL_ARG_INVALID && bad == NULL, "argument count mismatch");
        printf("(the build error below is expected)\n");
        CHECK(make_kernel(ctx, "__kernel void axpy(__global float* x, __global float* y, P pc) { undefined_call(); }",
                          &m2, &bad) == AC_ERR_KERNEL_BUILD_FAILED && bad == NULL, "build failure");
        const uint32_t word = 0x07230203u;
        AcKernelDesc spv = { AC_ABI_VERSION, (uint32_t)sizeof(AcKernelDesc), &word, 4, &m2, 0, NULL };
        CHECK(ac_runtime_kernel_create(ctx, &spv, &bad) == AC_ERR_FEATURE_NOT_SUPPORTED, "SPIR-V refused on OpenCL");
    }

    // 4. a batch: three dispatches over y with x read from a binding offset
    const uint32_t n = 100000;
    const size_t xoff_elems = 16; // binding offset 64 bytes
    AcBuffer* bx = make_buffer(ctx, (n + xoff_elems) * 4);
    AcBuffer* by = make_buffer(ctx, n * 4);
    float* hx = (float*)malloc((n + xoff_elems) * 4);
    float* hy = (float*)malloc(n * 4);
    float* want = (float*)malloc(n * 4);
    for (uint32_t i = 0; i < n + xoff_elems; ++i) hx[i] = (float)((int)(i % 1000) - 500) * 0.5f;
    for (uint32_t i = 0; i < n; ++i) hy[i] = (float)(i % 77);
    CHECK(ac_runtime_buffer_write(bx, 0, hx, (n + xoff_elems) * 4) == AC_SUCCESS, "write x");
    CHECK(ac_runtime_buffer_write(by, 0, hy, n * 4) == AC_SUCCESS, "write y");
    AcBatchDesc bdesc = { AC_ABI_VERSION, (uint32_t)sizeof(AcBatchDesc), 8,
                          AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_TIMESTAMPS | AC_BATCH_FLAG_DISPATCH_TIMESTAMPS };
    AcCommandBatch* batch = NULL;
    CHECK(ac_runtime_batch_begin_ex(ctx, &bdesc, &batch) == AC_SUCCESS && batch, "batch_begin_ex");
    AcBufferBinding bind[2] = { { bx, xoff_elems * 4, (uint64_t)n * 4 }, { by, 0, (uint64_t)n * 4 } };
    const struct { uint32_t n; float a; float b; } p1 = { n, 2.0f, 1.0f }, p2 = { n, -1.0f, 0.5f }, p3 = { n, 0.5f, -3.0f };
    const AcDim3 grid = { (n + 63) / 64, 1, 1 };
    CHECK(ac_runtime_batch_dispatch_bindings(batch, k, bind, 2, &p1, sizeof(p1), grid) == AC_SUCCESS, "dispatch 1");
    CHECK(ac_runtime_batch_barrier(batch) == AC_SUCCESS, "barrier");
    CHECK(ac_runtime_batch_dispatch_bindings(batch, k, bind, 2, &p2, sizeof(p2), grid) == AC_SUCCESS, "dispatch 2");
    CHECK(ac_runtime_batch_flush(batch) == AC_SUCCESS, "flush");
    CHECK(ac_runtime_batch_barrier(batch) == AC_SUCCESS, "barrier 2");
    CHECK(ac_runtime_batch_dispatch_bindings(batch, k, bind, 2, &p3, sizeof(p3), grid) == AC_SUCCESS, "dispatch 3");
    {
        // misaligned binding offset and a range past the end are refused at recording
        AcBufferBinding badb[2] = { { bx, 4, (uint64_t)n * 4 }, { by, 0, (uint64_t)n * 4 } };
        CHECK(ac_runtime_batch_dispatch_bindings(batch, k, badb, 2, &p1, sizeof(p1), grid) == AC_ERR_BUFFER_ALIGNMENT_INVALID,
              "misaligned binding offset");
        badb[0].offset = 0;
        badb[1].range = (uint64_t)n * 4 + 4;
        CHECK(ac_runtime_batch_dispatch_bindings(batch, k, badb, 2, &p1, sizeof(p1), grid) == AC_ERR_INVALID_BUFFER_SIZE,
              "range past the end");
    }
    AcEvent* ev = NULL;
    CHECK(ac_runtime_batch_submit(batch, &ev) == AC_SUCCESS && ev, "submit");
    CHECK(ac_runtime_event_wait(ev, UINT64_MAX) == AC_SUCCESS, "wait");
    CHECK(ac_runtime_event_wait(ev, 0) == AC_SUCCESS, "second wait");
    uint64_t wall = 0, gpu = 0;
    CHECK(ac_runtime_event_get_timing(ev, &wall, &gpu) == AC_SUCCESS && wall > 0 && gpu > 0, "timing %llu %llu",
          (unsigned long long)wall, (unsigned long long)gpu);
    uint64_t times[8] = { 0 };
    uint32_t nt = 0;
    CHECK(ac_runtime_batch_get_dispatch_times(batch, times, 8, &nt) == AC_SUCCESS && nt == 3, "dispatch times (%u)", nt);
    AcBatchStats st;
    memset(&st, 0, sizeof(st));
    st.abi_version = AC_ABI_VERSION;
    st.struct_size = (uint32_t)sizeof(st);
    CHECK(ac_runtime_batch_get_stats(batch, &st) == AC_SUCCESS && st.dispatches == 3 && st.barriers == 2 && st.flushes == 1,
          "stats %u %u %u", st.dispatches, st.barriers, st.flushes);
    float* got = (float*)malloc(n * 4);
    CHECK(ac_runtime_buffer_read(by, 0, got, n * 4) == AC_SUCCESS, "read y");
    for (uint32_t i = 0; i < n; ++i) {
        float y = hy[i];
        const float x = hx[i + xoff_elems];
        y = p1.a * x + y + p1.b;
        y = p2.a * x + y + p2.b;
        y = p3.a * x + y + p3.b;
        want[i] = y;
    }
    CHECK(memcmp(got, want, n * 4) == 0, "batch result differs (first: %g, want %g)", got[0], want[0]);
    printf("batch: 3 dispatches of %u items, GPU %.3f ms, wall %.3f ms, per dispatch %.3f / %.3f / %.3f ms\n", n,
           gpu * 1e-6, wall * 1e-6, times[0] * 1e-6, times[1] * 1e-6, times[2] * 1e-6);
    CHECK(ac_runtime_event_destroy(ev) == AC_SUCCESS, "event destroy");
    ac_runtime_batch_destroy(batch);

    // nothing runs before submit: a host write after recording is seen by the dispatch
    {
        AcBatchDesc d1 = { AC_ABI_VERSION, (uint32_t)sizeof(AcBatchDesc), 4, 0 };
        AcCommandBatch* b2 = NULL;
        CHECK(ac_runtime_batch_begin_ex(ctx, &d1, &b2) == AC_SUCCESS, "batch 2");
        const struct { uint32_t n; float a; float b; } q = { n, 1.0f, 0.0f };
        CHECK(ac_runtime_batch_dispatch_bindings(b2, k, bind, 2, &q, sizeof(q), grid) == AC_SUCCESS, "record");
        for (uint32_t i = 0; i < n; ++i) hy[i] = 1000.0f + (float)i;
        CHECK(ac_runtime_buffer_write(by, 0, hy, n * 4) == AC_SUCCESS, "write after recording");
        AcEvent* e2 = NULL;
        CHECK(ac_runtime_batch_submit(b2, &e2) == AC_SUCCESS && ac_runtime_event_wait(e2, UINT64_MAX) == AC_SUCCESS,
              "submit 2");
        CHECK(ac_runtime_buffer_read(by, 0, got, n * 4) == AC_SUCCESS, "read 2");
        uint32_t wrong = 0, first = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (got[i] != hx[i + xoff_elems] + (1000.0f + (float)i) && wrong++ == 0) first = i;
        }
        CHECK(wrong == 0, "recorded dispatch ran before submit or missed the write (%u values, first i=%u: %g)", wrong,
              first, got[first]);
        ac_runtime_event_destroy(e2);
        ac_runtime_batch_destroy(b2);
    }

    // a kernel that takes the binding ranges receives the bytes of each bound range
    {
        AcKernel* kr = NULL;
        CHECK(make_kernel(ctx, kRanges, &m2, &kr) == AC_ERR_KERNEL_ARG_INVALID && kr == NULL,
              "a fourth argument without AC_KERNEL_SOURCE_BINDING_RANGES");
        CHECK(make_kernel_flags(ctx, kAxpy, &m2, AC_KERNEL_SOURCE_BINDING_RANGES, &kr) == AC_ERR_KERNEL_ARG_INVALID,
              "AC_KERNEL_SOURCE_BINDING_RANGES without the argument");
        CHECK(make_kernel_flags(ctx, kAxpy, &m2, 1u << 7, &kr) == AC_ERR_INVALID_ARGUMENT, "unknown source flag");
        CHECK(make_kernel_flags(ctx, kRanges, &m2, AC_KERNEL_SOURCE_BINDING_RANGES, &kr) == AC_SUCCESS && kr,
              "kernel with a binding-range argument");
        AcBatchDesc d1 = { AC_ABI_VERSION, (uint32_t)sizeof(AcBatchDesc), 2, 0 };
        AcCommandBatch* b3 = NULL;
        CHECK(ac_runtime_batch_begin_ex(ctx, &d1, &b3) == AC_SUCCESS, "batch 3");
        const uint32_t part = 1024; // floats of y written; x bound from its offset, y for part floats only
        AcBufferBinding br[2] = { { bx, xoff_elems * 4, (uint64_t)n * 4 }, { by, 0, (uint64_t)part * 4 } };
        const struct { uint32_t n; float a; float b; } q = { part, 0.0f, 0.0f };
        const AcDim3 g3 = { part / 64, 1, 1 };
        CHECK(ac_runtime_batch_dispatch_bindings(b3, kr, br, 2, &q, sizeof(q), g3) == AC_SUCCESS, "record 3");
        AcEvent* e3 = NULL;
        CHECK(ac_runtime_batch_submit(b3, &e3) == AC_SUCCESS && ac_runtime_event_wait(e3, UINT64_MAX) == AC_SUCCESS,
              "submit 3");
        CHECK(ac_runtime_buffer_read(by, 0, got, part * 4) == AC_SUCCESS, "read 3");
        const float want3 = (float)n + (float)(part * 4);
        uint32_t wrong = 0;
        for (uint32_t i = 0; i < part; ++i) wrong += got[i] != want3;
        CHECK(wrong == 0, "binding ranges: %u values differ (first %g, want %g)", wrong, got[0], want3);
        ac_runtime_event_destroy(e3);
        ac_runtime_batch_destroy(b3);
        CHECK(ac_runtime_kernel_destroy(kr) == AC_SUCCESS, "range kernel destroy");
    }

    // 6. lifetime
    CHECK(ac_runtime_context_destroy(ctx) == AC_ERR_RESOURCE_IN_USE, "context destroy refused with live children");

    // 5. program binary cache into a second context
    size_t csz = 0;
    CHECK(ac_runtime_pipeline_cache_get_data(ctx, NULL, &csz) == AC_SUCCESS && csz > 64, "cache size %zu", csz);
    unsigned char* cache = (unsigned char*)malloc(csz);
    CHECK(ac_runtime_pipeline_cache_get_data(ctx, cache, &csz) == AC_SUCCESS, "cache data");
    {
        AcContext* c2 = NULL;
        CHECK(ac_runtime_context_create(&cd, &c2) == AC_SUCCESS, "second context");
        CHECK(ac_runtime_pipeline_cache_merge(c2, cache, csz) == AC_SUCCESS, "merge");
        cache[9] ^= 0x55; // inside the device name
        CHECK(ac_runtime_pipeline_cache_merge(c2, cache, csz) == AC_ERR_INVALID_ARGUMENT, "damaged cache refused");
        AcKernel* k2 = NULL;
        CHECK(make_kernel(c2, kAxpy, &m2, &k2) == AC_SUCCESS, "kernel from the merged cache");
        char ident2[65] = { 0 };
        ac_runtime_kernel_get_pipeline_identity(k2, ident2, sizeof(ident2));
        CHECK(strcmp(ident, ident2) == 0, "same identity");
        CHECK(ac_runtime_kernel_destroy(k2) == AC_SUCCESS && ac_runtime_context_destroy(c2) == AC_SUCCESS, "second teardown");
    }
    free(cache);

    CHECK(ac_runtime_kernel_destroy(k) == AC_SUCCESS, "kernel destroy");
    CHECK(ac_runtime_buffer_destroy(bx) == AC_SUCCESS && ac_runtime_buffer_destroy(by) == AC_SUCCESS, "buffers destroy");
    CHECK(ac_runtime_context_destroy(ctx) == AC_SUCCESS, "context destroy");
    free(hx);
    free(hy);
    free(want);
    free(got);

    // 7. the host access calls on Vulkan
    AcContextDesc vd = { AC_ABI_VERSION, (uint32_t)sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcContext* vctx = NULL;
    if (ac_runtime_context_create(&vd, &vctx) == AC_SUCCESS && vctx) {
        host_access(vctx, "vulkan");
        AcKernel* none = NULL;
        CHECK(make_kernel(vctx, kAxpy, &m2, &none) == AC_ERR_FEATURE_NOT_SUPPORTED, "OpenCL C refused on Vulkan");
        CHECK(ac_runtime_context_destroy(vctx) == AC_SUCCESS, "vulkan teardown");
        printf("vulkan: host access checks passed\n");
    } else {
        printf("vulkan: no device, host access checked on OpenCL only\n");
    }
    printf("AC_CL_KAT=PASS (%d checks)\n", g_checks);
    return 0;
}
