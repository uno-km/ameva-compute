// Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 OR Commercial-License
//
// disp_bench: cost of a dispatch and of a barrier between dispatches through ac_runtime (Vulkan or OpenCL), with a
// kernel that does almost nothing (one workgroup, one invocation increments one word), so that what is timed is the
// submission path, the dispatch and the barrier. Per batch size N: N dispatches with a barrier between every two
// (the graph engine's chain), and N dispatches without any barrier (EXPLICIT_BARRIERS, none inserted); each batch is
// submitted and waited for REPS times, wall time per batch (median), per dispatch = (median(N) - median(1)) / (N - 1).
//
//   disp_bench KERNEL.spv SHA256HEX [vulkan|opencl] [REPS]
//   KERNEL.spv: the tiny kernel (binding 0: one storage buffer, local size 64), e.g. from disp_tiny.comp
//
// Prints DISP_INFO, then DISP mode=barrier|free n=<N> median_us=<..> per_dispatch_us=<..>
#include <ac_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_d(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

#define CHECK(call)                                                                  \
    do {                                                                             \
        AcResult r_ = (call);                                                        \
        if (r_ != AC_SUCCESS) {                                                      \
            fprintf(stderr, "DISP_FAIL %s -> %d (line %d)\n", #call, r_, __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static double run_batch(AcContext* ctx, AcKernel* k, AcBuffer* buf, uint32_t n, int barriers) {
    const AcBatchDesc bd = { AC_ABI_VERSION, sizeof(AcBatchDesc), n, AC_BATCH_FLAG_EXPLICIT_BARRIERS };
    AcCommandBatch* batch = NULL;
    const double t0 = now_us();
    CHECK(ac_runtime_batch_begin_ex(ctx, &bd, &batch));
    const AcBufferBinding b = { buf, 0, 256 };
    const AcDim3 grid = { 1, 1, 1 };
    for (uint32_t i = 0; i < n; ++i) {
        if (i && barriers) CHECK(ac_runtime_batch_barrier(batch));
        CHECK(ac_runtime_batch_dispatch_bindings(batch, k, &b, 1, NULL, 0, grid));
    }
    AcEvent* ev = NULL;
    CHECK(ac_runtime_batch_submit(batch, &ev));
    CHECK(ac_runtime_event_wait(ev, 10000000000ull));
    const double t1 = now_us();
    CHECK(ac_runtime_event_destroy(ev));
    ac_runtime_batch_destroy(batch);
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: disp_bench KERNEL.spv SHA256HEX [vulkan|opencl] [REPS]\n");
        return 2;
    }
    const int opencl = argc > 3 && !strcmp(argv[3], "opencl");
    const int reps = argc > 4 ? atoi(argv[4]) : 15;
    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "DISP_FAIL cannot open %s\n", argv[1]);
        return 1;
    }
    static uint32_t spv[65536];
    const size_t bytes = fread(spv, 1, sizeof(spv), f);
    fclose(f);
    if (opencl) {
        fprintf(stderr, "DISP_FAIL opencl needs the kernel as OpenCL C; this build times Vulkan only\n");
        return 1;
    }
    const AcContextDesc cd = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcContext* ctx = NULL;
    CHECK(ac_runtime_context_create(&cd, &ctx));
    const AcRuntimeBindingDesc bind = { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE };
    const AcKernelRuntimeMetadata meta = { "disp_tiny", "main", 1, &bind, 0, { 64, 1, 1 }, 0, argv[2], 0, NULL };
    const AcKernelDesc kd = { AC_ABI_VERSION, sizeof(AcKernelDesc), spv, bytes, &meta, 0, NULL };
    AcKernel* k = NULL;
    CHECK(ac_runtime_kernel_create(ctx, &kd, &k));
    const AcBufferDesc bfd = { AC_ABI_VERSION, sizeof(AcBufferDesc), 4096, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBuffer* buf = NULL;
    CHECK(ac_runtime_buffer_create(ctx, &bfd, &buf));
    printf("DISP_INFO backend=vulkan reps=%d\n", reps);
    static const uint32_t ns[] = { 1, 50, 200, 500 };
    double* t = (double*)malloc(sizeof(double) * (size_t)reps);
    for (int mode = 0; mode < 2; ++mode) {
        double med1 = 0.0;
        for (size_t ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ++ni) {
            run_batch(ctx, k, buf, ns[ni], !mode); // warm-up
            for (int r = 0; r < reps; ++r) t[r] = run_batch(ctx, k, buf, ns[ni], !mode);
            qsort(t, (size_t)reps, sizeof(double), cmp_d);
            const double med = t[reps / 2];
            if (ns[ni] == 1) med1 = med;
            printf("DISP mode=%s n=%u median_us=%.1f per_dispatch_us=%.2f\n", mode ? "free" : "barrier", ns[ni], med,
                   ns[ni] > 1 ? (med - med1) / (ns[ni] - 1) : 0.0);
        }
    }
    free(t);
    ac_runtime_buffer_destroy(buf);
    ac_runtime_kernel_destroy(k);
    ac_runtime_context_destroy(ctx);
    return 0;
}
