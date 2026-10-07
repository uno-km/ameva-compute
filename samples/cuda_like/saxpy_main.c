/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
#include "amu_sample_util.h"
#include "saxpy.acu.h"

/* y is updated in place: every timed launch first restores y0 so the checked result is exactly one saxpy.
 * The restore is a host->mapped-memory copy and is part of wall_ms (not gpu_ms). */
static amuStatus saxpy_reset_and_launch(amuFunction* fn, amuDim3 grid, amuDim3 block, float alpha, amuBuffer* x,
                                        amuBuffer* y, const float* y0, uint32_t n) {
    amuStatus s = amuMemcpyHtoD(y, 0, y0, (size_t)n * sizeof(float));
    if (s != AMU_SUCCESS) return s;
    return amu_launch_saxpy(fn, grid, block, alpha, x, y, n);
}

int main(int argc, char** argv) {
    amu_sample_begin("saxpy");
    int narg = argc > 1 ? atoi(argv[1]) : (1 << 22);
    if (narg <= 0) amu_sample_die("BAD_ARGUMENT", "n must be > 0");
    const uint32_t n = (uint32_t)narg;
    const size_t bytes = (size_t)n * sizeof(float);
    const float alpha = 1.75f;

    float* hx = (float*)amu_host_alloc(bytes);
    float* y0 = (float*)amu_host_alloc(bytes);
    float* hy = (float*)amu_host_alloc(bytes);
    float* ref = (float*)amu_host_alloc(bytes);
    uint32_t seed = 777u;
    for (uint32_t i = 0; i < n; ++i) {
        hx[i] = amu_rand_unit(&seed) * 2.0f - 1.0f;
        y0[i] = amu_rand_unit(&seed);
    }

    amuContext* ctx;
    AMU_CHECK(amuInitFromEnv(&ctx));
    amu_print_header(ctx, "saxpy");

    amuBuffer *x, *y;
    AMU_CHECK(amuMalloc(ctx, bytes, &x));
    AMU_CHECK(amuMalloc(ctx, bytes, &y));
    AMU_CHECK(amuMemcpyHtoD(x, 0, hx, bytes));

    amuFunction* fn;
    AMU_CHECK(amu_load_saxpy(ctx, &fn));
    const uint32_t block = 256;
    amuDim3 grid = amuDim((n + block - 1) / block, 1, 1);
    amu_timing t;
    AMU_TIME_LAUNCHES(ctx, 10, t, saxpy_reset_and_launch(fn, grid, amuDim(block, 1, 1), alpha, x, y, y0, n));

    double t0 = amu_now_ms();
    AMU_CHECK(amuMemcpyDtoH(hy, y, 0, bytes));
    double d2h_ms = amu_now_ms() - t0;

    t0 = amu_now_ms();
    for (uint32_t i = 0; i < n; ++i) ref[i] = fmaf(alpha, hx[i], y0[i]);
    double cpu_ms = amu_now_ms() - t0;

    double max_rel = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double e = fabs((double)hy[i] - (double)ref[i]) / (1.0 + fabs((double)ref[i]));
        if (e > max_rel) max_rel = e;
    }
    char extra[96];
    snprintf(extra, sizeof(extra), "n=%u alpha=%.2f d2h_ms=%.3f wall_includes=y_reset", n, alpha, d2h_ms);
    int pass = max_rel <= 1e-6;
    amu_record_result(ctx, "saxpy", pass, max_rel, "max_rel_err", &t, cpu_ms, extra);

    AMU_CHECK(amuModuleUnload(fn));
    AMU_CHECK(amuFree(x)); AMU_CHECK(amuFree(y));
    AMU_CHECK(amuDestroy(ctx));
    free(hx); free(y0); free(hy); free(ref);
    return amu_sample_end(pass);
}
