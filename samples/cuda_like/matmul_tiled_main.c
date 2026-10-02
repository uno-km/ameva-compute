/* SPDX-License-Identifier: Apache-2.0 */
#include "amu_sample_util.h"
#include "matmul_tiled.acu.h"

int main(int argc, char** argv) {
    amu_sample_begin("matmul_tiled");
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : M;
    int K = argc > 3 ? atoi(argv[3]) : M;
    if (M <= 0 || N <= 0 || K <= 0) amu_sample_die("BAD_ARGUMENT", "M, N, K must be > 0");
    const uint32_t TILE = 16; /* must match #define TILE in matmul_tiled.acu */
    const size_t bA = (size_t)M * (size_t)K * sizeof(float);
    const size_t bB = (size_t)K * (size_t)N * sizeof(float);
    const size_t bC = (size_t)M * (size_t)N * sizeof(float);

    float* hA = (float*)amu_host_alloc(bA);
    float* hB = (float*)amu_host_alloc(bB);
    float* hC = (float*)amu_host_alloc(bC);
    float* ref = (float*)amu_host_alloc(bC);
    uint32_t seed = 4242u;
    for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = amu_rand_unit(&seed) - 0.5f;
    for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = amu_rand_unit(&seed) - 0.5f;

    amuContext* ctx;
    AMU_CHECK(amuInitFromEnv(&ctx));
    amu_print_header(ctx, "matmul_tiled");

    amuBuffer *A, *B, *C;
    AMU_CHECK(amuMalloc(ctx, bA, &A));
    AMU_CHECK(amuMalloc(ctx, bB, &B));
    AMU_CHECK(amuMalloc(ctx, bC, &C));
    AMU_CHECK(amuMemcpyHtoD(A, 0, hA, bA));
    AMU_CHECK(amuMemcpyHtoD(B, 0, hB, bB));

    amuFunction* fn;
    AMU_CHECK(amu_load_matmul_tiled(ctx, &fn));
    amuDim3 grid = amuDim(((uint32_t)N + TILE - 1) / TILE, ((uint32_t)M + TILE - 1) / TILE, 1);
    amu_timing t;
    AMU_TIME_LAUNCHES(ctx, 5, t, amu_launch_matmul_tiled(fn, grid, amuDim(TILE, TILE, 1), A, B, C, M, N, K));

    double t0 = amu_now_ms();
    AMU_CHECK(amuMemcpyDtoH(hC, C, 0, bC));
    double d2h_ms = amu_now_ms() - t0;

    t0 = amu_now_ms();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += hA[(size_t)i * K + k] * hB[(size_t)k * N + j];
            ref[(size_t)i * N + j] = acc;
        }
    double cpu_ms = amu_now_ms() - t0;

    double max_rel = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        double e = fabs((double)hC[i] - (double)ref[i]) / (1.0 + fabs((double)ref[i]));
        if (e > max_rel) max_rel = e;
    }
    double flops = 2.0 * M * N * K;
    char extra[128];
    snprintf(extra, sizeof(extra), "M=%d N=%d K=%d gpu_gflops=%.2f d2h_ms=%.3f", M, N, K,
             t.gpu_med_ms > 0 ? flops / (t.gpu_med_ms * 1e6) : -1.0, d2h_ms);
    int pass = max_rel <= 1e-4;
    amu_record_result(ctx, "matmul_tiled", pass, max_rel, "max_rel_err", &t, cpu_ms, extra);

    AMU_CHECK(amuModuleUnload(fn));
    AMU_CHECK(amuFree(A)); AMU_CHECK(amuFree(B)); AMU_CHECK(amuFree(C));
    AMU_CHECK(amuDestroy(ctx));
    free(hA); free(hB); free(hC); free(ref);
    return amu_sample_end(pass);
}
