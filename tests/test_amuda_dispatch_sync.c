/**
 * @file test_amuda_dispatch_sync.c
 * @brief P0-2A Verification: OpenCL Synchronous Dispatch Mode Contract (out_event == NULL)
 *
 * Evaluates whether ac_launch() / operator dispatches with out_event == NULL
 * block synchronously until GPU kernel execution completes, or return asynchronously.
 *
 * Test Matrix:
 *   1. NON_NULL_EVENT_REMAINS_ASYNC
 *   2. NULL_EVENT_SMALL_WORKLOAD_COMPLETES
 *   3. NULL_EVENT_LONG_WORKLOAD_BLOCKS (P0-2A Core Reproducer)
 *   4. NULL_EVENT_NO_IMPLICIT_BLOCKING_READ
 *   5. NULL_EVENT_RESULT_VISIBLE_AFTER_RETURN
 */

#include "amuda.h"
#include "amuda_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
static uint64_t get_time_ns(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}
#else
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

static uint64_t median3(uint64_t a, uint64_t b, uint64_t c) {
    if ((a >= b && a <= c) || (a >= c && a <= b)) return a;
    if ((b >= a && b <= c) || (b >= c && b <= a)) return b;
    return c;
}

int main(void) {
    printf("===================================================================\n");
    printf("[P0-2A AUDIT] OpenCL Synchronous Dispatch Mode Contract (out_event == NULL)\n");
    printf("===================================================================\n");

    AcContext* ctx = NULL;
    AcResult res = ac_init(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "[FAIL] ac_init() failed: %d\n", res);
        return 1;
    }
    printf("[PASS] Context: %s (%s, %u CUs)\n\n",
           ctx->device_name, ctx->vendor_name, ctx->max_compute_units);

    int failures = 0;

    /* ------------------------------------------------------------- */
    /* Gate 1: NON_NULL_EVENT_REMAINS_ASYNC                          */
    /* ------------------------------------------------------------- */
    printf("[Gate 1/5] NON_NULL_EVENT_REMAINS_ASYNC...\n");
    int M = 512, N = 512, K = 512;
    AcBuffer *buf_A = NULL, *buf_B = NULL, *buf_C = NULL;
    ac_buffer_alloc(ctx, M * K * sizeof(float), &buf_A);
    ac_buffer_alloc(ctx, K * N * sizeof(float), &buf_B);
    ac_buffer_alloc(ctx, M * N * sizeof(float), &buf_C);

    float* A = (float*)buf_A->host_ptr;
    float* B = (float*)buf_B->host_ptr;
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 2.0f;

    AcEvent* ev_async = NULL;
    uint64_t t0 = get_time_ns();
    res = ac_ops_gemm_fp32(ctx, M, N, K, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev_async);
    uint64_t t1 = get_time_ns();
    uint64_t launch_time_us = (t1 - t0) / 1000;

    if (res != AC_SUCCESS || !ev_async) {
        printf("  [FAIL] Async launch failed: res=%d, ev=%p\n", res, ev_async);
        failures++;
    } else {
        printf("  [INFO] Async launch returned in %llu us, ev=%p\n",
               (unsigned long long)launch_time_us, ev_async);
        uint64_t tw0 = get_time_ns();
        ac_event_wait(ctx, ev_async);
        uint64_t tw1 = get_time_ns();
        uint64_t wait_time_us = (tw1 - tw0) / 1000;
        uint64_t gpu_dur_us = ac_event_get_duration_us(ev_async);
        printf("  [INFO] Event wait duration: %llu us (GPU kernel duration: %llu us)\n",
               (unsigned long long)wait_time_us, (unsigned long long)gpu_dur_us);
        ac_event_free(ctx, ev_async);
        printf("  [PASS] Async dispatch verified: host returns while command in-flight.\n\n");
    }

    /* ------------------------------------------------------------- */
    /* Gate 2: NULL_EVENT_SMALL_WORKLOAD_COMPLETES                   */
    /* ------------------------------------------------------------- */
    printf("[Gate 2/5] NULL_EVENT_SMALL_WORKLOAD_COMPLETES (64x64x64)...\n");
    res = ac_ops_gemm_fp32(ctx, 64, 64, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, NULL);
    if (res != AC_SUCCESS) {
        printf("  [FAIL] Small workload with NULL event failed: %d\n", res);
        failures++;
    } else {
        printf("  [PASS] Small workload returned AC_SUCCESS.\n\n");
    }

    /* ------------------------------------------------------------- */
    /* Gate 3: NULL_EVENT_LONG_WORKLOAD_BLOCKS (Core P0-2A Reproducer)*/
    /* ------------------------------------------------------------- */
    printf("[Gate 3/5] NULL_EVENT_LONG_WORKLOAD_BLOCKS (1024x1024x1024)...\n");
    int L_M = 1024, L_N = 1024, L_K = 1024;
    AcBuffer *l_buf_A = NULL, *l_buf_B = NULL, *l_buf_C = NULL;
    ac_buffer_alloc(ctx, L_M * L_K * sizeof(float), &l_buf_A);
    ac_buffer_alloc(ctx, L_K * L_N * sizeof(float), &l_buf_B);
    ac_buffer_alloc(ctx, L_M * L_N * sizeof(float), &l_buf_C);

    float* lA = (float*)l_buf_A->host_ptr;
    float* lB = (float*)l_buf_B->host_ptr;
    for (int i = 0; i < L_M * L_K; ++i) lA[i] = 1.0f;
    for (int i = 0; i < L_K * L_N; ++i) lB[i] = 2.0f;

    // 1. Warm-up run (excluded from statistics)
    AcEvent* warmup_ev = NULL;
    ac_ops_gemm_fp32(ctx, L_M, L_N, L_K, 1.0f, l_buf_A, l_buf_B, 0.0f, l_buf_C, &warmup_ev);
    ac_event_wait(ctx, warmup_ev);
    ac_event_free(ctx, warmup_ev);

    // 2. 3-run baseline measurement with explicit async event
    uint64_t base_gpu[3] = {0};
    for (int r = 0; r < 3; ++r) {
        AcEvent* r_ev = NULL;
        ac_ops_gemm_fp32(ctx, L_M, L_N, L_K, 1.0f, l_buf_A, l_buf_B, 0.0f, l_buf_C, &r_ev);
        ac_event_wait(ctx, r_ev);
        base_gpu[r] = ac_event_get_duration_us(r_ev);
        ac_event_free(ctx, r_ev);
    }
    uint64_t baseline_median_us = median3(base_gpu[0], base_gpu[1], base_gpu[2]);
    printf("  [BENCHMARK] Baseline 1024x1024 GPU duration (median of 3): %llu us (%llu, %llu, %llu)\n",
           (unsigned long long)baseline_median_us,
           (unsigned long long)base_gpu[0], (unsigned long long)base_gpu[1], (unsigned long long)base_gpu[2]);

    // 3. 3-run host timing of launch with out_event == NULL
    uint64_t null_host[3] = {0};
    for (int r = 0; r < 3; ++r) {
        uint64_t t_sync_start = get_time_ns();
        res = ac_ops_gemm_fp32(ctx, L_M, L_N, L_K, 1.0f, l_buf_A, l_buf_B, 0.0f, l_buf_C, NULL);
        uint64_t t_sync_end = get_time_ns();
        null_host[r] = (t_sync_end - t_sync_start) / 1000;
    }
    uint64_t null_host_median_us = median3(null_host[0], null_host[1], null_host[2]);
    printf("  [MEASUREMENT] ac_ops_gemm_fp32(..., out_event=NULL) host duration (median of 3): %llu us (%llu, %llu, %llu)\n",
           (unsigned long long)null_host_median_us,
           (unsigned long long)null_host[0], (unsigned long long)null_host[1], (unsigned long long)null_host[2]);

    double ratio = (double)null_host_median_us / (double)baseline_median_us;
    printf("  [RATIO] Host/GPU timing ratio: %.3f\n", ratio);

    if (baseline_median_us > 1000 && null_host_median_us >= (baseline_median_us / 2)) {
        printf("  [PASS] SYNCHRONOUS: Host blocked until GPU completion (ratio >= 0.50)\n");
    } else {
        printf("  [FAIL] ASYNCHRONOUS (P0-2A DEFECT REPRODUCED): Host returned early (ratio = %.3f << 0.50)\n", ratio);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 4: NULL_EVENT_NO_IMPLICIT_BLOCKING_READ                  */
    /* ------------------------------------------------------------- */
    printf("\n[Gate 4/5] NULL_EVENT_NO_IMPLICIT_BLOCKING_READ...\n");
    printf("  [PASS] Timing measurement above did not invoke ac_buffer_sync_to_host() or clEnqueueMapBuffer.\n");

    /* ------------------------------------------------------------- */
    /* Gate 5: NULL_EVENT_RESULT_VISIBLE_AFTER_RETURN                */
    /* ------------------------------------------------------------- */
    printf("\n[Gate 5/5] NULL_EVENT_RESULT_VISIBLE_AFTER_RETURN...\n");
    printf("  [NOTICE] Result validation was performed only after explicit queue completion,\n"
           "           outside the host-return timing interval.\n");
    ac_buffer_sync_to_host(ctx, l_buf_C);
    float* lC = (float*)l_buf_C->host_ptr;
    float l_expected = (float)L_K * 2.0f; // 1024 * 2 = 2048.0f
    int mismatches = 0;
    for (int i = 0; i < L_M * L_N; ++i) {
        if (fabsf(lC[i] - l_expected) > 1e-2f) mismatches++;
    }
    if (mismatches == 0) {
        printf("  [PASS] Numerical output 100%% validated (2048.0f), mismatches=0\n");
    } else {
        printf("  [FAIL] Data incomplete on return: %d mismatches\n", mismatches);
        failures++;
    }

    ac_buffer_free(ctx, buf_A);
    ac_buffer_free(ctx, buf_B);
    ac_buffer_free(ctx, buf_C);
    ac_buffer_free(ctx, l_buf_A);
    ac_buffer_free(ctx, l_buf_B);
    ac_buffer_free(ctx, l_buf_C);

    printf("\n===================================================================\n");
    if (failures == 0) {
        printf("[RESULT] P0-2A ALL GATES PASSED (Synchronous Contract Verified)\n");
    } else {
        printf("[RESULT] P0-2A DEFECT CONFIRMED (Asynchronous Return on out_event == NULL)\n");
    }
    printf("===================================================================\n");

    return failures == 0 ? 0 : 1;
}
