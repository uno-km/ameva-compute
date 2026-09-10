/**
 * @file test_amuda_dispatch_validation.c
 * @brief P0-2B Verification: Dispatch Input Validation & Output Event Fail-Closed Contract
 *
 * Evaluates whether operator APIs and ac_launch() properly reject non-positive dimensions,
 * zero-sized grid components, arithmetic overflow, and enforce *out_event = NULL on failure.
 *
 * Test Matrix:
 *   1. GEMM_REJECTS_NONPOSITIVE_M
 *   2. GEMM_REJECTS_NONPOSITIVE_N
 *   3. GEMM_REJECTS_NONPOSITIVE_K
 *   4. LAUNCH_REJECTS_ZERO_GRID_X
 *   5. LAUNCH_REJECTS_ZERO_GRID_Y
 *   6. LAUNCH_REJECTS_ZERO_GRID_Z
 *   7. LAUNCH_CLEARS_OUT_EVENT_ON_FAILURE
 *   8. GRID_CALCULATION_REJECTS_OVERFLOW
 */

#include "amuda.h"
#include "amuda_ops.h"
#include "amuda_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SENTINEL_EVENT ((AcEvent*)(uintptr_t)0xDEADBEEF)

static const char* k_noop_src =
"__kernel void noop(__global float* x) {\n"
"    int id = get_global_id(0);\n"
"    x[id] = x[id] + 1.0f;\n"
"}\n";

int main(void) {
    printf("===================================================================\n");
    printf("[P0-2B AUDIT] Dispatch Input Validation & Fail-Closed Event Contract\n");
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

    AcBuffer *buf_A = NULL, *buf_B = NULL, *buf_C = NULL;
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_A);
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_B);
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_C);

    AcKernel* valid_kernel = NULL;
    res = ac_kernel_load_cached(ctx, k_noop_src, "noop", &valid_kernel);
    if (res != AC_SUCCESS || !valid_kernel) {
        fprintf(stderr, "[FAIL] Failed to compile test kernel: %d\n", res);
        return 1;
    }

    void* k_args[1] = { &buf_A->native_mem };
    size_t k_arg_sizes[1] = { sizeof(void*) };

    /* ------------------------------------------------------------- */
    /* Gate 1: GEMM_REJECTS_NONPOSITIVE_M                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 1/8] GEMM_REJECTS_NONPOSITIVE_M...\n");
    AcEvent* ev1 = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, -1, 64, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev1);
    if (res != AC_SUCCESS && ev1 == NULL) {
        printf("  [PASS] Rejected non-positive M (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Failed to reject non-positive M (res=%d, ev=%p)\n", res, ev1);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 2: GEMM_REJECTS_NONPOSITIVE_N                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 2/8] GEMM_REJECTS_NONPOSITIVE_N...\n");
    AcEvent* ev2 = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 0, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev2);
    if (res != AC_SUCCESS && ev2 == NULL) {
        printf("  [PASS] Rejected non-positive N (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Failed to reject non-positive N (res=%d, ev=%p)\n", res, ev2);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 3: GEMM_REJECTS_NONPOSITIVE_K                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 3/8] GEMM_REJECTS_NONPOSITIVE_K...\n");
    AcEvent* ev3 = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 64, -10, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev3);
    if (res != AC_SUCCESS && ev3 == NULL) {
        printf("  [PASS] Rejected non-positive K (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Failed to reject non-positive K (res=%d, ev=%p)\n", res, ev3);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 4: LAUNCH_REJECTS_ZERO_GRID_X                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 4/8] LAUNCH_REJECTS_ZERO_GRID_X...\n");
    AcEvent* ev4 = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(0, 1, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev4);
    if (res != AC_SUCCESS && ev4 == NULL) {
        printf("  [PASS] ac_launch rejected zero grid.x (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] ac_launch accepted zero grid.x or failed to clear ev (res=%d, ev=%p)\n", res, ev4);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 5: LAUNCH_REJECTS_ZERO_GRID_Y                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 5/8] LAUNCH_REJECTS_ZERO_GRID_Y...\n");
    AcEvent* ev5 = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(1, 0, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev5);
    if (res != AC_SUCCESS && ev5 == NULL) {
        printf("  [PASS] ac_launch rejected zero grid.y (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] ac_launch accepted zero grid.y or failed to clear ev (res=%d, ev=%p)\n", res, ev5);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 6: LAUNCH_REJECTS_ZERO_GRID_Z                            */
    /* ------------------------------------------------------------- */
    printf("[Gate 6/8] LAUNCH_REJECTS_ZERO_GRID_Z...\n");
    AcEvent* ev6 = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(1, 1, 0), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev6);
    if (res != AC_SUCCESS && ev6 == NULL) {
        printf("  [PASS] ac_launch rejected zero grid.z (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] ac_launch accepted zero grid.z or failed to clear ev (res=%d, ev=%p)\n", res, ev6);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 7: LAUNCH_CLEARS_OUT_EVENT_ON_FAILURE                    */
    /* ------------------------------------------------------------- */
    printf("[Gate 7/8] LAUNCH_CLEARS_OUT_EVENT_ON_FAILURE...\n");
    AcEvent* ev7 = SENTINEL_EVENT;
    // Invalid context must fail-closed and clear out_event to NULL
    res = ac_launch(NULL, valid_kernel, ac_dim3(1, 1, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev7);
    if (res == AC_ERR_INVALID_CONTEXT && ev7 == NULL) {
        printf("  [PASS] Invalid context cleared out_event to NULL (res=%d)\n", res);
    } else {
        printf("  [FAIL] out_event was not cleared to NULL on failure: res=%d, ev=%p\n", res, ev7);
        failures++;
    }

    /* ------------------------------------------------------------- */
    /* Gate 8: GRID_CALCULATION_REJECTS_OVERFLOW                     */
    /* ------------------------------------------------------------- */
    printf("[Gate 8/8] GRID_CALCULATION_REJECTS_OVERFLOW...\n");
    AcEvent* ev8 = SENTINEL_EVENT;
    // Huge dimension that would overflow uint32 grid calculation
    res = ac_ops_gemm_fp32(ctx, 0x7FFFFFFF, 0x7FFFFFFF, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev8);
    if (res != AC_SUCCESS && ev8 == NULL) {
        printf("  [PASS] Overflowing dimension rejected fail-closed (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Overflowing dimension accepted or ev not cleared (res=%d, ev=%p)\n", res, ev8);
        failures++;
    }

    ac_kernel_free(ctx, valid_kernel);
    ac_buffer_free(ctx, buf_A);
    ac_buffer_free(ctx, buf_B);
    ac_buffer_free(ctx, buf_C);

    printf("\n===================================================================\n");
    if (failures == 0) {
        printf("[RESULT] P0-2B ALL VALIDATION GATES PASSED\n");
    } else {
        printf("[RESULT] P0-2B DEFECTS DETECTED (%d failures)\n", failures);
    }
    printf("===================================================================\n");

    return failures == 0 ? 0 : 1;
}
