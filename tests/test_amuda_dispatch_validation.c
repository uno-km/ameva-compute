/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_amuda_dispatch_validation.c
 * @brief P0-2B Comprehensive Verification: Dispatch Input Validation, Overflow Disambiguation & Fail-Closed Contract
 *
 * Test Matrix (28 Distinct Verification Gates):
 *   1-3:   GEMM non-positive M, N, K rejection (AC_ERR_KERNEL_ARG_INVALID)
 *   4-6:   ac_launch zero grid.x, grid.y, grid.z rejection (AC_ERR_INVALID_WORK_ITEM_DIMS)
 *   7:     ac_launch fail-closed out_event clearance on invalid context
 *   8:     ELEMENT_COUNT_OVERFLOW (OVERFLOW_SITE=checked_mul -> AC_ERR_NUMERICAL_OVERFLOW)
 *   9:     GRID_TIMES_BLOCK_OVERFLOW (OVERFLOW_SITE=global_size -> AC_ERR_NUMERICAL_OVERFLOW)
 *   10:    BUFFER_TOO_SMALL_WITHOUT_OVERFLOW (OVERFLOW_SITE=buffer_required_bytes -> AC_ERR_INVALID_BUFFER_SIZE)
 *   11:    DISPATCH_BUDGET_EXCEEDED_WITHOUT_OVERFLOW (OVERFLOW_SITE=dispatch_budget -> AC_ERR_INVALID_WORK_GROUP_SIZE)
 *   12-14: GEMM rejects undersized buffers A, B, C (AC_ERR_INVALID_BUFFER_SIZE)
 *   15-17: BitNet I2 invalid dimension, fail-closed out_event, and undersized packed weight
 *   18-19: BitNet I2-S Exact invalid dimension and fail-closed out_event
 *   20-22: RMSNorm invalid dimension, fail-closed out_event, and undersized input buffer
 *   23-25: SwiGLU invalid dimension, fail-closed out_event, and undersized input buffer
 *   26-28: RoPE invalid dimension, fail-closed out_event, and undersized input buffer
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
    printf("[P0-2B AUDIT] Comprehensive Input Validation & Fail-Closed Contract (28 Gates)\n");
    printf("===================================================================\n");

    AcContext* ctx = NULL;
    AcResult res = ac_init(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "[FAIL] ac_init() failed: %d\n", res);
        return 1;
    }
    printf("[PASS] Context: %s (%s, %u CUs, max_wg=%llu)\n\n",
           ctx->device_name, ctx->vendor_name, ctx->max_compute_units,
           (unsigned long long)ctx->max_work_group_size);

    int failures = 0;

    AcBuffer *buf_A = NULL, *buf_B = NULL, *buf_C = NULL;
    AcBuffer *buf_small = NULL, *buf_w_packed = NULL, *buf_scales = NULL;
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_A);
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_B);
    ac_buffer_alloc(ctx, 64 * 64 * sizeof(float), &buf_C);
    ac_buffer_alloc(ctx, 16, &buf_small); // Intentionally tiny: 16 bytes
    ac_buffer_alloc(ctx, 64 * 16, &buf_w_packed);
    ac_buffer_alloc(ctx, 64 * sizeof(float), &buf_scales);

    AcKernel* valid_kernel = NULL;
    res = ac_kernel_load_cached(ctx, k_noop_src, "noop", &valid_kernel);
    if (res != AC_SUCCESS || !valid_kernel) {
        fprintf(stderr, "[FAIL] Failed to compile test kernel: %d\n", res);
        return 1;
    }

    void* k_args[1] = { &buf_A->native_mem };
    size_t k_arg_sizes[1] = { sizeof(void*) };

    /* --- [1-3] GEMM Non-Positive Dimensions --- */
    printf("[Gate 1/28] GEMM_REJECTS_NONPOSITIVE_M...\n");
    AcEvent* ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, -1, 64, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) {
        printf("  [PASS] Rejected non-positive M (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 2/28] GEMM_REJECTS_NONPOSITIVE_N...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 0, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) {
        printf("  [PASS] Rejected non-positive N (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 3/28] GEMM_REJECTS_NONPOSITIVE_K...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 64, -10, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) {
        printf("  [PASS] Rejected non-positive K (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    /* --- [4-6] ac_launch Zero Grid Axes --- */
    printf("[Gate 4/28] LAUNCH_REJECTS_ZERO_GRID_X...\n");
    ev = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(0, 1, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_INVALID_WORK_ITEM_DIMS && ev == NULL) {
        printf("  [PASS] Rejected zero grid.x (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 5/28] LAUNCH_REJECTS_ZERO_GRID_Y...\n");
    ev = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(1, 0, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_INVALID_WORK_ITEM_DIMS && ev == NULL) {
        printf("  [PASS] Rejected zero grid.y (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 6/28] LAUNCH_REJECTS_ZERO_GRID_Z...\n");
    ev = SENTINEL_EVENT;
    res = ac_launch(ctx, valid_kernel, ac_dim3(1, 1, 0), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_INVALID_WORK_ITEM_DIMS && ev == NULL) {
        printf("  [PASS] Rejected zero grid.z (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    /* --- [7] ac_launch Fail-Closed out_event --- */
    printf("[Gate 7/28] LAUNCH_CLEARS_OUT_EVENT_ON_FAILURE...\n");
    ev = SENTINEL_EVENT;
    res = ac_launch(NULL, valid_kernel, ac_dim3(1, 1, 1), ac_dim3(16, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_INVALID_CONTEXT && ev == NULL) {
        printf("  [PASS] Cleared out_event to NULL on failure (res=%d)\n", res);
    } else {
        printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++;
    }

    /* --- [8-11] Overflow Disambiguation vs Undersized Buffer & Dispatch Budget --- */
    printf("[Gate 8/28] ELEMENT_COUNT_OVERFLOW (OVERFLOW_SITE=checked_mul)...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 0x7FFFFFFF, 0x7FFFFFFF, 64, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_NUMERICAL_OVERFLOW && ev == NULL) {
        printf("  [PASS] OVERFLOW_SITE=checked_mul rejected (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Expected -502, got res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 9/28] GRID_TIMES_BLOCK_OVERFLOW (OVERFLOW_SITE=global_size)...\n");
    ev = SENTINEL_EVENT;
    // grid.x * block.x = 0x80000000 * 2 = 0x100000000ULL (> UINT32_MAX)
    res = ac_launch(ctx, valid_kernel, ac_dim3(0x80000000, 1, 1), ac_dim3(2, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_NUMERICAL_OVERFLOW && ev == NULL) {
        printf("  [PASS] OVERFLOW_SITE=global_size rejected (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Expected -502, got res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 10/28] BUFFER_TOO_SMALL_WITHOUT_OVERFLOW (OVERFLOW_SITE=buffer_required_bytes)...\n");
    ev = SENTINEL_EVENT;
    // Normal small dimensions (64x64), but buf_small is 16 bytes (< 16384 bytes)
    res = ac_ops_gemm_fp32(ctx, 64, 64, 64, 1.0f, buf_small, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) {
        printf("  [PASS] OVERFLOW_SITE=buffer_required_bytes correctly identified (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Expected -206, got res=%d, ev=%p\n", res, ev); failures++;
    }

    printf("[Gate 11/28] DISPATCH_BUDGET_EXCEEDED_WITHOUT_OVERFLOW (OVERFLOW_SITE=dispatch_budget)...\n");
    ev = SENTINEL_EVENT;
    // block.x = 2048 exceeds max_work_group_size without arithmetic overflow
    res = ac_launch(ctx, valid_kernel, ac_dim3(1, 1, 1), ac_dim3(2048, 1, 1), 1, k_args, k_arg_sizes, &ev);
    if (res == AC_ERR_INVALID_WORK_GROUP_SIZE && ev == NULL) {
        printf("  [PASS] OVERFLOW_SITE=dispatch_budget correctly identified (res=%d, ev=NULL)\n", res);
    } else {
        printf("  [FAIL] Expected -401, got res=%d, ev=%p\n", res, ev); failures++;
    }

    /* --- [12-14] GEMM Undersized Buffers --- */
    printf("[Gate 12/28] GEMM_REJECTS_UNDERSIZED_A...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 64, 64, 1.0f, buf_small, buf_B, 0.0f, buf_C, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized A rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 13/28] GEMM_REJECTS_UNDERSIZED_B...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 64, 64, 1.0f, buf_A, buf_small, 0.0f, buf_C, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized B rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 14/28] GEMM_REJECTS_UNDERSIZED_C...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_gemm_fp32(ctx, 64, 64, 64, 1.0f, buf_A, buf_B, 0.0f, buf_small, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized C rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    /* --- [15-17] BitNet I2 --- */
    printf("[Gate 15/28] BITNET_I2_REJECTS_INVALID_DIMENSIONS...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_bitnet_gemv_i2(ctx, -1, 64, buf_A, buf_w_packed, buf_scales, buf_B, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] Invalid dimension rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 16/28] BITNET_I2_CLEARS_OUT_EVENT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_bitnet_gemv_i2(ctx, 64, 65, buf_A, buf_w_packed, buf_scales, buf_B, &ev); // K%4 != 0
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] out_event cleared on failure\n");
    else { printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++; }

    printf("[Gate 17/28] BITNET_REJECTS_UNDERSIZED_PACKED_WEIGHT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_bitnet_gemv_i2(ctx, 64, 64, buf_A, buf_small, buf_scales, buf_B, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized weight rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    /* --- [18-19] BitNet I2-S Exact --- */
    printf("[Gate 18/28] BITNET_I2S_EXACT_REJECTS_INVALID_DIMENSIONS...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_bitnet_gemv_i2_s_exact(ctx, 64, 64, buf_A, buf_w_packed, 1.0f, buf_B, &ev); // K%128 != 0
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] Invalid dimension rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 19/28] BITNET_I2S_EXACT_CLEARS_OUT_EVENT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_bitnet_gemv_i2_s_exact(ctx, -1, 128, buf_A, buf_w_packed, 1.0f, buf_B, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] out_event cleared on failure\n");
    else { printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++; }

    /* --- [20-22] RMSNorm --- */
    printf("[Gate 20/28] RMSNORM_REJECTS_INVALID_DIMENSIONS...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rmsnorm(ctx, -1, 64, 1e-5f, buf_A, buf_scales, buf_B, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] Invalid dimension rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 21/28] RMSNORM_CLEARS_OUT_EVENT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rmsnorm(ctx, 0, 64, 1e-5f, buf_A, buf_scales, buf_B, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] out_event cleared on failure\n");
    else { printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++; }

    printf("[Gate 22/28] RMSNORM_REJECTS_UNDERSIZED_INPUT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rmsnorm(ctx, 64, 64, 1e-5f, buf_small, buf_scales, buf_B, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized input rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    /* --- [23-25] SwiGLU --- */
    printf("[Gate 23/28] SWIGLU_REJECTS_INVALID_DIMENSIONS...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_swiglu(ctx, -10, buf_A, buf_B, buf_C, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] Invalid dimension rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 24/28] SWIGLU_CLEARS_OUT_EVENT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_swiglu(ctx, 0, buf_A, buf_B, buf_C, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] out_event cleared on failure\n");
    else { printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++; }

    printf("[Gate 25/28] SWIGLU_REJECTS_UNDERSIZED_INPUT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_swiglu(ctx, 64 * 64, buf_small, buf_B, buf_C, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized input rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    /* --- [26-28] RoPE --- */
    printf("[Gate 26/28] ROPE_REJECTS_INVALID_DIMENSIONS...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rope(ctx, -1, 4, 16, buf_A, buf_B, &ev);
    if (res == AC_ERR_KERNEL_ARG_INVALID && ev == NULL) printf("  [PASS] Invalid dimension rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    printf("[Gate 27/28] ROPE_CLEARS_OUT_EVENT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rope(ctx, 16, 4, 15, buf_A, buf_B, &ev); // head_dim % 2 != 0
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] out_event cleared on failure\n");
    else { printf("  [FAIL] res=%d, ev=%p\n", res, ev); failures++; }

    printf("[Gate 28/28] ROPE_REJECTS_UNDERSIZED_INPUT...\n");
    ev = SENTINEL_EVENT;
    res = ac_ops_rope(ctx, 16, 4, 16, buf_small, buf_B, &ev);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && ev == NULL) printf("  [PASS] Undersized input rejected\n");
    else { printf("  [FAIL] res=%d\n", res); failures++; }

    ac_kernel_free(ctx, valid_kernel);
    ac_buffer_free(ctx, buf_A);
    ac_buffer_free(ctx, buf_B);
    ac_buffer_free(ctx, buf_C);
    ac_buffer_free(ctx, buf_small);
    ac_buffer_free(ctx, buf_w_packed);
    ac_buffer_free(ctx, buf_scales);

    printf("\n===================================================================\n");
    if (failures == 0) {
        printf("[RESULT] P0-2B ALL 28 VALIDATION GATES PASSED (100%% COMPLIANCE)\n");
    } else {
        printf("[RESULT] P0-2B DEFECTS DETECTED (%d failures)\n", failures);
    }
    printf("===================================================================\n");

    return failures == 0 ? 0 : 1;
}
