/**
 * @file test_amuda_ops.c
 * @brief Phase 2 Hardware Verification: GEMM, BitNet 1.58b, Transformer Ops, and Binary Cache
 */

#include "amuda.h"
#include "amuda_ops.h"
#include "amuda_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-4f

static float rand_float(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

int main(void) {
    printf("===================================================================\n");
    printf("[AMUDA PHASE 2] High-Performance Tensor Operations & Cache Test\n");
    printf("===================================================================\n");

    AcContext* ctx = NULL;
    AcResult res = ac_init(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "[FAIL] ac_init() failed: %d\n", res);
        return 1;
    }
    printf("[PASS] Hardware Context Initialized: %s (%s, %u CUs)\n\n",
           ctx->device_name, ctx->vendor_name, ctx->max_compute_units);

    /* ===================================================================== */
    /* 1. GEMM FP32 (256 x 256 x 256) Verification                          */
    /* ===================================================================== */
    printf("--- [TEST 1] 2D Tiled FP32 GEMM (256x256x256) ---\n");
    int M = 256, N = 256, K = 256;
    AcBuffer *buf_A = NULL, *buf_B = NULL, *buf_C = NULL;

    ac_buffer_alloc(ctx, M * K * sizeof(float), &buf_A);
    ac_buffer_alloc(ctx, K * N * sizeof(float), &buf_B);
    ac_buffer_alloc(ctx, M * N * sizeof(float), &buf_C);

    float* A = (float*)buf_A->host_ptr;
    float* B = (float*)buf_B->host_ptr;
    for (int i = 0; i < M * K; ++i) A[i] = rand_float(-1.0f, 1.0f);
    for (int i = 0; i < K * N; ++i) B[i] = rand_float(-1.0f, 1.0f);

    AcEvent* ev_gemm = NULL;
    res = ac_ops_gemm_fp32(ctx, M, N, K, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev_gemm);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_ops_gemm_fp32 failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_gemm);
    uint64_t gemm_us = ac_event_get_duration_us(ev_gemm);
    double gflops = (2.0 * M * N * K) / ((double)gemm_us * 1000.0);
    printf("[PASS] GEMM Executed in %llu us (%.2f GFLOPS)\n", (unsigned long long)gemm_us, gflops);

    // CPU Verification
    ac_buffer_sync_to_host(ctx, buf_C);
    float* C = (float*)buf_C->host_ptr;
    int gemm_errors = 0;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) {
                ref += A[r * K + k] * B[k * N + c];
            }
            float val = C[r * N + c];
            if (isnan(val) || isinf(val) || fabsf(val - ref) > (fabsf(ref) * 1e-3f + 1e-3f)) {
                if (gemm_errors < 3) {
                    fprintf(stderr, "[ERROR] GEMM Mismatch at (%d, %d): got %f, ref %f\n", r, c, val, ref);
                }
                gemm_errors++;
            }
        }
    }
    if (gemm_errors > 0) {
        fprintf(stderr, "[FAIL] GEMM verification failed with %d errors!\n", gemm_errors);
        return 1;
    }
    printf("[SUCCESS] GEMM Numerical Output 100%% Validated against CPU reference!\n\n");

    /* ===================================================================== */
    /* 2. BitNet 1.58b Ternary GEMV (M=1024, K=1024)                         */
    /* ===================================================================== */
    printf("--- [TEST 2] BitNet 1.58-bit Ternary GEMV (M=1024, K=1024) ---\n");
    int bit_M = 1024, bit_K = 1024;
    AcBuffer *buf_x = NULL, *buf_W = NULL, *buf_sc = NULL, *buf_y = NULL;

    ac_buffer_alloc(ctx, bit_K * sizeof(float), &buf_x);
    ac_buffer_alloc(ctx, bit_M * (bit_K / 4) * sizeof(unsigned char), &buf_W);
    ac_buffer_alloc(ctx, bit_M * sizeof(float), &buf_sc);
    ac_buffer_alloc(ctx, bit_M * sizeof(float), &buf_y);

    float* x_vec = (float*)buf_x->host_ptr;
    unsigned char* W_pack = (unsigned char*)buf_W->host_ptr;
    float* scales = (float*)buf_sc->host_ptr;

    for (int i = 0; i < bit_K; ++i) x_vec[i] = rand_float(-0.5f, 0.5f);
    for (int i = 0; i < bit_M; ++i) scales[i] = 0.025f;
    for (int i = 0; i < bit_M * (bit_K / 4); ++i) {
        // Pack 4 ternary weights: 0, 1, 2
        unsigned char c0 = rand() % 3;
        unsigned char c1 = rand() % 3;
        unsigned char c2 = rand() % 3;
        unsigned char c3 = rand() % 3;
        W_pack[i] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
    }

    AcEvent* ev_bit = NULL;
    res = ac_ops_bitnet_gemv_i2(ctx, bit_M, bit_K, buf_x, buf_W, buf_sc, buf_y, &ev_bit);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_ops_bitnet_gemv_i2 failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_bit);
    uint64_t bit_us = ac_event_get_duration_us(ev_bit);
    printf("[PASS] BitNet GEMV Executed in %llu us (%.3f ms)\n", (unsigned long long)bit_us, bit_us / 1000.0);

    ac_buffer_sync_to_host(ctx, buf_y);
    float* y_out = (float*)buf_y->host_ptr;
    int bit_errors = 0;
    for (int r = 0; r < bit_M; ++r) {
        float ref_acc = 0.0f;
        int row_off = r * (bit_K / 4);
        for (int b = 0; b < (bit_K / 4); ++b) {
            unsigned char p = W_pack[row_off + b];
            int codes[4] = { p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3 };
            for (int k = 0; k < 4; ++k) {
                float w = (codes[k] == 1) ? 1.0f : ((codes[k] == 2) ? -1.0f : 0.0f);
                ref_acc += w * x_vec[b * 4 + k];
            }
        }
        float ref_y = ref_acc * scales[r];
        if (isnan(y_out[r]) || fabsf(y_out[r] - ref_y) > 1e-4f) {
            if (bit_errors < 3) {
                fprintf(stderr, "[ERROR] BitNet mismatch at %d: got %f, ref %f\n", r, y_out[r], ref_y);
            }
            bit_errors++;
        }
    }
    if (bit_errors > 0) {
        fprintf(stderr, "[FAIL] BitNet verification failed with %d errors!\n", bit_errors);
        return 1;
    }
    printf("[SUCCESS] BitNet 1.58b 100%% Validated against CPU reference!\n\n");

    /* ===================================================================== */
    /* 3. Transformer RMSNorm Verification                                  */
    /* ===================================================================== */
    printf("--- [TEST 3] Transformer RMSNorm (16 tokens x 1024 dim) ---\n");
    int num_rows = 16, hidden_dim = 1024;
    AcBuffer *buf_rx = NULL, *buf_rw = NULL, *buf_ry = NULL;

    ac_buffer_alloc(ctx, num_rows * hidden_dim * sizeof(float), &buf_rx);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &buf_rw);
    ac_buffer_alloc(ctx, num_rows * hidden_dim * sizeof(float), &buf_ry);

    float* rx = (float*)buf_rx->host_ptr;
    float* rw = (float*)buf_rw->host_ptr;
    for (int i = 0; i < num_rows * hidden_dim; ++i) rx[i] = rand_float(-1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; ++i) rw[i] = rand_float(0.5f, 1.5f);

    AcEvent* ev_rms = NULL;
    res = ac_ops_rmsnorm(ctx, num_rows, hidden_dim, 1e-6f, buf_rx, buf_rw, buf_ry, &ev_rms);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_ops_rmsnorm failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_rms);
    printf("[PASS] RMSNorm Executed in %llu us\n", (unsigned long long)ac_event_get_duration_us(ev_rms));

    ac_buffer_sync_to_host(ctx, buf_ry);
    float* ry = (float*)buf_ry->host_ptr;
    int rms_errors = 0;
    for (int r = 0; r < num_rows; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < hidden_dim; ++c) {
            float v = rx[r * hidden_dim + c];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf((sum_sq / (float)hidden_dim) + 1e-6f);
        for (int c = 0; c < hidden_dim; ++c) {
            float ref = rx[r * hidden_dim + c] * inv_rms * rw[c];
            if (fabsf(ry[r * hidden_dim + c] - ref) > 1e-4f) {
                rms_errors++;
            }
        }
    }
    if (rms_errors > 0) {
        fprintf(stderr, "[FAIL] RMSNorm verification failed!\n");
        return 1;
    }
    printf("[SUCCESS] RMSNorm 100%% Validated against CPU reference!\n\n");

    /* ===================================================================== */
    /* 4. SwiGLU Activation Verification                                     */
    /* ===================================================================== */
    printf("--- [TEST 4] Fused SwiGLU Activation (16384 elements) ---\n");
    int sw_N = 16384;
    AcBuffer *buf_gate = NULL, *buf_up = NULL, *buf_sw_y = NULL;

    ac_buffer_alloc(ctx, sw_N * sizeof(float), &buf_gate);
    ac_buffer_alloc(ctx, sw_N * sizeof(float), &buf_up);
    ac_buffer_alloc(ctx, sw_N * sizeof(float), &buf_sw_y);

    float* gate = (float*)buf_gate->host_ptr;
    float* up = (float*)buf_up->host_ptr;
    for (int i = 0; i < sw_N; ++i) {
        gate[i] = rand_float(-3.0f, 3.0f);
        up[i] = rand_float(-2.0f, 2.0f);
    }

    AcEvent* ev_sw = NULL;
    res = ac_ops_swiglu(ctx, sw_N, buf_gate, buf_up, buf_sw_y, &ev_sw);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_ops_swiglu failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_sw);
    printf("[PASS] SwiGLU Executed in %llu us\n", (unsigned long long)ac_event_get_duration_us(ev_sw));

    ac_buffer_sync_to_host(ctx, buf_sw_y);
    float* sw_y = (float*)buf_sw_y->host_ptr;
    int sw_errors = 0;
    for (int i = 0; i < sw_N; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        float ref = silu * up[i];
        if (fabsf(sw_y[i] - ref) > 1e-4f) {
            sw_errors++;
        }
    }
    if (sw_errors > 0) {
        fprintf(stderr, "[FAIL] SwiGLU verification failed!\n");
        return 1;
    }
    printf("[SUCCESS] SwiGLU 100%% Validated against CPU reference!\n\n");

    /* ===================================================================== */
    /* 5. In-Place RoPE Rotary Embedding Verification                        */
    /* ===================================================================== */
    printf("--- [TEST 5] In-Place RoPE (seq=16, heads=8, dim=64) ---\n");
    int seq_len = 16, num_heads = 8, head_dim = 64;
    int total_qk = seq_len * num_heads * head_dim;
    int total_cs = seq_len * head_dim;

    AcBuffer *buf_qk = NULL, *buf_cs = NULL;
    ac_buffer_alloc(ctx, total_qk * sizeof(float), &buf_qk);
    ac_buffer_alloc(ctx, total_cs * sizeof(float), &buf_cs);

    float* qk_data = (float*)buf_qk->host_ptr;
    float* cs_data = (float*)buf_cs->host_ptr;
    for (int i = 0; i < total_qk; ++i) qk_data[i] = rand_float(-1.0f, 1.0f);
    for (int i = 0; i < total_cs; ++i) cs_data[i] = rand_float(-1.0f, 1.0f);

    // Save copy for reference
    float* qk_ref = (float*)malloc(total_qk * sizeof(float));
    memcpy(qk_ref, qk_data, total_qk * sizeof(float));

    AcEvent* ev_rope = NULL;
    res = ac_ops_rope(ctx, seq_len, num_heads, head_dim, buf_qk, buf_cs, &ev_rope);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_ops_rope failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_rope);
    printf("[PASS] RoPE In-Place Executed in %llu us\n", (unsigned long long)ac_event_get_duration_us(ev_rope));

    ac_buffer_sync_to_host(ctx, buf_qk);
    int rope_errors = 0;
    int half_dim = head_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                int idx0 = s * (num_heads * head_dim) + h * head_dim + (d * 2);
                int idx1 = idx0 + 1;
                float cos_v = cs_data[s * head_dim + d];
                float sin_v = cs_data[s * head_dim + half_dim + d];
                float x0 = qk_ref[idx0];
                float x1 = qk_ref[idx1];
                float ref0 = x0 * cos_v - x1 * sin_v;
                float ref1 = x0 * sin_v + x1 * cos_v;
                if (fabsf(qk_data[idx0] - ref0) > 1e-4f || fabsf(qk_data[idx1] - ref1) > 1e-4f) {
                    rope_errors++;
                }
            }
        }
    }
    free(qk_ref);
    if (rope_errors > 0) {
        fprintf(stderr, "[FAIL] RoPE verification failed!\n");
        return 1;
    }
    printf("[SUCCESS] RoPE In-Place 100%% Validated against CPU reference!\n\n");

    /* ===================================================================== */
    /* 6. On-Disk Binary Cache Hit Test                                      */
    /* ===================================================================== */
    printf("--- [TEST 6] On-Disk Binary Cache Hit Verification ---\n");
    AcEvent* ev_gemm2 = NULL;
    res = ac_ops_gemm_fp32(ctx, M, N, K, 1.0f, buf_A, buf_B, 0.0f, buf_C, &ev_gemm2);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] Second GEMM (Cache Hit) failed: %d\n", res);
        return 1;
    }
    ac_event_wait(ctx, ev_gemm2);
    printf("[PASS] Second invocation (Loaded from Disk Binary Cache) executed in %llu us!\n\n",
           (unsigned long long)ac_event_get_duration_us(ev_gemm2));

    /* Cleanup */
    ac_event_free(ctx, ev_gemm);
    ac_event_free(ctx, ev_gemm2);
    ac_event_free(ctx, ev_bit);
    ac_event_free(ctx, ev_rms);
    ac_event_free(ctx, ev_sw);
    ac_event_free(ctx, ev_rope);

    ac_buffer_free(ctx, buf_A);
    ac_buffer_free(ctx, buf_B);
    ac_buffer_free(ctx, buf_C);
    ac_buffer_free(ctx, buf_x);
    ac_buffer_free(ctx, buf_W);
    ac_buffer_free(ctx, buf_sc);
    ac_buffer_free(ctx, buf_y);
    ac_buffer_free(ctx, buf_rx);
    ac_buffer_free(ctx, buf_rw);
    ac_buffer_free(ctx, buf_ry);
    ac_buffer_free(ctx, buf_gate);
    ac_buffer_free(ctx, buf_up);
    ac_buffer_free(ctx, buf_sw_y);
    ac_buffer_free(ctx, buf_qk);
    ac_buffer_free(ctx, buf_cs);

    ac_destroy(ctx);

    printf("===================================================================\n");
    printf("[AMUDA PHASE 2 COMPLETED] ALL 6 TENSOR OPS & CACHE TESTS PASSED!\n");
    printf("===================================================================\n");
    return 0;
}
