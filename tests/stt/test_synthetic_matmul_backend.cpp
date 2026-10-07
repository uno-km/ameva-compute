/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_synthetic_matmul_backend.cpp
 * @brief Formal verification of synthetic GGML graph MatMul execution via AMEVA Vulkan backend
 */

#include "ggml-backend-ameva.h"
#include "../../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp  = (h >> 10) & 0x001f;
    uint32_t mant = h & 0x03ff;
    uint32_t val;
    if (exp == 0) {
        if (mant == 0) {
            val = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            val = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        val = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        val = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &val, sizeof(float));
    return out;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x007fffff;
    if (exp <= 0) {
        return (uint16_t)sign;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

int main() {
    printf("=== [AMEVA FORMAL BACKEND SYNTHETIC GRAPH MATMUL TEST] ===\n");

    const uint32_t M = 1500;
    const uint32_t N = 384;
    const uint32_t K = 384;

    // 1. Initialize GGML context
    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FATAL: Failed to init ggml context\n");
        return 1;
    }

    // 2. Create tensors matching target Whisper encoder projection
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, w, a);

    // 3. Populate deterministic synthetic data
    uint16_t * w_data = (uint16_t *)w->data;
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        float val = sinf((float)i * 0.01f) * 0.1f;
        w_data[i] = fp32_to_fp16(val);
    }

    float * a_data = (float *)a->data;
    for (size_t i = 0; i < (size_t)M * K; ++i) {
        a_data[i] = cosf((float)i * 0.02f) * 0.1f;
    }

    // Zero out output
    memset(dst->data, 0, (size_t)M * N * sizeof(float));

    // 4. Build compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // 5. Initialize AMEVA backend and reset counters
    ameva_backend_reset_counters();
    ggml_backend_t backend = ggml_backend_ameva_init(nullptr);
    if (!backend) {
        fprintf(stderr, "FATAL: Failed to init AMEVA backend\n");
        ggml_free(ctx);
        return 2;
    }

    // 6. Execute compute graph via AMEVA backend
    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: ggml_backend_graph_compute failed with status %d\n", status);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 3;
    }

    // 7. Verify execution counters
    AmevaBackendCounters counters = {};
    ameva_backend_get_counters(&counters);

    printf("PATTERN_MATCH_COUNT=%llu\n", (unsigned long long)counters.supported_pattern_match_count);
    printf("AMEVA_DISPATCH_ATTEMPT_COUNT=%llu\n", (unsigned long long)counters.ameva_dispatch_attempt_count);
    printf("AMEVA_DISPATCH_SUCCESS_COUNT=%llu\n", (unsigned long long)counters.ameva_dispatch_success_count);
    printf("AMEVA_GPU_OP_COUNT=%llu\n", (unsigned long long)counters.ameva_gpu_op_count);

    if (counters.ameva_gpu_op_count != 1 || counters.ameva_dispatch_success_count != 1) {
        fprintf(stderr, "FATAL: GPU op counter invariant violation! Expected 1, got gpu_op=%llu, success=%llu\n",
                (unsigned long long)counters.ameva_gpu_op_count, (unsigned long long)counters.ameva_dispatch_success_count);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 4;
    }

    // 8. Compute CPU Golden Reference and evaluate numerical parity
    std::vector<float> golden((size_t)M * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float w_val = fp16_to_fp32(w_data[n * K + k]);
                float a_val = a_data[m * K + k];
                sum += w_val * a_val;
            }
            golden[m * N + n] = sum;
        }
    }

    float * gpu_out = (float *)dst->data;
    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    bool output_written = false;

    for (size_t i = 0; i < (size_t)M * N; ++i) {
        if (gpu_out[i] != 0.0f) output_written = true;
        if (isnan(gpu_out[i]) || isinf(gpu_out[i])) {
            fprintf(stderr, "FATAL: NaN/Inf detected in GPU output at index %zu\n", i);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return 5;
        }
        double diff = fabs((double)gpu_out[i] - (double)golden[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        sum_sq_err += diff * diff;
    }

    double rmse = sqrt(sum_sq_err / ((double)M * N));

    printf("MAX_ABS_ERROR=%f\n", max_abs_err);
    printf("RMSE=%f\n", rmse);
    printf("OUTPUT_WRITTEN=%s\n", output_written ? "true" : "false");

    bool parity_pass = (output_written && max_abs_err < 0.05 && rmse < 0.005);
    printf("SYNTHETIC_MUL_MAT_PARITY=%s\n", parity_pass ? "PASS" : "FAIL");

    ggml_backend_free(backend);
    ggml_free(ctx);

    if (!parity_pass) {
        fprintf(stderr, "FATAL: Parity verification failed\n");
        return 6;
    }

    printf("=== [AMEVA FORMAL BACKEND SYNTHETIC GRAPH TEST PASSED] ===\n");
    return 0;
}
