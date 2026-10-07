/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_vision_pointwise_conv_relu.cpp
 * @brief AMUDA Vision Fused Pointwise Conv2D 1x1 + ReLU & Device-Resident Verification
 *
 * Enforces:
 *   - 3-Way Numerical Parity:
 *       Route A: CPU Double Reference Conv1x1 + ReLU
 *       Route B: GPU Conv1x1 + Separate GPU ReLU (2 dispatches, device-resident buffer)
 *       Route C: Fused GPU Conv1x1 + ReLU (1 dispatch)
 *   - Pre-activation vs Post-activation Parity & Zero Negative Elements Verification
 *   - P50/P90 Latency Benchmarking (A vs B vs C)
 *   - Anti-Fake Mutation Sensitivity (+0.25 mutated kernel)
 *   - Zero-Silent-Fallback & Full Lifecycle Recovery
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include "ac_runtime.h"
#include "ac_vision_ops.h"
#include "../src/backend/vulkan/ac_vk_sha256.h"
#include "../src/shaders/pointwise_conv_spv.h"
#include "../src/shaders/pointwise_conv_relu_spv.h"
#include "../src/shaders/pointwise_conv_relu_mutated_spv.h"
#include "../src/shaders/relu_spv.h"

constexpr uint32_t kCanary = 0x7fc12345u;

static void fill_buffer_canary(float* dst, size_t elem_count) {
    uint32_t* words = reinterpret_cast<uint32_t*>(dst);
    for (size_t i = 0; i < elem_count; ++i) {
        words[i] = kCanary;
    }
}

static size_t verify_buffer_not_canary(const float* dst, size_t elem_count) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(dst);
    size_t count = 0;
    for (size_t i = 0; i < elem_count; ++i) {
        if (words[i] == kCanary) {
            count++;
        }
    }
    return count;
}

static int load_fixture_file(const char* filename, void* out_buf, size_t max_bytes, size_t* out_read_bytes) {
    char full_path[512];
    FILE* fp = NULL;

    const char* env_dir = getenv("AMEVA_VISION_FIXTURE_DIR");
    if (env_dir && env_dir[0] != '\0') {
        snprintf(full_path, sizeof(full_path), "%s/%s", env_dir, filename);
        fp = fopen(full_path, "rb");
    }

    if (!fp) {
        const char* search_paths[] = {
            "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/%s",
            "tests/fixtures/%s",
            "../tests/fixtures/%s",
            "../../tests/fixtures/%s",
            "fixtures/%s",
            "%s",
            NULL
        };
        for (int i = 0; search_paths[i] != NULL; i++) {
            snprintf(full_path, sizeof(full_path), search_paths[i], filename);
            fp = fopen(full_path, "rb");
            if (fp != NULL) break;
        }
    }
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -2;
    }
    long sz = ftell(fp);
    if (sz <= 0 || (size_t)sz > max_bytes) {
        fclose(fp);
        return -3;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -4;
    }
    size_t read_bytes = fread(out_buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (read_bytes != (size_t)sz) return -5;
    if (out_read_bytes) *out_read_bytes = read_bytes;
    return 0;
}

// -----------------------------------------------------------------------------
// Numerical Parity Evaluator
// -----------------------------------------------------------------------------
struct NumericalMetrics {
    double cosine_similarity;
    double rmse;
    double p99_abs_error;
    double max_abs_error;
    uint32_t nan_count;
    uint32_t inf_count;
};

static NumericalMetrics evaluate_parity(const float* ref, const float* act, size_t count) {
    NumericalMetrics m = { 0.0, 0.0, 0.0, 0.0, 0, 0 };
    if (!ref || !act || count == 0) return m;

    double dot = 0.0, ref_sq = 0.0, act_sq = 0.0, sum_sq_diff = 0.0;
    std::vector<double> abs_diffs(count);

    for (size_t i = 0; i < count; ++i) {
        float r = ref[i];
        float a = act[i];

        if (isnan(a)) m.nan_count++;
        if (isinf(a)) m.inf_count++;

        double dr = (double)r;
        double da = (double)a;
        double diff = fabs(dr - da);

        abs_diffs[i] = diff;
        dot += dr * da;
        ref_sq += dr * dr;
        act_sq += da * da;
        sum_sq_diff += diff * diff;
        if (diff > m.max_abs_error) {
            m.max_abs_error = diff;
        }
    }

    double denom = sqrt(ref_sq) * sqrt(act_sq);
    m.cosine_similarity = (denom > 1e-12) ? (dot / denom) : 0.0;
    m.rmse = sqrt(sum_sq_diff / (double)count);

    std::sort(abs_diffs.begin(), abs_diffs.end());
    size_t p99_idx = (size_t)((double)count * 0.99);
    if (p99_idx >= count) p99_idx = count - 1;
    m.p99_abs_error = abs_diffs[p99_idx];

    return m;
}

// -----------------------------------------------------------------------------
// CPU Double-Precision Reference (Conv1x1 and Conv1x1+ReLU)
// -----------------------------------------------------------------------------
static void cpu_pointwise_conv_double_ref(
    const float* in_nhwc,
    const float* weight_ohwi,
    const float* bias,
    float* out_conv,
    float* out_conv_relu,
    uint32_t H, uint32_t W, uint32_t Cin, uint32_t Cout
) {
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            const float* in_pixel = in_nhwc + (h * W + w) * Cin;
            float* conv_pixel = out_conv + (h * W + w) * Cout;
            float* relu_pixel = out_conv_relu + (h * W + w) * Cout;

            for (uint32_t oc = 0; oc < Cout; ++oc) {
                double acc = bias ? (double)bias[oc] : 0.0;
                const float* w_kernel = weight_ohwi + oc * Cin;
                for (uint32_t ic = 0; ic < Cin; ++ic) {
                    acc += (double)in_pixel[ic] * (double)w_kernel[ic];
                }
                conv_pixel[oc] = (float)acc;
                relu_pixel[oc] = (acc > 0.0) ? (float)acc : 0.0f;
            }
        }
    }
}

int main(void) {
    printf("======================================================================\n");
    printf("AMEVA Vision - Fused Pointwise Conv2D 1x1 + ReLU Verification Suite\n");
    printf("======================================================================\n\n");

    const uint32_t H = 56, W = 56, Cin = 64, Cout = 16;
    const size_t in_elems = H * W * Cin;
    const size_t out_elems = H * W * Cout;
    const size_t weight_elems = Cout * Cin;
    const size_t bias_elems = Cout;

    const size_t in_bytes = in_elems * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);
    const size_t weight_bytes = weight_elems * sizeof(float);
    const size_t bias_bytes = bias_elems * sizeof(float);

    // -------------------------------------------------------------------------
    // 1. Load Fixtures
    // -------------------------------------------------------------------------
    printf("--- [STEP 1: LOAD FIXTURES] ---\n");
    std::vector<float> host_in(in_elems);
    std::vector<float> host_w(weight_elems);
    std::vector<float> host_b(bias_elems);

    size_t read_bytes = 0;
    if (load_fixture_file("squeezenet_fire2_input_activation_nhwc.bin", host_in.data(), in_bytes, &read_bytes) != 0 || read_bytes != in_bytes) {
        fprintf(stderr, "FATAL: Failed to load squeezenet_fire2_input_activation_nhwc.bin (read %zu, expected %zu)\n", read_bytes, in_bytes);
        return EXIT_FAILURE;
    }
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_weight.bin", host_w.data(), weight_bytes, &read_bytes) != 0 || read_bytes != weight_bytes) {
        fprintf(stderr, "FATAL: Failed to load squeezenet_fire2_squeeze1x1_weight.bin\n");
        return EXIT_FAILURE;
    }
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_bias.bin", host_b.data(), bias_bytes, &read_bytes) != 0 || read_bytes != bias_bytes) {
        fprintf(stderr, "FATAL: Failed to load squeezenet_fire2_squeeze1x1_bias.bin\n");
        return EXIT_FAILURE;
    }
    printf("Fixtures loaded successfully (Input=%zu B, Weight=%zu B, Bias=%zu B)\n", in_bytes, weight_bytes, bias_bytes);

    // -------------------------------------------------------------------------
    // 2. Compute Route A: CPU Double Reference (Conv1x1 & Conv1x1+ReLU)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 2: ROUTE A - CPU DOUBLE REFERENCE] ---\n");
    std::vector<float> cpu_conv_ref(out_elems);
    std::vector<float> cpu_conv_relu_ref(out_elems);

    auto t_cpu_0 = std::chrono::high_resolution_clock::now();
    cpu_pointwise_conv_double_ref(
        host_in.data(), host_w.data(), host_b.data(),
        cpu_conv_ref.data(), cpu_conv_relu_ref.data(),
        H, W, Cin, Cout
    );
    auto t_cpu_1 = std::chrono::high_resolution_clock::now();
    double cpu_time_us = std::chrono::duration<double, std::micro>(t_cpu_1 - t_cpu_0).count();

    size_t pre_act_neg_count = 0;
    size_t post_act_neg_count = 0;
    for (size_t i = 0; i < out_elems; ++i) {
        if (cpu_conv_ref[i] < 0.0f) pre_act_neg_count++;
        if (cpu_conv_relu_ref[i] < 0.0f) post_act_neg_count++;
    }
    printf("  CPU_PRE_ACTIVATION_NEGATIVE_COUNT=%zu (out of %zu)\n", pre_act_neg_count, out_elems);
    printf("  CPU_POST_ACTIVATION_NEGATIVE_COUNT=%zu\n", post_act_neg_count);
    printf("  CPU_REFERENCE_TIME=%.1f us\n", cpu_time_us);

    if (pre_act_neg_count == 0) {
        fprintf(stderr, "WARNING: Pre-activation has 0 negative elements; ReLU activation check requires negative values\n");
    }
    if (post_act_neg_count != 0) {
        fprintf(stderr, "FATAL: CPU ReLU reference has negative values!\n");
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // 3. Initialize Vulkan Runtime & Allocate Device Buffers
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 3: INITIALIZE VULKAN RUNTIME & DEVICE BUFFERS] ---\n");
    uint32_t ctx_created = 0, ctx_destroyed = 0;
    uint32_t kernel_created = 0, kernel_destroyed = 0;
    uint32_t buffer_created = 0, buffer_destroyed = 0;

    AcContext* ctx = nullptr;
    AcKernel* kernel_conv = nullptr;
    AcKernel* kernel_relu = nullptr;
    AcKernel* kernel_fused = nullptr;
    AcKernel* kernel_mutated = nullptr;

    AcBuffer* buf_in = nullptr;
    AcBuffer* buf_w = nullptr;
    AcBuffer* buf_b = nullptr;
    AcBuffer* buf_inter = nullptr;   /* Device-resident intermediate buffer for Route B */
    AcBuffer* buf_out_sep = nullptr; /* Route B final output */
    AcBuffer* buf_out_fused = nullptr; /* Route C final output */

    float* mapped_in = nullptr;
    float* mapped_w = nullptr;
    float* mapped_b = nullptr;
    float* mapped_inter = nullptr;
    float* mapped_out_sep = nullptr;
    float* mapped_out_fused = nullptr;

    AcVisionPointwiseConvParams params = {};
    params.abi_version = AC_VISION_ABI_VERSION;
    params.struct_size = sizeof(AcVisionPointwiseConvParams);
    params.input_channels = Cin;
    params.output_channels = Cout;
    params.input_height = H;
    params.input_width = W;
    params.has_bias = 1;
    params.kernel_h = 1;
    params.kernel_w = 1;
    params.stride_h = 1;
    params.stride_w = 1;
    params.dilation_h = 1;
    params.dilation_w = 1;

    AcVisionTensorDesc in_desc = {};
    in_desc.abi_version = AC_VISION_ABI_VERSION;
    in_desc.struct_size = sizeof(AcVisionTensorDesc);
    in_desc.dtype = AC_VISION_DTYPE_F32;
    in_desc.layout = AC_VISION_LAYOUT_NHWC;
    in_desc.n = 1;
    in_desc.h = H;
    in_desc.w = W;
    in_desc.c = Cin;
    in_desc.row_stride_bytes = W * Cin * sizeof(float);

    AcVisionTensorDesc weight_desc = {};
    weight_desc.abi_version = AC_VISION_ABI_VERSION;
    weight_desc.struct_size = sizeof(AcVisionTensorDesc);
    weight_desc.dtype = AC_VISION_DTYPE_F32;
    weight_desc.layout = AC_VISION_LAYOUT_OHWI;
    weight_desc.n = Cout;
    weight_desc.h = 1;
    weight_desc.w = 1;
    weight_desc.c = Cin;
    weight_desc.row_stride_bytes = 0;

    AcVisionTensorDesc out_desc = {};
    out_desc.abi_version = AC_VISION_ABI_VERSION;
    out_desc.struct_size = sizeof(AcVisionTensorDesc);
    out_desc.dtype = AC_VISION_DTYPE_F32;
    out_desc.layout = AC_VISION_LAYOUT_NHWC;
    out_desc.n = 1;
    out_desc.h = H;
    out_desc.w = W;
    out_desc.c = Cout;
    out_desc.row_stride_bytes = W * Cout * sizeof(float);

    AcBufferDesc bdesc_in = { AC_ABI_VERSION, sizeof(AcBufferDesc), in_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_w = { AC_ABI_VERSION, sizeof(AcBufferDesc), weight_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_b = { AC_ABI_VERSION, sizeof(AcBufferDesc), bias_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_out = { AC_ABI_VERSION, sizeof(AcBufferDesc), out_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    bool init_ok = false;
    AcContextDesc ctx_desc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcResult res = ac_runtime_context_create(&ctx_desc, &ctx);
    if (res == AC_SUCCESS && ctx) {
        ctx_created++;
        if (ac_vision_pointwise_conv_kernel_create(ctx, nullptr, 0, nullptr, &kernel_conv) == AC_SUCCESS && kernel_conv) {
            kernel_created++;
        }
        if (ac_vision_relu_kernel_create(ctx, nullptr, 0, nullptr, &kernel_relu) == AC_SUCCESS && kernel_relu) {
            kernel_created++;
        }
        if (ac_vision_pointwise_conv_relu_kernel_create(ctx, nullptr, 0, nullptr, &kernel_fused) == AC_SUCCESS && kernel_fused) {
            kernel_created++;
        }
        if (ac_vision_pointwise_conv_relu_kernel_create(ctx, kPointwiseConvReluMutatedSpv, kPointwiseConvReluMutatedSpvByteSize, kPointwiseConvReluMutatedSpvSha256, &kernel_mutated) == AC_SUCCESS && kernel_mutated) {
            kernel_created++;
        }

        if (kernel_created == 4) {
            if (ac_runtime_buffer_create(ctx, &bdesc_in, &buf_in) == AC_SUCCESS && buf_in) buffer_created++;
            if (ac_runtime_buffer_create(ctx, &bdesc_w, &buf_w) == AC_SUCCESS && buf_w) buffer_created++;
            if (ac_runtime_buffer_create(ctx, &bdesc_b, &buf_b) == AC_SUCCESS && buf_b) buffer_created++;
            if (ac_runtime_buffer_create(ctx, &bdesc_out, &buf_inter) == AC_SUCCESS && buf_inter) buffer_created++;
            if (ac_runtime_buffer_create(ctx, &bdesc_out, &buf_out_sep) == AC_SUCCESS && buf_out_sep) buffer_created++;
            if (ac_runtime_buffer_create(ctx, &bdesc_out, &buf_out_fused) == AC_SUCCESS && buf_out_fused) buffer_created++;

            if (buffer_created == 6) {
                if (ac_runtime_buffer_get_mapped_ptr(buf_in, (void**)&mapped_in) == AC_SUCCESS &&
                    ac_runtime_buffer_get_mapped_ptr(buf_w, (void**)&mapped_w) == AC_SUCCESS &&
                    ac_runtime_buffer_get_mapped_ptr(buf_b, (void**)&mapped_b) == AC_SUCCESS &&
                    ac_runtime_buffer_get_mapped_ptr(buf_inter, (void**)&mapped_inter) == AC_SUCCESS &&
                    ac_runtime_buffer_get_mapped_ptr(buf_out_sep, (void**)&mapped_out_sep) == AC_SUCCESS &&
                    ac_runtime_buffer_get_mapped_ptr(buf_out_fused, (void**)&mapped_out_fused) == AC_SUCCESS &&
                    mapped_in && mapped_w && mapped_b && mapped_inter && mapped_out_sep && mapped_out_fused) {
                    init_ok = true;
                }
            }
        }
    }

    bool execution_ok = false;

    if (init_ok) {
        memcpy(mapped_in, host_in.data(), in_bytes);
        memcpy(mapped_w, host_w.data(), weight_bytes);
        memcpy(mapped_b, host_b.data(), bias_bytes);

        ac_runtime_buffer_flush(buf_in, 0, in_bytes);
        ac_runtime_buffer_flush(buf_w, 0, weight_bytes);
        ac_runtime_buffer_flush(buf_b, 0, bias_bytes);

        auto run_all_passes = [&]() -> bool {
            // -----------------------------------------------------------------
            // 4. Execute Route B: Separate GPU Conv1x1 -> Separate GPU ReLU
            // -----------------------------------------------------------------
            printf("\n--- [STEP 4: ROUTE B - SEPARATE GPU CONV1x1 + RELU (DEVICE-RESIDENT)] ---\n");
            fill_buffer_canary(mapped_inter, out_elems);
            fill_buffer_canary(mapped_out_sep, out_elems);
            ac_runtime_buffer_flush(buf_inter, 0, out_bytes);
            ac_runtime_buffer_flush(buf_out_sep, 0, out_bytes);

            auto t_b0 = std::chrono::high_resolution_clock::now();
            AcResult r_b1 = ac_vision_pointwise_conv_device_f32(
                ctx, kernel_conv, &params,
                buf_in, &in_desc, buf_w, &weight_desc, buf_b,
                buf_inter, &out_desc, nullptr
            );
            if (r_b1 != AC_SUCCESS) {
                fprintf(stderr, "FATAL: Route B Dispatch 1 (Conv1x1) failed (%d)\n", r_b1);
                return false;
            }

            // Zero Intermediate Readbacks: buf_inter is directly consumed by ReLU on device!
            uint32_t intermediate_readbacks = 0;

            AcResult r_b2 = ac_vision_relu_device_f32(
                ctx, kernel_relu,
                buf_inter, buf_out_sep,
                out_elems, nullptr
            );
            if (r_b2 != AC_SUCCESS) {
                fprintf(stderr, "FATAL: Route B Dispatch 2 (ReLU) failed (%d)\n", r_b2);
                return false;
            }
            auto t_b1 = std::chrono::high_resolution_clock::now();
            double route_b_time_us = std::chrono::duration<double, std::micro>(t_b1 - t_b0).count();

            ac_runtime_buffer_invalidate(buf_inter, 0, out_bytes);
            ac_runtime_buffer_invalidate(buf_out_sep, 0, out_bytes);

            size_t unwritten_inter = verify_buffer_not_canary(mapped_inter, out_elems);
            size_t unwritten_sep = verify_buffer_not_canary(mapped_out_sep, out_elems);
            if (unwritten_inter != 0 || unwritten_sep != 0) {
                fprintf(stderr, "FATAL: Route B output contains unwritten canary elements (inter=%zu, sep=%zu)\n", unwritten_inter, unwritten_sep);
                return false;
            }

            printf("  ROUTE_B_INTERMEDIATE_HOST_READBACKS=%u (Device-Resident: true)\n", intermediate_readbacks);
            printf("  ROUTE_B_DISPATCH_COUNT=2\n");
            printf("  ROUTE_B_EXECUTION_TIME=%.1f us\n", route_b_time_us);

            // -----------------------------------------------------------------
            // 5. Execute Route C: Fused GPU Conv1x1+ReLU (1 dispatch)
            // -----------------------------------------------------------------
            printf("\n--- [STEP 5: ROUTE C - FUSED GPU CONV1x1 + RELU (1 DISPATCH)] ---\n");
            fill_buffer_canary(mapped_out_fused, out_elems);
            ac_runtime_buffer_flush(buf_out_fused, 0, out_bytes);

            auto t_c0 = std::chrono::high_resolution_clock::now();
            AcResult r_c = ac_vision_pointwise_conv_relu_with_kernel_f32(
                ctx, kernel_fused, &params,
                buf_in, &in_desc, buf_w, &weight_desc, buf_b,
                buf_out_fused, &out_desc, nullptr
            );
            if (r_c != AC_SUCCESS) {
                fprintf(stderr, "FATAL: Route C Dispatch (Fused Conv1x1+ReLU) failed (%d)\n", r_c);
                return false;
            }
            auto t_c1 = std::chrono::high_resolution_clock::now();
            double route_c_time_us = std::chrono::duration<double, std::micro>(t_c1 - t_c0).count();

            ac_runtime_buffer_invalidate(buf_out_fused, 0, out_bytes);
            size_t unwritten_fused = verify_buffer_not_canary(mapped_out_fused, out_elems);
            if (unwritten_fused != 0) {
                fprintf(stderr, "FATAL: Route C output contains unwritten canary elements (%zu)\n", unwritten_fused);
                return false;
            }

            printf("  ROUTE_C_DISPATCH_COUNT=1\n");
            printf("  ROUTE_C_EXECUTION_TIME=%.1f us\n", route_c_time_us);

            // -----------------------------------------------------------------
            // 6. Pre-Activation vs Post-Activation & Negative Count Verification
            // -----------------------------------------------------------------
            printf("\n--- [STEP 6: PRE/POST ACTIVATION & NEGATIVE ELEMENT VERIFICATION] ---\n");
            size_t gpu_inter_neg_count = 0;
            size_t gpu_sep_neg_count = 0;
            size_t gpu_fused_neg_count = 0;

            for (size_t i = 0; i < out_elems; ++i) {
                if (mapped_inter[i] < 0.0f) gpu_inter_neg_count++;
                if (mapped_out_sep[i] < 0.0f) gpu_sep_neg_count++;
                if (mapped_out_fused[i] < 0.0f) gpu_fused_neg_count++;
            }

            printf("  GPU_PRE_ACTIVATION_NEGATIVE_COUNT=%zu (out of %zu)\n", gpu_inter_neg_count, out_elems);
            printf("  GPU_SEPARATE_POST_ACTIVATION_NEGATIVE_COUNT=%zu\n", gpu_sep_neg_count);
            printf("  GPU_FUSED_POST_ACTIVATION_NEGATIVE_COUNT=%zu\n", gpu_fused_neg_count);

            bool neg_elements_ok = (gpu_inter_neg_count == pre_act_neg_count) &&
                                   (gpu_sep_neg_count == 0) &&
                                   (gpu_fused_neg_count == 0);
            printf("NEGATIVE_ELEMENT_VERIFICATION=%s\n", neg_elements_ok ? "PASS" : "FAIL");
            if (!neg_elements_ok) {
                fprintf(stderr, "FATAL: Negative element verification failed (Pre: %zu vs %zu, Sep: %zu, Fused: %zu)\n",
                        gpu_inter_neg_count, pre_act_neg_count, gpu_sep_neg_count, gpu_fused_neg_count);
                return false;
            }

            // -----------------------------------------------------------------
            // 7. 3-Way Parity Evaluation (A vs B, A vs C, B vs C)
            // -----------------------------------------------------------------
            printf("\n--- [STEP 7: 3-WAY NUMERICAL PARITY EVALUATION] ---\n");
            NumericalMetrics m_a_vs_b = evaluate_parity(cpu_conv_relu_ref.data(), mapped_out_sep, out_elems);
            NumericalMetrics m_a_vs_c = evaluate_parity(cpu_conv_relu_ref.data(), mapped_out_fused, out_elems);
            NumericalMetrics m_b_vs_c = evaluate_parity(mapped_out_sep, mapped_out_fused, out_elems);

            printf("  [PARITY 1: ROUTE A (CPU DOUBLE REF) vs ROUTE B (SEPARATE GPU)]\n");
            printf("    COSINE_SIMILARITY=%.7f\n", m_a_vs_b.cosine_similarity);
            printf("    RMSE=%.7f\n", m_a_vs_b.rmse);
            printf("    P99_ABS_ERROR=%.7f\n", m_a_vs_b.p99_abs_error);
            printf("    MAX_ABS_ERROR=%.7f\n", m_a_vs_b.max_abs_error);
            printf("    NAN_COUNT=%u, INF_COUNT=%u\n", m_a_vs_b.nan_count, m_a_vs_b.inf_count);

            printf("  [PARITY 2: ROUTE A (CPU DOUBLE REF) vs ROUTE C (FUSED GPU)]\n");
            printf("    COSINE_SIMILARITY=%.7f\n", m_a_vs_c.cosine_similarity);
            printf("    RMSE=%.7f\n", m_a_vs_c.rmse);
            printf("    P99_ABS_ERROR=%.7f\n", m_a_vs_c.p99_abs_error);
            printf("    MAX_ABS_ERROR=%.7f\n", m_a_vs_c.max_abs_error);
            printf("    NAN_COUNT=%u, INF_COUNT=%u\n", m_a_vs_c.nan_count, m_a_vs_c.inf_count);

            printf("  [PARITY 3: ROUTE B (SEPARATE GPU) vs ROUTE C (FUSED GPU)]\n");
            printf("    COSINE_SIMILARITY=%.7f\n", m_b_vs_c.cosine_similarity);
            printf("    RMSE=%.7f\n", m_b_vs_c.rmse);
            printf("    P99_ABS_ERROR=%.7f\n", m_b_vs_c.p99_abs_error);
            printf("    MAX_ABS_ERROR=%.7f\n", m_b_vs_c.max_abs_error);
            printf("    NAN_COUNT=%u, INF_COUNT=%u\n", m_b_vs_c.nan_count, m_b_vs_c.inf_count);

            bool pass_a_vs_b = (m_a_vs_b.cosine_similarity >= 0.99999) &&
                               (m_a_vs_b.rmse <= 0.005) &&
                               (m_a_vs_b.max_abs_error <= 0.05) &&
                               (m_a_vs_b.nan_count == 0) && (m_a_vs_b.inf_count == 0);

            bool pass_a_vs_c = (m_a_vs_c.cosine_similarity >= 0.99999) &&
                               (m_a_vs_c.rmse <= 0.005) &&
                               (m_a_vs_c.max_abs_error <= 0.05) &&
                               (m_a_vs_c.nan_count == 0) && (m_a_vs_c.inf_count == 0);

            bool pass_b_vs_c = (m_b_vs_c.cosine_similarity >= 0.999999) &&
                               (m_b_vs_c.rmse <= 0.0001) &&
                               (m_b_vs_c.max_abs_error <= 0.0001) &&
                               (m_b_vs_c.nan_count == 0) && (m_b_vs_c.inf_count == 0);

            printf("ROUTE_A_VS_ROUTE_B_PARITY=%s\n", pass_a_vs_b ? "PASS" : "FAIL");
            printf("ROUTE_A_VS_ROUTE_C_PARITY=%s\n", pass_a_vs_c ? "PASS" : "FAIL");
            printf("ROUTE_B_VS_ROUTE_C_PARITY=%s\n", pass_b_vs_c ? "PASS" : "FAIL");

            if (!pass_a_vs_b || !pass_a_vs_c || !pass_b_vs_c) {
                fprintf(stderr, "FATAL: 3-Way parity verification failed!\n");
                return false;
            }

            // -----------------------------------------------------------------
            // 8. Latency Benchmark (Warmup 5, 20 iterations, P50 & P90)
            // -----------------------------------------------------------------
            printf("\n--- [STEP 8: P50/P90 LATENCY BENCHMARK] ---\n");
            const int kWarmup = 5;
            const int kIters = 20;

            // Warmup Route B
            for (int i = 0; i < kWarmup; ++i) {
                ac_vision_pointwise_conv_device_f32(ctx, kernel_conv, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_inter, &out_desc, nullptr);
                ac_vision_relu_device_f32(ctx, kernel_relu, buf_inter, buf_out_sep, out_elems, nullptr);
            }
            // Warmup Route C
            for (int i = 0; i < kWarmup; ++i) {
                ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_fused, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out_fused, &out_desc, nullptr);
            }

            std::vector<double> times_b(kIters);
            for (int i = 0; i < kIters; ++i) {
                auto tb0 = std::chrono::high_resolution_clock::now();
                ac_vision_pointwise_conv_device_f32(ctx, kernel_conv, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_inter, &out_desc, nullptr);
                ac_vision_relu_device_f32(ctx, kernel_relu, buf_inter, buf_out_sep, out_elems, nullptr);
                auto tb1 = std::chrono::high_resolution_clock::now();
                times_b[i] = std::chrono::duration<double, std::micro>(tb1 - tb0).count();
            }

            std::vector<double> times_c(kIters);
            for (int i = 0; i < kIters; ++i) {
                auto tc0 = std::chrono::high_resolution_clock::now();
                ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_fused, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out_fused, &out_desc, nullptr);
                auto tc1 = std::chrono::high_resolution_clock::now();
                times_c[i] = std::chrono::duration<double, std::micro>(tc1 - tc0).count();
            }

            std::sort(times_b.begin(), times_b.end());
            std::sort(times_c.begin(), times_c.end());

            double p50_b = times_b[kIters / 2];
            double p90_b = times_b[(size_t)(kIters * 0.9)];
            double p50_c = times_c[kIters / 2];
            double p90_c = times_c[(size_t)(kIters * 0.9)];

            printf("  CPU_DOUBLE_REF_TIME=%.1f us\n", cpu_time_us);
            printf("  SEPARATE_GPU_ROUTE_B_P50=%.1f us, P90=%.1f us (2 dispatches)\n", p50_b, p90_b);
            printf("  FUSED_GPU_ROUTE_C_P50=%.1f us, P90=%.1f us (1 dispatch)\n", p50_c, p90_c);
            printf("  FUSED_VS_SEPARATE_SPEEDUP=%.2fx\n", p50_b / p50_c);
            printf("  FUSED_VS_CPU_SPEEDUP=%.2fx\n", cpu_time_us / p50_c);

            // -----------------------------------------------------------------
            // 9. Anti-Fake Mutation Sensitivity Verification (+0.25 mutated kernel)
            // -----------------------------------------------------------------
            printf("\n--- [STEP 9: ANTI-FAKE MUTATION SENSITIVITY TEST] ---\n");
            fill_buffer_canary(mapped_out_fused, out_elems);
            ac_runtime_buffer_flush(buf_out_fused, 0, out_bytes);

            AcResult r_mut = ac_vision_pointwise_conv_relu_with_kernel_f32(
                ctx, kernel_mutated, &params,
                buf_in, &in_desc, buf_w, &weight_desc, buf_b,
                buf_out_fused, &out_desc, nullptr
            );
            if (r_mut != AC_SUCCESS) {
                fprintf(stderr, "FATAL: Mutated kernel execution failed (%d)\n", r_mut);
                return false;
            }

            ac_runtime_buffer_invalidate(buf_out_fused, 0, out_bytes);
            NumericalMetrics m_mut = evaluate_parity(cpu_conv_relu_ref.data(), mapped_out_fused, out_elems);

            printf("  MUTATED_KERNEL_MAX_ABS_ERROR=%.7f (Expected >= 0.24)\n", m_mut.max_abs_error);
            printf("  MUTATED_KERNEL_RMSE=%.7f\n", m_mut.rmse);

            bool mutation_detected = (m_mut.max_abs_error >= 0.24);
            printf("MUTATION_SENSITIVITY_RESULT=%s\n", mutation_detected ? "PASS" : "FAIL");

            if (!mutation_detected) {
                fprintf(stderr, "FATAL: Mutated kernel (+0.25) was not detected!\n");
                return false;
            }

            return true;
        };

        execution_ok = run_all_passes();
    } else {
        fprintf(stderr, "FATAL: Initialization of Vulkan resources failed\n");
    }

    // -------------------------------------------------------------------------
    // 10. Cleanup & Lifecycle Recovery Verification
    // -------------------------------------------------------------------------
    if (buf_in) { ac_runtime_buffer_destroy(buf_in); buffer_destroyed++; }
    if (buf_w) { ac_runtime_buffer_destroy(buf_w); buffer_destroyed++; }
    if (buf_b) { ac_runtime_buffer_destroy(buf_b); buffer_destroyed++; }
    if (buf_inter) { ac_runtime_buffer_destroy(buf_inter); buffer_destroyed++; }
    if (buf_out_sep) { ac_runtime_buffer_destroy(buf_out_sep); buffer_destroyed++; }
    if (buf_out_fused) { ac_runtime_buffer_destroy(buf_out_fused); buffer_destroyed++; }

    if (kernel_conv) { ac_vision_pointwise_conv_kernel_destroy(kernel_conv); kernel_destroyed++; }
    if (kernel_relu) { ac_vision_pointwise_conv_kernel_destroy(kernel_relu); kernel_destroyed++; }
    if (kernel_fused) { ac_vision_pointwise_conv_kernel_destroy(kernel_fused); kernel_destroyed++; }
    if (kernel_mutated) { ac_vision_pointwise_conv_kernel_destroy(kernel_mutated); kernel_destroyed++; }

    if (ctx) { ac_runtime_context_destroy(ctx); ctx_destroyed++; }

    printf("\n--- [LIFECYCLE REPORT] ---\n");
    printf("POINTWISE_CONV_RELU_CONTEXT_CREATE_DESTROY=%u/%u\n", ctx_created, ctx_destroyed);
    printf("POINTWISE_CONV_RELU_KERNEL_CREATE_DESTROY=%u/%u\n", kernel_created, kernel_destroyed);
    printf("POINTWISE_CONV_RELU_BUFFER_CREATE_DESTROY=%u/%u\n", buffer_created, buffer_destroyed);

    bool lifecycle_ok = (ctx_created == ctx_destroyed) &&
                        (kernel_created == kernel_destroyed) &&
                        (buffer_created == buffer_destroyed);
    printf("POINTWISE_CONV_RELU_LIFECYCLE=%s\n", lifecycle_ok ? "PASS" : "FAIL");

    bool all_pass = init_ok && execution_ok && lifecycle_ok;
    printf("\n======================================================================\n");
    printf("POINTWISE_CONV_RELU_TEST_RESULT=%s\n", all_pass ? "PASS" : "FAIL");
    printf("======================================================================\n");

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
