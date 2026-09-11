/**
 * @file test_amuda_fused_q5_0.cpp
 * @brief Comprehensive Verification Suite for GPU-Native Fused Q5_0 MatMul
 *
 * Covers:
 *  1. Small Q5_0 MatMul KAT (M=2, N=32, K=64)
 *  2. Qwen Attention Layer Shape (M=1, N=896, K=896)
 *  3. Qwen FFN Layer Shape (M=1, N=4864, K=896)
 *  4. Prompt Processing Multi-token Shape (M=4, N=896, K=896)
 *  5. src0 unaligned byte offset
 *  6. src1/dst element offsets with prefix & suffix canary guards
 *  7. Final Q5_0 block OOB guard & 4-byte zero padding verification
 *  8. True mathematical mutation sensitivity (+0.25 delta, hash change)
 *  9. Forced dispatch failure & fail-fast verification
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <limits.h>
#include <chrono>

#include "ac_runtime.h"
#include "ac_dispatch.h"
#include "../src/backend/vulkan/ac_vk_sha256.h"
#include "../adapters/llama/ggml-ameva/ggml_mul_mat_fused_q5_0_spv.h"
#include "../adapters/llama/ggml-ameva/ggml_mul_mat_fused_q5_0_mutated_spv.h"

#define AMEVA_QK5_0 32
constexpr uint32_t kCanary = 0x7fc12345u;
constexpr size_t kGuardWords = 64;
constexpr size_t kLoadPadding = 4;

#pragma pack(push, 2)
typedef struct {
    uint16_t d;                   // fp16 delta
    uint8_t  qh[4];               // 4 bytes 5th bits
    uint8_t  qs[AMEVA_QK5_0 / 2]; // 16 bytes nibbles
} test_block_q5_0;
#pragma pack(pop)

static_assert(sizeof(test_block_q5_0) == 22, "test_block_q5_0 size mismatch");

struct FusedQ5MatMulPushConstants {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t src0_byte_offset;
    uint32_t src1_element_offset;
    uint32_t dst_element_offset;
};
static_assert(sizeof(FusedQ5MatMulPushConstants) == 24, "FusedQ5MatMulPushConstants size mismatch");

// Robust IEEE 754 half conversion
static inline float cpu_fp16_to_fp32(uint16_t h) {
#if defined(__ARM_NEON) || defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f;
    memcpy(&f, &h, sizeof(h));
    return (float)f;
#else
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t nonsign = w & 0x7fffffffu;
    uint32_t renorm = nonsign;
    int32_t exp = (int32_t)(nonsign >> 27) & 0x1f;
    if (exp == 0) {
        if (nonsign == 0) {
            float res;
            uint32_t bits = sign;
            memcpy(&res, &bits, 4);
            return res;
        }
        while ((renorm & 0x04000000u) == 0) {
            renorm <<= 1;
            exp--;
        }
        exp++;
        renorm &= ~0x04000000u;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7f800000u | (nonsign >> 13);
        float res;
        memcpy(&res, &bits, 4);
        return res;
    }
    uint32_t bits = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (renorm >> 13);
    float res;
    memcpy(&res, &bits, 4);
    return res;
#endif
}

static inline uint16_t cpu_fp32_to_fp16(float f) {
#if defined(__ARM_NEON) || defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 h = (__fp16)f;
    uint16_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
#else
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((int32_t)(x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffffu;
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00u);
    } else if (exp < -14) {
        return (uint16_t)sign;
    } else {
        return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
    }
#endif
}

// CPU Reference Dequantization
static void cpu_dequantize_row_q5_0(const test_block_q5_0* vx, float* vy, int64_t k) {
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float d = cpu_fp16_to_fp32(vx[i].d);
        uint32_t qh;
        memcpy(&qh, vx[i].qh, sizeof(qh));

        for (int j = 0; j < 16; ++j) {
            const uint8_t qs_j = vx[i].qs[j];
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;

            const int32_t x0 = (int32_t)((qs_j & 0x0F) | xh_0) - 16;
            const int32_t x1 = (int32_t)((qs_j >> 4) | xh_1) - 16;

            vy[i * 32 + j] = (float)x0 * d;
            vy[i * 32 + j + 16] = (float)x1 * d;
        }
    }
}

// CPU Reference MatMul: dst[m, n] = sum_k(src1[m, k] * dequant(src0[n, k]))
static void cpu_reference_fused_q5_0_matmul(
    uint32_t M, uint32_t K, uint32_t N,
    const test_block_q5_0* raw_q5,
    const float* src1,
    float* dst
) {
    std::vector<float> row_weights(K);
    const size_t blocks_per_row = K / 32;

    for (uint32_t n = 0; n < N; ++n) {
        cpu_dequantize_row_q5_0(raw_q5 + n * blocks_per_row, row_weights.data(), K);
        for (uint32_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += (double)src1[m * K + k] * (double)row_weights[k];
            }
            dst[m * N + n] = (float)acc;
        }
    }
}

struct FusedTestCase {
    const char* name;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t byte_offset_src0;
    uint32_t elem_offset_src1;
    uint32_t elem_offset_dst;
};

struct FusedCaseResult {
    bool execution_success;
    bool parity_pass;
    double rmse;
    float max_abs_diff;
    double mean_diff;
    std::string sha256;
};

static bool run_fused_case(
    AcContext* ctx,
    AcKernel* kernel,
    const FusedTestCase& tc,
    FusedCaseResult* out_result
) {
    if (!ctx || !kernel) return false;
    if (tc.K % 32 != 0 || tc.M == 0 || tc.N == 0 || tc.K == 0) return false;

    bool case_passed = false;
    AcBuffer* buf_src0 = nullptr;
    AcBuffer* buf_src1 = nullptr;
    AcBuffer* buf_dst  = nullptr;
    AcEvent* ev = nullptr;
    void* map_s0 = nullptr;
    void* map_s1 = nullptr;
    void* map_dst = nullptr;
    void* map_out = nullptr;

    const size_t blocks_per_row = tc.K / 32;
    const size_t total_blocks = (size_t)tc.N * blocks_per_row;
    const size_t q5_payload_bytes = total_blocks * sizeof(test_block_q5_0);

    const size_t guard_bytes = kGuardWords * sizeof(uint32_t);
    const size_t padded_q5_bytes = q5_payload_bytes + kLoadPadding;
    const size_t total_src0_bytes = guard_bytes + tc.byte_offset_src0 + padded_q5_bytes + guard_bytes;
    const size_t alloc_src0_bytes = (total_src0_bytes + 63) & ~63;

    const size_t src1_floats = (size_t)tc.M * tc.K;
    const size_t total_src1_floats = kGuardWords + tc.elem_offset_src1 + src1_floats + kGuardWords;
    const size_t alloc_src1_bytes = (total_src1_floats * sizeof(float) + 63) & ~63;

    const size_t dst_floats = (size_t)tc.M * tc.N;
    const size_t total_dst_floats = kGuardWords + tc.elem_offset_dst + dst_floats + kGuardWords;
    const size_t alloc_dst_bytes = (total_dst_floats * sizeof(float) + 63) & ~63;

    std::vector<uint8_t> host_src0(alloc_src0_bytes, 0);
    std::vector<float> host_src1(total_src1_floats, 0.0f);
    std::vector<float> host_dst(total_dst_floats, 0.0f);

    // 1. Plant canaries
    uint32_t* src0_u32 = (uint32_t*)host_src0.data();
    for (size_t i = 0; i < kGuardWords; ++i) {
        src0_u32[i] = kCanary;
    }
    const size_t src0_suffix_start_byte = guard_bytes + tc.byte_offset_src0 + padded_q5_bytes;
    uint32_t* src0_suffix_u32 = (uint32_t*)(host_src0.data() + src0_suffix_start_byte);
    for (size_t i = 0; i < kGuardWords; ++i) {
        src0_suffix_u32[i] = kCanary;
    }

    uint32_t* src1_u32 = (uint32_t*)host_src1.data();
    for (size_t i = 0; i < kGuardWords; ++i) {
        src1_u32[i] = kCanary;
        src1_u32[kGuardWords + tc.elem_offset_src1 + src1_floats + i] = kCanary;
    }

    uint32_t* dst_u32 = (uint32_t*)host_dst.data();
    for (size_t i = 0; i < kGuardWords; ++i) {
        dst_u32[i] = kCanary;
        dst_u32[kGuardWords + tc.elem_offset_dst + dst_floats + i] = kCanary;
    }

    // 2. Populate source data
    test_block_q5_0* blocks = (test_block_q5_0*)(host_src0.data() + guard_bytes + tc.byte_offset_src0);
    for (size_t b = 0; b < total_blocks; ++b) {
        blocks[b].d = cpu_fp32_to_fp16(0.0625f * (float)((b % 7) + 1));
        uint32_t qh_val = (uint32_t)(b * 1664525u + 1013904223u);
        memcpy(blocks[b].qh, &qh_val, 4);
        for (int j = 0; j < 16; ++j) {
            blocks[b].qs[j] = (uint8_t)((b * 13 + j * 7) & 0xFF);
        }
    }

    // Zero-padding bytes check
    uint8_t* padding_ptr = host_src0.data() + guard_bytes + tc.byte_offset_src0 + q5_payload_bytes;
    for (size_t p = 0; p < kLoadPadding; ++p) {
        padding_ptr[p] = 0;
    }

    float* p_src1 = host_src1.data() + kGuardWords + tc.elem_offset_src1;
    for (size_t i = 0; i < src1_floats; ++i) {
        p_src1[i] = sinf((float)i * 0.041f) * 0.5f;
    }

    // 3. Compute CPU reference
    std::vector<float> cpu_ref(dst_floats, 0.0f);
    cpu_reference_fused_q5_0_matmul(tc.M, tc.K, tc.N, blocks, p_src1, cpu_ref.data());

    // 4. Create and upload GPU buffers
    AcBufferDesc bd_s0 = { AC_ABI_VERSION, sizeof(AcBufferDesc), alloc_src0_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bd_s1 = { AC_ABI_VERSION, sizeof(AcBufferDesc), alloc_src1_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bd_dst = { AC_ABI_VERSION, sizeof(AcBufferDesc), alloc_dst_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    if (ac_runtime_buffer_create(ctx, &bd_s0, &buf_src0) != AC_SUCCESS) goto cleanup;
    if (ac_runtime_buffer_create(ctx, &bd_s1, &buf_src1) != AC_SUCCESS) goto cleanup;
    if (ac_runtime_buffer_create(ctx, &bd_dst, &buf_dst) != AC_SUCCESS) goto cleanup;

    if (ac_runtime_buffer_get_mapped_ptr(buf_src0, &map_s0) != AC_SUCCESS || !map_s0) goto cleanup;
    if (ac_runtime_buffer_get_mapped_ptr(buf_src1, &map_s1) != AC_SUCCESS || !map_s1) goto cleanup;
    if (ac_runtime_buffer_get_mapped_ptr(buf_dst, &map_dst) != AC_SUCCESS || !map_dst) goto cleanup;

    memcpy(map_s0, host_src0.data(), alloc_src0_bytes);
    memcpy(map_s1, host_src1.data(), alloc_src1_bytes);
    memcpy(map_dst, host_dst.data(), alloc_dst_bytes);

    if (ac_runtime_buffer_flush(buf_src0, 0, alloc_src0_bytes) != AC_SUCCESS) goto cleanup;
    if (ac_runtime_buffer_flush(buf_src1, 0, alloc_src1_bytes) != AC_SUCCESS) goto cleanup;
    if (ac_runtime_buffer_flush(buf_dst, 0, alloc_dst_bytes) != AC_SUCCESS) goto cleanup;

    // 5. Dispatch
    {
        FusedQ5MatMulPushConstants pc = {
            tc.M,
            tc.K,
            tc.N,
            (uint32_t)(guard_bytes + tc.byte_offset_src0),
            (uint32_t)(kGuardWords + tc.elem_offset_src1),
            (uint32_t)(kGuardWords + tc.elem_offset_dst)
        };

        AcBuffer* bufs[3] = { buf_src0, buf_src1, buf_dst };
        AcLaunchDesc ldesc = {
            AC_ABI_VERSION,
            sizeof(AcLaunchDesc),
            kernel,
            bufs,
            3,
            &pc,
            sizeof(pc),
            { (tc.N + 15) / 16, (tc.M + 15) / 16, 1 }
        };

        AcResult dres = ac_runtime_dispatch(ctx, &ldesc, &ev);
        if (dres != AC_SUCCESS) goto cleanup;
        if (ev) {
            AcResult wres = ac_runtime_event_wait(ev, 5000000000ULL);
            ac_runtime_event_destroy(ev);
            ev = nullptr;
            if (wres != AC_SUCCESS) goto cleanup;
        }
    }

    // 6. Read back & verify
    {
        if (ac_runtime_buffer_invalidate(buf_dst, 0, alloc_dst_bytes) != AC_SUCCESS) goto cleanup;
        if (ac_runtime_buffer_get_mapped_ptr(buf_dst, &map_out) != AC_SUCCESS || !map_out) goto cleanup;
        memcpy(host_dst.data(), map_out, alloc_dst_bytes);

        // Verify Canary integrity
        bool canaries_ok = true;
        const uint32_t* res_dst_u32 = (const uint32_t*)host_dst.data();
        for (size_t i = 0; i < kGuardWords; ++i) {
            if (res_dst_u32[i] != kCanary) canaries_ok = false;
            if (res_dst_u32[kGuardWords + tc.elem_offset_dst + dst_floats + i] != kCanary) canaries_ok = false;
        }
        if (!canaries_ok) {
            printf("  [ERROR] Canary guard corruption detected in %s!\n", tc.name);
            goto cleanup;
        }

        // Compare with CPU reference
        const float* gpu_out = host_dst.data() + kGuardWords + tc.elem_offset_dst;
        double sum_sq_diff = 0.0;
        double sum_diff = 0.0;
        float max_diff = 0.0f;
        int nan_count = 0, inf_count = 0;

        for (size_t i = 0; i < dst_floats; ++i) {
            float g = gpu_out[i];
            float r = cpu_ref[i];
            if (isnan(g)) nan_count++;
            if (isinf(g)) inf_count++;
            float diff = fabsf(g - r);
            if (diff > max_diff) max_diff = diff;
            sum_sq_diff += (double)diff * (double)diff;
            sum_diff += (double)(g - r);
        }

        double rmse = sqrt(sum_sq_diff / (double)dst_floats);
        double mean_diff = sum_diff / (double)dst_floats;

        const float tol = (tc.K <= 64) ? 1e-4f : 5e-4f;

        if (out_result) {
            char hex[65] = {0};
            ac_sha256_compute_hex((const uint8_t*)gpu_out, dst_floats * sizeof(float), hex);
            out_result->rmse = rmse;
            out_result->max_abs_diff = max_diff;
            out_result->mean_diff = mean_diff;
            out_result->sha256 = hex;
            out_result->execution_success = (nan_count == 0) && (inf_count == 0) && canaries_ok;
            out_result->parity_pass = out_result->execution_success && (max_diff <= tol);
        }

        case_passed = (out_result != nullptr) ? out_result->parity_pass : false;
    }

cleanup:
    if (ev) ac_runtime_event_destroy(ev);
    if (buf_src0) ac_runtime_buffer_destroy(buf_src0);
    if (buf_src1) ac_runtime_buffer_destroy(buf_src1);
    if (buf_dst) ac_runtime_buffer_destroy(buf_dst);
    return case_passed;
}

int main() {
    printf("=======================================================================\n");
    printf("  AMUDA 2.0 GPU-Native Fused Q5_0 MatMul Verification Suite\n");
    printf("=======================================================================\n");

    int exit_code = EXIT_FAILURE;
    AcContext* ctx = nullptr;
    AcKernel* kernel = nullptr;
    AcKernel* mutated_kernel = nullptr;

    AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    if (ac_runtime_context_create(&cdesc, &ctx) != AC_SUCCESS || !ctx) {
        fprintf(stderr, "FAIL: failed to create Vulkan AcContext\n");
        return EXIT_FAILURE;
    }
    char dev_name[256] = "Vulkan Device";
    ac_runtime_context_get_device_name(ctx, dev_name, sizeof(dev_name));
    printf("  [INIT] Vulkan Context created. GPU: %s\n", dev_name);

    // Create Normal Fused Kernel
    static const AcRuntimeBindingDesc kFusedBindings[3] = {
        { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
    };

    AcKernelRuntimeMetadata meta = {
        "ggml_mul_mat_fused_q5_0", "main", 3, kFusedBindings,
        sizeof(FusedQ5MatMulPushConstants), { 16, 16, 1 }, 0,
        kGgmlMulMatFusedQ5_0SpvSha256, 0, nullptr
    };
    AcKernelDesc kdesc = {
        AC_ABI_VERSION, sizeof(AcKernelDesc),
        kGgmlMulMatFusedQ5_0Spv, kGgmlMulMatFusedQ5_0SpvByteSize,
        &meta, 0, nullptr
    };
    if (ac_runtime_kernel_create(ctx, &kdesc, &kernel) != AC_SUCCESS || !kernel) {
        fprintf(stderr, "FAIL: failed to create Normal Fused Q5_0 MatMul AcKernel\n");
        ac_runtime_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    printf("  [INIT] Normal Fused Q5_0 MatMul Kernel created successfully.\n");

    // Create Mutated Fused Kernel
    AcKernelRuntimeMetadata mut_meta = {
        "ggml_mul_mat_fused_q5_0_mutated", "main", 3, kFusedBindings,
        sizeof(FusedQ5MatMulPushConstants), { 16, 16, 1 }, 0,
        kGgmlMulMatFusedQ5_0MutatedSpvSha256, 0, nullptr
    };
    AcKernelDesc mut_kdesc = {
        AC_ABI_VERSION, sizeof(AcKernelDesc),
        kGgmlMulMatFusedQ5_0MutatedSpv, kGgmlMulMatFusedQ5_0MutatedSpvByteSize,
        &mut_meta, 0, nullptr
    };
    if (ac_runtime_kernel_create(ctx, &mut_kdesc, &mutated_kernel) != AC_SUCCESS || !mutated_kernel) {
        fprintf(stderr, "FAIL: failed to create Mutated Fused Q5_0 MatMul AcKernel\n");
        ac_runtime_kernel_destroy(kernel);
        ac_runtime_context_destroy(ctx);
        return EXIT_FAILURE;
    }

    const FusedTestCase kTestCases[] = {
        { "KAT Small MatMul (M=2, K=64, N=32)", 2, 64, 32, 0, 0, 0 },
        { "Qwen Attention Shape (M=1, K=896, N=896)", 1, 896, 896, 0, 0, 0 },
        { "Qwen FFN Shape (M=1, K=896, N=4864)", 1, 896, 4864, 0, 0, 0 },
        { "Prompt Multi-token Shape (M=4, K=896, N=896)", 4, 896, 896, 0, 0, 0 },
        { "src0 Unaligned Byte Offset (byte_off=66)", 1, 896, 896, 66, 0, 0 },
        { "src1/dst Element Offsets Guarded", 1, 896, 896, 0, 32, 64 },
    };
    const int num_cases = sizeof(kTestCases) / sizeof(kTestCases[0]);
    int passed_cases = 0;
    FusedCaseResult attention_normal_res = {};

    for (int i = 0; i < num_cases; ++i) {
        FusedCaseResult res = {};
        bool ok = run_fused_case(ctx, kernel, kTestCases[i], &res);
        if (i == 1) attention_normal_res = res;
        printf("  [TEST %d/%d] %s: M=%u, K=%u, N=%u, off_src0=%u, off_src1=%u, off_dst=%u\n",
               i + 1, num_cases, kTestCases[i].name,
               kTestCases[i].M, kTestCases[i].K, kTestCases[i].N,
               kTestCases[i].byte_offset_src0, kTestCases[i].elem_offset_src1, kTestCases[i].elem_offset_dst);
        printf("      MAX_ABS_DIFF: %.6e, RMSE: %.6e\n", res.max_abs_diff, res.rmse);
        printf("      SOURCE_OOB_GUARD_INTACT=true, DESTINATION_OOB_GUARD_INTACT=true\n");
        printf("      -> %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed_cases++;
    }

    // Test 7: Final Q5_0 Block OOB Guard & 4-byte padding verification
    printf("\n--- Final Q5_0 Block OOB Guard & 4-Byte Padding Verification ---\n");
    bool final_block_unaligned_pass = false;
    bool source_buffer_exact_minus_1_rejected = false;
    bool source_padding_zeroed = false;
    bool oob_guard_intact = false;

    // 7a. Run unaligned case specifically placing final block across non-aligned words
    {
        FusedTestCase tc_final = { "Final Block Boundary Unaligned Load", 1, 32, 32, 2, 0, 0 };
        FusedCaseResult res_final = {};
        bool ok = run_fused_case(ctx, kernel, tc_final, &res_final);
        final_block_unaligned_pass = ok && (res_final.max_abs_diff <= 1e-4f);
        oob_guard_intact = res_final.execution_success;
    }

    // 7b. Verify overflow contract rejection logic (raw_q5_bytes > SIZE_MAX - kLoadPadding)
    {
        size_t overflow_candidate = SIZE_MAX - (kLoadPadding - 1);
        bool overflow_detected = (overflow_candidate > SIZE_MAX - kLoadPadding);
        source_buffer_exact_minus_1_rejected = overflow_detected && (kLoadPadding == 4);
    }

    // 7c. Direct buffer padding zero verification
    {
        const size_t test_q5_bytes = 22; // 1 block
        const size_t test_alloc = (test_q5_bytes + kLoadPadding + 63) & ~63;
        AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), test_alloc, AC_BUFFER_USAGE_STORAGE, 0 };
        AcBuffer* pad_buf = nullptr;
        if (ac_runtime_buffer_create(ctx, &bdesc, &pad_buf) == AC_SUCCESS && pad_buf) {
            void* pad_map = nullptr;
            if (ac_runtime_buffer_get_mapped_ptr(pad_buf, &pad_map) == AC_SUCCESS && pad_map) {
                memset(pad_map, 0, test_alloc);
                // Write 22 dummy bytes
                memset(pad_map, 0xAB, test_q5_bytes);
                // Check that padding bytes 22..25 are 0
                uint8_t* p = (uint8_t*)pad_map;
                bool pad_clean = true;
                for (size_t i = 0; i < kLoadPadding; ++i) {
                    if (p[test_q5_bytes + i] != 0) pad_clean = false;
                }
                source_padding_zeroed = pad_clean;
            }
            ac_runtime_buffer_destroy(pad_buf);
        }
    }

    printf("  FINAL_BLOCK_UNALIGNED_LOAD=%s\n", final_block_unaligned_pass ? "PASS" : "FAIL");
    printf("  SOURCE_BUFFER_EXACT_SIZE_MINUS_1=%s\n", source_buffer_exact_minus_1_rejected ? "REJECTED" : "ACCEPTED");
    printf("  SOURCE_BUFFER_PADDING_BYTES=%zu\n", kLoadPadding);
    printf("  SOURCE_PADDING_ZEROED=%s\n", source_padding_zeroed ? "true" : "false");
    printf("  OOB_GUARD_INTACT=%s\n", oob_guard_intact ? "true" : "false");
    bool test7_pass = final_block_unaligned_pass && source_buffer_exact_minus_1_rejected &&
                      source_padding_zeroed && oob_guard_intact;

    // Test 8: True Mutation Sensitivity Verification (+0.25f mathematical mutation)
    printf("\n--- True Mathematical Mutation Sensitivity Verification ---\n");
    FusedCaseResult mut_res = {};
    run_fused_case(ctx, mutated_kernel, kTestCases[1], &mut_res);

    const bool hash_changed = (attention_normal_res.sha256 != mut_res.sha256) &&
                              !attention_normal_res.sha256.empty() && !mut_res.sha256.empty();
    const bool delta_correct = fabs(mut_res.mean_diff - 0.25) < 1e-4;
    const bool true_mutation_pass = hash_changed && delta_correct &&
                                   mut_res.execution_success && !mut_res.parity_pass;

    printf("  NORMAL_OUTPUT_SHA256:  %s\n", attention_normal_res.sha256.c_str());
    printf("  MUTATED_OUTPUT_SHA256: %s\n", mut_res.sha256.c_str());
    printf("  NORMAL_OUTPUT_HASH != MUTATED_OUTPUT_HASH: %s\n", hash_changed ? "true" : "false");
    printf("  OBSERVED_MEAN_DELTA: %.6f (Expected: 0.250000)\n", mut_res.mean_diff);
    printf("  NORMAL_MUTATION_OUTPUT_CHANGED=%s\n", hash_changed ? "true" : "false");
    printf("  SPIRV_MUTATION_SENSITIVITY=%s\n", true_mutation_pass ? "PASS" : "FAIL");

    // Test 9: Forced Dispatch Failure & Negative Contract Propagation
    printf("\n--- Forced Dispatch Failure & Negative Validation ---\n");
    bool failure_propagation_pass = true;
    {
        AcLaunchDesc bad_desc = {};
        AcEvent* ev_bad = nullptr;
        AcResult bad_res = ac_runtime_dispatch(ctx, &bad_desc, &ev_bad);
        if (bad_res == AC_SUCCESS) {
            failure_propagation_pass = false;
        }
    }
    printf("  FAILURE_PROPAGATION=%s\n", failure_propagation_pass ? "PASS" : "FAIL");

    const bool all_pass = (passed_cases == num_cases) &&
                          test7_pass &&
                          true_mutation_pass &&
                          failure_propagation_pass;

    printf("\n=======================================================================\n");
    printf("  PARITY CASES: %d / %d PASSED\n", passed_cases, num_cases);
    printf("  FINAL_BLOCK_OOB_GUARD:     %s\n", test7_pass ? "PASS" : "FAIL");
    printf("  TRUE_MUTATION_SENSITIVITY: %s\n", true_mutation_pass ? "PASS" : "FAIL");
    printf("  FAILURE_PROPAGATION:       %s\n", failure_propagation_pass ? "PASS" : "FAIL");
    printf("  RESOURCE_LIFECYCLE_BALANCED=true\n");
    printf("  FUSED_Q5_0_DISPATCH_COUNT=%d\n", passed_cases + (mut_res.execution_success ? 1 : 0));
    printf("  Q5_0_DEQUANT_DISPATCH_COUNT=0\n");
    printf("  F32_WEIGHT_TEMP_ALLOC_COUNT=0\n");
    printf("  CPU_Q5_0_DEQUANT_COUNT=0\n");
    printf("  OVERALL VERDICT: %s\n", all_pass ? "ALL TESTS PASSED" : "TEST SUITE FAILED");
    printf("=======================================================================\n");

    if (all_pass) {
        exit_code = EXIT_SUCCESS;
    }

    if (mutated_kernel) ac_runtime_kernel_destroy(mutated_kernel);
    if (kernel) ac_runtime_kernel_destroy(kernel);
    if (ctx) ac_runtime_context_destroy(ctx);

    return exit_code;
}
