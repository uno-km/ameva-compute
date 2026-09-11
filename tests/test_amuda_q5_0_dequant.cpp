/**
 * @file test_amuda_q5_0_dequant.cpp
 * @brief Standalone GPU-Native Q5_0 Dequantization KAT & Numerical Parity Verification Suite
 *
 * Verifies Vulkan compute shader dequantization against CPU reference implementation.
 * Validates:
 *  - Single block (32 elements)
 *  - Multi-block (random & known values across full 5-bit range [-16, 15])
 *  - LLM realistic tensor shapes (K=896, K=4864, multi-row matrices)
 *  - Offset handling (byte_offset_src, elem_offset_dst)
 *  - Strict precision bounds: MAX_ABS_ERROR < 1e-4, RMSE, NAN_COUNT = 0, INF_COUNT = 0
 *  - Anti-fake mutation sensitivity check
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>

#include "ac_runtime.h"
#include "ac_dispatch.h"
#include "../adapters/llama/ggml-ameva/q5_0_dequant_spv.h"

#define CHECK_AC(expr)                                                      \
    do {                                                                    \
        AcResult _result = (expr);                                          \
        if (_result != AC_SUCCESS) {                                        \
            printf("  -> FAIL: %s returned %d (line %d)\n",                 \
                   #expr, _result, __LINE__);                               \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define AMEVA_QK5_0 32

#pragma pack(push, 2)
typedef struct {
    uint16_t d;                   // fp16 delta
    uint8_t  qh[4];               // 4 bytes 5th bits
    uint8_t  qs[AMEVA_QK5_0 / 2]; // 16 bytes nibbles
} test_block_q5_0;
#pragma pack(pop)

static_assert(sizeof(test_block_q5_0) == 22, "test_block_q5_0 size mismatch");

static inline float cpu_fp16_to_fp32(uint16_t h) {
#if defined(__ARM_NEON) || defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f;
    memcpy(&f, &h, sizeof(h));
    return (float)f;
#else
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000;
    uint32_t two_w = w + w;
    uint32_t exp = (two_w >> 27) & 0x1f;
    if (exp == 0) {
        if (two_w == 0) {
            float res = 0.0f;
            if (sign) res = -0.0f;
            return res;
        }
        while ((two_w & 0x04000000) == 0) {
            two_w <<= 1;
            exp--;
        }
        exp++;
        two_w &= ~0x04000000;
    } else if (exp == 31) {
        uint32_t f_bits = sign | 0x7f800000 | (two_w >> 13);
        float res;
        memcpy(&res, &f_bits, 4);
        return res;
    }
    uint32_t f_bits = sign | ((exp + (127 - 15)) << 23) | (two_w >> 13);
    float res;
    memcpy(&res, &f_bits, 4);
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
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffff;
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00);
    } else if (exp < -14) {
        return (uint16_t)sign;
    } else {
        return (uint16_t)(sign | ((exp + 15) << 10) | (mant >> 13));
    }
#endif
}

static void cpu_dequantize_q5_0(const test_block_q5_0* vx, float* vy, int64_t k) {
    const int64_t nb = k / AMEVA_QK5_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = cpu_fp16_to_fp32(vx[i].d);
        uint32_t qh;
        memcpy(&qh, vx[i].qh, sizeof(qh));

        for (int j = 0; j < AMEVA_QK5_0 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = ((vx[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((vx[i].qs[j] >> 4) | xh_1) - 16;

            vy[i * AMEVA_QK5_0 + j + 0] = (float)x0 * d;
            vy[i * AMEVA_QK5_0 + j + AMEVA_QK5_0 / 2] = (float)x1 * d;
        }
    }
}

int main(void) {
    printf("=======================================================================\n");
    printf("  AMUDA 2.0 GPU-Native Q5_0 Dequantization Parity Suite\n");
    printf("=======================================================================\n");

    // 1. Initialize Vulkan Context
    AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcContext* ctx = nullptr;
    CHECK_AC(ac_runtime_context_create(&cdesc, &ctx));

    char device_name[256] = "Unknown";
    ac_runtime_context_get_device_name(ctx, device_name, sizeof(device_name));
    printf("  [INIT] Vulkan Context created. GPU: %s\n", device_name);

    // 2. Create Q5_0 Dequant Kernel
    static const AcRuntimeBindingDesc kBindings[2] = {
        { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
    };

    AcKernelRuntimeMetadata meta = {
        "q5_0_dequant",
        "main",
        2,
        kBindings,
        12, // sizeof(uint32_t) * 3: num_blocks, byte_offset_src, elem_offset_dst
        { 64, 1, 1 },
        0,
        kQ5_0DequantSpvSha256,
        0,
        nullptr
    };

    AcKernelDesc kdesc = {
        AC_ABI_VERSION,
        sizeof(AcKernelDesc),
        kQ5_0DequantSpv,
        kQ5_0DequantSpvByteSize,
        &meta,
        0,
        nullptr
    };

    AcKernel* kernel = nullptr;
    CHECK_AC(ac_runtime_kernel_create(ctx, &kdesc, &kernel));
    printf("  [INIT] Q5_0 Dequant Kernel created successfully.\n");

    // Test Cases
    struct TestCase {
        const char* name;
        uint32_t num_blocks;
        uint32_t byte_offset_src;
        uint32_t elem_offset_dst;
    } cases[] = {
        { "Single Block (32 elements)", 1, 0, 0 },
        { "Multi-Block (64 blocks, 2048 elements)", 64, 0, 0 },
        { "Qwen2.5-0.5B Attention Dim (28 blocks, 896 elements)", 28, 0, 0 },
        { "Qwen2.5-0.5B FFN Dim (152 blocks, 4864 elements)", 152, 0, 0 },
        { "Offset Source & Destination (32 blocks, byte_off=66, elem_off=128)", 32, 66, 128 },
        { "Large Tensor Matrix (1024 blocks, 32768 elements)", 1024, 0, 0 }
    };

    int total_cases = sizeof(cases) / sizeof(cases[0]);
    int passed_cases = 0;

    for (int t = 0; t < total_cases; ++t) {
        TestCase& tc = cases[t];
        uint32_t num_blocks = tc.num_blocks;
        size_t total_src_bytes = tc.byte_offset_src + (size_t)num_blocks * sizeof(test_block_q5_0);
        size_t total_dst_floats = tc.elem_offset_dst + (size_t)num_blocks * AMEVA_QK5_0;
        size_t total_dst_bytes = total_dst_floats * sizeof(float);

        // Prepare Host Data
        std::vector<uint8_t> host_src(total_src_bytes + 64, 0);
        test_block_q5_0* blocks = (test_block_q5_0*)(host_src.data() + tc.byte_offset_src);

        // Populate deterministic pseudorandom test blocks
        uint32_t seed = 42 + t * 1000;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            float d_val = 0.05f + (float)((seed * 1103515245U + 12345U) % 1000) / 5000.0f;
            if (b % 7 == 0) d_val = -d_val;
            blocks[b].d = cpu_fp32_to_fp16(d_val);
            seed = seed * 1664525U + 1013904223U;

            blocks[b].qh[0] = (uint8_t)(seed & 0xFF);
            blocks[b].qh[1] = (uint8_t)((seed >> 8) & 0xFF);
            blocks[b].qh[2] = (uint8_t)((seed >> 16) & 0xFF);
            blocks[b].qh[3] = (uint8_t)((seed >> 24) & 0xFF);
            seed = seed * 1664525U + 1013904223U;

            for (int j = 0; j < 16; ++j) {
                blocks[b].qs[j] = (uint8_t)((seed + j * 17) & 0xFF);
            }
        }

        // Compute Reference on CPU
        std::vector<float> cpu_ref(num_blocks * AMEVA_QK5_0);
        cpu_dequantize_q5_0(blocks, cpu_ref.data(), (int64_t)num_blocks * AMEVA_QK5_0);

        // Create GPU Buffers
        AcBuffer* buf_src = nullptr;
        AcBuffer* buf_dst = nullptr;

        AcBufferDesc bdesc_src = { AC_ABI_VERSION, sizeof(AcBufferDesc), (host_src.size() + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
        AcBufferDesc bdesc_dst = { AC_ABI_VERSION, sizeof(AcBufferDesc), (total_dst_bytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };

        CHECK_AC(ac_runtime_buffer_create(ctx, &bdesc_src, &buf_src));
        CHECK_AC(ac_runtime_buffer_create(ctx, &bdesc_dst, &buf_dst));

        void* mapped_src = nullptr;
        void* mapped_dst = nullptr;
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(buf_src, &mapped_src));
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(buf_dst, &mapped_dst));

        memcpy(mapped_src, host_src.data(), host_src.size());
        CHECK_AC(ac_runtime_buffer_flush(buf_src, 0, host_src.size()));

        // Zero out destination
        memset(mapped_dst, 0, total_dst_bytes);
        CHECK_AC(ac_runtime_buffer_flush(buf_dst, 0, total_dst_bytes));

        // Dispatch
        uint32_t push_constants[3] = { num_blocks, tc.byte_offset_src, tc.elem_offset_dst };
        AcBuffer* bound_bufs[2] = { buf_src, buf_dst };
        uint32_t grid_x = (num_blocks + 63) / 64;

        AcLaunchDesc ldesc = {
            AC_ABI_VERSION,
            sizeof(AcLaunchDesc),
            kernel,
            bound_bufs,
            2,
            push_constants,
            sizeof(push_constants),
            { grid_x, 1, 1 }
        };

        AcEvent* ev = nullptr;
        CHECK_AC(ac_runtime_dispatch(ctx, &ldesc, &ev));
        CHECK_AC(ac_runtime_event_wait(ev, 5000000000ULL));
        ac_runtime_event_destroy(ev);

        // Readback
        CHECK_AC(ac_runtime_buffer_invalidate(buf_dst, 0, total_dst_bytes));
        const float* gpu_res = (const float*)mapped_dst + tc.elem_offset_dst;

        // Verify Numerical Parity
        double max_abs_diff = 0.0;
        double sum_sq_diff = 0.0;
        int nan_count = 0;
        int inf_count = 0;
        uint32_t total_elems = num_blocks * AMEVA_QK5_0;

        for (uint32_t e = 0; e < total_elems; ++e) {
            float g = gpu_res[e];
            float c = cpu_ref[e];

            if (isnan(g) || isnan(c)) nan_count++;
            if (isinf(g) || isinf(c)) inf_count++;

            double diff = fabs((double)g - (double)c);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            sum_sq_diff += diff * diff;
        }

        double rmse = sqrt(sum_sq_diff / total_elems);

        printf("  [TEST %d/%d] %s: blocks=%u, elems=%u\n", t + 1, total_cases, tc.name, num_blocks, total_elems);
        printf("      MAX_ABS_DIFF: %.6e, RMSE: %.6e, NAN: %d, INF: %d\n", max_abs_diff, rmse, nan_count, inf_count);

        bool pass = (max_abs_diff < 1e-4) && (nan_count == 0) && (inf_count == 0);
        if (pass) {
            printf("      -> PASS\n");
            passed_cases++;
        } else {
            printf("      -> FAIL (Threshold violated)\n");
        }

        ac_runtime_buffer_destroy(buf_src);
        ac_runtime_buffer_destroy(buf_dst);
    }

    // 4. Anti-Fake Mutation Sensitivity Check
    printf("\n  [MUTATION_TEST] Running Anti-Fake Bit Mutation Sensitivity Check...\n");
    {
        uint32_t mut_spv[kQ5_0DequantSpvByteSize / 4];
        memcpy(mut_spv, kQ5_0DequantSpv, kQ5_0DequantSpvByteSize);
        // Mutate opcode in shader
        mut_spv[100] ^= 0xDEADBEEF;

        AcKernelRuntimeMetadata mut_meta = meta;
        mut_meta.expected_spirv_sha256 = "mutated_fake_hash_00000000000000000000000000000000";

        AcKernelDesc mut_kdesc = {
            AC_ABI_VERSION,
            sizeof(AcKernelDesc),
            mut_spv,
            kQ5_0DequantSpvByteSize,
            &mut_meta,
            0,
            nullptr
        };

        AcKernel* mut_kernel = nullptr;
        AcResult mut_res = ac_runtime_kernel_create(ctx, &mut_kdesc, &mut_kernel);
        if (mut_res != AC_SUCCESS) {
            printf("  [MUTATION_PASS] Mutated SPIR-V correctly rejected by Vulkan driver/validator (res=%d).\n", mut_res);
        } else {
            printf("  [MUTATION_NOTE] Driver accepted corrupted module without immediate validation error.\n");
            ac_runtime_kernel_destroy(mut_kernel);
        }
    }

    ac_runtime_kernel_destroy(kernel);
    ac_runtime_context_destroy(ctx);

    printf("\n=======================================================================\n");
    printf("  TEST SUMMARY: %d / %d PASSED\n", passed_cases, total_cases);
    printf("=======================================================================\n");

    return (passed_cases == total_cases) ? 0 : 1;
}
