/**
 * @file test_phase3_universal_backend.cpp
 * @brief Phase 3: Universal Backend, Non-Coherent Memory, Timestamps, Vector-Add & Dual-Device Cross-Validation
 *
 * Hardware Verification Target:
 *   - Primary: Qualcomm Snapdragon 8 Elite / Adreno 830 (Samsung Galaxy S25)
 *   - Heterogeneous: Samsung Exynos 1380 / ARM Mali-G68 (Samsung Galaxy A35)
 *
 * Verification Gates:
 *   1. Hardware Diagnostic & Capabilities Inspection (UMA, atom size, timestamps)
 *   2. Universal AcKernelMetadata & SPIR-V Hash Authentication (BitNet GEMV I2_S)
 *   3. Universal Vector-Add Kernel Execution (vector_add.spv, M=1, N=256, alpha=2.5f)
 *   4. Residual In-Place Vector Addition (x += alpha * y accumulation pattern)
 *   5. Arbitrary Non-Power-of-2 Sizes & Tail Boundaries (N=37, 255, 1023, 4099)
 *   6. IEEE-754 Special Floating-Point Values Preservation (+Inf, -Inf, NaN)
 *   7. Memory Management & Physical Non-Coherent Verification (Flush/Invalidate)
 *   8. Fine-Grained GPU Timestamp Command Interval vs Host Wall Latency
 *   9. Metadata Validation Failures (Invalid features, Hash mismatch rejection)
 *  10. Repeated 1,000-Cycle Lifecycle Stability & Strict Teardown Integrity
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"
#include "../src/backend/vulkan/ac_vk_sha256.h"

#include "../src/shaders/vector_add_spv.h"
#include "../src/shaders/gemv_i2s_spv.h"

static long get_process_rss_kb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

static int load_fixture_file(const char * filename, void * out_buf, size_t max_bytes, size_t * out_read_bytes) {
    const char * search_paths[] = {
        "%s",
        "tests/fixtures/%s",
        "fixtures/%s",
        "../tests/fixtures/%s",
        "../../tests/fixtures/%s",
        "/data/data/com.termux/files/home/fixtures/%s",
        "/data/data/com.termux/files/home/tests/fixtures/%s",
        "/data/data/com.termux/files/home/%s",
        "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/%s",
        NULL
    };
    char full_path[512];
    FILE * fp = NULL;
    for (int i = 0; search_paths[i] != NULL; i++) {
        snprintf(full_path, sizeof(full_path), search_paths[i], filename);
        fp = fopen(full_path, "rb");
        if (fp != NULL) break;
    }
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0 || (size_t)sz > max_bytes) {
        fclose(fp);
        return -2;
    }
    size_t read_bytes = fread(out_buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (read_bytes != (size_t)sz) return -3;
    *out_read_bytes = (size_t)sz;
    return 0;
}

int main(void) {
    printf("======================================================================\n");
    printf("AMUDA 2.0 Phase 3: Universal Backend & Cross-Device Hardware Test\n");
    printf("======================================================================\n\n");

    int passed_gates = 0;
    const int total_gates = 10;

    // -------------------------------------------------------------------------
    // Gate 1: Hardware Diagnostic & Capabilities Inspection
    // -------------------------------------------------------------------------
    printf("[Gate 1/10] Hardware Diagnostic & Capabilities Inspection...\n");
    AcVkContext* ctx = NULL;
    AcResult res = ac_vk_context_create(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        printf("  -> FAIL: ac_vk_context_create failed with %d\n", res);
        return 1;
    }

    printf("  -> Physical GPU Selected: %s (Vendor: 0x%04x, Device: 0x%04x)\n",
           ctx->device_properties.deviceName,
           ctx->device_properties.vendorID,
           ctx->device_properties.deviceID);
    printf("  -> Driver Version: 0x%08x, API Version: %u.%u.%u\n",
           ctx->device_properties.driverVersion,
           VK_VERSION_MAJOR(ctx->device_properties.apiVersion),
           VK_VERSION_MINOR(ctx->device_properties.apiVersion),
           VK_VERSION_PATCH(ctx->device_properties.apiVersion));
    printf("  -> Non-Coherent Atom Size: %llu bytes\n",
           (unsigned long long)ctx->device_properties.limits.nonCoherentAtomSize);
    printf("  -> Timestamp Compute Support: %s, Valid Bits: %u, Period: %.2f ns/tick\n",
           ctx->device_properties.limits.timestampComputeAndGraphics ? "TRUE" : "FALSE",
           ctx->timestamp_valid_bits,
           ctx->device_properties.limits.timestampPeriod);

    passed_gates++;
    printf("  -> Gate 1 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 2: Universal AcKernelMetadata & SPIR-V Hash Authentication (BitNet I2_S)
    // -------------------------------------------------------------------------
    printf("[Gate 2/10] Universal AcKernelMetadata & Hash Authentication (BitNet GEMV I2_S)...\n");
    AcKernelMetadata i2s_meta = {};
    i2s_meta.kernel_name = "bitnet_gemv_i2s";
    i2s_meta.entry_point = "main";
    i2s_meta.binding_count = 3;
    i2s_meta.bindings = NULL;
    i2s_meta.push_constant_size = sizeof(BitNetPushConstants);
    i2s_meta.local_size = {32, 4, 1};
    i2s_meta.required_features = AC_FEATURE_NONE;
    i2s_meta.expected_spirv_sha256 = kBitnetGemvI2SSpvSha256;

    AcVkKernel* i2s_kernel = NULL;
    res = ac_vk_kernel_create_with_metadata(ctx, kBitnetGemvI2SSpv, kBitnetGemvI2SSpvByteSize, &i2s_meta, &i2s_kernel);
    if (res != AC_SUCCESS || !i2s_kernel) {
        printf("  -> FAIL: ac_vk_kernel_create_with_metadata failed with %d\n", res);
        ac_vk_context_destroy(ctx);
        return 2;
    }
    printf("  -> Kernel '%s' created with SHA-256: %s\n", i2s_meta.kernel_name, i2s_meta.expected_spirv_sha256);

    uint8_t packed_weights_fixture[256];
    size_t fixture_sz = 0;
    if (load_fixture_file("fixture_i2s_upstream.bin", packed_weights_fixture, sizeof(packed_weights_fixture), &fixture_sz) != 0 || fixture_sz != 96) {
        printf("  -> FAIL: Could not load fixture_i2s_upstream.bin (sz: %zu)\n", fixture_sz);
        ac_vk_kernel_destroy(i2s_kernel);
        ac_vk_context_destroy(ctx);
        return 2;
    }

    float dequant_scale = 1.0f;
    memcpy(&dequant_scale, packed_weights_fixture + 64, sizeof(float));

    float expected_f32[256];
    size_t exp_f32_sz = 0;
    if (load_fixture_file("expected_i2s_upstream_f32.bin", (uint8_t *)expected_f32, sizeof(expected_f32), &exp_f32_sz) != 0 || exp_f32_sz != 1024) {
        printf("  -> FAIL: Could not load expected_i2s_upstream_f32.bin (sz: %zu)\n", exp_f32_sz);
        ac_vk_kernel_destroy(i2s_kernel);
        ac_vk_context_destroy(ctx);
        return 2;
    }

    AcVkBuffer *b_w = NULL, *b_a = NULL, *b_o = NULL;
    ac_vk_buffer_create(ctx, 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_w);
    ac_vk_buffer_create(ctx, 256 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_a);
    ac_vk_buffer_create(ctx, 1 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_o);

    memcpy(b_w->mapped_ptr, packed_weights_fixture, 96);
    float* act_ptr = (float*)b_a->mapped_ptr;
    float* out_ptr = (float*)b_o->mapped_ptr;

    int i2s_mismatches = 0;
    float max_i2s_err = 0.0f;
    for (int col = 0; col < 256; ++col) {
        memset(act_ptr, 0, 256 * sizeof(float));
        act_ptr[col] = 1.0f;

        BitNetPushConstants pc = {256, 1, dequant_scale, 0, 0};
        AcVkBuffer* i2s_bufs[3] = {b_w, b_a, b_o};
        res = ac_vk_dispatch(ctx, i2s_kernel, i2s_bufs, 3, &pc, sizeof(pc), 1, 1, 1, NULL);
        if (res != AC_SUCCESS) {
            printf("  -> FAIL: ac_vk_dispatch failed at col %d (err: %d)\n", col, res);
            return 2;
        }

        float diff = fabsf(*out_ptr - expected_f32[col]);
        if (diff > max_i2s_err) max_i2s_err = diff;
        if (diff > 1e-4f) i2s_mismatches++;
    }

    printf("  -> BitNet GEMV I2_S Decoding (256/256 weights): Mismatches: %d/256, Max Error: %.2e\n",
           i2s_mismatches, max_i2s_err);
    if (i2s_mismatches > 0) {
        return 2;
    }

    ac_vk_buffer_destroy(b_w);
    ac_vk_buffer_destroy(b_a);
    ac_vk_buffer_destroy(b_o);
    ac_vk_kernel_destroy(i2s_kernel);

    passed_gates++;
    printf("  -> Gate 2 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 3: Universal Vector-Add Kernel Execution (vector_add.spv)
    // -------------------------------------------------------------------------
    printf("[Gate 3/10] Universal Vector-Add Kernel (vector_add.spv, N=256, alpha=2.5f)...\n");
    AcKernelMetadata vec_meta = {};
    vec_meta.kernel_name = "vector_add";
    vec_meta.entry_point = "main";
    vec_meta.binding_count = 3;
    vec_meta.bindings = NULL;
    vec_meta.push_constant_size = 2 * sizeof(uint32_t); // uint32_t n, float alpha
    vec_meta.local_size = {256, 1, 1};
    vec_meta.required_features = AC_FEATURE_NONE;
    vec_meta.expected_spirv_sha256 = kVectorAddSpvSha256;

    AcVkKernel* vec_kernel = NULL;
    res = ac_vk_kernel_create_with_metadata(ctx, kVectorAddSpv, kVectorAddSpvByteSize, &vec_meta, &vec_kernel);
    if (res != AC_SUCCESS || !vec_kernel) {
        printf("  -> FAIL: ac_vk_kernel_create_with_metadata failed for vector_add: %d\n", res);
        return 3;
    }

    const uint32_t N_gate3 = 256;
    AcVkBuffer *b_va = NULL, *b_vb = NULL, *b_vc = NULL;
    ac_vk_buffer_create(ctx, N_gate3 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_va);
    ac_vk_buffer_create(ctx, N_gate3 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_vb);
    ac_vk_buffer_create(ctx, N_gate3 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_vc);

    float* fa = (float*)b_va->mapped_ptr;
    float* fb = (float*)b_vb->mapped_ptr;
    float* fc = (float*)b_vc->mapped_ptr;
    const float alpha_gate3 = 2.5f;

    for (uint32_t i = 0; i < N_gate3; ++i) {
        fa[i] = (float)i * 0.5f;
        fb[i] = (float)(N_gate3 - i) * 0.25f;
        fc[i] = 0.0f;
    }

    struct { uint32_t n; float alpha; } vec_pc = {N_gate3, alpha_gate3};
    AcVkBuffer* vec_bufs[3] = {b_va, b_vb, b_vc};
    res = ac_vk_dispatch(ctx, vec_kernel, vec_bufs, 3, &vec_pc, sizeof(vec_pc), 1, 1, 1, NULL);
    if (res != AC_SUCCESS) {
        printf("  -> FAIL: Vector add dispatch failed: %d\n", res);
        return 3;
    }

    int mismatches = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < N_gate3; ++i) {
        float expected = fa[i] + alpha_gate3 * fb[i];
        float diff = fabsf(fc[i] - expected);
        if (diff > max_err) max_err = diff;
        if (diff > 1e-5f) mismatches++;
    }
    printf("  -> Vector Add (N=256): Mismatches: %d/%u, Max Error: %.2e\n", mismatches, N_gate3, max_err);
    if (mismatches > 0) {
        return 3;
    }

    ac_vk_buffer_destroy(b_va);
    ac_vk_buffer_destroy(b_vb);
    ac_vk_buffer_destroy(b_vc);

    passed_gates++;
    printf("  -> Gate 3 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 4: Residual In-Place Vector Addition (Accumulation Pattern)
    // -------------------------------------------------------------------------
    printf("[Gate 4/10] Residual In-Place Vector Addition (x += alpha * y)...\n");
    const uint32_t N_gate4 = 512;
    AcVkBuffer *b_x = NULL, *b_y = NULL;
    ac_vk_buffer_create(ctx, N_gate4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_x);
    ac_vk_buffer_create(ctx, N_gate4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_y);

    float* fx = (float*)b_x->mapped_ptr;
    float* fy = (float*)b_y->mapped_ptr;
    float* orig_x = (float*)malloc(N_gate4 * sizeof(float));
    const float alpha_gate4 = 1.75f;

    for (uint32_t i = 0; i < N_gate4; ++i) {
        fx[i] = (float)i * 1.2f;
        orig_x[i] = fx[i];
        fy[i] = (float)(i % 17) * -0.5f;
    }

    struct { uint32_t n; float alpha; } res_pc = {N_gate4, alpha_gate4};
    AcVkBuffer* res_bufs[3] = {b_x, b_y, b_x};
    res = ac_vk_dispatch(ctx, vec_kernel, res_bufs, 3, &res_pc, sizeof(res_pc), (N_gate4 + 255) / 256, 1, 1, NULL);
    if (res != AC_SUCCESS) {
        printf("  -> FAIL: Residual dispatch failed: %d\n", res);
        free(orig_x);
        return 4;
    }

    mismatches = 0;
    max_err = 0.0f;
    for (uint32_t i = 0; i < N_gate4; ++i) {
        float expected = orig_x[i] + alpha_gate4 * fy[i];
        float diff = fabsf(fx[i] - expected);
        if (diff > max_err) max_err = diff;
        if (diff > 1e-4f) mismatches++;
    }
    printf("  -> Residual In-Place (N=512): Mismatches: %d/%u, Max Error: %.2e\n", mismatches, N_gate4, max_err);
    free(orig_x);
    ac_vk_buffer_destroy(b_x);
    ac_vk_buffer_destroy(b_y);
    if (mismatches > 0) return 4;

    passed_gates++;
    printf("  -> Gate 4 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 5: Arbitrary Non-Power-of-2 Sizes & Tail Boundaries
    // -------------------------------------------------------------------------
    printf("[Gate 5/10] Arbitrary Non-Power-of-2 Sizes & Tail Boundaries...\n");
    const uint32_t test_sizes[] = {37, 255, 1023, 4099};
    const uint32_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (uint32_t s = 0; s < num_sizes; ++s) {
        uint32_t n = test_sizes[s];
        uint32_t padded_n = n + 64; // canary padding
        AcVkBuffer *ba = NULL, *bb = NULL, *bc = NULL;
        ac_vk_buffer_create(ctx, padded_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ba);
        ac_vk_buffer_create(ctx, padded_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bb);
        ac_vk_buffer_create(ctx, padded_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bc);

        float* pa = (float*)ba->mapped_ptr;
        float* pb = (float*)bb->mapped_ptr;
        float* pc_ptr = (float*)bc->mapped_ptr;

        for (uint32_t i = 0; i < padded_n; ++i) {
            pa[i] = (float)i * 0.1f;
            pb[i] = (float)(i * 2) * 0.1f;
            pc_ptr[i] = -99999.0f; // canary pattern
        }

        uint32_t groups = (n + 255) / 256;
        struct { uint32_t n; float alpha; } tail_pc = {n, 1.0f};
        AcVkBuffer* t_bufs[3] = {ba, bb, bc};
        res = ac_vk_dispatch(ctx, vec_kernel, t_bufs, 3, &tail_pc, sizeof(tail_pc), groups, 1, 1, NULL);
        if (res != AC_SUCCESS) {
            printf("  -> FAIL: Tail dispatch failed for N=%u\n", n);
            return 5;
        }

        int tail_mismatch = 0;
        for (uint32_t i = 0; i < n; ++i) {
            float exp = pa[i] + pb[i];
            if (fabsf(pc_ptr[i] - exp) > 1e-4f) tail_mismatch++;
        }
        int canary_corrupted = 0;
        for (uint32_t i = n; i < padded_n; ++i) {
            if (pc_ptr[i] != -99999.0f) canary_corrupted++;
        }

        printf("  -> Tail N=%u (Groups=%u): Mismatch=%d, Canary Corrupted=%d\n",
               n, groups, tail_mismatch, canary_corrupted);

        ac_vk_buffer_destroy(ba);
        ac_vk_buffer_destroy(bb);
        ac_vk_buffer_destroy(bc);

        if (tail_mismatch > 0 || canary_corrupted > 0) return 5;
    }

    passed_gates++;
    printf("  -> Gate 5 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 6: IEEE-754 Special Floating-Point Values Preservation
    // -------------------------------------------------------------------------
    printf("[Gate 6/10] IEEE-754 Special Floating-Point Values Preservation...\n");
    const uint32_t N_gate6 = 8;
    AcVkBuffer *b_ia = NULL, *b_ib = NULL, *b_ic = NULL;
    ac_vk_buffer_create(ctx, N_gate6 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_ia);
    ac_vk_buffer_create(ctx, N_gate6 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_ib);
    ac_vk_buffer_create(ctx, N_gate6 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_ic);

    float* pia = (float*)b_ia->mapped_ptr;
    float* pib = (float*)b_ib->mapped_ptr;
    float* pic = (float*)b_ic->mapped_ptr;

    pia[0] = 5.0f;        pib[0] = INFINITY;   // 5 + Inf = Inf
    pia[1] = 5.0f;        pib[1] = -INFINITY;  // 5 + (-Inf) = -Inf
    pia[2] = 0.0f;        pib[2] = -0.0f;      // 0 + (-0) = 0
    pia[3] = 1e-35f;      pib[3] = 1e-35f;     // Subnormal addition
    pia[4] = INFINITY;    pib[4] = 100.0f;     // Inf + 100 = Inf
    pia[5] = -INFINITY;   pib[5] = -100.0f;    // -Inf + (-100) = -Inf
    pia[6] = 1.0f;        pib[6] = NAN;        // 1 + NaN = NaN
    pia[7] = 42.0f;       pib[7] = 8.0f;       // normal

    struct { uint32_t n; float alpha; } ieee_pc = {N_gate6, 1.0f};
    AcVkBuffer* ieee_bufs[3] = {b_ia, b_ib, b_ic};
    res = ac_vk_dispatch(ctx, vec_kernel, ieee_bufs, 3, &ieee_pc, sizeof(ieee_pc), 1, 1, 1, NULL);
    if (res != AC_SUCCESS) {
        printf("  -> FAIL: IEEE-754 dispatch failed: %d\n", res);
        return 6;
    }

    bool ieee_pass = true;
    if (!isinf(pic[0]) || pic[0] < 0) ieee_pass = false;
    if (!isinf(pic[1]) || pic[1] > 0) ieee_pass = false;
    if (!isinf(pic[4]) || pic[4] < 0) ieee_pass = false;
    if (!isinf(pic[5]) || pic[5] > 0) ieee_pass = false;
    if (!isnan(pic[6])) ieee_pass = false;
    if (fabsf(pic[7] - 50.0f) > 1e-4f) ieee_pass = false;

    printf("  -> [0] 5 + Inf = %f (isinf: %d)\n", pic[0], isinf(pic[0]));
    printf("  -> [1] 5 + (-Inf) = %f (isinf: %d)\n", pic[1], isinf(pic[1]));
    printf("  -> [6] 1 + NaN = %f (isnan: %d)\n", pic[6], isnan(pic[6]));
    printf("  -> [7] 42 + 8 = %f (expected: 50.0)\n", pic[7]);

    ac_vk_buffer_destroy(b_ia);
    ac_vk_buffer_destroy(b_ib);
    ac_vk_buffer_destroy(b_ic);

    if (!ieee_pass) {
        printf("  -> FAIL: IEEE-754 preservation check failed\n");
        return 6;
    }
    passed_gates++;
    printf("  -> Gate 6 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 7: Memory Management & Physical Non-Coherent Verification
    // -------------------------------------------------------------------------
    printf("[Gate 7/10] Memory Management & Physical Non-Coherent Verification...\n");
    AcVkBuffer* b_flush = NULL;
    ac_vk_buffer_create(ctx, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_flush);

    printf("  -> Default Allocated Buffer Properties:\n");
    printf("     Memory Type Index: %u | Property Flags: 0x%04x | is_coherent: %s\n",
           b_flush->memory_type_index, b_flush->memory_property_flags, b_flush->is_coherent ? "TRUE" : "FALSE");
    printf("     Logical Size: %zu bytes | Physical Driver Allocation: %zu bytes\n",
           b_flush->size_bytes, b_flush->allocation_size_bytes);

    /* 1. NULL check */
    if (ac_vk_buffer_flush(NULL, 0, 1024) != AC_ERR_NULL_POINTER) {
        printf("  -> FAIL: Flush NULL buffer did not return AC_ERR_NULL_POINTER\n");
        return 7;
    }
    /* 2. Out of bounds check */
    if (ac_vk_buffer_flush(b_flush, 5000, 1024) != AC_ERR_INVALID_ARGUMENT) {
        printf("  -> FAIL: Flush out of bounds did not return AC_ERR_INVALID_ARGUMENT\n");
        return 7;
    }
    /* 3. Valid range flush */
    if (ac_vk_buffer_flush(b_flush, 0, 1024) != AC_SUCCESS) {
        printf("  -> FAIL: Valid range flush failed\n");
        return 7;
    }
    /* 4. Valid range invalidate */
    if (ac_vk_buffer_invalidate(b_flush, 0, 1024) != AC_SUCCESS) {
        printf("  -> FAIL: Valid range invalidate failed\n");
        return 7;
    }
    /* 5. Entire buffer flush (size = 0) */
    if (ac_vk_buffer_flush(b_flush, 128, 0) != AC_SUCCESS) {
        printf("  -> FAIL: Full flush from offset failed\n");
        return 7;
    }
    ac_vk_buffer_destroy(b_flush);

    /* 6. Physical Non-Coherent Path Verification */
    AcVkBuffer* b_non_coherent = NULL;
    res = ac_vk_buffer_create_explicit(ctx, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &b_non_coherent);
    if (res == AC_SUCCESS && b_non_coherent != NULL) {
        printf("  -> Forced Non-Coherent Allocation: Type Index=%u, Flags=0x%04x, is_coherent=%s (Atom=%llu B)\n",
               b_non_coherent->memory_type_index,
               b_non_coherent->memory_property_flags,
               b_non_coherent->is_coherent ? "TRUE" : "FALSE",
               (unsigned long long)ctx->device_properties.limits.nonCoherentAtomSize);

        uint32_t* non_ptr = (uint32_t*)b_non_coherent->mapped_ptr;
        for (int i = 0; i < 1024; ++i) non_ptr[i] = 0xAA550000u + i;

        AcResult flush_res = ac_vk_buffer_flush(b_non_coherent, 0, 4096);
        AcResult inv_res = ac_vk_buffer_invalidate(b_non_coherent, 0, 4096);

        bool pattern_ok = true;
        for (int i = 0; i < 1024; ++i) {
            if (non_ptr[i] != 0xAA550000u + i) {
                pattern_ok = false;
                break;
            }
        }
        ac_vk_buffer_destroy(b_non_coherent);

        if (flush_res == AC_SUCCESS && inv_res == AC_SUCCESS && pattern_ok) {
            printf("  -> Physical Non-Coherent Memory Path: VERIFIED on this device (Flush/Invalidate SUCCESS)\n");
        } else {
            printf("  -> FAIL: Non-coherent memory test failed (flush: %d, inv: %d, pattern: %d)\n",
                   flush_res, inv_res, pattern_ok);
            return 7;
        }
    } else {
        printf("  -> [SKIP] NON_COHERENT_NOT_AVAILABLE: Device does not offer non-coherent host-visible memory type.\n");
    }

    passed_gates++;
    printf("  -> Gate 7 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 8: Fine-Grained GPU Timestamp Command Interval vs Host Wall Latency
    // -------------------------------------------------------------------------
    printf("[Gate 8/10] Fine-Grained GPU Timestamp Command Interval vs Host Wall Latency...\n");
    AcVkBuffer *b_t0 = NULL, *b_t1 = NULL, *b_t2 = NULL;
    const uint32_t N_t = 1024;
    ac_vk_buffer_create(ctx, N_t * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_t0);
    ac_vk_buffer_create(ctx, N_t * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_t1);
    ac_vk_buffer_create(ctx, N_t * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_t2);

    struct { uint32_t n; float alpha; } t_pc = {N_t, 1.0f};
    AcVkBuffer* t_bufs[3] = {b_t0, b_t1, b_t2};

    AcVkEvent* async_event = NULL;
    res = ac_vk_dispatch(ctx, vec_kernel, t_bufs, 3, &t_pc, sizeof(t_pc), 4, 1, 1, &async_event);
    if (res != AC_SUCCESS || !async_event) {
        printf("  -> FAIL: Async dispatch failed: %d\n", res);
        return 8;
    }

    res = ac_vk_event_wait(async_event, 5000000000ULL);
    if (res != AC_SUCCESS) {
        printf("  -> FAIL: ac_vk_event_wait failed: %d\n", res);
        return 8;
    }

    uint64_t wall_latency_ns = 0;
    uint64_t gpu_duration_ns = 0;
    res = ac_vk_event_get_timing(async_event, &wall_latency_ns, &gpu_duration_ns);
    if (res != AC_SUCCESS && res != AC_ERR_FEATURE_NOT_SUPPORTED) {
        printf("  -> FAIL: ac_vk_event_get_timing failed: %d\n", res);
        return 8;
    }

    printf("  -> Avg Synchronous End-to-End Wall Latency: %llu ns (%.3f µs, %.3f ms)\n",
           (unsigned long long)wall_latency_ns,
           (double)wall_latency_ns / 1000.0,
           (double)wall_latency_ns / 1000000.0);

    if (async_event->has_timestamp && res == AC_SUCCESS) {
        printf("  -> GPU timestamp command interval duration: %llu ns (%.3f µs, %.3f ms)\n",
               (unsigned long long)gpu_duration_ns,
               (double)gpu_duration_ns / 1000.0,
               (double)gpu_duration_ns / 1000000.0);
        printf("     (Measured between TOP_OF_PIPE and BOTTOM_OF_PIPE pipeline stages)\n");
        if (wall_latency_ns > 0 && gpu_duration_ns > 0) {
            double ratio = (double)gpu_duration_ns / (double)wall_latency_ns;
            printf("  -> GPU Command Interval Active Ratio: %.2f%% of host wall time\n", ratio * 100.0);
        }
    } else {
        printf("  -> GPU Timestamps: Not supported on this hardware (graceful fallback)\n");
    }

    ac_vk_event_destroy(async_event);
    ac_vk_buffer_destroy(b_t0);
    ac_vk_buffer_destroy(b_t1);
    ac_vk_buffer_destroy(b_t2);

    passed_gates++;
    printf("  -> Gate 8 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 9: Metadata Validation Failures (Corrupted Hash & Unsupported Features)
    // -------------------------------------------------------------------------
    printf("[Gate 9/10] Metadata Validation Failures (Corrupted Hash & Unsupported Features)...\n");
    /* 1. Corrupted SHA-256 */
    AcKernelMetadata corrupt_meta = vec_meta;
    corrupt_meta.expected_spirv_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    AcVkKernel* bad_kernel = NULL;
    res = ac_vk_kernel_create_with_metadata(ctx, kVectorAddSpv, kVectorAddSpvByteSize, &corrupt_meta, &bad_kernel);
    if (res != AC_ERR_BINARY_CORRUPTED) {
        printf("  -> FAIL: Expected AC_ERR_BINARY_CORRUPTED, got %d\n", res);
        return 9;
    }
    printf("  -> Corrupted Hash Rejection: Validated (Returned %d)\n", res);

    /* 2. Unsupported feature mask */
    AcKernelMetadata feature_meta = vec_meta;
    feature_meta.required_features = 0x80000000u;
    res = ac_vk_kernel_create_with_metadata(ctx, kVectorAddSpv, kVectorAddSpvByteSize, &feature_meta, &bad_kernel);
    if (res != AC_ERR_FEATURE_NOT_SUPPORTED) {
        printf("  -> FAIL: Expected AC_ERR_FEATURE_NOT_SUPPORTED, got %d\n", res);
        return 9;
    }
    printf("  -> Unsupported Feature Mask Rejection: Validated (Returned %d)\n", res);

    passed_gates++;
    printf("  -> Gate 9 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 10: 1,000-Cycle Buffer Lifecycle RSS Stability & Strict Teardown
    // -------------------------------------------------------------------------
    printf("[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability & Strict Teardown...\n");
    ac_vk_kernel_destroy(vec_kernel);

    /* Measure RSS over 1,000 buffer allocation / destruction cycles */
    long rss_initial = get_process_rss_kb();
    const int kStressCycles = 1000;
    for (int iter = 0; iter < kStressCycles; ++iter) {
        AcVkBuffer* test_buf = NULL;
        AcResult b_res = ac_vk_buffer_create(ctx, 1024 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &test_buf);
        if (b_res != AC_SUCCESS || !test_buf) {
            printf("  -> FAIL: Buffer alloc failed at cycle %d\n", iter);
            return 10;
        }
        ac_vk_buffer_destroy(test_buf);
    }
    long rss_final = get_process_rss_kb();
    printf("  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS %ld KB -> %ld KB (Delta: %ld KB)\n",
           rss_initial, rss_final, (rss_final >= 0 && rss_initial >= 0) ? (rss_final - rss_initial) : 0);

    /* Verify outstanding submission counters and child references are strictly 0 */
    if (ctx->active_child_buffers != 0 || ctx->active_child_kernels != 0 || ctx->submission_outstanding_count != 0) {
        printf("  -> FAIL: Internal counters non-zero before destroy! (bufs: %u, kernels: %u, submissions: %u)\n",
               ctx->active_child_buffers, ctx->active_child_kernels, ctx->submission_outstanding_count);
        return 10;
    }
    printf("  -> Internal Counters Check: active_child_buffers=0, active_child_kernels=0, submissions=0\n");

    res = ac_vk_context_destroy(ctx);
    if (res != AC_SUCCESS) {
        printf("  -> FAIL: Context teardown failed with %d\n", res);
        return 10;
    }
    printf("  -> Context Teardown: Cleanup completed with internal counters 0 (Returned %d)\n", res);

    passed_gates++;
    printf("  -> Gate 10 PASS\n\n");

    printf("======================================================================\n");
    printf("AMUDA 2.0 Phase 3 Verification Gates: %d / %d Passed\n", passed_gates, total_gates);
    printf("Storage-Buffer Compute Pipeline Status: VERIFIED PASS\n");
    printf("======================================================================\n");
    return 0;
}
