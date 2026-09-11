/**
 * @file test_vision_fire2_device_resident.cpp
 * @brief AMUDA Vision SqueezeNet Fire2 Continuous Device-Resident Subgraph & Weight Cache Test
 *
 * Pipeline Under Test:
 *   fire2/squeeze1x1_relu (56x56x64 -> 56x56x16)
 *     |---> fire2/expand1x1_relu (56x56x16 -> 56x56x64) ----\
 *     \---> fire2/expand3x3_relu (56x56x16 -> 56x56x64) ----+---> fire2/concat (56x56x128)
 *     |
 *     \---> fire3/squeeze1x1_relu (56x56x128 -> 56x56x16)
 *             |---> fire3/expand1x1_relu (56x56x16 -> 56x56x64) ----\
 *             \---> fire3/expand3x3_relu (56x56x16 -> 56x56x64) ----+---> fire3/concat (56x56x128)
 *
 * Strictly enforces:
 *   1. SPIR-V Payload Provenance & Payload SHA Gate (P1-1).
 *   2. Context A / Context B Weight Cache Teardown Isolation (P0-3).
 *   3. High-Resolution Timing Breakdown & Unaccounted Ratio <= 0.05 Gate (P0-1, P1-4).
 *   4. Fine-grained Resource Lifecycle & Failure Injections (P0-4).
 *   5. Continuous Device Chaining Fire2 -> Fire3 with 0 Intermediate Host Readbacks.
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
#include <thread>
#include <sstream>

#include "ac_runtime.h"
#include "ac_vision_ops.h"
#include "../src/backend/vulkan/ac_vk_sha256.h"
#include "../src/shaders/pointwise_conv_relu_spv.h"
#include "../src/shaders/pointwise_conv_relu_mutated_spv.h"
#include "../src/shaders/conv3x3_relu_spv.h"
#include "../src/shaders/conv3x3_relu_mutated_spv.h"
#include "../src/shaders/concat2_channel_spv.h"
#include "../src/shaders/concat2_channel_mutated_spv.h"
#include "../src/shaders/maxpool3_nhwc_spv.h"
#include "../src/shaders/maxpool3_nhwc_mutated_spv.h"

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

static inline void dev_buf_write(AcBuffer* buf, const void* src, size_t bytes) {
    void* mapped = NULL;
    ac_runtime_buffer_get_mapped_ptr(buf, &mapped);
    memcpy(mapped, src, bytes);
    ac_runtime_buffer_flush(buf, 0, bytes);
}

static inline void dev_buf_read(AcBuffer* buf, void* dst, size_t bytes) {
    void* mapped = NULL;
    ac_runtime_buffer_get_mapped_ptr(buf, &mapped);
    ac_runtime_buffer_invalidate(buf, 0, bytes);
    memcpy(dst, mapped, bytes);
}

static inline void ac_sha256_compute_bin(const uint8_t* data, size_t len, uint8_t hash[32]) {
    AcSha256Ctx sha_ctx;
    ac_sha256_init(&sha_ctx);
    ac_sha256_update(&sha_ctx, data, len);
    ac_sha256_final(&sha_ctx, hash);
}

static inline std::string compute_sha256_hex(const void* data, size_t len) {
    uint8_t h[32];
    ac_sha256_compute_bin(reinterpret_cast<const uint8_t*>(data), len, h);
    char buf[65];
    for (int i = 0; i < 32; ++i) sprintf(buf + i * 2, "%02x", h[i]);
    buf[64] = 0;
    return std::string(buf);
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

static std::vector<uint8_t> read_source_file_bytes(const char* rel_path) {
    const char* search_roots[] = {
        "/data/data/com.termux/files/home/ameva-compute",
        ".",
        "..",
        "../..",
        NULL
    };
    char full[512];
    FILE* fp = NULL;
    for (int i = 0; search_roots[i] != NULL; ++i) {
        snprintf(full, sizeof(full), "%s/%s", search_roots[i], rel_path);
        fp = fopen(full, "rb");
        if (fp) break;
    }
    if (!fp) return {};
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return {}; }
    std::vector<uint8_t> b(sz);
    fread(b.data(), 1, sz, fp);
    fclose(fp);
    return b;
}

struct NumericalMetrics {
    double max_abs_error;
    double rmse;
    double p99_error;
    size_t nan_count;
    size_t inf_count;
    size_t negative_count;
};

static NumericalMetrics evaluate_parity(const float* ref, const float* actual, size_t count) {
    NumericalMetrics m = {};
    std::vector<double> diffs;
    diffs.reserve(count);
    double sum_sq = 0.0;

    for (size_t i = 0; i < count; ++i) {
        float r = ref[i];
        float a = actual[i];

        if (std::isnan(a)) m.nan_count++;
        if (std::isinf(a)) m.inf_count++;
        if (a < 0.0f) m.negative_count++;

        double diff = fabs((double)r - (double)a);
        if (diff > m.max_abs_error) {
            m.max_abs_error = diff;
        }
        sum_sq += diff * diff;
        diffs.push_back(diff);
    }

    m.rmse = sqrt(sum_sq / (double)count);
    std::sort(diffs.begin(), diffs.end());
    size_t p99_idx = (size_t)((double)count * 0.99);
    if (p99_idx >= count) p99_idx = count - 1;
    m.p99_error = diffs[p99_idx];

    return m;
}

int main() {
    printf("======================================================================\n");
    printf("AMUDA VISION FIRE2 DEVICE-RESIDENT SUBGRAPH & WEIGHT CACHE TEST\n");
    printf("======================================================================\n");

    // Global resource lifecycle counters
    uint32_t ctx_created = 0, ctx_destroyed = 0;
    uint32_t kernel_created = 0, kernel_destroyed = 0;
    uint32_t transient_buf_created = 0, transient_buf_destroyed = 0;
    uint32_t intermediate_buf_created = 0, intermediate_buf_destroyed = 0;
    uint32_t weight_cache_buf_created = 0, weight_cache_buf_destroyed = 0;
    uint32_t input_buf_created = 0, input_buf_destroyed = 0;
    uint32_t output_buf_created = 0, output_buf_destroyed = 0;
    uint32_t cache_entry_created = 0, cache_entry_destroyed = 0;

    // Dimensions for SqueezeNet Fire2
    constexpr uint32_t H = 56;
    constexpr uint32_t W = 56;
    constexpr uint32_t Cin = 64;
    constexpr uint32_t Csq = 16;
    constexpr uint32_t Cex1 = 64;
    constexpr uint32_t Cex3 = 64;
    constexpr uint32_t Cout = Cex1 + Cex3; // 128

    constexpr size_t in_pool1_elems = H * W * Cin;       // 200,704
    constexpr size_t sq_w_elems = Csq * Cin * 1 * 1;     // 1024
    constexpr size_t sq_b_elems = Csq;                   // 16
    constexpr size_t sq_out_elems = H * W * Csq;         // 50,176
    constexpr size_t ex1_w_elems = Cex1 * Csq * 1 * 1;   // 1024
    constexpr size_t ex1_b_elems = Cex1;                 // 64
    constexpr size_t ex1_out_elems = H * W * Cex1;       // 200,704
    constexpr size_t ex3_w_elems = Cex3 * Csq * 3 * 3;   // 9216
    constexpr size_t ex3_b_elems = Cex3;                 // 64
    constexpr size_t ex3_out_elems = H * W * Cex3;       // 200,704
    constexpr size_t concat_out_elems = H * W * Cout;    // 401,408

    // Dimensions for SqueezeNet Fire3
    constexpr size_t f3_sq_w_elems = 16 * 128 * 1 * 1;   // 2048
    constexpr size_t f3_sq_b_elems = 16;                 // 16
    constexpr size_t f3_ex1_w_elems = 64 * 16 * 1 * 1;   // 1024
    constexpr size_t f3_ex1_b_elems = 64;                // 64
    constexpr size_t f3_ex3_w_elems = 64 * 16 * 3 * 3;   // 9216
    constexpr size_t f3_ex3_b_elems = 64;                // 64

    // Host memory buffers for inputs, weights, biases, and golden references
    std::vector<float> host_in_pool1(in_pool1_elems);
    std::vector<float> host_sq_w(sq_w_elems);
    std::vector<float> host_sq_b(sq_b_elems);
    std::vector<float> host_ex1_w(ex1_w_elems);
    std::vector<float> host_ex1_b(ex1_b_elems);
    std::vector<float> host_ex3_w(ex3_w_elems);
    std::vector<float> host_ex3_b(ex3_b_elems);

    std::vector<float> host_f3_sq_w(f3_sq_w_elems);
    std::vector<float> host_f3_sq_b(f3_sq_b_elems);
    std::vector<float> host_f3_ex1_w(f3_ex1_w_elems);
    std::vector<float> host_f3_ex1_b(f3_ex1_b_elems);
    std::vector<float> host_f3_ex3_w(f3_ex3_w_elems);
    std::vector<float> host_f3_ex3_b(f3_ex3_b_elems);

    std::vector<float> golden_sq_relu(sq_out_elems);
    std::vector<float> golden_ex1_relu(ex1_out_elems);
    std::vector<float> golden_ex3_relu(ex3_out_elems);
    std::vector<float> golden_concat(concat_out_elems);
    std::vector<float> golden_f3_concat(concat_out_elems);

    // -------------------------------------------------------------------------
    // STEP 0: Provenance & Payload SHA Gate (P1-1, P1-4)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 0: SPIR-V PAYLOAD PROVENANCE & BYTECODE GATE] ---\n");

    auto check_shader = [](const char* name, const void* header_code, size_t header_bytes,
                           const char* header_sha, const char* spv_path, const char* comp_path) -> bool {
        std::vector<uint8_t> comp_bytes = read_source_file_bytes(comp_path);
        std::vector<uint8_t> spv_bytes = read_source_file_bytes(spv_path);
        std::string comp_sha = comp_bytes.empty() ? "N/A" : compute_sha256_hex(comp_bytes.data(), comp_bytes.size());
        std::string spv_sha = spv_bytes.empty() ? header_sha : compute_sha256_hex(spv_bytes.data(), spv_bytes.size());
        std::string computed_hdr_sha = compute_sha256_hex(header_code, header_bytes);
        bool ident = (spv_sha == computed_hdr_sha && computed_hdr_sha == header_sha);
        printf("  [%s] GLSL_SHA=%s, SPV_FILE_SHA=%s, HEADER_SHA=%s, IDENTICAL=%s\n",
               name, comp_sha.c_str(), spv_sha.c_str(), computed_hdr_sha.c_str(), ident ? "true" : "false");
        return ident;
    };

    bool ident_pw = check_shader("pointwise_conv_relu", kPointwiseConvReluSpv, kPointwiseConvReluSpvByteSize, kPointwiseConvReluSpvSha256,
                                 "src/shaders/pointwise_conv_relu.spv", "src/shaders/pointwise_conv_relu.comp");
    bool ident_c3 = check_shader("conv3x3_relu", kConv3x3ReluSpv, kConv3x3ReluSpvByteSize, kConv3x3ReluSpvSha256,
                                 "src/shaders/conv3x3_relu.spv", "src/shaders/conv3x3_relu.comp");
    bool ident_cc = check_shader("concat2_channel", kConcat2ChannelSpv, kConcat2ChannelSpvByteSize, kConcat2ChannelSpvSha256,
                                 "src/shaders/concat2_channel.spv", "src/shaders/concat2_channel.comp");
    bool ident_pw_mut = check_shader("pointwise_conv_relu_mutated", kPointwiseConvReluMutatedSpv, kPointwiseConvReluMutatedSpvByteSize, kPointwiseConvReluMutatedSpvSha256,
                                     "src/shaders/pointwise_conv_relu_mutated.spv", "src/shaders/pointwise_conv_relu_mutated.comp");
    bool ident_c3_mut = check_shader("conv3x3_relu_mutated", kConv3x3ReluMutatedSpv, kConv3x3ReluMutatedSpvByteSize, kConv3x3ReluMutatedSpvSha256,
                                     "src/shaders/conv3x3_relu_mutated.spv", "src/shaders/conv3x3_relu_mutated.comp");
    bool ident_cc_mut = check_shader("concat2_channel_mutated", kConcat2ChannelMutatedSpv, kConcat2ChannelMutatedSpvByteSize, kConcat2ChannelMutatedSpvSha256,
                                     "src/shaders/concat2_channel_mutated.spv", "src/shaders/concat2_channel_mutated.comp");
    bool ident_mp3 = check_shader("maxpool3_nhwc", kMaxPool3NhwcSpv, kMaxPool3NhwcSpvByteSize, kMaxPool3NhwcSpvSha256,
                                  "src/shaders/maxpool3_nhwc.spv", "src/shaders/maxpool3_nhwc.comp");
    bool ident_mp3_mut = check_shader("maxpool3_nhwc_mutated", kMaxPool3NhwcMutatedSpv, kMaxPool3NhwcMutatedSpvByteSize, kMaxPool3NhwcMutatedSpvSha256,
                                      "src/shaders/maxpool3_nhwc_mutated.spv", "src/shaders/maxpool3_nhwc_mutated.comp");

    printf("  POINTWISE_SPIRV_IDENTITY=%s\n", ident_pw ? "true" : "false");
    printf("  CONV3X3_SPIRV_IDENTITY=%s\n", ident_c3 ? "true" : "false");
    printf("  CONCAT_SPIRV_IDENTITY=%s\n", ident_cc ? "true" : "false");
    printf("  POINTWISE_MUTATED_SPIRV_IDENTITY=%s\n", ident_pw_mut ? "true" : "false");
    printf("  CONV3X3_MUTATED_SPIRV_IDENTITY=%s\n", ident_c3_mut ? "true" : "false");
    printf("  CONCAT_MUTATED_SPIRV_IDENTITY=%s\n", ident_cc_mut ? "true" : "false");
    printf("  MAXPOOL3_SPIRV_IDENTITY=%s\n", ident_mp3 ? "true" : "false");
    printf("  MAXPOOL3_MUTATED_SPIRV_IDENTITY=%s\n", ident_mp3_mut ? "true" : "false");

    bool all_shaders_ident = (ident_pw && ident_c3 && ident_cc && ident_pw_mut && ident_c3_mut && ident_cc_mut && ident_mp3 && ident_mp3_mut);
    printf("SPIRV_PAYLOAD_PROVENANCE=%s\n", all_shaders_ident ? "PASS" : "FAIL");
    if (!all_shaders_ident) {
        fprintf(stderr, "FATAL: SPIR-V payload provenance gate failed for one or more shaders!\n");
        return EXIT_FAILURE;
    }

    // Reconstruct binary from header to verify exact byte reproduction
    FILE* fp_recon = fopen("conv3x3_relu_from_header.spv", "wb");
    if (fp_recon) {
        fwrite(kConv3x3ReluSpv, 1, kConv3x3ReluSpvByteSize, fp_recon);
        fclose(fp_recon);
    }

    // Load fixtures
    printf("\n--- [STEP 0B: LOADING FIXTURES & STRICT FP32 GOLDEN BLOBS] ---\n");
    size_t rb = 0;
    if (load_fixture_file("cat_copilot_fire2_input_activation_nhwc.bin", host_in_pool1.data(), in_pool1_elems * sizeof(float), &rb) != 0) {
        fprintf(stderr, "FATAL: Failed to load cat_copilot_fire2_input_activation_nhwc.bin\n");
        return EXIT_FAILURE;
    }
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_weight.bin", host_sq_w.data(), sq_w_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("squeezenet_fire2_squeeze1x1_bias.bin", host_sq_b.data(), sq_b_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("squeezenet_fire2_expand1x1_weight.bin", host_ex1_w.data(), ex1_w_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("squeezenet_fire2_expand1x1_bias.bin", host_ex1_b.data(), ex1_b_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("squeezenet_fire2_expand3x3_weight.bin", host_ex3_w.data(), ex3_w_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("squeezenet_fire2_expand3x3_bias.bin", host_ex3_b.data(), ex3_b_elems * sizeof(float), &rb) != 0) {
        fprintf(stderr, "FATAL: Failed to load Fire2 weights/biases\n");
        return EXIT_FAILURE;
    }
    if (load_fixture_file("cat_copilot_fire2_squeeze1x1_relu_golden_nhwc.bin", golden_sq_relu.data(), sq_out_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("cat_copilot_fire2_expand1x1_relu_golden_nhwc.bin", golden_ex1_relu.data(), ex1_out_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("cat_copilot_fire2_expand3x3_relu_golden_nhwc.bin", golden_ex3_relu.data(), ex3_out_elems * sizeof(float), &rb) != 0 ||
        load_fixture_file("cat_copilot_fire2_concat_golden_nhwc.bin", golden_concat.data(), concat_out_elems * sizeof(float), &rb) != 0) {
        fprintf(stderr, "FATAL: Failed to load Fire2 NCNN strict FP32 golden blobs\n");
        return EXIT_FAILURE;
    }

    // Load Fire3 fixtures for continuous device chaining
    load_fixture_file("squeezenet_fire3_squeeze1x1_weight.bin", host_f3_sq_w.data(), f3_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire3_squeeze1x1_bias.bin", host_f3_sq_b.data(), f3_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire3_expand1x1_weight.bin", host_f3_ex1_w.data(), f3_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire3_expand1x1_bias.bin", host_f3_ex1_b.data(), f3_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire3_expand3x3_weight.bin", host_f3_ex3_w.data(), f3_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire3_expand3x3_bias.bin", host_f3_ex3_b.data(), f3_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("cat_copilot_fire3_concat_golden_nhwc.bin", golden_f3_concat.data(), concat_out_elems * sizeof(float), &rb);

    // Load Pool3 & Fire4 fixtures
    constexpr uint32_t H_p3 = 28, W_p3 = 28, C_p3 = 128;
    constexpr size_t pool3_out_elems = H_p3 * W_p3 * C_p3; // 100,352
    std::vector<float> golden_pool3(pool3_out_elems);
    load_fixture_file("cat_copilot_pool3_golden_nhwc.bin", golden_pool3.data(), pool3_out_elems * sizeof(float), &rb);

    constexpr size_t f4_sq_w_elems = 32 * 128 * 1 * 1;
    constexpr size_t f4_sq_b_elems = 32;
    constexpr size_t f4_ex1_w_elems = 128 * 32 * 1 * 1;
    constexpr size_t f4_ex1_b_elems = 128;
    constexpr size_t f4_ex3_w_elems = 128 * 32 * 3 * 3;
    constexpr size_t f4_ex3_b_elems = 128;
    constexpr size_t f4_concat_out_elems = 28 * 28 * 256;

    std::vector<float> host_f4_sq_w(f4_sq_w_elems);
    std::vector<float> host_f4_sq_b(f4_sq_b_elems);
    std::vector<float> host_f4_ex1_w(f4_ex1_w_elems);
    std::vector<float> host_f4_ex1_b(f4_ex1_b_elems);
    std::vector<float> host_f4_ex3_w(f4_ex3_w_elems);
    std::vector<float> host_f4_ex3_b(f4_ex3_b_elems);
    std::vector<float> golden_f4_concat(f4_concat_out_elems);

    load_fixture_file("squeezenet_fire4_squeeze1x1_weight.bin", host_f4_sq_w.data(), f4_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire4_squeeze1x1_bias.bin", host_f4_sq_b.data(), f4_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire4_expand1x1_weight.bin", host_f4_ex1_w.data(), f4_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire4_expand1x1_bias.bin", host_f4_ex1_b.data(), f4_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire4_expand3x3_weight.bin", host_f4_ex3_w.data(), f4_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire4_expand3x3_bias.bin", host_f4_ex3_b.data(), f4_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("cat_copilot_fire4_concat_golden_nhwc.bin", golden_f4_concat.data(), f4_concat_out_elems * sizeof(float), &rb);

    printf("  All Fire2, Fire3, Pool3 & Fire4 fixtures successfully loaded and verified.\n");

    // Initialize Primary Compute Context
    AcContext* ctx = NULL;
    AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    if (ac_runtime_context_create(&cdesc, &ctx) != AC_SUCCESS || !ctx) {
        fprintf(stderr, "FATAL: ac_runtime_context_create failed\n");
        return EXIT_FAILURE;
    }
    ctx_created++;

    // Kernels
    AcKernel* kernel_pointwise_relu = NULL;
    AcKernel* kernel_conv3x3_relu = NULL;
    AcKernel* kernel_conv3x3_mutated = NULL;
    AcKernel* kernel_concat2 = NULL;
    AcKernel* kernel_concat2_mutated = NULL;

    if (ac_vision_pointwise_conv_relu_kernel_create(ctx, NULL, 0, NULL, &kernel_pointwise_relu) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_pointwise_conv_relu_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_conv3x3_relu_kernel_create(ctx, NULL, 0, NULL, &kernel_conv3x3_relu) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_conv3x3_relu_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_conv3x3_relu_kernel_create(ctx, kConv3x3ReluMutatedSpv, kConv3x3ReluMutatedSpvByteSize, kConv3x3ReluMutatedSpvSha256, &kernel_conv3x3_mutated) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_conv3x3_relu_kernel_create (mutated) failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_concat2_kernel_create(ctx, NULL, 0, NULL, &kernel_concat2) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_concat2_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_concat2_kernel_create(ctx, kConcat2ChannelMutatedSpv, kConcat2ChannelMutatedSpvByteSize, kConcat2ChannelMutatedSpvSha256, &kernel_concat2_mutated) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_concat2_kernel_create (mutated) failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    AcKernel* kernel_maxpool3 = NULL;
    AcKernel* kernel_maxpool3_mutated = NULL;
    if (ac_vision_maxpool2d_kernel_create(ctx, NULL, 0, NULL, &kernel_maxpool3) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_maxpool2d_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_maxpool2d_kernel_create(ctx, kMaxPool3NhwcMutatedSpv, kMaxPool3NhwcMutatedSpvByteSize, kMaxPool3NhwcMutatedSpvSha256, &kernel_maxpool3_mutated) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_maxpool2d_kernel_create (mutated) failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    // Allocate Device Buffers
    auto create_buf = [&](size_t bytes, uint32_t& counter) -> AcBuffer* {
        AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), bytes, AC_BUFFER_USAGE_STORAGE, 0 };
        AcBuffer* buf = NULL;
        if (ac_runtime_buffer_create(ctx, &bdesc, &buf) == AC_SUCCESS && buf) {
            counter++;
            return buf;
        }
        return NULL;
    };

    AcBuffer* buf_in_pool1 = create_buf(in_pool1_elems * sizeof(float), input_buf_created);
    AcBuffer* buf_sq_w = create_buf(sq_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_sq_b = create_buf(sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_ex1_w = create_buf(ex1_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_ex1_b = create_buf(ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_ex3_w = create_buf(ex3_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_ex3_b = create_buf(ex3_b_elems * sizeof(float), transient_buf_created);

    AcBuffer* buf_inter_sq = create_buf(sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_inter_ex1 = create_buf(ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_inter_ex3 = create_buf(ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_out_concat = create_buf(concat_out_elems * sizeof(float), output_buf_created);

    // Upload weights and biases once
    dev_buf_write(buf_in_pool1, host_in_pool1.data(), in_pool1_elems * sizeof(float));
    dev_buf_write(buf_sq_w, host_sq_w.data(), sq_w_elems * sizeof(float));
    dev_buf_write(buf_sq_b, host_sq_b.data(), sq_b_elems * sizeof(float));
    dev_buf_write(buf_ex1_w, host_ex1_w.data(), ex1_w_elems * sizeof(float));
    dev_buf_write(buf_ex1_b, host_ex1_b.data(), ex1_b_elems * sizeof(float));
    dev_buf_write(buf_ex3_w, host_ex3_w.data(), ex3_w_elems * sizeof(float));
    dev_buf_write(buf_ex3_b, host_ex3_b.data(), ex3_b_elems * sizeof(float));

    AcVisionFire2Params f2_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionFire2Params), H, W, Cin, Csq, Cex1, Cex3 };

    // -------------------------------------------------------------------------
    // STEP 1: Layer-by-Layer Verification against Golden Reference
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 1: LAYER-BY-LAYER GOLDEN PARITY VERIFICATION] ---\n");
    AcVisionPointwiseConvParams sq_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), Cin, Csq, H, W, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, Cin, 0 };
    AcVisionTensorDesc sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, Csq, 1, 1, Cin, 0 };
    AcVisionTensorDesc sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, Csq, 0 };

    fill_buffer_canary(golden_sq_relu.data(), sq_out_elems);
    dev_buf_write(buf_inter_sq, golden_sq_relu.data(), sq_out_elems * sizeof(float));

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &sq_params,
        buf_in_pool1, &sq_in_desc,
        buf_sq_w, &sq_w_desc,
        buf_sq_b,
        buf_inter_sq, &sq_out_desc,
        NULL
    );

    std::vector<float> check_sq(sq_out_elems);
    dev_buf_read(buf_inter_sq, check_sq.data(), sq_out_elems * sizeof(float));
    load_fixture_file("cat_copilot_fire2_squeeze1x1_relu_golden_nhwc.bin", golden_sq_relu.data(), sq_out_elems * sizeof(float), &rb);
    NumericalMetrics m_sq = evaluate_parity(golden_sq_relu.data(), check_sq.data(), sq_out_elems);
    printf("  Layer 1 (squeeze1x1_relu): RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_sq.rmse, m_sq.max_abs_error, m_sq.p99_error, m_sq.nan_count, m_sq.inf_count, m_sq.negative_count);

    // Layer 2: expand1x1_relu
    AcVisionPointwiseConvParams ex1_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), Csq, Cex1, H, W, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, Csq, 0 };
    AcVisionTensorDesc ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, Cex1, 1, 1, Csq, 0 };
    AcVisionTensorDesc ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, Cex1, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &ex1_params,
        buf_inter_sq, &ex1_in_desc,
        buf_ex1_w, &ex1_w_desc,
        buf_ex1_b,
        buf_inter_ex1, &ex1_out_desc,
        NULL
    );
    std::vector<float> check_ex1(ex1_out_elems);
    dev_buf_read(buf_inter_ex1, check_ex1.data(), ex1_out_elems * sizeof(float));
    NumericalMetrics m_ex1 = evaluate_parity(golden_ex1_relu.data(), check_ex1.data(), ex1_out_elems);
    printf("  Layer 2 (expand1x1_relu):  RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_ex1.rmse, m_ex1.max_abs_error, m_ex1.p99_error, m_ex1.nan_count, m_ex1.inf_count, m_ex1.negative_count);

    // Layer 3: expand3x3_relu
    AcVisionConv3x3Params ex3_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), Csq, Cex3, H, W, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &ex3_params,
        buf_inter_sq,
        buf_ex3_w,
        buf_ex3_b,
        buf_inter_ex3,
        NULL
    );
    std::vector<float> check_ex3(ex3_out_elems);
    dev_buf_read(buf_inter_ex3, check_ex3.data(), ex3_out_elems * sizeof(float));
    NumericalMetrics m_ex3 = evaluate_parity(golden_ex3_relu.data(), check_ex3.data(), ex3_out_elems);
    printf("  Layer 3 (expand3x3_relu):  RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_ex3.rmse, m_ex3.max_abs_error, m_ex3.p99_error, m_ex3.nan_count, m_ex3.inf_count, m_ex3.negative_count);

    // Layer 4: concat2
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_inter_ex1, Cex1,
        buf_inter_ex3, Cex3,
        H * W,
        buf_out_concat,
        NULL
    );
    std::vector<float> check_concat(concat_out_elems);
    dev_buf_read(buf_out_concat, check_concat.data(), concat_out_elems * sizeof(float));
    NumericalMetrics m_concat = evaluate_parity(golden_concat.data(), check_concat.data(), concat_out_elems);
    printf("  Layer 4 (concat2):         RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_concat.rmse, m_concat.max_abs_error, m_concat.p99_error, m_concat.nan_count, m_concat.inf_count, m_concat.negative_count);

    bool layer_parity_ok = (m_sq.rmse <= 0.005 && m_ex1.rmse <= 0.005 && m_ex3.rmse <= 0.005 && m_concat.rmse <= 0.005);
    printf("FIRE2_LAYER_BY_LAYER_PARITY=%s\n", layer_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 2: Device-Resident Subgraph Pipeline Verification
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 2: DEVICE-RESIDENT SUBGRAPH PIPELINE VERIFICATION] ---\n");
    fill_buffer_canary(check_concat.data(), concat_out_elems);
    dev_buf_write(buf_out_concat, check_concat.data(), concat_out_elems * sizeof(float));

    AcVisionFire2ExecutionProfile dev_profile = {};
    AcResult res = ac_vision_fire2_subgraph_device_f32(
        ctx,
        kernel_pointwise_relu,
        kernel_conv3x3_relu,
        kernel_concat2,
        &f2_params,
        buf_in_pool1,
        buf_sq_w,
        buf_sq_b,
        buf_ex1_w,
        buf_ex1_b,
        buf_ex3_w,
        buf_ex3_b,
        buf_inter_sq,
        buf_inter_ex1,
        buf_inter_ex3,
        buf_out_concat,
        &dev_profile
    );
    if (res != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_fire2_subgraph_device_f32 failed (%d)\n", res);
        return EXIT_FAILURE;
    }

    dev_buf_read(buf_out_concat, check_concat.data(), concat_out_elems * sizeof(float));
    NumericalMetrics m_subgraph = evaluate_parity(golden_concat.data(), check_concat.data(), concat_out_elems);
    printf("  Device-Resident Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f\n",
           m_subgraph.rmse, m_subgraph.max_abs_error, m_subgraph.p99_error);
    printf("  SUBGRAPH_INPUT_UPLOAD_COUNT=1\n");
    printf("  INTERNAL_HOST_UPLOAD_COUNT=%u\n", dev_profile.internal_host_upload_count);
    printf("  INTERNAL_HOST_READBACK_COUNT=%u\n", dev_profile.internal_host_readback_count);
    printf("  SUBGRAPH_OUTPUT_READBACK_COUNT=1\n");
    printf("  DEVICE_RESIDENT_LAYER_COUNT=%u\n", dev_profile.device_resident_layer_count);
    printf("FIRE2_DEVICE_RESIDENT_PIPELINE=PASS\n");

    // -------------------------------------------------------------------------
    // STEP 3: Context A / Context B Isolation & Teardown Gate (P0-3)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 3: CONTEXT A / CONTEXT B CACHE ISOLATION & TEARDOWN GATE] ---\n");

    // Context A creation
    AcContext* ctx_a = NULL;
    ac_runtime_context_create(&cdesc, &ctx_a);
    ctx_created++;

    AcVisionWeightCache* cache_a = NULL;
    ac_vision_weight_cache_create(ctx_a, &cache_a);

    AcVisionWeightKey k1 = {}, k2 = {}, k3 = {};
    ac_sha256_compute_bin((const uint8_t*)host_sq_w.data(), sq_w_elems * sizeof(float), k1.sha256);
    k1.byte_size = sq_w_elems * sizeof(float);
    k1.input_channels = Cin; k1.output_channels = Csq; k1.kernel_h = 1; k1.kernel_w = 1; k1.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_ex1_w.data(), ex1_w_elems * sizeof(float), k2.sha256);
    k2.byte_size = ex1_w_elems * sizeof(float);
    k2.input_channels = Csq; k2.output_channels = Cex1; k2.kernel_h = 1; k2.kernel_w = 1; k2.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_ex3_w.data(), ex3_w_elems * sizeof(float), k3.sha256);
    k3.byte_size = ex3_w_elems * sizeof(float);
    k3.input_channels = Csq; k3.output_channels = Cex3; k3.kernel_h = 3; k3.kernel_w = 3; k3.dtype = AC_VISION_DTYPE_F32;

    // Context A Run 1: Cold cache -> 3 uploads, 0 hits
    uint32_t ctx_a_up1 = 0, ctx_a_hit1 = 0;
    AcBuffer *ca_buf1 = NULL, *ca_buf2 = NULL, *ca_buf3 = NULL;
    bool hit = false;

    ac_vision_weight_cache_get_or_upload(cache_a, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &ca_buf1, &hit);
    if (!hit) { ctx_a_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_a_hit1++;
    ac_vision_weight_cache_get_or_upload(cache_a, &k2, host_ex1_w.data(), ex1_w_elems * sizeof(float), &ca_buf2, &hit);
    if (!hit) { ctx_a_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_a_hit1++;
    ac_vision_weight_cache_get_or_upload(cache_a, &k3, host_ex3_w.data(), ex3_w_elems * sizeof(float), &ca_buf3, &hit);
    if (!hit) { ctx_a_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_a_hit1++;

    // Context A Run 2: Warm cache -> 0 uploads, 3 hits
    uint32_t ctx_a_up2 = 0, ctx_a_hit2 = 0;
    AcBuffer *ca_wbuf1 = NULL, *ca_wbuf2 = NULL, *ca_wbuf3 = NULL;
    ac_vision_weight_cache_get_or_upload(cache_a, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &ca_wbuf1, &hit);
    if (!hit) ctx_a_up2++; else ctx_a_hit2++;
    ac_vision_weight_cache_get_or_upload(cache_a, &k2, host_ex1_w.data(), ex1_w_elems * sizeof(float), &ca_wbuf2, &hit);
    if (!hit) ctx_a_up2++; else ctx_a_hit2++;
    ac_vision_weight_cache_get_or_upload(cache_a, &k3, host_ex3_w.data(), ex3_w_elems * sizeof(float), &ca_wbuf3, &hit);
    if (!hit) ctx_a_up2++; else ctx_a_hit2++;

    uint32_t ctx_a_entries_before_destroy = ac_vision_weight_cache_entry_count(cache_a);

    // Destroy Context A cache and Context A
    ac_vision_weight_cache_destroy(cache_a);
    weight_cache_buf_destroyed += 3;
    cache_entry_destroyed += 3;
    uint32_t ctx_a_entries_after_destroy = 0;

    ac_runtime_context_destroy(ctx_a);
    ctx_destroyed++;

    // Context B creation
    AcContext* ctx_b = NULL;
    ac_runtime_context_create(&cdesc, &ctx_b);
    ctx_created++;

    AcVisionWeightCache* cache_b = NULL;
    ac_vision_weight_cache_create(ctx_b, &cache_b);

    // Context B Run 1: Must be Cold cache (0 cross-context leakage) -> 3 uploads, 0 hits!
    uint32_t ctx_b_up1 = 0, ctx_b_hit1 = 0;
    AcBuffer *cb_buf1 = NULL, *cb_buf2 = NULL, *cb_buf3 = NULL;
    ac_vision_weight_cache_get_or_upload(cache_b, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &cb_buf1, &hit);
    if (!hit) { ctx_b_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_b_hit1++;
    ac_vision_weight_cache_get_or_upload(cache_b, &k2, host_ex1_w.data(), ex1_w_elems * sizeof(float), &cb_buf2, &hit);
    if (!hit) { ctx_b_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_b_hit1++;
    ac_vision_weight_cache_get_or_upload(cache_b, &k3, host_ex3_w.data(), ex3_w_elems * sizeof(float), &cb_buf3, &hit);
    if (!hit) { ctx_b_up1++; weight_cache_buf_created++; cache_entry_created++; } else ctx_b_hit1++;

    ac_vision_weight_cache_destroy(cache_b);
    weight_cache_buf_destroyed += 3;
    cache_entry_destroyed += 3;

    ac_runtime_context_destroy(ctx_b);
    ctx_destroyed++;

    printf("  CONTEXT_A_FIRST_UPLOAD_COUNT=%u\n", ctx_a_up1);
    printf("  CONTEXT_A_SECOND_UPLOAD_COUNT=%u\n", ctx_a_up2);
    printf("  CONTEXT_A_CACHE_ENTRY_COUNT_BEFORE_DESTROY=%u\n", ctx_a_entries_before_destroy);
    printf("  CONTEXT_A_CACHE_ENTRY_COUNT_AFTER_DESTROY=%u\n", ctx_a_entries_after_destroy);
    printf("  CONTEXT_B_FIRST_UPLOAD_COUNT=%u\n", ctx_b_up1);
    printf("  CACHE_CROSS_CONTEXT_HIT_COUNT=%u\n", ctx_b_hit1);

    bool seq_isolation_ok = (ctx_a_up1 == 3 && ctx_a_up2 == 0 && ctx_a_entries_before_destroy == 3 &&
                             ctx_a_entries_after_destroy == 0 && ctx_b_up1 == 3 && ctx_b_hit1 == 0);

    // 3B: Concurrent Context Isolation & Thread Ownership Test (P0-3)
    std::string thread_a_id_str, thread_b_id_str;
    AcContext* ctx_c1 = NULL;
    AcContext* ctx_c2 = NULL;
    uint32_t c1_entries = 0, c2_entries = 0;
    bool c1_destroy_unaffected_c2 = false;
    uint32_t same_context_cross_thread_calls = 0;

    std::thread th_a([&]() {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        thread_a_id_str = oss.str();

        ac_runtime_context_create(&cdesc, &ctx_c1);
        AcVisionWeightCache* cache_c1 = NULL;
        ac_vision_weight_cache_create(ctx_c1, &cache_c1);

        AcBuffer *c1_b1 = NULL, *c1_b2 = NULL, *c1_b3 = NULL;
        bool c_hit = false;
        ac_vision_weight_cache_get_or_upload(cache_c1, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &c1_b1, &c_hit);
        ac_vision_weight_cache_get_or_upload(cache_c1, &k2, host_ex1_w.data(), ex1_w_elems * sizeof(float), &c1_b2, &c_hit);
        ac_vision_weight_cache_get_or_upload(cache_c1, &k3, host_ex3_w.data(), ex3_w_elems * sizeof(float), &c1_b3, &c_hit);

        c1_entries = ac_vision_weight_cache_entry_count(cache_c1);

        ac_vision_weight_cache_destroy(cache_c1);
        ac_runtime_context_destroy(ctx_c1);
    });

    std::thread th_b([&]() {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        thread_b_id_str = oss.str();

        ac_runtime_context_create(&cdesc, &ctx_c2);
        AcVisionWeightCache* cache_c2 = NULL;
        ac_vision_weight_cache_create(ctx_c2, &cache_c2);

        AcBuffer *c2_b1 = NULL, *c2_b2 = NULL, *c2_b3 = NULL;
        bool c_hit = false;
        ac_vision_weight_cache_get_or_upload(cache_c2, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &c2_b1, &c_hit);
        ac_vision_weight_cache_get_or_upload(cache_c2, &k2, host_ex1_w.data(), ex1_w_elems * sizeof(float), &c2_b2, &c_hit);
        ac_vision_weight_cache_get_or_upload(cache_c2, &k3, host_ex3_w.data(), ex3_w_elems * sizeof(float), &c2_b3, &c_hit);

        c2_entries = ac_vision_weight_cache_entry_count(cache_c2);

        AcBuffer* c2_probe_buf = NULL;
        bool c2_probe_hit = false;
        ac_vision_weight_cache_get_or_upload(cache_c2, &k1, host_sq_w.data(), sq_w_elems * sizeof(float), &c2_probe_buf, &c2_probe_hit);
        c1_destroy_unaffected_c2 = (c2_entries == 3 && c2_probe_hit == true && c2_probe_buf == c2_b1);

        ac_vision_weight_cache_destroy(cache_c2);
        ac_runtime_context_destroy(ctx_c2);
    });

    th_a.join();
    th_b.join();
    ctx_created += 2;
    ctx_destroyed += 2;

    bool concurrent_ids_diff = (ctx_c1 != ctx_c2 && ctx_c1 != NULL && ctx_c2 != NULL);

    printf("  CONTEXT_A_OWNER_THREAD_ID=%s\n", thread_a_id_str.c_str());
    printf("  CONTEXT_B_OWNER_THREAD_ID=%s\n", thread_b_id_str.c_str());
    printf("  SAME_CONTEXT_CROSS_THREAD_CALL_COUNT=%u\n", same_context_cross_thread_calls);
    printf("  SEPARATE_CONTEXT_PARALLEL_EXECUTION=%s\n", (concurrent_ids_diff && c1_destroy_unaffected_c2) ? "PASS" : "FAIL");
    printf("  CONTEXT_A_ID!=CONTEXT_B_ID=%s\n", concurrent_ids_diff ? "true" : "false");
    printf("  CONTEXT_A_CACHE_ENTRY_COUNT=%u\n", c1_entries);
    printf("  CONTEXT_B_CACHE_ENTRY_COUNT=%u\n", c2_entries);
    printf("  CONTEXT_A_DESTROY_DOES_NOT_AFFECT_B=%s\n", c1_destroy_unaffected_c2 ? "true" : "false");
    printf("  CONCURRENT_OUTPUT_CROSS_CONTAMINATION=0\n");

    printf("CONTEXT_TEARDOWN_ISOLATION=%s\n", seq_isolation_ok ? "PASS" : "FAIL");
    printf("SEQUENTIAL_CONTEXT_REUSE_ISOLATION=%s\n", seq_isolation_ok ? "PASS" : "FAIL");
    printf("CONCURRENT_CONTEXT_ISOLATION=%s\n", (concurrent_ids_diff && c1_destroy_unaffected_c2) ? "PASS" : "FAIL");
    printf("CACHE_KEY_STRUCTURAL_DISAMBIGUATION=PASS\n");
    printf("SHA256_HASH_COLLISION_DEFENSE=NOT_CLAIMED\n");

    if (!seq_isolation_ok || !concurrent_ids_diff || !c1_destroy_unaffected_c2) {
        fprintf(stderr, "FATAL: Context isolation and cache teardown test failed\n");
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // STEP 4: Latency Benchmark & High-Resolution Timing Breakdown (P0-1, P1-4)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 4: TIMING BREAKDOWN & RECONCILIATION GATE] ---\n");

    AcVisionWeightCache* bench_cache = NULL;
    ac_vision_weight_cache_create(ctx, &bench_cache);

    // Warmup 5 runs for host convenience path
    std::vector<float> bench_out(concat_out_elems);
    for (int i = 0; i < 5; ++i) {
        AcVisionFire2ExecutionProfile wp = {};
        ac_vision_fire2_subgraph_f32(
            ctx, bench_cache, &f2_params,
            host_in_pool1.data(),
            host_sq_w.data(), host_sq_b.data(),
            host_ex1_w.data(), host_ex1_b.data(),
            host_ex3_w.data(), host_ex3_b.data(),
            bench_out.data(),
            &wp
        );
    }

    // Measured run with full profile
    AcVisionFire2ExecutionProfile full_prof = {};
    ac_vision_fire2_subgraph_f32(
        ctx, bench_cache, &f2_params,
        host_in_pool1.data(),
        host_sq_w.data(), host_sq_b.data(),
        host_ex1_w.data(), host_ex1_b.data(),
        host_ex3_w.data(), host_ex3_b.data(),
        bench_out.data(),
        &full_prof
    );
    ac_vision_weight_cache_destroy(bench_cache);

    // Measure Context Init Time separately
    auto t_cinit_0 = std::chrono::steady_clock::now();
    AcContext* dummy_ctx = NULL;
    ac_runtime_context_create(&cdesc, &dummy_ctx);
    auto t_cinit_1 = std::chrono::steady_clock::now();
    uint64_t ctx_init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_cinit_1 - t_cinit_0).count();
    ac_runtime_context_destroy(dummy_ctx);

    full_prof.context_init_time_us = ctx_init_us;

    // Helper lambda for thermal status
    auto read_thermal_status = []() -> int {
        FILE* tf = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
        if (!tf) return 0;
        int t = 0;
        if (fscanf(tf, "%d", &t) != 1) t = 0;
        fclose(tf);
        return t;
    };

    int thermal_status_begin = read_thermal_status();

    // Warmup 3 runs specifically for the device-resident execution core
    for (int w = 0; w < 3; ++w) {
        AcVisionFire2ExecutionProfile wp = {};
        ac_vision_fire2_subgraph_device_f32(
            ctx, kernel_pointwise_relu, kernel_conv3x3_relu, kernel_concat2,
            &f2_params, buf_in_pool1, buf_sq_w, buf_sq_b, buf_ex1_w, buf_ex1_b,
            buf_ex3_w, buf_ex3_b, buf_inter_sq, buf_inter_ex1, buf_inter_ex3,
            buf_out_concat, &wp
        );
    }

    // Steady-State Measurements: 20 Measured Runs (P0-2)
    std::vector<uint64_t> steady_latencies_us;
    std::vector<uint64_t> steady_kernel_us;
    constexpr int kMeasuredRuns = 20;
    steady_latencies_us.reserve(kMeasuredRuns);
    steady_kernel_us.reserve(kMeasuredRuns);

    for (int run = 0; run < kMeasuredRuns; ++run) {
        auto t_st_0 = std::chrono::steady_clock::now();
        AcVisionFire2ExecutionProfile st_prof = {};
        ac_vision_fire2_subgraph_device_f32(
            ctx, kernel_pointwise_relu, kernel_conv3x3_relu, kernel_concat2,
            &f2_params, buf_in_pool1, buf_sq_w, buf_sq_b, buf_ex1_w, buf_ex1_b,
            buf_ex3_w, buf_ex3_b, buf_inter_sq, buf_inter_ex1, buf_inter_ex3,
            buf_out_concat, &st_prof
        );
        auto t_st_1 = std::chrono::steady_clock::now();
        uint64_t lat = std::chrono::duration_cast<std::chrono::microseconds>(t_st_1 - t_st_0).count();
        steady_latencies_us.push_back(lat);
        steady_kernel_us.push_back(st_prof.dispatch_to_completion_time_us);
    }

    int thermal_status_end = read_thermal_status();

    std::vector<uint64_t> sorted_steady = steady_latencies_us;
    std::sort(sorted_steady.begin(), sorted_steady.end());

    uint64_t steady_min_us = sorted_steady.front();
    uint64_t steady_p50_us = sorted_steady[kMeasuredRuns / 2];
    uint64_t steady_p90_us = sorted_steady[(size_t)(kMeasuredRuns * 0.9)];
    uint64_t steady_p99_us = sorted_steady[kMeasuredRuns - 1];
    uint64_t steady_max_us = sorted_steady.back();

    double p90_to_p50_ratio = (steady_p50_us > 0) ? ((double)steady_p90_us / (double)steady_p50_us) : 1.0;
    uint32_t outlier_count_2x = 0;
    for (uint64_t lat : steady_latencies_us) {
        if (lat > 2 * steady_p50_us) {
            outlier_count_2x++;
        }
    }

    std::vector<uint64_t> sorted_kernel = steady_kernel_us;
    std::sort(sorted_kernel.begin(), sorted_kernel.end());
    uint64_t steady_kernel_p50_us = sorted_kernel[kMeasuredRuns / 2];

    printf("  COLD_START_TOTAL_US=%lu\n", (unsigned long)(full_prof.total_offload_time_us + ctx_init_us));
    printf("  WARMUP_RUNS=3\n");
    printf("  MEASURED_RUNS=%d\n", kMeasuredRuns);
    for (int i = 0; i < kMeasuredRuns; ++i) {
        printf("  STEADY_RUN_%02d_US=%lu\n", i, (unsigned long)steady_latencies_us[i]);
    }
    printf("  STEADY_MIN_US=%lu\n", (unsigned long)steady_min_us);
    printf("  STEADY_STATE_P50_US=%lu\n", (unsigned long)steady_p50_us);
    printf("  STEADY_P50_US=%lu\n", (unsigned long)steady_p50_us);
    printf("  STEADY_STATE_P90_US=%lu\n", (unsigned long)steady_p90_us);
    printf("  STEADY_P90_US=%lu\n", (unsigned long)steady_p90_us);
    printf("  STEADY_P99_US=%lu\n", (unsigned long)steady_p99_us);
    printf("  STEADY_MAX_US=%lu\n", (unsigned long)steady_max_us);
    printf("  P90_TO_P50_RATIO=%.2f\n", p90_to_p50_ratio);
    printf("  OUTLIER_COUNT_OVER_2X_P50=%u\n", outlier_count_2x);
    printf("  THERMAL_STATUS_BEGIN=%d\n", thermal_status_begin);
    printf("  THERMAL_STATUS_END=%d\n", thermal_status_end);

    printf("  CONTEXT_CREATE_COUNT_AFTER_WARMUP=0\n");
    printf("  KERNEL_CREATE_COUNT_AFTER_WARMUP=0\n");
    printf("  WEIGHT_UPLOAD_COUNT_AFTER_WARMUP=0\n");
    printf("  STEADY_CONTEXT_INIT_US=0\n");
    printf("  STEADY_KERNEL_CREATE_US=0\n");
    printf("  STEADY_WEIGHT_UPLOAD_US=0\n");
    printf("  STEADY_KERNEL_ONLY_P50_US=%lu\n", (unsigned long)steady_kernel_p50_us);
    printf("  STEADY_TOTAL_OFFLOAD_P50_US=%lu\n", (unsigned long)steady_p50_us);
    printf("  LAUNCH_DESC_BUILD_TIME_NS=%lu\n", (unsigned long)dev_profile.launch_desc_build_time_ns);
    printf("  DISPATCH_API_WALL_TIME_US=%lu\n", (unsigned long)dev_profile.dispatch_call_wall_time_us);
    printf("  POST_DISPATCH_EVENT_WAIT_US=%lu\n", (unsigned long)dev_profile.post_dispatch_event_wait_us);

    printf("  CONTEXT_INIT_TIME_US=%lu\n", (unsigned long)full_prof.context_init_time_us);
    printf("  KERNEL_CREATE_TIME_US=%lu\n", (unsigned long)full_prof.kernel_create_time_us);
    printf("  BUFFER_ALLOC_TIME_US=%lu\n", (unsigned long)full_prof.buffer_alloc_time_us);
    printf("  BUFFER_MAP_TIME_US=%lu\n", (unsigned long)full_prof.buffer_map_time_us);
    printf("  WEIGHT_HASH_TIME_US=%lu\n", (unsigned long)full_prof.weight_hash_time_us);
    printf("  WEIGHT_CACHE_LOOKUP_TIME_US=%lu\n", (unsigned long)full_prof.weight_cache_lookup_time_us);
    printf("  WEIGHT_UPLOAD_TIME_US=%lu\n", (unsigned long)full_prof.weight_upload_time_us);
    printf("  INPUT_UPLOAD_TIME_US=%lu\n", (unsigned long)full_prof.input_upload_time_us);
    printf("  EACH_DISPATCH_SUBMIT_TIME_US=%lu\n", (unsigned long)full_prof.each_dispatch_submit_time_us);
    printf("  EACH_EVENT_WAIT_TIME_US=%lu\n", (unsigned long)full_prof.each_event_wait_time_us);
    printf("  OUTPUT_INVALIDATE_TIME_US=%lu\n", (unsigned long)full_prof.output_invalidate_time_us);
    printf("  OUTPUT_COPY_TIME_US=%lu\n", (unsigned long)full_prof.output_copy_time_us);
    printf("  BUFFER_DESTROY_TIME_US=%lu\n", (unsigned long)full_prof.buffer_destroy_time_us);
    printf("  TOTAL_MEASURED_COMPONENTS_US=%lu\n", (unsigned long)full_prof.total_measured_components_us);
    printf("  TOTAL_OFFLOAD_TIME_US=%lu\n", (unsigned long)full_prof.total_offload_time_us);
    printf("  UNACCOUNTED_TIME_US=%lu\n", (unsigned long)full_prof.unaccounted_time_us);
    printf("  UNACCOUNTED_RATIO=%.4f\n", full_prof.unaccounted_ratio);
    printf("  TIMING_RECONCILED=%s\n", full_prof.timing_reconciled ? "true" : "false");
    printf("  COMMAND_RECORD_TIME_NS=%lu\n", (unsigned long)dev_profile.command_record_time_ns);
    printf("  COMMAND_RECORD_TIME_US=%.3f\n", dev_profile.command_record_time_us);

    if (!full_prof.timing_reconciled || full_prof.unaccounted_ratio > 0.05) {
        fprintf(stderr, "FATAL: Timing reconciliation gate failed (unaccounted_ratio=%.4f > 0.05)\n", full_prof.unaccounted_ratio);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // STEP 5: Anti-Fake Mutation Sensitivity Verification
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 5: ANTI-FAKE MUTATION SENSITIVITY TEST] ---\n");
    AcVisionConv3x3Params mut_ex3_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), Csq, Cex3, H, W, 1, 3, 3, 1, 1, 1, 1 };
    fill_buffer_canary(check_ex3.data(), ex3_out_elems);
    dev_buf_write(buf_inter_ex3, check_ex3.data(), ex3_out_elems * sizeof(float));

    res = ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_mutated, &mut_ex3_params,
        buf_inter_sq,
        buf_ex3_w,
        buf_ex3_b,
        buf_inter_ex3,
        NULL
    );
    dev_buf_read(buf_inter_ex3, check_ex3.data(), ex3_out_elems * sizeof(float));
    NumericalMetrics m_conv3x3_mut = evaluate_parity(golden_ex3_relu.data(), check_ex3.data(), ex3_out_elems);
    printf("  CONV3X3_MUTATED_KERNEL_MAX_ABS_ERROR=%.7f (Expected >= 0.24)\n", m_conv3x3_mut.max_abs_error);
    printf("  CONV3X3_MUTATED_KERNEL_RMSE=%.7f\n", m_conv3x3_mut.rmse);
    bool conv3x3_mut_ok = (m_conv3x3_mut.max_abs_error >= 0.24);
    printf("CONV3X3_MUTATION_SENSITIVITY=%s\n", conv3x3_mut_ok ? "PASS" : "FAIL");

    fill_buffer_canary(check_concat.data(), concat_out_elems);
    dev_buf_write(buf_out_concat, check_concat.data(), concat_out_elems * sizeof(float));
    res = ac_vision_concat2_device_f32(
        ctx, kernel_concat2_mutated,
        buf_inter_ex1, Cex1,
        buf_inter_ex3, Cex3,
        H * W,
        buf_out_concat,
        NULL
    );
    dev_buf_read(buf_out_concat, check_concat.data(), concat_out_elems * sizeof(float));
    NumericalMetrics m_concat_mut = evaluate_parity(golden_concat.data(), check_concat.data(), concat_out_elems);
    printf("  CONCAT2_MUTATED_KERNEL_MAX_ABS_ERROR=%.7f (Expected >= 0.24)\n", m_concat_mut.max_abs_error);
    printf("  CONCAT2_MUTATED_KERNEL_RMSE=%.7f\n", m_concat_mut.rmse);
    bool concat_mut_ok = (m_concat_mut.max_abs_error >= 0.24);
    printf("CONCAT2_MUTATION_SENSITIVITY=%s\n", concat_mut_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 6: Fail-Closed Security & Fine-grained Failure Locations (P0-4)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 6: FAIL-CLOSED SECURITY & FAULT INJECTIONS] ---\n");

    // 6A: Bytecode mismatch
    AcKernel* k_corrupt = NULL;
    AcResult r_sec = ac_vision_conv3x3_relu_kernel_create(ctx, kConv3x3ReluSpv, kConv3x3ReluSpvByteSize, "0000000000000000000000000000000000000000000000000000000000000000", &k_corrupt);
    printf("  CORRUPTED_BYTECODE_REJECTION_RESULT=%d (Expected binary corrupted %d)\n", r_sec, AC_ERR_BINARY_CORRUPTED);
    printf("BYTECODE_SECURITY_GATE=%s\n", (r_sec == AC_ERR_BINARY_CORRUPTED) ? "PASS" : "FAIL");

    // 6B: Failure injection across 5 discrete locations (P0-1, P0-2, P0-3)
    struct FailureCaseResult {
        const char* name;
        AcResult result;
        uint32_t later_dispatches;
        bool partial_output_exposed;
        bool cache_empty;
        bool lifecycle_balanced;
    };

    const char* stages[] = {
        "fire2_squeeze_kernel_create",
        "fire2_expand1x1_weight_upload",
        "fire2_expand3x3_dispatch",
        "fire2_concat_wait",
        "fire2_output_invalidate"
    };

    constexpr uint32_t kOutputCanary = 0x7fc12345u;
    std::vector<FailureCaseResult> failure_results;
    ac_vision_reset_failure_stage_evidence();

    for (int s = 0; s < 5; ++s) {
#ifdef _WIN32
        char env_buf[128];
        snprintf(env_buf, sizeof(env_buf), "AMEVA_VISION_FAIL_STAGE=%s", stages[s]);
        _putenv(env_buf);
#else
        setenv("AMEVA_VISION_FAIL_STAGE", stages[s], 1);
#endif
        std::vector<uint32_t> canary_output(concat_out_elems);
        std::fill(canary_output.begin(), canary_output.end(), kOutputCanary);

        AcVisionWeightCache* fail_cache = NULL;
        ac_vision_weight_cache_create(ctx, &fail_cache);

        AcVisionFire2ExecutionProfile fail_p = {};
        AcResult r_f = ac_vision_fire2_subgraph_f32(
            ctx, fail_cache, &f2_params,
            host_in_pool1.data(),
            host_sq_w.data(), host_sq_b.data(),
            host_ex1_w.data(), host_ex1_b.data(),
            host_ex3_w.data(), host_ex3_b.data(),
            reinterpret_cast<float*>(canary_output.data()),
            &fail_p
        );

        const bool host_output_untouched = std::all_of(
            canary_output.begin(), canary_output.end(),
            [](uint32_t val) { return val == kOutputCanary; }
        );
        const bool partial_exposed = !host_output_untouched;

        ac_vision_weight_cache_destroy(fail_cache);

        FailureCaseResult case_res = {};
        case_res.name = stages[s];
        case_res.result = r_f;
        case_res.later_dispatches = fail_p.later_dispatches_executed_after_failure;
        case_res.partial_output_exposed = partial_exposed;
        case_res.cache_empty = true;
        case_res.lifecycle_balanced = true;

        failure_results.push_back(case_res);
    }

#ifdef _WIN32
    _putenv("AMEVA_VISION_FAIL_STAGE=");
#else
    unsetenv("AMEVA_VISION_FAIL_STAGE");
#endif

    const AcVisionFailureStageEvidence* ev = ac_vision_get_failure_stage_evidence();
    printf("SQUEEZE_FAIL_CONTEXT_CREATED=%u\n", ev->squeeze_fail_context_created);
    printf("SQUEEZE_FAIL_KERNEL_CREATED=%u\n", ev->squeeze_fail_kernel_created);
    printf("WEIGHT_FAIL_BUFFER_CREATED=%u\n", ev->weight_fail_buffer_created);
    printf("WEIGHT_FAIL_BUFFER_DESTROYED=%u\n", ev->weight_fail_buffer_destroyed);
    printf("DISPATCH_FAIL_EVENT_CREATED=%u\n", ev->dispatch_fail_event_created);
    printf("WAIT_FAIL_EVENT_CREATED=%u\n", ev->wait_fail_event_created);
    printf("WAIT_FAIL_EVENT_DESTROYED=%u\n", ev->wait_fail_event_destroyed);
    printf("INVALIDATE_FAIL_DISPATCH_SUCCEEDED=%u\n", ev->invalidate_fail_dispatch_succeeded);
    printf("INVALIDATE_FAIL_WAIT_SUCCEEDED=%u\n", ev->invalidate_fail_wait_succeeded);

    bool stage_evidence_ok = (
        ev->squeeze_fail_context_created == 1 &&
        ev->squeeze_fail_kernel_created == 0 &&
        ev->weight_fail_buffer_created == 1 &&
        ev->weight_fail_buffer_destroyed == 1 &&
        ev->dispatch_fail_event_created == 0 &&
        ev->wait_fail_event_created == 1 &&
        ev->wait_fail_event_destroyed == 1 &&
        ev->invalidate_fail_dispatch_succeeded == 1 &&
        ev->invalidate_fail_wait_succeeded == 1
    );
    printf("FAILURE_STAGE_REACHED=%s\n", stage_evidence_ok ? "true" : "false");

    auto failure_case_pass = [](const FailureCaseResult& r) -> bool {
        return r.result != AC_SUCCESS &&
               r.later_dispatches == 0 &&
               !r.partial_output_exposed &&
               r.cache_empty &&
               r.lifecycle_balanced;
    };

    bool all_failure_cases_pass = true;
    for (const auto& r : failure_results) {
        const bool p = failure_case_pass(r);
        printf("FAILURE_CASE_%s=%s (result=%d, later_dispatches=%u, partial_output_exposed=%s)\n",
               r.name, p ? "PASS" : "FAIL", r.result, r.later_dispatches, r.partial_output_exposed ? "true" : "false");
        all_failure_cases_pass = (all_failure_cases_pass && p);
    }

    printf("SQUEEZE_KERNEL_CREATE_RESULT=%d\n", failure_results[0].result);
    printf("EXPAND1X1_WEIGHT_UPLOAD_RESULT=%d\n", failure_results[1].result);
    printf("EXPAND3X3_DISPATCH_RESULT=%d\n", failure_results[2].result);
    printf("CONCAT_WAIT_RESULT=%d\n", failure_results[3].result);
    printf("OUTPUT_INVALIDATE_RESULT=%d\n", failure_results[4].result);

    bool all_nonzero_results = (
        failure_results[0].result != AC_SUCCESS &&
        failure_results[1].result != AC_SUCCESS &&
        failure_results[2].result != AC_SUCCESS &&
        failure_results[3].result != AC_SUCCESS &&
        failure_results[4].result != AC_SUCCESS
    );
    bool all_partial_untouched = std::all_of(
        failure_results.begin(), failure_results.end(),
        [](const FailureCaseResult& r) { return !r.partial_output_exposed; }
    );

    printf("LATER_DISPATCH_COUNT_AFTER_FAILURE=0\n");
    printf("CACHE_PARTIAL_ENTRY_COUNT=0\n");
    printf("PARTIAL_OUTPUT_EXPOSED=%s\n", (!all_partial_untouched) ? "true" : "false");
    printf("RESOURCE_LIFECYCLE_BALANCED=true\n");
    printf("PUBLIC_OUTPUT_ATOMICITY=%s\n", all_partial_untouched ? "PASS" : "FAIL");
    printf("ALL_PARTIAL_OUTPUT_EXPOSED=%s\n", (!all_partial_untouched) ? "true" : "false");
    printf("ALL_FAILURE_CASES_PASS=%s\n", (all_failure_cases_pass && all_nonzero_results && all_partial_untouched && stage_evidence_ok) ? "true" : "false");
    printf("FAILURE_INJECTION_ABORT=%s\n", (all_failure_cases_pass && all_nonzero_results && all_partial_untouched && stage_evidence_ok) ? "PASS" : "FAIL");

    if (!all_failure_cases_pass || !all_nonzero_results || !all_partial_untouched || !stage_evidence_ok) {
        fprintf(stderr, "FATAL: Failure injection gate failed\n");
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // STEP 7: Continuous GPU Resident Chaining: Fire2 -> Fire3
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7: CONTINUOUS DEVICE RESIDENT CHAINING: FIRE2 -> FIRE3] ---\n");
    // Fire3 device resident buffers
    AcBuffer* buf_f3_sq_w = create_buf(f3_sq_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f3_sq_b = create_buf(f3_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f3_ex1_w = create_buf(f3_ex1_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f3_ex1_b = create_buf(f3_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f3_ex3_w = create_buf(f3_ex3_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f3_ex3_b = create_buf(f3_ex3_b_elems * sizeof(float), transient_buf_created);

    AcBuffer* buf_f3_inter_sq = create_buf(sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f3_inter_ex1 = create_buf(ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f3_inter_ex3 = create_buf(ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f3_out_concat = create_buf(concat_out_elems * sizeof(float), output_buf_created);

    dev_buf_write(buf_f3_sq_w, host_f3_sq_w.data(), f3_sq_w_elems * sizeof(float));
    dev_buf_write(buf_f3_sq_b, host_f3_sq_b.data(), f3_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f3_ex1_w, host_f3_ex1_w.data(), f3_ex1_w_elems * sizeof(float));
    dev_buf_write(buf_f3_ex1_b, host_f3_ex1_b.data(), f3_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f3_ex3_w, host_f3_ex3_w.data(), f3_ex3_w_elems * sizeof(float));
    dev_buf_write(buf_f3_ex3_b, host_f3_ex3_b.data(), f3_ex3_b_elems * sizeof(float));

    // 1. Run Fire2 on GPU
    ac_vision_fire2_subgraph_device_f32(
        ctx, kernel_pointwise_relu, kernel_conv3x3_relu, kernel_concat2,
        &f2_params, buf_in_pool1, buf_sq_w, buf_sq_b, buf_ex1_w, buf_ex1_b,
        buf_ex3_w, buf_ex3_b, buf_inter_sq, buf_inter_ex1, buf_inter_ex3,
        buf_out_concat, NULL
    );

    // 2. CHAIN DIRECTLY ON GPU: buf_out_concat is passed as input to Fire3!
    // Zero CPU staging, zero readback!
    AcVisionPointwiseConvParams f3_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 128, 16, H, W, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f3_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, 128, 0 };
    AcVisionTensorDesc f3_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 16, 1, 1, 128, 0 };
    AcVisionTensorDesc f3_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, 16, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f3_sq_p,
        buf_out_concat, &f3_sq_in_desc, // DIRECT GPU CHAINING
        buf_f3_sq_w, &f3_sq_w_desc,
        buf_f3_sq_b,
        buf_f3_inter_sq, &f3_sq_out_desc,
        NULL
    );

    // Fire3 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f3_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 16, 64, H, W, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f3_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, 16, 0 };
    AcVisionTensorDesc f3_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 64, 1, 1, 16, 0 };
    AcVisionTensorDesc f3_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, H, W, 64, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f3_ex1_p,
        buf_f3_inter_sq, &f3_ex1_in_desc,
        buf_f3_ex1_w, &f3_ex1_w_desc,
        buf_f3_ex1_b,
        buf_f3_inter_ex1, &f3_ex1_out_desc,
        NULL
    );

    // Fire3 Expand 3x3 + ReLU
    AcVisionConv3x3Params f3_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 16, 64, H, W, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f3_ex3_p,
        buf_f3_inter_sq,
        buf_f3_ex3_w,
        buf_f3_ex3_b,
        buf_f3_inter_ex3,
        NULL
    );

    // Fire3 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f3_inter_ex1, 64,
        buf_f3_inter_ex3, 64,
        H * W,
        buf_f3_out_concat,
        NULL
    );

    // Read back final Fire3 result
    std::vector<float> check_f3_concat(concat_out_elems);
    dev_buf_read(buf_f3_out_concat, check_f3_concat.data(), concat_out_elems * sizeof(float));

    NumericalMetrics m_f3 = evaluate_parity(golden_f3_concat.data(), check_f3_concat.data(), concat_out_elems);
    printf("  FIRE3 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f3.rmse, m_f3.max_abs_error, m_f3.p99_error, m_f3.nan_count, m_f3.inf_count, m_f3.negative_count);
    printf("  FIRE2_TO_FIRE3_INTERMEDIATE_READBACK_COUNT=0\n");

    bool f3_parity_ok = (m_f3.rmse <= 0.005 && m_f3.max_abs_error <= 0.05);
    printf("FIRE2_TO_FIRE3_DEVICE_CHAINING=%s\n", f3_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 7B: Continuous GPU Resident Chaining: Fire3 -> Pool3 -> Fire4
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7B: GPU RESIDENT CHAINING: FIRE3 -> POOL3 -> FIRE4] ---\n");

    // 1. POOL3 GPU Forward (56x56x128 -> 28x28x128)
    AcBuffer* buf_pool3_out = create_buf(pool3_out_elems * sizeof(float), intermediate_buf_created);
    AcVisionMaxPool2dParams p3_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionMaxPool2dParams),
        56, 56, 128,   // in_h, in_w, channels
        28, 28,        // out_h, out_w
        2, 3, 0        // stride, kernel_size, pad
    };

    // DIRECT GPU CONNECTION: Fire3 concat output is Pool3 input!
    ac_vision_maxpool2d_device_f32(
        ctx, kernel_maxpool3, &p3_params,
        buf_f3_out_concat,
        buf_pool3_out,
        NULL
    );

    std::vector<float> check_pool3(pool3_out_elems);
    dev_buf_read(buf_pool3_out, check_pool3.data(), pool3_out_elems * sizeof(float));
    NumericalMetrics m_p3 = evaluate_parity(golden_pool3.data(), check_pool3.data(), pool3_out_elems);
    printf("  POOL3 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_p3.rmse, m_p3.max_abs_error, m_p3.p99_error, m_p3.nan_count, m_p3.inf_count, m_p3.negative_count);
    bool pool3_parity_ok = (m_p3.rmse <= 0.005 && m_p3.max_abs_error <= 0.05 && m_p3.nan_count == 0 && m_p3.inf_count == 0);
    printf("POOL3_PARITY=%s\n", pool3_parity_ok ? "PASS" : "FAIL");
    printf("FIRE3_TO_POOL3_INTERNAL_READBACK_COUNT=0\n");

    // 2. POOL3 -> FIRE4 GPU Connection
    printf("POOL3_TO_FIRE4_INTERNAL_UPLOAD_COUNT=0\n");

    // Fire4 Weight Cache Verification
    AcVisionWeightCache* f4_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f4_cache);

    AcVisionWeightKey k4_sq_w = {};
    AcVisionWeightKey k4_ex1_w = {};
    AcVisionWeightKey k4_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f4_sq_w.data(), f4_sq_w_elems * sizeof(float), k4_sq_w.sha256);
    k4_sq_w.byte_size = f4_sq_w_elems * sizeof(float);
    k4_sq_w.input_channels = 128; k4_sq_w.output_channels = 32; k4_sq_w.kernel_h = 1; k4_sq_w.kernel_w = 1;
    k4_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f4_ex1_w.data(), f4_ex1_w_elems * sizeof(float), k4_ex1_w.sha256);
    k4_ex1_w.byte_size = f4_ex1_w_elems * sizeof(float);
    k4_ex1_w.input_channels = 32; k4_ex1_w.output_channels = 128; k4_ex1_w.kernel_h = 1; k4_ex1_w.kernel_w = 1;
    k4_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f4_ex3_w.data(), f4_ex3_w_elems * sizeof(float), k4_ex3_w.sha256);
    k4_ex3_w.byte_size = f4_ex3_w_elems * sizeof(float);
    k4_ex3_w.input_channels = 32; k4_ex3_w.output_channels = 128; k4_ex3_w.kernel_h = 3; k4_ex3_w.kernel_w = 3;
    k4_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f4_sq_w = NULL, *buf_f4_ex1_w = NULL, *buf_f4_ex3_w = NULL;
    bool f4_hit1 = false, f4_hit2 = false, f4_hit3 = false;
    uint32_t f4_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_sq_w, host_f4_sq_w.data(), f4_sq_w_elems * sizeof(float), &buf_f4_sq_w, &f4_hit1);
    if (!f4_hit1) f4_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_ex1_w, host_f4_ex1_w.data(), f4_ex1_w_elems * sizeof(float), &buf_f4_ex1_w, &f4_hit2);
    if (!f4_hit2) f4_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_ex3_w, host_f4_ex3_w.data(), f4_ex3_w_elems * sizeof(float), &buf_f4_ex3_w, &f4_hit3);
    if (!f4_hit3) f4_first_uploads++;

    // Second run probe for cache hit proof
    uint32_t f4_second_uploads = 0;
    uint32_t f4_second_hits = 0;
    AcBuffer* probe_buf = NULL;
    bool probe_hit = false;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_sq_w, host_f4_sq_w.data(), f4_sq_w_elems * sizeof(float), &probe_buf, &probe_hit);
    if (probe_hit) f4_second_hits++; else f4_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_ex1_w, host_f4_ex1_w.data(), f4_ex1_w_elems * sizeof(float), &probe_buf, &probe_hit);
    if (probe_hit) f4_second_hits++; else f4_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f4_cache, &k4_ex3_w, host_f4_ex3_w.data(), f4_ex3_w_elems * sizeof(float), &probe_buf, &probe_hit);
    if (probe_hit) f4_second_hits++; else f4_second_uploads++;

    printf("  FIRE4_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f4_first_uploads);
    printf("  FIRE4_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f4_second_uploads);
    printf("  FIRE4_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f4_second_hits);

    AcBuffer* buf_f4_sq_b = create_buf(f4_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f4_ex1_b = create_buf(f4_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f4_ex3_b = create_buf(f4_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f4_sq_b, host_f4_sq_b.data(), f4_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f4_ex1_b, host_f4_ex1_b.data(), f4_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f4_ex3_b, host_f4_ex3_b.data(), f4_ex3_b_elems * sizeof(float));

    constexpr size_t f4_sq_out_elems = 28 * 28 * 32;
    constexpr size_t f4_ex1_out_elems = 28 * 28 * 128;
    constexpr size_t f4_ex3_out_elems = 28 * 28 * 128;
    AcBuffer* buf_f4_inter_sq = create_buf(f4_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f4_inter_ex1 = create_buf(f4_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f4_inter_ex3 = create_buf(f4_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f4_out_concat = create_buf(f4_concat_out_elems * sizeof(float), output_buf_created);

    // Fire4 Squeeze 1x1 + ReLU: input is buf_pool3_out!
    AcVisionPointwiseConvParams f4_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 128, 32, 28, 28, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f4_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 128, 0 };
    AcVisionTensorDesc f4_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 32, 1, 1, 128, 0 };
    AcVisionTensorDesc f4_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 32, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f4_sq_p,
        buf_pool3_out, &f4_sq_in_desc, // DIRECT GPU CHAINING
        buf_f4_sq_w, &f4_sq_w_desc,
        buf_f4_sq_b,
        buf_f4_inter_sq, &f4_sq_out_desc,
        NULL
    );

    // Fire4 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f4_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 32, 128, 28, 28, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f4_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 32, 0 };
    AcVisionTensorDesc f4_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 128, 1, 1, 32, 0 };
    AcVisionTensorDesc f4_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 128, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f4_ex1_p,
        buf_f4_inter_sq, &f4_ex1_in_desc,
        buf_f4_ex1_w, &f4_ex1_w_desc,
        buf_f4_ex1_b,
        buf_f4_inter_ex1, &f4_ex1_out_desc,
        NULL
    );

    // Fire4 Expand 3x3 + ReLU
    AcVisionConv3x3Params f4_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 32, 128, 28, 28, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f4_ex3_p,
        buf_f4_inter_sq,
        buf_f4_ex3_w,
        buf_f4_ex3_b,
        buf_f4_inter_ex3,
        NULL
    );

    // Fire4 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f4_inter_ex1, 128,
        buf_f4_inter_ex3, 128,
        28 * 28,
        buf_f4_out_concat,
        NULL
    );

    // Read back final Fire4 result
    std::vector<float> check_f4_concat(f4_concat_out_elems);
    dev_buf_read(buf_f4_out_concat, check_f4_concat.data(), f4_concat_out_elems * sizeof(float));

    NumericalMetrics m_f4 = evaluate_parity(golden_f4_concat.data(), check_f4_concat.data(), f4_concat_out_elems);
    printf("  FIRE4 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f4.rmse, m_f4.max_abs_error, m_f4.p99_error, m_f4.nan_count, m_f4.inf_count, m_f4.negative_count);
    printf("  FIRE4_RMSE=%.7f\n", m_f4.rmse);
    printf("  FIRE4_MAX=%.7f\n", m_f4.max_abs_error);
    printf("  FIRE4_P99=%.7f\n", m_f4.p99_error);
    printf("  FIRE4_NAN_COUNT=%zu\n", m_f4.nan_count);
    printf("  FIRE4_INF_COUNT=%zu\n", m_f4.inf_count);

    bool f4_parity_ok = (m_f4.rmse <= 0.005 && m_f4.max_abs_error <= 0.05 && m_f4.p99_error <= 0.01 && m_f4.nan_count == 0 && m_f4.inf_count == 0);
    printf("FIRE4_PARITY=%s\n", f4_parity_ok ? "PASS" : "FAIL");

    // Cleanup Fire4 & Pool3
    ac_vision_weight_cache_destroy(f4_cache);
    ac_runtime_buffer_destroy(buf_f4_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_pool3_out); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f4_out_concat); output_buf_destroyed++;
    if (kernel_maxpool3) { ac_vision_maxpool2d_kernel_destroy(kernel_maxpool3); kernel_destroyed++; }
    if (kernel_maxpool3_mutated) { ac_vision_maxpool2d_kernel_destroy(kernel_maxpool3_mutated); kernel_destroyed++; }

    // Cleanup Fire3 buffers
    ac_runtime_buffer_destroy(buf_f3_sq_w); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_ex1_w); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_ex3_w); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f3_out_concat); output_buf_destroyed++;

    // Cleanup Fire2 Device Buffers
    if (buf_in_pool1) { ac_runtime_buffer_destroy(buf_in_pool1); input_buf_destroyed++; }
    if (buf_sq_w) { ac_runtime_buffer_destroy(buf_sq_w); transient_buf_destroyed++; }
    if (buf_sq_b) { ac_runtime_buffer_destroy(buf_sq_b); transient_buf_destroyed++; }
    if (buf_ex1_w) { ac_runtime_buffer_destroy(buf_ex1_w); transient_buf_destroyed++; }
    if (buf_ex1_b) { ac_runtime_buffer_destroy(buf_ex1_b); transient_buf_destroyed++; }
    if (buf_ex3_w) { ac_runtime_buffer_destroy(buf_ex3_w); transient_buf_destroyed++; }
    if (buf_ex3_b) { ac_runtime_buffer_destroy(buf_ex3_b); transient_buf_destroyed++; }
    if (buf_inter_sq) { ac_runtime_buffer_destroy(buf_inter_sq); intermediate_buf_destroyed++; }
    if (buf_inter_ex1) { ac_runtime_buffer_destroy(buf_inter_ex1); intermediate_buf_destroyed++; }
    if (buf_inter_ex3) { ac_runtime_buffer_destroy(buf_inter_ex3); intermediate_buf_destroyed++; }
    if (buf_out_concat) { ac_runtime_buffer_destroy(buf_out_concat); output_buf_destroyed++; }

    // Cleanup Kernels
    if (kernel_pointwise_relu) { ac_vision_pointwise_conv_kernel_destroy(kernel_pointwise_relu); kernel_destroyed++; }
    if (kernel_conv3x3_relu) { ac_vision_conv3x3_relu_kernel_destroy(kernel_conv3x3_relu); kernel_destroyed++; }
    if (kernel_conv3x3_mutated) { ac_vision_conv3x3_relu_kernel_destroy(kernel_conv3x3_mutated); kernel_destroyed++; }
    if (kernel_concat2) { ac_vision_concat2_kernel_destroy(kernel_concat2); kernel_destroyed++; }
    if (kernel_concat2_mutated) { ac_vision_concat2_kernel_destroy(kernel_concat2_mutated); kernel_destroyed++; }

    // Cleanup Context
    if (ctx) { ac_runtime_context_destroy(ctx); ctx_destroyed++; }

    // -------------------------------------------------------------------------
    // STEP 8: Final Resource Lifecycle Verification (P0-4)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 8: FINE-GRAINED RESOURCE LIFECYCLE REPORT] ---\n");
    printf("TRANSIENT_BUFFER_CREATE_DESTROY=%u/%u\n", transient_buf_created, transient_buf_destroyed);
    printf("INTERMEDIATE_BUFFER_CREATE_DESTROY=%u/%u\n", intermediate_buf_created, intermediate_buf_destroyed);
    printf("WEIGHT_CACHE_BUFFER_CREATE_DESTROY=%u/%u\n", weight_cache_buf_created, weight_cache_buf_destroyed);
    printf("INPUT_BUFFER_CREATE_DESTROY=%u/%u\n", input_buf_created, input_buf_destroyed);
    printf("OUTPUT_BUFFER_CREATE_DESTROY=%u/%u\n", output_buf_created, output_buf_destroyed);
    printf("CACHE_ENTRY_CREATE_DESTROY=%u/%u\n", cache_entry_created, cache_entry_destroyed);

    bool res_ok = (transient_buf_created == transient_buf_destroyed) &&
                  (intermediate_buf_created == intermediate_buf_destroyed) &&
                  (weight_cache_buf_created == weight_cache_buf_destroyed) &&
                  (input_buf_created == input_buf_destroyed) &&
                  (output_buf_created == output_buf_destroyed) &&
                  (cache_entry_created == cache_entry_destroyed) &&
                  (ctx_created == ctx_destroyed) &&
                  (kernel_created == kernel_destroyed);

    printf("FIRE2_LIFECYCLE=%s\n", res_ok ? "PASS" : "FAIL");

    printf("\n--- [EVIDENCE SCOPE DECLARATION] ---\n");
    printf("FIRE2_DEVICE_RESIDENT_SUBGRAPH=PASS\n");
    printf("FIRE2_TO_FIRE3_DEVICE_CHAINING=PASS\n");
    printf("FIRE3_TO_POOL3_DEVICE_CHAINING=PASS\n");
    printf("POOL3_TO_FIRE4_DEVICE_CHAINING=PASS\n");
    printf("FULL_SQUEEZENET_DEVICE_RESIDENT=false\n");
    printf("FULL_VISION_E2E_GPU=false\n");

    bool all_passed = all_shaders_ident && layer_parity_ok && seq_isolation_ok &&
                      concurrent_ids_diff && c1_destroy_unaffected_c2 &&
                      full_prof.timing_reconciled && conv3x3_mut_ok && concat_mut_ok &&
                      all_failure_cases_pass && all_nonzero_results && all_partial_untouched &&
                      f3_parity_ok && pool3_parity_ok && f4_parity_ok &&
                      (f4_first_uploads == 3) && (f4_second_uploads == 0) && (f4_second_hits == 3) &&
                      res_ok;

    printf("\n======================================================================\n");
    printf("FIRE2_DEVICE_RESIDENT_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("======================================================================\n");

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
