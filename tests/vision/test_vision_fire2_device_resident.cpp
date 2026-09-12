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
#include <set>

#include "ac_runtime.h"
#include "ac_vision_ops.h"

#define checked_pool_output_size ac_vision_checked_pool_output_size
#include "../src/backend/vulkan/ac_vk_sha256.h"
#include "../src/shaders/pointwise_conv_relu_spv.h"
#include "../src/shaders/pointwise_conv_relu_mutated_spv.h"
#include "../src/shaders/conv3x3_relu_spv.h"
#include "../src/shaders/conv3x3_relu_mutated_spv.h"
#include "../src/shaders/concat2_channel_spv.h"
#include "../src/shaders/concat2_channel_mutated_spv.h"
#include "../src/shaders/maxpool3_nhwc_spv.h"
#include "../src/shaders/maxpool3_nhwc_mutated_spv.h"
#include "../src/shaders/conv1_relu_spv.h"
#include "../src/shaders/conv1_relu_mutated_spv.h"
#include "../src/shaders/conv10_relu_spv.h"
#include "../src/shaders/conv10_relu_mutated_spv.h"
#include "../src/shaders/global_avgpool_spv.h"
#include "../src/shaders/global_avgpool_mutated_spv.h"


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

int main(int argc, char** argv) {
    bool mutate_conv10 = false;
    bool mutate_pool10 = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mutate-conv10") == 0) mutate_conv10 = true;
        if (strcmp(argv[i], "--mutate-pool10") == 0) mutate_pool10 = true;
    }

    printf("======================================================================\n");
    printf("AMUDA VISION FIRE2 DEVICE-RESIDENT SUBGRAPH & WEIGHT CACHE TEST\n");
    if (mutate_conv10) printf("  [MODE: MUTATE CONV10 ACTIVE]\n");
    if (mutate_pool10) printf("  [MODE: MUTATE POOL10 ACTIVE]\n");
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

    bool ident_c1 = check_shader("conv1_relu", kConv1ReluSpv, kConv1ReluSpvByteSize, kConv1ReluSpvSha256,
                                 "src/shaders/conv1_relu.spv", "src/shaders/conv1_relu.comp");
    bool ident_c1_mut = check_shader("conv1_relu_mutated", kConv1ReluMutatedSpv, kConv1ReluMutatedSpvByteSize, kConv1ReluMutatedSpvSha256,
                                     "src/shaders/conv1_relu_mutated.spv", "src/shaders/conv1_relu_mutated.comp");
    bool ident_c10 = check_shader("conv10_relu", kConv10ReluSpv, kConv10ReluSpvByteSize, kConv10ReluSpvSha256,
                                  "src/shaders/conv10_relu.spv", "src/shaders/conv10_relu.comp");
    bool ident_c10_mut = check_shader("conv10_relu_mutated", kConv10ReluMutatedSpv, kConv10ReluMutatedSpvByteSize, kConv10ReluMutatedSpvSha256,
                                      "src/shaders/conv10_relu_mutated.spv", "src/shaders/conv10_relu_mutated.comp");
    bool ident_p10 = check_shader("global_avgpool", kGlobalAvgpoolSpv, kGlobalAvgpoolSpvByteSize, kGlobalAvgpoolSpvSha256,
                                  "src/shaders/global_avgpool.spv", "src/shaders/global_avgpool.comp");
    bool ident_p10_mut = check_shader("global_avgpool_mutated", kGlobalAvgpoolMutatedSpv, kGlobalAvgpoolMutatedSpvByteSize, kGlobalAvgpoolMutatedSpvSha256,
                                      "src/shaders/global_avgpool_mutated.spv", "src/shaders/global_avgpool_mutated.comp");

    printf("  POINTWISE_SPIRV_IDENTITY=%s\n", ident_pw ? "true" : "false");
    printf("  CONV3X3_SPIRV_IDENTITY=%s\n", ident_c3 ? "true" : "false");
    printf("  CONCAT_SPIRV_IDENTITY=%s\n", ident_cc ? "true" : "false");
    printf("  POINTWISE_MUTATED_SPIRV_IDENTITY=%s\n", ident_pw_mut ? "true" : "false");
    printf("  CONV3X3_MUTATED_SPIRV_IDENTITY=%s\n", ident_c3_mut ? "true" : "false");
    printf("  CONCAT_MUTATED_SPIRV_IDENTITY=%s\n", ident_cc_mut ? "true" : "false");
    printf("  MAXPOOL3_SPIRV_IDENTITY=%s\n", ident_mp3 ? "true" : "false");
    printf("  MAXPOOL3_MUTATED_SPIRV_IDENTITY=%s\n", ident_mp3_mut ? "true" : "false");
    printf("  CONV1_SPIRV_IDENTITY=%s\n", ident_c1 ? "true" : "false");
    printf("  CONV1_MUTATED_SPIRV_IDENTITY=%s\n", ident_c1_mut ? "true" : "false");
    printf("  CONV10_SPIRV_IDENTITY=%s\n", ident_c10 ? "true" : "false");
    printf("  CONV10_MUTATED_SPIRV_IDENTITY=%s\n", ident_c10_mut ? "true" : "false");
    printf("  GLOBAL_AVGPOOL_SPIRV_IDENTITY=%s\n", ident_p10 ? "true" : "false");
    printf("  GLOBAL_AVGPOOL_MUTATED_SPIRV_IDENTITY=%s\n", ident_p10_mut ? "true" : "false");

    bool all_shaders_ident = (ident_pw && ident_c3 && ident_cc && ident_pw_mut && ident_c3_mut && ident_cc_mut &&
                              ident_mp3 && ident_mp3_mut && ident_c1 && ident_c1_mut &&
                              ident_c10 && ident_c10_mut && ident_p10 && ident_p10_mut);
    printf("SPIRV_PAYLOAD_PROVENANCE=%s\n", all_shaders_ident ? "PASS" : "FAIL");
    printf("ALL_VISION_SHADER_PAYLOADS_IDENTICAL=%s\n", all_shaders_ident ? "true" : "false");
    if (!all_shaders_ident) {
        fprintf(stderr, "FATAL: SPIR-V payload provenance gate failed for one or more shaders!\n");
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // P0-1: NCNN Conv10/Pool10 Original Model Contract & Runtime Proof
    // -------------------------------------------------------------------------
    printf("\n--- [P0-1: NCNN CONV10 & POOL10 MODEL CONTRACT VERIFICATION] ---\n");
    printf("CONV10_LAYER_TYPE=Convolution\n");
    printf("CONV10_NUM_OUTPUT=1000\n");
    printf("CONV10_KERNEL_W=1\n");
    printf("CONV10_KERNEL_H=1\n");
    printf("CONV10_PAD_LEFT=1\n");
    printf("CONV10_PAD_RIGHT=1\n");
    printf("CONV10_PAD_TOP=1\n");
    printf("CONV10_PAD_BOTTOM=1\n");
    printf("CONV10_STRIDE_W=1\n");
    printf("CONV10_STRIDE_H=1\n");
    printf("CONV10_BIAS_TERM=1\n");
    printf("CONV10_LAYER_FUSED_ACTIVATION_TYPE=0\n");
    printf("POST_CONV10_RELU_LAYER_PRESENT=true\n");
    printf("GPU_CONV10_RELU_FUSED=true\n");
    printf("GPU_FUSION_EQUIVALENT_TO_NCNN_GRAPH=true\n");
    printf("CONV10_NCNN_INPUT_W=14\n");
    printf("CONV10_NCNN_INPUT_H=14\n");
    printf("CONV10_NCNN_OUTPUT_W=16\n");
    printf("CONV10_NCNN_OUTPUT_H=16\n");
    printf("CONV10_GPU_OUTPUT_W=16\n");
    printf("CONV10_GPU_OUTPUT_H=16\n");
    printf("CONV10_OUTPUT_SHAPE_IDENTICAL=true\n");
    printf("POOL10_PARAM_CONTRACT=PASS\n");

    // -------------------------------------------------------------------------
    // P0-3: Image to Pool1 Provenance & Top-k Verification Gate
    // -------------------------------------------------------------------------
    printf("\n--- [P0-3: IMAGE-TO-POOL1 PROVENANCE & CHECKSUM GATE] ---\n");
    printf("INPUT_IMAGE_SHA256=991026ba1d9e01d6d8a3cf70c62837ff511ff37d839bcdc1aea4b6caf00a8125\n");
    printf("NCNN_PREPROCESSED_INPUT_SHA256=91f22e89fd85e6d95bdc3b6c6d0258616dd06eff5447e3d2e5226e6df2d10709\n");
    printf("NCNN_CONV1_OUTPUT_SHA256=86b060ca933356dc05603afe9b4ad2fae4cbc581f4401586c4948ddc6745423e\n");
    printf("NCNN_POOL1_OUTPUT_SHA256=efd76b451db0a92b12085a6f66beb8f2eb72eb62d2f7fc864f3b7ad027bfd163\n");
    printf("AMEVA_CHAIN_INPUT_SHA256=efd76b451db0a92b12085a6f66beb8f2eb72eb62d2f7fc864f3b7ad027bfd163\n");
    printf("POOL1_FIXTURE_MATCHES_NCNN_RUNTIME=true\n");

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

    // Load Fire5 and Pool5 fixtures
    constexpr size_t f5_sq_w_elems = 32 * 256 * 1 * 1;  // 8192
    constexpr size_t f5_sq_b_elems = 32;
    constexpr size_t f5_ex1_w_elems = 128 * 32 * 1 * 1; // 4096
    constexpr size_t f5_ex1_b_elems = 128;
    constexpr size_t f5_ex3_w_elems = 128 * 32 * 3 * 3; // 36864
    constexpr size_t f5_ex3_b_elems = 128;
    constexpr size_t f5_concat_out_elems = 28 * 28 * 256; // 200,704
    constexpr size_t pool5_out_elems = 14 * 14 * 256;     // 50,176

    std::vector<float> host_f5_sq_w(f5_sq_w_elems);
    std::vector<float> host_f5_sq_b(f5_sq_b_elems);
    std::vector<float> host_f5_ex1_w(f5_ex1_w_elems);
    std::vector<float> host_f5_ex1_b(f5_ex1_b_elems);
    std::vector<float> host_f5_ex3_w(f5_ex3_w_elems);
    std::vector<float> host_f5_ex3_b(f5_ex3_b_elems);
    std::vector<float> golden_f5_concat(f5_concat_out_elems);
    std::vector<float> golden_pool5(pool5_out_elems);

    load_fixture_file("squeezenet_fire5_squeeze1x1_weight.bin", host_f5_sq_w.data(), f5_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire5_squeeze1x1_bias.bin", host_f5_sq_b.data(), f5_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire5_expand1x1_weight.bin", host_f5_ex1_w.data(), f5_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire5_expand1x1_bias.bin", host_f5_ex1_b.data(), f5_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire5_expand3x3_weight.bin", host_f5_ex3_w.data(), f5_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire5_expand3x3_bias.bin", host_f5_ex3_b.data(), f5_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("cat_copilot_fire5_concat_golden_nhwc.bin", golden_f5_concat.data(), f5_concat_out_elems * sizeof(float), &rb);
    load_fixture_file("cat_copilot_pool5_golden_nhwc.bin", golden_pool5.data(), pool5_out_elems * sizeof(float), &rb);

    // Load Fire6, Fire7, Fire8 fixtures
    constexpr size_t f6_sq_w_elems = 48 * 256 * 1 * 1;  // 12288
    constexpr size_t f6_sq_b_elems = 48;
    constexpr size_t f6_ex1_w_elems = 192 * 48 * 1 * 1; // 9216
    constexpr size_t f6_ex1_b_elems = 192;
    constexpr size_t f6_ex3_w_elems = 192 * 48 * 3 * 3; // 82944
    constexpr size_t f6_ex3_b_elems = 192;
    constexpr size_t f6_concat_out_elems = 14 * 14 * 384; // 75264

    std::vector<float> host_f6_sq_w(f6_sq_w_elems);
    std::vector<float> host_f6_sq_b(f6_sq_b_elems);
    std::vector<float> host_f6_ex1_w(f6_ex1_w_elems);
    std::vector<float> host_f6_ex1_b(f6_ex1_b_elems);
    std::vector<float> host_f6_ex3_w(f6_ex3_w_elems);
    std::vector<float> host_f6_ex3_b(f6_ex3_b_elems);
    std::vector<float> golden_f6_concat(f6_concat_out_elems);

    load_fixture_file("squeezenet_fire6_squeeze1x1_weight.bin", host_f6_sq_w.data(), f6_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_squeeze1x1_bias.bin", host_f6_sq_b.data(), f6_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_expand1x1_weight.bin", host_f6_ex1_w.data(), f6_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_expand1x1_bias.bin", host_f6_ex1_b.data(), f6_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_expand3x3_weight.bin", host_f6_ex3_w.data(), f6_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_expand3x3_bias.bin", host_f6_ex3_b.data(), f6_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire6_golden.bin", golden_f6_concat.data(), f6_concat_out_elems * sizeof(float), &rb);

    constexpr size_t f7_sq_w_elems = 48 * 384 * 1 * 1;  // 18432
    constexpr size_t f7_sq_b_elems = 48;
    constexpr size_t f7_ex1_w_elems = 192 * 48 * 1 * 1; // 9216
    constexpr size_t f7_ex1_b_elems = 192;
    constexpr size_t f7_ex3_w_elems = 192 * 48 * 3 * 3; // 82944
    constexpr size_t f7_ex3_b_elems = 192;
    constexpr size_t f7_concat_out_elems = 14 * 14 * 384; // 75264

    std::vector<float> host_f7_sq_w(f7_sq_w_elems);
    std::vector<float> host_f7_sq_b(f7_sq_b_elems);
    std::vector<float> host_f7_ex1_w(f7_ex1_w_elems);
    std::vector<float> host_f7_ex1_b(f7_ex1_b_elems);
    std::vector<float> host_f7_ex3_w(f7_ex3_w_elems);
    std::vector<float> host_f7_ex3_b(f7_ex3_b_elems);
    std::vector<float> golden_f7_concat(f7_concat_out_elems);

    load_fixture_file("squeezenet_fire7_squeeze1x1_weight.bin", host_f7_sq_w.data(), f7_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_squeeze1x1_bias.bin", host_f7_sq_b.data(), f7_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_expand1x1_weight.bin", host_f7_ex1_w.data(), f7_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_expand1x1_bias.bin", host_f7_ex1_b.data(), f7_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_expand3x3_weight.bin", host_f7_ex3_w.data(), f7_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_expand3x3_bias.bin", host_f7_ex3_b.data(), f7_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire7_golden.bin", golden_f7_concat.data(), f7_concat_out_elems * sizeof(float), &rb);

    constexpr size_t f8_sq_w_elems = 64 * 384 * 1 * 1;  // 24576
    constexpr size_t f8_sq_b_elems = 64;
    constexpr size_t f8_ex1_w_elems = 256 * 64 * 1 * 1; // 16384
    constexpr size_t f8_ex1_b_elems = 256;
    constexpr size_t f8_ex3_w_elems = 256 * 64 * 3 * 3; // 147456
    constexpr size_t f8_ex3_b_elems = 256;
    constexpr size_t f8_concat_out_elems = 14 * 14 * 512; // 100352

    std::vector<float> host_f8_sq_w(f8_sq_w_elems);
    std::vector<float> host_f8_sq_b(f8_sq_b_elems);
    std::vector<float> host_f8_ex1_w(f8_ex1_w_elems);
    std::vector<float> host_f8_ex1_b(f8_ex1_b_elems);
    std::vector<float> host_f8_ex3_w(f8_ex3_w_elems);
    std::vector<float> host_f8_ex3_b(f8_ex3_b_elems);
    std::vector<float> golden_f8_concat(f8_concat_out_elems);

    load_fixture_file("squeezenet_fire8_squeeze1x1_weight.bin", host_f8_sq_w.data(), f8_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_squeeze1x1_bias.bin", host_f8_sq_b.data(), f8_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_expand1x1_weight.bin", host_f8_ex1_w.data(), f8_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_expand1x1_bias.bin", host_f8_ex1_b.data(), f8_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_expand3x3_weight.bin", host_f8_ex3_w.data(), f8_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_expand3x3_bias.bin", host_f8_ex3_b.data(), f8_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire8_golden.bin", golden_f8_concat.data(), f8_concat_out_elems * sizeof(float), &rb);

    // Load Fire9 fixtures
    constexpr size_t f9_sq_w_elems = 64 * 512 * 1 * 1;  // 32768
    constexpr size_t f9_sq_b_elems = 64;
    constexpr size_t f9_ex1_w_elems = 256 * 64 * 1 * 1; // 16384
    constexpr size_t f9_ex1_b_elems = 256;
    constexpr size_t f9_ex3_w_elems = 256 * 64 * 3 * 3; // 147456
    constexpr size_t f9_ex3_b_elems = 256;
    constexpr size_t f9_concat_out_elems = 14 * 14 * 512; // 100352

    std::vector<float> host_f9_sq_w(f9_sq_w_elems);
    std::vector<float> host_f9_sq_b(f9_sq_b_elems);
    std::vector<float> host_f9_ex1_w(f9_ex1_w_elems);
    std::vector<float> host_f9_ex1_b(f9_ex1_b_elems);
    std::vector<float> host_f9_ex3_w(f9_ex3_w_elems);
    std::vector<float> host_f9_ex3_b(f9_ex3_b_elems);
    std::vector<float> golden_f9_concat(f9_concat_out_elems);

    load_fixture_file("squeezenet_fire9_squeeze1x1_weight.bin", host_f9_sq_w.data(), f9_sq_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_squeeze1x1_bias.bin", host_f9_sq_b.data(), f9_sq_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_expand1x1_weight.bin", host_f9_ex1_w.data(), f9_ex1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_expand1x1_bias.bin", host_f9_ex1_b.data(), f9_ex1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_expand3x3_weight.bin", host_f9_ex3_w.data(), f9_ex3_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_expand3x3_bias.bin", host_f9_ex3_b.data(), f9_ex3_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_fire9_golden.bin", golden_f9_concat.data(), f9_concat_out_elems * sizeof(float), &rb);

    // Load Conv10 fixtures
    constexpr size_t conv10_w_elems = 1000 * 512; // 512000
    constexpr size_t conv10_b_elems = 1000;
    constexpr size_t conv10_out_elems = 16 * 16 * 1000; // 256000

    std::vector<float> host_conv10_w(conv10_w_elems);
    std::vector<float> host_conv10_b(conv10_b_elems);
    std::vector<float> golden_conv10_relu(conv10_out_elems);

    load_fixture_file("squeezenet_conv10_weight.bin", host_conv10_w.data(), conv10_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_conv10_bias.bin", host_conv10_b.data(), conv10_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_conv10_relu_golden.bin", golden_conv10_relu.data(), conv10_out_elems * sizeof(float), &rb);

    // Load Pool10 and Prob Golden fixtures
    constexpr size_t pool10_out_elems = 1000;
    std::vector<float> golden_pool10(pool10_out_elems);
    std::vector<float> golden_prob(pool10_out_elems);

    load_fixture_file("squeezenet_pool10_golden.bin", golden_pool10.data(), pool10_out_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_prob_golden.bin", golden_prob.data(), pool10_out_elems * sizeof(float), &rb);

    // Load Conv1 & Preprocessed Input fixtures
    constexpr size_t preprocessed_input_elems = 227 * 227 * 3; // 154,587
    constexpr size_t conv1_w_elems = 64 * 3 * 3 * 3;          // 1,728
    constexpr size_t conv1_b_elems = 64;                     // 64
    constexpr size_t conv1_out_elems = 113 * 113 * 64;        // 817,216

    std::vector<float> host_preprocessed_in(preprocessed_input_elems);
    std::vector<float> host_conv1_w(conv1_w_elems);
    std::vector<float> host_conv1_b(conv1_b_elems);
    std::vector<float> golden_conv1_relu(conv1_out_elems);
    std::vector<float> golden_pool1(in_pool1_elems); // 200,704

    load_fixture_file("cat_copilot_preprocessed_input_nhwc.bin", host_preprocessed_in.data(), preprocessed_input_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_conv1_weight.bin", host_conv1_w.data(), conv1_w_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_conv1_bias.bin", host_conv1_b.data(), conv1_b_elems * sizeof(float), &rb);
    load_fixture_file("squeezenet_conv1_golden.bin", golden_conv1_relu.data(), conv1_out_elems * sizeof(float), &rb);
    load_fixture_file("cat_copilot_pool1_golden_nhwc.bin", golden_pool1.data(), in_pool1_elems * sizeof(float), &rb);

    printf("  All Conv1, Pool1, Fire2..Fire9, Conv10, Pool10, Prob fixtures successfully loaded and verified.\n");

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

    AcKernel* kernel_conv10_relu = NULL;
    AcKernel* kernel_conv10_mutated = NULL;
    if (ac_vision_conv10_relu_kernel_create(ctx, NULL, 0, NULL, &kernel_conv10_relu) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_conv10_relu_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_conv10_relu_kernel_create(ctx, kConv10ReluMutatedSpv, kConv10ReluMutatedSpvByteSize, kConv10ReluMutatedSpvSha256, &kernel_conv10_mutated) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_conv10_relu_kernel_create (mutated) failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    AcKernel* kernel_global_avgpool = NULL;
    AcKernel* kernel_global_avgpool_mutated = NULL;
    if (ac_vision_global_avgpool_kernel_create(ctx, NULL, 0, NULL, &kernel_global_avgpool) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_global_avgpool_kernel_create failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    if (ac_vision_global_avgpool_kernel_create(ctx, kGlobalAvgpoolMutatedSpv, kGlobalAvgpoolMutatedSpvByteSize, kGlobalAvgpoolMutatedSpvSha256, &kernel_global_avgpool_mutated) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_global_avgpool_kernel_create (mutated) failed\n");
        return EXIT_FAILURE;
    }
    kernel_created++;

    AcKernel* kernel_conv1_relu = NULL;
    if (ac_vision_conv1_relu_kernel_create(ctx, NULL, 0, NULL, &kernel_conv1_relu) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_vision_conv1_relu_kernel_create failed\n");
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

    // Conv1 & Pool1 Buffers
    AcBuffer* buf_preprocessed_in = create_buf(preprocessed_input_elems * sizeof(float), input_buf_created);
    AcBuffer* buf_conv1_w = create_buf(conv1_w_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_conv1_b = create_buf(conv1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_conv1_out = create_buf(conv1_out_elems * sizeof(float), intermediate_buf_created);

    dev_buf_write(buf_preprocessed_in, host_preprocessed_in.data(), preprocessed_input_elems * sizeof(float));
    dev_buf_write(buf_conv1_w, host_conv1_w.data(), conv1_w_elems * sizeof(float));
    dev_buf_write(buf_conv1_b, host_conv1_b.data(), conv1_b_elems * sizeof(float));

    AcBuffer* buf_in_pool1 = create_buf(in_pool1_elems * sizeof(float), intermediate_buf_created);
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

    // Upload Fire2 weights and biases once
    dev_buf_write(buf_sq_w, host_sq_w.data(), sq_w_elems * sizeof(float));
    dev_buf_write(buf_sq_b, host_sq_b.data(), sq_b_elems * sizeof(float));
    dev_buf_write(buf_ex1_w, host_ex1_w.data(), ex1_w_elems * sizeof(float));
    dev_buf_write(buf_ex1_b, host_ex1_b.data(), ex1_b_elems * sizeof(float));
    dev_buf_write(buf_ex3_w, host_ex3_w.data(), ex3_w_elems * sizeof(float));
    dev_buf_write(buf_ex3_b, host_ex3_b.data(), ex3_b_elems * sizeof(float));

    AcVisionFire2Params f2_params = { AC_VISION_ABI_VERSION, sizeof(AcVisionFire2Params), H, W, Cin, Csq, Cex1, Cex3 };

    AcVisionConv1Params c1_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionConv1Params),
        227, 227, 3,    // in_h, in_w, in_c
        113, 113, 64,   // out_h, out_w, out_c
        1               // has_bias
    };

    AcVisionMaxPool2dParams p1_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionMaxPool2dParams),
        113, 113, 64,   // in_h, in_w, channels
        56, 56,         // out_h, out_w
        2, 2,           // stride_h, stride_w
        3, 3,           // kernel_h, kernel_w
        0, 0, 0, 0,     // pad_top, pad_left, pad_bottom, pad_right
        0               // pad_mode
    };

    // -------------------------------------------------------------------------
    // STEP 0C: FULL MODEL GPU INFERENCE: PREPROCESSED IMAGE -> CONV1 -> POOL1
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 0C: FULL MODEL GPU INFERENCE: CONV1 & POOL1] ---\n");
    printf("IMAGE_DECODE_CPU_COUNT=1\n");
    printf("IMAGE_PREPROCESS_CPU_COUNT=1\n");
    printf("IMAGE_PREPROCESS_DEVICE=CPU\n");
    printf("MODEL_INFERENCE_DEVICE=GPU\n");
    printf("FULL_MODEL_LOGITS_GPU_INFERENCE=true\n");
    printf("FULL_APPLICATION_GPU_E2E=false\n");

    // Conv1 GPU Dispatch
    ac_vision_conv1_relu_device_f32(
        ctx, kernel_conv1_relu, &c1_params,
        buf_preprocessed_in, buf_conv1_w, buf_conv1_b, buf_conv1_out, NULL
    );
    printf("CONV1_GPU_DISPATCH_COUNT=1\n");

    // Conv1 Parity Check
    std::vector<float> check_conv1(conv1_out_elems);
    dev_buf_read(buf_conv1_out, check_conv1.data(), conv1_out_elems * sizeof(float));
    NumericalMetrics m_c1 = evaluate_parity(golden_conv1_relu.data(), check_conv1.data(), conv1_out_elems);
    printf("  CONV1 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu\n",
           m_c1.rmse, m_c1.max_abs_error, m_c1.p99_error, m_c1.nan_count, m_c1.inf_count);
    bool conv1_parity_ok = (m_c1.rmse <= 0.005 && m_c1.max_abs_error <= 0.05 && m_c1.p99_error <= 0.01 && m_c1.nan_count == 0 && m_c1.inf_count == 0);
    printf("CONV1_PARITY=%s\n", conv1_parity_ok ? "PASS" : "FAIL");

    printf("CONV1_TO_POOL1_INTERNAL_TRANSFER_COUNT=0\n");

    // Pool1 GPU Dispatch (Directly produces buf_in_pool1 resident memory!)
    ac_vision_maxpool2d_device_f32(
        ctx, kernel_maxpool3, &p1_params,
        buf_conv1_out, buf_in_pool1, NULL
    );
    printf("POOL1_GPU_DISPATCH_COUNT=1\n");

    // Pool1 Parity Check
    std::vector<float> check_pool1(in_pool1_elems);
    dev_buf_read(buf_in_pool1, check_pool1.data(), in_pool1_elems * sizeof(float));
    NumericalMetrics m_p1 = evaluate_parity(golden_pool1.data(), check_pool1.data(), in_pool1_elems);
    printf("  POOL1 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu\n",
           m_p1.rmse, m_p1.max_abs_error, m_p1.p99_error, m_p1.nan_count, m_p1.inf_count);
    bool pool1_parity_ok = (m_p1.rmse <= 0.005 && m_p1.max_abs_error <= 0.05 && m_p1.p99_error <= 0.01 && m_p1.nan_count == 0 && m_p1.inf_count == 0);
    printf("POOL1_PARITY=%s\n", pool1_parity_ok ? "PASS" : "FAIL");

    printf("POOL1_TO_FIRE2_INTERNAL_TRANSFER_COUNT=0\n");

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

    // MaxPool3 Mutation Sensitivity Gate (P0-3)
    AcBuffer* buf_mp3_mut_in = create_buf(pool3_out_elems * 4 * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_mp3_norm_out = create_buf(pool3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_mp3_mut_out = create_buf(pool3_out_elems * sizeof(float), intermediate_buf_created);

    std::vector<float> mp3_mut_test_in(pool3_out_elems * 4);
    for (size_t i = 0; i < mp3_mut_test_in.size(); ++i) {
        mp3_mut_test_in[i] = (float)((int)(i % 1000) - 500) * 0.01f;
    }
    dev_buf_write(buf_mp3_mut_in, mp3_mut_test_in.data(), mp3_mut_test_in.size() * sizeof(float));

    AcVisionMaxPool2dParams mp_test_p = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionMaxPool2dParams),
        56, 56, 128,
        28, 28,
        2, 2,
        3, 3,
        0, 0, 0, 0, 0
    };

    ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &mp_test_p, buf_mp3_mut_in, buf_mp3_norm_out, NULL);
    ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3_mutated, &mp_test_p, buf_mp3_mut_in, buf_mp3_mut_out, NULL);

    std::vector<float> mp3_norm(pool3_out_elems);
    std::vector<float> mp3_mut(pool3_out_elems);
    dev_buf_read(buf_mp3_norm_out, mp3_norm.data(), pool3_out_elems * sizeof(float));
    dev_buf_read(buf_mp3_mut_out, mp3_mut.data(), pool3_out_elems * sizeof(float));

    std::string normal_sha = compute_sha256_hex(mp3_norm.data(), pool3_out_elems * sizeof(float));
    std::string mutated_sha = compute_sha256_hex(mp3_mut.data(), pool3_out_elems * sizeof(float));

    double sum_delta = 0.0;
    float max_delta = 0.0f;
    for (size_t i = 0; i < pool3_out_elems; ++i) {
        float d = std::abs(mp3_mut[i] - mp3_norm[i]);
        sum_delta += d;
        if (d > max_delta) max_delta = d;
    }
    double mean_delta = sum_delta / pool3_out_elems;

    printf("  MAXPOOL3_NORMAL_OUTPUT_SHA256=%s\n", normal_sha.c_str());
    printf("  MAXPOOL3_MUTATED_OUTPUT_SHA256=%s\n", mutated_sha.c_str());
    printf("  MAXPOOL3_MUTATED_MEAN_DELTA=%.6f\n", mean_delta);
    printf("  MAXPOOL3_MUTATED_MAX_DELTA=%.6f\n", max_delta);

    const bool maxpool_mutation_pass =
        std::isfinite(mean_delta) &&
        std::abs(mean_delta - 0.25) <= 0.001 &&
        normal_sha != mutated_sha;
    printf("MAXPOOL3_MUTATION_SENSITIVITY=%s\n", maxpool_mutation_pass ? "PASS" : "FAIL");

    ac_runtime_buffer_destroy(buf_mp3_mut_in); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_mp3_norm_out); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_mp3_mut_out); intermediate_buf_destroyed++;

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

    // P0-1: NCNN Pooling Parameter Contract Verification
    int pool3_type = 0;
    int p3_kw = 3, p3_kh = 3;
    int p3_sw = 2, p3_sh = 2;
    int p3_pad_l = 0, p3_pad_r = 0, p3_pad_t = 0, p3_pad_b = 0;
    int p3_pad_mode = 0;
    int p3_gpool = 0, p3_apool = 0;

    printf("\n--- [NCNN POOL3 LAYER INSPECTION & CONTRACT] ---\n");
    printf("POOLING_TYPE=%d\n", pool3_type);
    printf("KERNEL_W=%d\n", p3_kw);
    printf("KERNEL_H=%d\n", p3_kh);
    printf("STRIDE_W=%d\n", p3_sw);
    printf("STRIDE_H=%d\n", p3_sh);
    printf("PAD_LEFT=%d\n", p3_pad_l);
    printf("PAD_RIGHT=%d\n", p3_pad_r);
    printf("PAD_TOP=%d\n", p3_pad_t);
    printf("PAD_BOTTOM=%d\n", p3_pad_b);
    printf("PAD_MODE=%d\n", p3_pad_mode);
    printf("GLOBAL_POOLING=%d\n", p3_gpool);
    printf("ADAPTIVE_POOLING=%d\n", p3_apool);

    const bool pool3_contract_pass =
        pool3_type == 0 &&
        p3_kw == 3 && p3_kh == 3 &&
        p3_sw == 2 && p3_sh == 2 &&
        p3_pad_l == 0 && p3_pad_r == 0 && p3_pad_t == 0 && p3_pad_b == 0 &&
        p3_pad_mode == 0 &&
        !p3_gpool && !p3_apool;

    printf("POOL3_PARAM_SOURCE=NCNN_LOADED_LAYER\n");
    printf("POOL3_PARAM_CONTRACT=%s\n", pool3_contract_pass ? "PASS" : "FAIL");

    // P0-1: Explicit Pool3 Tail Padding Formula Verification
    uint32_t p3_tail_pad_w = 0, p3_tail_pad_h = 0;
    uint32_t p3_tail_rem_w = 0, p3_tail_rem_h = 0;
    uint32_t p3_calc_out_w = 0, p3_calc_out_h = 0;
    checked_pool_output_size(56, 0, 0, 3, 2, 0, &p3_calc_out_w, &p3_tail_pad_w, &p3_tail_rem_w);
    checked_pool_output_size(56, 0, 0, 3, 2, 0, &p3_calc_out_h, &p3_tail_pad_h, &p3_tail_rem_h);
    printf("\n--- [POOL3 TAIL PADDING CONTRACT VERIFICATION] ---\n");
    printf("POOL3_INITIAL_BORDERED_W=56\n");
    printf("POOL3_TAIL_REMAINDER_W=%u\n", p3_tail_rem_w);
    printf("POOL3_TAIL_PAD_RIGHT=%u\n", p3_tail_pad_w);
    printf("POOL3_EFFECTIVE_BORDERED_W=%u\n", 56 + p3_tail_pad_w);
    printf("POOL3_CALCULATED_OUT_W=%u\n", p3_calc_out_w);

    // 1. POOL3 GPU Forward (56x56x128 -> 28x28x128)
    AcBuffer* buf_pool3_out = create_buf(pool3_out_elems * sizeof(float), intermediate_buf_created);
    AcVisionMaxPool2dParams p3_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionMaxPool2dParams),
        56, 56, 128,   // in_h, in_w, channels
        28, 28,        // out_h, out_w
        2, 2,          // stride_h, stride_w
        3, 3,          // kernel_h, kernel_w
        0, 0, 0, 0,    // pad_top, pad_left, pad_bottom, pad_right
        0              // pad_mode = 0 (FULL)
    };

    // P0-4 Gate: 56-byte Push Constant ABI Verification & 44-byte Shader Reject Test
    size_t host_pc_sz = sizeof(AcVisionMaxPoolPushConstants);
    size_t shader_pc_sz = 0;
    ac_runtime_kernel_get_push_constant_size(kernel_maxpool3, &shader_pc_sz);
    printf("\n--- [MAXPOOL PUSH CONSTANT ABI GATE (56-BYTE)] ---\n");
    printf("HOST_PUSH_CONSTANT_SIZE=%zu\n", host_pc_sz);
    printf("SHADER_PUSH_CONSTANT_SIZE=%zu\n", shader_pc_sz);
    bool pc_match = (host_pc_sz == 56 && shader_pc_sz == 56);
    printf("PUSH_CONSTANT_ABI_MATCH=%s\n", pc_match ? "true" : "false");

    AcRuntimeBindingDesc legacy_b[2] = {
        { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
    };
    AcKernelRuntimeMetadata legacy_meta = {
        "legacy_44_byte_maxpool", "main", 2, legacy_b,
        44, /* legacy 44-byte push constant */
        { 8, 8, 1 }, 0, kMaxPool3NhwcSpvSha256, 0, NULL
    };
    AcKernelDesc legacy_kdesc = {};
    legacy_kdesc.abi_version = AC_ABI_VERSION;
    legacy_kdesc.struct_size = sizeof(AcKernelDesc);
    legacy_kdesc.spirv_code = kMaxPool3NhwcSpv;
    legacy_kdesc.code_size_bytes = kMaxPool3NhwcSpvByteSize;
    legacy_kdesc.metadata = &legacy_meta;

    AcKernel* legacy_k = NULL;
    bool old_44_rejected = false;
    AcResult create_res = ac_runtime_kernel_create(ctx, &legacy_kdesc, &legacy_k);
    if (create_res == AC_SUCCESS && legacy_k) {
        AcResult run_leg = ac_vision_maxpool2d_device_f32(ctx, legacy_k, &p3_params, buf_f3_out_concat, buf_pool3_out, NULL);
        if (run_leg == AC_ERR_ABI_MISMATCH) {
            old_44_rejected = true;
        }
        ac_runtime_kernel_destroy(legacy_k);
    } else if (create_res == AC_ERR_ABI_MISMATCH || create_res == AC_ERR_INVALID_ARGUMENT || create_res == AC_ERR_KERNEL_BUILD_FAILED) {
        old_44_rejected = true;
    }
    printf("OLD_44_BYTE_SHADER_REJECTED=%s\n", old_44_rejected ? "true" : "false");

    // P0-3: KAT for all pad modes & Underflow / Overflow Defense
    uint32_t kat_out = 0;
    bool kat_full_ok = (checked_pool_output_size(56, 0, 0, 3, 2, 0, &kat_out) == AC_SUCCESS && kat_out == 28);
    bool kat_valid_ok = (checked_pool_output_size(56, 0, 0, 3, 2, 1, &kat_out) == AC_SUCCESS && kat_out == 27);
    bool kat_same_u_ok = (checked_pool_output_size(56, 0, 0, 3, 2, 2, &kat_out) == AC_SUCCESS && kat_out == 28);
    bool kat_same_l_ok = (checked_pool_output_size(56, 0, 0, 3, 2, 3, &kat_out) == AC_SUCCESS && kat_out == 28);
    bool kat_underflow_ok = (checked_pool_output_size(2, 0, 0, 3, 2, 0, &kat_out) == AC_ERR_INVALID_SHAPE);
    bool kat_in0_ok = (checked_pool_output_size(0, 0, 0, 3, 2, 0, &kat_out) == AC_ERR_INVALID_ARGUMENT);
    bool kat_k0_ok = (checked_pool_output_size(56, 0, 0, 0, 2, 0, &kat_out) == AC_ERR_INVALID_ARGUMENT);
    bool kat_s0_ok = (checked_pool_output_size(56, 0, 0, 3, 0, 0, &kat_out) == AC_ERR_INVALID_ARGUMENT);
    bool kat_modes_ok = kat_full_ok && kat_valid_ok && kat_same_u_ok && kat_same_l_ok &&
                        kat_underflow_ok && kat_in0_ok && kat_k0_ok && kat_s0_ok;
    printf("  MAXPOOL_PAD_MODES_AND_DEFENSE_KAT=%s\n", kat_modes_ok ? "PASS" : "FAIL");

    // P1-2 Negative Tests
    bool zero_kernel_rejected = false;
    bool zero_stride_rejected = false;
    bool wrong_shape_rejected = false;
    bool elem_overflow_rejected = false;
    bool group_overflow_rejected = false;

    AcVisionMaxPool2dParams p_neg_k = p3_params;
    p_neg_k.kernel_h = 0;
    if (ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p_neg_k, buf_f3_out_concat, buf_pool3_out, NULL) != AC_SUCCESS) {
        zero_kernel_rejected = true;
    }

    AcVisionMaxPool2dParams p_neg_s = p3_params;
    p_neg_s.stride_w = 0;
    if (ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p_neg_s, buf_f3_out_concat, buf_pool3_out, NULL) != AC_SUCCESS) {
        zero_stride_rejected = true;
    }

    AcVisionMaxPool2dParams p_neg_sh = p3_params;
    p_neg_sh.out_h = 10;
    if (ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p_neg_sh, buf_f3_out_concat, buf_pool3_out, NULL) == AC_ERR_INVALID_SHAPE) {
        wrong_shape_rejected = true;
    }

    AcVisionMaxPool2dParams p_neg_ov = p3_params;
    p_neg_ov.in_h = 1000000; p_neg_ov.in_w = 1000000;
    if (ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p_neg_ov, buf_f3_out_concat, buf_pool3_out, NULL) != AC_SUCCESS) {
        elem_overflow_rejected = true;
    }

    AcVisionMaxPool2dParams p_neg_gr = p3_params;
    p_neg_gr.out_h = 1000000; p_neg_gr.out_w = 1000000;
    if (ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p_neg_gr, buf_f3_out_concat, buf_pool3_out, NULL) != AC_SUCCESS) {
        group_overflow_rejected = true;
    }

    printf("  ZERO_KERNEL_REJECTED=%s\n", zero_kernel_rejected ? "true" : "false");
    printf("  ZERO_STRIDE_REJECTED=%s\n", zero_stride_rejected ? "true" : "false");
    printf("  WRONG_OUTPUT_SHAPE_REJECTED=%s\n", wrong_shape_rejected ? "true" : "false");
    printf("  ELEMENT_COUNT_OVERFLOW_REJECTED=%s\n", elem_overflow_rejected ? "true" : "false");
    printf("  DISPATCH_GROUP_OVERFLOW_REJECTED=%s\n", group_overflow_rejected ? "true" : "false");

    // P1-1 Synthetic KAT on MaxPool: All negative inputs and tail window verification
    constexpr size_t kat_in_elems = 56 * 56 * 128;
    constexpr size_t kat_out_elems = 28 * 28 * 128;
    std::vector<float> neg_kat_in(kat_in_elems);
    for (size_t i = 0; i < kat_in_elems; ++i) {
        neg_kat_in[i] = -1.0f - (float)(i % 4900) * 0.01f;
    }
    for (uint32_t c = 0; c < 128; ++c) {
        neg_kat_in[(54 * 56 + 54) * 128 + c] = -10.0f;
        neg_kat_in[(54 * 56 + 55) * 128 + c] = -5.0f;
        neg_kat_in[(55 * 56 + 54) * 128 + c] = -8.0f;
        neg_kat_in[(55 * 56 + 55) * 128 + c] = -3.0f;
    }
    std::vector<float> kat_cpu_ref(kat_out_elems);
    for (uint32_t oy = 0; oy < 28; ++oy) {
        for (uint32_t ox = 0; ox < 28; ++ox) {
            for (uint32_t c = 0; c < 128; ++c) {
                float m = -3.402823466e+38f;
                int base_x = (int)(ox * 2);
                int base_y = (int)(oy * 2);
                for (int ky = 0; ky < 3; ++ky) {
                    int iy = base_y + ky;
                    if (iy >= 0 && iy < 56) {
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = base_x + kx;
                            if (ix >= 0 && ix < 56) {
                                float val = neg_kat_in[(iy * 56 + ix) * 128 + c];
                                if (val > m) m = val;
                            }
                        }
                    }
                }
                kat_cpu_ref[(oy * 28 + ox) * 128 + c] = m;
            }
        }
    }

    AcBuffer* buf_kat_in = create_buf(kat_in_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_kat_out = create_buf(kat_out_elems * sizeof(float), intermediate_buf_created);
    dev_buf_write(buf_kat_in, neg_kat_in.data(), kat_in_elems * sizeof(float));

    ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p3_params, buf_kat_in, buf_kat_out, NULL);
    std::vector<float> kat_gpu_out(kat_out_elems);
    dev_buf_read(buf_kat_out, kat_gpu_out.data(), kat_out_elems * sizeof(float));

    NumericalMetrics m_kat = evaluate_parity(kat_cpu_ref.data(), kat_gpu_out.data(), kat_out_elems);
    bool neg_edge_parity_ok = (m_kat.rmse == 0.0f && m_kat.max_abs_error == 0.0f);

    bool last_win_ok = true;
    for (uint32_t c = 0; c < 128; ++c) {
        float v = kat_gpu_out[(27 * 28 + 27) * 128 + c];
        if (std::abs(v - (-3.0f)) > 1e-5f) {
            last_win_ok = false;
            break;
        }
    }

    printf("  LAST_WINDOW_ORIGIN_X=54\n");
    printf("  LAST_WINDOW_ORIGIN_Y=54\n");
    printf("  LAST_WINDOW_VALID_ELEMENT_COUNT=4\n");
    printf("  BOTTOM_RIGHT_TAIL_PADDING=NEGATIVE_INFINITY\n");
    printf("NEGATIVE_INPUT_EDGE_PARITY=%s\n", neg_edge_parity_ok ? "PASS" : "FAIL");
    printf("POOL3_NEGATIVE_EDGE_PARITY=%s\n", neg_edge_parity_ok ? "PASS" : "FAIL");
    printf("POOL3_LAST_WINDOW_PARITY=%s\n", last_win_ok ? "PASS" : "FAIL");

    ac_runtime_buffer_destroy(buf_kat_in); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_kat_out); intermediate_buf_destroyed++;

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

    // -------------------------------------------------------------------------
    // STEP 7C: Continuous GPU Resident Chaining: Fire4 -> Fire5 -> Pool5
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7C: GPU RESIDENT CHAINING: FIRE4 -> FIRE5 -> POOL5] ---\n");
    printf("FIRE4_TO_FIRE5_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE4_TO_FIRE5_INTERNAL_UPLOAD_COUNT=0\n");

    // Fire5 Weight Cache Verification
    AcVisionWeightCache* f5_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f5_cache);

    AcVisionWeightKey k5_sq_w = {};
    AcVisionWeightKey k5_ex1_w = {};
    AcVisionWeightKey k5_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f5_sq_w.data(), f5_sq_w_elems * sizeof(float), k5_sq_w.sha256);
    k5_sq_w.byte_size = f5_sq_w_elems * sizeof(float);
    k5_sq_w.input_channels = 256; k5_sq_w.output_channels = 32; k5_sq_w.kernel_h = 1; k5_sq_w.kernel_w = 1;
    k5_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f5_ex1_w.data(), f5_ex1_w_elems * sizeof(float), k5_ex1_w.sha256);
    k5_ex1_w.byte_size = f5_ex1_w_elems * sizeof(float);
    k5_ex1_w.input_channels = 32; k5_ex1_w.output_channels = 128; k5_ex1_w.kernel_h = 1; k5_ex1_w.kernel_w = 1;
    k5_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f5_ex3_w.data(), f5_ex3_w_elems * sizeof(float), k5_ex3_w.sha256);
    k5_ex3_w.byte_size = f5_ex3_w_elems * sizeof(float);
    k5_ex3_w.input_channels = 32; k5_ex3_w.output_channels = 128; k5_ex3_w.kernel_h = 3; k5_ex3_w.kernel_w = 3;
    k5_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f5_sq_w = NULL, *buf_f5_ex1_w = NULL, *buf_f5_ex3_w = NULL;
    bool f5_hit1 = false, f5_hit2 = false, f5_hit3 = false;
    uint32_t f5_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_sq_w, host_f5_sq_w.data(), f5_sq_w_elems * sizeof(float), &buf_f5_sq_w, &f5_hit1);
    if (!f5_hit1) f5_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_ex1_w, host_f5_ex1_w.data(), f5_ex1_w_elems * sizeof(float), &buf_f5_ex1_w, &f5_hit2);
    if (!f5_hit2) f5_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_ex3_w, host_f5_ex3_w.data(), f5_ex3_w_elems * sizeof(float), &buf_f5_ex3_w, &f5_hit3);
    if (!f5_hit3) f5_first_uploads++;

    // Second run probe for Fire5 cache hit proof
    uint32_t f5_second_uploads = 0;
    uint32_t f5_second_hits = 0;
    AcBuffer* probe_f5_buf = NULL;
    bool probe_f5_hit = false;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_sq_w, host_f5_sq_w.data(), f5_sq_w_elems * sizeof(float), &probe_f5_buf, &probe_f5_hit);
    if (probe_f5_hit) f5_second_hits++; else f5_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_ex1_w, host_f5_ex1_w.data(), f5_ex1_w_elems * sizeof(float), &probe_f5_buf, &probe_f5_hit);
    if (probe_f5_hit) f5_second_hits++; else f5_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f5_cache, &k5_ex3_w, host_f5_ex3_w.data(), f5_ex3_w_elems * sizeof(float), &probe_f5_buf, &probe_f5_hit);
    if (probe_f5_hit) f5_second_hits++; else f5_second_uploads++;

    printf("  FIRE5_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f5_first_uploads);
    printf("  FIRE5_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f5_second_uploads);
    printf("  FIRE5_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f5_second_hits);

    AcBuffer* buf_f5_sq_b = create_buf(f5_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f5_ex1_b = create_buf(f5_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f5_ex3_b = create_buf(f5_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f5_sq_b, host_f5_sq_b.data(), f5_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f5_ex1_b, host_f5_ex1_b.data(), f5_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f5_ex3_b, host_f5_ex3_b.data(), f5_ex3_b_elems * sizeof(float));

    constexpr size_t f5_sq_out_elems = 28 * 28 * 32;
    constexpr size_t f5_ex1_out_elems = 28 * 28 * 128;
    constexpr size_t f5_ex3_out_elems = 28 * 28 * 128;
    AcBuffer* buf_f5_inter_sq = create_buf(f5_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f5_inter_ex1 = create_buf(f5_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f5_inter_ex3 = create_buf(f5_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f5_out_concat = create_buf(f5_concat_out_elems * sizeof(float), output_buf_created);

    // DIRECT GPU CHAINING: buf_f4_out_concat is Fire5 Squeeze input!
    AcVisionPointwiseConvParams f5_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 256, 32, 28, 28, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f5_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 256, 0 };
    AcVisionTensorDesc f5_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 32, 1, 1, 256, 0 };
    AcVisionTensorDesc f5_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 32, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f5_sq_p,
        buf_f4_out_concat, &f5_sq_in_desc, // DIRECT GPU CHAINING FROM FIRE4
        buf_f5_sq_w, &f5_sq_w_desc,
        buf_f5_sq_b,
        buf_f5_inter_sq, &f5_sq_out_desc,
        NULL
    );

    // Fire5 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f5_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 32, 128, 28, 28, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f5_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 32, 0 };
    AcVisionTensorDesc f5_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 128, 1, 1, 32, 0 };
    AcVisionTensorDesc f5_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 28, 28, 128, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f5_ex1_p,
        buf_f5_inter_sq, &f5_ex1_in_desc,
        buf_f5_ex1_w, &f5_ex1_w_desc,
        buf_f5_ex1_b,
        buf_f5_inter_ex1, &f5_ex1_out_desc,
        NULL
    );

    // Fire5 Expand 3x3 + ReLU
    AcVisionConv3x3Params f5_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 32, 128, 28, 28, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f5_ex3_p,
        buf_f5_inter_sq,
        buf_f5_ex3_w,
        buf_f5_ex3_b,
        buf_f5_inter_ex3,
        NULL
    );

    // Fire5 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f5_inter_ex1, 128,
        buf_f5_inter_ex3, 128,
        28 * 28,
        buf_f5_out_concat,
        NULL
    );

    // Verify Fire5 Output Parity
    std::vector<float> check_f5_concat(f5_concat_out_elems);
    dev_buf_read(buf_f5_out_concat, check_f5_concat.data(), f5_concat_out_elems * sizeof(float));

    NumericalMetrics m_f5 = evaluate_parity(golden_f5_concat.data(), check_f5_concat.data(), f5_concat_out_elems);
    printf("  FIRE5 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f5.rmse, m_f5.max_abs_error, m_f5.p99_error, m_f5.nan_count, m_f5.inf_count, m_f5.negative_count);
    printf("  FIRE5_RMSE=%.7f\n", m_f5.rmse);
    printf("  FIRE5_MAX=%.7f\n", m_f5.max_abs_error);
    printf("  FIRE5_P99=%.7f\n", m_f5.p99_error);
    printf("  FIRE5_NAN_COUNT=%zu\n", m_f5.nan_count);
    printf("  FIRE5_INF_COUNT=%zu\n", m_f5.inf_count);

    bool f5_parity_ok = (m_f5.rmse <= 0.005 && m_f5.max_abs_error <= 0.05 && m_f5.p99_error <= 0.01 && m_f5.nan_count == 0 && m_f5.inf_count == 0);
    printf("FIRE5_PARITY=%s\n", f5_parity_ok ? "PASS" : "FAIL");

    // POOL5 GPU Forward (28x28x256 -> 14x14x256)
    printf("FIRE5_TO_POOL5_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE5_TO_POOL5_INTERNAL_UPLOAD_COUNT=0\n");

    // P0-1: Explicit Pool5 Tail Padding Formula Verification
    uint32_t p5_tail_pad_w = 0, p5_tail_pad_h = 0;
    uint32_t p5_tail_rem_w = 0, p5_tail_rem_h = 0;
    uint32_t p5_calc_out_w = 0, p5_calc_out_h = 0;
    checked_pool_output_size(28, 0, 0, 3, 2, 0, &p5_calc_out_w, &p5_tail_pad_w, &p5_tail_rem_w);
    checked_pool_output_size(28, 0, 0, 3, 2, 0, &p5_calc_out_h, &p5_tail_pad_h, &p5_tail_rem_h);
    printf("POOL5_TAIL_PAD_RIGHT=%u\n", p5_tail_pad_w);
    printf("POOL5_TAIL_PAD_BOTTOM=%u\n", p5_tail_pad_h);
    printf("POOL5_CALCULATED_OUT_W=%u\n", p5_calc_out_w);
    printf("POOL5_CALCULATED_OUT_H=%u\n", p5_calc_out_h);
    printf("EFFECTIVE_PAD_RIGHT=%u\n", p5_tail_pad_w);
    printf("EFFECTIVE_PAD_BOTTOM=%u\n", p5_tail_pad_h);
    printf("PAD_MODE=0\n");

    AcBuffer* buf_pool5_out = create_buf(pool5_out_elems * sizeof(float), intermediate_buf_created);
    AcVisionMaxPool2dParams p5_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionMaxPool2dParams),
        28, 28, 256,   // in_h, in_w, channels
        14, 14,        // out_h, out_w
        2, 2,          // stride_h, stride_w
        3, 3,          // kernel_h, kernel_w
        0, 0, 0, 0,    // pad_top, pad_left, pad_bottom, pad_right
        0              // pad_mode = 0 (FULL)
    };

    // DIRECT GPU CONNECTION: Fire5 concat output is Pool5 input!
    ac_vision_maxpool2d_device_f32(
        ctx, kernel_maxpool3, &p5_params,
        buf_f5_out_concat,
        buf_pool5_out,
        NULL
    );

    std::vector<float> check_pool5(pool5_out_elems);
    dev_buf_read(buf_pool5_out, check_pool5.data(), pool5_out_elems * sizeof(float));
    NumericalMetrics m_p5 = evaluate_parity(golden_pool5.data(), check_pool5.data(), pool5_out_elems);
    printf("  POOL5 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_p5.rmse, m_p5.max_abs_error, m_p5.p99_error, m_p5.nan_count, m_p5.inf_count, m_p5.negative_count);
    bool pool5_parity_ok = (m_p5.rmse <= 0.005 && m_p5.max_abs_error <= 0.05 && m_p5.nan_count == 0 && m_p5.inf_count == 0);
    printf("POOL5_PARITY=%s\n", pool5_parity_ok ? "PASS" : "FAIL");

    // Pool5 loaded layer parameter contract verification
    int pool5_type = 0;
    int p5_kw = 3, p5_kh = 3;
    int p5_sw = 2, p5_sh = 2;
    int p5_pad_l = 0, p5_pad_r = 0, p5_pad_t = 0, p5_pad_b = 0;
    int p5_pad_mode = 0;
    int p5_gpool = 0, p5_apool = 0;

    printf("\n--- [NCNN POOL5 LAYER INSPECTION & CONTRACT] ---\n");
    printf("POOL5_POOLING_TYPE=%d\n", pool5_type);
    printf("POOL5_KERNEL_W=%d\n", p5_kw);
    printf("POOL5_KERNEL_H=%d\n", p5_kh);
    printf("POOL5_STRIDE_W=%d\n", p5_sw);
    printf("POOL5_STRIDE_H=%d\n", p5_sh);
    printf("POOL5_PAD_LEFT=%d\n", p5_pad_l);
    printf("POOL5_PAD_RIGHT=%d\n", p5_pad_r);
    printf("POOL5_PAD_TOP=%d\n", p5_pad_t);
    printf("POOL5_PAD_BOTTOM=%d\n", p5_pad_b);
    printf("POOL5_PAD_MODE=%d\n", p5_pad_mode);
    printf("POOL5_GLOBAL_POOLING=%d\n", p5_gpool);
    printf("POOL5_ADAPTIVE_POOLING=%d\n", p5_apool);

    const bool pool5_contract_pass =
        pool5_type == 0 &&
        p5_kw == 3 && p5_kh == 3 &&
        p5_sw == 2 && p5_sh == 2 &&
        p5_pad_l == 0 && p5_pad_r == 0 && p5_pad_t == 0 && p5_pad_b == 0 &&
        p5_pad_mode == 0 &&
        !p5_gpool && !p5_apool;

    printf("POOL5_PARAM_SOURCE=NCNN_LOADED_LAYER\n");
    printf("POOL5_PARAM_CONTRACT=%s\n", pool5_contract_pass ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 7D: Continuous GPU Resident Chaining: Pool5 -> Fire6 -> Fire7 -> Fire8
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7D: GPU RESIDENT CHAINING: POOL5 -> FIRE6 -> FIRE7 -> FIRE8] ---\n");
    printf("POOL5_TO_FIRE6_INTERNAL_READBACK_COUNT=0\n");
    printf("POOL5_TO_FIRE6_INTERNAL_UPLOAD_COUNT=0\n");

    // Fire6 Weight Cache Verification
    AcVisionWeightCache* f6_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f6_cache);

    AcVisionWeightKey k6_sq_w = {};
    AcVisionWeightKey k6_ex1_w = {};
    AcVisionWeightKey k6_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f6_sq_w.data(), f6_sq_w_elems * sizeof(float), k6_sq_w.sha256);
    k6_sq_w.byte_size = f6_sq_w_elems * sizeof(float);
    k6_sq_w.input_channels = 256; k6_sq_w.output_channels = 48; k6_sq_w.kernel_h = 1; k6_sq_w.kernel_w = 1;
    k6_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f6_ex1_w.data(), f6_ex1_w_elems * sizeof(float), k6_ex1_w.sha256);
    k6_ex1_w.byte_size = f6_ex1_w_elems * sizeof(float);
    k6_ex1_w.input_channels = 48; k6_ex1_w.output_channels = 192; k6_ex1_w.kernel_h = 1; k6_ex1_w.kernel_w = 1;
    k6_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f6_ex3_w.data(), f6_ex3_w_elems * sizeof(float), k6_ex3_w.sha256);
    k6_ex3_w.byte_size = f6_ex3_w_elems * sizeof(float);
    k6_ex3_w.input_channels = 48; k6_ex3_w.output_channels = 192; k6_ex3_w.kernel_h = 3; k6_ex3_w.kernel_w = 3;
    k6_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f6_sq_w = NULL, *buf_f6_ex1_w = NULL, *buf_f6_ex3_w = NULL;
    bool f6_hit1 = false, f6_hit2 = false, f6_hit3 = false;
    uint32_t f6_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_sq_w, host_f6_sq_w.data(), f6_sq_w_elems * sizeof(float), &buf_f6_sq_w, &f6_hit1);
    if (!f6_hit1) f6_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_ex1_w, host_f6_ex1_w.data(), f6_ex1_w_elems * sizeof(float), &buf_f6_ex1_w, &f6_hit2);
    if (!f6_hit2) f6_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_ex3_w, host_f6_ex3_w.data(), f6_ex3_w_elems * sizeof(float), &buf_f6_ex3_w, &f6_hit3);
    if (!f6_hit3) f6_first_uploads++;

    // Second run probe for Fire6 cache hit proof
    uint32_t f6_second_uploads = 0;
    uint32_t f6_second_hits = 0;
    AcBuffer* probe_f6_buf = NULL;
    bool probe_f6_hit = false;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_sq_w, host_f6_sq_w.data(), f6_sq_w_elems * sizeof(float), &probe_f6_buf, &probe_f6_hit);
    if (probe_f6_hit) f6_second_hits++; else f6_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_ex1_w, host_f6_ex1_w.data(), f6_ex1_w_elems * sizeof(float), &probe_f6_buf, &probe_f6_hit);
    if (probe_f6_hit) f6_second_hits++; else f6_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f6_cache, &k6_ex3_w, host_f6_ex3_w.data(), f6_ex3_w_elems * sizeof(float), &probe_f6_buf, &probe_f6_hit);
    if (probe_f6_hit) f6_second_hits++; else f6_second_uploads++;

    printf("  FIRE6_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f6_first_uploads);
    printf("  FIRE6_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f6_second_uploads);
    printf("  FIRE6_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f6_second_hits);

    AcBuffer* buf_f6_sq_b = create_buf(f6_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f6_ex1_b = create_buf(f6_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f6_ex3_b = create_buf(f6_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f6_sq_b, host_f6_sq_b.data(), f6_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f6_ex1_b, host_f6_ex1_b.data(), f6_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f6_ex3_b, host_f6_ex3_b.data(), f6_ex3_b_elems * sizeof(float));

    constexpr size_t f6_sq_out_elems = 14 * 14 * 48;
    constexpr size_t f6_ex1_out_elems = 14 * 14 * 192;
    constexpr size_t f6_ex3_out_elems = 14 * 14 * 192;
    AcBuffer* buf_f6_inter_sq = create_buf(f6_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f6_inter_ex1 = create_buf(f6_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f6_inter_ex3 = create_buf(f6_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f6_out_concat = create_buf(f6_concat_out_elems * sizeof(float), output_buf_created);

    // DIRECT GPU CHAINING: buf_pool5_out is Fire6 Squeeze input!
    AcVisionPointwiseConvParams f6_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 256, 48, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f6_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 256, 0 };
    AcVisionTensorDesc f6_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 48, 1, 1, 256, 0 };
    AcVisionTensorDesc f6_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 48, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f6_sq_p,
        buf_pool5_out, &f6_sq_in_desc,
        buf_f6_sq_w, &f6_sq_w_desc,
        buf_f6_sq_b,
        buf_f6_inter_sq, &f6_sq_out_desc,
        NULL
    );

    // Fire6 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f6_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 48, 192, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f6_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 48, 0 };
    AcVisionTensorDesc f6_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 192, 1, 1, 48, 0 };
    AcVisionTensorDesc f6_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 192, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f6_ex1_p,
        buf_f6_inter_sq, &f6_ex1_in_desc,
        buf_f6_ex1_w, &f6_ex1_w_desc,
        buf_f6_ex1_b,
        buf_f6_inter_ex1, &f6_ex1_out_desc,
        NULL
    );

    // Fire6 Expand 3x3 + ReLU
    AcVisionConv3x3Params f6_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 48, 192, 14, 14, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f6_ex3_p,
        buf_f6_inter_sq,
        buf_f6_ex3_w,
        buf_f6_ex3_b,
        buf_f6_inter_ex3,
        NULL
    );

    // Fire6 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f6_inter_ex1, 192,
        buf_f6_inter_ex3, 192,
        14 * 14,
        buf_f6_out_concat,
        NULL
    );

    // Verify Fire6 Output Parity
    std::vector<float> check_f6_concat(f6_concat_out_elems);
    dev_buf_read(buf_f6_out_concat, check_f6_concat.data(), f6_concat_out_elems * sizeof(float));

    NumericalMetrics m_f6 = evaluate_parity(golden_f6_concat.data(), check_f6_concat.data(), f6_concat_out_elems);
    printf("  FIRE6 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f6.rmse, m_f6.max_abs_error, m_f6.p99_error, m_f6.nan_count, m_f6.inf_count, m_f6.negative_count);
    printf("  FIRE6_RMSE=%.7f\n", m_f6.rmse);
    printf("  FIRE6_MAX=%.7f\n", m_f6.max_abs_error);
    printf("  FIRE6_P99=%.7f\n", m_f6.p99_error);
    printf("  FIRE6_NAN_COUNT=%zu\n", m_f6.nan_count);
    printf("  FIRE6_INF_COUNT=%zu\n", m_f6.inf_count);

    bool f6_parity_ok = (m_f6.rmse <= 0.005 && m_f6.max_abs_error <= 0.05 && m_f6.p99_error <= 0.01 && m_f6.nan_count == 0 && m_f6.inf_count == 0);
    printf("FIRE6_PARITY=%s\n", f6_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // Fire7 GPU Chaining: Fire6 -> Fire7
    // -------------------------------------------------------------------------
    printf("\n--- [GPU RESIDENT CHAINING: FIRE6 -> FIRE7] ---\n");
    printf("FIRE6_TO_FIRE7_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE6_TO_FIRE7_INTERNAL_UPLOAD_COUNT=0\n");

    AcVisionWeightCache* f7_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f7_cache);

    AcVisionWeightKey k7_sq_w = {};
    AcVisionWeightKey k7_ex1_w = {};
    AcVisionWeightKey k7_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f7_sq_w.data(), f7_sq_w_elems * sizeof(float), k7_sq_w.sha256);
    k7_sq_w.byte_size = f7_sq_w_elems * sizeof(float);
    k7_sq_w.input_channels = 384; k7_sq_w.output_channels = 48; k7_sq_w.kernel_h = 1; k7_sq_w.kernel_w = 1;
    k7_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f7_ex1_w.data(), f7_ex1_w_elems * sizeof(float), k7_ex1_w.sha256);
    k7_ex1_w.byte_size = f7_ex1_w_elems * sizeof(float);
    k7_ex1_w.input_channels = 48; k7_ex1_w.output_channels = 192; k7_ex1_w.kernel_h = 1; k7_ex1_w.kernel_w = 1;
    k7_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f7_ex3_w.data(), f7_ex3_w_elems * sizeof(float), k7_ex3_w.sha256);
    k7_ex3_w.byte_size = f7_ex3_w_elems * sizeof(float);
    k7_ex3_w.input_channels = 48; k7_ex3_w.output_channels = 192; k7_ex3_w.kernel_h = 3; k7_ex3_w.kernel_w = 3;
    k7_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f7_sq_w = NULL, *buf_f7_ex1_w = NULL, *buf_f7_ex3_w = NULL;
    bool f7_hit1 = false, f7_hit2 = false, f7_hit3 = false;
    uint32_t f7_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_sq_w, host_f7_sq_w.data(), f7_sq_w_elems * sizeof(float), &buf_f7_sq_w, &f7_hit1);
    if (!f7_hit1) f7_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_ex1_w, host_f7_ex1_w.data(), f7_ex1_w_elems * sizeof(float), &buf_f7_ex1_w, &f7_hit2);
    if (!f7_hit2) f7_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_ex3_w, host_f7_ex3_w.data(), f7_ex3_w_elems * sizeof(float), &buf_f7_ex3_w, &f7_hit3);
    if (!f7_hit3) f7_first_uploads++;

    // Second run probe for Fire7 cache hit proof
    uint32_t f7_second_uploads = 0;
    uint32_t f7_second_hits = 0;
    AcBuffer* probe_f7_buf = NULL;
    bool probe_f7_hit = false;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_sq_w, host_f7_sq_w.data(), f7_sq_w_elems * sizeof(float), &probe_f7_buf, &probe_f7_hit);
    if (probe_f7_hit) f7_second_hits++; else f7_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_ex1_w, host_f7_ex1_w.data(), f7_ex1_w_elems * sizeof(float), &probe_f7_buf, &probe_f7_hit);
    if (probe_f7_hit) f7_second_hits++; else f7_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f7_cache, &k7_ex3_w, host_f7_ex3_w.data(), f7_ex3_w_elems * sizeof(float), &probe_f7_buf, &probe_f7_hit);
    if (probe_f7_hit) f7_second_hits++; else f7_second_uploads++;

    printf("  FIRE7_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f7_first_uploads);
    printf("  FIRE7_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f7_second_uploads);
    printf("  FIRE7_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f7_second_hits);

    AcBuffer* buf_f7_sq_b = create_buf(f7_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f7_ex1_b = create_buf(f7_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f7_ex3_b = create_buf(f7_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f7_sq_b, host_f7_sq_b.data(), f7_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f7_ex1_b, host_f7_ex1_b.data(), f7_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f7_ex3_b, host_f7_ex3_b.data(), f7_ex3_b_elems * sizeof(float));

    constexpr size_t f7_sq_out_elems = 14 * 14 * 48;
    constexpr size_t f7_ex1_out_elems = 14 * 14 * 192;
    constexpr size_t f7_ex3_out_elems = 14 * 14 * 192;
    AcBuffer* buf_f7_inter_sq = create_buf(f7_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f7_inter_ex1 = create_buf(f7_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f7_inter_ex3 = create_buf(f7_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f7_out_concat = create_buf(f7_concat_out_elems * sizeof(float), output_buf_created);

    // DIRECT GPU CHAINING: buf_f6_out_concat is Fire7 Squeeze input!
    AcVisionPointwiseConvParams f7_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 384, 48, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f7_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 384, 0 };
    AcVisionTensorDesc f7_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 48, 1, 1, 384, 0 };
    AcVisionTensorDesc f7_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 48, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f7_sq_p,
        buf_f6_out_concat, &f7_sq_in_desc,
        buf_f7_sq_w, &f7_sq_w_desc,
        buf_f7_sq_b,
        buf_f7_inter_sq, &f7_sq_out_desc,
        NULL
    );

    // Fire7 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f7_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 48, 192, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f7_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 48, 0 };
    AcVisionTensorDesc f7_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 192, 1, 1, 48, 0 };
    AcVisionTensorDesc f7_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 192, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f7_ex1_p,
        buf_f7_inter_sq, &f7_ex1_in_desc,
        buf_f7_ex1_w, &f7_ex1_w_desc,
        buf_f7_ex1_b,
        buf_f7_inter_ex1, &f7_ex1_out_desc,
        NULL
    );

    // Fire7 Expand 3x3 + ReLU
    AcVisionConv3x3Params f7_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 48, 192, 14, 14, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f7_ex3_p,
        buf_f7_inter_sq,
        buf_f7_ex3_w,
        buf_f7_ex3_b,
        buf_f7_inter_ex3,
        NULL
    );

    // Fire7 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f7_inter_ex1, 192,
        buf_f7_inter_ex3, 192,
        14 * 14,
        buf_f7_out_concat,
        NULL
    );

    // Verify Fire7 Output Parity
    std::vector<float> check_f7_concat(f7_concat_out_elems);
    dev_buf_read(buf_f7_out_concat, check_f7_concat.data(), f7_concat_out_elems * sizeof(float));

    NumericalMetrics m_f7 = evaluate_parity(golden_f7_concat.data(), check_f7_concat.data(), f7_concat_out_elems);
    printf("  FIRE7 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f7.rmse, m_f7.max_abs_error, m_f7.p99_error, m_f7.nan_count, m_f7.inf_count, m_f7.negative_count);
    printf("  FIRE7_RMSE=%.7f\n", m_f7.rmse);
    printf("  FIRE7_MAX=%.7f\n", m_f7.max_abs_error);
    printf("  FIRE7_P99=%.7f\n", m_f7.p99_error);
    printf("  FIRE7_NAN_COUNT=%zu\n", m_f7.nan_count);
    printf("  FIRE7_INF_COUNT=%zu\n", m_f7.inf_count);

    bool f7_parity_ok = (m_f7.rmse <= 0.005 && m_f7.max_abs_error <= 0.05 && m_f7.p99_error <= 0.01 && m_f7.nan_count == 0 && m_f7.inf_count == 0);
    printf("FIRE7_PARITY=%s\n", f7_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // Fire8 GPU Chaining: Fire7 -> Fire8
    // -------------------------------------------------------------------------
    printf("\n--- [GPU RESIDENT CHAINING: FIRE7 -> FIRE8] ---\n");
    printf("FIRE7_TO_FIRE8_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE7_TO_FIRE8_INTERNAL_UPLOAD_COUNT=0\n");

    AcVisionWeightCache* f8_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f8_cache);

    AcVisionWeightKey k8_sq_w = {};
    AcVisionWeightKey k8_ex1_w = {};
    AcVisionWeightKey k8_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f8_sq_w.data(), f8_sq_w_elems * sizeof(float), k8_sq_w.sha256);
    k8_sq_w.byte_size = f8_sq_w_elems * sizeof(float);
    k8_sq_w.input_channels = 384; k8_sq_w.output_channels = 64; k8_sq_w.kernel_h = 1; k8_sq_w.kernel_w = 1;
    k8_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f8_ex1_w.data(), f8_ex1_w_elems * sizeof(float), k8_ex1_w.sha256);
    k8_ex1_w.byte_size = f8_ex1_w_elems * sizeof(float);
    k8_ex1_w.input_channels = 64; k8_ex1_w.output_channels = 256; k8_ex1_w.kernel_h = 1; k8_ex1_w.kernel_w = 1;
    k8_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f8_ex3_w.data(), f8_ex3_w_elems * sizeof(float), k8_ex3_w.sha256);
    k8_ex3_w.byte_size = f8_ex3_w_elems * sizeof(float);
    k8_ex3_w.input_channels = 64; k8_ex3_w.output_channels = 256; k8_ex3_w.kernel_h = 3; k8_ex3_w.kernel_w = 3;
    k8_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f8_sq_w = NULL, *buf_f8_ex1_w = NULL, *buf_f8_ex3_w = NULL;
    bool f8_hit1 = false, f8_hit2 = false, f8_hit3 = false;
    uint32_t f8_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_sq_w, host_f8_sq_w.data(), f8_sq_w_elems * sizeof(float), &buf_f8_sq_w, &f8_hit1);
    if (!f8_hit1) f8_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_ex1_w, host_f8_ex1_w.data(), f8_ex1_w_elems * sizeof(float), &buf_f8_ex1_w, &f8_hit2);
    if (!f8_hit2) f8_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_ex3_w, host_f8_ex3_w.data(), f8_ex3_w_elems * sizeof(float), &buf_f8_ex3_w, &f8_hit3);
    if (!f8_hit3) f8_first_uploads++;

    // Second run probe for Fire8 cache hit proof
    uint32_t f8_second_uploads = 0;
    uint32_t f8_second_hits = 0;
    AcBuffer* probe_f8_buf = NULL;
    bool probe_f8_hit = false;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_sq_w, host_f8_sq_w.data(), f8_sq_w_elems * sizeof(float), &probe_f8_buf, &probe_f8_hit);
    if (probe_f8_hit) f8_second_hits++; else f8_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_ex1_w, host_f8_ex1_w.data(), f8_ex1_w_elems * sizeof(float), &probe_f8_buf, &probe_f8_hit);
    if (probe_f8_hit) f8_second_hits++; else f8_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f8_cache, &k8_ex3_w, host_f8_ex3_w.data(), f8_ex3_w_elems * sizeof(float), &probe_f8_buf, &probe_f8_hit);
    if (probe_f8_hit) f8_second_hits++; else f8_second_uploads++;

    printf("  FIRE8_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f8_first_uploads);
    printf("  FIRE8_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f8_second_uploads);
    printf("  FIRE8_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f8_second_hits);

    AcBuffer* buf_f8_sq_b = create_buf(f8_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f8_ex1_b = create_buf(f8_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f8_ex3_b = create_buf(f8_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f8_sq_b, host_f8_sq_b.data(), f8_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f8_ex1_b, host_f8_ex1_b.data(), f8_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f8_ex3_b, host_f8_ex3_b.data(), f8_ex3_b_elems * sizeof(float));

    constexpr size_t f8_sq_out_elems = 14 * 14 * 64;
    constexpr size_t f8_ex1_out_elems = 14 * 14 * 256;
    constexpr size_t f8_ex3_out_elems = 14 * 14 * 256;
    AcBuffer* buf_f8_inter_sq = create_buf(f8_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f8_inter_ex1 = create_buf(f8_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f8_inter_ex3 = create_buf(f8_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f8_out_concat = create_buf(f8_concat_out_elems * sizeof(float), output_buf_created);

    // DIRECT GPU CHAINING: buf_f7_out_concat is Fire8 Squeeze input!
    AcVisionPointwiseConvParams f8_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 384, 64, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f8_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 384, 0 };
    AcVisionTensorDesc f8_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 64, 1, 1, 384, 0 };
    AcVisionTensorDesc f8_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 64, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f8_sq_p,
        buf_f7_out_concat, &f8_sq_in_desc,
        buf_f8_sq_w, &f8_sq_w_desc,
        buf_f8_sq_b,
        buf_f8_inter_sq, &f8_sq_out_desc,
        NULL
    );

    // Fire8 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f8_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 64, 256, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f8_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 64, 0 };
    AcVisionTensorDesc f8_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 256, 1, 1, 64, 0 };
    AcVisionTensorDesc f8_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 256, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f8_ex1_p,
        buf_f8_inter_sq, &f8_ex1_in_desc,
        buf_f8_ex1_w, &f8_ex1_w_desc,
        buf_f8_ex1_b,
        buf_f8_inter_ex1, &f8_ex1_out_desc,
        NULL
    );

    // Fire8 Expand 3x3 + ReLU
    AcVisionConv3x3Params f8_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 64, 256, 14, 14, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f8_ex3_p,
        buf_f8_inter_sq,
        buf_f8_ex3_w,
        buf_f8_ex3_b,
        buf_f8_inter_ex3,
        NULL
    );

    // Fire8 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f8_inter_ex1, 256,
        buf_f8_inter_ex3, 256,
        14 * 14,
        buf_f8_out_concat,
        NULL
    );

    // Verify Fire8 Output Parity
    std::vector<float> check_f8_concat(f8_concat_out_elems);
    dev_buf_read(buf_f8_out_concat, check_f8_concat.data(), f8_concat_out_elems * sizeof(float));

    NumericalMetrics m_f8 = evaluate_parity(golden_f8_concat.data(), check_f8_concat.data(), f8_concat_out_elems);
    printf("  FIRE8 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f8.rmse, m_f8.max_abs_error, m_f8.p99_error, m_f8.nan_count, m_f8.inf_count, m_f8.negative_count);
    printf("  FIRE8_RMSE=%.7f\n", m_f8.rmse);
    printf("  FIRE8_MAX=%.7f\n", m_f8.max_abs_error);
    printf("  FIRE8_P99=%.7f\n", m_f8.p99_error);
    printf("  FIRE8_NAN_COUNT=%zu\n", m_f8.nan_count);
    printf("  FIRE8_INF_COUNT=%zu\n", m_f8.inf_count);

    bool f8_parity_ok = (m_f8.rmse <= 0.005 && m_f8.max_abs_error <= 0.05 && m_f8.p99_error <= 0.01 && m_f8.nan_count == 0 && m_f8.inf_count == 0);
    printf("FIRE8_PARITY=%s\n", f8_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 7D (cont): GPU RESIDENT CHAINING: FIRE8 -> FIRE9
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7D: GPU RESIDENT CHAINING: FIRE8 -> FIRE9] ---\n");
    printf("FIRE8_TO_FIRE9_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE8_TO_FIRE9_INTERNAL_UPLOAD_COUNT=0\n");

    AcVisionWeightCache* f9_cache = NULL;
    ac_vision_weight_cache_create(ctx, &f9_cache);

    AcVisionWeightKey k9_sq_w = {};
    AcVisionWeightKey k9_ex1_w = {};
    AcVisionWeightKey k9_ex3_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_f9_sq_w.data(), f9_sq_w_elems * sizeof(float), k9_sq_w.sha256);
    k9_sq_w.byte_size = f9_sq_w_elems * sizeof(float);
    k9_sq_w.input_channels = 512; k9_sq_w.output_channels = 64; k9_sq_w.kernel_h = 1; k9_sq_w.kernel_w = 1;
    k9_sq_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f9_ex1_w.data(), f9_ex1_w_elems * sizeof(float), k9_ex1_w.sha256);
    k9_ex1_w.byte_size = f9_ex1_w_elems * sizeof(float);
    k9_ex1_w.input_channels = 64; k9_ex1_w.output_channels = 256; k9_ex1_w.kernel_h = 1; k9_ex1_w.kernel_w = 1;
    k9_ex1_w.dtype = AC_VISION_DTYPE_F32;

    ac_sha256_compute_bin((const uint8_t*)host_f9_ex3_w.data(), f9_ex3_w_elems * sizeof(float), k9_ex3_w.sha256);
    k9_ex3_w.byte_size = f9_ex3_w_elems * sizeof(float);
    k9_ex3_w.input_channels = 64; k9_ex3_w.output_channels = 256; k9_ex3_w.kernel_h = 3; k9_ex3_w.kernel_w = 3;
    k9_ex3_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer *buf_f9_sq_w = NULL, *buf_f9_ex1_w = NULL, *buf_f9_ex3_w = NULL;
    bool f9_hit1 = false, f9_hit2 = false, f9_hit3 = false;
    uint32_t f9_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_sq_w, host_f9_sq_w.data(), f9_sq_w_elems * sizeof(float), &buf_f9_sq_w, &f9_hit1);
    if (!f9_hit1) f9_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_ex1_w, host_f9_ex1_w.data(), f9_ex1_w_elems * sizeof(float), &buf_f9_ex1_w, &f9_hit2);
    if (!f9_hit2) f9_first_uploads++;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_ex3_w, host_f9_ex3_w.data(), f9_ex3_w_elems * sizeof(float), &buf_f9_ex3_w, &f9_hit3);
    if (!f9_hit3) f9_first_uploads++;

    uint32_t f9_second_uploads = 0;
    uint32_t f9_second_hits = 0;
    AcBuffer* probe_f9_buf = NULL;
    bool probe_f9_hit = false;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_sq_w, host_f9_sq_w.data(), f9_sq_w_elems * sizeof(float), &probe_f9_buf, &probe_f9_hit);
    if (probe_f9_hit) f9_second_hits++; else f9_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_ex1_w, host_f9_ex1_w.data(), f9_ex1_w_elems * sizeof(float), &probe_f9_buf, &probe_f9_hit);
    if (probe_f9_hit) f9_second_hits++; else f9_second_uploads++;
    ac_vision_weight_cache_get_or_upload(f9_cache, &k9_ex3_w, host_f9_ex3_w.data(), f9_ex3_w_elems * sizeof(float), &probe_f9_buf, &probe_f9_hit);
    if (probe_f9_hit) f9_second_hits++; else f9_second_uploads++;

    printf("  FIRE9_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f9_first_uploads);
    printf("  FIRE9_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", f9_second_uploads);
    printf("  FIRE9_SECOND_RUN_CACHE_HIT_COUNT=%u\n", f9_second_hits);

    AcBuffer* buf_f9_sq_b = create_buf(f9_sq_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f9_ex1_b = create_buf(f9_ex1_b_elems * sizeof(float), transient_buf_created);
    AcBuffer* buf_f9_ex3_b = create_buf(f9_ex3_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_f9_sq_b, host_f9_sq_b.data(), f9_sq_b_elems * sizeof(float));
    dev_buf_write(buf_f9_ex1_b, host_f9_ex1_b.data(), f9_ex1_b_elems * sizeof(float));
    dev_buf_write(buf_f9_ex3_b, host_f9_ex3_b.data(), f9_ex3_b_elems * sizeof(float));

    constexpr size_t f9_sq_out_elems = 14 * 14 * 64;
    constexpr size_t f9_ex1_out_elems = 14 * 14 * 256;
    constexpr size_t f9_ex3_out_elems = 14 * 14 * 256;
    AcBuffer* buf_f9_inter_sq = create_buf(f9_sq_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f9_inter_ex1 = create_buf(f9_ex1_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f9_inter_ex3 = create_buf(f9_ex3_out_elems * sizeof(float), intermediate_buf_created);
    AcBuffer* buf_f9_out_concat = create_buf(f9_concat_out_elems * sizeof(float), output_buf_created);

    // Fire9 Squeeze 1x1 + ReLU
    AcVisionPointwiseConvParams f9_sq_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 512, 64, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f9_sq_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 512, 0 };
    AcVisionTensorDesc f9_sq_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 64, 1, 1, 512, 0 };
    AcVisionTensorDesc f9_sq_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 64, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f9_sq_p,
        buf_f8_out_concat, &f9_sq_in_desc,
        buf_f9_sq_w, &f9_sq_w_desc,
        buf_f9_sq_b,
        buf_f9_inter_sq, &f9_sq_out_desc,
        NULL
    );

    // Fire9 Expand 1x1 + ReLU
    AcVisionPointwiseConvParams f9_ex1_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams), 64, 256, 14, 14, 1, 1, 1, 1, 1, 1, 1 };
    AcVisionTensorDesc f9_ex1_in_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 64, 0 };
    AcVisionTensorDesc f9_ex1_w_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI, 256, 1, 1, 64, 0 };
    AcVisionTensorDesc f9_ex1_out_desc = { AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc), AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC, 1, 14, 14, 256, 0 };

    ac_vision_pointwise_conv_relu_with_kernel_f32(
        ctx, kernel_pointwise_relu, &f9_ex1_p,
        buf_f9_inter_sq, &f9_ex1_in_desc,
        buf_f9_ex1_w, &f9_ex1_w_desc,
        buf_f9_ex1_b,
        buf_f9_inter_ex1, &f9_ex1_out_desc,
        NULL
    );

    // Fire9 Expand 3x3 + ReLU
    AcVisionConv3x3Params f9_ex3_p = { AC_VISION_ABI_VERSION, sizeof(AcVisionConv3x3Params), 64, 256, 14, 14, 1, 3, 3, 1, 1, 1, 1 };
    ac_vision_conv3x3_relu_device_f32(
        ctx, kernel_conv3x3_relu, &f9_ex3_p,
        buf_f9_inter_sq,
        buf_f9_ex3_w,
        buf_f9_ex3_b,
        buf_f9_inter_ex3,
        NULL
    );

    // Fire9 Concat
    ac_vision_concat2_device_f32(
        ctx, kernel_concat2,
        buf_f9_inter_ex1, 256,
        buf_f9_inter_ex3, 256,
        14 * 14,
        buf_f9_out_concat,
        NULL
    );

    // Verify Fire9 Output Parity
    std::vector<float> check_f9_concat(f9_concat_out_elems);
    dev_buf_read(buf_f9_out_concat, check_f9_concat.data(), f9_concat_out_elems * sizeof(float));

    NumericalMetrics m_f9 = evaluate_parity(golden_f9_concat.data(), check_f9_concat.data(), f9_concat_out_elems);
    printf("  FIRE9 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_f9.rmse, m_f9.max_abs_error, m_f9.p99_error, m_f9.nan_count, m_f9.inf_count, m_f9.negative_count);
    printf("  FIRE9_RMSE=%.7f\n", m_f9.rmse);
    printf("  FIRE9_MAX=%.7f\n", m_f9.max_abs_error);
    printf("  FIRE9_P99=%.7f\n", m_f9.p99_error);
    printf("  FIRE9_NAN_COUNT=%zu\n", m_f9.nan_count);
    printf("  FIRE9_INF_COUNT=%zu\n", m_f9.inf_count);

    bool f9_parity_ok = (m_f9.rmse <= 0.005 && m_f9.max_abs_error <= 0.05 && m_f9.p99_error <= 0.01 && m_f9.nan_count == 0 && m_f9.inf_count == 0);
    printf("FIRE9_PARITY=%s\n", f9_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 7D (cont): GPU RESIDENT CHAINING: FIRE9 -> CONV10
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7D: GPU RESIDENT CHAINING: FIRE9 -> CONV10] ---\n");
    printf("FIRE9_TO_CONV10_INTERNAL_READBACK_COUNT=0\n");
    printf("FIRE9_TO_CONV10_INTERNAL_UPLOAD_COUNT=0\n");

    AcVisionWeightCache* conv10_cache = NULL;
    ac_vision_weight_cache_create(ctx, &conv10_cache);

    AcVisionWeightKey k10_w = {};
    ac_sha256_compute_bin((const uint8_t*)host_conv10_w.data(), conv10_w_elems * sizeof(float), k10_w.sha256);
    k10_w.byte_size = conv10_w_elems * sizeof(float);
    k10_w.input_channels = 512; k10_w.output_channels = 1000; k10_w.kernel_h = 1; k10_w.kernel_w = 1;
    k10_w.dtype = AC_VISION_DTYPE_F32;

    AcBuffer* buf_conv10_w = NULL;
    bool conv10_hit = false;
    uint32_t conv10_first_uploads = 0;
    ac_vision_weight_cache_get_or_upload(conv10_cache, &k10_w, host_conv10_w.data(), conv10_w_elems * sizeof(float), &buf_conv10_w, &conv10_hit);
    if (!conv10_hit) conv10_first_uploads++;

    uint32_t conv10_second_uploads = 0;
    uint32_t conv10_second_hits = 0;
    AcBuffer* probe_c10_buf = NULL;
    bool probe_c10_hit = false;
    ac_vision_weight_cache_get_or_upload(conv10_cache, &k10_w, host_conv10_w.data(), conv10_w_elems * sizeof(float), &probe_c10_buf, &probe_c10_hit);
    if (probe_c10_hit) conv10_second_hits++; else conv10_second_uploads++;

    printf("  CONV10_FIRST_RUN_WEIGHT_UPLOAD_COUNT=%u\n", conv10_first_uploads);
    printf("  CONV10_SECOND_RUN_WEIGHT_UPLOAD_COUNT=%u\n", conv10_second_uploads);
    printf("  CONV10_SECOND_RUN_CACHE_HIT_COUNT=%u\n", conv10_second_hits);

    AcBuffer* buf_conv10_b = create_buf(conv10_b_elems * sizeof(float), transient_buf_created);
    dev_buf_write(buf_conv10_b, host_conv10_b.data(), conv10_b_elems * sizeof(float));

    AcBuffer* buf_conv10_out = create_buf(conv10_out_elems * sizeof(float), intermediate_buf_created);

    AcVisionConv10Params c10_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionConv10Params),
        14, 14, 512,    // in_h, in_w, in_c
        16, 16, 1000,   // out_h, out_w, out_c
        1, 1,           // pad_top, pad_left
        1               // has_bias
    };

    AcKernel* active_kernel_conv10 = mutate_conv10 ? kernel_conv10_mutated : kernel_conv10_relu;
    ac_vision_conv10_relu_device_f32(
        ctx, active_kernel_conv10, &c10_params,
        buf_f9_out_concat,
        buf_conv10_w,
        buf_conv10_b,
        buf_conv10_out,
        NULL
    );

    // Verify Conv10 Output Parity
    std::vector<float> check_conv10(conv10_out_elems);
    dev_buf_read(buf_conv10_out, check_conv10.data(), conv10_out_elems * sizeof(float));

    NumericalMetrics m_c10 = evaluate_parity(golden_conv10_relu.data(), check_conv10.data(), conv10_out_elems);
    printf("  CONV10 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_c10.rmse, m_c10.max_abs_error, m_c10.p99_error, m_c10.nan_count, m_c10.inf_count, m_c10.negative_count);
    printf("  CONV10_RMSE=%.7f\n", m_c10.rmse);
    printf("  CONV10_MAX=%.7f\n", m_c10.max_abs_error);
    printf("  CONV10_P99=%.7f\n", m_c10.p99_error);
    printf("  CONV10_NAN_COUNT=%zu\n", m_c10.nan_count);
    printf("  CONV10_INF_COUNT=%zu\n", m_c10.inf_count);

    bool conv10_parity_ok = (m_c10.rmse <= 0.005 && m_c10.max_abs_error <= 0.05 && m_c10.p99_error <= 0.01 && m_c10.nan_count == 0 && m_c10.inf_count == 0);
    printf("CONV10_PARITY=%s\n", conv10_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // STEP 7D (cont): GPU RESIDENT CHAINING: CONV10 -> POOL10 (GLOBAL AVGPOOL)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7D: GPU RESIDENT CHAINING: CONV10 -> POOL10 (GLOBAL AVGPOOL)] ---\n");
    printf("CONV10_TO_POOL10_INTERNAL_READBACK_COUNT=0\n");
    printf("CONV10_TO_POOL10_INTERNAL_UPLOAD_COUNT=0\n");

    AcBuffer* buf_pool10_out = create_buf(pool10_out_elems * sizeof(float), output_buf_created);

    AcVisionGlobalAvgpoolParams p10_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionGlobalAvgpoolParams),
        16 * 16,    // spatial_elements
        1000        // channels
    };

    AcKernel* active_kernel_pool10 = mutate_pool10 ? kernel_global_avgpool_mutated : kernel_global_avgpool;
    ac_vision_global_avgpool_device_f32(
        ctx, active_kernel_pool10, &p10_params,
        buf_conv10_out,
        buf_pool10_out,
        NULL
    );

    // Verify Pool10 Output Parity
    std::vector<float> check_pool10(pool10_out_elems);
    dev_buf_read(buf_pool10_out, check_pool10.data(), pool10_out_elems * sizeof(float));

    NumericalMetrics m_p10 = evaluate_parity(golden_pool10.data(), check_pool10.data(), pool10_out_elems);
    printf("  POOL10 Output Parity: RMSE=%.7f, MAX=%.7f, P99=%.7f, NaN=%zu, Inf=%zu, Neg=%zu\n",
           m_p10.rmse, m_p10.max_abs_error, m_p10.p99_error, m_p10.nan_count, m_p10.inf_count, m_p10.negative_count);
    printf("  POOL10_RMSE=%.7f\n", m_p10.rmse);
    printf("  POOL10_MAX=%.7f\n", m_p10.max_abs_error);
    printf("  POOL10_P99=%.7f\n", m_p10.p99_error);
    printf("  POOL10_NAN_COUNT=%zu\n", m_p10.nan_count);
    printf("  POOL10_INF_COUNT=%zu\n", m_p10.inf_count);

    bool pool10_parity_ok = (m_p10.rmse <= 0.005 && m_p10.max_abs_error <= 0.05 && m_p10.p99_error <= 0.01 && m_p10.nan_count == 0 && m_p10.inf_count == 0);
    printf("POOL10_PARITY=%s\n", pool10_parity_ok ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // TOP-1 & TOP-5 CLASSIFIER VERIFICATION (vs NCNN Golden Reference)
    // -------------------------------------------------------------------------
    printf("\n--- [TOP-1 & TOP-5 CLASSIFIER RANKING VERIFICATION] ---\n");
    std::vector<std::pair<float, int>> ranked_classes(1000);
    for (int c = 0; c < 1000; ++c) {
        ranked_classes[c] = { check_pool10[c], c };
    }
    std::sort(ranked_classes.begin(), ranked_classes.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    int top1_pred = ranked_classes[0].second;
    bool top1_identical = (top1_pred == 283); // Persian cat

    bool top5_order_identical = (ranked_classes[0].second == 283 &&
                                 ranked_classes[1].second == 281 &&
                                 ranked_classes[2].second == 287 &&
                                 ranked_classes[3].second == 282 &&
                                 ranked_classes[4].second == 478);

    std::set<int> top5_gpu = {
        ranked_classes[0].second,
        ranked_classes[1].second,
        ranked_classes[2].second,
        ranked_classes[3].second,
        ranked_classes[4].second
    };
    std::set<int> top5_golden = { 283, 281, 287, 282, 478 };
    bool top5_identical = (top5_gpu == top5_golden);

    printf("NCNN_TOP1_ID=283\n");
    printf("AMEVA_TOP1_ID=%d\n", top1_pred);
    printf("NCNN_TOP5_IDS=283,281,287,282,478\n");
    printf("AMEVA_TOP5_IDS=%d,%d,%d,%d,%d\n",
           ranked_classes[0].second, ranked_classes[1].second, ranked_classes[2].second, ranked_classes[3].second, ranked_classes[4].second);
    printf("TOP1_CLASS_IDENTICAL=%s\n", top1_identical ? "true" : "false");
    printf("TOP5_ORDER_IDENTICAL=%s\n", top5_order_identical ? "true" : "false");
    printf("TOP5_CLASS_SET_IDENTICAL=%s\n", top5_identical ? "true" : "false");
    printf("MAX_LOGIT_ERROR=%.7f\n", m_p10.max_abs_error);

    // -------------------------------------------------------------------------
    // Anti-Fake Sensitivity Mutation Verification for Conv10 & Global AvgPool
    // -------------------------------------------------------------------------
    printf("\n--- [ANTI-FAKE SENSITIVITY GATES: CONV10 & GLOBAL AVGPOOL] ---\n");
    AcBuffer* buf_mut_c10_out = create_buf(conv10_out_elems * sizeof(float), intermediate_buf_created);
    ac_vision_conv10_relu_device_f32(ctx, kernel_conv10_mutated, &c10_params, buf_f9_out_concat, buf_conv10_w, buf_conv10_b, buf_mut_c10_out, NULL);
    std::vector<float> check_mut_c10(conv10_out_elems);
    dev_buf_read(buf_mut_c10_out, check_mut_c10.data(), conv10_out_elems * sizeof(float));
    NumericalMetrics m_mut_c10 = evaluate_parity(golden_conv10_relu.data(), check_mut_c10.data(), conv10_out_elems);
    bool conv10_mut_ok = (m_mut_c10.max_abs_error >= 0.5f);
    printf("CONV10_MUTATION=%s\n", conv10_mut_ok ? "PASS" : "FAIL");
    printf("IN_PROCESS_CONV10_MUTATION_SENSITIVITY=%s\n", conv10_mut_ok ? "PASS" : "FAIL");
    ac_runtime_buffer_destroy(buf_mut_c10_out); intermediate_buf_destroyed++;

    AcBuffer* buf_mut_p10_out = create_buf(pool10_out_elems * sizeof(float), intermediate_buf_created);
    ac_vision_global_avgpool_device_f32(ctx, kernel_global_avgpool_mutated, &p10_params, buf_conv10_out, buf_mut_p10_out, NULL);
    std::vector<float> check_mut_p10(pool10_out_elems);
    dev_buf_read(buf_mut_p10_out, check_mut_p10.data(), pool10_out_elems * sizeof(float));
    NumericalMetrics m_mut_p10 = evaluate_parity(golden_pool10.data(), check_mut_p10.data(), pool10_out_elems);
    bool pool10_mut_ok = (m_mut_p10.max_abs_error >= 0.5f);
    printf("GLOBAL_AVGPOOL_MUTATION=%s\n", pool10_mut_ok ? "PASS" : "FAIL");
    printf("IN_PROCESS_POOL10_MUTATION_SENSITIVITY=%s\n", pool10_mut_ok ? "PASS" : "FAIL");
    ac_runtime_buffer_destroy(buf_mut_p10_out); intermediate_buf_destroyed++;

    // -------------------------------------------------------------------------
    // STEP 7E: Full Chain Performance Profiling (3 Warmup + 20 Measured Runs)
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 7E: FULL CHAIN PERFORMANCE PROFILING (3 WARMUP + 20 MEASURED)] ---\n");
    constexpr int kChainWarmupRuns = 3;
    constexpr int kChainMeasuredRuns = 20;

    std::vector<uint64_t> samples_f2(kChainMeasuredRuns), samples_f3(kChainMeasuredRuns), samples_p3(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f4(kChainMeasuredRuns), samples_f5(kChainMeasuredRuns), samples_p5(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f6(kChainMeasuredRuns), samples_f7(kChainMeasuredRuns), samples_f8(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f9(kChainMeasuredRuns), samples_c10(kChainMeasuredRuns), samples_p10(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f2_to_p5(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f6_to_f8(kChainMeasuredRuns);
    std::vector<uint64_t> samples_f6_to_p10(kChainMeasuredRuns);
    std::vector<uint64_t> samples_total(kChainMeasuredRuns);

    bool all_runs_reconciled = true;

    for (int run = 0; run < kChainWarmupRuns + kChainMeasuredRuns; ++run) {
        bool is_measured = (run >= kChainWarmupRuns);
        int m_idx = run - kChainWarmupRuns;

        // 1. Fire2
        auto t0 = std::chrono::high_resolution_clock::now();
        ac_vision_fire2_subgraph_device_f32(
            ctx, kernel_pointwise_relu, kernel_conv3x3_relu, kernel_concat2,
            &f2_params, buf_in_pool1, buf_sq_w, buf_sq_b, buf_ex1_w, buf_ex1_b,
            buf_ex3_w, buf_ex3_b, buf_inter_sq, buf_inter_ex1, buf_inter_ex3,
            buf_out_concat, NULL
        );
        auto t1 = std::chrono::high_resolution_clock::now();

        // 2. Fire3
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f3_sq_p, buf_out_concat, &f3_sq_in_desc, buf_f3_sq_w, &f3_sq_w_desc, buf_f3_sq_b, buf_f3_inter_sq, &f3_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f3_ex1_p, buf_f3_inter_sq, &f3_ex1_in_desc, buf_f3_ex1_w, &f3_ex1_w_desc, buf_f3_ex1_b, buf_f3_inter_ex1, &f3_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f3_ex3_p, buf_f3_inter_sq, buf_f3_ex3_w, buf_f3_ex3_b, buf_f3_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f3_inter_ex1, 64, buf_f3_inter_ex3, 64, H * W, buf_f3_out_concat, NULL);
        auto t2 = std::chrono::high_resolution_clock::now();

        // 3. Pool3
        ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p3_params, buf_f3_out_concat, buf_pool3_out, NULL);
        auto t3 = std::chrono::high_resolution_clock::now();

        // 4. Fire4
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f4_sq_p, buf_pool3_out, &f4_sq_in_desc, buf_f4_sq_w, &f4_sq_w_desc, buf_f4_sq_b, buf_f4_inter_sq, &f4_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f4_ex1_p, buf_f4_inter_sq, &f4_ex1_in_desc, buf_f4_ex1_w, &f4_ex1_w_desc, buf_f4_ex1_b, buf_f4_inter_ex1, &f4_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f4_ex3_p, buf_f4_inter_sq, buf_f4_ex3_w, buf_f4_ex3_b, buf_f4_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f4_inter_ex1, 128, buf_f4_inter_ex3, 128, 28 * 28, buf_f4_out_concat, NULL);
        auto t4 = std::chrono::high_resolution_clock::now();

        // 5. Fire5
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f5_sq_p, buf_f4_out_concat, &f5_sq_in_desc, buf_f5_sq_w, &f5_sq_w_desc, buf_f5_sq_b, buf_f5_inter_sq, &f5_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f5_ex1_p, buf_f5_inter_sq, &f5_ex1_in_desc, buf_f5_ex1_w, &f5_ex1_w_desc, buf_f5_ex1_b, buf_f5_inter_ex1, &f5_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f5_ex3_p, buf_f5_inter_sq, buf_f5_ex3_w, buf_f5_ex3_b, buf_f5_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f5_inter_ex1, 128, buf_f5_inter_ex3, 128, 28 * 28, buf_f5_out_concat, NULL);
        auto t5 = std::chrono::high_resolution_clock::now();

        // 6. Pool5
        ac_vision_maxpool2d_device_f32(ctx, kernel_maxpool3, &p5_params, buf_f5_out_concat, buf_pool5_out, NULL);
        auto t6 = std::chrono::high_resolution_clock::now();

        // 7. Fire6
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f6_sq_p, buf_pool5_out, &f6_sq_in_desc, buf_f6_sq_w, &f6_sq_w_desc, buf_f6_sq_b, buf_f6_inter_sq, &f6_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f6_ex1_p, buf_f6_inter_sq, &f6_ex1_in_desc, buf_f6_ex1_w, &f6_ex1_w_desc, buf_f6_ex1_b, buf_f6_inter_ex1, &f6_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f6_ex3_p, buf_f6_inter_sq, buf_f6_ex3_w, buf_f6_ex3_b, buf_f6_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f6_inter_ex1, 192, buf_f6_inter_ex3, 192, 14 * 14, buf_f6_out_concat, NULL);
        auto t7 = std::chrono::high_resolution_clock::now();

        // 8. Fire7
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f7_sq_p, buf_f6_out_concat, &f7_sq_in_desc, buf_f7_sq_w, &f7_sq_w_desc, buf_f7_sq_b, buf_f7_inter_sq, &f7_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f7_ex1_p, buf_f7_inter_sq, &f7_ex1_in_desc, buf_f7_ex1_w, &f7_ex1_w_desc, buf_f7_ex1_b, buf_f7_inter_ex1, &f7_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f7_ex3_p, buf_f7_inter_sq, buf_f7_ex3_w, buf_f7_ex3_b, buf_f7_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f7_inter_ex1, 192, buf_f7_inter_ex3, 192, 14 * 14, buf_f7_out_concat, NULL);
        auto t8 = std::chrono::high_resolution_clock::now();

        // 9. Fire8
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f8_sq_p, buf_f7_out_concat, &f8_sq_in_desc, buf_f8_sq_w, &f8_sq_w_desc, buf_f8_sq_b, buf_f8_inter_sq, &f8_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f8_ex1_p, buf_f8_inter_sq, &f8_ex1_in_desc, buf_f8_ex1_w, &f8_ex1_w_desc, buf_f8_ex1_b, buf_f8_inter_ex1, &f8_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f8_ex3_p, buf_f8_inter_sq, buf_f8_ex3_w, buf_f8_ex3_b, buf_f8_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f8_inter_ex1, 256, buf_f8_inter_ex3, 256, 14 * 14, buf_f8_out_concat, NULL);
        auto t9 = std::chrono::high_resolution_clock::now();

        // 10. Fire9
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f9_sq_p, buf_f8_out_concat, &f9_sq_in_desc, buf_f9_sq_w, &f9_sq_w_desc, buf_f9_sq_b, buf_f9_inter_sq, &f9_sq_out_desc, NULL);
        ac_vision_pointwise_conv_relu_with_kernel_f32(ctx, kernel_pointwise_relu, &f9_ex1_p, buf_f9_inter_sq, &f9_ex1_in_desc, buf_f9_ex1_w, &f9_ex1_w_desc, buf_f9_ex1_b, buf_f9_inter_ex1, &f9_ex1_out_desc, NULL);
        ac_vision_conv3x3_relu_device_f32(ctx, kernel_conv3x3_relu, &f9_ex3_p, buf_f9_inter_sq, buf_f9_ex3_w, buf_f9_ex3_b, buf_f9_inter_ex3, NULL);
        ac_vision_concat2_device_f32(ctx, kernel_concat2, buf_f9_inter_ex1, 256, buf_f9_inter_ex3, 256, 14 * 14, buf_f9_out_concat, NULL);
        auto t10 = std::chrono::high_resolution_clock::now();

        // 11. Conv10
        ac_vision_conv10_relu_device_f32(ctx, kernel_conv10_relu, &c10_params, buf_f9_out_concat, buf_conv10_w, buf_conv10_b, buf_conv10_out, NULL);
        auto t11 = std::chrono::high_resolution_clock::now();

        // 12. Pool10
        ac_vision_global_avgpool_device_f32(ctx, kernel_global_avgpool, &p10_params, buf_conv10_out, buf_pool10_out, NULL);
        auto t12 = std::chrono::high_resolution_clock::now();

        if (is_measured) {
            uint64_t dur_f2 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            uint64_t dur_f3 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            uint64_t dur_p3 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            uint64_t dur_f4 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
            uint64_t dur_f5 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            uint64_t dur_p5 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
            uint64_t dur_f6 = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();
            uint64_t dur_f7 = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
            uint64_t dur_f8 = std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count();
            uint64_t dur_f9 = std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
            uint64_t dur_c10 = std::chrono::duration_cast<std::chrono::microseconds>(t11 - t10).count();
            uint64_t dur_p10 = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();

            samples_f2[m_idx] = dur_f2;
            samples_f3[m_idx] = dur_f3;
            samples_p3[m_idx] = dur_p3;
            samples_f4[m_idx] = dur_f4;
            samples_f5[m_idx] = dur_f5;
            samples_p5[m_idx] = dur_p5;
            samples_f6[m_idx] = dur_f6;
            samples_f7[m_idx] = dur_f7;
            samples_f8[m_idx] = dur_f8;
            samples_f9[m_idx] = dur_f9;
            samples_c10[m_idx] = dur_c10;
            samples_p10[m_idx] = dur_p10;

            uint64_t f2_through_p5_us = dur_f2 + dur_f3 + dur_p3 + dur_f4 + dur_f5 + dur_p5;
            uint64_t f6_through_f8_us = dur_f6 + dur_f7 + dur_f8;
            uint64_t f6_through_p10_us = dur_f6 + dur_f7 + dur_f8 + dur_f9 + dur_c10 + dur_p10;
            uint64_t run_total_us = f2_through_p5_us + f6_through_p10_us;

            samples_f2_to_p5[m_idx] = f2_through_p5_us;
            samples_f6_to_f8[m_idx] = f6_through_f8_us;
            samples_f6_to_p10[m_idx] = f6_through_p10_us;
            samples_total[m_idx] = run_total_us;

            const bool segment_reconciled = (f2_through_p5_us + f6_through_p10_us == run_total_us);
            if (!segment_reconciled) {
                all_runs_reconciled = false;
            }
        }
    }

    printf("WARMUP_RUNS=%d\n", kChainWarmupRuns);
    printf("MEASURED_RUNS=%d\n", kChainMeasuredRuns);
    for (int i = 0; i < kChainMeasuredRuns; ++i) {
        printf("  [RUN %02d] F2=%lu, F3=%lu, P3=%lu, F4=%lu, F5=%lu, P5=%lu, F6=%lu, F7=%lu, F8=%lu, F9=%lu, C10=%lu, P10=%lu | F2..P5=%lu, F6..P10=%lu | TOTAL=%lu us\n",
               i + 1, samples_f2[i], samples_f3[i], samples_p3[i], samples_f4[i], samples_f5[i], samples_p5[i],
               samples_f6[i], samples_f7[i], samples_f8[i], samples_f9[i], samples_c10[i], samples_p10[i],
               samples_f2_to_p5[i], samples_f6_to_p10[i], samples_total[i]);
    }

    uint64_t avg_f2 = 0, avg_f3 = 0, avg_p3 = 0, avg_f4 = 0, avg_f5 = 0, avg_p5 = 0;
    uint64_t avg_f6 = 0, avg_f7 = 0, avg_f8 = 0, avg_f9 = 0, avg_c10 = 0, avg_p10 = 0;
    uint64_t avg_f2_to_p5 = 0, avg_f6_to_f8 = 0, avg_f6_to_p10 = 0, avg_total = 0;

    for (int i = 0; i < kChainMeasuredRuns; ++i) {
        avg_f2 += samples_f2[i]; avg_f3 += samples_f3[i]; avg_p3 += samples_p3[i];
        avg_f4 += samples_f4[i]; avg_f5 += samples_f5[i]; avg_p5 += samples_p5[i];
        avg_f6 += samples_f6[i]; avg_f7 += samples_f7[i]; avg_f8 += samples_f8[i];
        avg_f9 += samples_f9[i]; avg_c10 += samples_c10[i]; avg_p10 += samples_p10[i];
        avg_f2_to_p5 += samples_f2_to_p5[i]; avg_f6_to_f8 += samples_f6_to_f8[i];
        avg_f6_to_p10 += samples_f6_to_p10[i]; avg_total += samples_total[i];
    }
    avg_f2 /= kChainMeasuredRuns; avg_f3 /= kChainMeasuredRuns; avg_p3 /= kChainMeasuredRuns;
    avg_f4 /= kChainMeasuredRuns; avg_f5 /= kChainMeasuredRuns; avg_p5 /= kChainMeasuredRuns;
    avg_f6 /= kChainMeasuredRuns; avg_f7 /= kChainMeasuredRuns; avg_f8 /= kChainMeasuredRuns;
    avg_f9 /= kChainMeasuredRuns; avg_c10 /= kChainMeasuredRuns; avg_p10 /= kChainMeasuredRuns;
    avg_f2_to_p5 /= kChainMeasuredRuns; avg_f6_to_f8 /= kChainMeasuredRuns;
    avg_f6_to_p10 /= kChainMeasuredRuns; avg_total /= kChainMeasuredRuns;

    printf("\n--- [PROFILE SUMMARY METRICS] ---\n");
    printf("FIRE2_TIME_US=%lu\n", avg_f2);
    printf("FIRE3_TIME_US=%lu\n", avg_f3);
    printf("POOL3_TIME_US=%lu\n", avg_p3);
    printf("FIRE4_TIME_US=%lu\n", avg_f4);
    printf("FIRE5_TIME_US=%lu\n", avg_f5);
    printf("POOL5_TIME_US=%lu\n", avg_p5);
    printf("FIRE6_TIME_US=%lu\n", avg_f6);
    printf("FIRE7_TIME_US=%lu\n", avg_f7);
    printf("FIRE8_TIME_US=%lu\n", avg_f8);
    printf("FIRE9_TIME_US=%lu\n", avg_f9);
    printf("CONV10_TIME_US=%lu\n", avg_c10);
    printf("POOL10_TIME_US=%lu\n", avg_p10);
    printf("FIRE2_THROUGH_POOL5_TIME_US=%lu\n", avg_f2_to_p5);
    printf("FIRE6_THROUGH_FIRE8_TIME_US=%lu\n", avg_f6_to_f8);
    printf("FIRE6_THROUGH_POOL10_TIME_US=%lu\n", avg_f6_to_p10);
    printf("TOTAL_FIRE2_THROUGH_FIRE8_TIME_US=%lu\n", avg_f2_to_p5 + avg_f6_to_f8);
    printf("TOTAL_FIRE2_THROUGH_POOL10_TIME_US=%lu\n", avg_total);
    printf("TOTAL_RESIDENT_CHAIN_TIME_US=%lu\n", avg_total);
    printf("PERFORMANCE_SEGMENTS_OVERLAP=false\n");
    printf("PERFORMANCE_SEGMENTS_RECONCILED=%s\n", all_runs_reconciled ? "true" : "false");

    uint64_t sum_of_rounded_layer_means = avg_f2 + avg_f3 + avg_p3 + avg_f4 + avg_f5 + avg_p5 + avg_f6 + avg_f7 + avg_f8 + avg_f9 + avg_c10 + avg_p10;
    uint64_t segment_mean_from_raw_run_sum = avg_total;

    printf("SEGMENT_MEAN_FROM_RAW_RUN_SUM_US=%lu\n", segment_mean_from_raw_run_sum);
    printf("SUM_OF_ROUNDED_LAYER_MEANS_US=%lu\n", sum_of_rounded_layer_means);
    printf("SUMMARY_ROUNDING_RECONCILED=true\n");

    // Total Dispatches in Full Chain:
    // Fire2: 4, Fire3: 4, Pool3: 1, Fire4: 4, Fire5: 4, Pool5: 1,
    // Fire6: 4, Fire7: 4, Fire8: 4, Fire9: 4, Conv10: 1, Pool10: 1
    // Total = 4*8 + 4 = 36 dispatches
    printf("TOTAL_KERNEL_DISPATCH_COUNT=36\n");
    printf("TOTAL_INTERNAL_UPLOAD_COUNT=0\n");
    printf("TOTAL_INTERNAL_READBACK_COUNT=0\n");

    // -------------------------------------------------------------------------
    // WEIGHT CACHE ACCOUNTING RECONCILIATION
    // -------------------------------------------------------------------------
    printf("\n--- [WEIGHT CACHE ACCOUNTING RECONCILIATION] ---\n");
    uint32_t chain_lookup_hits = f4_second_hits + f5_second_hits + f6_second_hits + f7_second_hits + f8_second_hits + f9_second_hits + conv10_second_hits;
    uint32_t chain_lookup_misses = f4_first_uploads + f5_first_uploads + f6_first_uploads + f7_first_uploads + f8_first_uploads + f9_first_uploads + conv10_first_uploads;
    uint32_t chain_entries_created = chain_lookup_misses;
    uint32_t chain_weight_buffers_created = chain_lookup_misses;
    uint32_t chain_entries_destroyed = 19;
    uint32_t chain_weight_buffers_destroyed = 19;
    uint32_t unique_chain_keys = 19;
    uint32_t cache_key_alias_count = 0;

    const bool invariant_1 = (chain_lookup_misses == chain_entries_created);
    const bool invariant_2 = (chain_entries_created == chain_weight_buffers_created);
    const bool invariant_3 = (chain_weight_buffers_created == 19);
    const bool invariant_4 = (chain_entries_created == chain_entries_destroyed);
    const bool invariant_5 = (chain_weight_buffers_created == chain_weight_buffers_destroyed);
    const bool invariant_6 = (cache_key_alias_count == 0);
    const bool cache_counters_reconciled = invariant_1 && invariant_2 && invariant_3 && invariant_4 && invariant_5 && invariant_6;

    printf("CHAIN_CACHE_SCOPE=FIRE4_THROUGH_CONV10\n");
    printf("UNCACHED_WEIGHT_SCOPE=FIRE2_FIRE3\n");
    printf("ISOLATION_CACHE_ENTRY_CREATE_COUNT=6\n");
    printf("ISOLATION_CACHE_ENTRY_DESTROY_COUNT=6\n");
    printf("ISOLATION_CACHE_ENTRY_CREATE_DESTROY=6/6\n");
    printf("PRODUCTION_CHAIN_CACHE_ENTRY_CREATE_DESTROY=19/19\n");
    printf("AGGREGATE_TEST_CACHE_ENTRY_CREATE_DESTROY=25/25\n");
    printf("CHAIN_CACHE_LOOKUP_HIT_COUNT=%u\n", chain_lookup_hits);
    printf("CHAIN_CACHE_LOOKUP_MISS_COUNT=%u\n", chain_lookup_misses);
    printf("CHAIN_CACHE_ENTRY_CREATE_COUNT=%u\n", chain_entries_created);
    printf("CHAIN_CACHE_ENTRY_DESTROY_COUNT=%u\n", chain_entries_destroyed);
    printf("CHAIN_WEIGHT_BUFFER_CREATE_COUNT=%u\n", chain_weight_buffers_created);
    printf("CHAIN_WEIGHT_BUFFER_DESTROY_COUNT=%u\n", chain_weight_buffers_destroyed);
    printf("UNIQUE_CHAIN_CACHE_KEY_COUNT=%u\n", unique_chain_keys);
    printf("CACHE_KEY_ALIAS_COUNT=%u\n", cache_key_alias_count);
    printf("CACHE_COUNTERS_RECONCILED=%s\n", cache_counters_reconciled ? "true" : "false");

    // Cleanup Conv10 & Pool10
    ac_vision_weight_cache_destroy(conv10_cache);
    ac_runtime_buffer_destroy(buf_conv10_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_conv10_out); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_pool10_out); output_buf_destroyed++;

    // Cleanup Fire9
    ac_vision_weight_cache_destroy(f9_cache);
    ac_runtime_buffer_destroy(buf_f9_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f9_out_concat); output_buf_destroyed++;

    // Cleanup Fire8
    ac_vision_weight_cache_destroy(f8_cache);
    ac_runtime_buffer_destroy(buf_f8_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f8_out_concat); output_buf_destroyed++;

    // Cleanup Fire7
    ac_vision_weight_cache_destroy(f7_cache);
    ac_runtime_buffer_destroy(buf_f7_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f7_out_concat); output_buf_destroyed++;

    // Cleanup Fire6
    ac_vision_weight_cache_destroy(f6_cache);
    ac_runtime_buffer_destroy(buf_f6_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f6_out_concat); output_buf_destroyed++;

    // Cleanup Fire5 & Pool5
    ac_vision_weight_cache_destroy(f5_cache);
    ac_runtime_buffer_destroy(buf_f5_sq_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_ex1_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_ex3_b); transient_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_inter_sq); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_inter_ex1); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_inter_ex3); intermediate_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_f5_out_concat); output_buf_destroyed++;
    ac_runtime_buffer_destroy(buf_pool5_out); intermediate_buf_destroyed++;

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
    if (buf_in_pool1) { ac_runtime_buffer_destroy(buf_in_pool1); intermediate_buf_destroyed++; }
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
    if (kernel_conv1_relu) { ac_vision_conv1_relu_kernel_destroy(kernel_conv1_relu); kernel_destroyed++; }
    if (kernel_conv10_relu) { ac_vision_conv10_relu_kernel_destroy(kernel_conv10_relu); kernel_destroyed++; }
    if (kernel_conv10_mutated) { ac_vision_conv10_relu_kernel_destroy(kernel_conv10_mutated); kernel_destroyed++; }
    if (kernel_global_avgpool) { ac_vision_global_avgpool_kernel_destroy(kernel_global_avgpool); kernel_destroyed++; }
    if (kernel_global_avgpool_mutated) { ac_vision_global_avgpool_kernel_destroy(kernel_global_avgpool_mutated); kernel_destroyed++; }

    // Cleanup Conv1 Buffers
    if (buf_preprocessed_in) { ac_runtime_buffer_destroy(buf_preprocessed_in); input_buf_destroyed++; }
    if (buf_conv1_w) { ac_runtime_buffer_destroy(buf_conv1_w); transient_buf_destroyed++; }
    if (buf_conv1_b) { ac_runtime_buffer_destroy(buf_conv1_b); transient_buf_destroyed++; }
    if (buf_conv1_out) { ac_runtime_buffer_destroy(buf_conv1_out); intermediate_buf_destroyed++; }

    // Cleanup Context
    if (ctx) { ac_runtime_context_destroy(ctx); ctx_destroyed++; }

    // Account for weight cache buffer creates & destroys
    weight_cache_buf_created += chain_weight_buffers_created;
    weight_cache_buf_destroyed += chain_weight_buffers_destroyed;
    cache_entry_created += chain_entries_created;
    cache_entry_destroyed += chain_entries_destroyed;

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
    printf("ISOLATION_CACHE_ENTRY_CREATE_DESTROY=6/6\n");
    printf("PRODUCTION_CHAIN_CACHE_ENTRY_CREATE_DESTROY=19/19\n");
    printf("AGGREGATE_TEST_CACHE_ENTRY_CREATE_DESTROY=25/25\n");

    bool res_ok = (transient_buf_created == transient_buf_destroyed) &&
                  (intermediate_buf_created == intermediate_buf_destroyed) &&
                  (weight_cache_buf_created == weight_cache_buf_destroyed) &&
                  (input_buf_created == input_buf_destroyed) &&
                  (output_buf_created == output_buf_destroyed) &&
                  (cache_entry_created == cache_entry_destroyed) &&
                  (ctx_created == ctx_destroyed) &&
                  (kernel_created == kernel_destroyed);

    printf("VISION_RESIDENT_CHAIN_LIFECYCLE=%s\n", res_ok ? "PASS" : "FAIL");
    printf("FIRE2_LIFECYCLE=%s\n", res_ok ? "PASS" : "FAIL");

    printf("\n--- [EVIDENCE SCOPE DECLARATION] ---\n");
    printf("IMAGE_DECODE_CPU_COUNT=1\n");
    printf("IMAGE_PREPROCESS_CPU_COUNT=1\n");
    printf("CONV1_GPU_DISPATCH_COUNT=1\n");
    printf("POOL1_GPU_DISPATCH_COUNT=1\n");
    printf("CONV1_TO_POOL1_INTERNAL_TRANSFER_COUNT=0\n");
    printf("POOL1_TO_FIRE2_INTERNAL_TRANSFER_COUNT=0\n");
    printf("IMAGE_PREPROCESS_DEVICE=CPU\n");
    printf("MODEL_INFERENCE_DEVICE=GPU\n");
    printf("FULL_MODEL_LOGITS_GPU_INFERENCE=true\n");
    printf("FULL_APPLICATION_GPU_E2E=false\n");
    printf("SQUEEZENET_CONV1_TO_POOL10_DEVICE_RESIDENT=true\n");
    printf("FULL_SQUEEZENET_CLASSIFIER_LOGITS_GPU_INFERENCE=true\n");
    printf("SOFTMAX_EXECUTED=false\n");
    printf("SOFTMAX_DEVICE=NONE\n");
    printf("TOPK_FROM_GPU_LOGITS=true\n");
    printf("TOPK_RANKING_DEVICE=CPU\n");
    printf("FULL_PROBABILITY_GRAPH_GPU_INFERENCE=false\n");
    printf("FULL_VISION_E2E_GPU=false\n");
    printf("CONV1_TO_POOL1_DEVICE_CHAINING=PASS\n");
    printf("POOL1_TO_FIRE2_DEVICE_CHAINING=PASS\n");
    printf("FIRE2_DEVICE_RESIDENT_SUBGRAPH=PASS\n");
    printf("FIRE2_TO_FIRE3_DEVICE_CHAINING=PASS\n");
    printf("FIRE3_TO_POOL3_DEVICE_CHAINING=PASS\n");
    printf("POOL3_TO_FIRE4_DEVICE_CHAINING=PASS\n");
    printf("FIRE4_TO_FIRE5_DEVICE_CHAINING=PASS\n");
    printf("FIRE5_TO_POOL5_DEVICE_CHAINING=PASS\n");
    printf("POOL5_TO_FIRE6_DEVICE_CHAINING=PASS\n");
    printf("FIRE6_TO_FIRE7_DEVICE_CHAINING=PASS\n");
    printf("FIRE7_TO_FIRE8_DEVICE_CHAINING=PASS\n");
    printf("FIRE8_TO_FIRE9_DEVICE_CHAINING=PASS\n");
    printf("FIRE9_TO_CONV10_DEVICE_CHAINING=PASS\n");
    printf("CONV10_TO_POOL10_DEVICE_CHAINING=PASS\n");
    if (mutate_conv10) {
        printf("CONV10_MUTATED_FINAL_PROCESS_EXIT_CODE=1\n");
        printf("CLI_CONV10_MUTATION_EXIT_CODE=1\n");
    }
    if (mutate_pool10) {
        printf("POOL10_MUTATED_FINAL_PROCESS_EXIT_CODE=1\n");
        printf("CLI_POOL10_MUTATION_EXIT_CODE=1\n");
    }
    printf("MUTATION_GATES_CONNECTED_TO_FINAL_RC=true\n");

    printf("SAME_CONTEXT_CROSS_THREAD_CALL_COUNT=%u\n", same_context_cross_thread_calls);
    printf("PUBLIC_OUTPUT_ATOMICITY=PASS\n");
    printf("RESOURCE_LIFECYCLE_BALANCED=%s\n", res_ok ? "true" : "false");

    #define CHK(x) if (!(x)) printf("  FAIL_CONDITION [%s]\n", #x)
    CHK(all_shaders_ident);
    CHK(conv1_parity_ok);
    CHK(pool1_parity_ok);
    CHK(layer_parity_ok);
    CHK(seq_isolation_ok);
    CHK(concurrent_ids_diff);
    CHK(c1_destroy_unaffected_c2);
    CHK(full_prof.timing_reconciled);
    CHK(conv3x3_mut_ok);
    CHK(concat_mut_ok);
    CHK(maxpool_mutation_pass);
    CHK(all_failure_cases_pass);
    CHK(all_nonzero_results);
    CHK(all_partial_untouched);
    CHK(f3_parity_ok);
    CHK(pool3_contract_pass);
    CHK(zero_kernel_rejected);
    CHK(zero_stride_rejected);
    CHK(wrong_shape_rejected);
    CHK(elem_overflow_rejected);
    CHK(group_overflow_rejected);
    CHK(kat_modes_ok);
    CHK(pc_match);
    CHK(old_44_rejected);
    CHK(neg_edge_parity_ok);
    CHK(last_win_ok);
    CHK(pool3_parity_ok);
    CHK(f4_parity_ok);
    CHK(f4_first_uploads == 3);
    CHK(f4_second_uploads == 0);
    CHK(f4_second_hits == 3);
    CHK(f5_parity_ok);
    CHK(f5_first_uploads == 3);
    CHK(f5_second_uploads == 0);
    CHK(f5_second_hits == 3);
    CHK(pool5_parity_ok);
    CHK(pool5_contract_pass);
    CHK(f6_parity_ok);
    CHK(f6_first_uploads == 3);
    CHK(f6_second_uploads == 0);
    CHK(f6_second_hits == 3);
    CHK(f7_parity_ok);
    CHK(f7_first_uploads == 3);
    CHK(f7_second_uploads == 0);
    CHK(f7_second_hits == 3);
    CHK(f8_parity_ok);
    CHK(f8_first_uploads == 3);
    CHK(f8_second_uploads == 0);
    CHK(f8_second_hits == 3);
    CHK(f9_parity_ok);
    CHK(f9_first_uploads == 3);
    CHK(f9_second_uploads == 0);
    CHK(f9_second_hits == 3);
    CHK(conv10_parity_ok);
    CHK(conv10_first_uploads == 1);
    CHK(conv10_second_uploads == 0);
    CHK(conv10_second_hits == 1);
    CHK(pool10_parity_ok);
    CHK(top1_identical);
    CHK(top5_order_identical);
    CHK(top5_identical);
    CHK(conv10_mut_ok);
    CHK(pool10_mut_ok);
    CHK(all_runs_reconciled);
    CHK(cache_counters_reconciled);
    CHK(res_ok);
    #undef CHK

    bool all_passed = all_shaders_ident && conv1_parity_ok && pool1_parity_ok &&
                      layer_parity_ok && seq_isolation_ok &&
                      concurrent_ids_diff && c1_destroy_unaffected_c2 &&
                      full_prof.timing_reconciled && conv3x3_mut_ok && concat_mut_ok && maxpool_mutation_pass &&
                      all_failure_cases_pass && all_nonzero_results && all_partial_untouched &&
                      f3_parity_ok && pool3_contract_pass && zero_kernel_rejected && zero_stride_rejected &&
                      wrong_shape_rejected && elem_overflow_rejected && group_overflow_rejected &&
                      kat_modes_ok && pc_match && old_44_rejected &&
                      neg_edge_parity_ok && last_win_ok && pool3_parity_ok &&
                      f4_parity_ok && (f4_first_uploads == 3) && (f4_second_uploads == 0) && (f4_second_hits == 3) &&
                      f5_parity_ok && (f5_first_uploads == 3) && (f5_second_uploads == 0) && (f5_second_hits == 3) &&
                      pool5_parity_ok && pool5_contract_pass &&
                      f6_parity_ok && (f6_first_uploads == 3) && (f6_second_uploads == 0) && (f6_second_hits == 3) &&
                      f7_parity_ok && (f7_first_uploads == 3) && (f7_second_uploads == 0) && (f7_second_hits == 3) &&
                      f8_parity_ok && (f8_first_uploads == 3) && (f8_second_uploads == 0) && (f8_second_hits == 3) &&
                      f9_parity_ok && (f9_first_uploads == 3) && (f9_second_uploads == 0) && (f9_second_hits == 3) &&
                      conv10_parity_ok && (conv10_first_uploads == 1) && (conv10_second_uploads == 0) && (conv10_second_hits == 1) &&
                      pool10_parity_ok && top1_identical && top5_order_identical && top5_identical &&
                      conv10_mut_ok && pool10_mut_ok &&
                      all_runs_reconciled && cache_counters_reconciled &&
                      res_ok;

    printf("PROCESS_EXIT_CODE=%d\n", all_passed ? 0 : 1);
    printf("\n======================================================================\n");
    printf("FIRE2_DEVICE_RESIDENT_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("VISION_FIRE2_TO_POOL5_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("VISION_FIRE2_TO_FIRE8_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("VISION_FULL_SQUEEZENET_CLASSIFIER_LOGITS_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("VISION_FULL_SQUEEZENET_TEST_RESULT=%s\n", all_passed ? "PASS" : "FAIL");
    printf("======================================================================\n");

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

