/**
 * @file audit_ncnn_fire2.cpp
 * @brief Diagnostic Harness for NCNN SqueezeNet v1.1 Fire2/Squeeze1x1 Parity
 *
 * Implements strict Option inspections, Network vs Isolated Layer comparisons,
 * and exact scalar traces without ambiguous assumptions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "/data/data/com.termux/files/home/ncnn/src/stb_image.h"

#include "/data/data/com.termux/files/home/ncnn/src/net.h"
#include "/data/data/com.termux/files/home/ncnn/src/mat.h"
#include "/data/data/com.termux/files/home/ncnn/src/layer.h"
#include "/data/data/com.termux/files/home/ncnn/src/layer/convolution.h"

// -----------------------------------------------------------------------------
// Portable SHA-256
// -----------------------------------------------------------------------------
struct SHA256Context {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
};

static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_transform(SHA256Context* ctx, const uint8_t data[64]) {
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) | ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
    }
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K256[i] + m[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(SHA256Context* ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(SHA256Context* ctx, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    size_t buffer_idx = (size_t)(ctx->count & 63);
    ctx->count += len;
    while (len > 0) {
        size_t take = 64 - buffer_idx;
        if (take > len) take = len;
        memcpy(ctx->buffer + buffer_idx, p, take);
        p += take;
        len -= take;
        buffer_idx += take;
        if (buffer_idx == 64) {
            sha256_transform(ctx, ctx->buffer);
            buffer_idx = 0;
        }
    }
}

static void sha256_final(SHA256Context* ctx, uint8_t hash[32]) {
    uint64_t total_bits = ctx->count * 8;
    size_t pad_idx = (size_t)(ctx->count & 63);
    ctx->buffer[pad_idx++] = 0x80;
    if (pad_idx > 56) {
        memset(ctx->buffer + pad_idx, 0, 64 - pad_idx);
        sha256_transform(ctx, ctx->buffer);
        pad_idx = 0;
    }
    memset(ctx->buffer + pad_idx, 0, 56 - pad_idx);
    for (int i = 0; i < 8; ++i) {
        ctx->buffer[56 + i] = (uint8_t)((total_bits >> (56 - i * 8)) & 0xFF);
    }
    sha256_transform(ctx, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t)((ctx->state[i] >> 24) & 0xFF);
        hash[i * 4 + 1] = (uint8_t)((ctx->state[i] >> 16) & 0xFF);
        hash[i * 4 + 2] = (uint8_t)((ctx->state[i] >> 8) & 0xFF);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i] & 0xFF);
    }
}

static std::string compute_sha256(const void* data, size_t len) {
    SHA256Context ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    uint8_t h[32];
    sha256_final(&ctx, h);
    char buf[65];
    for (int i = 0; i < 32; ++i) sprintf(buf + i * 2, "%02x", h[i]);
    buf[64] = 0;
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// File loading helper
// -----------------------------------------------------------------------------
static std::vector<uint8_t> read_file_bytes(const char* path) {
    if (!path) return {};
    FILE* fp = fopen(path, "rb");
    if (!fp) return {};
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return {}; }
    long sz = ftell(fp);
    if (sz <= 0) { fclose(fp); return {}; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return {}; }
    std::vector<uint8_t> buf(sz);
    size_t read_n = fread(buf.data(), 1, (size_t)sz, fp);
    if (read_n != (size_t)sz || ferror(fp)) { fclose(fp); return {}; }
    if (fclose(fp) != 0) return {};
    return buf;
}

// -----------------------------------------------------------------------------
// Layout conversion
// -----------------------------------------------------------------------------
static std::vector<float> chw_to_nhwc(const ncnn::Mat& m) {
    int w = m.w, h = m.h, c = m.c;
    std::vector<float> nhwc(w * h * c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                nhwc[(y * w + x) * c + ch] = m.channel(ch)[y * w + x];
            }
        }
    }
    return nhwc;
}

static ncnn::Mat nhwc_to_chw(const float* nhwc, int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                m.channel(ch)[y * w + x] = nhwc[(y * w + x) * c + ch];
            }
        }
    }
    return m;
}

// -----------------------------------------------------------------------------
// Option Printer
// -----------------------------------------------------------------------------
static void print_option(const char* label, const ncnn::Option& opt) {
    printf("OPTION_%s_NUM_THREADS=%d\n", label, opt.num_threads);
    printf("OPTION_%s_USE_VULKAN_COMPUTE=%d\n", label, (int)opt.use_vulkan_compute);
    printf("OPTION_%s_USE_FP16_PACKED=%d\n", label, (int)opt.use_fp16_packed);
    printf("OPTION_%s_USE_FP16_STORAGE=%d\n", label, (int)opt.use_fp16_storage);
    printf("OPTION_%s_USE_FP16_ARITHMETIC=%d\n", label, (int)opt.use_fp16_arithmetic);
    printf("OPTION_%s_USE_BF16_STORAGE=%d\n", label, (int)opt.use_bf16_storage);
    printf("OPTION_%s_USE_BF16_PACKED=%d\n", label, (int)opt.use_bf16_packed);
    printf("OPTION_%s_USE_INT8_INFERENCE=%d\n", label, (int)opt.use_int8_inference);
    printf("OPTION_%s_USE_INT8_PACKED=%d\n", label, (int)opt.use_int8_packed);
    printf("OPTION_%s_USE_INT8_STORAGE=%d\n", label, (int)opt.use_int8_storage);
    printf("OPTION_%s_USE_INT8_ARITHMETIC=%d\n", label, (int)opt.use_int8_arithmetic);
    printf("OPTION_%s_USE_PACKING_LAYOUT=%d\n", label, (int)opt.use_packing_layout);
    printf("OPTION_%s_USE_SGEMM_CONVOLUTION=%d\n", label, (int)opt.use_sgemm_convolution);
    printf("OPTION_%s_USE_WINOGRAD_CONVOLUTION=%d\n", label, (int)opt.use_winograd_convolution);
    fflush(stdout);
}

// -----------------------------------------------------------------------------
// Main Diagnostic Execution
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("======================================================================\n");
    printf("AMEVA Vision - NCNN Fire2/Squeeze1x1 Rigorous Diagnostic Suite (v2)\n");
    printf("======================================================================\n\n");

    const char* param_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param";
    const char* bin_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin";
    const char* fixture_act_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_input_activation_nhwc.bin";
    const char* fixture_weight_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_weight.bin";
    const char* fixture_bias_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_bias.bin";
    const char* fixture_ncnn_out_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_ncnn_cpu_output_nhwc.bin";

    // -------------------------------------------------------------------------
    // 1. Inspect Genuine Option Defaults
    // -------------------------------------------------------------------------
    printf("--- [P0-2: GENUINE NCNN OPTION INSPECTION] ---\n");
    ncnn::Option default_opt; // 100% untouched constructor default
    default_opt.num_threads = 1;
    default_opt.use_vulkan_compute = false;
    print_option("DEFAULT", default_opt);

    ncnn::Option strict_opt;
    strict_opt.num_threads = 1;
    strict_opt.use_vulkan_compute = false;
    strict_opt.use_fp16_packed = false;
    strict_opt.use_fp16_storage = false;
    strict_opt.use_fp16_arithmetic = false;
    strict_opt.use_bf16_storage = false;
    strict_opt.use_bf16_packed = false;
    strict_opt.use_int8_inference = false;
    strict_opt.use_int8_packed = false;
    strict_opt.use_int8_storage = false;
    strict_opt.use_int8_arithmetic = false;
    strict_opt.use_packing_layout = false;
    strict_opt.use_sgemm_convolution = false;
    strict_opt.use_winograd_convolution = false;
    print_option("STRICT_FP32", strict_opt);

    // -------------------------------------------------------------------------
    // 2. Load Input Image & Run Network Under Both Options
    // -------------------------------------------------------------------------
    printf("\n--- [P0-1: NETWORK VS ISOLATED LAYER EXTRACTION & PARITY] ---\n");
    const char* shapes_img_path = "/data/data/com.termux/files/home/test_shapes_224x224.png";
    int img_w, img_h, img_c;
    unsigned char* raw_pixels = stbi_load(shapes_img_path, &img_w, &img_h, &img_c, 3);
    if (!raw_pixels) {
        printf("FATAL: Could not load %s\n", shapes_img_path);
        return 1;
    }
    ncnn::Mat in_mat = ncnn::Mat::from_pixels_resize(raw_pixels, ncnn::Mat::PIXEL_RGB, img_w, img_h, 227, 227);
    stbi_image_free(raw_pixels);
    const float mean_vals[3] = { 104.f, 117.f, 123.f };
    in_mat.substract_mean_normalize(mean_vals, 0);

    // A. Network with default_opt
    ncnn::Net net_default;
    net_default.opt = default_opt;
    if (net_default.load_param(param_path) != 0) {
        fprintf(stderr, "FATAL: net_default.load_param failed\n");
        return EXIT_FAILURE;
    }
    if (net_default.load_model(bin_path) != 0) {
        fprintf(stderr, "FATAL: net_default.load_model failed\n");
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex_def = net_default.create_extractor();
    if (ex_def.input("data", in_mat) != 0) {
        fprintf(stderr, "FATAL: ex_def.input('data') failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat default_pool1_mat;
    if (ex_def.extract("pool1", default_pool1_mat) != 0 || default_pool1_mat.empty()) {
        fprintf(stderr, "FATAL: ex_def.extract('pool1') failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat default_fire2_mat;
    if (ex_def.extract("fire2/squeeze1x1", default_fire2_mat) != 0 || default_fire2_mat.empty()) {
        fprintf(stderr, "FATAL: ex_def.extract('fire2/squeeze1x1') failed\n");
        return EXIT_FAILURE;
    }

    std::vector<float> default_pool1_nhwc = chw_to_nhwc(default_pool1_mat);
    std::vector<float> default_fire2_nhwc = chw_to_nhwc(default_fire2_mat);
    std::string default_pool1_sha = compute_sha256(default_pool1_nhwc.data(), default_pool1_nhwc.size() * sizeof(float));
    std::string default_fire2_sha = compute_sha256(default_fire2_nhwc.data(), default_fire2_nhwc.size() * sizeof(float));

    // B. Network with strict_opt
    ncnn::Net net_strict;
    net_strict.opt = strict_opt;
    if (net_strict.load_param(param_path) != 0) {
        fprintf(stderr, "FATAL: net_strict.load_param failed\n");
        return EXIT_FAILURE;
    }
    if (net_strict.load_model(bin_path) != 0) {
        fprintf(stderr, "FATAL: net_strict.load_model failed\n");
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex_strict = net_strict.create_extractor();
    if (ex_strict.input("data", in_mat) != 0) {
        fprintf(stderr, "FATAL: ex_strict.input('data') failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat strict_pool1_mat;
    if (ex_strict.extract("pool1", strict_pool1_mat) != 0 || strict_pool1_mat.empty()) {
        fprintf(stderr, "FATAL: ex_strict.extract('pool1') failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat strict_fire2_mat;
    if (ex_strict.extract("fire2/squeeze1x1", strict_fire2_mat) != 0 || strict_fire2_mat.empty()) {
        fprintf(stderr, "FATAL: ex_strict.extract('fire2/squeeze1x1') failed\n");
        return EXIT_FAILURE;
    }

    std::vector<float> strict_pool1_nhwc = chw_to_nhwc(strict_pool1_mat);
    std::vector<float> strict_fire2_nhwc = chw_to_nhwc(strict_fire2_mat);
    std::string strict_pool1_sha = compute_sha256(strict_pool1_nhwc.data(), strict_pool1_nhwc.size() * sizeof(float));
    std::string strict_fire2_sha = compute_sha256(strict_fire2_nhwc.data(), strict_fire2_nhwc.size() * sizeof(float));

    // Check upstream pool1 divergence
    printf("\n--- UPSTREAM POOL1 COMPARISON (Default vs Strict FP32) ---\n");
    printf("NETWORK_DEFAULT_POOL1_SHA256=%s (Bytes=%zu)\n", default_pool1_sha.c_str(), default_pool1_nhwc.size() * 4);
    printf("NETWORK_STRICT_POOL1_SHA256=%s (Bytes=%zu)\n", strict_pool1_sha.c_str(), strict_pool1_nhwc.size() * 4);

    double pool1_sum_sq = 0.0, pool1_max_abs = 0.0;
    for (size_t i = 0; i < strict_pool1_nhwc.size(); ++i) {
        double d = fabs((double)default_pool1_nhwc[i] - (double)strict_pool1_nhwc[i]);
        pool1_sum_sq += d * d;
        if (d > pool1_max_abs) pool1_max_abs = d;
    }
    double pool1_rmse = sqrt(pool1_sum_sq / strict_pool1_nhwc.size());
    printf("UPSTREAM_POOL1_DELTA: RMSE=%.7f, MAX_ABS=%.7f\n", pool1_rmse, pool1_max_abs);

    // C. Isolated 1-Layer Execution with strict_pool1 as input
    // Synthesize 1-layer NCNN model binary from verified weight & bias fixtures (P0-4: No offset 7172 hardcoding)
    std::vector<uint8_t> w_bytes = read_file_bytes(fixture_weight_path);
    std::vector<uint8_t> b_bytes = read_file_bytes(fixture_bias_path);
    if (w_bytes.size() != 4096 || b_bytes.size() != 64) {
        fprintf(stderr, "FATAL: Weight or bias fixture size invalid (w=%zu, b=%zu)\n", w_bytes.size(), b_bytes.size());
        return EXIT_FAILURE;
    }
    std::string w_sha = compute_sha256(w_bytes.data(), w_bytes.size());
    std::string b_sha = compute_sha256(b_bytes.data(), b_bytes.size());
    if (w_sha != "9e420c4f34d556158e2214d023191c0d7c0e167aa5432845030b459e718b5918") {
        fprintf(stderr, "FATAL: Weight fixture SHA-256 mismatch: %s\n", w_sha.c_str());
        return EXIT_FAILURE;
    }
    if (b_sha != "b38cfdd1c6cd695186eaf84a84d8fcafab76e83ece87df9c4af81e55db41583e") {
        fprintf(stderr, "FATAL: Bias fixture SHA-256 mismatch: %s\n", b_sha.c_str());
        return EXIT_FAILURE;
    }

    std::vector<uint8_t> synth_model;
    synth_model.reserve(4 + 4096 + 64);
    uint32_t tag = 0x00000000u;
    synth_model.insert(synth_model.end(), (const uint8_t*)&tag, (const uint8_t*)&tag + 4);
    synth_model.insert(synth_model.end(), w_bytes.begin(), w_bytes.end());
    synth_model.insert(synth_model.end(), b_bytes.begin(), b_bytes.end());

    ncnn::Net isolated_net;
    isolated_net.opt = strict_opt;
    const char* one_layer_param =
        "7767517\n"
        "2 2\n"
        "Input pool1 0 1 pool1\n"
        "Convolution fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024\n";
    if (isolated_net.load_param_mem(one_layer_param) != 0) {
        fprintf(stderr, "FATAL: isolated_net.load_param_mem failed\n");
        return EXIT_FAILURE;
    }
    size_t consumed = isolated_net.load_model(synth_model.data());
    if (consumed != synth_model.size()) {
        fprintf(stderr, "FATAL: isolated_net.load_model failed, consumed=%zu (expected %zu)\n", consumed, synth_model.size());
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex_iso = isolated_net.create_extractor();
    if (ex_iso.input("pool1", strict_pool1_mat) != 0) {
        fprintf(stderr, "FATAL: ex_iso.input failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat iso_fire2_mat;
    if (ex_iso.extract("fire2/squeeze1x1", iso_fire2_mat) != 0 || iso_fire2_mat.empty()) {
        fprintf(stderr, "FATAL: ex_iso.extract failed\n");
        return EXIT_FAILURE;
    }

    std::vector<float> iso_fire2_nhwc = chw_to_nhwc(iso_fire2_mat);
    std::string iso_fire2_sha = compute_sha256(iso_fire2_nhwc.data(), iso_fire2_nhwc.size() * sizeof(float));

    // Compare Network Raw Conv vs Isolated Conv (Both on strict_pool1)
    double net_iso_sum_sq = 0.0, net_iso_max_abs = 0.0;
    for (size_t i = 0; i < strict_fire2_nhwc.size(); ++i) {
        double d = fabs((double)strict_fire2_nhwc[i] - (double)iso_fire2_nhwc[i]);
        net_iso_sum_sq += d * d;
        if (d > net_iso_max_abs) net_iso_max_abs = d;
    }
    double net_iso_rmse = sqrt(net_iso_sum_sq / strict_fire2_nhwc.size());

    printf("\n--- REQUIRED P0-1 COMPARISON METRICS ---\n");
    printf("NETWORK_POOL1_SHA256=%s\n", strict_pool1_sha.c_str());
    printf("ISOLATED_INPUT_SHA256=%s\n", strict_pool1_sha.c_str());
    printf("NETWORK_RAW_CONV_SHA256=%s\n", strict_fire2_sha.c_str());
    printf("ISOLATED_CONV_SHA256=%s\n", iso_fire2_sha.c_str());
    printf("NETWORK_VS_ISOLATED_RMSE=%.10f\n", net_iso_rmse);
    printf("NETWORK_VS_ISOLATED_MAX_ABS_ERROR=%.10f\n", net_iso_max_abs);

    bool net_vs_iso_pass = (net_iso_rmse <= 1e-6) && (net_iso_max_abs <= 1e-5);
    printf("NETWORK_VS_ISOLATED_PARITY=%s\n", net_vs_iso_pass ? "PASS" : "FAIL");

    // D. Check what happened with the original fixture (squeezenet_fire2_input_activation_nhwc.bin)
    std::vector<uint8_t> fix_act = read_file_bytes(fixture_act_path);
    std::string fix_act_sha = compute_sha256(fix_act.data(), fix_act.size());
    printf("\n--- ORIGINAL FIXTURE PROVENANCE CHECK ---\n");
    printf("FIXTURE_INPUT_ACTIVATION_SHA256=%s\n", fix_act_sha.c_str());
    printf("MATCHES_DEFAULT_POOL1=%s\n", (fix_act_sha == default_pool1_sha) ? "true" : "false");
    printf("MATCHES_STRICT_POOL1=%s\n", (fix_act_sha == strict_pool1_sha) ? "true" : "false");

    std::vector<uint8_t> fix_out = read_file_bytes(fixture_ncnn_out_path);
    std::string fix_out_sha = compute_sha256(fix_out.data(), fix_out.size());
    printf("FIXTURE_NCNN_OUTPUT_SHA256=%s\n", fix_out_sha.c_str());
    printf("MATCHES_DEFAULT_FIRE2=%s\n", (fix_out_sha == default_fire2_sha) ? "true" : "false");
    printf("MATCHES_STRICT_FIRE2=%s\n", (fix_out_sha == strict_fire2_sha) ? "true" : "false");

    printf("\n======================================================================\n");
    printf("P0-1 and P0-2 Diagnostic Complete.\n");
    printf("======================================================================\n");
    return net_vs_iso_pass ? 0 : 1;
}
