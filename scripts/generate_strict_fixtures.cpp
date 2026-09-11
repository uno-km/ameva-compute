/**
 * @file generate_strict_fixtures.cpp
 * @brief Generates verified Strict FP32 reference binaries for test_shapes and cat images
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "/data/data/com.termux/files/home/ncnn/src/stb_image.h"

#include "/data/data/com.termux/files/home/ncnn/src/net.h"
#include "/data/data/com.termux/files/home/ncnn/src/mat.h"

// Portable SHA-256
struct SHA256Context { uint32_t state[8]; uint64_t count; uint8_t buffer[64]; };
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
    for (int i = 16; i < 64; ++i) m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K256[i] + m[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
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
        p += take; len -= take; buffer_idx += take;
        if (buffer_idx == 64) { sha256_transform(ctx, ctx->buffer); buffer_idx = 0; }
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
    for (int i = 0; i < 8; ++i) ctx->buffer[56 + i] = (uint8_t)((total_bits >> (56 - i * 8)) & 0xFF);
    sha256_transform(ctx, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t)((ctx->state[i] >> 24) & 0xFF);
        hash[i * 4 + 1] = (uint8_t)((ctx->state[i] >> 16) & 0xFF);
        hash[i * 4 + 2] = (uint8_t)((ctx->state[i] >> 8) & 0xFF);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i] & 0xFF);
    }
}
static std::string compute_sha256(const void* data, size_t len) {
    SHA256Context ctx; sha256_init(&ctx); sha256_update(&ctx, data, len);
    uint8_t h[32]; sha256_final(&ctx, h);
    char buf[65]; for (int i = 0; i < 32; ++i) sprintf(buf + i * 2, "%02x", h[i]);
    buf[64] = 0; return std::string(buf);
}

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

static bool write_file_bytes(const char* path, const void* data, size_t len) {
    if (!path || !data || len == 0) return false;
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "FATAL: Failed to open %s for write\n", path);
        return false;
    }
    size_t written = fwrite(data, 1, len, fp);
    if (written != len || ferror(fp)) {
        fprintf(stderr, "FATAL: Failed to write %s (wrote %zu / %zu)\n", path, written, len);
        fclose(fp);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "FATAL: Failed to close %s after write\n", path);
        return false;
    }
    printf("Wrote %s (%zu bytes)\n", path, len);
    return true;
}

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

int main() {
    const char* param_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param";
    const char* bin_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin";
    const char* fixture_act_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_input_activation_nhwc.bin";
    const char* fixture_weight_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_weight.bin";
    const char* fixture_bias_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_bias.bin";

    std::vector<uint8_t> fix_act = read_file_bytes(fixture_act_path);
    if (fix_act.empty()) {
        fprintf(stderr, "FATAL: Could not read %s\n", fixture_act_path);
        return EXIT_FAILURE;
    }

    ncnn::Option opt_fp32;
    opt_fp32.num_threads = 1;
    opt_fp32.use_vulkan_compute = false;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_storage = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_int8_inference = false;
    opt_fp32.use_int8_packed = false;
    opt_fp32.use_int8_storage = false;
    opt_fp32.use_int8_arithmetic = false;
    opt_fp32.use_packing_layout = false;
    opt_fp32.use_sgemm_convolution = false;
    opt_fp32.use_winograd_convolution = false;

    // 1. Generate Strict FP32 output for test_shapes input activation
    const int H = 56, W = 56, Cin = 64;
    ncnn::Mat pool1_isolated(W, H, Cin);
    const float* fix_act_ptr = (const float*)fix_act.data();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int ch = 0; ch < Cin; ++ch) {
                pool1_isolated.channel(ch)[y * W + x] = fix_act_ptr[(y * W + x) * Cin + ch];
            }
        }
    }

    // Synthesize 1-layer NCNN model binary from verified weight & bias fixtures (P0-4)
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

    ncnn::Net one_layer_net;
    one_layer_net.opt = opt_fp32;
    const char* p_str =
        "7767517\n"
        "2 2\n"
        "Input pool1 0 1 pool1\n"
        "Convolution fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024\n";
    if (one_layer_net.load_param_mem(p_str) != 0) {
        fprintf(stderr, "FATAL: one_layer_net.load_param_mem failed\n");
        return EXIT_FAILURE;
    }
    size_t consumed = one_layer_net.load_model(synth_model.data());
    if (consumed != synth_model.size()) {
        fprintf(stderr, "FATAL: one_layer_net.load_model failed, consumed=%zu\n", consumed);
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex_1l = one_layer_net.create_extractor();
    if (ex_1l.input("pool1", pool1_isolated) != 0) {
        fprintf(stderr, "FATAL: ex_1l.input('pool1') failed\n");
        return EXIT_FAILURE;
    }
    ncnn::Mat top_blob;
    if (ex_1l.extract("fire2/squeeze1x1", top_blob) != 0 || top_blob.empty()) {
        fprintf(stderr, "FATAL: ex_1l.extract('fire2/squeeze1x1') failed\n");
        return EXIT_FAILURE;
    }
    std::vector<float> strict_f32_out = chw_to_nhwc(top_blob);
    std::string strict_f32_sha = compute_sha256(strict_f32_out.data(), strict_f32_out.size() * sizeof(float));

    printf("STRICT_FP32_OUTPUT_SHA256=%s\n", strict_f32_sha.c_str());
    if (!write_file_bytes("/data/data/com.termux/files/home/squeezenet_fire2_ncnn_strict_fp32_output_nhwc.bin",
                          strict_f32_out.data(), strict_f32_out.size() * sizeof(float))) {
        return EXIT_FAILURE;
    }

    // 2. Generate Cat Image fixtures under Strict FP32
    const char* cat_img_path = "/data/data/com.termux/files/home/cat_copilot_input.png";
    int cat_w, cat_h, cat_c;
    unsigned char* cat_raw = stbi_load(cat_img_path, &cat_w, &cat_h, &cat_c, 3);
    if (!cat_raw) {
        fprintf(stderr, "FATAL: Failed to load cat image %s\n", cat_img_path);
        return EXIT_FAILURE;
    }
    ncnn::Mat cat_mat = ncnn::Mat::from_pixels_resize(cat_raw, ncnn::Mat::PIXEL_RGB, cat_w, cat_h, 227, 227);
    stbi_image_free(cat_raw);
    const float mean_vals[3] = { 104.f, 117.f, 123.f };
    cat_mat.substract_mean_normalize(mean_vals, 0);

    ncnn::Net cat_net;
    cat_net.opt = opt_fp32;
    if (cat_net.load_param(param_path) != 0) {
        fprintf(stderr, "FATAL: cat_net.load_param failed\n");
        return EXIT_FAILURE;
    }
    if (cat_net.load_model(bin_path) != 0) {
        fprintf(stderr, "FATAL: cat_net.load_model failed\n");
        return EXIT_FAILURE;
    }

    ncnn::Extractor cat_ex = cat_net.create_extractor();
    if (cat_ex.input("data", cat_mat) != 0) {
        fprintf(stderr, "FATAL: cat_ex.input('data') failed\n");
        return EXIT_FAILURE;
    }

    ncnn::Mat cat_pool1;
    if (cat_ex.extract("pool1", cat_pool1) != 0 || cat_pool1.empty()) {
        fprintf(stderr, "FATAL: cat_ex.extract('pool1') failed\n");
        return EXIT_FAILURE;
    }
    std::vector<float> cat_pool1_nhwc = chw_to_nhwc(cat_pool1);
    std::string cat_pool1_sha = compute_sha256(cat_pool1_nhwc.data(), cat_pool1_nhwc.size() * sizeof(float));
    printf("CAT_POOL1_SHA256=%s\n", cat_pool1_sha.c_str());
    if (!write_file_bytes("/data/data/com.termux/files/home/cat_copilot_fire2_input_activation_nhwc.bin",
                          cat_pool1_nhwc.data(), cat_pool1_nhwc.size() * sizeof(float))) {
        return EXIT_FAILURE;
    }

    ncnn::Mat cat_fire2;
    if (cat_ex.extract("fire2/squeeze1x1", cat_fire2) != 0 || cat_fire2.empty()) {
        fprintf(stderr, "FATAL: cat_ex.extract('fire2/squeeze1x1') failed\n");
        return EXIT_FAILURE;
    }
    std::vector<float> cat_fire2_nhwc = chw_to_nhwc(cat_fire2);
    std::string cat_fire2_sha = compute_sha256(cat_fire2_nhwc.data(), cat_fire2_nhwc.size() * sizeof(float));
    printf("CAT_FIRE2_STRICT_FP32_SHA256=%s\n", cat_fire2_sha.c_str());
    if (!write_file_bytes("/data/data/com.termux/files/home/cat_copilot_fire2_ncnn_strict_fp32_output_nhwc.bin",
                          cat_fire2_nhwc.data(), cat_fire2_nhwc.size() * sizeof(float))) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
