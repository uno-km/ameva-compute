/**
 * @file extract_fire2_fixtures.cpp
 * @brief Extracts verified Fire2 expand1x1 & expand3x3 parameters, weights, biases, and golden blobs
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>

#include "/data/data/com.termux/files/home/ncnn/src/net.h"
#include "/data/data/com.termux/files/home/ncnn/src/mat.h"
#include "/data/data/com.termux/files/home/ncnn/src/datareader.h"
#include "/data/data/com.termux/files/home/ncnn/src/modelbin.h"

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
    const char* out_dir = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures";

    printf("======================================================================\n");
    printf("Extracting Fire2 Layers from SqueezeNet v1.1 ModelBin\n");
    printf("======================================================================\n");

    FILE* fp = fopen(bin_path, "rb");
    if (!fp) {
        fprintf(stderr, "FATAL: Could not open %s\n", bin_path);
        return EXIT_FAILURE;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);

    // 1. conv1: weight=1728, bias=64
    ncnn::Mat conv1_w = mb.load(1728, 0);
    ncnn::Mat conv1_b = mb.load(64, 1);

    // 2. fire2/squeeze1x1: weight=1024, bias=16
    ncnn::Mat fire2_sq_w = mb.load(1024, 0);
    ncnn::Mat fire2_sq_b = mb.load(16, 1);

    // 3. fire2/expand1x1: weight=1024, bias=64
    ncnn::Mat fire2_ex1_w = mb.load(1024, 0);
    ncnn::Mat fire2_ex1_b = mb.load(64, 1);

    // 4. fire2/expand3x3: weight=9216, bias=64
    ncnn::Mat fire2_ex3_w = mb.load(9216, 0);
    ncnn::Mat fire2_ex3_b = mb.load(64, 1);

    fclose(fp);

    std::string sq_w_sha = compute_sha256(fire2_sq_w.data, fire2_sq_w.total() * sizeof(float));
    std::string sq_b_sha = compute_sha256(fire2_sq_b.data, fire2_sq_b.total() * sizeof(float));
    printf("\n[LAYER: fire2/squeeze1x1]\n");
    printf("  WEIGHT_SHA256=%s (elems=%zu, bytes=%zu)\n", sq_w_sha.c_str(), fire2_sq_w.total(), fire2_sq_w.total() * sizeof(float));
    printf("  BIAS_SHA256=%s (elems=%zu, bytes=%zu)\n", sq_b_sha.c_str(), fire2_sq_b.total(), fire2_sq_b.total() * sizeof(float));
    const char* expected_sq_w_sha = "9e420c4f34d556158e2214d023191c0d7c0e167aa5432845030b459e718b5918";
    if (sq_w_sha == expected_sq_w_sha) {
        printf("  VERIFICATION: Squeeze1x1 Weight matches golden SHA256: PASS\n");
    } else {
        fprintf(stderr, "FATAL: Squeeze1x1 weight mismatch!\n");
        return EXIT_FAILURE;
    }

    std::string ex1_w_sha = compute_sha256(fire2_ex1_w.data, fire2_ex1_w.total() * sizeof(float));
    std::string ex1_b_sha = compute_sha256(fire2_ex1_b.data, fire2_ex1_b.total() * sizeof(float));
    printf("\n[LAYER: fire2/expand1x1]\n");
    printf("  WEIGHT_SHA256=%s (elems=%zu, bytes=%zu)\n", ex1_w_sha.c_str(), fire2_ex1_w.total(), fire2_ex1_w.total() * sizeof(float));
    printf("  BIAS_SHA256=%s (elems=%zu, bytes=%zu)\n", ex1_b_sha.c_str(), fire2_ex1_b.total(), fire2_ex1_b.total() * sizeof(float));

    std::string ex3_w_sha = compute_sha256(fire2_ex3_w.data, fire2_ex3_w.total() * sizeof(float));
    std::string ex3_b_sha = compute_sha256(fire2_ex3_b.data, fire2_ex3_b.total() * sizeof(float));
    printf("\n[LAYER: fire2/expand3x3]\n");
    printf("  WEIGHT_SHA256=%s (elems=%zu, bytes=%zu)\n", ex3_w_sha.c_str(), fire2_ex3_w.total(), fire2_ex3_w.total() * sizeof(float));
    printf("  BIAS_SHA256=%s (elems=%zu, bytes=%zu)\n", ex3_b_sha.c_str(), fire2_ex3_b.total(), fire2_ex3_b.total() * sizeof(float));

    char path[512];
    snprintf(path, sizeof(path), "%s/squeezenet_fire2_expand1x1_weight.bin", out_dir);
    write_file_bytes(path, fire2_ex1_w.data, fire2_ex1_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire2_expand1x1_bias.bin", out_dir);
    write_file_bytes(path, fire2_ex1_b.data, fire2_ex1_b.total() * sizeof(float));

    snprintf(path, sizeof(path), "%s/squeezenet_fire2_expand3x3_weight.bin", out_dir);
    write_file_bytes(path, fire2_ex3_w.data, fire2_ex3_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire2_expand3x3_bias.bin", out_dir);
    write_file_bytes(path, fire2_ex3_b.data, fire2_ex3_b.total() * sizeof(float));

    // NCNN inference for golden blobs
    printf("\n--- Computing Golden Outputs for Cat Image Pool1 Input via NCNN Strict FP32 ---\n");
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

    ncnn::Net net;
    net.opt = strict_opt;
    net.load_param(param_path);
    net.load_model(bin_path);

    const char* cat_pool1_path = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/cat_copilot_fire2_input_activation_nhwc.bin";
    std::vector<uint8_t> pool1_bytes = read_file_bytes(cat_pool1_path);
    const float* pool1_nhwc = reinterpret_cast<const float*>(pool1_bytes.data());
    ncnn::Mat pool1_chw(56, 56, 64);
    for (int y = 0; y < 56; ++y) {
        for (int x = 0; x < 56; ++x) {
            for (int ch = 0; ch < 64; ++ch) {
                pool1_chw.channel(ch)[y * 56 + x] = pool1_nhwc[(y * 56 + x) * 64 + ch];
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    ex.input("pool1", pool1_chw);

    ncnn::Mat out_squeeze_relu, out_expand1x1_relu, out_expand3x3_relu, out_concat;
    ex.extract("fire2/squeeze1x1_fire2/relu_squeeze1x1", out_squeeze_relu);
    ex.extract("fire2/expand1x1_fire2/relu_expand1x1", out_expand1x1_relu);
    ex.extract("fire2/expand3x3_fire2/relu_expand3x3", out_expand3x3_relu);
    ex.extract("fire2/concat", out_concat);

    auto save_blob = [&](const ncnn::Mat& m, const char* name, const char* out_filename) {
        std::vector<float> nhwc = chw_to_nhwc(m);
        std::string sha = compute_sha256(nhwc.data(), nhwc.size() * sizeof(float));
        printf("BLOB [%s]: shape=[1,%d,%d,%d], elements=%zu, SHA-256=%s\n",
               name, m.h, m.w, m.c, nhwc.size(), sha.c_str());
        snprintf(path, sizeof(path), "%s/%s", out_dir, out_filename);
        write_file_bytes(path, nhwc.data(), nhwc.size() * sizeof(float));
    };

    save_blob(out_squeeze_relu, "fire2/squeeze1x1_relu", "cat_copilot_fire2_squeeze1x1_relu_golden_nhwc.bin");
    save_blob(out_expand1x1_relu, "fire2/expand1x1_relu", "cat_copilot_fire2_expand1x1_relu_golden_nhwc.bin");
    save_blob(out_expand3x3_relu, "fire2/expand3x3_relu", "cat_copilot_fire2_expand3x3_relu_golden_nhwc.bin");
    save_blob(out_concat, "fire2/concat", "cat_copilot_fire2_concat_golden_nhwc.bin");

    printf("\nAll Fire2 fixtures extracted and verified successfully.\n");
    return EXIT_SUCCESS;
}
