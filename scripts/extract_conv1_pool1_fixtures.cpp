#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include <algorithm>

#include "net.h"
#include "mat.h"
#include "datareader.h"
#include "modelbin.h"
#include "layer/convolution.h"
#include "layer/pooling.h"

#define STB_IMAGE_IMPLEMENTATION
#include "/data/data/com.termux/files/home/ncnn/src/stb_image.h"

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
static void sha256_update(SHA256Context* ctx, const uint8_t* data, size_t len) {
    size_t i = 0;
    size_t index = (ctx->count >> 3) & 0x3f;
    ctx->count += (uint64_t)len << 3;
    size_t part_len = 64 - index;
    if (len >= part_len) {
        memcpy(&ctx->buffer[index], data, part_len);
        sha256_transform(ctx, ctx->buffer);
        for (i = part_len; i + 63 < len; i += 64) sha256_transform(ctx, &data[i]);
        index = 0;
    }
    memcpy(&ctx->buffer[index], &data[i], len - i);
}
static void sha256_final(SHA256Context* ctx, uint8_t digest[32]) {
    uint8_t bits[8];
    for (int i = 0; i < 8; ++i) bits[i] = (uint8_t)((ctx->count >> ((7 - i) * 8)) & 0xff);
    size_t index = (ctx->count >> 3) & 0x3f;
    size_t pad_len = (index < 56) ? (56 - index) : (120 - index);
    static const uint8_t padding[64] = { 0x80 };
    sha256_update(ctx, padding, pad_len);
    sha256_update(ctx, bits, 8);
    for (int i = 0; i < 8; ++i) {
        digest[i * 4]     = (uint8_t)((ctx->state[i] >> 24) & 0xff);
        digest[i * 4 + 1] = (uint8_t)((ctx->state[i] >> 16) & 0xff);
        digest[i * 4 + 2] = (uint8_t)((ctx->state[i] >> 8) & 0xff);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i] & 0xff);
    }
}
static std::string compute_sha256(const void* data, size_t len) {
    SHA256Context ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t*)data, len);
    uint8_t dig[32];
    sha256_final(&ctx, dig);
    char buf[65];
    for (int i = 0; i < 32; ++i) snprintf(buf + i * 2, 3, "%02x", dig[i]);
    return std::string(buf);
}

static void write_file_bytes(const char* path, const void* data, size_t len) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Failed to write %s\n", path); exit(1); }
    fwrite(data, 1, len, fp);
    fclose(fp);
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

int main(int argc, char** argv) {
    const char* param_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param";
    const char* bin_path = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin";
    const char* img_path = "/data/data/com.termux/files/home/cat_copilot_input.png";
    const char* out_dir = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures";

    FILE* fp_b = fopen(bin_path, "rb");
    if (!fp_b) { fprintf(stderr, "Cannot open %s\n", bin_path); return 1; }
    ncnn::DataReaderFromStdio dr(fp_b);
    ncnn::ModelBinFromDataReader mb(dr);

    // Load Conv1 weights & bias from ModelBin
    ncnn::Mat conv1_w = mb.load(1728, 0); // 64 * 3 * 3 * 3
    ncnn::Mat conv1_b = mb.load(64, 1);   // 64
    fclose(fp_b);

    char path[512];
    snprintf(path, sizeof(path), "%s/squeezenet_conv1_weight.bin", out_dir);
    write_file_bytes(path, conv1_w.data, conv1_w.total() * sizeof(float));
    printf("Wrote conv1 weight: %zu floats, SHA256=%s\n", conv1_w.total(), compute_sha256(conv1_w.data, conv1_w.total() * sizeof(float)).c_str());

    snprintf(path, sizeof(path), "%s/squeezenet_conv1_bias.bin", out_dir);
    write_file_bytes(path, conv1_b.data, conv1_b.total() * sizeof(float));
    printf("Wrote conv1 bias: %zu floats, SHA256=%s\n", conv1_b.total(), compute_sha256(conv1_b.data, conv1_b.total() * sizeof(float)).c_str());

    // Load Image
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* raw = stbi_load(img_path, &img_w, &img_h, &img_c, 3);
    if (!raw) { fprintf(stderr, "Failed to load %s\n", img_path); return 1; }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw, ncnn::Mat::PIXEL_RGB, img_w, img_h, 227, 227);
    stbi_image_free(raw);

    const float mean_vals[3] = { 104.f, 117.f, 123.f };
    in.substract_mean_normalize(mean_vals, 0);

    // Strict Net
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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);

    ncnn::Mat out_conv1, out_pool1;
    ex.extract("conv1_relu_conv1", out_conv1);
    ex.extract("pool1", out_pool1);

    std::vector<float> nhwc_conv1 = chw_to_nhwc(out_conv1);
    std::vector<float> nhwc_pool1 = chw_to_nhwc(out_pool1);

    snprintf(path, sizeof(path), "%s/squeezenet_conv1_golden.bin", out_dir);
    write_file_bytes(path, nhwc_conv1.data(), nhwc_conv1.size() * sizeof(float));
    printf("Wrote conv1 golden: %zu floats, SHA256=%s\n", nhwc_conv1.size(), compute_sha256(nhwc_conv1.data(), nhwc_conv1.size() * sizeof(float)).c_str());

    snprintf(path, sizeof(path), "%s/cat_copilot_pool1_golden_nhwc.bin", out_dir);
    write_file_bytes(path, nhwc_pool1.data(), nhwc_pool1.size() * sizeof(float));
    printf("Wrote pool1 golden: %zu floats, SHA256=%s\n", nhwc_pool1.size(), compute_sha256(nhwc_pool1.data(), nhwc_pool1.size() * sizeof(float)).c_str());

    // Preprocessed input
    std::vector<float> nhwc_in = chw_to_nhwc(in);
    snprintf(path, sizeof(path), "%s/cat_copilot_preprocessed_input_nhwc.bin", out_dir);
    write_file_bytes(path, nhwc_in.data(), nhwc_in.size() * sizeof(float));
    printf("Wrote preprocessed input: %zu floats, SHA256=%s\n", nhwc_in.size(), compute_sha256(nhwc_in.data(), nhwc_in.size() * sizeof(float)).c_str());

    return 0;
}
