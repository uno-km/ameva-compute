/**
 * @file extract_fire6_7_8_fixtures.cpp
 * @brief Extracts SqueezeNet v1.1 Fire6, Fire7, Fire8 weights/biases and golden blobs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>

#include "net.h"
#include "mat.h"
#include "datareader.h"
#include "modelbin.h"
#include "layer/pooling.h"

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
    size_t idx = (ctx->count >> 3) & 0x3f;
    ctx->count += (uint64_t)len << 3;
    size_t part = 64 - idx;
    if (len >= part) {
        memcpy(&ctx->buffer[idx], data, part);
        sha256_transform(ctx, ctx->buffer);
        for (i = part; i + 63 < len; i += 64) sha256_transform(ctx, &data[i]);
        idx = 0;
    }
    memcpy(&ctx->buffer[idx], &data[i], len - i);
}
static void sha256_final(SHA256Context* ctx, uint8_t digest[32]) {
    static const uint8_t pad[64] = { 0x80 };
    uint8_t count_bytes[8];
    for (int i = 0; i < 8; ++i) count_bytes[i] = (uint8_t)(ctx->count >> ((7 - i) * 8));
    size_t idx = (ctx->count >> 3) & 0x3f;
    size_t padlen = (idx < 56) ? (56 - idx) : (120 - idx);
    sha256_update(ctx, pad, padlen);
    sha256_update(ctx, count_bytes, 8);
    for (int i = 0; i < 8; ++i) {
        digest[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

static std::string compute_sha256(const void* data, size_t len) {
    SHA256Context ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const uint8_t*>(data), len);
    uint8_t digest[32];
    sha256_final(&ctx, digest);
    char buf[65];
    for (int i = 0; i < 32; ++i) snprintf(buf + i * 2, 3, "%02x", digest[i]);
    buf[64] = '\0';
    return std::string(buf);
}

static std::vector<uint8_t> read_file_bytes(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return {};
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, fp);
    fclose(fp);
    return buf;
}

static bool write_file_bytes(const char* path, const void* data, size_t len) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return false;
    size_t written = fwrite(data, 1, len, fp);
    fclose(fp);
    return written == len;
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

    FILE* fp = fopen(bin_path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open bin: %s\n", bin_path);
        return EXIT_FAILURE;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);

    // 1. conv1
    mb.load(1728, 0); mb.load(64, 1);
    // 2. fire2
    mb.load(1024, 0); mb.load(16, 1);
    mb.load(1024, 0); mb.load(64, 1);
    mb.load(9216, 0); mb.load(64, 1);
    // 3. fire3
    mb.load(2048, 0); mb.load(16, 1);
    mb.load(1024, 0); mb.load(64, 1);
    mb.load(9216, 0); mb.load(64, 1);
    // 4. fire4
    mb.load(4096, 0); mb.load(32, 1);
    mb.load(4096, 0); mb.load(128, 1);
    mb.load(36864, 0); mb.load(128, 1);
    // 5. fire5
    mb.load(8192, 0); mb.load(32, 1);
    mb.load(4096, 0); mb.load(128, 1);
    mb.load(36864, 0); mb.load(128, 1);

    // 6. fire6
    ncnn::Mat fire6_sq_w = mb.load(12288, 0);
    ncnn::Mat fire6_sq_b = mb.load(48, 1);
    ncnn::Mat fire6_ex1_w = mb.load(9216, 0);
    ncnn::Mat fire6_ex1_b = mb.load(192, 1);
    ncnn::Mat fire6_ex3_w = mb.load(82944, 0);
    ncnn::Mat fire6_ex3_b = mb.load(192, 1);

    // 7. fire7
    ncnn::Mat fire7_sq_w = mb.load(18432, 0);
    ncnn::Mat fire7_sq_b = mb.load(48, 1);
    ncnn::Mat fire7_ex1_w = mb.load(9216, 0);
    ncnn::Mat fire7_ex1_b = mb.load(192, 1);
    ncnn::Mat fire7_ex3_w = mb.load(82944, 0);
    ncnn::Mat fire7_ex3_b = mb.load(192, 1);

    // 8. fire8
    ncnn::Mat fire8_sq_w = mb.load(24576, 0);
    ncnn::Mat fire8_sq_b = mb.load(64, 1);
    ncnn::Mat fire8_ex1_w = mb.load(16384, 0);
    ncnn::Mat fire8_ex1_b = mb.load(256, 1);
    ncnn::Mat fire8_ex3_w = mb.load(147456, 0);
    ncnn::Mat fire8_ex3_b = mb.load(256, 1);

    fclose(fp);

    char path[512];
    // Write Fire6 weights
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_squeeze1x1_weight.bin", out_dir);
    write_file_bytes(path, fire6_sq_w.data, fire6_sq_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_squeeze1x1_bias.bin", out_dir);
    write_file_bytes(path, fire6_sq_b.data, fire6_sq_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_expand1x1_weight.bin", out_dir);
    write_file_bytes(path, fire6_ex1_w.data, fire6_ex1_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_expand1x1_bias.bin", out_dir);
    write_file_bytes(path, fire6_ex1_b.data, fire6_ex1_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_expand3x3_weight.bin", out_dir);
    write_file_bytes(path, fire6_ex3_w.data, fire6_ex3_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire6_expand3x3_bias.bin", out_dir);
    write_file_bytes(path, fire6_ex3_b.data, fire6_ex3_b.total() * sizeof(float));

    // Write Fire7 weights
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_squeeze1x1_weight.bin", out_dir);
    write_file_bytes(path, fire7_sq_w.data, fire7_sq_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_squeeze1x1_bias.bin", out_dir);
    write_file_bytes(path, fire7_sq_b.data, fire7_sq_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_expand1x1_weight.bin", out_dir);
    write_file_bytes(path, fire7_ex1_w.data, fire7_ex1_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_expand1x1_bias.bin", out_dir);
    write_file_bytes(path, fire7_ex1_b.data, fire7_ex1_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_expand3x3_weight.bin", out_dir);
    write_file_bytes(path, fire7_ex3_w.data, fire7_ex3_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire7_expand3x3_bias.bin", out_dir);
    write_file_bytes(path, fire7_ex3_b.data, fire7_ex3_b.total() * sizeof(float));

    // Write Fire8 weights
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_squeeze1x1_weight.bin", out_dir);
    write_file_bytes(path, fire8_sq_w.data, fire8_sq_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_squeeze1x1_bias.bin", out_dir);
    write_file_bytes(path, fire8_sq_b.data, fire8_sq_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_expand1x1_weight.bin", out_dir);
    write_file_bytes(path, fire8_ex1_w.data, fire8_ex1_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_expand1x1_bias.bin", out_dir);
    write_file_bytes(path, fire8_ex1_b.data, fire8_ex1_b.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_expand3x3_weight.bin", out_dir);
    write_file_bytes(path, fire8_ex3_w.data, fire8_ex3_w.total() * sizeof(float));
    snprintf(path, sizeof(path), "%s/squeezenet_fire8_expand3x3_bias.bin", out_dir);
    write_file_bytes(path, fire8_ex3_b.data, fire8_ex3_b.total() * sizeof(float));

    printf("Extracted Fire6, Fire7, Fire8 weights successfully.\n");

    // Strict single-thread FP32 forward pass
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

    // Read input tensor from existing fire2 input fixture (which is pool1 output)
    std::vector<uint8_t> p1_bytes = read_file_bytes("/data/data/com.termux/files/home/ameva-compute/tests/fixtures/cat_copilot_fire2_input_activation_nhwc.bin");
    ncnn::Mat in_pool1;
    if (p1_bytes.size() == 56 * 56 * 64 * sizeof(float)) {
        in_pool1.create(56, 56, 64);
        // NHWC to NCNN CHW
        const float* src = reinterpret_cast<const float*>(p1_bytes.data());
        for (int ch = 0; ch < 64; ++ch) {
            float* dst = in_pool1.channel(ch);
            for (int y = 0; y < 56; ++y) {
                for (int x = 0; x < 56; ++x) {
                    dst[y * 56 + x] = src[(y * 56 + x) * 64 + ch];
                }
            }
        }
    } else {
        fprintf(stderr, "Failed to read cat_copilot_fire2_input_activation_nhwc.bin (size=%zu)\n", p1_bytes.size());
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("pool1", in_pool1);

    ncnn::Mat out_pool5, out_fire6, out_fire7, out_fire8;
    ex.extract("pool5", out_pool5);
    ex.extract("fire6/concat", out_fire6);
    ex.extract("fire7/concat", out_fire7);
    ex.extract("fire8/concat", out_fire8);

    printf("NCNN Extracted Shapes:\n");
    printf("  pool5: %d x %d x %d (total: %zu)\n", out_pool5.w, out_pool5.h, out_pool5.c, out_pool5.total());
    printf("  fire6: %d x %d x %d (total: %zu)\n", out_fire6.w, out_fire6.h, out_fire6.c, out_fire6.total());
    printf("  fire7: %d x %d x %d (total: %zu)\n", out_fire7.w, out_fire7.h, out_fire7.c, out_fire7.total());
    printf("  fire8: %d x %d x %d (total: %zu)\n", out_fire8.w, out_fire8.h, out_fire8.c, out_fire8.total());

    // Convert CHW -> NHWC and write golden files
    std::vector<float> nhwc_f6 = chw_to_nhwc(out_fire6);
    std::vector<float> nhwc_f7 = chw_to_nhwc(out_fire7);
    std::vector<float> nhwc_f8 = chw_to_nhwc(out_fire8);

    snprintf(path, sizeof(path), "%s/squeezenet_fire6_golden.bin", out_dir);
    write_file_bytes(path, nhwc_f6.data(), nhwc_f6.size() * sizeof(float));
    printf("  Wrote fire6 golden: %zu floats, SHA256=%s\n", nhwc_f6.size(), compute_sha256(nhwc_f6.data(), nhwc_f6.size() * sizeof(float)).c_str());

    snprintf(path, sizeof(path), "%s/squeezenet_fire7_golden.bin", out_dir);
    write_file_bytes(path, nhwc_f7.data(), nhwc_f7.size() * sizeof(float));
    printf("  Wrote fire7 golden: %zu floats, SHA256=%s\n", nhwc_f7.size(), compute_sha256(nhwc_f7.data(), nhwc_f7.size() * sizeof(float)).c_str());

    snprintf(path, sizeof(path), "%s/squeezenet_fire8_golden.bin", out_dir);
    write_file_bytes(path, nhwc_f8.data(), nhwc_f8.size() * sizeof(float));
    printf("  Wrote fire8 golden: %zu floats, SHA256=%s\n", nhwc_f8.size(), compute_sha256(nhwc_f8.data(), nhwc_f8.size() * sizeof(float)).c_str());

    return 0;
}

