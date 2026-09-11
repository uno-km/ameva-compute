/**
 * @file test_vision_cat_fire2_integration.cpp
 * @brief End-to-End Cat Image Integration Test (stbi_load -> NCNN Strict FP32 -> AMEVA Vulkan GPU)
 *
 * Principles:
 *   - Real PNG decoding via stb_image (no mock/synthetic image)
 *   - Genuine NCNN Strict FP32 extraction (load_param/load_model/extract return checks)
 *   - AMEVA Vulkan Pointwise Conv2D 1x1 execution with Canary injection and return checks
 *   - Comprehensive Numerical Parity verification against CPU Double Reference and NCNN Strict FP32
 *   - Fail-Closed: any error exits with EXIT_FAILURE immediately.
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

#define STB_IMAGE_IMPLEMENTATION
#include "/data/data/com.termux/files/home/ncnn/src/stb_image.h"

#include "/data/data/com.termux/files/home/ncnn/src/net.h"
#include "/data/data/com.termux/files/home/ncnn/src/mat.h"

#include "ac_base.h"
#include "ac_runtime.h"
#include "ac_vision_ops.h"

// -----------------------------------------------------------------------------
// Constants & Configuration
// -----------------------------------------------------------------------------
constexpr uint32_t kCanary = 0x7fc12345u;

static const char* kParamPath = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param";
static const char* kBinPath = "/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin";
static const char* kCatImagePath = "/data/data/com.termux/files/home/cat_copilot_input.png";
static const char* kWeightPath = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_weight.bin";
static const char* kBiasPath = "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/squeezenet_fire2_squeeze1x1_bias.bin";

static const char* kExpectedCatPool1Sha = "efd76b451db0a92b12085a6f66beb8f2eb72eb62d2f7fc864f3b7ad027bfd163";
static const char* kExpectedCatFire2Sha = "7009eff14c198737c5eb9461dec98d7356dd3c502e7708dff2ea89e81f750c86";
static const char* kExpectedWeightSha = "9e420c4f34d556158e2214d023191c0d7c0e167aa5432845030b459e718b5918";
static const char* kExpectedBiasSha = "b38cfdd1c6cd695186eaf84a84d8fcafab76e83ece87df9c4af81e55db41583e";

// -----------------------------------------------------------------------------
// Portable SHA-256
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// File I/O Helper
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
// Layout Conversion Helper
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

// -----------------------------------------------------------------------------
// Numerical Parity Evaluation
// -----------------------------------------------------------------------------
struct NumericalMetrics {
    double cosine_similarity;
    double rmse;
    double p99_abs_error;
    double max_abs_error;
    uint32_t nan_count;
    uint32_t inf_count;
};

static NumericalMetrics evaluate_parity(const float* ref, const float* actual, size_t n) {
    NumericalMetrics m = {0.0, 0.0, 0.0, 0.0, 0, 0};
    if (!ref || !actual || n == 0) return m;

    double dot = 0.0, norm_ref_sq = 0.0, norm_act_sq = 0.0;
    double sum_sq_diff = 0.0;
    std::vector<double> abs_diffs(n);

    for (size_t i = 0; i < n; ++i) {
        float r = ref[i];
        float a = actual[i];

        if (std::isnan(a)) m.nan_count++;
        if (std::isinf(a)) m.inf_count++;

        double rd = (double)r;
        double ad = (double)a;
        double d = fabs(rd - ad);
        abs_diffs[i] = d;

        dot += rd * ad;
        norm_ref_sq += rd * rd;
        norm_act_sq += ad * ad;
        sum_sq_diff += d * d;
        if (d > m.max_abs_error) m.max_abs_error = d;
    }

    double denom = sqrt(norm_ref_sq) * sqrt(norm_act_sq);
    m.cosine_similarity = (denom > 1e-12) ? (dot / denom) : 0.0;
    m.rmse = sqrt(sum_sq_diff / (double)n);

    std::sort(abs_diffs.begin(), abs_diffs.end());
    size_t p99_idx = (size_t)std::ceil(0.99 * (double)n);
    if (p99_idx > 0) p99_idx -= 1;
    if (p99_idx >= n) p99_idx = n - 1;
    m.p99_abs_error = abs_diffs[p99_idx];

    return m;
}

// -----------------------------------------------------------------------------
// CPU Double-Precision Reference
// -----------------------------------------------------------------------------
static void cpu_pointwise_conv_f32(
    const float* in, const float* weight, const float* bias, float* out,
    uint32_t h, uint32_t w, uint32_t c_in, uint32_t c_out, uint32_t has_bias
) {
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const float* in_pixel = in + (y * w + x) * c_in;
            float* out_pixel = out + (y * w + x) * c_out;
            for (uint32_t oc = 0; oc < c_out; ++oc) {
                double acc = has_bias ? (double)bias[oc] : 0.0;
                const float* w_row = weight + oc * c_in;
                for (uint32_t ic = 0; ic < c_in; ++ic) {
                    acc += (double)in_pixel[ic] * (double)w_row[ic];
                }
                out_pixel[oc] = (float)acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Canary Pattern Helpers
// -----------------------------------------------------------------------------
static void fill_buffer_canary(void* ptr, size_t count_floats) {
    uint32_t* u32 = reinterpret_cast<uint32_t*>(ptr);
    for (size_t i = 0; i < count_floats; ++i) {
        u32[i] = kCanary;
    }
}

static size_t verify_buffer_not_canary(const void* ptr, size_t count_floats) {
    const uint32_t* u32 = reinterpret_cast<const uint32_t*>(ptr);
    size_t canary_matches = 0;
    for (size_t i = 0; i < count_floats; ++i) {
        if (u32[i] == kCanary) {
            canary_matches++;
        }
    }
    return canary_matches;
}

// -----------------------------------------------------------------------------
// Main Test Execution
// -----------------------------------------------------------------------------
int main(void) {
    printf("======================================================================\n");
    printf("AMEVA Vision - Cat Image Fire2 End-to-End Integration Test\n");
    printf("======================================================================\n\n");

    uint32_t vision_gpu_op_count = 0;

    // -------------------------------------------------------------------------
    // 1. Decode Cat Image via stb_image
    // -------------------------------------------------------------------------
    printf("--- [STEP 1: REAL PNG DECODE & PREPROCESS] ---\n");
    const char* env_cat_img = getenv("AMEVA_VISION_CAT_IMAGE_PATH");
    const char* cat_img_path = (env_cat_img && env_cat_img[0] != '\0') ? env_cat_img : kCatImagePath;
    int cat_w = 0, cat_h = 0, cat_c = 0;
    unsigned char* raw_pixels = stbi_load(cat_img_path, &cat_w, &cat_h, &cat_c, 3);
    if (!raw_pixels) {
        fprintf(stderr, "FATAL: Could not load %s\n", cat_img_path);
        return EXIT_FAILURE;
    }
    printf("Loaded image: %s (w=%d, h=%d, c=%d)\n", cat_img_path, cat_w, cat_h, cat_c);

    ncnn::Mat in_mat = ncnn::Mat::from_pixels_resize(raw_pixels, ncnn::Mat::PIXEL_RGB, cat_w, cat_h, 227, 227);
    stbi_image_free(raw_pixels);
    if (in_mat.empty()) {
        fprintf(stderr, "FATAL: from_pixels_resize produced empty Mat\n");
        return EXIT_FAILURE;
    }

    const float mean_vals[3] = { 104.f, 117.f, 123.f };
    in_mat.substract_mean_normalize(mean_vals, 0);

    // -------------------------------------------------------------------------
    // 2. NCNN Strict FP32 Extraction
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 2: NCNN STRICT FP32 FEATURE EXTRACTION] ---\n");
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
    if (net.load_param(kParamPath) != 0) {
        fprintf(stderr, "FATAL: net.load_param failed: %s\n", kParamPath);
        return EXIT_FAILURE;
    }
    if (net.load_model(kBinPath) != 0) {
        fprintf(stderr, "FATAL: net.load_model failed: %s\n", kBinPath);
        return EXIT_FAILURE;
    }

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("data", in_mat) != 0) {
        fprintf(stderr, "FATAL: ex.input('data') failed\n");
        return EXIT_FAILURE;
    }

    ncnn::Mat ncnn_pool1;
    if (ex.extract("pool1", ncnn_pool1) != 0 || ncnn_pool1.empty()) {
        fprintf(stderr, "FATAL: ex.extract('pool1') failed or empty\n");
        return EXIT_FAILURE;
    }
    std::vector<float> pool1_nhwc = chw_to_nhwc(ncnn_pool1);
    std::string pool1_sha = compute_sha256(pool1_nhwc.data(), pool1_nhwc.size() * sizeof(float));

    ncnn::Mat ncnn_fire2;
    if (ex.extract("fire2/squeeze1x1", ncnn_fire2) != 0 || ncnn_fire2.empty()) {
        fprintf(stderr, "FATAL: ex.extract('fire2/squeeze1x1') failed or empty\n");
        return EXIT_FAILURE;
    }
    std::vector<float> fire2_ncnn_nhwc = chw_to_nhwc(ncnn_fire2);
    std::string fire2_ncnn_sha = compute_sha256(fire2_ncnn_nhwc.data(), fire2_ncnn_nhwc.size() * sizeof(float));

    printf("CAT_POOL1_SHA256=%s (Elements=%zu)\n", pool1_sha.c_str(), pool1_nhwc.size());
    printf("CAT_FIRE2_STRICT_FP32_SHA256=%s (Elements=%zu)\n", fire2_ncnn_sha.c_str(), fire2_ncnn_nhwc.size());

    if (pool1_sha != kExpectedCatPool1Sha) {
        fprintf(stderr, "FATAL: Cat pool1 SHA-256 mismatch! Got %s, expected %s\n", pool1_sha.c_str(), kExpectedCatPool1Sha);
        return EXIT_FAILURE;
    }
    if (fire2_ncnn_sha != kExpectedCatFire2Sha) {
        fprintf(stderr, "FATAL: Cat fire2 NCNN strict FP32 SHA-256 mismatch! Got %s, expected %s\n", fire2_ncnn_sha.c_str(), kExpectedCatFire2Sha);
        return EXIT_FAILURE;
    }
    printf("NCNN Feature Extraction verified against golden SHA-256: PASS\n");

    // -------------------------------------------------------------------------
    // 3. Load Model Weight and Bias Fixtures
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 3: LOAD WEIGHT AND BIAS FIXTURES] ---\n");
    std::string weight_path = kWeightPath;
    std::string bias_path = kBiasPath;
    const char* env_dir = getenv("AMEVA_VISION_FIXTURE_DIR");
    if (env_dir && env_dir[0] != '\0') {
        weight_path = std::string(env_dir) + "/squeezenet_fire2_squeeze1x1_weight.bin";
        bias_path = std::string(env_dir) + "/squeezenet_fire2_squeeze1x1_bias.bin";
    }

    std::vector<uint8_t> w_bytes = read_file_bytes(weight_path.c_str());
    std::vector<uint8_t> b_bytes = read_file_bytes(bias_path.c_str());
    if (w_bytes.size() != 4096 || b_bytes.size() != 64) {
        fprintf(stderr, "FATAL: Weight (%zu) or bias (%zu) size invalid\n", w_bytes.size(), b_bytes.size());
        return EXIT_FAILURE;
    }
    std::string w_sha = compute_sha256(w_bytes.data(), w_bytes.size());
    std::string b_sha = compute_sha256(b_bytes.data(), b_bytes.size());
    printf("WEIGHT_SHA256=%s\n", w_sha.c_str());
    printf("BIAS_SHA256=%s\n", b_sha.c_str());
    if (w_sha != kExpectedWeightSha || b_sha != kExpectedBiasSha) {
        fprintf(stderr, "FATAL: Weight or bias SHA-256 mismatch!\n");
        return EXIT_FAILURE;
    }

    const float* host_w = reinterpret_cast<const float*>(w_bytes.data());
    const float* host_b = reinterpret_cast<const float*>(b_bytes.data());

    // -------------------------------------------------------------------------
    // 4. Compute CPU Double-Precision Reference
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 4: COMPUTE CPU DOUBLE-PRECISION REFERENCE] ---\n");
    const uint32_t H = 56, W = 56, Cin = 64, Cout = 16;
    const size_t out_elems = H * W * Cout;
    std::vector<float> cpu_double_ref(out_elems);

    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_pointwise_conv_f32(
        pool1_nhwc.data(), host_w, host_b, cpu_double_ref.data(),
        H, W, Cin, Cout, 1
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ref_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::string cpu_double_sha = compute_sha256(cpu_double_ref.data(), out_elems * sizeof(float));
    printf("CPU_DOUBLE_REF_SHA256=%s (Time=%.1f us)\n", cpu_double_sha.c_str(), cpu_ref_time_us);

    // -------------------------------------------------------------------------
    // 5. Initialize AMEVA Vulkan Runtime & Execute on GPU
    // -------------------------------------------------------------------------
    printf("\n--- [STEP 5: AMEVA VULKAN GPU EXECUTION] ---\n");
    uint32_t ctx_created = 0, ctx_destroyed = 0;
    uint32_t kernel_created = 0, kernel_destroyed = 0;
    uint32_t buffer_created = 0, buffer_destroyed = 0;
    int test_ret_code = EXIT_FAILURE;

    AcContext* ctx = nullptr;
    AcKernel* kernel = nullptr;
    AcBuffer* buf_in = nullptr;
    AcBuffer* buf_w = nullptr;
    AcBuffer* buf_b = nullptr;
    AcBuffer* buf_out = nullptr;

    float* gpu_in_ptr = nullptr;
    float* gpu_w_ptr = nullptr;
    float* gpu_b_ptr = nullptr;
    float* gpu_out_ptr = nullptr;

    bool double_parity_pass = false;
    bool ncnn_parity_pass = false;

    const size_t in_bytes = H * W * Cin * sizeof(float);
    const size_t weight_bytes = Cout * Cin * sizeof(float);
    const size_t bias_bytes = Cout * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);

    AcVisionPointwiseConvParams params = {};
    params.abi_version = AC_VISION_ABI_VERSION;
    params.struct_size = sizeof(AcVisionPointwiseConvParams);
    params.input_channels = Cin;
    params.output_channels = Cout;
    params.input_height = H;
    params.input_width = W;
    params.has_bias = 1;
    params.kernel_h = 1;
    params.kernel_w = 1;
    params.stride_h = 1;
    params.stride_w = 1;
    params.dilation_h = 1;
    params.dilation_w = 1;

    AcVisionTensorDesc in_desc = {};
    in_desc.abi_version = AC_VISION_ABI_VERSION;
    in_desc.struct_size = sizeof(AcVisionTensorDesc);
    in_desc.dtype = AC_VISION_DTYPE_F32;
    in_desc.layout = AC_VISION_LAYOUT_NHWC;
    in_desc.n = 1;
    in_desc.h = H;
    in_desc.w = W;
    in_desc.c = Cin;
    in_desc.row_stride_bytes = W * Cin * sizeof(float);

    AcVisionTensorDesc weight_desc = {};
    weight_desc.abi_version = AC_VISION_ABI_VERSION;
    weight_desc.struct_size = sizeof(AcVisionTensorDesc);
    weight_desc.dtype = AC_VISION_DTYPE_F32;
    weight_desc.layout = AC_VISION_LAYOUT_OHWI;
    weight_desc.n = Cout;
    weight_desc.h = 1;
    weight_desc.w = 1;
    weight_desc.c = Cin;
    weight_desc.row_stride_bytes = 0;

    AcVisionTensorDesc out_desc = {};
    out_desc.abi_version = AC_VISION_ABI_VERSION;
    out_desc.struct_size = sizeof(AcVisionTensorDesc);
    out_desc.dtype = AC_VISION_DTYPE_F32;
    out_desc.layout = AC_VISION_LAYOUT_NHWC;
    out_desc.n = 1;
    out_desc.h = H;
    out_desc.w = W;
    out_desc.c = Cout;
    out_desc.row_stride_bytes = W * Cout * sizeof(float);

    AcBufferDesc bdesc_in = { AC_ABI_VERSION, sizeof(AcBufferDesc), in_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_w = { AC_ABI_VERSION, sizeof(AcBufferDesc), weight_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_b = { AC_ABI_VERSION, sizeof(AcBufferDesc), bias_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_out = { AC_ABI_VERSION, sizeof(AcBufferDesc), out_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    AcContextDesc ctx_desc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcResult res = ac_runtime_context_create(&ctx_desc, &ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "FATAL: ac_runtime_context_create failed with code %d\n", res);
        goto cleanup;
    }
    ctx_created++;

    res = ac_vision_pointwise_conv_kernel_create(ctx, nullptr, 0, nullptr, &kernel);
    if (res != AC_SUCCESS || !kernel) {
        fprintf(stderr, "FATAL: ac_vision_pointwise_conv_kernel_create failed with code %d\n", res);
        goto cleanup;
    }
    kernel_created++;

    if (ac_runtime_buffer_create(ctx, &bdesc_in, &buf_in) != AC_SUCCESS || !buf_in) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_create buf_in failed\n");
        goto cleanup;
    }
    buffer_created++;

    if (ac_runtime_buffer_create(ctx, &bdesc_w, &buf_w) != AC_SUCCESS || !buf_w) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_create buf_w failed\n");
        goto cleanup;
    }
    buffer_created++;

    if (ac_runtime_buffer_create(ctx, &bdesc_b, &buf_b) != AC_SUCCESS || !buf_b) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_create buf_b failed\n");
        goto cleanup;
    }
    buffer_created++;

    if (ac_runtime_buffer_create(ctx, &bdesc_out, &buf_out) != AC_SUCCESS || !buf_out) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_create buf_out failed\n");
        goto cleanup;
    }
    buffer_created++;

    if (ac_runtime_buffer_get_mapped_ptr(buf_in, (void**)&gpu_in_ptr) != AC_SUCCESS ||
        ac_runtime_buffer_get_mapped_ptr(buf_w, (void**)&gpu_w_ptr) != AC_SUCCESS ||
        ac_runtime_buffer_get_mapped_ptr(buf_b, (void**)&gpu_b_ptr) != AC_SUCCESS ||
        ac_runtime_buffer_get_mapped_ptr(buf_out, (void**)&gpu_out_ptr) != AC_SUCCESS ||
        !gpu_in_ptr || !gpu_w_ptr || !gpu_b_ptr || !gpu_out_ptr) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_get_mapped_ptr failed\n");
        goto cleanup;
    }

    // Upload Input, Weight, Bias
    memcpy(gpu_in_ptr, pool1_nhwc.data(), in_bytes);
    memcpy(gpu_w_ptr, host_w, weight_bytes);
    memcpy(gpu_b_ptr, host_b, bias_bytes);

    if (ac_runtime_buffer_flush(buf_in, 0, in_bytes) != AC_SUCCESS ||
        ac_runtime_buffer_flush(buf_w, 0, weight_bytes) != AC_SUCCESS ||
        ac_runtime_buffer_flush(buf_b, 0, bias_bytes) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_flush failed\n");
        goto cleanup;
    }

    // Canary Injection into Output Buffer (P0-2)
    fill_buffer_canary(gpu_out_ptr, out_elems);
    if (ac_runtime_buffer_flush(buf_out, 0, out_bytes) != AC_SUCCESS) {
        fprintf(stderr, "FATAL: ac_runtime_buffer_flush for output canary failed\n");
        goto cleanup;
    }

    // GPU Dispatch with return check (P0-1)
    {
        auto tg0 = std::chrono::high_resolution_clock::now();
        res = ac_vision_pointwise_conv_with_kernel_f32(
            ctx, kernel, &params,
            buf_in, &in_desc, buf_w, &weight_desc, buf_b,
            buf_out, &out_desc, nullptr
        );
        auto tg1 = std::chrono::high_resolution_clock::now();
        double gpu_dispatch_time_us = std::chrono::duration<double, std::micro>(tg1 - tg0).count();

        if (res != AC_SUCCESS) {
            fprintf(stderr, "CAT_GPU_DISPATCH_FAILED=%d\n", res);
            goto cleanup;
        }
        vision_gpu_op_count++;

        // Invalidate Output Buffer (P0-1)
        if (ac_runtime_buffer_invalidate(buf_out, 0, out_bytes) != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_runtime_buffer_invalidate failed\n");
            goto cleanup;
        }

        // Verify No Canary Remaining (P0-2)
        size_t unwritten_canary_count = verify_buffer_not_canary(gpu_out_ptr, out_elems);
        if (unwritten_canary_count != 0) {
            fprintf(stderr, "FATAL: CAT_OUTPUT_UNWRITTEN_ELEMENT_COUNT=%zu\n", unwritten_canary_count);
            goto cleanup;
        }

        const float* host_gpu_out = reinterpret_cast<const float*>(gpu_out_ptr);
        std::string gpu_out_sha = compute_sha256(host_gpu_out, out_bytes);
        printf("CAT_AMEVA_GPU_OUTPUT_SHA256=%s (Time=%.1f us)\n", gpu_out_sha.c_str(), gpu_dispatch_time_us);

        // -------------------------------------------------------------------------
        // 6. Numerical Parity Evaluation
        // -------------------------------------------------------------------------
        printf("\n--- [STEP 6: NUMERICAL PARITY EVALUATION] ---\n");
        NumericalMetrics nm_double = evaluate_parity(cpu_double_ref.data(), host_gpu_out, out_elems);
        NumericalMetrics nm_ncnn = evaluate_parity(fire2_ncnn_nhwc.data(), host_gpu_out, out_elems);

        printf("  [PARITY 1: AMEVA GPU vs CPU DOUBLE REFERENCE]\n");
        printf("    COSINE_SIMILARITY=%.7f\n", nm_double.cosine_similarity);
        printf("    RMSE=%.7f\n", nm_double.rmse);
        printf("    P99_ABS_ERROR=%.7f\n", nm_double.p99_abs_error);
        printf("    MAX_ABS_ERROR=%.7f\n", nm_double.max_abs_error);
        printf("    NAN_COUNT=%u, INF_COUNT=%u\n", nm_double.nan_count, nm_double.inf_count);

        printf("  [PARITY 2: AMEVA GPU vs NCNN STRICT FP32]\n");
        printf("    COSINE_SIMILARITY=%.7f\n", nm_ncnn.cosine_similarity);
        printf("    RMSE=%.7f\n", nm_ncnn.rmse);
        printf("    P99_ABS_ERROR=%.7f\n", nm_ncnn.p99_abs_error);
        printf("    MAX_ABS_ERROR=%.7f\n", nm_ncnn.max_abs_error);
        printf("    NAN_COUNT=%u, INF_COUNT=%u\n", nm_ncnn.nan_count, nm_ncnn.inf_count);

        double_parity_pass = (nm_double.cosine_similarity >= 0.99999) &&
                             (nm_double.rmse <= 0.005) &&
                             (nm_double.p99_abs_error <= 0.01) &&
                             (nm_double.max_abs_error <= 0.05) &&
                             (nm_double.nan_count == 0) &&
                             (nm_double.inf_count == 0);

        ncnn_parity_pass = (nm_ncnn.cosine_similarity >= 0.99999) &&
                           (nm_ncnn.rmse <= 0.005) &&
                           (nm_ncnn.p99_abs_error <= 0.01) &&
                           (nm_ncnn.max_abs_error <= 0.05) &&
                           (nm_ncnn.nan_count == 0) &&
                           (nm_ncnn.inf_count == 0);

        printf("AMEVA_VS_DOUBLE_PARITY=%s\n", double_parity_pass ? "PASS" : "FAIL");
        printf("AMEVA_VS_NCNN_STRICT_FP32_PARITY=%s\n", ncnn_parity_pass ? "PASS" : "FAIL");

        printf("\n--- [FINAL AUDIT REPORT] ---\n");
        printf("TOTAL_VISION_GPU_OP_COUNT=%u\n", vision_gpu_op_count);
        printf("COUNT_RECONCILED=%s\n", (vision_gpu_op_count == 1) ? "true" : "false");
        printf("FALLBACK_PATH_IMPLEMENTED=false\n");

        bool all_pass = double_parity_pass && ncnn_parity_pass && (vision_gpu_op_count == 1);
        printf("INTEGRATION_TEST_RESULT=%s\n", all_pass ? "PASS" : "FAIL");

        if (all_pass) {
            test_ret_code = EXIT_SUCCESS;
        }
    }

cleanup:
    // -------------------------------------------------------------------------
    // 7. Cleanup & Lifecycle Verification (P0-3)
    // -------------------------------------------------------------------------
    if (buf_in) { ac_runtime_buffer_destroy(buf_in); buffer_destroyed++; }
    if (buf_w) { ac_runtime_buffer_destroy(buf_w); buffer_destroyed++; }
    if (buf_b) { ac_runtime_buffer_destroy(buf_b); buffer_destroyed++; }
    if (buf_out) { ac_runtime_buffer_destroy(buf_out); buffer_destroyed++; }
    if (kernel) { ac_vision_pointwise_conv_kernel_destroy(kernel); kernel_destroyed++; }
    if (ctx) { ac_runtime_context_destroy(ctx); ctx_destroyed++; }

    const char* fail_stage_env = getenv("AMEVA_VISION_FAIL_STAGE");
    bool is_dispatch_fail_mode = (fail_stage_env && strcmp(fail_stage_env, "dispatch") == 0);

    if (is_dispatch_fail_mode) {
        printf("\n--- [DISPATCH FAILURE LIFECYCLE RECOVERY REPORT] ---\n");
        printf("DISPATCH_FAILURE_CONTEXT_CREATE_DESTROY=%u/%u\n", ctx_created, ctx_destroyed);
        printf("DISPATCH_FAILURE_KERNEL_CREATE_DESTROY=%u/%u\n", kernel_created, kernel_destroyed);
        printf("DISPATCH_FAILURE_BUFFER_CREATE_DESTROY=%u/%u\n", buffer_created, buffer_destroyed);
        bool recovery_ok = (ctx_created == ctx_destroyed) &&
                           (kernel_created == kernel_destroyed) &&
                           (buffer_created == buffer_destroyed);
        printf("DISPATCH_FAILURE_RECOVERY=%s\n", recovery_ok ? "PASS" : "FAIL");
    } else {
        printf("\n--- [NORMAL LIFECYCLE RECOVERY REPORT] ---\n");
        printf("NORMAL_CONTEXT_CREATE_DESTROY=%u/%u\n", ctx_created, ctx_destroyed);
        printf("NORMAL_KERNEL_CREATE_DESTROY=%u/%u\n", kernel_created, kernel_destroyed);
        printf("NORMAL_BUFFER_CREATE_DESTROY=%u/%u\n", buffer_created, buffer_destroyed);
        bool normal_lifecycle_ok = (ctx_created == ctx_destroyed) &&
                                   (kernel_created == kernel_destroyed) &&
                                   (buffer_created == buffer_destroyed);
        printf("NORMAL_LIFECYCLE_VERIFIED=%s\n", normal_lifecycle_ok ? "PASS" : "FAIL");
    }

    return test_ret_code;
}
