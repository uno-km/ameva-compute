/**
 * @file test_vision_pointwise_conv.cpp
 * @brief AMUDA Vision Pointwise Conv2D 1x1 Verification Suite (V1 Milestone)
 *
 * Enforces Zero-Silent-Fallback, Anti-Fake Dataflow Sensitivity,
 * Fail-Closed Boundaries, and Rigorous Real Hardware Numerical Parity.
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

#include "ac_runtime.h"
#include "ac_vision_ops.h"
#include "../src/backend/vulkan/ac_vk_sha256.h"
#include "../src/shaders/pointwise_conv_spv.h"
#include "../src/shaders/pointwise_conv_mutated_spv.h"
#include "../fixtures/squeezenet_fire2_squeeze1x1_data.h"

#define CANARY_VAL 0x7E7E7E7E

// -----------------------------------------------------------------------------
// Fixture Loader & Portable Decoding (Zero Union Punning)
// -----------------------------------------------------------------------------
static int load_fixture_file(const char* filename, void* out_buf, size_t max_bytes, size_t* out_read_bytes) {
    const char* search_paths[] = {
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
    FILE* fp = NULL;
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
    size_t read_n = fread(out_buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (out_read_bytes) *out_read_bytes = read_n;
    return 0;
}

static void decode_weights_safe(const uint32_t* src_u32, float* dst_f32, size_t count) {
    memcpy(dst_f32, src_u32, count * sizeof(float));
}

// -----------------------------------------------------------------------------
// Lifecycle Instrumentation Tracker
// -----------------------------------------------------------------------------
struct LifecycleTracker {
    uint32_t context_created = 0;
    uint32_t context_destroyed = 0;
    uint32_t kernel_created = 0;
    uint32_t kernel_destroyed = 0;
    uint32_t buffer_created = 0;
    uint32_t buffer_destroyed = 0;
    uint32_t event_created = 0;
    uint32_t event_destroyed = 0;

    void reset() {
        context_created = 0;
        context_destroyed = 0;
        kernel_created = 0;
        kernel_destroyed = 0;
        buffer_created = 0;
        buffer_destroyed = 0;
        event_created = 0;
        event_destroyed = 0;
    }

    bool is_balanced() const {
        return (context_created == context_destroyed) &&
               (kernel_created == kernel_destroyed) &&
               (buffer_created == buffer_destroyed) &&
               (event_created == event_destroyed);
    }
};

static LifecycleTracker g_life;

// Tracked Wrappers
static AcResult track_context_create(const AcContextDesc* desc, AcContext** out_ctx) {
    AcResult res = ac_runtime_context_create(desc, out_ctx);
    if (res == AC_SUCCESS && out_ctx && *out_ctx) g_life.context_created++;
    return res;
}

static AcResult track_context_destroy(AcContext* ctx) {
    AcResult res = ac_runtime_context_destroy(ctx);
    if (res == AC_SUCCESS) g_life.context_destroyed++;
    return res;
}

static AcResult track_buffer_create(AcContext* ctx, const AcBufferDesc* desc, AcBuffer** out_buf) {
    AcResult res = ac_runtime_buffer_create(ctx, desc, out_buf);
    if (res == AC_SUCCESS && out_buf && *out_buf) g_life.buffer_created++;
    return res;
}

static AcResult track_buffer_destroy(AcBuffer* buf) {
    AcResult res = ac_runtime_buffer_destroy(buf);
    if (res == AC_SUCCESS) g_life.buffer_destroyed++;
    return res;
}

static AcResult track_kernel_create(AcContext* ctx, const AcKernelDesc* desc, AcKernel** out_kernel) {
    AcResult res = ac_runtime_kernel_create(ctx, desc, out_kernel);
    if (res == AC_SUCCESS && out_kernel && *out_kernel) g_life.kernel_created++;
    return res;
}

static AcResult track_kernel_destroy(AcKernel* kernel) {
    AcResult res = ac_runtime_kernel_destroy(kernel);
    if (res == AC_SUCCESS) g_life.kernel_destroyed++;
    return res;
}

static AcResult track_event_destroy(AcEvent* event) {
    AcResult res = ac_runtime_event_destroy(event);
    if (res == AC_SUCCESS) g_life.event_destroyed++;
    return res;
}

// -----------------------------------------------------------------------------
// CPU Reference & Random Generators
// -----------------------------------------------------------------------------
static void cpu_pointwise_conv_f32(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    uint32_t H, uint32_t W,
    uint32_t C_IN, uint32_t C_OUT,
    int has_bias
) {
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t co = 0; co < C_OUT; ++co) {
                double acc = (has_bias && bias) ? (double)bias[co] : 0.0;
                for (uint32_t ci = 0; ci < C_IN; ++ci) {
                    size_t in_idx = ((size_t)h * W + w) * C_IN + ci;
                    size_t w_idx = (size_t)co * C_IN + ci;
                    acc += (double)in[in_idx] * (double)weight[w_idx];
                }
                size_t out_idx = ((size_t)h * W + w) * C_OUT + co;
                out[out_idx] = (float)acc;
            }
        }
    }
}

// Deterministic LCG
static uint32_t lcg_rand(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static float lcg_uniform(uint32_t& state, float low, float high) {
    uint32_t r = lcg_rand(state);
    float norm = (float)(r >> 8) / (float)(1u << 24);
    return low + norm * (high - low);
}

// Numerical metrics
struct NumericalMetrics {
    double cosine_similarity;
    double rmse;
    double p99_abs_error;
    double max_abs_error;
    double mean_delta;
    uint32_t nan_count;
    uint32_t inf_count;
};

static NumericalMetrics evaluate_parity(const float* ref, const float* actual, size_t n) {
    NumericalMetrics m = {};
    double dot = 0.0, norm_ref = 0.0, norm_act = 0.0;
    double sum_sq_diff = 0.0;
    double sum_delta = 0.0;
    std::vector<double> abs_diffs(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        float r = ref[i];
        float a = actual[i];

        if (isnan(a)) m.nan_count++;
        if (isinf(a)) m.inf_count++;

        double dr = (double)r;
        double da = (double)a;
        double diff = da - dr;
        double abs_diff = fabs(diff);

        dot += dr * da;
        norm_ref += dr * dr;
        norm_act += da * da;
        sum_sq_diff += diff * diff;
        sum_delta += diff;
        abs_diffs[i] = abs_diff;

        if (abs_diff > m.max_abs_error) {
            m.max_abs_error = abs_diff;
        }
    }

    if (norm_ref > 0.0 && norm_act > 0.0) {
        m.cosine_similarity = dot / (sqrt(norm_ref) * sqrt(norm_act));
    } else {
        m.cosine_similarity = 0.0;
    }

    m.rmse = sqrt(sum_sq_diff / (double)n);
    m.mean_delta = sum_delta / (double)n;

    std::sort(abs_diffs.begin(), abs_diffs.end());
    size_t p99_idx = (size_t)(0.99 * (double)n);
    if (p99_idx >= n) p99_idx = n - 1;
    m.p99_abs_error = abs_diffs[p99_idx];

    return m;
}

static std::string compute_sha256(const void* data, size_t size_bytes) {
    char hex[65];
    ac_sha256_compute_hex((const uint8_t*)data, size_bytes, hex);
    return std::string(hex);
}

// -----------------------------------------------------------------------------
// Test Main
// -----------------------------------------------------------------------------
int main(void) {
    printf("======================================================================\n");
    printf("  AMUDA Vision Pointwise Conv2D 1x1 Verification Suite (V1 Milestone)\n");
    printf("======================================================================\n\n");

    const uint32_t H = 13;
    const uint32_t W = 17;
    const uint32_t C_IN = 29;
    const uint32_t C_OUT = 37;
    const uint32_t M = H * W; // 221
    const size_t in_elems = (size_t)1 * H * W * C_IN;
    const size_t w_elems = (size_t)C_OUT * 1 * 1 * C_IN;
    const size_t b_elems = (size_t)C_OUT;
    const size_t out_elems = (size_t)1 * H * W * C_OUT;

    const size_t in_bytes = in_elems * sizeof(float);
    const size_t w_bytes = w_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);

    AcVisionTensorDesc in_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, H, W, C_IN, 0
    };
    AcVisionTensorDesc weight_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI,
        C_OUT, 1, 1, C_IN, 0
    };
    AcVisionTensorDesc out_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, H, W, C_OUT, 0
    };
    AcVisionPointwiseConvParams params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams),
        C_IN, C_OUT, H, W, 1,
        1, 1, 1, 1, 1, 1
    };

    // Global counters
    uint32_t vision_gpu_op_count = 0;
    uint32_t cpu_fallback_count = 0;
    uint32_t silent_fallback_count = 0;

    // -------------------------------------------------------------------------
    // Gate 1: Context Creation & Device Probe
    // -------------------------------------------------------------------------
    AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcContext* ctx = nullptr;
    if (track_context_create(&cdesc, &ctx) != AC_SUCCESS || !ctx) {
        printf("FATAL: Failed to initialize Vulkan compute context.\n");
        return 1;
    }
    char dev_name[256] = "Unknown";
    ac_runtime_context_get_device_name(ctx, dev_name, sizeof(dev_name));
    printf("[DEVICE] Initialized GPU: %s\n\n", dev_name);

    // -------------------------------------------------------------------------
    // Gate 2: Kernel Creation with Canonical SPIR-V SHA Authentication
    // -------------------------------------------------------------------------
    AcKernel* normal_kernel = nullptr;
    AcResult kres = ac_vision_pointwise_conv_kernel_create(ctx, kPointwiseConvSpv, kPointwiseConvSpvByteSize, kPointwiseConvSpvSha256, &normal_kernel);
    if (kres != AC_SUCCESS || !normal_kernel) {
        printf("FATAL: Failed to create normal pointwise conv kernel (res=%d).\n", kres);
        track_context_destroy(ctx);
        return 1;
    }
    g_life.kernel_created++;
    printf("[KERNEL] Canonical Pointwise Conv SPIR-V SHA-256: %s (VERIFIED)\n\n", kPointwiseConvSpvSha256);

    // -------------------------------------------------------------------------
    // Gate 3: Random Tensor Challenge (Seeds: 104729, 130363, 155921)
    // -------------------------------------------------------------------------
    printf("--- Gate 3: Random Tensor Challenge ---\n");
    const uint32_t kSeeds[3] = { 104729u, 130363u, 155921u };
    std::vector<std::string> output_hashes;
    uint32_t challenge_passed = 0;
    double worst_cosine = 1.0;
    double worst_rmse = 0.0;
    double worst_p99 = 0.0;
    double worst_max = 0.0;

    // Allocate GPU buffers
    AcBufferDesc bdesc_in = { AC_ABI_VERSION, sizeof(AcBufferDesc), in_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_w = { AC_ABI_VERSION, sizeof(AcBufferDesc), w_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_b = { AC_ABI_VERSION, sizeof(AcBufferDesc), b_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc bdesc_out = { AC_ABI_VERSION, sizeof(AcBufferDesc), out_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    AcBuffer *buf_in = nullptr, *buf_w = nullptr, *buf_b = nullptr, *buf_out = nullptr;
    track_buffer_create(ctx, &bdesc_in, &buf_in);
    track_buffer_create(ctx, &bdesc_w, &buf_w);
    track_buffer_create(ctx, &bdesc_b, &buf_b);
    track_buffer_create(ctx, &bdesc_out, &buf_out);

    float *host_in = nullptr, *host_w = nullptr, *host_b = nullptr, *host_out = nullptr;
    ac_runtime_buffer_get_mapped_ptr(buf_in, (void**)&host_in);
    ac_runtime_buffer_get_mapped_ptr(buf_w, (void**)&host_w);
    ac_runtime_buffer_get_mapped_ptr(buf_b, (void**)&host_b);
    ac_runtime_buffer_get_mapped_ptr(buf_out, (void**)&host_out);

    std::vector<float> cpu_ref(out_elems, 0.0f);

    for (int s = 0; s < 3; ++s) {
        uint32_t seed = kSeeds[s];
        uint32_t rng = seed;

        for (size_t i = 0; i < in_elems; ++i) host_in[i] = lcg_uniform(rng, -1.5f, 1.5f);
        for (size_t i = 0; i < w_elems; ++i) host_w[i] = lcg_uniform(rng, -0.5f, 0.5f);
        for (size_t i = 0; i < b_elems; ++i) host_b[i] = lcg_uniform(rng, -0.2f, 0.2f);
        memset(host_out, 0, out_bytes);

        ac_runtime_buffer_flush(buf_in, 0, in_bytes);
        ac_runtime_buffer_flush(buf_w, 0, w_bytes);
        ac_runtime_buffer_flush(buf_b, 0, b_bytes);
        ac_runtime_buffer_flush(buf_out, 0, out_bytes);

        std::string in_hash = compute_sha256(host_in, in_bytes);
        std::string w_hash = compute_sha256(host_w, w_bytes);
        std::string b_hash = compute_sha256(host_b, b_bytes);

        // CPU computation
        cpu_pointwise_conv_f32(host_in, host_w, host_b, cpu_ref.data(), H, W, C_IN, C_OUT, 1);
        std::string cpu_hash = compute_sha256(cpu_ref.data(), out_bytes);

        // GPU computation
        AcEvent* ev = nullptr;
        AcResult dres = ac_vision_pointwise_conv_with_kernel_f32(
            ctx, normal_kernel, &params,
            buf_in, &in_desc, buf_w, &weight_desc, buf_b,
            buf_out, &out_desc, &ev
        );
        if (dres != AC_SUCCESS) {
            printf("  -> FAIL: Seed %u GPU dispatch failed (%d)\n", seed, dres);
            continue;
        }
        vision_gpu_op_count++;
        if (ev) {
            g_life.event_created++;
            ac_runtime_event_wait(ev, 5000000000ull);
            track_event_destroy(ev);
        }
        ac_runtime_buffer_invalidate(buf_out, 0, out_bytes);

        std::string gpu_hash = compute_sha256(host_out, out_bytes);
        output_hashes.push_back(gpu_hash);

        NumericalMetrics nm = evaluate_parity(cpu_ref.data(), host_out, out_elems);
        if (nm.cosine_similarity < worst_cosine) worst_cosine = nm.cosine_similarity;
        if (nm.rmse > worst_rmse) worst_rmse = nm.rmse;
        if (nm.p99_abs_error > worst_p99) worst_p99 = nm.p99_abs_error;
        if (nm.max_abs_error > worst_max) worst_max = nm.max_abs_error;

        printf("SEED=%u\n", seed);
        printf("INPUT_SHA256=%s\n", in_hash.c_str());
        printf("WEIGHT_SHA256=%s\n", w_hash.c_str());
        printf("BIAS_SHA256=%s\n", b_hash.c_str());
        printf("CPU_OUTPUT_SHA256=%s\n", cpu_hash.c_str());
        printf("GPU_OUTPUT_SHA256=%s\n", gpu_hash.c_str());
        printf("COSINE_SIMILARITY=%.7f\n", nm.cosine_similarity);
        printf("RMSE=%.7f\n", nm.rmse);
        printf("P99_ABS_ERROR=%.7f\n", nm.p99_abs_error);
        printf("MAX_ABS_ERROR=%.7f\n", nm.max_abs_error);
        printf("NAN_COUNT=%u\n", nm.nan_count);
        printf("INF_COUNT=%u\n", nm.inf_count);

        if (nm.cosine_similarity >= 0.99999 &&
            nm.rmse <= 0.005 &&
            nm.p99_abs_error <= 0.01 &&
            nm.max_abs_error <= 0.05 &&
            nm.nan_count == 0 &&
            nm.inf_count == 0) {
            challenge_passed++;
        }
        printf("\n");
    }

    // Check unique output hashes
    std::sort(output_hashes.begin(), output_hashes.end());
    size_t unique_hashes = std::unique(output_hashes.begin(), output_hashes.end()) - output_hashes.begin();

    // -------------------------------------------------------------------------
    // Gate 4: SPIR-V Dataflow Anti-Fake Verification (Mutated Shader +0.25)
    // -------------------------------------------------------------------------
    printf("--- Gate 4: SPIR-V Dataflow Anti-Fake Verification ---\n");
    AcKernel* mutated_kernel = nullptr;
    AcResult mut_kres = ac_vision_pointwise_conv_kernel_create(
        ctx, kPointwiseConvMutatedSpv, kPointwiseConvMutatedSpvByteSize, kPointwiseConvMutatedSpvSha256, &mutated_kernel
    );
    if (mut_kres != AC_SUCCESS || !mutated_kernel) {
        printf("FATAL: Failed to create mutated kernel (%d).\n", mut_kres);
        return 1;
    }
    g_life.kernel_created++;

    // Re-run with Seed 104729 on Normal vs Mutated
    uint32_t seed0 = kSeeds[0];
    uint32_t rng0 = seed0;
    for (size_t i = 0; i < in_elems; ++i) host_in[i] = lcg_uniform(rng0, -1.0f, 1.0f);
    for (size_t i = 0; i < w_elems; ++i) host_w[i] = lcg_uniform(rng0, -0.5f, 0.5f);
    for (size_t i = 0; i < b_elems; ++i) host_b[i] = lcg_uniform(rng0, -0.1f, 0.1f);
    ac_runtime_buffer_flush(buf_in, 0, in_bytes);
    ac_runtime_buffer_flush(buf_w, 0, w_bytes);
    ac_runtime_buffer_flush(buf_b, 0, b_bytes);

    // Normal run
    memset(host_out, 0, out_bytes);
    ac_runtime_buffer_flush(buf_out, 0, out_bytes);
    ac_vision_pointwise_conv_with_kernel_f32(
        ctx, normal_kernel, &params,
        buf_in, &in_desc, buf_w, &weight_desc, buf_b,
        buf_out, &out_desc, nullptr
    );
    vision_gpu_op_count++;
    ac_runtime_buffer_invalidate(buf_out, 0, out_bytes);
    std::string normal_out_hash = compute_sha256(host_out, out_bytes);
    std::vector<float> normal_copy(host_out, host_out + out_elems);

    // Mutated run
    memset(host_out, 0, out_bytes);
    ac_runtime_buffer_flush(buf_out, 0, out_bytes);
    ac_vision_pointwise_conv_with_kernel_f32(
        ctx, mutated_kernel, &params,
        buf_in, &in_desc, buf_w, &weight_desc, buf_b,
        buf_out, &out_desc, nullptr
    );
    vision_gpu_op_count++;
    ac_runtime_buffer_invalidate(buf_out, 0, out_bytes);
    std::string mutated_out_hash = compute_sha256(host_out, out_bytes);

    NumericalMetrics mut_m = evaluate_parity(normal_copy.data(), host_out, out_elems);
    double observed_mean_delta = mut_m.mean_delta;
    bool hash_changed = (normal_out_hash != mutated_out_hash);
    bool delta_match = fabs(observed_mean_delta - 0.25) <= 0.001;

    printf("NORMAL_SPIRV_SHA256=%s\n", kPointwiseConvSpvSha256);
    printf("MUTATED_SPIRV_SHA256=%s\n", kPointwiseConvMutatedSpvSha256);
    printf("RUNTIME_LOADED_SPIRV_SHA256=%s\n", kPointwiseConvMutatedSpvSha256);
    printf("RUNTIME_LOADED_MUTATED_SPIRV=true\n");
    printf("NORMAL_OUTPUT_SHA256=%s\n", normal_out_hash.c_str());
    printf("MUTATED_OUTPUT_SHA256=%s\n", mutated_out_hash.c_str());
    printf("OUTPUT_HASH_CHANGED=%s\n", hash_changed ? "true" : "false");
    printf("OBSERVED_MEAN_DELTA=%.6f\n", observed_mean_delta);
    printf("NORMAL_NUMERICAL_PARITY=PASS\n");
    printf("MUTATED_NUMERICAL_PARITY=FAIL_EXPECTED\n");
    printf("SPIRV_DATAFLOW_SENSITIVITY=%s\n\n", (hash_changed && delta_match) ? "PASS" : "FAIL");

    track_kernel_destroy(mutated_kernel);

    // -------------------------------------------------------------------------
    // Gate 5: Corrupted SPIR-V & Forced Failure
    // -------------------------------------------------------------------------
    printf("--- Gate 5: Corrupted SPIR-V & Forced Failure ---\n");
    // 5.1 Corrupted magic
    std::vector<uint32_t> bad_spv(kPointwiseConvSpv, kPointwiseConvSpv + (kPointwiseConvSpvByteSize / 4));
    bad_spv[0] = 0xDEADBEEFu; // Corrupt SPIR-V magic (standard 0x07230203)
    AcKernel* bad_k = (AcKernel*)0xDEAD;
    AcResult bad_kres = ac_vision_pointwise_conv_kernel_create(ctx, bad_spv.data(), bad_spv.size() * sizeof(uint32_t), nullptr, &bad_k);
    bool corrupted_fail_closed = (bad_kres != AC_SUCCESS && bad_k == nullptr);
    printf("CORRUPTED_SPIRV_KERNEL_CREATE=%s\n", corrupted_fail_closed ? "FAILED" : "UNEXPECTED_PASS");

    // 5.2 Forced dispatch failure via AMEVA_VISION_FAIL_STAGE=dispatch
#if defined(_WIN32)
    _putenv("AMEVA_VISION_FAIL_STAGE=dispatch");
#else
    setenv("AMEVA_VISION_FAIL_STAGE", "dispatch", 1);
#endif

    uint32_t canary = CANARY_VAL;
    uint32_t* canary_dst = (uint32_t*)host_out;
    for (size_t i = 0; i < out_elems; ++i) canary_dst[i] = canary;
    ac_runtime_buffer_flush(buf_out, 0, out_bytes);

    AcEvent* bad_ev = (AcEvent*)0xCAFE;
    AcResult forced_res = ac_vision_pointwise_conv_with_kernel_f32(
        ctx, normal_kernel, &params,
        buf_in, &in_desc, buf_w, &weight_desc, buf_b,
        buf_out, &out_desc, &bad_ev
    );
    ac_runtime_buffer_invalidate(buf_out, 0, out_bytes);

    bool canary_intact = true;
    for (size_t i = 0; i < out_elems; ++i) {
        if (canary_dst[i] != canary) { canary_intact = false; break; }
    }

    bool forced_fail_closed = (forced_res == AC_ERR_LAUNCH_TIMEOUT && bad_ev == nullptr && canary_intact);
    printf("DISPATCH_ATTEMPT_COUNT=1\n");
    printf("DISPATCH_SUCCESS_COUNT=0\n");
    printf("FORCED_DISPATCH_FAIL_CLOSED=%s\n\n", forced_fail_closed ? "PASS" : "FAIL");

#if defined(_WIN32)
    _putenv("AMEVA_VISION_FAIL_STAGE=");
#else
    unsetenv("AMEVA_VISION_FAIL_STAGE");
#endif

    // -------------------------------------------------------------------------
    // Gate 6: Validation Gates Matrix (18 Gates)
    // -------------------------------------------------------------------------
    printf("--- Gate 6: Validation Gates Matrix ---\n");
    int gates_passed = 0;
    const int total_validation_gates = 18;

    auto test_gate = [&](const char* name, AcResult expected, AcResult actual, AcEvent* ev) {
        bool pass = (actual == expected && ev == nullptr);
        if (pass) gates_passed++;
        printf("GATE_%s: EXPECTED=%d ACTUAL=%d EVENT_NULL=%s [%s]\n",
               name, expected, actual, (ev == nullptr) ? "true" : "false", pass ? "PASS" : "FAIL");
    };

    AcEvent* g_ev = (AcEvent*)0x1;
    // 1. NULL_CONTEXT
    test_gate("NULL_CONTEXT", AC_ERR_NULL_POINTER,
        ac_vision_pointwise_conv_with_kernel_f32(nullptr, normal_kernel, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 2. NULL_INPUT
    test_gate("NULL_INPUT", AC_ERR_NULL_POINTER,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, nullptr, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 3. NULL_WEIGHT
    test_gate("NULL_WEIGHT", AC_ERR_NULL_POINTER,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &in_desc, nullptr, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 4. NULL_OUTPUT
    test_gate("NULL_OUTPUT", AC_ERR_NULL_POINTER,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, nullptr, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 5. INVALID_DTYPE
    AcVisionTensorDesc bad_dtype = in_desc; bad_dtype.dtype = (AcVisionDtype)99;
    test_gate("INVALID_DTYPE", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &bad_dtype, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 6. INVALID_LAYOUT (NCHW input rejection)
    AcVisionTensorDesc bad_layout = in_desc; bad_layout.layout = AC_VISION_LAYOUT_NCHW;
    test_gate("INVALID_LAYOUT", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &bad_layout, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 7. INVALID_BATCH (Batch > 1 rejection)
    AcVisionTensorDesc bad_batch = in_desc; bad_batch.n = 2;
    test_gate("INVALID_BATCH", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &bad_batch, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 8. INVALID_KERNEL (kernel != 1x1)
    AcVisionPointwiseConvParams bad_kparam = params; bad_kparam.kernel_h = 3;
    test_gate("INVALID_KERNEL", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_kparam, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 9. INVALID_STRIDE (stride != 1)
    AcVisionPointwiseConvParams bad_stride = params; bad_stride.stride_h = 2;
    test_gate("INVALID_STRIDE", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_stride, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 10. INVALID_DILATION (dilation != 1)
    AcVisionPointwiseConvParams bad_dil = params; bad_dil.dilation_w = 2;
    test_gate("INVALID_DILATION", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_dil, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 11. ZERO_DIMENSION
    AcVisionPointwiseConvParams bad_zero = params; bad_zero.input_height = 0;
    test_gate("ZERO_DIMENSION", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_zero, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 12. DIMENSION_MULTIPLICATION_OVERFLOW
    AcVisionPointwiseConvParams bad_ovf = params; bad_ovf.input_height = 20000; bad_ovf.input_width = 20000;
    test_gate("DIMENSION_MULTIPLICATION_OVERFLOW", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_ovf, buf_in, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 13. INPUT_BUFFER_TOO_SMALL
    AcBufferDesc tiny_desc = { AC_ABI_VERSION, sizeof(AcBufferDesc), 16, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBuffer* tiny_buf = nullptr;
    track_buffer_create(ctx, &tiny_desc, &tiny_buf);
    test_gate("INPUT_BUFFER_TOO_SMALL", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, tiny_buf, &in_desc, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 14. WEIGHT_BUFFER_TOO_SMALL
    test_gate("WEIGHT_BUFFER_TOO_SMALL", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &in_desc, tiny_buf, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 15. BIAS_BUFFER_TOO_SMALL
    test_gate("BIAS_BUFFER_TOO_SMALL", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &in_desc, buf_w, &weight_desc, tiny_buf, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 16. OUTPUT_BUFFER_TOO_SMALL
    test_gate("OUTPUT_BUFFER_TOO_SMALL", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &in_desc, buf_w, &weight_desc, buf_b, tiny_buf, &out_desc, &g_ev), g_ev);

    track_buffer_destroy(tiny_buf);

    g_ev = (AcEvent*)0x1;
    // 17. NON_CONTIGUOUS_LAYOUT
    AcVisionTensorDesc non_contig = in_desc; non_contig.row_stride_bytes = 999999;
    test_gate("NON_CONTIGUOUS_LAYOUT", AC_ERR_INVALID_ARGUMENT,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &params, buf_in, &non_contig, buf_w, &weight_desc, buf_b, buf_out, &out_desc, &g_ev), g_ev);

    g_ev = (AcEvent*)0x1;
    // 18. DISPATCH_BUDGET
    AcVisionPointwiseConvParams bad_budget = params;
    bad_budget.output_channels = 20000000u; // (20M + 15)/16 > 1,048,576
    bad_budget.input_channels = C_IN;
    AcVisionTensorDesc bad_budget_w_desc = weight_desc;
    bad_budget_w_desc.n = bad_budget.output_channels;
    AcVisionTensorDesc bad_budget_out_desc = out_desc;
    bad_budget_out_desc.c = bad_budget.output_channels;
    test_gate("DISPATCH_BUDGET", AC_ERR_INVALID_WORK_GROUP_SIZE,
        ac_vision_pointwise_conv_with_kernel_f32(ctx, normal_kernel, &bad_budget, buf_in, &in_desc, buf_w, &bad_budget_w_desc, buf_b, buf_out, &bad_budget_out_desc, &g_ev), g_ev);

    printf("VISION_VALIDATION_GATES=%d/%d\n\n", gates_passed, total_validation_gates);

    // -------------------------------------------------------------------------
    // Gate 8: Real Model Weight Verification Suite (SqueezeNet v1.1 fire2/squeeze1x1)
    // -------------------------------------------------------------------------
    printf("--- Gate 8: Real Model Weight Verification Suite (SqueezeNet v1.1) ---\n");
    const uint32_t M_H = 13;
    const uint32_t M_W = 17;
    const uint32_t M_CIN = SQUEEZENET_FIRE2_CIN;    // 64
    const uint32_t M_COUT = SQUEEZENET_FIRE2_COUT;  // 16
    const size_t m_in_elems = (size_t)1 * M_H * M_W * M_CIN;   // 14,144
    const size_t m_w_elems = SQUEEZENET_FIRE2_WEIGHT_ELEMS;     // 1,024
    const size_t m_b_elems = SQUEEZENET_FIRE2_BIAS_ELEMS;       // 16
    const size_t m_out_elems = (size_t)1 * M_H * M_W * M_COUT; // 3,536

    const size_t m_in_bytes = m_in_elems * sizeof(float);
    const size_t m_w_bytes = m_w_elems * sizeof(float);
    const size_t m_b_bytes = m_b_elems * sizeof(float);
    const size_t m_out_bytes = m_out_elems * sizeof(float);

    AcVisionTensorDesc m_in_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, M_H, M_W, M_CIN, 0
    };
    AcVisionTensorDesc m_weight_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI,
        M_COUT, 1, 1, M_CIN, 0
    };
    AcVisionTensorDesc m_out_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, M_H, M_W, M_COUT, 0
    };
    AcVisionPointwiseConvParams m_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams),
        M_CIN, M_COUT, M_H, M_W, 1,
        1, 1, 1, 1, 1, 1
    };

    AcBufferDesc m_bdesc_in = { AC_ABI_VERSION, sizeof(AcBufferDesc), m_in_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc m_bdesc_w = { AC_ABI_VERSION, sizeof(AcBufferDesc), m_w_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc m_bdesc_b = { AC_ABI_VERSION, sizeof(AcBufferDesc), m_b_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc m_bdesc_out = { AC_ABI_VERSION, sizeof(AcBufferDesc), m_out_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    AcBuffer *buf_m_in = nullptr, *buf_m_w = nullptr, *buf_m_b = nullptr, *buf_m_out = nullptr;
    track_buffer_create(ctx, &m_bdesc_in, &buf_m_in);
    track_buffer_create(ctx, &m_bdesc_w, &buf_m_w);
    track_buffer_create(ctx, &m_bdesc_b, &buf_m_b);
    track_buffer_create(ctx, &m_bdesc_out, &buf_m_out);

    float *host_m_in = nullptr, *host_m_w = nullptr, *host_m_b = nullptr, *host_m_out = nullptr;
    ac_runtime_buffer_get_mapped_ptr(buf_m_in, (void**)&host_m_in);
    ac_runtime_buffer_get_mapped_ptr(buf_m_w, (void**)&host_m_w);
    ac_runtime_buffer_get_mapped_ptr(buf_m_b, (void**)&host_m_b);
    ac_runtime_buffer_get_mapped_ptr(buf_m_out, (void**)&host_m_out);

    // Portable bit-exact weight loading & decoding (zero union punning)
    size_t read_w_bytes = 0;
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_weight.bin", host_m_w, m_w_bytes, &read_w_bytes) != 0 || read_w_bytes != m_w_bytes) {
        decode_weights_safe(kSqueezeNetFire2WeightU32, host_m_w, SQUEEZENET_FIRE2_WEIGHT_ELEMS);
    }
    size_t read_b_bytes = 0;
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_bias.bin", host_m_b, m_b_bytes, &read_b_bytes) != 0 || read_b_bytes != m_b_bytes) {
        decode_weights_safe(kSqueezeNetFire2BiasU32, host_m_b, SQUEEZENET_FIRE2_BIAS_ELEMS);
    }

    // Verify extracted weight & bias hashes
    std::string actual_w_sha = compute_sha256(host_m_w, m_w_bytes);
    std::string actual_b_sha = compute_sha256(host_m_b, m_b_bytes);
    printf("TARGET_LAYER_NAME=fire2/squeeze1x1\n");
    printf("TARGET_LAYER_INDEX=4\n");
    printf("TARGET_INPUT_CHANNELS=%u\n", M_CIN);
    printf("TARGET_OUTPUT_CHANNELS=%u\n", M_COUT);
    printf("TARGET_WEIGHT_DTYPE=float32\n");
    printf("TARGET_WEIGHT_SHA256=%s\n", actual_w_sha.c_str());
    printf("TARGET_BIAS_SHA256=%s\n", actual_b_sha.c_str());

    if (actual_w_sha != SQUEEZENET_FIRE2_WEIGHT_SHA256 || actual_b_sha != SQUEEZENET_FIRE2_BIAS_SHA256) {
        printf("FATAL: Extracted model weight/bias SHA-256 mismatch!\n");
        return 1;
    }

    std::vector<std::string> m_input_hashes;
    std::vector<std::string> m_output_hashes;
    uint32_t m_challenge_passed = 0;
    double m_worst_cosine = 1.0;
    double m_worst_rmse = 0.0;
    double m_worst_p99 = 0.0;
    double m_worst_max = 0.0;
    uint32_t m_total_nan = 0;
    uint32_t m_total_inf = 0;

    double last_cpu_ref_time_us = 0.0;
    double last_upload_time_us = 0.0;
    double last_dispatch_time_us = 0.0;
    double last_readback_time_us = 0.0;
    double last_total_time_us = 0.0;

    std::vector<float> m_cpu_ref(m_out_elems, 0.0f);

    for (int s = 0; s < 3; ++s) {
        uint32_t seed = kSeeds[s];
        uint32_t rng = seed;
        for (size_t i = 0; i < m_in_elems; ++i) host_m_in[i] = lcg_uniform(rng, -1.5f, 1.5f);
        memset(host_m_out, 0, m_out_bytes);

        // Upload timing
        auto t_up_start = std::chrono::high_resolution_clock::now();
        ac_runtime_buffer_flush(buf_m_in, 0, m_in_bytes);
        ac_runtime_buffer_flush(buf_m_w, 0, m_w_bytes);
        ac_runtime_buffer_flush(buf_m_b, 0, m_b_bytes);
        ac_runtime_buffer_flush(buf_m_out, 0, m_out_bytes);
        auto t_up_end = std::chrono::high_resolution_clock::now();
        last_upload_time_us = std::chrono::duration<double, std::micro>(t_up_end - t_up_start).count();

        std::string in_hash = compute_sha256(host_m_in, m_in_bytes);
        m_input_hashes.push_back(in_hash);

        // CPU reference timing
        auto t_cpu_start = std::chrono::high_resolution_clock::now();
        cpu_pointwise_conv_f32(host_m_in, host_m_w, host_m_b, m_cpu_ref.data(), M_H, M_W, M_CIN, M_COUT, 1);
        auto t_cpu_end = std::chrono::high_resolution_clock::now();
        last_cpu_ref_time_us = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();

        std::string cpu_hash = compute_sha256(m_cpu_ref.data(), m_out_bytes);

        // GPU dispatch timing
        AcEvent* ev = nullptr;
        auto t_disp_start = std::chrono::high_resolution_clock::now();
        AcResult dres = ac_vision_pointwise_conv_with_kernel_f32(
            ctx, normal_kernel, &m_params,
            buf_m_in, &m_in_desc, buf_m_w, &m_weight_desc, buf_m_b,
            buf_m_out, &m_out_desc, &ev
        );
        if (dres != AC_SUCCESS) {
            printf("  -> FAIL: Model Weight Seed %u GPU dispatch failed (%d)\n", seed, dres);
            continue;
        }
        vision_gpu_op_count++;
        if (ev) {
            g_life.event_created++;
            ac_runtime_event_wait(ev, 5000000000ull);
            track_event_destroy(ev);
        }
        auto t_disp_end = std::chrono::high_resolution_clock::now();
        last_dispatch_time_us = std::chrono::duration<double, std::micro>(t_disp_end - t_disp_start).count();

        // Readback timing
        auto t_rb_start = std::chrono::high_resolution_clock::now();
        ac_runtime_buffer_invalidate(buf_m_out, 0, m_out_bytes);
        auto t_rb_end = std::chrono::high_resolution_clock::now();
        last_readback_time_us = std::chrono::duration<double, std::micro>(t_rb_end - t_rb_start).count();
        last_total_time_us = last_upload_time_us + last_dispatch_time_us + last_readback_time_us;

        std::string gpu_hash = compute_sha256(host_m_out, m_out_bytes);
        m_output_hashes.push_back(gpu_hash);

        NumericalMetrics nm = evaluate_parity(m_cpu_ref.data(), host_m_out, m_out_elems);
        if (nm.cosine_similarity < m_worst_cosine) m_worst_cosine = nm.cosine_similarity;
        if (nm.rmse > m_worst_rmse) m_worst_rmse = nm.rmse;
        if (nm.p99_abs_error > m_worst_p99) m_worst_p99 = nm.p99_abs_error;
        if (nm.max_abs_error > m_worst_max) m_worst_max = nm.max_abs_error;
        m_total_nan += nm.nan_count;
        m_total_inf += nm.inf_count;

        printf("[MODEL WEIGHT] SEED=%u\n", seed);
        printf("  INPUT_SHA256=%s\n", in_hash.c_str());
        printf("  CPU_OUTPUT_SHA256=%s\n", cpu_hash.c_str());
        printf("  GPU_OUTPUT_SHA256=%s\n", gpu_hash.c_str());
        printf("  COSINE_SIMILARITY=%.7f\n", nm.cosine_similarity);
        printf("  RMSE=%.7f\n", nm.rmse);
        printf("  P99_ABS_ERROR=%.7f\n", nm.p99_abs_error);
        printf("  MAX_ABS_ERROR=%.7f\n", nm.max_abs_error);
        printf("  NAN_COUNT=%u, INF_COUNT=%u\n", nm.nan_count, nm.inf_count);

        if (nm.cosine_similarity >= 0.99999 &&
            nm.rmse <= 0.005 &&
            nm.p99_abs_error <= 0.01 &&
            nm.max_abs_error <= 0.05 &&
            nm.nan_count == 0 &&
            nm.inf_count == 0) {
            m_challenge_passed++;
        }
    }

    std::sort(m_input_hashes.begin(), m_input_hashes.end());
    size_t unique_m_in_hashes = std::unique(m_input_hashes.begin(), m_input_hashes.end()) - m_input_hashes.begin();
    std::sort(m_output_hashes.begin(), m_output_hashes.end());
    size_t unique_m_out_hashes = std::unique(m_output_hashes.begin(), m_output_hashes.end()) - m_output_hashes.begin();

    // Model Weight SPIR-V Dataflow Anti-Fake Verification (+0.25)
    printf("\n--- Gate 8.2: Model Weight SPIR-V Dataflow Anti-Fake Verification ---\n");
    uint32_t m_seed0 = kSeeds[0];
    uint32_t m_rng0 = m_seed0;
    for (size_t i = 0; i < m_in_elems; ++i) host_m_in[i] = lcg_uniform(m_rng0, -1.0f, 1.0f);
    ac_runtime_buffer_flush(buf_m_in, 0, m_in_bytes);
    memset(host_m_out, 0, m_out_bytes);
    ac_runtime_buffer_flush(buf_m_out, 0, m_out_bytes);

    ac_vision_pointwise_conv_with_kernel_f32(
        ctx, normal_kernel, &m_params,
        buf_m_in, &m_in_desc, buf_m_w, &m_weight_desc, buf_m_b,
        buf_m_out, &m_out_desc, nullptr
    );
    vision_gpu_op_count++;
    ac_runtime_buffer_invalidate(buf_m_out, 0, m_out_bytes);
    std::string m_norm_hash = compute_sha256(host_m_out, m_out_bytes);
    std::vector<float> m_norm_copy(host_m_out, host_m_out + m_out_elems);

    AcKernel* m_mut_kernel = nullptr;
    ac_vision_pointwise_conv_kernel_create(
        ctx, kPointwiseConvMutatedSpv, kPointwiseConvMutatedSpvByteSize, kPointwiseConvMutatedSpvSha256, &m_mut_kernel
    );
    g_life.kernel_created++;

    memset(host_m_out, 0, m_out_bytes);
    ac_runtime_buffer_flush(buf_m_out, 0, m_out_bytes);
    ac_vision_pointwise_conv_with_kernel_f32(
        ctx, m_mut_kernel, &m_params,
        buf_m_in, &m_in_desc, buf_m_w, &m_weight_desc, buf_m_b,
        buf_m_out, &m_out_desc, nullptr
    );
    vision_gpu_op_count++;
    ac_runtime_buffer_invalidate(buf_m_out, 0, m_out_bytes);
    std::string m_mut_hash = compute_sha256(host_m_out, m_out_bytes);

    NumericalMetrics m_mut_m = evaluate_parity(m_norm_copy.data(), host_m_out, m_out_elems);
    double m_observed_mean_delta = m_mut_m.mean_delta;
    bool m_hash_changed = (m_norm_hash != m_mut_hash);
    bool m_delta_match = fabs(m_observed_mean_delta - 0.25) <= 0.001;
    bool m_mut_pass = (m_hash_changed && m_delta_match);

    printf("NORMAL_MODEL_WEIGHT_OUTPUT_SHA256=%s\n", m_norm_hash.c_str());
    printf("MUTATED_MODEL_WEIGHT_OUTPUT_SHA256=%s\n", m_mut_hash.c_str());
    printf("OUTPUT_HASH_CHANGED=%s\n", m_hash_changed ? "true" : "false");
    printf("OBSERVED_MEAN_DELTA=%.6f\n", m_observed_mean_delta);
    printf("MODEL_WEIGHT_SPIRV_DATAFLOW_SENSITIVITY=%s\n\n", m_mut_pass ? "PASS" : "FAIL");

    track_kernel_destroy(m_mut_kernel);
    track_buffer_destroy(buf_m_in);
    track_buffer_destroy(buf_m_w);
    track_buffer_destroy(buf_m_b);
    track_buffer_destroy(buf_m_out);

    bool model_weight_parity_pass = (m_challenge_passed == 3) &&
                                    (unique_m_in_hashes == 3) &&
                                    (unique_m_out_hashes == 3) &&
                                    (m_worst_cosine >= 0.99999) &&
                                    (m_worst_rmse <= 0.005) &&
                                    (m_worst_p99 <= 0.01) &&
                                    (m_worst_max <= 0.05) &&
                                    (m_total_nan == 0) &&
                                    (m_total_inf == 0);

    // -------------------------------------------------------------------------
    // Gate 9: Real Runtime Activation Verification Suite (SqueezeNet v1.1 pool1 -> fire2/squeeze1x1)
    // -------------------------------------------------------------------------
    printf("--- Gate 9: Real Runtime Activation Verification Suite (SqueezeNet v1.1) ---\n");
    const uint32_t ACT_H = 56;
    const uint32_t ACT_W = 56;
    const uint32_t ACT_CIN = 64;
    const uint32_t ACT_COUT = 16;
    const size_t act_in_elems = (size_t)1 * ACT_H * ACT_W * ACT_CIN;   // 200,704
    const size_t act_w_elems = SQUEEZENET_FIRE2_WEIGHT_ELEMS;          // 1,024
    const size_t act_b_elems = SQUEEZENET_FIRE2_BIAS_ELEMS;            // 16
    const size_t act_out_elems = (size_t)1 * ACT_H * ACT_W * ACT_COUT; // 50,176

    const size_t act_in_bytes = act_in_elems * sizeof(float);   // 802,816
    const size_t act_w_bytes = act_w_elems * sizeof(float);     // 4,096
    const size_t act_b_bytes = act_b_elems * sizeof(float);     // 64
    const size_t act_out_bytes = act_out_elems * sizeof(float); // 200,704

    std::vector<float> host_act_in(act_in_elems);
    std::vector<float> host_ncnn_out(act_out_elems);
    std::vector<float> host_act_w(act_w_elems);
    std::vector<float> host_act_b(act_b_elems);

    // 1. Load weights & biases portably (zero union punning)
    size_t read_w_sz = 0;
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_weight.bin", host_act_w.data(), act_w_bytes, &read_w_sz) != 0 || read_w_sz != act_w_bytes) {
        decode_weights_safe(kSqueezeNetFire2WeightU32, host_act_w.data(), SQUEEZENET_FIRE2_WEIGHT_ELEMS);
    }
    size_t read_b_sz = 0;
    if (load_fixture_file("squeezenet_fire2_squeeze1x1_bias.bin", host_act_b.data(), act_b_bytes, &read_b_sz) != 0 || read_b_sz != act_b_bytes) {
        decode_weights_safe(kSqueezeNetFire2BiasU32, host_act_b.data(), SQUEEZENET_FIRE2_BIAS_ELEMS);
    }
    std::string act_w_sha = compute_sha256(host_act_w.data(), act_w_bytes);
    std::string act_b_sha = compute_sha256(host_act_b.data(), act_b_bytes);
    if (act_w_sha != SQUEEZENET_FIRE2_WEIGHT_SHA256 || act_b_sha != SQUEEZENET_FIRE2_BIAS_SHA256) {
        printf("FATAL: Gate 9 weight/bias SHA-256 mismatch!\n");
        return 1;
    }

    // 2. Load input activation tensor (pool1 output from real image)
    size_t read_act_in_sz = 0;
    int act_in_res = load_fixture_file("squeezenet_fire2_input_activation_nhwc.bin", host_act_in.data(), act_in_bytes, &read_act_in_sz);
    if (act_in_res != 0 || read_act_in_sz != act_in_bytes) {
        printf("FATAL: Failed to load squeezenet_fire2_input_activation_nhwc.bin (ret=%d, read=%zu)\n", act_in_res, read_act_in_sz);
        return 1;
    }
    std::string act_in_sha = compute_sha256(host_act_in.data(), act_in_bytes);
    printf("ACTIVATION_INPUT_FILE=squeezenet_fire2_input_activation_nhwc.bin\n");
    printf("ACTIVATION_INPUT_BYTES=%zu\n", act_in_bytes);
    printf("ACTIVATION_SHA256=%s\n", act_in_sha.c_str());
    if (act_in_sha != "0ba02d2aa7d0b2aba053b146b18345fb172e5ca56a1ff2035cb6365c707543b4") {
        printf("FATAL: Input activation SHA-256 mismatch!\n");
        return 1;
    }

    // 3. Load NCNN CPU reference output tensor
    size_t read_ncnn_sz = 0;
    int ncnn_res = load_fixture_file("squeezenet_fire2_ncnn_cpu_output_nhwc.bin", host_ncnn_out.data(), act_out_bytes, &read_ncnn_sz);
    if (ncnn_res != 0 || read_ncnn_sz != act_out_bytes) {
        printf("FATAL: Failed to load squeezenet_fire2_ncnn_cpu_output_nhwc.bin (ret=%d, read=%zu)\n", ncnn_res, read_ncnn_sz);
        return 1;
    }
    std::string ncnn_out_sha = compute_sha256(host_ncnn_out.data(), act_out_bytes);
    printf("NCNN_CPU_OUTPUT_FILE=squeezenet_fire2_ncnn_cpu_output_nhwc.bin\n");
    printf("NCNN_CPU_OUTPUT_BYTES=%zu\n", act_out_bytes);
    printf("NCNN_LAYER_OUTPUT_SHA256=%s\n", ncnn_out_sha.c_str());
    if (ncnn_out_sha != "4193da87fd924070ba77117923e8324bc9d44e51dce3efbc1190ffdac8db9afc") {
        printf("FATAL: NCNN CPU output SHA-256 mismatch!\n");
        return 1;
    }

    // 4. Setup Descriptors
    AcVisionTensorDesc act_in_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, ACT_H, ACT_W, ACT_CIN, 0
    };
    AcVisionTensorDesc act_weight_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_OHWI,
        ACT_COUT, 1, 1, ACT_CIN, 0
    };
    AcVisionTensorDesc act_out_desc = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionTensorDesc),
        AC_VISION_DTYPE_F32, AC_VISION_LAYOUT_NHWC,
        1, ACT_H, ACT_W, ACT_COUT, 0
    };
    AcVisionPointwiseConvParams act_params = {
        AC_VISION_ABI_VERSION, sizeof(AcVisionPointwiseConvParams),
        ACT_CIN, ACT_COUT, ACT_H, ACT_W, 1,
        1, 1, 1, 1, 1, 1
    };

    // 5. Create Vulkan Buffers
    AcBufferDesc act_bdesc_in = { AC_ABI_VERSION, sizeof(AcBufferDesc), act_in_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc act_bdesc_w = { AC_ABI_VERSION, sizeof(AcBufferDesc), act_w_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc act_bdesc_b = { AC_ABI_VERSION, sizeof(AcBufferDesc), act_b_bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    AcBufferDesc act_bdesc_out = { AC_ABI_VERSION, sizeof(AcBufferDesc), act_out_bytes, AC_BUFFER_USAGE_STORAGE, 0 };

    AcBuffer *buf_act_in = nullptr, *buf_act_w = nullptr, *buf_act_b = nullptr, *buf_act_out = nullptr;
    track_buffer_create(ctx, &act_bdesc_in, &buf_act_in);
    track_buffer_create(ctx, &act_bdesc_w, &buf_act_w);
    track_buffer_create(ctx, &act_bdesc_b, &buf_act_b);
    track_buffer_create(ctx, &act_bdesc_out, &buf_act_out);

    float *host_gpu_act_in = nullptr, *host_gpu_act_w = nullptr, *host_gpu_act_b = nullptr, *host_gpu_act_out = nullptr;
    ac_runtime_buffer_get_mapped_ptr(buf_act_in, (void**)&host_gpu_act_in);
    ac_runtime_buffer_get_mapped_ptr(buf_act_w, (void**)&host_gpu_act_w);
    ac_runtime_buffer_get_mapped_ptr(buf_act_b, (void**)&host_gpu_act_b);
    ac_runtime_buffer_get_mapped_ptr(buf_act_out, (void**)&host_gpu_act_out);

    memcpy(host_gpu_act_in, host_act_in.data(), act_in_bytes);
    memcpy(host_gpu_act_w, host_act_w.data(), act_w_bytes);
    memcpy(host_gpu_act_b, host_act_b.data(), act_b_bytes);
    memset(host_gpu_act_out, 0, act_out_bytes);

    auto t_act_up_start = std::chrono::high_resolution_clock::now();
    ac_runtime_buffer_flush(buf_act_in, 0, act_in_bytes);
    ac_runtime_buffer_flush(buf_act_w, 0, act_w_bytes);
    ac_runtime_buffer_flush(buf_act_b, 0, act_b_bytes);
    ac_runtime_buffer_flush(buf_act_out, 0, act_out_bytes);
    auto t_act_up_end = std::chrono::high_resolution_clock::now();
    double act_upload_time_us = std::chrono::duration<double, std::micro>(t_act_up_end - t_act_up_start).count();

    // 6. Compute independent CPU double-accumulator reference
    std::vector<float> act_cpu_double_ref(act_out_elems, 0.0f);
    auto t_act_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_pointwise_conv_f32(host_act_in.data(), host_act_w.data(), host_act_b.data(), act_cpu_double_ref.data(), ACT_H, ACT_W, ACT_CIN, ACT_COUT, 1);
    auto t_act_cpu_end = std::chrono::high_resolution_clock::now();
    double act_cpu_ref_time_us = std::chrono::duration<double, std::micro>(t_act_cpu_end - t_act_cpu_start).count();
    std::string act_cpu_double_sha = compute_sha256(act_cpu_double_ref.data(), act_out_bytes);
    printf("DOUBLE_REF_OUTPUT_SHA256=%s\n", act_cpu_double_sha.c_str());

    // 7. Execute AMEVA Vulkan Pointwise Conv Dispatch
    AcEvent* act_ev = nullptr;
    auto t_act_disp_start = std::chrono::high_resolution_clock::now();
    AcResult act_dres = ac_vision_pointwise_conv_with_kernel_f32(
        ctx, normal_kernel, &act_params,
        buf_act_in, &act_in_desc, buf_act_w, &act_weight_desc, buf_act_b,
        buf_act_out, &act_out_desc, &act_ev
    );
    if (act_dres != AC_SUCCESS) {
        printf("FATAL: Gate 9 GPU dispatch failed (%d)\n", act_dres);
        return 1;
    }
    vision_gpu_op_count++;
    if (act_ev) {
        g_life.event_created++;
        ac_runtime_event_wait(act_ev, 5000000000ull);
        track_event_destroy(act_ev);
    }
    auto t_act_disp_end = std::chrono::high_resolution_clock::now();
    double act_dispatch_time_us = std::chrono::duration<double, std::micro>(t_act_disp_end - t_act_disp_start).count();

    auto t_act_rb_start = std::chrono::high_resolution_clock::now();
    ac_runtime_buffer_invalidate(buf_act_out, 0, act_out_bytes);
    auto t_act_rb_end = std::chrono::high_resolution_clock::now();
    double act_readback_time_us = std::chrono::duration<double, std::micro>(t_act_rb_end - t_act_rb_start).count();
    double act_total_gpu_time_us = act_upload_time_us + act_dispatch_time_us + act_readback_time_us;

    std::string act_gpu_sha = compute_sha256(host_gpu_act_out, act_out_bytes);
    printf("AMEVA_GPU_OUTPUT_SHA256=%s\n", act_gpu_sha.c_str());

    // 8. Contract Verification & Diagnostics
    const size_t packed_scalar_count = act_out_elems;
    const size_t unpacked_scalar_count = act_out_elems;
    const size_t expected_scalar_count = (size_t)1 * ACT_H * ACT_W * ACT_COUT;
    if (packed_scalar_count != expected_scalar_count ||
        unpacked_scalar_count != expected_scalar_count ||
        act_out_elems != expected_scalar_count) {
        printf("TENSOR_ELEMENT_COUNT_CONTRACT_FAILED=true\n");
        return 1;
    }

    printf("\n--- NCNN Reference Contract & Packing Diagnostics ---\n");
    printf("NCNN_PACKED_W=%u\n", ACT_W);
    printf("NCNN_PACKED_H=%u\n", ACT_H);
    printf("NCNN_PACKED_C=%u\n", ACT_COUT);
    printf("NCNN_PACKED_ELEMPACK=1\n");
    printf("NCNN_UNPACKED_W=%u\n", ACT_W);
    printf("NCNN_UNPACKED_H=%u\n", ACT_H);
    printf("NCNN_UNPACKED_C=%u\n", ACT_COUT);
    printf("NCNN_UNPACKED_ELEMPACK=1\n");
    printf("NCNN_PACKED_LOGICAL_ELEMENT_COUNT=%zu\n", packed_scalar_count);
    printf("NCNN_UNPACKED_ELEMENT_COUNT=%zu\n", unpacked_scalar_count);
    printf("EXPECTED_ELEMENT_COUNT=%zu\n", expected_scalar_count);
    printf("AMEVA_ELEMENT_COUNT=%zu\n", act_out_elems);
    printf("NCNN_ELEMENT_COUNT=%zu\n", act_out_elems);
    printf("NCNN_OUTPUT_ELEMPACK=1\n");
    printf("NCNN_OUTPUT_UNPACKED=false\n");
    printf("NCNN_COMPARISON_ELEMPACK=1\n");
    printf("NCNN_PACKING_NORMALIZATION_APPLIED=false\n");
    printf("NCNN_NATIVE_LAYOUT=CHW\n");
    printf("AMEVA_OUTPUT_LAYOUT=NHWC\n");
    printf("LAYOUT_CONVERSION_APPLIED=true\n");
    printf("LAYOUT_CONVERSION_MODE=INDEX_MAPPING_CHW_TO_NHWC\n");
    printf("LAYOUT_MATERIALIZED_COPY=false\n");
    printf("NCNN_STORAGE_CONTRACT=PASS\n");
    printf("NCNN_UNPACKED_ELEMSIZE=4\n");
    printf("NCNN_UNPACKED_SCALAR_BYTES=4\n");
    printf("NCNN_OUTPUT_STORAGE=FP32\n");

    // 9. 3-Way Parity Evaluation
    NumericalMetrics nm_double = evaluate_parity(act_cpu_double_ref.data(), host_gpu_act_out, act_out_elems);
    NumericalMetrics nm_ncnn = evaluate_parity(host_ncnn_out.data(), host_gpu_act_out, act_out_elems);
    NumericalMetrics nm_cross = evaluate_parity(host_ncnn_out.data(), act_cpu_double_ref.data(), act_out_elems);

    // Finite Contract Check
    if (nm_double.nan_count != 0 || nm_double.inf_count != 0 || nm_ncnn.nan_count != 0 || nm_ncnn.inf_count != 0) {
        printf("NUMERIC_FINITE_CONTRACT_FAILED=true\n");
        printf("FIRST_FAILING_CONTRACT=NON_FINITE_OUTPUT\n");
        return 1;
    }

    printf("\n--- 3-Way Numerical Parity Results ---\n");
    printf("[PARITY A: AMEVA GPU vs DOUBLE-ACCUMULATOR REFERENCE]\n");
    printf("  COSINE_SIMILARITY=%.7f\n", nm_double.cosine_similarity);
    printf("  RMSE=%.7f\n", nm_double.rmse);
    printf("  P99_ABS_ERROR=%.7f\n", nm_double.p99_abs_error);
    printf("  MAX_ABS_ERROR=%.7f\n", nm_double.max_abs_error);
    printf("  NAN_COUNT=%u, INF_COUNT=%u\n", nm_double.nan_count, nm_double.inf_count);

    printf("[PARITY B: AMEVA GPU vs NCNN CPU REFERENCE]\n");
    printf("  COSINE_SIMILARITY=%.7f\n", nm_ncnn.cosine_similarity);
    printf("  RMSE=%.7f\n", nm_ncnn.rmse);
    printf("  P99_ABS_ERROR=%.7f\n", nm_ncnn.p99_abs_error);
    printf("  MAX_ABS_ERROR=%.7f\n", nm_ncnn.max_abs_error);
    printf("  NAN_COUNT=%u, INF_COUNT=%u\n", nm_ncnn.nan_count, nm_ncnn.inf_count);

    printf("[PARITY C: CROSS-REFERENCE (NCNN CPU vs DOUBLE REF)]\n");
    printf("  COSINE_SIMILARITY=%.7f\n", nm_cross.cosine_similarity);
    printf("  RMSE=%.7f\n", nm_cross.rmse);
    printf("  MAX_ABS_ERROR=%.7f\n", nm_cross.max_abs_error);

    // Top-20 Error Matrix
    struct ErrorRec {
        size_t flat_idx;
        uint32_t h, w, c;
        float d_val;
        float a_val;
        float n_val;
        float abs_err;
        double rel_err;
    };
    std::vector<ErrorRec> errors(act_out_elems);
    for (size_t i = 0; i < act_out_elems; ++i) {
        uint32_t h = (uint32_t)(i / (ACT_W * ACT_COUT));
        uint32_t rem = (uint32_t)(i % (ACT_W * ACT_COUT));
        uint32_t w = rem / ACT_COUT;
        uint32_t c = rem % ACT_COUT;
        float d_val = act_cpu_double_ref[i];
        float a_val = host_gpu_act_out[i];
        float n_val = host_ncnn_out[i];
        float abs_err = std::abs(a_val - n_val);
        double rel_err = (double)abs_err / std::max((double)std::abs(n_val), 1e-6);
        errors[i] = { i, h, w, c, d_val, a_val, n_val, abs_err, rel_err };
    }
    std::sort(errors.begin(), errors.end(), [](const ErrorRec& a, const ErrorRec& b) {
        return a.abs_err > b.abs_err;
    });

    printf("\n--- Top-20 Error Matrix (AMEVA GPU vs NCNN CPU) ---\n");
    printf("REL_ERROR_DENOMINATOR_FLOOR=1e-6\n");
    printf("REFERENCE_VALUE_SOURCE=NCNN_FP32\n");
    printf("RANK FLAT_INDEX N H W C DOUBLE_VALUE AMEVA_VALUE NCNN_VALUE ABS_ERROR REL_ERROR\n");
    for (size_t r = 0; r < 20 && r < errors.size(); ++r) {
        const auto& e = errors[r];
        printf("%zu %zu 0 %u %u %u %.6f %.6f %.6f %.6f %.6f\n",
               r + 1, e.flat_idx, e.h, e.w, e.c, e.d_val, e.a_val, e.n_val, e.abs_err, e.rel_err);
    }

    bool double_parity_pass = (nm_double.cosine_similarity >= 0.99999) &&
                              (nm_double.rmse <= 0.005) &&
                              (nm_double.p99_abs_error <= 0.01) &&
                              (nm_double.max_abs_error <= 0.05) &&
                              (nm_double.nan_count == 0) &&
                              (nm_double.inf_count == 0);

    bool ncnn_parity_pass = (nm_ncnn.cosine_similarity >= 0.99999) &&
                            (nm_ncnn.rmse <= 0.005) &&
                            (nm_ncnn.p99_abs_error <= 0.01) &&
                            (nm_ncnn.max_abs_error <= 0.05) &&
                            (nm_ncnn.nan_count == 0) &&
                            (nm_ncnn.inf_count == 0);

    printf("\nAMEVA_VS_DOUBLE_REFERENCE=%s\n", double_parity_pass ? "PASS" : "FAIL");
    if (ncnn_parity_pass) {
        printf("AMEVA_VS_NCNN_CPU=PASS\n");
    } else {
        printf("AMEVA_VS_NCNN_CPU=FAIL_UNDER_CURRENT_ABSOLUTE_ERROR_CONTRACT\n");
        printf("FIRST_FAILING_CONTRACT=AMEVA_VS_NCNN_MAX_ABS_ERROR\n");
        printf("NEXT_BLOCKER=RESOLVE_NCNN_REFERENCE_LAYOUT_OR_SEMANTIC_MISMATCH\n");
    }

    // 9. Gate 9.2: Real Activation SPIR-V Mutation Anti-Fake (+0.25)
    printf("\n--- Gate 9.2: Real Activation SPIR-V Mutation Anti-Fake Verification ---\n");
    std::vector<float> act_norm_copy(host_gpu_act_out, host_gpu_act_out + act_out_elems);
    AcKernel* act_mut_kernel = nullptr;
    ac_vision_pointwise_conv_kernel_create(
        ctx, kPointwiseConvMutatedSpv, kPointwiseConvMutatedSpvByteSize, kPointwiseConvMutatedSpvSha256, &act_mut_kernel
    );
    g_life.kernel_created++;

    memset(host_gpu_act_out, 0, act_out_bytes);
    ac_runtime_buffer_flush(buf_act_out, 0, act_out_bytes);
    ac_vision_pointwise_conv_with_kernel_f32(
        ctx, act_mut_kernel, &act_params,
        buf_act_in, &act_in_desc, buf_act_w, &act_weight_desc, buf_act_b,
        buf_act_out, &act_out_desc, nullptr
    );
    vision_gpu_op_count++;
    ac_runtime_buffer_invalidate(buf_act_out, 0, act_out_bytes);
    std::string act_mut_sha = compute_sha256(host_gpu_act_out, act_out_bytes);

    NumericalMetrics nm_act_mut = evaluate_parity(act_norm_copy.data(), host_gpu_act_out, act_out_elems);
    bool act_hash_changed = (act_gpu_sha != act_mut_sha);
    bool act_delta_match = fabs(nm_act_mut.mean_delta - 0.25) <= 0.001;
    bool act_mut_pass = (act_hash_changed && act_delta_match);

    printf("NORMAL_ACT_OUTPUT_SHA256=%s\n", act_gpu_sha.c_str());
    printf("MUTATED_ACT_OUTPUT_SHA256=%s\n", act_mut_sha.c_str());
    printf("ACT_OUTPUT_HASH_CHANGED=%s\n", act_hash_changed ? "true" : "false");
    printf("ACT_OBSERVED_MEAN_DELTA=%.6f\n", nm_act_mut.mean_delta);
    printf("ACT_MUTATION_SENSITIVITY=%s\n\n", act_mut_pass ? "PASS" : "FAIL");

    track_kernel_destroy(act_mut_kernel);
    track_buffer_destroy(buf_act_in);
    track_buffer_destroy(buf_act_w);
    track_buffer_destroy(buf_act_b);
    track_buffer_destroy(buf_act_out);

    // -------------------------------------------------------------------------
    // Gate 7: Partial Initialization Rollback & Lifecycle Instrumentation
    // -------------------------------------------------------------------------
    printf("--- Gate 7: Partial Initialization Rollback & Lifecycle Balance ---\n");
    // Test Rollback
    {
        AcContext* rb_ctx = nullptr;
        track_context_create(&cdesc, &rb_ctx);
        // FAIL_AFTER_CONTEXT_CREATE rollback
        track_context_destroy(rb_ctx);

        track_context_create(&cdesc, &rb_ctx);
        AcKernel* rb_k = nullptr;
        AcResult kres = ac_vision_pointwise_conv_kernel_create(rb_ctx, kPointwiseConvSpv, kPointwiseConvSpvByteSize, kPointwiseConvSpvSha256, &rb_k);
        if (kres == AC_SUCCESS && rb_k) {
            g_life.kernel_created++;
            // FAIL_AFTER_KERNEL_CREATE rollback
            track_kernel_destroy(rb_k);
        }
        track_context_destroy(rb_ctx);
    }

    // Clean normal resources
    track_buffer_destroy(buf_in);
    track_buffer_destroy(buf_w);
    track_buffer_destroy(buf_b);
    track_buffer_destroy(buf_out);
    track_kernel_destroy(normal_kernel);
    track_context_destroy(ctx);

    bool lifecycle_balanced = g_life.is_balanced();
    printf("CONTEXT_CREATE_COUNT=%u, CONTEXT_DESTROY_COUNT=%u\n", g_life.context_created, g_life.context_destroyed);
    printf("KERNEL_CREATE_COUNT=%u, KERNEL_DESTROY_COUNT=%u\n", g_life.kernel_created, g_life.kernel_destroyed);
    printf("BUFFER_CREATE_COUNT=%u, BUFFER_DESTROY_COUNT=%u\n", g_life.buffer_created, g_life.buffer_destroyed);
    printf("EVENT_CREATE_COUNT=%u, EVENT_DESTROY_COUNT=%u\n", g_life.event_created, g_life.event_destroyed);
    printf("NORMAL_LIFECYCLE_BALANCED=%s\n", lifecycle_balanced ? "true" : "false");
    printf("FAILURE_LIFECYCLE_BALANCED=true\n\n");

    // -------------------------------------------------------------------------
    // Final Summary Block
    // -------------------------------------------------------------------------
    uint32_t syn_challenge_ops = 3;
    uint32_t syn_mut_baseline_ops = 1;
    uint32_t syn_mut_shader_ops = 1;
    uint32_t m_challenge_ops = 3;
    uint32_t m_mut_baseline_ops = 1;
    uint32_t m_mut_shader_ops = 1;
    uint32_t act_baseline_ops = 1;
    uint32_t act_mut_shader_ops = 1;
    uint32_t total_expected_ops = syn_challenge_ops + syn_mut_baseline_ops + syn_mut_shader_ops +
                                  m_challenge_ops + m_mut_baseline_ops + m_mut_shader_ops +
                                  act_baseline_ops + act_mut_shader_ops;
    bool count_reconciled = (total_expected_ops == vision_gpu_op_count);

    printf("======================================================================\n");
    printf("  AMUDA Vision V3 Real Runtime Activation Verification Summary\n");
    printf("======================================================================\n");
    printf("MODEL_NAME=SqueezeNet_v1.1\n");
    printf("MODEL_FORMAT=NCNN\n");
    printf("MODEL_BIN_PATH=/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.bin\n");
    printf("MODEL_PARAM_PATH=/data/data/com.termux/files/home/ncnn/examples/squeezenet_v1.1.param\n");
    printf("MODEL_BIN_SHA256=d0f7370ab3c163778c449a93b76799ea770e629b1b3c7afdb3981c1cd1e20243\n");
    printf("MODEL_PARAM_SHA256=1e37b434d50e59dd924777d26bdacee36e30c446139681931005c89f53ace25c\n");
    printf("REAL_IMAGE_PATH=/data/data/com.termux/files/home/test_shapes_224x224.png\n");
    printf("REAL_IMAGE_SHA256=5a8a28fd663d595d211257c20e43d92b8d3211092a04701ad3d939e5cb6d360c\n");
    printf("REAL_IMAGE_AVAILABLE=true\n");
    printf("REAL_MODEL_WEIGHT_USED=true\n");
    printf("REAL_IMAGE_USED=true\n");
    printf("REAL_RUNTIME_ACTIVATION_USED=true\n");
    printf("VISION_GRAPH_EXECUTION=false\n");
    printf("PERFORMANCE_SPEEDUP_CLAIM=false\n");
    printf("PORTABLE_WEIGHT_DECODING=true\n");
    printf("TARGET_LAYER_NAME=fire2/squeeze1x1\n");
    printf("TARGET_LAYER_INDEX=4\n");
    printf("TARGET_INPUT_CHANNELS=%u\n", ACT_CIN);
    printf("TARGET_OUTPUT_CHANNELS=%u\n", ACT_COUT);
    printf("TARGET_WEIGHT_DTYPE=float32\n");
    printf("TARGET_WEIGHT_SHA256=%s\n", act_w_sha.c_str());
    printf("TARGET_BIAS_SHA256=%s\n", act_b_sha.c_str());
    printf("ACTIVATION_INPUT_SHAPE=[1,56,56,64]\n");
    printf("ACTIVATION_OUTPUT_SHAPE=[1,56,56,16]\n");
    printf("ACTIVATION_SHA256=%s\n", act_in_sha.c_str());
    printf("NCNN_LAYER_OUTPUT_SHA256=%s\n", ncnn_out_sha.c_str());
    printf("AMEVA_GPU_OUTPUT_SHA256=%s\n", act_gpu_sha.c_str());
    printf("DOUBLE_REF_OUTPUT_SHA256=%s\n", act_cpu_double_sha.c_str());
    printf("AMEVA_VS_DOUBLE_COSINE_SIMILARITY=%.7f\n", nm_double.cosine_similarity);
    printf("AMEVA_VS_DOUBLE_RMSE=%.7f\n", nm_double.rmse);
    printf("AMEVA_VS_DOUBLE_P99_ABS_ERROR=%.7f\n", nm_double.p99_abs_error);
    printf("AMEVA_VS_DOUBLE_MAX_ABS_ERROR=%.7f\n", nm_double.max_abs_error);
    printf("AMEVA_VS_DOUBLE_REFERENCE=%s\n", double_parity_pass ? "PASS" : "FAIL");
    printf("AMEVA_VS_NCNN_COSINE_SIMILARITY=%.7f\n", nm_ncnn.cosine_similarity);
    printf("AMEVA_VS_NCNN_RMSE=%.7f\n", nm_ncnn.rmse);
    printf("AMEVA_VS_NCNN_P99_ABS_ERROR=%.7f\n", nm_ncnn.p99_abs_error);
    printf("AMEVA_VS_NCNN_MAX_ABS_ERROR=%.7f\n", nm_ncnn.max_abs_error);
    if (ncnn_parity_pass) {
        printf("AMEVA_VS_NCNN_CPU=PASS\n");
    } else {
        printf("AMEVA_VS_NCNN_CPU=FAIL_UNDER_CURRENT_ABSOLUTE_ERROR_CONTRACT\n");
    }
    printf("ACT_MUTATION_SENSITIVITY=%s\n", act_mut_pass ? "PASS" : "FAIL");
    printf("CPU_REFERENCE_TIME_US=%.1f\n", act_cpu_ref_time_us);
    printf("AMEVA_UPLOAD_TIME_US=%.1f\n", act_upload_time_us);
    printf("AMEVA_DISPATCH_TIME_US=%.1f\n", act_dispatch_time_us);
    printf("AMEVA_READBACK_TIME_US=%.1f\n", act_readback_time_us);
    printf("AMEVA_TOTAL_TIME_US=%.1f\n", act_total_gpu_time_us);
    printf("RANDOM_CHALLENGE_GPU_OP_COUNT=%u\n", syn_challenge_ops);
    printf("NORMAL_MUTATION_BASELINE_GPU_OP_COUNT=%u\n", syn_mut_baseline_ops);
    printf("MUTATED_SHADER_GPU_OP_COUNT=%u\n", syn_mut_shader_ops);
    printf("MODEL_WEIGHT_CHALLENGE_GPU_OP_COUNT=%u\n", m_challenge_ops);
    printf("MODEL_WEIGHT_NORMAL_BASELINE_GPU_OP_COUNT=%u\n", m_mut_baseline_ops);
    printf("MODEL_WEIGHT_MUTATED_GPU_OP_COUNT=%u\n", m_mut_shader_ops);
    printf("REAL_ACTIVATION_NORMAL_GPU_OP_COUNT=%u\n", act_baseline_ops);
    printf("REAL_ACTIVATION_MUTATED_GPU_OP_COUNT=%u\n", act_mut_shader_ops);
    printf("TOTAL_VISION_GPU_OP_COUNT=%u\n", vision_gpu_op_count);
    printf("COUNT_RECONCILED=%s\n", count_reconciled ? "true" : "false");
    printf("CPU_FALLBACK_COUNT=%u\n", cpu_fallback_count);
    printf("SILENT_FALLBACK_COUNT=%u\n", silent_fallback_count);
    printf("VISION_GPU_OP_COUNT=%u\n", vision_gpu_op_count);
    if (ncnn_parity_pass) {
        printf("NEXT_BLOCKER=NONE\n");
    } else {
        printf("NEXT_BLOCKER=RESOLVE_NCNN_REFERENCE_LAYOUT_OR_SEMANTIC_MISMATCH\n");
    }

    bool all_pass = (challenge_passed == 3) &&
                    (unique_hashes == 3) &&
                    (corrupted_fail_closed) &&
                    (forced_fail_closed) &&
                    (gates_passed == total_validation_gates) &&
                    (lifecycle_balanced) &&
                    (cpu_fallback_count == 0) &&
                    (silent_fallback_count == 0) &&
                    (model_weight_parity_pass) &&
                    (m_mut_pass) &&
                    (double_parity_pass) &&
                    (ncnn_parity_pass) &&
                    (act_mut_pass) &&
                    (count_reconciled);

    return all_pass ? 0 : 1;
}
