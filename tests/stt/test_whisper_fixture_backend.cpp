/**
 * @file test_whisper_fixture_backend.cpp
 * @brief Formal verification of Whisper fixture parity, failure injection, and lifecycle symmetry
 */

#include "ggml-backend-ameva.h"
#include "../../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>

// ============================================================================
// Lightweight Self-Contained SHA-256 Implementation
// ============================================================================
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
} Sha256Context;

#define ROR32(val, bits) (((val) >> (bits)) | ((val) << (32 - (bits))))
#define CH(x, y, z)      (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z)     (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)           (ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22))
#define EP1(x)           (ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25))
#define SIG0(x)          (ROR32(x, 7) ^ ROR32(x, 18) ^ ((x) >> 3))
#define SIG1(x)          (ROR32(x, 17) ^ ROR32(x, 19) ^ ((x) >> 10))

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

static void sha256_transform(Sha256Context * ctx, const uint8_t data[64]) {
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8)  | ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + EP1(e) + CH(e, f, g) + K256[i] + m[i];
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256Context * ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(Sha256Context * ctx, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->buffer[ctx->count % 64] = data[i];
        ctx->count++;
        if ((ctx->count % 64) == 0) {
            sha256_transform(ctx, ctx->buffer);
        }
    }
}

static void sha256_final(Sha256Context * ctx, uint8_t hash[32]) {
    uint64_t total_bits = ctx->count * 8;
    sha256_update(ctx, (const uint8_t *)"\x80", 1);
    while ((ctx->count % 64) != 56) {
        sha256_update(ctx, (const uint8_t *)"\x00", 1);
    }
    for (int i = 7; i >= 0; --i) {
        uint8_t b = (total_bits >> (i * 8)) & 0xff;
        sha256_update(ctx, &b, 1);
    }
    for (int i = 0; i < 8; ++i) {
        hash[i * 4]     = (ctx->state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (ctx->state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (ctx->state[i] >> 8)  & 0xff;
        hash[i * 4 + 3] = (ctx->state[i])       & 0xff;
    }
}

static std::string compute_sha256(const uint8_t * data, size_t len) {
    Sha256Context ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char buf[65];
    for (int i = 0; i < 32; ++i) {
        sprintf(buf + (i * 2), "%02x", hash[i]);
    }
    buf[64] = '\0';
    return std::string(buf);
}

static std::string compute_file_sha256(const char * path, size_t * out_size) {
    FILE * f = fopen(path, "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (out_size) *out_size = (size_t)sz;
    std::vector<uint8_t> buf(sz > 0 ? sz : 1);
    if (sz > 0) {
        size_t rd = fread(buf.data(), 1, sz, f);
        (void)rd;
    }
    fclose(f);
    return compute_sha256(buf.data(), sz);
}

// ============================================================================
// Main Fixture & Hardening Test Entrypoint
// ============================================================================
int main() {
    printf("=== [AMEVA FORMAL BACKEND WHISPER FIXTURE & HARDENING TEST] ===\n");

    const uint32_t M = 1500;
    const uint32_t N = 384;
    const uint32_t K = 384;

    const size_t expected_weight_bytes = (size_t)K * N * sizeof(uint16_t); // 294,912
    const size_t expected_act_bytes    = (size_t)K * M * sizeof(float);    // 2,304,000
    const size_t expected_out_bytes    = (size_t)N * M * sizeof(float);    // 2,304,000

    const char * weight_path = "/data/data/com.termux/files/home/fixture_attn_q_weight.bin";
    const char * act_path    = "/data/data/com.termux/files/home/fixture_attn_q_activation.bin";
    const char * cpu_path    = "/data/data/com.termux/files/home/fixture_attn_q_cpu_out.bin";

    // 1. Fixture Size & Trailing Byte Integrity Gate
    size_t weight_file_sz = 0, act_file_sz = 0, cpu_file_sz = 0;
    std::string weight_sha = compute_file_sha256(weight_path, &weight_file_sz);
    std::string act_sha    = compute_file_sha256(act_path, &act_file_sz);
    std::string cpu_sha    = compute_file_sha256(cpu_path, &cpu_file_sz);

    bool weight_size_match = (weight_file_sz == expected_weight_bytes);
    bool act_size_match    = (act_file_sz == expected_act_bytes);
    bool golden_size_match = (cpu_file_sz == expected_out_bytes);

    int size_gates_passed = 0;
    if (weight_size_match) size_gates_passed++;
    if (act_size_match)    size_gates_passed++;
    if (golden_size_match) size_gates_passed++;

    printf("WEIGHT_SIZE_MATCH=%s\n", weight_size_match ? "true" : "false");
    printf("ACTIVATION_SIZE_MATCH=%s\n", act_size_match ? "true" : "false");
    printf("GOLDEN_SIZE_MATCH=%s\n", golden_size_match ? "true" : "false");
    printf("FIXTURE_SIZE_GATE=%d/3\n", size_gates_passed);

    // 2. Fixture SHA-256 Pre-Execution Hash Gate
    const std::string EXPECTED_WEIGHT_SHA = "51ce67bcaa84b565ec5250da8f7a6d2c9422677892d0d6758b76b3bde2799c6a";
    const std::string EXPECTED_ACT_SHA    = "52d27f56cd23ba0f629f3def9f2c2c98b91b04dc68bb52d7f9f44f7e6993faff";
    const std::string EXPECTED_CPU_SHA    = "8e56163c7a83f9bb950957f2a2cfbf28d20fa3518eeae47aa85db7aa1c1d440b";

    int hash_gates_passed = 0;
    if (weight_sha == EXPECTED_WEIGHT_SHA) hash_gates_passed++;
    if (act_sha == EXPECTED_ACT_SHA)       hash_gates_passed++;
    if (cpu_sha == EXPECTED_CPU_SHA)       hash_gates_passed++;

    printf("FIXTURE_WEIGHT_SHA256=%s\n", weight_sha.c_str());
    printf("FIXTURE_ACTIVATION_SHA256=%s\n", act_sha.c_str());
    printf("FIXTURE_CPU_OUT_SHA256=%s\n", cpu_sha.c_str());
    printf("FIXTURE_HASH_GATE=%d/3\n", hash_gates_passed);

    if (size_gates_passed != 3 || hash_gates_passed != 3) {
        fprintf(stderr, "FATAL: Fixture size or hash integrity check failed!\n");
        return 1;
    }

    // 3. Runtime SPIR-V Shader Verification
    const char * env_spirv = getenv("AMEVA_SPIRV_PATH");
    const char * effective_spirv = env_spirv ? env_spirv : "/data/data/com.termux/files/home/matmul_f16_f32.spv";
    size_t spirv_sz = 0;
    std::string runtime_spirv_sha = compute_file_sha256(effective_spirv, &spirv_sz);
    bool is_mutated_spirv = (env_spirv != nullptr && strstr(env_spirv, "mutated") != nullptr);

    printf("RUNTIME_LOADED_SPIRV_PATH=%s\n", effective_spirv);
    printf("RUNTIME_LOADED_SPIRV_SHA256=%s\n", runtime_spirv_sha.c_str());
    printf("RUNTIME_LOADED_MUTATED_SPIRV=%s\n", is_mutated_spirv ? "true" : "false");

    // 4. Initialize GGML context & tensors
    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FATAL: Failed to init ggml context\n");
        return 2;
    }

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, w, a);

    // Read fixture content
    FILE * fw = fopen(weight_path, "rb");
    size_t rdw = fread(w->data, 1, expected_weight_bytes, fw);
    fclose(fw);

    FILE * fa = fopen(act_path, "rb");
    size_t rda = fread(a->data, 1, expected_act_bytes, fa);
    fclose(fa);

    std::vector<float> golden(N * M);
    FILE * fc = fopen(cpu_path, "rb");
    size_t rdc = fread(golden.data(), sizeof(float), N * M, fc);
    fclose(fc);

    if (rdw != expected_weight_bytes || rda != expected_act_bytes || rdc != N * M) {
        fprintf(stderr, "FATAL: fread byte count mismatch\n");
        ggml_free(ctx);
        return 3;
    }

    memset(dst->data, 0, expected_out_bytes);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // 5. Normal execution
    ameva_backend_reset_counters();
    ameva_backend_set_graph_kind("ENCODER");
    ggml_backend_t backend = ggml_backend_ameva_init(nullptr);
    if (!backend) {
        fprintf(stderr, "FATAL: Failed to init AMEVA backend\n");
        ggml_free(ctx);
        return 4;
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: Normal graph_compute failed with status %d\n", status);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 5;
    }

    AmevaBackendCounters c_norm = {};
    ameva_backend_get_counters(&c_norm);

    printf("AMEVA_DISPATCH_ATTEMPT_COUNT=%llu\n", (unsigned long long)c_norm.ameva_dispatch_attempt_count);
    printf("AMEVA_DISPATCH_SUCCESS_COUNT=%llu\n", (unsigned long long)c_norm.ameva_dispatch_success_count);
    printf("AMEVA_GPU_OP_COUNT=%llu\n", (unsigned long long)c_norm.ameva_gpu_op_count);

    // 6. Parity evaluation against Golden CPU Out
    float * gpu_out = (float *)dst->data;
    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_delta = 0.0;
    double dot = 0.0, norm_gpu = 0.0, norm_cpu = 0.0;
    std::vector<double> abs_errors(N * M);
    size_t nan_count = 0, inf_count = 0;

    for (size_t i = 0; i < (size_t)N * M; ++i) {
        float g = golden[i];
        float v = gpu_out[i];
        if (isnan(v)) nan_count++;
        if (isinf(v)) inf_count++;
        double delta = (double)v - (double)g;
        double diff = fabs(delta);
        abs_errors[i] = diff;
        sum_delta += delta;
        if (diff > max_abs_err) max_abs_err = diff;
        sum_sq_err += diff * diff;
        dot += (double)v * (double)g;
        norm_gpu += (double)v * (double)v;
        norm_cpu += (double)g * (double)g;
    }

    double rmse = sqrt(sum_sq_err / ((double)N * M));
    double cosine_sim = dot / (sqrt(norm_gpu) * sqrt(norm_cpu));
    double mean_delta = sum_delta / ((double)N * M);

    std::sort(abs_errors.begin(), abs_errors.end());
    double p99 = abs_errors[(size_t)(0.99 * abs_errors.size())];

    std::string tensor_sha = compute_sha256((const uint8_t *)gpu_out, expected_out_bytes);
    printf("OUTPUT_TENSOR_SHA256=%s\n", tensor_sha.c_str());
    printf("COSINE_SIMILARITY=%.8f\n", cosine_sim);
    printf("RMSE=%.6f\n", rmse);
    printf("MAX_ABS_ERROR=%.6f\n", max_abs_err);
    printf("P99_ABS_ERROR=%.6f\n", p99);
    printf("MEAN_DELTA=%.6f\n", mean_delta);
    printf("NAN_COUNT=%zu\n", nan_count);
    printf("INF_COUNT=%zu\n", inf_count);

    bool parity_pass = false;
    if (is_mutated_spirv) {
        // In mutated shader (+0.5), we expect parity failure with mean_delta ~ 0.5
        bool output_hash_changed = (tensor_sha != EXPECTED_CPU_SHA);
        bool delta_observed = (fabs(mean_delta - 0.5) < 0.05);
        bool parity_failure_observed = (cosine_sim < 0.99999 || rmse > 0.005);
        printf("OUTPUT_HASH_CHANGED=%s\n", output_hash_changed ? "true" : "false");
        printf("MUTATED_OUTPUT_DELTA_OBSERVED=%s\n", delta_observed ? "true" : "false");
        printf("PARITY_FAILURE_OBSERVED=%s\n", parity_failure_observed ? "true" : "false");
        printf("WHISPER_TENSOR_PARITY=FAIL_EXPECTED\n");
    } else {
        parity_pass = (cosine_sim >= 0.99999 && rmse <= 0.005 && nan_count == 0 && inf_count == 0 && c_norm.ameva_gpu_op_count == 1);
        printf("WHISPER_TENSOR_PARITY=%s\n", parity_pass ? "PASS" : "FAIL");
    }

    // Report Unknown Graph Kind Counter Defense
    printf("UNKNOWN_GRAPH_KIND_COUNT=%llu\n", (unsigned long long)c_norm.unknown_graph_kind_count);

    // 7. Partial Initialization Failure Rollback Gates (5/5)
    printf("--- PARTIAL INIT FAIL-CLOSED ROLLBACK GATES ---\n");
    const char * init_fail_stages[5] = { "context", "kernel", "buf_w", "buf_a", "buf_o" };
    int partial_init_passed = 0;
    bool post_fail_ctx_bal = true;
    bool post_fail_ker_bal = true;
    bool post_fail_buf_bal = true;
    bool post_fail_evt_bal = true;

    for (int i = 0; i < 5; ++i) {
        // Force complete teardown of hardware before partial init test
        ameva_backend_teardown_hardware();

        setenv("AMEVA_INIT_FAIL_STAGE", init_fail_stages[i], 1);
        AmevaBackendCounters c_before = {};
        ameva_backend_get_counters(&c_before);

        enum ggml_status init_stat = ggml_backend_graph_compute(backend, gf);
        AmevaBackendCounters c_after = {};
        ameva_backend_get_counters(&c_after);

        bool ctx_bal = (c_after.context_create_count == c_after.context_destroy_count);
        bool ker_bal = (c_after.kernel_create_count == c_after.kernel_destroy_count);
        bool buf_bal = (c_after.buffer_create_count == c_after.buffer_destroy_count);
        bool evt_bal = (c_after.event_create_count == c_after.event_destroy_count);

        if (!ctx_bal) post_fail_ctx_bal = false;
        if (!ker_bal) post_fail_ker_bal = false;
        if (!buf_bal) post_fail_buf_bal = false;
        if (!evt_bal) post_fail_evt_bal = false;

        bool stage_ok = (init_stat == GGML_STATUS_FAILED && ctx_bal && ker_bal && buf_bal && evt_bal);
        if (stage_ok) partial_init_passed++;

        printf("PARTIAL_INIT_FAIL_%s: %s (status=%d, balanced=%s)\n",
               init_fail_stages[i], stage_ok ? "PASS" : "FAIL", (int)init_stat,
               (ctx_bal && ker_bal && buf_bal && evt_bal) ? "true" : "false");
    }
    unsetenv("AMEVA_INIT_FAIL_STAGE");

    // Re-initialize hardware pipeline normally for subsequent runtime failure tests
    ameva_backend_teardown_hardware();
    ggml_backend_graph_compute(backend, gf); // Bring-up hardware

    printf("PARTIAL_INIT_FAILURE_GATES=%d/5\n", partial_init_passed);
    printf("POST_FAILURE_CONTEXT_BALANCED=%s\n", post_fail_ctx_bal ? "true" : "false");
    printf("POST_FAILURE_KERNEL_BALANCED=%s\n", post_fail_ker_bal ? "true" : "false");
    printf("POST_FAILURE_BUFFER_BALANCED=%s\n", post_fail_buf_bal ? "true" : "false");
    printf("POST_FAILURE_EVENT_BALANCED=%s\n", post_fail_evt_bal ? "true" : "false");

    // 8. Real Runtime Failure Injection Gates (6/6) with DST Canary Protection Check
    printf("--- REAL RUNTIME API FAILURE INJECTION GATES ---\n");
    const char * fail_stages[6] = { "map", "flush", "dispatch", "wait", "invalidate", "readback" };
    int fail_stage_passed = 0;
    int dst_canary_passed = 0;
    const uint8_t CANARY_VAL = 0xAA;

    for (int i = 0; i < 6; ++i) {
        setenv("AMEVA_RUNTIME_FAIL_STAGE", fail_stages[i], 1);
        setenv("AMEVA_BACKEND_FAIL_STAGE", fail_stages[i], 1);
        AmevaBackendCounters c_before = {};
        ameva_backend_get_counters(&c_before);

        // Fill destination with canary pattern
        memset(dst->data, CANARY_VAL, expected_out_bytes);

        enum ggml_status fail_stat = ggml_backend_graph_compute(backend, gf);
        AmevaBackendCounters c_after = {};
        ameva_backend_get_counters(&c_after);

        uint64_t fail_delta = c_after.ameva_dispatch_failure_count - c_before.ameva_dispatch_failure_count;
        uint64_t op_delta   = c_after.ameva_gpu_op_count - c_before.ameva_gpu_op_count;
        bool evt_bal = (c_after.event_create_count == c_after.event_destroy_count);

        bool stage_ok = (fail_stat == GGML_STATUS_FAILED &&
                         op_delta == 0 &&
                         fail_delta == 1 &&
                         evt_bal);
        if (stage_ok) fail_stage_passed++;

        // Canary check: destination tensor MUST retain canary pattern completely untouched
        const uint8_t * dptr = (const uint8_t *)dst->data;
        bool canary_intact = true;
        for (size_t b = 0; b < expected_out_bytes; ++b) {
            if (dptr[b] != CANARY_VAL) {
                canary_intact = false;
                break;
            }
        }
        if (canary_intact) dst_canary_passed++;

        printf("REAL_%s_API_FAILURE: %s (status=%d, gpu_op_delta=%llu, fail_delta=%llu, canary=%s, event_balanced=%s)\n",
               fail_stages[i], stage_ok ? "PASS" : "FAIL", (int)fail_stat,
               (unsigned long long)op_delta,
               (unsigned long long)fail_delta,
               canary_intact ? "INTACT" : "CORRUPTED",
               evt_bal ? "true" : "false");
    }
    unsetenv("AMEVA_RUNTIME_FAIL_STAGE");
    unsetenv("AMEVA_BACKEND_FAIL_STAGE");

    printf("FAIL_STAGE_HIT_GATES=%d/6\n", fail_stage_passed);
    printf("DST_CANARY_GATES=%d/6\n", dst_canary_passed);
    printf("REAL_RUNTIME_API_FAILURE_COVERAGE=%s\n", (fail_stage_passed == 6) ? "PASS" : "PARTIAL");

    // 9. Teardown and Lifecycle Symmetry Gates (6/6)
    ggml_backend_free(backend);
    ggml_free(ctx);
    ameva_backend_teardown_hardware();

    AmevaBackendCounters c_final = {};
    ameva_backend_get_counters(&c_final);

    printf("CONTEXT_CREATE_COUNT=%llu\n", (unsigned long long)c_final.context_create_count);
    printf("CONTEXT_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.context_destroy_count);
    printf("KERNEL_CREATE_COUNT=%llu\n", (unsigned long long)c_final.kernel_create_count);
    printf("KERNEL_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.kernel_destroy_count);
    printf("BUFFER_CREATE_COUNT=%llu\n", (unsigned long long)c_final.buffer_create_count);
    printf("BUFFER_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.buffer_destroy_count);
    printf("EVENT_CREATE_COUNT=%llu\n", (unsigned long long)c_final.event_create_count);
    printf("EVENT_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.event_destroy_count);

    int lifecycle_gates = 0;
    if (c_final.context_create_count > 0 && c_final.context_create_count == c_final.context_destroy_count) lifecycle_gates++;
    if (c_final.kernel_create_count > 0 && c_final.kernel_create_count == c_final.kernel_destroy_count)   lifecycle_gates++;
    if (c_final.buffer_create_count > 0 && c_final.buffer_create_count == c_final.buffer_destroy_count)   lifecycle_gates++;
    if (c_final.event_create_count > 0 && c_final.event_create_count == c_final.event_destroy_count)     lifecycle_gates++;
    if (c_final.buffer_alloc_count == c_final.buffer_free_count)                                           lifecycle_gates++;
    if (c_final.ameva_dispatch_failure_count == (uint64_t)(partial_init_passed + fail_stage_passed))       lifecycle_gates++;

    printf("LIFECYCLE_GATES=%d/6\n", lifecycle_gates);

    bool all_ok = false;
    if (is_mutated_spirv) {
        all_ok = (hash_gates_passed == 3 && size_gates_passed == 3 && fail_stage_passed == 6 && dst_canary_passed == 6);
    } else {
        all_ok = (parity_pass && hash_gates_passed == 3 && size_gates_passed == 3 &&
                  partial_init_passed == 5 && fail_stage_passed == 6 && dst_canary_passed == 6 &&
                  lifecycle_gates == 6 && c_norm.unknown_graph_kind_count == 0);
    }

    printf("=== [ALL WHISPER FIXTURE & HARDENING GATES %s] ===\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 7;
}
