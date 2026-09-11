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
#include <algorithm>

int main() {
    printf("=== [AMEVA FORMAL BACKEND WHISPER FIXTURE & HARDENING TEST] ===\n");

    const uint32_t M = 1500;
    const uint32_t N = 384;
    const uint32_t K = 384;

    const size_t weight_bytes = (size_t)K * N * sizeof(uint16_t); // 294,912
    const size_t act_bytes    = (size_t)K * M * sizeof(float);    // 2,304,000
    const size_t out_bytes    = (size_t)N * M * sizeof(float);    // 2,304,000

    // 1. Initialize GGML context
    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FATAL: Failed to init ggml context\n");
        return 1;
    }

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, w, a);

    // Print shape and byte stride contract
    printf("SYNTHETIC_SRC0_NE=[%lld,%lld,%lld,%lld]\n", (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2], (long long)w->ne[3]);
    printf("SYNTHETIC_SRC1_NE=[%lld,%lld,%lld,%lld]\n", (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3]);
    printf("SYNTHETIC_DST_NE=[%lld,%lld,%lld,%lld]\n",  (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
    printf("SYNTHETIC_SRC0_NB=[%zu,%zu,%zu,%zu]\n", w->nb[0], w->nb[1], w->nb[2], w->nb[3]);
    printf("SYNTHETIC_SRC1_NB=[%zu,%zu,%zu,%zu]\n", a->nb[0], a->nb[1], a->nb[2], a->nb[3]);
    printf("SYNTHETIC_DST_NB=[%zu,%zu,%zu,%zu]\n",  dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);

    // 2. Read real Whisper fixture tensors
    const char * weight_path = "/data/data/com.termux/files/home/fixture_attn_q_weight.bin";
    const char * act_path    = "/data/data/com.termux/files/home/fixture_attn_q_activation.bin";
    const char * cpu_path    = "/data/data/com.termux/files/home/fixture_attn_q_cpu_out.bin";

    FILE * fw = fopen(weight_path, "rb");
    if (!fw || fread(w->data, 1, weight_bytes, fw) != weight_bytes) {
        fprintf(stderr, "FATAL: Cannot read %s\n", weight_path);
        return 2;
    }
    fclose(fw);

    FILE * fa = fopen(act_path, "rb");
    if (!fa || fread(a->data, 1, act_bytes, fa) != act_bytes) {
        fprintf(stderr, "FATAL: Cannot read %s\n", act_path);
        return 3;
    }
    fclose(fa);

    std::vector<float> golden(N * M);
    FILE * fc = fopen(cpu_path, "rb");
    if (!fc || fread(golden.data(), sizeof(float), N * M, fc) != N * M) {
        fprintf(stderr, "FATAL: Cannot read %s\n", cpu_path);
        return 4;
    }
    fclose(fc);

    memset(dst->data, 0, out_bytes);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // 3. Normal execution
    ameva_backend_reset_counters();
    ggml_backend_t backend = ggml_backend_ameva_init(nullptr);
    if (!backend) {
        fprintf(stderr, "FATAL: Failed to init AMEVA backend\n");
        return 5;
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: Normal graph_compute failed with status %d\n", status);
        return 6;
    }

    AmevaBackendCounters c_norm = {};
    ameva_backend_get_counters(&c_norm);

    printf("AMEVA_DISPATCH_ATTEMPT_COUNT=%llu\n", (unsigned long long)c_norm.ameva_dispatch_attempt_count);
    printf("AMEVA_DISPATCH_SUCCESS_COUNT=%llu\n", (unsigned long long)c_norm.ameva_dispatch_success_count);
    printf("AMEVA_GPU_OP_COUNT=%llu\n", (unsigned long long)c_norm.ameva_gpu_op_count);

    // 4. Parity evaluation against Golden CPU Out
    float * gpu_out = (float *)dst->data;
    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double dot = 0.0, norm_gpu = 0.0, norm_cpu = 0.0;
    std::vector<double> abs_errors(N * M);
    size_t nan_count = 0, inf_count = 0;

    for (size_t i = 0; i < (size_t)N * M; ++i) {
        float g = golden[i];
        float v = gpu_out[i];
        if (isnan(v)) nan_count++;
        if (isinf(v)) inf_count++;
        double diff = fabs((double)v - (double)g);
        abs_errors[i] = diff;
        if (diff > max_abs_err) max_abs_err = diff;
        sum_sq_err += diff * diff;
        dot += (double)v * (double)g;
        norm_gpu += (double)v * (double)v;
        norm_cpu += (double)g * (double)g;
    }

    double rmse = sqrt(sum_sq_err / ((double)N * M));
    double cosine_sim = dot / (sqrt(norm_gpu) * sqrt(norm_cpu));

    std::sort(abs_errors.begin(), abs_errors.end());
    double p99 = abs_errors[(size_t)(0.99 * abs_errors.size())];

    printf("COSINE_SIMILARITY=%.8f\n", cosine_sim);
    printf("RMSE=%.6f\n", rmse);
    printf("MAX_ABS_ERROR=%.6f\n", max_abs_err);
    printf("P99_ABS_ERROR=%.6f\n", p99);
    printf("NAN_COUNT=%zu\n", nan_count);
    printf("INF_COUNT=%zu\n", inf_count);

    bool parity_pass = (cosine_sim >= 0.99999 && rmse <= 0.005 && nan_count == 0 && inf_count == 0 && c_norm.ameva_gpu_op_count == 1);
    printf("WHISPER_TENSOR_PARITY=%s\n", parity_pass ? "PASS" : "FAIL");

    // 5. Failure Injection Gates (6/6)
    const char * fail_stages[6] = { "map", "flush", "dispatch", "wait", "invalidate", "readback" };
    int fail_passed = 0;

    for (int i = 0; i < 6; ++i) {
        setenv("AMEVA_BACKEND_FAIL_STAGE", fail_stages[i], 1);
        ameva_backend_reset_counters();
        memset(dst->data, 0, out_bytes);

        enum ggml_status fail_stat = ggml_backend_graph_compute(backend, gf);
        AmevaBackendCounters c_fail = {};
        ameva_backend_get_counters(&c_fail);

        bool stage_ok = (fail_stat == GGML_STATUS_FAILED &&
                         c_fail.ameva_gpu_op_count == 0 &&
                         c_fail.ameva_dispatch_failure_count == 1);
        if (stage_ok) fail_passed++;
        printf("FAIL_STAGE_%s: %s (status=%d, gpu_op=%llu, fail=%llu)\n",
               fail_stages[i], stage_ok ? "PASS" : "FAIL", (int)fail_stat,
               (unsigned long long)c_fail.ameva_gpu_op_count,
               (unsigned long long)c_fail.ameva_dispatch_failure_count);
    }
    unsetenv("AMEVA_BACKEND_FAIL_STAGE");
    printf("BACKEND_FAILURE_INJECTION_GATES=%d/6\n", fail_passed);

    // 6. Teardown and Lifecycle Symmetry
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

    bool balanced = (c_final.context_create_count == c_final.context_destroy_count &&
                     c_final.kernel_create_count == c_final.kernel_destroy_count &&
                     c_final.buffer_create_count == c_final.buffer_destroy_count &&
                     c_final.event_create_count == c_final.event_destroy_count);
    printf("LIFECYCLE_BALANCED=%s\n", balanced ? "true" : "false");

    bool all_ok = (parity_pass && fail_passed == 6 && balanced);
    printf("=== [ALL WHISPER FIXTURE & HARDENING GATES %s] ===\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 7;
}
