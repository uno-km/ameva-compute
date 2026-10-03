// SPDX-License-Identifier: Apache-2.0
// AMEVA MUL_MAT timing: one MUL_MAT node through the ggml backend, repeated, for kernel tuning.
//
//   ameva_mm_bench TYPE M N K [ITER]
//
// Runs dst = W^T x A with W of type TYPE (N rows of K weights) and A f32 (M rows of K) twice: with W in a buffer of
// usage ANY (ggml layout, ameva_qmm) and with W in a weight buffer (repacked layout, ameva_qmv; only for types with a
// repacked layout and repacking enabled). Prints the mean wall time of graph_compute per iteration, its effective
// weight bandwidth and the GPU time reported by AMEVA_PROFILE when it is set. The two results are compared to each
// other (max relative difference) so that a tuning run cannot silently time a wrong kernel.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0x2545F491u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static enum ggml_type type_by_name(const char* n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* tn = ggml_type_name((enum ggml_type)t);
        if (tn && strcmp(tn, n) == 0) return (enum ggml_type)t;
    }
    return GGML_TYPE_COUNT;
}

// mean milliseconds per graph_compute; out receives the result of the last run
static double run(ggml_backend_t be, enum ggml_type type, int M, int N, int K, int iters, bool weights,
                  const std::vector<uint8_t>& wq, const std::vector<float>& af, std::vector<float>& out) {
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, type, K, N);
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* d = ggml_mul_mat(ctx, w, a);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    if (!buf) {
        fprintf(stderr, "allocation failed\n");
        exit(2);
    }
    if (weights) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_supports_op(be, d)) {
        fprintf(stderr, "MUL_MAT not supported for this shape\n");
        exit(2);
    }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(a, af.data(), 0, af.size() * sizeof(float));
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { // warm-up: kernel creation, descriptor cache
        fprintf(stderr, "graph_compute failed\n");
        exit(1);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph_compute failed\n");
            exit(1);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    out.resize((size_t)M * N);
    ggml_backend_tensor_get(d, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s TYPE M N K [ITER]\n", argv[0]);
        return 2;
    }
    const enum ggml_type type = type_by_name(argv[1]);
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    const int iters = argc > 5 ? atoi(argv[5]) : 50;
    if (type == GGML_TYPE_COUNT || M <= 0 || N <= 0 || K <= 0 || K % ggml_blck_size(type) || iters <= 0) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    std::vector<float> wf((size_t)N * K), af((size_t)M * K);
    for (float& v : wf) v = rnd();
    for (float& v : af) v = rnd();
    std::vector<uint8_t> wq(ggml_row_size(type, K) * N);
    if (type == GGML_TYPE_F32) memcpy(wq.data(), wf.data(), wq.size());
    else ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, nullptr);
    const double mb = wq.size() / 1e6;

    std::vector<float> r_qmm, r_qmv;
    const int64_t q0 = ggml_backend_ameva_get_qmv_mul_mat_count();
    const double t_qmm = run(be, type, M, N, K, iters, false, wq, af, r_qmm);
    const bool qmm_used = ggml_backend_ameva_get_qmv_mul_mat_count() == q0;
    printf("MM_BENCH type=%s M=%d N=%d K=%d path=qmm ms=%.4f weight_GBps=%.2f%s\n", argv[1], M, N, K, t_qmm,
           mb / t_qmm, qmm_used ? "" : " (ERROR: ran ameva_qmv)");
    const bool repack_on = !ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(be));
    int rc = qmm_used ? 0 : 1;
    if (repack_on && type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        const int64_t q1 = ggml_backend_ameva_get_qmv_mul_mat_count();
        const double t_qmv = run(be, type, M, N, K, iters, true, wq, af, r_qmv);
        const bool qmv_used = ggml_backend_ameva_get_qmv_mul_mat_count() - q1 == iters + 1;
        // largest difference relative to the RMS of the outputs: an output that sums to almost zero would make a
        // per-element relative difference meaningless
        double maxd = 0.0, ss = 0.0;
        for (size_t i = 0; i < r_qmm.size(); ++i) {
            ss += (double)r_qmm[i] * (double)r_qmm[i];
            maxd = std::fmax(maxd, std::fabs((double)r_qmv[i] - (double)r_qmm[i]));
        }
        const double rms = std::sqrt(ss / (double)r_qmm.size());
        const double rel = rms > 0.0 ? maxd / rms : maxd;
        printf("MM_BENCH type=%s M=%d N=%d K=%d path=qmv ms=%.4f weight_GBps=%.2f speedup=%.2f max_diff_over_rms=%.2e%s\n",
               argv[1], M, N, K, t_qmv, mb / t_qmv, t_qmm / t_qmv, rel, qmv_used ? "" : " (ERROR: qmv not used)");
        if (!qmv_used || rel > 1e-4) rc = 1; // f32 summation order over K terms differs between kernels
    }
    ggml_backend_free(be);
    return rc;
}
