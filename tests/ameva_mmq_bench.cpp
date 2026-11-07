// SPDX-License-Identifier: Apache-2.0
// AMEVA MUL_MAT timing on the production path: weights in a WEIGHTS buffer (stored repacked, as llama.cpp loads a
// model), activations f32, one MUL_MAT per graph through ggml_backend_graph_compute. The kernel that ran is proven by
// the backend's counters (repacked tensors, ameva_mmq / ameva_mmqb MUL_MATs), not assumed.
//   ameva_mmq_bench <type> <M activation rows> <N weight rows> <K> [reps]
// Prints the median and minimum wall time of REPS graphs after one warm-up graph, and 2 M N K / time. With
// AMEVA_PROFILE=2 the backend adds the GPU time per kernel kind at exit (warm-up included). The result is checked
// against a double reference on 64 sampled outputs with the tolerance of the kernel's activation rounding, so that a
// timing of a kernel text under test (AMEVA_CL_SOURCE_DIR) states whether that text still computes the product.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

static double now_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ameva_mmq_bench <type> <M> <N> <K> [reps]\n");
        return 2;
    }
    enum ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((enum ggml_type)t);
        if (n && strcmp(n, argv[1]) == 0) type = (enum ggml_type)t;
    }
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 9;
    if (type == GGML_TYPE_COUNT || M < 1 || N < 1 || K < 1 || reps < 1 || K % ggml_blck_size(type) != 0) {
        fprintf(stderr, "bad arguments (type %s, K must be a whole number of blocks)\n", argv[1]);
        return 2;
    }
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<float> wf((size_t)N * K), af((size_t)M * K);
    for (float& v : wf) v = U(rng);
    for (float& v : af) v = U(rng);
    const size_t row_bytes = ggml_row_size(type, K), wbytes = row_bytes * N;
    std::vector<uint8_t> wq(wbytes);
    std::vector<float> wref((size_t)N * K);
    if (type == GGML_TYPE_F32) {
        memcpy(wq.data(), wf.data(), wbytes);
        wref = wf;
    } else {
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, nullptr);
        const struct ggml_type_traits* tt = ggml_get_type_traits(type);
        for (int n = 0; n < N; ++n) tt->to_float(wq.data() + row_bytes * n, wref.data() + (size_t)n * K, K);
    }

    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* wctx = ggml_init(ip);
    struct ggml_context* actx = ggml_init(ip);
    struct ggml_tensor* w = ggml_new_tensor_2d(wctx, type, K, N);
    struct ggml_tensor* a = ggml_new_tensor_2d(actx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* out = ggml_mul_mat(actx, w, a);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
    if (!wbuf || !abuf) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_supports_op(be, out)) {
        fprintf(stderr, "MMQ_BENCH FAIL: the backend declines this MUL_MAT\n");
        return 1;
    }
    const int64_t rep0 = ggml_backend_ameva_get_repacked_tensor_count();
    ggml_backend_tensor_set(w, wq.data(), 0, wbytes);
    ggml_backend_tensor_set(a, af.data(), 0, af.size() * sizeof(float));
    const int64_t repacked = ggml_backend_ameva_get_repacked_tensor_count() - rep0;
    struct ggml_cgraph* gf = ggml_new_graph(actx);
    ggml_build_forward_expand(gf, out);

    const int64_t mmq0 = ggml_backend_ameva_get_mmq_mul_mat_count(), mmqb0 = ggml_backend_ameva_get_mmqb_mul_mat_count();
    std::vector<double> t;
    for (int r = -1; r < reps; ++r) {
        const double t0 = now_ms();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "MMQ_BENCH FAIL: graph_compute failed\n");
            return 1;
        }
        if (r >= 0) t.push_back(now_ms() - t0);
    }
    const int64_t mmq = ggml_backend_ameva_get_mmq_mul_mat_count() - mmq0;
    const int64_t mmqb = ggml_backend_ameva_get_mmqb_mul_mat_count() - mmqb0;
    const char* path = mmqb == reps + 1 ? "ameva_mmqb" : mmq == reps + 1 ? "ameva_mmq" : mmq == 0 ? "other" : "mixed";

    // 64 sampled outputs against a double reference; int8 activations when an int8 kernel ran
    std::vector<float> got((size_t)M * N);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    const int blk = mmqb ? 256 : 32;
    double worst = 0.0;
    std::mt19937 pick(99);
    for (int s = 0; s < 64; ++s) {
        const int m = (int)(pick() % M), n = (int)(pick() % N);
        std::vector<double> x(K);
        for (int k = 0; k < K; ++k) x[k] = af[(size_t)m * K + k];
        if (mmq) {
            for (int k0 = 0; k0 < K; k0 += blk) {
                double amax = 0.0;
                for (int k = k0; k < k0 + blk && k < K; ++k) amax = std::max(amax, std::fabs(x[k]));
                const double d = amax / 127.0;
                for (int k = k0; k < k0 + blk && k < K; ++k) x[k] = d > 0 ? std::nearbyint(x[k] / d) * d : 0.0;
            }
        }
        double ref = 0.0, mag = 0.0;
        for (int k = 0; k < K; ++k) {
            const double p = (double)wref[(size_t)n * K + k] * x[k];
            ref += p;
            mag += std::fabs(p);
        }
        worst = std::max(worst, std::fabs(got[(size_t)m * N + n] - ref) / (mag + 1e-30));
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2], mn = t.front(), gop = 2.0 * M * N * K / 1e9;
    printf("MMQ_BENCH type=%s M=%d N=%d K=%d path=%s repacked=%lld reps=%d median_ms=%.3f min_ms=%.3f "
           "GOP/s_median=%.1f GOP/s_min=%.1f check_max_rel=%.2e check=%s\n",
           argv[1], M, N, K, path, (long long)repacked, reps, med, mn, gop / med * 1e3, gop / mn * 1e3, worst,
           worst < 2e-3 ? "PASS" : "FAIL");
    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(abuf);
    ggml_free(wctx);
    ggml_free(actx);
    ggml_backend_free(be);
    return 0;
}
