// SPDX-License-Identifier: Apache-2.0
// AMEVA graph-engine fusion (ameva_find_fusion): the node patterns of one token of a transformer layer, computed
// through the ggml backend with repacked weights and one activation row:
//   y    = RMS_NORM(x) * w_norm                         RMS_NORM -> MUL
//   q    = W_q y + b_q                                  MUL_MAT -> ADD (bias, product first)
//   o    = r + W_o y                                    MUL_MAT -> ADD (residual, product second)
//   f    = silu(W_g y) * (W_u y)                        MUL_MAT (up), MUL_MAT (gate) -> SWIGLU (product = gate)
//   f2   = silu(W_g2 y) * (W_u2 y)                      MUL_MAT (gate), MUL_MAT (up) -> SWIGLU (product = up)
// (ggml visits src0 before src1, so a SWIGLU built from two fresh products follows the up projection, as in
// llama.cpp; the first pair expands the up projection beforehand so that the gate product comes last.)
// Checks: every output and every intermediate (the fused dispatches still write the first node's result) equals a
// double-precision reference from ggml's dequantization to 1e-5 relative; the fused-pair counter moved by 5 when
// fusion is on (AMEVA_FUSE unset or 1) and by 0 with AMEVA_FUSE=0. It prints FUSE_HASH, a hash of all result bits:
// the same build must print the same hash with AMEVA_FUSE=0 and 1 (fusion uses the f32 operations of the unfused
// nodes). The check sets AMEVA_MMVQ=0 (f32 token-generation kernels, see main). Exit 0 only if every check passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0xa54ff53au;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static const int K = 512, N = 384;

struct Weight {
    ggml_tensor* t;
    enum ggml_type type;
    std::vector<uint8_t> q;
    std::vector<float> deq; // N x K
};

static void make_weight(Weight& w) {
    std::vector<float> f((size_t)N * K);
    for (auto& v : f) v = rnd();
    w.q.resize(ggml_row_size(w.type, K) * N);
    ggml_quantize_chunk(w.type, f.data(), w.q.data(), 0, N, K, nullptr);
    w.deq.resize(f.size());
    ggml_get_type_traits(w.type)->to_float(w.q.data(), w.deq.data(), (int64_t)f.size());
}

static std::vector<double> matvec(const Weight& w, const std::vector<double>& y, std::vector<double>& mag) {
    std::vector<double> r(N, 0.0);
    mag.assign(N, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const double p = (double)w.deq[(size_t)n * K + k] * y[k];
            r[n] += p;
            mag[n] += std::fabs(p);
        }
    }
    return r;
}

static int g_failed = 0;
static uint64_t g_hash = 1469598103934665603ull;

static void check(const char* name, ggml_tensor* t, const std::vector<double>& ref, const std::vector<double>& mag) {
    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(t, got.data(), 0, got.size() * 4);
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double rel = std::fabs((double)got[i] - ref[i]) / (mag[i] + 1e-30);
        if (!std::isfinite(got[i])) worst = 1e30;
        else if (rel > worst) worst = rel;
        uint32_t u;
        memcpy(&u, &got[i], 4);
        g_hash = (g_hash ^ u) * 1099511628211ull;
    }
    const bool ok = worst <= 1e-5;
    if (!ok) ++g_failed;
    printf("FUSE %-14s max_rel=%.3g %s\n", name, worst, ok ? "PASS" : "FAIL");
}

int main() {
    // The reference is f32 arithmetic on ggml's dequantized weights. The measured int8 token-generation layouts
    // (ameva_mmvq, chosen where they are faster) quantize the activations and differ from it by about 1e-3, so this
    // check runs with them off; another setting is refused instead of being compared against the wrong tolerance.
    const char* mv = getenv("AMEVA_MMVQ");
    if (mv && strcmp(mv, "0") != 0) {
        fprintf(stderr, "ameva_fuse_check compares the f32 kernels: AMEVA_MMVQ must be unset or 0 (got \"%s\")\n", mv);
        return 2;
    }
    setenv("AMEVA_MMVQ", "0", 1);
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* wctx = ggml_init(ip); // weights (repacked: weight buffer)
    ggml_context* ctx = ggml_init(ip);  // activations and results
    Weight wq{ nullptr, GGML_TYPE_Q4_K, {}, {} }, wo{ nullptr, GGML_TYPE_Q5_0, {}, {} };
    Weight wu{ nullptr, GGML_TYPE_Q8_0, {}, {} }, wg{ nullptr, GGML_TYPE_Q6_K, {}, {} };
    Weight wg2{ nullptr, GGML_TYPE_IQ4_NL, {}, {} }, wu2{ nullptr, GGML_TYPE_Q4_0, {}, {} };
    Weight* ws[6] = { &wq, &wo, &wu, &wg, &wg2, &wu2 };
    for (Weight* w : ws) {
        w->t = ggml_new_tensor_2d(wctx, w->type, K, N);
        make_weight(*w);
    }
    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_tensor* wn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_tensor* bq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_tensor* r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, 1);
    const float eps = 1e-6f;
    ggml_tensor* yn = ggml_rms_norm(ctx, x, eps);
    ggml_tensor* y = ggml_mul(ctx, yn, wn);
    ggml_tensor* mq = ggml_mul_mat(ctx, wq.t, y);
    ggml_tensor* q = ggml_add(ctx, mq, bq);
    ggml_tensor* mo = ggml_mul_mat(ctx, wo.t, y);
    ggml_tensor* o = ggml_add(ctx, r, mo);
    ggml_tensor* mu = ggml_mul_mat(ctx, wu.t, y);
    ggml_tensor* mg = ggml_mul_mat(ctx, wg.t, y);
    ggml_tensor* f = ggml_swiglu_split(ctx, mg, mu);
    ggml_tensor* mg2 = ggml_mul_mat(ctx, wg2.t, y);
    ggml_tensor* mu2 = ggml_mul_mat(ctx, wu2.t, y);
    ggml_tensor* f2 = ggml_swiglu_split(ctx, mg2, mu2);
    ggml_tensor* outs[] = { y, yn, q, mq, o, mo, f, mu, mg, f2, mg2, mu2 };
    for (ggml_tensor* t : outs) ggml_set_output(t); // keep every result readable (no buffer reuse between them)

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) return 2;
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (Weight* w : ws) ggml_backend_tensor_set(w->t, w->q.data(), 0, w->q.size());
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!abuf) return 2;
    std::vector<float> xf(K), wnf(K), bqf(N), rf(N);
    for (auto& v : xf) v = 3.0f * rnd();
    for (auto& v : wnf) v = rnd();
    for (auto& v : bqf) v = rnd();
    for (auto& v : rf) v = rnd();
    ggml_backend_tensor_set(x, xf.data(), 0, K * 4);
    ggml_backend_tensor_set(wn, wnf.data(), 0, K * 4);
    ggml_backend_tensor_set(bq, bqf.data(), 0, N * 4);
    ggml_backend_tensor_set(r, rf.data(), 0, N * 4);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, mu);
    ggml_build_forward_expand(gf, f);
    ggml_build_forward_expand(gf, f2);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            printf("  FAIL supports_op declined %s\n", ggml_op_desc(ggml_graph_node(gf, i)));
            return 1;
        }
    }
    const int64_t fused0 = ggml_backend_ameva_get_fused_pair_count();
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  FAIL graph_compute\n");
        return 1;
    }
    const int64_t fused = ggml_backend_ameva_get_fused_pair_count() - fused0;

    // reference in double
    double ss = 0.0;
    for (int k = 0; k < K; ++k) ss += (double)xf[k] * xf[k];
    const double sc = 1.0 / std::sqrt(ss / K + eps);
    std::vector<double> ynr(K), yr(K), magk(K);
    for (int k = 0; k < K; ++k) {
        ynr[k] = xf[k] * sc;
        yr[k] = ynr[k] * wnf[k];
    }
    for (int k = 0; k < K; ++k) magk[k] = std::fabs(ynr[k]) + 1e-30;
    check("rms_norm", yn, ynr, magk);
    for (int k = 0; k < K; ++k) magk[k] = std::fabs(yr[k]) + 1e-30;
    check("rms_norm*w", y, yr, magk);
    std::vector<double> m1, m2, m3, m4, m5, m6;
    const std::vector<double> rq = matvec(wq, yr, m1), ro = matvec(wo, yr, m2), ru = matvec(wu, yr, m3);
    const std::vector<double> rg = matvec(wg, yr, m4), rg2 = matvec(wg2, yr, m5), ru2 = matvec(wu2, yr, m6);
    check("W_q y", mq, rq, m1);
    std::vector<double> t(N), tm(N);
    for (int n = 0; n < N; ++n) {
        t[n] = rq[n] + bqf[n];
        tm[n] = m1[n] + std::fabs(bqf[n]);
    }
    check("W_q y + b", q, t, tm);
    check("W_o y", mo, ro, m2);
    for (int n = 0; n < N; ++n) {
        t[n] = rf[n] + ro[n];
        tm[n] = m2[n] + std::fabs(rf[n]);
    }
    check("r + W_o y", o, t, tm);
    check("W_u y", mu, ru, m3);
    check("W_g y", mg, rg, m4);
    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    for (int n = 0; n < N; ++n) {
        t[n] = silu(rg[n]) * ru[n];
        tm[n] = std::fabs(silu(rg[n])) * m3[n] + m4[n] * std::fabs(ru[n]);
    }
    check("swiglu(g, u)", f, t, tm);
    check("W_g2 y", mg2, rg2, m5);
    check("W_u2 y", mu2, ru2, m6);
    for (int n = 0; n < N; ++n) {
        t[n] = silu(rg2[n]) * ru2[n];
        tm[n] = std::fabs(silu(rg2[n])) * m6[n] + m5[n] * std::fabs(ru2[n]);
    }
    check("swiglu(g2, u2)", f2, t, tm);

    const char* fe = getenv("AMEVA_FUSE");
    const int64_t expect = (fe && strcmp(fe, "0") == 0) ? 0 : 5;
    if (fused != expect) {
        printf("  FAIL fused pairs %lld, expected %lld\n", (long long)fused, (long long)expect);
        ++g_failed;
    }
    printf("FUSE_HASH=%016llx fused_pairs=%lld\n", (unsigned long long)g_hash, (long long)fused);
    ggml_backend_buffer_free(abuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    ggml_free(wctx);
    ggml_backend_free(be);
    printf("AMEVA_FUSE_CHECK=%s (%d failed checks)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
