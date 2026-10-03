// SPDX-License-Identifier: Apache-2.0
// AMEVA MUL_MAT exactness check: every weight type through the real ggml backend path (supports_op -> graph_compute).
//
// For each type and shape the weights are produced by ggml itself (ggml_quantize_chunk / fp16 / bf16 conversion),
// the reference is computed in double from ggml's own dequantization (type_traits.to_float), and the GPU result
// must match it to a relative error that only fp32 accumulation can explain. A wrong bit in any decoder shows up
// as an error of the size of a quantization step, orders of magnitude above the bound.
//
// Placements of the weight (all tensors live in AMEVA buffers; the engine never stages other buffers):
//   resident  buffer usage ANY: ggml layout, ameva_qmm
//   weights   buffer usage WEIGHTS: quantized weights are stored repacked (ggml-ameva-repack.h), ameva_qmv
//   split     as weights, uploaded in two writes that do not end on a row boundary (read-modify-write path)
//   view      as weights, src0 is a range of rows of a larger repacked weight
// For the weight placements the repacked-tensor and qmv counters must move exactly as expected (ground truth that
// the repacked path ran, or that a type without a repacked layout did not take it), and reading the weight back
// through the buffer interface must return the uploaded ggml bytes.
// Exit 0 only if every case passes and supports_op accepted every case.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0x9e3779b9u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

enum Placement { RESIDENT, WEIGHTS, SPLIT, VIEW };
static const char* placement_name(Placement p) {
    return p == RESIDENT ? "resident" : p == WEIGHTS ? "weights" : p == SPLIT ? "split" : "view";
}

struct Result {
    bool supported;
    bool pass;
    double max_rel;
    const char* why; // first failed check other than the numeric bound
};

static bool repackable(enum ggml_type t) {
    return t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16;
}

static Result run_case(ggml_backend_t be, enum ggml_type type, int M, int N, int K, Placement place, bool repack_on) {
    Result res = {false, false, 0.0, nullptr};
    const int VIEW_ROW0 = 3, VIEW_EXTRA = 5; // the view takes rows [3, 3 + N) of a weight with N + 5 rows
    const int NW = place == VIEW ? N + VIEW_EXTRA : N;
    std::vector<float> wf((size_t)NW * K), af((size_t)M * K), wref((size_t)NW * K);
    for (float& v : wf) v = rnd();
    for (float& v : af) v = rnd();

    const size_t row_bytes = ggml_row_size(type, K);
    const size_t wbytes = row_bytes * NW;
    std::vector<uint8_t> wq(wbytes);
    if (type == GGML_TYPE_F32) {
        memcpy(wq.data(), wf.data(), wbytes);
        wref = wf;
    } else {
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, NW, K, nullptr);
        const struct ggml_type_traits* tt = ggml_get_type_traits(type);
        for (int n = 0; n < NW; ++n) {
            tt->to_float(wq.data() + row_bytes * n, wref.data() + (size_t)n * K, K);
        }
    }

    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, type, K, NW);
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* src0 = place == VIEW ? ggml_view_2d(ctx, w, K, N, w->nb[1], (size_t)VIEW_ROW0 * w->nb[1]) : w;
    struct ggml_tensor* out = ggml_mul_mat(ctx, src0, a);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    if (!buf) {
        fprintf(stderr, "allocation failed\n");
        ggml_free(ctx);
        return res;
    }
    if (place != RESIDENT) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    res.supported = ggml_backend_supports_op(be, out);
    if (!res.supported) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    const int64_t repacked0 = ggml_backend_ameva_get_repacked_tensor_count();
    const int64_t qmv0 = ggml_backend_ameva_get_qmv_mul_mat_count();
    if (place == SPLIT) {
        // the first write ends inside a row and is not the whole tensor
        const size_t cut = wbytes / 3 + 1;
        ggml_backend_tensor_set(w, wq.data(), 0, cut);
        ggml_backend_tensor_set(w, wq.data() + cut, cut, wbytes - cut);
    } else {
        ggml_backend_tensor_set(w, wq.data(), 0, wbytes);
    }
    ggml_backend_tensor_set(a, af.data(), 0, af.size() * sizeof(float));
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        res.why = "graph_compute failed";
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    std::vector<float> got((size_t)M * N);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    const bool expect_repacked = repack_on && place != RESIDENT && repackable(type);
    const int64_t d_repacked = ggml_backend_ameva_get_repacked_tensor_count() - repacked0;
    const int64_t d_qmv = ggml_backend_ameva_get_qmv_mul_mat_count() - qmv0;
    if (d_repacked != (expect_repacked ? 1 : 0)) res.why = "repacked tensor count did not move as expected";
    else if (d_qmv != (expect_repacked ? 1 : 0)) res.why = "qmv MUL_MAT count did not move as expected";
    if (!res.why) {
        // the buffer interface returns the ggml bytes that were uploaded, whatever the stored layout
        std::vector<uint8_t> back(wbytes);
        ggml_backend_tensor_get(w, back.data(), 0, wbytes);
        if (memcmp(back.data(), wq.data(), wbytes) != 0) res.why = "weight bytes read back differ from the upload";
        // and so does a read that starts and ends inside rows
        const size_t o = row_bytes / 2 + 1, n = wbytes - row_bytes;
        std::vector<uint8_t> part(n);
        if (!res.why && o + n <= wbytes) {
            ggml_backend_tensor_get(w, part.data(), o, n);
            if (memcmp(part.data(), wq.data() + o, n) != 0) res.why = "partial read differs from the upload";
        }
    }

    bool pass = true;
    double max_rel = 0.0;
    const int r0 = place == VIEW ? VIEW_ROW0 : 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double p = (double)wref[(size_t)(r0 + n) * K + k] * (double)af[(size_t)m * K + k];
                ref += p;
                mag += std::fabs(p);
            }
            const double g = got[(size_t)m * N + n];
            const double rel = std::fabs(g - ref) / (mag + 1e-30);
            if (!std::isfinite(g) || rel > 1e-5) pass = false;
            if (rel > max_rel) max_rel = rel;
        }
    }
    res.pass = pass && !res.why;
    res.max_rel = max_rel;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    // AMEVA buffers are host buffers exactly when repacking is off (AMEVA_REPACK=0)
    const bool repack_on = !ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(be));
    const enum ggml_type types[] = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_1,
                                    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,   GGML_TYPE_Q3_K,
                                    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS};
    struct Shape { int M, N, K; };
    // M = 1: one-column kernels; 5 and 3: ameva_qmv with 8 activation columns (partial tile); 8, 13, 33 and 40:
    // repacked weights through ameva_qgemm (from 8 activation rows), partial 64-row and 32-column tiles. Odd N
    // exercises the row bound; K = 2560 and 4096 give long rows (many units per lane in ameva_qmv).
    const Shape shapes[] = {{1, 37, 256},  {5, 64, 512},   {8, 16, 256},  {13, 33, 768},
                            {1, 70, 2560}, {3, 19, 4096}, {40, 70, 512}, {33, 130, 256}};
    const Placement places[] = {RESIDENT, WEIGHTS, SPLIT, VIEW};
    printf("repacking %s\n", repack_on ? "on" : "off (AMEVA_REPACK=0)");

    int total = 0, failed = 0;
    for (enum ggml_type t : types) {
        double worst = 0.0;
        bool type_ok = true;
        int cases = 0;
        for (const Shape& s : shapes) {
            for (Placement p : places) {
                ++total;
                ++cases;
                const Result r = run_case(be, t, s.M, s.N, s.K, p, repack_on);
                const bool ok = r.supported && r.pass;
                if (!ok) {
                    ++failed;
                    type_ok = false;
                    printf("  FAIL type=%s M=%d N=%d K=%d %s supported=%d max_rel=%.3g%s%s\n", ggml_type_name(t), s.M,
                           s.N, s.K, placement_name(p), (int)r.supported, r.max_rel, r.why ? " : " : "",
                           r.why ? r.why : "");
                }
                if (r.max_rel > worst) worst = r.max_rel;
            }
        }
        printf("QMM type=%-6s cases=%d max_rel_err=%.3g status=%s\n", ggml_type_name(t), cases, worst,
               type_ok ? "PASS" : "FAIL");
    }
    ggml_backend_free(be);
    printf("AMEVA_QMM_CHECK=%s (%d/%d cases)\n", failed ? "FAIL" : "PASS", total - failed, total);
    return failed ? 1 : 0;
}
