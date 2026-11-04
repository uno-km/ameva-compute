// SPDX-License-Identifier: Apache-2.0
// AMEVA flash-attention timing at diffusion shapes: one GGML_OP_FLASH_ATTN_EXT node (q f32 [D, NQ, H], k and v f16
// [D, NKV, H], no mask, scale 1/sqrt(D)) through the AMEVA backend, the way stable-diffusion.cpp's UNet calls it with
// diffusion flash attention on. SD1.5 at 512x512: D 40 / NQ = NKV 4096 / H 8 (self-attention at 64x64), D 40 / NKV 77
// (cross-attention), D 80 / 1024, D 160 / 256.
//
//   ameva_fa_bench D NQ NKV H [ITER]
//
// One run is checked against a reference in double on CPU_ROWS query rows of every head (default 48: the first,
// the last and rows spread between): largest |gpu - ref| over the largest |ref|, must stay below 1e-4. Then ITER
// timed runs (default 5) after one that is not counted; wall time per run from graph_compute to its return (the
// backend waits for the GPU). The variant is the engine's choice, or AMEVA_FA_VARIANT=<index> (ggml-ameva).
// Exit status 0 only when the check passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t g_seed = 0x2545F491u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s D NQ NKV H [ITER]\n", argv[0]);
        return 2;
    }
    const int D = atoi(argv[1]), NQ = atoi(argv[2]), NKV = atoi(argv[3]), H = atoi(argv[4]);
    const int ITER = argc > 5 ? atoi(argv[5]) : 5;
    const int CPU_ROWS = getenv("CPU_ROWS") ? atoi(getenv("CPU_ROWS")) : 48;
    if (D <= 0 || NQ <= 0 || NKV <= 0 || H <= 0 || ITER <= 0) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, H);
    struct ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, NKV, H);
    struct ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, NKV, H);
    const float scale = 1.0f / std::sqrt((float)D);
    struct ggml_tensor* o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    if (!buf) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    if (!ggml_backend_supports_op(be, o)) {
        fprintf(stderr, "FA_BENCH supports_op says no for D=%d\n", D);
        return 1;
    }
    const size_t nq = (size_t)D * NQ * H, nk = (size_t)D * NKV * H;
    std::vector<float> qv(nq), kf(nk), vf(nk);
    std::vector<ggml_fp16_t> kh(nk), vh(nk);
    for (float& x : qv) x = rnd();
    for (size_t i = 0; i < nk; ++i) {
        kh[i] = ggml_fp32_to_fp16(rnd());
        vh[i] = ggml_fp32_to_fp16(rnd());
        kf[i] = ggml_fp16_to_fp32(kh[i]);
        vf[i] = ggml_fp16_to_fp32(vh[i]);
    }
    ggml_backend_tensor_set(q, qv.data(), 0, nq * sizeof(float));
    ggml_backend_tensor_set(k, kh.data(), 0, nk * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, vh.data(), 0, nk * sizeof(ggml_fp16_t));
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FA_BENCH graph_compute failed\n");
        return 1;
    }
    // o: [D, H, NQ] contiguous
    std::vector<float> got((size_t)D * H * NQ);
    ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
    double worst = 0, refmax = 0;
    size_t nan = 0;
    std::vector<double> s(NKV), acc(D);
    for (int h = 0; h < H; ++h) {
        for (int r = 0; r < CPU_ROWS && r < NQ; ++r) {
            const int iq = CPU_ROWS >= NQ ? r : (int)((int64_t)r * (NQ - 1) / std::max(1, CPU_ROWS - 1));
            const float* qr = qv.data() + ((size_t)h * NQ + iq) * D;
            double m = -1e300;
            for (int j = 0; j < NKV; ++j) {
                const float* kr = kf.data() + ((size_t)h * NKV + j) * D;
                double d = 0;
                for (int c = 0; c < D; ++c) d += (double)qr[c] * kr[c];
                s[j] = d * scale;
                m = std::max(m, s[j]);
            }
            double l = 0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < NKV; ++j) {
                const double p = std::exp(s[j] - m);
                l += p;
                const float* vr = vf.data() + ((size_t)h * NKV + j) * D;
                for (int c = 0; c < D; ++c) acc[c] += p * vr[c];
            }
            const float* g = got.data() + ((size_t)iq * H + h) * D;
            for (int c = 0; c < D; ++c) {
                const double ref = acc[c] / l;
                if (std::isnan(g[c])) ++nan;
                worst = std::max(worst, std::fabs(g[c] - ref));
                refmax = std::max(refmax, std::fabs(ref));
            }
        }
    }
    const double rel = refmax > 0 ? worst / refmax : 1.0;
    const bool ok = nan == 0 && rel < 1e-4;
    std::vector<double> ms;
    for (int it = 0; it <= ITER; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FA_BENCH graph_compute failed in run %d\n", it);
            return 1;
        }
        const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (it > 0) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double mean = 0;
    for (double t : ms) mean += t;
    mean /= (double)ms.size();
    const double gflop = 4.0 * D * (double)NQ * NKV * H / 1e9; // QK^T and PV
    printf("FA_BENCH D=%d NQ=%d NKV=%d H=%d variant=%s ms=%.1f min_ms=%.1f gflops=%.1f max_err_over_refmax=%.2e nan=%zu %s\n",
           D, NQ, NKV, H, getenv("AMEVA_FA_VARIANT") ? getenv("AMEVA_FA_VARIANT") : "auto", mean, ms[0],
           gflop / (mean / 1e3), rel, nan, ok ? "MATCH" : "MISMATCH");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return ok ? 0 : 1;
}
