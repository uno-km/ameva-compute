// SPDX-License-Identifier: Apache-2.0
// AMEVA CONV_2D exactness check: ggml_conv_2d_direct through the real ggml backend path (supports_op ->
// graph_compute) on the AMEVA backend, for f16 and f32 kernels, strides, paddings, dilations, channel counts that are
// not multiples of 32, a batch, and f16 weights in the subnormal range.
//
// Each result is compared with an f32 convolution computed here in the summation order of shaders/ameva_conv.comp
// (channel groups of 32, all taps of a group before the next group, taps along w then h, channels ascending), once
// with a fused multiply-add per term and once with a rounded product and a rounded sum. A case passes when the GPU
// result equals one of the two in every element (a driver may or may not fuse the kernel's fma(); zeros of either
// sign are equal). Also required: supports_op says yes for every case and no for a non-contiguous source, and the
// backend's convolution counter advances by one per computed graph, so that a result cannot come from anywhere but
// ameva_conv. Exit status 0 only when every case passes.
//
// Every case runs a second time with a bias in the form sd.cpp builds (ggml_ext_conv_2d): the bias reshaped to
// [1, 1, OC, 1] and added in place of the convolution's result. The graph engine gives that ADD to the kernel
// (ameva_find_chain); expected is the same reference plus the bias, one rounded addition, in every element. Required
// as well: the counter of convolutions with a fused bias and the counter of fused nodes advance by one each, or by
// zero when AMEVA_FUSE=0 runs the ADD as a node of its own (same expected result).
//
// The run is repeated by scripts/termux/lab/conv_check.sh with AMEVA_CONV_STAGE=scalar|vec4, AMEVA_CONV_F16=exact
// and AMEVA_FUSE=0 so that every kernel of the family and both forms of the bias are exercised.
//
// With AMEVA_BINDING_LIMIT=<bytes> the engine treats a tensor above that size as above one buffer binding
// (maxStorageBufferRange) and convolves it in passes over windows of channel groups and in slices of output planes.
// The passes add the terms of an output in the order of the one-pass kernel, so the expected values are the same,
// bit for bit. Under a limit a case may be declined (32 source channels, or the kernel tensor, above the limit): it
// is reported and not counted; the run fails unless the backend's window counter advanced, so a limit that windows
// nothing cannot pass. conv_check.sh runs two limits.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static uint32_t g_seed = 0x2545F491u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

struct Case {
    const char* name;
    enum ggml_type wtype;
    int W, H, IC, OC, B, KW, KH, s0, s1, p0, p1, d0, d1;
    float wscale; // weights are rnd() * wscale (a small scale puts f16 weights into the subnormal range)
};

// ref[b][oc][oh][ow] in ggml's layout; fused: fma per term, otherwise a rounded product and a rounded sum
static void reference(const Case& c, int OW, int OH, const float* x, const float* w, float* ref, bool fused) {
    const int taps = c.KW * c.KH, groups = (c.IC + 31) / 32;
    for (int b = 0; b < c.B; ++b)
        for (int oc = 0; oc < c.OC; ++oc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;
                    for (int g = 0; g < groups; ++g)
                        for (int t = 0; t < taps; ++t) {
                            const int iw = ow * c.s0 + (t % c.KW) * c.d0 - c.p0, ih = oh * c.s1 + (t / c.KW) * c.d1 - c.p1;
                            if (iw < 0 || iw >= c.W || ih < 0 || ih >= c.H) continue;
                            for (int ch = g * 32; ch < c.IC && ch < g * 32 + 32; ++ch) {
                                const float wv = w[t + (size_t)taps * (ch + (size_t)c.IC * oc)];
                                const float xv = x[iw + (size_t)c.W * (ih + (size_t)c.H * (ch + (size_t)c.IC * b))];
                                if (fused) {
                                    acc = __builtin_fmaf(wv, xv, acc);
                                } else {
#pragma clang fp contract(off)
                                    const float prod = wv * xv;
                                    acc = acc + prod;
                                }
                            }
                        }
                    ref[ow + (size_t)OW * (oh + (size_t)OH * (oc + (size_t)c.OC * b))] = acc + 0.0f; // the zero bias
                }
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    const Case cases[] = {
        { "3x3 4->3", GGML_TYPE_F16, 37, 29, 4, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        { "1x1 64->96", GGML_TYPE_F16, 40, 24, 64, 96, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1.0f },
        { "3x3 20->70 stride 2 dilation 2,1", GGML_TYPE_F16, 37, 29, 20, 70, 1, 3, 3, 2, 2, 1, 1, 2, 1, 1.0f },
        { "3x3 33->65 f32 kernel", GGML_TYPE_F32, 70, 33, 33, 65, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        { "3x3 40->33 batch 2", GGML_TYPE_F16, 19, 23, 40, 33, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        { "5x5 35->34 pad 2", GGML_TYPE_F16, 31, 17, 35, 34, 1, 5, 5, 1, 1, 2, 2, 1, 1, 1.0f },
        { "3x3 17->9 stride 2,1 pad 0,2 dilation 1,3 f32", GGML_TYPE_F32, 45, 38, 17, 9, 1, 3, 3, 2, 1, 0, 2, 1, 3, 1.0f },
        { "1x1 32->64 f16 subnormal weights", GGML_TYPE_F16, 23, 9, 32, 64, 1, 1, 1, 1, 1, 0, 0, 1, 1, 6.0e-5f },
        { "3x3 128->128 64x64", GGML_TYPE_F16, 64, 64, 128, 128, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        { "3x3 96->200 no padding f32", GGML_TYPE_F32, 50, 21, 96, 200, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1.0f },
        // sized for AMEVA_BINDING_LIMIT=131072: 32 source channels just fit, so every channel group is a pass of its
        // own (a partial last group, both staging forms, a stride, a batch, the result in slices of its planes), and
        // K-ordered weights that need two passes although source and kernel tensor fit
        { "3x3 70->40 batch 2 f32 (three channel groups)", GGML_TYPE_F32, 31, 33, 70, 40, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        { "3x3 100->70 stride 2 (four channel groups)", GGML_TYPE_F16, 30, 34, 100, 70, 1, 3, 3, 2, 2, 1, 1, 1, 1, 1.0f },
        { "1x1 96->64 (three channel groups)", GGML_TYPE_F16, 32, 31, 96, 64, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1.0f },
        { "3x3 33->96 f32 (weights of two channel groups)", GGML_TYPE_F32, 20, 12, 33, 96, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1.0f },
        // ggml gives one output row although the dilated kernel (9 rows) is larger than the padded source (5 rows):
        // every tap is outside, the result is zero
        { "1x3 dilation 4 on height 1 (kernel larger than the padded source)", GGML_TYPE_F16, 9, 1, 8, 4, 1, 1, 3, 1, 5, 0, 2, 1, 4, 1.0f },
    };
    int passed = 0, total = 0, declined_under_limit = 0;
    const char* limit_env = getenv("AMEVA_BINDING_LIMIT");
    const bool limited = limit_env && *limit_env;
    const int64_t win0 = ggml_backend_ameva_get_window_node_count();
    const char* fuse_env = getenv("AMEVA_FUSE");
    const int fuse_on = fuse_env && !strcmp(fuse_env, "0") ? 0 : 1; // the engine's default is on
    for (const Case& c : cases) {
        total += 2; // without and with a bias
        struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
        struct ggml_context* ctx = ggml_init(ip);
        struct ggml_tensor* a = ggml_new_tensor_4d(ctx, c.wtype, c.KW, c.KH, c.IC, c.OC);
        struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.W, c.H, c.IC, c.B);
        struct ggml_tensor* d = ggml_conv_2d_direct(ctx, a, b, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1);
        // the same convolution with a bias, as sd.cpp builds it
        struct ggml_tensor* bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.OC);
        struct ggml_tensor* dc = ggml_conv_2d_direct(ctx, a, b, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1);
        struct ggml_tensor* db = ggml_add_inplace(ctx, dc, ggml_reshape_4d(ctx, bias, 1, 1, c.OC, 1));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
        if (!buf) {
            printf("CONV_CHECK case=\"%s\" FAIL: allocation (both forms)\n", c.name);
            ggml_free(ctx);
            continue;
        }
        if (limited && !ggml_backend_supports_op(be, d)) {
            printf("CONV_CHECK case=\"%s\" DECLINED under AMEVA_BINDING_LIMIT=%s (both forms, not counted)\n", c.name, limit_env);
            total -= 2;
            ++declined_under_limit;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        const int OW = (int)d->ne[0], OH = (int)d->ne[1];
        const size_t nw = (size_t)ggml_nelements(a), nx = (size_t)ggml_nelements(b), nd = (size_t)ggml_nelements(d);
        std::vector<float> w(nw), x(nx), got(nd), ref_fma(nd), ref_ma(nd), bv((size_t)c.OC);
        for (float& v : w) v = rnd() * c.wscale;
        for (float& v : x) v = rnd();
        for (float& v : bv) v = rnd();
        ggml_backend_tensor_set(bias, bv.data(), 0, bv.size() * sizeof(float));
        size_t subnormal = 0;
        if (c.wtype == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(nw);
            ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t)nw);
            ggml_fp16_to_fp32_row(h.data(), w.data(), (int64_t)nw); // the reference uses the stored values
            for (ggml_fp16_t v : h) subnormal += (v & 0x7C00) == 0 && (v & 0x03FF) != 0;
            ggml_backend_tensor_set(a, h.data(), 0, nw * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(a, w.data(), 0, nw * sizeof(float));
        }
        ggml_backend_tensor_set(b, x.data(), 0, nx * sizeof(float));
        const char* fail = nullptr;
        if (!ggml_backend_supports_op(be, d)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, d);
        const int64_t n0 = ggml_backend_ameva_get_conv_count();
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail && ggml_backend_ameva_get_conv_count() - n0 != 1) fail = "the convolution counter did not advance by one";
        size_t bad_fma = 0, bad_ma = 0, nan = 0;
        double maxd = 0, ss = 0;
        if (!fail) {
            ggml_backend_tensor_get(d, got.data(), 0, nd * sizeof(float));
            reference(c, OW, OH, x.data(), w.data(), ref_fma.data(), true);
            reference(c, OW, OH, x.data(), w.data(), ref_ma.data(), false);
            for (size_t i = 0; i < nd; ++i) {
                nan += got[i] != got[i];
                bad_fma += !(got[i] == ref_fma[i]);
                bad_ma += !(got[i] == ref_ma[i]);
                ss += (double)ref_fma[i] * ref_fma[i];
                maxd = std::fmax(maxd, std::fabs((double)got[i] - (double)ref_fma[i]));
            }
            if (nan) fail = "NaN in the result";
            else if (bad_fma && bad_ma) fail = "differs from both references";
        }
        const double rms = std::sqrt(ss / (double)nd);
        printf("CONV_CHECK case=\"%s\" kernel=%s out=%dx%dx%dx%d f16_subnormal_weights=%zu equal_to=%s "
               "max_diff_over_rms_vs_fma=%.2e %s%s\n",
               c.name, ggml_type_name(c.wtype), OW, OH, c.OC, c.B, subnormal,
               fail && strcmp(fail, "differs from both references") != 0 ? "-" : !bad_fma ? "fma" : !bad_ma ? "muladd" : "neither",
               rms > 0 ? maxd / rms : maxd, fail ? "FAIL: " : "PASS", fail ? fail : "");
        if (!fail) ++passed;

        // with the bias: every element is the reference plus the bias of its output channel
        const char* failb = fail ? "the case without a bias failed" : nullptr;
        if (!failb && (!ggml_backend_supports_op(be, dc) || !ggml_backend_supports_op(be, db))) failb = "supports_op says no";
        struct ggml_cgraph* gb = ggml_new_graph(ctx);
        ggml_build_forward_expand(gb, db);
        const int64_t c0 = ggml_backend_ameva_get_conv_count(), cb0 = ggml_backend_ameva_get_conv_bias_count();
        const int64_t f0 = ggml_backend_ameva_get_fused_pair_count();
        if (!failb && ggml_backend_graph_compute(be, gb) != GGML_STATUS_SUCCESS) failb = "graph_compute failed";
        const int64_t dconv = ggml_backend_ameva_get_conv_count() - c0;
        const int64_t dbias = ggml_backend_ameva_get_conv_bias_count() - cb0;
        const int64_t dfused = ggml_backend_ameva_get_fused_pair_count() - f0;
        if (!failb && dconv != 1) failb = "the convolution counter did not advance by one";
        if (!failb && (dbias != fuse_on || dfused != fuse_on)) failb = "the bias was not computed the expected way (counters)";
        size_t badb_fma = 0, badb_ma = 0, nanb = 0;
        if (!failb) {
            ggml_backend_tensor_get(db, got.data(), 0, nd * sizeof(float));
            const size_t plane = (size_t)OW * OH;
            for (size_t i = 0; i < nd; ++i) {
                const float bo = bv[(i / plane) % (size_t)c.OC];
                nanb += got[i] != got[i];
                badb_fma += !(got[i] == ref_fma[i] + bo);
                badb_ma += !(got[i] == ref_ma[i] + bo);
            }
            if (nanb) failb = "NaN in the result";
            else if (badb_fma && badb_ma) failb = "differs from both references plus the bias";
        }
        printf("CONV_CHECK case=\"%s + bias\" bias=%s equal_to=%s mismatches_fma=%zu mismatches_muladd=%zu %s%s\n", c.name,
               dbias == 1 ? "fused" : "separate", failb ? "-" : !badb_fma ? "fma" : "muladd", badb_fma, badb_ma,
               failb ? "FAIL: " : "PASS", failb ? failb : "");
        if (!failb) ++passed;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // a source that is a permuted view is not contiguous: the op must be declined, not computed some other way
    ++total;
    {
        struct ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        struct ggml_context* ctx = ggml_init(ip);
        struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 8, 8);
        struct ggml_tensor* b = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 12, 10, 1), 2, 0, 1, 3); // [12,10,8,1]
        struct ggml_tensor* d = ggml_conv_2d_direct(ctx, a, b, 1, 1, 1, 1, 1, 1);
        const bool declined = !ggml_backend_supports_op(be, d);
        printf("CONV_CHECK case=\"non-contiguous source is declined\" %s\n", declined ? "PASS" : "FAIL: supports_op says yes");
        passed += declined;
        ggml_free(ctx);
    }
    // AMEVA_CONV_CHECK_TIME="W,H,IC,OC,K[ W,H,IC,OC,K...]": wall time of graph_compute for a K x K convolution with
    // an f16 kernel and padding K / 2 (one run that is not counted, then four). Timing only: these outputs are not
    // compared, the cases above are the check.
    if (const char* spec = getenv("AMEVA_CONV_CHECK_TIME")) {
        for (const char* p = spec; *p;) {
            int W = 0, H = 0, IC = 0, OC = 0, K = 0, used = 0;
            if (sscanf(p, "%d,%d,%d,%d,%d%n", &W, &H, &IC, &OC, &K, &used) != 5 || W < 1 || H < 1 || IC < 1 || OC < 1 || K < 1) {
                fprintf(stderr, "AMEVA_CONV_CHECK_TIME: bad shape at \"%s\"\n", p);
                return 2;
            }
            p += used;
            while (*p == ' ') ++p;
            struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            struct ggml_context* ctx = ggml_init(ip);
            struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, K, K, IC, OC);
            struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, 1);
            struct ggml_tensor* d = ggml_conv_2d_direct(ctx, a, b, 1, 1, K / 2, K / 2, 1, 1);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
            if (!buf || !ggml_backend_supports_op(be, d)) {
                printf("CONV_TIME %dx%dx%d->%d %dx%d: not allocated or not supported\n", W, H, IC, OC, K, K);
                return 1;
            }
            std::vector<float> w((size_t)ggml_nelements(a)), x((size_t)ggml_nelements(b));
            for (float& v : w) v = rnd();
            for (float& v : x) v = rnd();
            std::vector<ggml_fp16_t> h(w.size());
            ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t)w.size());
            ggml_backend_tensor_set(a, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, d);
            double ms = 0;
            bool ok = true;
            for (int it = -1; it < 4 && ok; ++it) {
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                if (it >= 0) ms += (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
            }
            const double gflop = 2.0 * (double)d->ne[0] * (double)d->ne[1] * OC * (double)IC * K * K * 1e-9;
            if (ok) printf("CONV_TIME %dx%dx%d->%d %dx%d f16: %.1f ms per graph_compute (%.2f GFLOP, %.1f GFLOPS)\n", W, H, IC, OC,
                           K, K, ms / 4, gflop, gflop / (ms / 4 * 1e-3));
            else printf("CONV_TIME %dx%dx%d->%d %dx%d: graph_compute failed\n", W, H, IC, OC, K, K);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            if (!ok) return 1;
        }
    }
    const int dec = ggml_backend_ameva_get_conv_f16_decoder();
    const int64_t windowed = ggml_backend_ameva_get_window_node_count() - win0;
    printf("CONV_CHECK %d/%d passed, f16 decoder=%s, convolutions on the GPU=%lld, with a fused bias=%lld, "
           "in windows=%lld, binding limit=%s, declined under it=%d\n", passed, total,
           dec == 1 ? "built-in conversion" : dec == 0 ? "exact" : "not determined",
           (long long)ggml_backend_ameva_get_conv_count(), (long long)ggml_backend_ameva_get_conv_bias_count(),
           (long long)windowed, limited ? limit_env : "device", declined_under_limit);
    if (limited && windowed == 0) {
        printf("CONV_CHECK FAIL: AMEVA_BINDING_LIMIT=%s put no convolution into windows\n", limit_env);
        ggml_backend_free(be);
        return 1;
    }
    ggml_backend_free(be);
    return passed == total ? 0 : 1;
}
