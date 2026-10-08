// SPDX-License-Identifier: Apache-2.0
// AMEVA GROUP_NORM check: ggml_group_norm through the real ggml backend path (supports_op -> graph_compute) on the
// AMEVA backend, alone and with the nodes sd.cpp puts behind it (GroupNorm32 + SiLU of the 2D decoders: MUL by a
// per-channel weight, ADD of a per-channel bias, SILU, each stored in place of the result before), which the graph
// engine computes in the dispatch of the norm (ameva_find_chain, shaders/ameva_cnorm.comp).
//
// Per case four graphs on the same input:
//   plain     GROUP_NORM
//   separate  GROUP_NORM, MUL, ADD, SILU as nodes with memory of their own (nothing is fused)
//   fused2    GROUP_NORM, then MUL and ADD in place (fused: 2 nodes)
//   fused3    GROUP_NORM in place of its input, then MUL, ADD, SILU in place (fused: 3 nodes)
// Required:
//   * plain, separate and fused3 lie within a rounding bound of a reference computed here as ggml's CPU code does
//     (mean and variance accumulated in double, every other step in f32). The kernel accumulates in f32 partial
//     sums, so it is not bit-equal to that reference; the bound is the worst case of those sums, see tolerance().
//   * fused2 equals the ADD of `separate` in every bit: the fused kernel rounds each step as the separate kernels do.
//     For fused3 against the SILU of `separate` the number of differing elements is printed, not judged (exp is
//     evaluated in two different kernels).
//   * supports_op says yes for every node and no for a non-contiguous source; the backend's GROUP_NORM counter
//     advances by one per graph and the counter of fused nodes by 0, 0, 2, 3 (by 0 for all with AMEVA_FUSE=0), so
//     that a result cannot come from another path than the one named.
// Then one block as in a decoder: GROUP_NORM, MUL, ADD, SILU, CONV_2D 3x3, ADD of the convolution bias, all in sd.cpp's
// form (5 fused nodes), against an f32 reference. Exit status 0 only when every case passes.
//
// AMEVA_CNORM_CHECK_TIME="W,H,C[ W,H,C...]": wall time of graph_compute for plain, separate and fused3 at 32 groups.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static uint32_t g_seed = 0x9E3779B9u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

struct Case {
    const char* name;
    int W, H, C, B, groups;
    float eps, offset, spread; // input = offset + spread * rnd()
};

struct Geometry {
    size_t plane, per_batch, n;
    int cpg, G; // channels per group, groups that have channels (ggml: ceil(C / groups) channels per group)
};

static Geometry geometry(const Case& c) {
    Geometry g;
    g.plane = (size_t)c.W * c.H;
    g.per_batch = g.plane * c.C;
    g.n = g.per_batch * c.B;
    g.cpg = (c.C + c.groups - 1) / c.groups;
    g.G = (c.C + g.cpg - 1) / g.cpg;
    return g;
}

// ggml_compute_forward_group_norm_f32. units[b * G + group] = (elements / 512 + 32) * (1 + |mean| * scale): the
// factor of FLT_EPSILON by which the kernel's f32 statistics can move a normalised value of that group (tolerance()).
static void reference_norm(const Case& c, const Geometry& g, const float* x, float* y, std::vector<double>& units) {
    units.assign((size_t)g.G * c.B, 0.0);
    for (int b = 0; b < c.B; ++b) {
        for (int gi = 0; gi < g.G; ++gi) {
            const int c0 = gi * g.cpg, c1 = c0 + g.cpg < c.C ? c0 + g.cpg : c.C;
            const size_t off = (size_t)b * g.per_batch + (size_t)c0 * g.plane, n = (size_t)(c1 - c0) * g.plane;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) sum += (double)x[off + i];
            const float mean = (float)(sum / (double)n);
            double sum2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const float v = x[off + i] - mean;
                y[off + i] = v;
                sum2 += (double)(v * v);
            }
            const float variance = (float)(sum2 / (double)n);
            const float scale = 1.0f / sqrtf(variance + c.eps);
            for (size_t i = 0; i < n; ++i) y[off + i] *= scale;
            units[(size_t)b * g.G + gi] = ((double)n / 512.0 + 32.0) * (1.0 + std::fabs((double)mean) * (double)scale);
        }
    }
}

// Allowed |GPU - reference| of one element.
//   norm (stage 0): the kernel sums a group in 512 f32 chains of n / 512 terms, so its mean is off by at most
//     (n / 512) eps (|mean| + sigma), which moves y by that times scale <= (n / 512) eps (1 + |mean| scale); the
//     variance sum moves the scale by the same relative amount, i.e. y by (n / 512) eps |y|. The 32 added to
//     n / 512 covers the rounding of the f32 steps and of sqrt and the division: units * eps * (1 + |y|).
//   MUL, ADD (stage 1): that error times |w|, plus the rounding of the product and of the sum.
//   SILU (stage 2): the slope of silu is below 1.1; exp within the 3 + 2 |a| ulp Vulkan allows.
static double tolerance(int stage, double units, float yn, float w, float a, float s) {
    const double eps = (double)FLT_EPSILON;
    const double tn = units * eps * (1.0 + std::fabs((double)yn));
    if (stage == 0) return tn;
    const double ta = tn * std::fabs((double)w) + 2.0 * eps * (std::fabs((double)yn * (double)w) + std::fabs((double)a)) + 1e-30;
    if (stage == 1) return ta;
    return 1.1 * ta + eps * (16.0 + 8.0 * std::fabs((double)a)) * (std::fabs((double)s) + 1e-30);
}

struct Compare {
    double worst = 0; // largest |got - ref| / tolerance
    double max_abs = 0;
    size_t nan = 0;
};

static double ms_between(const struct timespec& t0, const struct timespec& t1) {
    return (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    const char* fuse_env = getenv("AMEVA_FUSE");
    const int fuse_on = fuse_env && !strcmp(fuse_env, "0") ? 0 : 1; // the engine's default is on
    const Case cases[] = {
        { "16x16x64 32 groups", 16, 16, 64, 1, 32, 1e-6f, 0.5f, 1.0f },
        { "13x7x40 32 groups (2 channels per group, 12 groups without channels)", 13, 7, 40, 1, 32, 1e-6f, 0.25f, 1.0f },
        { "9x5x10 4 groups batch 2 (last group short, plane not a multiple of 4)", 9, 5, 10, 2, 4, 1e-6f, -0.5f, 2.0f },
        { "5x5x7 1 group batch 3", 5, 5, 7, 3, 1, 1e-5f, 1.0f, 1.0f },
        { "32x32x512 32 groups", 32, 32, 512, 1, 32, 1e-6f, 0.1f, 1.0f },
        { "128x128x256 32 groups", 128, 128, 256, 1, 32, 1e-6f, 0.3f, 1.5f },
        { "24x24x64 32 groups, mean 30 against spread 1", 24, 24, 64, 1, 32, 1e-6f, 30.0f, 1.0f },
        { "16x16x32 8 groups, variance below eps", 16, 16, 32, 1, 8, 1e-6f, 0.0f, 1e-3f },
        { "1x1x64 32 groups (2 elements per group)", 1, 1, 64, 1, 32, 1e-6f, 0.5f, 1.0f },
        { "8x8x16 16 groups (one channel per group)", 8, 8, 16, 1, 16, 1e-6f, 0.5f, 1.0f },
        { "6x6x3 8 groups (more groups than channels)", 6, 6, 3, 1, 8, 1e-6f, 0.5f, 1.0f },
        { "300x7x33 3 groups (plane wider than one workgroup)", 300, 7, 33, 1, 3, 1e-6f, 0.5f, 1.0f },
    };
    int passed = 0, total = 0;
    for (const Case& c : cases) {
        ++total;
        const Geometry g = geometry(c);
        struct ggml_init_params ip = { 40 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
        struct ggml_context* ctx = ggml_init(ip);
        struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.W, c.H, c.C, c.B);
        struct ggml_tensor* x2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.W, c.H, c.C, c.B);
        struct ggml_tensor* wt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.C);
        struct ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.C);
        struct ggml_tensor* plain = ggml_group_norm(ctx, x, c.groups, c.eps);
        struct ggml_tensor* sep_m = ggml_mul(ctx, plain, ggml_reshape_4d(ctx, wt, 1, 1, c.C, 1));
        struct ggml_tensor* sep_a = ggml_add(ctx, sep_m, ggml_reshape_4d(ctx, bt, 1, 1, c.C, 1));
        struct ggml_tensor* sep_s = ggml_silu(ctx, sep_a);
        // sd.cpp: ggml_ext_group_norm, then ggml_silu_inplace
        struct ggml_tensor* f2n = ggml_group_norm(ctx, x, c.groups, c.eps);
        struct ggml_tensor* f2 = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, f2n, ggml_reshape_4d(ctx, wt, 1, 1, c.C, 1)),
                                                  ggml_reshape_4d(ctx, bt, 1, 1, c.C, 1));
        struct ggml_tensor* f3n = ggml_group_norm_inplace(ctx, x2, c.groups, c.eps);
        struct ggml_tensor* f3 = ggml_silu_inplace(
            ctx, ggml_add_inplace(ctx, ggml_mul_inplace(ctx, f3n, ggml_reshape_4d(ctx, wt, 1, 1, c.C, 1)),
                                  ggml_reshape_4d(ctx, bt, 1, 1, c.C, 1)));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
        if (!buf) {
            printf("CNORM_CHECK case=\"%s\" FAIL: allocation\n", c.name);
            ggml_free(ctx);
            continue;
        }
        std::vector<float> xv(g.n), wv((size_t)c.C), bv((size_t)c.C);
        for (float& v : xv) v = c.offset + c.spread * rnd();
        for (float& v : wv) v = 0.2f + 1.5f * rnd();
        for (float& v : bv) v = rnd();
        ggml_backend_tensor_set(x, xv.data(), 0, g.n * sizeof(float));
        ggml_backend_tensor_set(x2, xv.data(), 0, g.n * sizeof(float));
        ggml_backend_tensor_set(wt, wv.data(), 0, wv.size() * sizeof(float));
        ggml_backend_tensor_set(bt, bv.data(), 0, bv.size() * sizeof(float));

        // reference of the norm, the ADD and the SILU
        std::vector<float> rn(g.n), ra(g.n), rs(g.n);
        std::vector<double> units;
        reference_norm(c, g, xv.data(), rn.data(), units);
        for (size_t i = 0; i < g.n; ++i) {
            const size_t ch = (i / g.plane) % (size_t)c.C;
            const float m = rn[i] * wv[ch];
            ra[i] = m + bv[ch];
            rs[i] = ra[i] / (1.0f + expf(-ra[i]));
        }
        auto compare = [&](const float* got, int stage) {
            Compare r;
            for (size_t i = 0; i < g.n; ++i) {
                const size_t ch = (i / g.plane) % (size_t)c.C;
                const double u = units[(i / g.per_batch) * (size_t)g.G + ch / (size_t)g.cpg];
                const float ref = stage == 0 ? rn[i] : stage == 1 ? ra[i] : rs[i];
                const double d = std::fabs((double)got[i] - (double)ref);
                r.nan += got[i] != got[i];
                r.max_abs = std::fmax(r.max_abs, d);
                r.worst = std::fmax(r.worst, d / tolerance(stage, u, rn[i], wv[ch], ra[i], rs[i]));
            }
            return r;
        };

        const char* fail = nullptr;
        for (struct ggml_tensor* t : { plain, sep_m, sep_a, sep_s, f2n, f2, f3n, f3 }) {
            if (!fail && !ggml_backend_supports_op(be, t)) fail = "supports_op says no";
        }
        // one graph: its result, and how far the two counters moved
        int64_t d_norm = 0, d_fused = 0;
        auto run = [&](struct ggml_tensor* out) {
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            const int64_t n0 = ggml_backend_ameva_get_group_norm_count(), f0 = ggml_backend_ameva_get_fused_pair_count();
            const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
            d_norm = ggml_backend_ameva_get_group_norm_count() - n0;
            d_fused = ggml_backend_ameva_get_fused_pair_count() - f0;
            return ok;
        };
        std::vector<float> got(g.n), sa(g.n), ss(g.n);
        Compare cp, ca, cs, c3;
        size_t f2_diff = 0, f3_diff = 0;
        int fused2 = -1, fused3 = -1;
        if (!fail && !run(plain)) fail = "graph_compute failed (plain)";
        if (!fail && (d_norm != 1 || d_fused != 0)) fail = "counters (plain)";
        if (!fail) {
            ggml_backend_tensor_get(plain, got.data(), 0, g.n * sizeof(float));
            cp = compare(got.data(), 0);
            if (cp.nan) fail = "NaN (plain)";
            else if (cp.worst > 1.0) fail = "plain outside the rounding bound";
        }
        if (!fail && !run(sep_s)) fail = "graph_compute failed (separate)";
        if (!fail && (d_norm != 1 || d_fused != 0)) fail = "counters (separate)";
        if (!fail) {
            ggml_backend_tensor_get(sep_a, sa.data(), 0, g.n * sizeof(float));
            ggml_backend_tensor_get(sep_s, ss.data(), 0, g.n * sizeof(float));
            ca = compare(sa.data(), 1);
            cs = compare(ss.data(), 2);
            if (ca.nan || cs.nan) fail = "NaN (separate)";
            else if (ca.worst > 1.0 || cs.worst > 1.0) fail = "separate outside the rounding bound";
        }
        if (!fail && !run(f2)) fail = "graph_compute failed (fused2)";
        fused2 = (int)d_fused;
        if (!fail && (d_norm != 1 || d_fused != 2 * fuse_on)) fail = "counters (fused2)";
        if (!fail) {
            ggml_backend_tensor_get(f2, got.data(), 0, g.n * sizeof(float));
            for (size_t i = 0; i < g.n; ++i) f2_diff += memcmp(&got[i], &sa[i], sizeof(float)) != 0;
            if (f2_diff) fail = "fused2 differs from the separate nodes";
        }
        if (!fail && !run(f3)) fail = "graph_compute failed (fused3)";
        fused3 = (int)d_fused;
        if (!fail && (d_norm != 1 || d_fused != 3 * fuse_on)) fail = "counters (fused3)";
        if (!fail) {
            ggml_backend_tensor_get(f3, got.data(), 0, g.n * sizeof(float));
            c3 = compare(got.data(), 2);
            for (size_t i = 0; i < g.n; ++i) f3_diff += memcmp(&got[i], &ss[i], sizeof(float)) != 0;
            if (c3.nan) fail = "NaN (fused3)";
            else if (c3.worst > 1.0) fail = "fused3 outside the rounding bound";
        }
        printf("CNORM_CHECK case=\"%s\" groups_with_channels=%d err_over_bound plain=%.3f add=%.3f silu=%.3f fused3=%.3f "
               "max_abs_err plain=%.2e fused3=%.2e fused_nodes=%d,%d fused2_vs_separate_diff=%zu fused3_vs_separate_diff=%zu "
               "%s%s\n",
               c.name, g.G, cp.worst, ca.worst, cs.worst, c3.worst, cp.max_abs, c3.max_abs, fused2, fused3, f2_diff, f3_diff,
               fail ? "FAIL: " : "PASS", fail ? fail : "");
        if (!fail) ++passed;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // a source that is a permuted view is not contiguous: the op must be declined, not computed some other way
    ++total;
    {
        struct ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        struct ggml_context* ctx = ggml_init(ip);
        struct ggml_tensor* x = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 12, 10, 1), 2, 0, 1, 3); // [12,10,8,1]
        struct ggml_tensor* d = ggml_group_norm(ctx, x, 4, 1e-6f);
        const bool declined = !ggml_backend_supports_op(be, d);
        printf("CNORM_CHECK case=\"non-contiguous source is declined\" %s\n", declined ? "PASS" : "FAIL: supports_op says yes");
        passed += declined;
        ggml_free(ctx);
    }

    // One block of a 2D decoder in sd.cpp's form: GroupNorm32 (GROUP_NORM, MUL, ADD in place), SILU in place, a 3x3
    // convolution with f16 weights and its bias added in place. Reference in f32 with the stored (f16) weights; the
    // convolution sums 576 products, so the bound is relative to the size of the result.
    ++total;
    {
        const int W = 24, H = 20, C = 64, OC = 32, G = 32;
        const Case c = { "block", W, H, C, 1, G, 1e-6f, 0.2f, 1.0f };
        const Geometry g = geometry(c);
        struct ggml_init_params ip = { 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        struct ggml_context* ctx = ggml_init(ip);
        struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
        struct ggml_tensor* wt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        struct ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        struct ggml_tensor* kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, C, OC);
        struct ggml_tensor* cb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        struct ggml_tensor* h = ggml_group_norm(ctx, x, G, c.eps);
        h = ggml_mul_inplace(ctx, h, ggml_reshape_4d(ctx, wt, 1, 1, C, 1));
        h = ggml_add_inplace(ctx, h, ggml_reshape_4d(ctx, bt, 1, 1, C, 1));
        h = ggml_silu_inplace(ctx, h);
        struct ggml_tensor* conv = ggml_conv_2d_direct(ctx, kt, h, 1, 1, 1, 1, 1, 1);
        struct ggml_tensor* out = ggml_add_inplace(ctx, conv, ggml_reshape_4d(ctx, cb, 1, 1, OC, 1));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
        const char* fail = buf ? nullptr : "allocation";
        const size_t nk = (size_t)9 * C * OC, nd = g.plane * OC;
        std::vector<float> xv(g.n), wv((size_t)C), bv((size_t)C), kv(nk), cbv((size_t)OC), got(nd), ref(nd), rn(g.n), act(g.n);
        std::vector<double> units;
        double worst = 0, rms = 0;
        int64_t d_norm = 0, d_conv = 0, d_bias = 0, d_fused = 0;
        if (!fail) {
            for (float& v : xv) v = c.offset + c.spread * rnd();
            for (float& v : wv) v = 0.2f + 1.5f * rnd();
            for (float& v : bv) v = rnd();
            for (float& v : kv) v = rnd() * 0.1f;
            for (float& v : cbv) v = rnd();
            std::vector<ggml_fp16_t> kh(nk);
            ggml_fp32_to_fp16_row(kv.data(), kh.data(), (int64_t)nk);
            ggml_fp16_to_fp32_row(kh.data(), kv.data(), (int64_t)nk);
            ggml_backend_tensor_set(x, xv.data(), 0, g.n * sizeof(float));
            ggml_backend_tensor_set(wt, wv.data(), 0, wv.size() * sizeof(float));
            ggml_backend_tensor_set(bt, bv.data(), 0, bv.size() * sizeof(float));
            ggml_backend_tensor_set(kt, kh.data(), 0, nk * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(cb, cbv.data(), 0, cbv.size() * sizeof(float));
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            for (int i = 0; i < ggml_graph_n_nodes(gf) && !fail; ++i) {
                if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) fail = "supports_op says no";
            }
            const int64_t n0 = ggml_backend_ameva_get_group_norm_count(), k0 = ggml_backend_ameva_get_conv_count();
            const int64_t b0 = ggml_backend_ameva_get_conv_bias_count(), f0 = ggml_backend_ameva_get_fused_pair_count();
            if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
            d_norm = ggml_backend_ameva_get_group_norm_count() - n0;
            d_conv = ggml_backend_ameva_get_conv_count() - k0;
            d_bias = ggml_backend_ameva_get_conv_bias_count() - b0;
            d_fused = ggml_backend_ameva_get_fused_pair_count() - f0;
            if (!fail && (d_norm != 1 || d_conv != 1 || d_bias != fuse_on || d_fused != 4 * fuse_on)) fail = "counters";
        }
        if (!fail) {
            ggml_backend_tensor_get(out, got.data(), 0, nd * sizeof(float));
            reference_norm(c, g, xv.data(), rn.data(), units);
            for (size_t i = 0; i < g.n; ++i) {
                const float a = rn[i] * wv[i / g.plane] + bv[i / g.plane];
                act[i] = a / (1.0f + expf(-a));
            }
            double ss = 0;
            for (int oc = 0; oc < OC; ++oc)
                for (int oh = 0; oh < H; ++oh)
                    for (int ow = 0; ow < W; ++ow) {
                        double acc = 0;
                        for (int ch = 0; ch < C; ++ch)
                            for (int t = 0; t < 9; ++t) {
                                const int iw = ow + t % 3 - 1, ih = oh + t / 3 - 1;
                                if (iw < 0 || iw >= W || ih < 0 || ih >= H) continue;
                                acc += (double)kv[t + 9 * ((size_t)ch + (size_t)C * oc)] * (double)act[iw + (size_t)W * (ih + (size_t)H * ch)];
                            }
                        const float r = (float)acc + cbv[oc];
                        ref[ow + (size_t)W * (oh + (size_t)H * oc)] = r;
                        ss += (double)r * r;
                    }
            rms = std::sqrt(ss / (double)nd);
            size_t nan = 0;
            for (size_t i = 0; i < nd; ++i) {
                nan += got[i] != got[i];
                worst = std::fmax(worst, std::fabs((double)got[i] - (double)ref[i]));
            }
            if (nan) fail = "NaN in the result";
            else if (worst > 1e-4 * rms) fail = "differs from the reference by more than 1e-4 of its rms";
        }
        printf("CNORM_CHECK case=\"decoder block: norm, weight, bias, silu, 3x3 convolution, bias\" group_norms=%lld "
               "convolutions=%lld with_fused_bias=%lld fused_nodes=%lld max_abs_err_over_rms=%.2e %s%s\n",
               (long long)d_norm, (long long)d_conv, (long long)d_bias, (long long)d_fused, rms > 0 ? worst / rms : worst,
               fail ? "FAIL: " : "PASS", fail ? fail : "");
        if (!fail) ++passed;
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Timing only: these outputs are not compared, the cases above are the check. One run that is not counted, then
    // four. fused3 normalises in place, so every run after the first works on the result of the run before.
    if (const char* spec = getenv("AMEVA_CNORM_CHECK_TIME")) {
        for (const char* p = spec; *p;) {
            int W = 0, H = 0, C = 0, used = 0;
            if (sscanf(p, "%d,%d,%d%n", &W, &H, &C, &used) != 3 || W < 1 || H < 1 || C < 1) {
                fprintf(stderr, "AMEVA_CNORM_CHECK_TIME: bad shape at \"%s\"\n", p);
                return 2;
            }
            p += used;
            while (*p == ' ') ++p;
            struct ggml_init_params ip = { 32 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(), nullptr, true };
            struct ggml_context* ctx = ggml_init(ip);
            struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
            struct ggml_tensor* x2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
            struct ggml_tensor* wt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            struct ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            struct ggml_tensor* plain = ggml_group_norm(ctx, x, 32, 1e-6f);
            struct ggml_tensor* sep = ggml_silu(
                ctx, ggml_add(ctx, ggml_mul(ctx, ggml_group_norm(ctx, x, 32, 1e-6f), ggml_reshape_4d(ctx, wt, 1, 1, C, 1)),
                              ggml_reshape_4d(ctx, bt, 1, 1, C, 1)));
            struct ggml_tensor* f3 = ggml_silu_inplace(
                ctx, ggml_add_inplace(ctx, ggml_mul_inplace(ctx, ggml_group_norm_inplace(ctx, x2, 32, 1e-6f),
                                                            ggml_reshape_4d(ctx, wt, 1, 1, C, 1)),
                                      ggml_reshape_4d(ctx, bt, 1, 1, C, 1)));
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
            if (!buf || !ggml_backend_supports_op(be, plain)) {
                printf("CNORM_TIME %dx%dx%d: not allocated or not supported\n", W, H, C);
                return 1;
            }
            std::vector<float> xv((size_t)W * H * C), wv((size_t)C, 1.0f), bv((size_t)C, 0.0f);
            for (float& v : xv) v = rnd();
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
            ggml_backend_tensor_set(x2, xv.data(), 0, xv.size() * sizeof(float));
            ggml_backend_tensor_set(wt, wv.data(), 0, wv.size() * sizeof(float));
            ggml_backend_tensor_set(bt, bv.data(), 0, bv.size() * sizeof(float));
            double ms[3] = { 0, 0, 0 };
            struct ggml_tensor* outs[3] = { plain, sep, f3 };
            bool ok = true;
            for (int k = 0; k < 3 && ok; ++k) {
                struct ggml_cgraph* gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, outs[k]);
                for (int it = -1; it < 4 && ok; ++it) {
                    struct timespec t0, t1;
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    if (it >= 0) ms[k] += ms_between(t0, t1) / 4;
                }
            }
            if (ok) printf("CNORM_TIME %dx%dx%d 32 groups, ms per graph_compute: norm alone %.2f, norm + weight + bias + silu as "
                           "4 nodes %.2f, as one fused node %.2f (AMEVA_FUSE=%d)\n", W, H, C, ms[0], ms[1], ms[2], fuse_on);
            else printf("CNORM_TIME %dx%dx%d: graph_compute failed\n", W, H, C);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            if (!ok) return 1;
        }
    }
    printf("CNORM_CHECK %d/%d passed, group norms on the GPU=%lld, AMEVA_FUSE=%d\n", passed, total,
           (long long)ggml_backend_ameva_get_group_norm_count(), fuse_on);
    ggml_backend_free(be);
    return passed == total ? 0 : 1;
}
