// SPDX-License-Identifier: Apache-2.0
// AMEVA K3 check: the ops a VAE graph needs besides convolution and group norm, through the real ggml backend path
// (supports_op -> graph_compute) on the AMEVA backend, each compared with a reference computed here as ggml's CPU
// code computes it.
//
//   UPSCALE   nearest (ggml_upscale, ggml_interpolate): equal to the reference in every bit. A fractional factor is
//             accepted only when the kernel's integer index equals ggml's float index for every position; the test
//             evaluates the same rule and requires supports_op to agree with it. Bilinear must be declined.
//   PAD       zeros and circular (ggml_pad_ext, ggml_pad_ext_circular): equal in every bit.
//   CONCAT    every dimension, f32 and f16: equal in every bit.
//   SOFT_MAX  plain, in place, with scale and an f32 or f16 mask (ggml_soft_max_ext): within a rounding bound of the
//             reference (sum in double there, an f32 tree here; exp within the 3 + 2 |x| ulp Vulkan allows). An
//             ALiBi bias must be declined.
//   MUL_MAT   f32 x f32 in the two shapes of sd.cpp's VAE attention (K = channels, K = latent positions), with K not
//             a multiple of 32 (a 30x30 latent), and with several matrices: within K eps sum |a b| of the reference.
// A source may be a transposed view (strides that are not those of a contiguous tensor). The AMEVA backend has no
// CPU path: a node it accepts is computed by its kernels or graph_compute fails. Exit status 0 only when every case
// passes.
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
#include <limits>
#include <utility>
#include <vector>

static uint32_t g_seed = 0x9E3779B9u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

// logical contents of a tensor, row-major in its own shape
struct Host {
    int64_t ne[4];
    std::vector<float> v;
    Host(int64_t n0, int64_t n1, int64_t n2, int64_t n3) : ne{ n0, n1, n2, n3 }, v((size_t)(n0 * n1 * n2 * n3)) {}
    size_t idx(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const {
        return (size_t)(i0 + ne[0] * (i1 + ne[1] * (i2 + ne[2] * i3)));
    }
    float at(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const { return v[idx(i0, i1, i2, i3)]; }
    void fill(float offset, float spread) {
        for (float& x : v) x = offset + spread * rnd();
    }
};

struct Graph {
    struct ggml_context* ctx;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<std::pair<struct ggml_tensor*, std::vector<uint8_t>>> fills;
    Graph() {
        struct ggml_init_params ip = { 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
    }
    ~Graph() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    // A source with the contents h: a contiguous tensor, or (view) dimensions 0 and 1 of a contiguous base exchanged.
    struct ggml_tensor* src(const Host& h, bool view, enum ggml_type type = GGML_TYPE_F32) {
        const size_t n = h.v.size();
        std::vector<float> data(n);
        struct ggml_tensor* t;
        struct ggml_tensor* base;
        if (view) {
            base = ggml_new_tensor_4d(ctx, type, h.ne[1], h.ne[0], h.ne[2], h.ne[3]);
            t = ggml_permute(ctx, base, 1, 0, 2, 3);
            for (int64_t i3 = 0; i3 < h.ne[3]; ++i3)
                for (int64_t i2 = 0; i2 < h.ne[2]; ++i2)
                    for (int64_t i1 = 0; i1 < h.ne[1]; ++i1)
                        for (int64_t i0 = 0; i0 < h.ne[0]; ++i0)
                            data[(size_t)(i1 + h.ne[1] * (i0 + h.ne[0] * (i2 + h.ne[2] * i3)))] = h.at(i0, i1, i2, i3);
        } else {
            base = t = ggml_new_tensor_4d(ctx, type, h.ne[0], h.ne[1], h.ne[2], h.ne[3]);
            data = h.v;
        }
        std::vector<uint8_t> bytes(n * ggml_type_size(type));
        if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(data.data(), (ggml_fp16_t*)bytes.data(), (int64_t)n);
        else memcpy(bytes.data(), data.data(), bytes.size());
        fills.emplace_back(base, std::move(bytes));
        return t;
    }
    // NULL, or what failed. The result is read into out (bytes of the result's type).
    const char* run(ggml_backend_t be, struct ggml_tensor* result, std::vector<uint8_t>& out) {
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
        if (!buf) return "allocation";
        for (auto& f : fills) ggml_backend_tensor_set(f.first, f.second.data(), 0, f.second.size());
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) return "graph_compute failed";
        out.resize(ggml_nbytes(result));
        ggml_backend_tensor_get(result, out.data(), 0, out.size());
        return nullptr;
    }
};

static int g_passed = 0, g_total = 0;
static void report(const char* op, const char* name, const char* fail, const char* detail) {
    ++g_total;
    if (!fail) ++g_passed;
    printf("K3_CHECK op=%s case=\"%s\" %s %s%s\n", op, name, detail, fail ? "FAIL: " : "PASS", fail ? fail : "");
}

// number of elements whose bits differ
static size_t bit_diff(const std::vector<uint8_t>& got, const void* ref, size_t esize) {
    size_t n = 0;
    for (size_t i = 0; i + esize <= got.size(); i += esize) n += memcmp(&got[i], (const uint8_t*)ref + i, esize) != 0;
    return n;
}

// ---------------------------------------------------------------------------------------------------- UPSCALE
struct UpCase {
    const char* name;
    int64_t a[4], d[4];
    bool view, bilinear;
};

static void check_upscale(ggml_backend_t be, const UpCase& c) {
    Host h(c.a[0], c.a[1], c.a[2], c.a[3]);
    h.fill(0.0f, 4.0f);
    Graph g;
    struct ggml_tensor* x = g.src(h, c.view);
    struct ggml_tensor* y = ggml_interpolate(g.ctx, x, c.d[0], c.d[1], c.d[2], c.d[3],
                                             c.bilinear ? GGML_SCALE_MODE_BILINEAR : GGML_SCALE_MODE_NEAREST);
    // the rule of the engine: the integer index equals ggml's float index everywhere
    bool same_index = true;
    float sf[4];
    for (int i = 0; i < 4; ++i) {
        sf[i] = (float)c.d[i] / c.a[i];
        for (int64_t k = 0; k < c.d[i]; ++k) same_index = same_index && (int64_t)(k / sf[i]) == k * c.a[i] / c.d[i];
    }
    const bool expect = same_index && !c.bilinear;
    const bool says = ggml_backend_supports_op(be, y);
    char detail[160];
    snprintf(detail, sizeof(detail), "src=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] supports_op=%d expected=%d",
             (long long)c.a[0], (long long)c.a[1], (long long)c.a[2], (long long)c.a[3], (long long)c.d[0],
             (long long)c.d[1], (long long)c.d[2], (long long)c.d[3], (int)says, (int)expect);
    if (says != expect) return report("UPSCALE", c.name, "supports_op differs from the rule", detail);
    if (!says) return report("UPSCALE", c.name, nullptr, detail);
    Host r(c.d[0], c.d[1], c.d[2], c.d[3]);
    for (int64_t i3 = 0; i3 < c.d[3]; ++i3)
        for (int64_t i2 = 0; i2 < c.d[2]; ++i2)
            for (int64_t i1 = 0; i1 < c.d[1]; ++i1)
                for (int64_t i0 = 0; i0 < c.d[0]; ++i0)
                    r.v[r.idx(i0, i1, i2, i3)] = h.at((int64_t)(i0 / sf[0]), (int64_t)(i1 / sf[1]), (int64_t)(i2 / sf[2]),
                                                      (int64_t)(i3 / sf[3]));
    std::vector<uint8_t> got;
    const char* fail = g.run(be, y, got);
    size_t diff = 0;
    if (!fail && (diff = bit_diff(got, r.v.data(), 4)) != 0) fail = "differs from the reference";
    char d2[224];
    snprintf(d2, sizeof(d2), "%s elements=%zu differing=%zu", detail, r.v.size(), diff);
    report("UPSCALE", c.name, fail, d2);
}

// -------------------------------------------------------------------------------------------------------- PAD
struct PadCase {
    const char* name;
    int64_t a[4];
    int lp[4], rp[4];
    bool circular, view;
};

static void check_pad(ggml_backend_t be, const PadCase& c) {
    Host h(c.a[0], c.a[1], c.a[2], c.a[3]);
    h.fill(0.5f, 2.0f);
    Graph g;
    struct ggml_tensor* x = g.src(h, c.view);
    struct ggml_tensor* y =
        c.circular ? ggml_pad_ext_circular(g.ctx, x, c.lp[0], c.rp[0], c.lp[1], c.rp[1], c.lp[2], c.rp[2], c.lp[3], c.rp[3])
                   : ggml_pad_ext(g.ctx, x, c.lp[0], c.rp[0], c.lp[1], c.rp[1], c.lp[2], c.rp[2], c.lp[3], c.rp[3]);
    Host r(y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
    for (int64_t i3 = 0; i3 < r.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < r.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < r.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < r.ne[0]; ++i0) {
                    const int64_t i[4] = { i0, i1, i2, i3 };
                    int64_t s[4];
                    bool inside = true;
                    for (int k = 0; k < 4; ++k) {
                        s[k] = i[k] - c.lp[k];
                        if (c.circular) s[k] = (s[k] + c.a[k]) % c.a[k]; // ggml_wrap_around
                        else inside = inside && s[k] >= 0 && s[k] < c.a[k];
                    }
                    r.v[r.idx(i0, i1, i2, i3)] = inside ? h.at(s[0], s[1], s[2], s[3]) : 0.0f;
                }
    char detail[200];
    snprintf(detail, sizeof(detail), "src=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] circular=%d", (long long)c.a[0],
             (long long)c.a[1], (long long)c.a[2], (long long)c.a[3], (long long)r.ne[0], (long long)r.ne[1],
             (long long)r.ne[2], (long long)r.ne[3], (int)c.circular);
    if (!ggml_backend_supports_op(be, y)) return report("PAD", c.name, "supports_op says no", detail);
    std::vector<uint8_t> got;
    const char* fail = g.run(be, y, got);
    size_t diff = 0;
    if (!fail && (diff = bit_diff(got, r.v.data(), 4)) != 0) fail = "differs from the reference";
    char d2[256];
    snprintf(d2, sizeof(d2), "%s elements=%zu differing=%zu", detail, r.v.size(), diff);
    report("PAD", c.name, fail, d2);
}

// ----------------------------------------------------------------------------------------------------- CONCAT
struct CatCase {
    const char* name;
    int64_t a[4], b[4];
    int dim;
    bool f16, view;
};

static void check_concat(ggml_backend_t be, const CatCase& c) {
    Host ha(c.a[0], c.a[1], c.a[2], c.a[3]), hb(c.b[0], c.b[1], c.b[2], c.b[3]);
    ha.fill(0.0f, 3.0f);
    hb.fill(1.0f, 3.0f);
    const enum ggml_type type = c.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    Graph g;
    struct ggml_tensor* xa = g.src(ha, c.view, type);
    struct ggml_tensor* xb = g.src(hb, false, type);
    struct ggml_tensor* y = ggml_concat(g.ctx, xa, xb, c.dim);
    Host r(y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
    for (int64_t i3 = 0; i3 < r.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < r.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < r.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < r.ne[0]; ++i0) {
                    int64_t i[4] = { i0, i1, i2, i3 };
                    const bool second = i[c.dim] >= c.a[c.dim];
                    if (second) i[c.dim] -= c.a[c.dim];
                    r.v[r.idx(i0, i1, i2, i3)] = second ? hb.at(i[0], i[1], i[2], i[3]) : ha.at(i[0], i[1], i[2], i[3]);
                }
    // the reference in the result's type (f16: the same conversion the sources went through)
    std::vector<uint8_t> ref(r.v.size() * ggml_type_size(type));
    if (c.f16) ggml_fp32_to_fp16_row(r.v.data(), (ggml_fp16_t*)ref.data(), (int64_t)r.v.size());
    else memcpy(ref.data(), r.v.data(), ref.size());
    char detail[200];
    snprintf(detail, sizeof(detail), "a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] dim=%d type=%s", (long long)c.a[0],
             (long long)c.a[1], (long long)c.a[2], (long long)c.a[3], (long long)c.b[0], (long long)c.b[1],
             (long long)c.b[2], (long long)c.b[3], c.dim, ggml_type_name(type));
    if (!ggml_backend_supports_op(be, y)) return report("CONCAT", c.name, "supports_op says no", detail);
    std::vector<uint8_t> got;
    const char* fail = g.run(be, y, got);
    size_t diff = 0;
    if (!fail && (diff = bit_diff(got, ref.data(), ggml_type_size(type))) != 0) fail = "differs from the reference";
    char d2[256];
    snprintf(d2, sizeof(d2), "%s elements=%zu differing=%zu", detail, r.v.size(), diff);
    report("CONCAT", c.name, fail, d2);
}

// --------------------------------------------------------------------------------------------------- SOFT_MAX
struct SmCase {
    const char* name;
    int64_t ne[4];
    bool inplace;
    float scale, spread;
    int mask;      // 0 none, 1 f32 random, 2 f16 with -inf above the diagonal (a causal mask)
    int64_t m2;    // ne2 of the mask (1: repeated over dimension 2)
    float max_bias; // != 0: must be declined
};

static void check_soft_max(ggml_backend_t be, const SmCase& c) {
    Host h(c.ne[0], c.ne[1], c.ne[2], c.ne[3]);
    h.fill(0.0f, c.spread);
    Host hm(c.ne[0], c.ne[1], c.mask ? c.m2 : 1, 1);
    if (c.mask == 1) hm.fill(0.0f, 2.0f);
    if (c.mask == 2) {
        for (int64_t i2 = 0; i2 < hm.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < hm.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < hm.ne[0]; ++i0)
                    hm.v[hm.idx(i0, i1, i2, 0)] = i0 > i1 ? -std::numeric_limits<float>::infinity() : 0.0f;
    }
    Graph g;
    struct ggml_tensor* x = g.src(h, false);
    struct ggml_tensor* m = c.mask ? g.src(hm, false, c.mask == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32) : nullptr;
    struct ggml_tensor* y = (c.mask || c.scale != 1.0f || c.max_bias != 0.0f)
                                ? ggml_soft_max_ext(g.ctx, x, m, c.scale, c.max_bias)
                                : c.inplace ? ggml_soft_max_inplace(g.ctx, x) : ggml_soft_max(g.ctx, x);
    char detail[200];
    snprintf(detail, sizeof(detail), "ne=[%lld,%lld,%lld,%lld] inplace=%d scale=%g mask=%s", (long long)c.ne[0],
             (long long)c.ne[1], (long long)c.ne[2], (long long)c.ne[3], (int)c.inplace, (double)c.scale,
             c.mask == 0 ? "none" : c.mask == 1 ? "f32" : "f16");
    const bool says = ggml_backend_supports_op(be, y);
    if (c.max_bias != 0.0f) return report("SOFT_MAX", c.name, says ? "an ALiBi bias was accepted" : nullptr, detail);
    if (!says) return report("SOFT_MAX", c.name, "supports_op says no", detail);
    std::vector<uint8_t> got;
    const char* fail = g.run(be, y, got);
    double worst = 0, max_abs = 0;
    size_t nan = 0;
    if (!fail) {
        const float* gv = (const float*)got.data();
        const int64_t n0 = c.ne[0];
        std::vector<float> xr((size_t)n0);
        for (int64_t i3 = 0; i3 < c.ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < c.ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < c.ne[1]; ++i1) {
                    // ggml_compute_forward_soft_max_f32: x = a * scale + mask, y = exp(x - max), y /= sum (double)
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        float v = h.at(i0, i1, i2, i3) * c.scale;
                        if (c.mask) v += hm.at(i0, i1, i2 % hm.ne[2], 0);
                        xr[(size_t)i0] = v;
                        mx = v > mx ? v : mx;
                    }
                    double sum = 0, tsum = 0;
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        const float e = expf(xr[(size_t)i0] - mx);
                        sum += (double)e;
                        if (e > 0.0f) tsum += (double)e * (3.0 + 2.0 * std::fabs((double)xr[(size_t)i0] - (double)mx));
                    }
                    tsum /= sum; // the mean ulp allowance of the terms, weighted as they enter the sum
                    const float inv = (float)(1.0 / sum);
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        const float ref = expf(xr[(size_t)i0] - mx) * inv;
                        const float gg = gv[h.idx(i0, i1, i2, i3)];
                        const double t = std::isinf(xr[(size_t)i0]) ? 0.0 : std::fabs((double)xr[(size_t)i0] - (double)mx);
                        // exp of this element and of the sum's terms, the rounding of x (one fused multiply-add
                        // moves x - max by an ulp of each), the tree sum and the division
                        const double tol = (double)ref * (double)FLT_EPSILON *
                                               (64.0 + 4.0 * t + 2.0 * tsum + 4.0 * (std::fabs((double)mx) + t) +
                                                (double)n0 / 64.0) + 1e-37;
                        const double d = std::fabs((double)gg - (double)ref);
                        nan += gg != gg;
                        max_abs = std::fmax(max_abs, d);
                        worst = std::fmax(worst, d / tol);
                    }
                }
        if (nan) fail = "NaN";
        else if (worst > 1.0) fail = "outside the rounding bound";
    }
    char d2[256];
    snprintf(d2, sizeof(d2), "%s err_over_bound=%.3f max_abs_err=%.2e", detail, worst, max_abs);
    report("SOFT_MAX", c.name, fail, d2);
}

// ---------------------------------------------------------------------------------------------------- MUL_MAT
struct MmCase {
    const char* name;
    int64_t K, N, M, a2, a3, b2, b3;
};

static void check_mul_mat(ggml_backend_t be, const MmCase& c) {
    Host ha(c.K, c.N, c.a2, c.a3), hb(c.K, c.M, c.b2, c.b3);
    ha.fill(0.0f, 1.0f);
    hb.fill(0.1f, 1.0f);
    Graph g;
    struct ggml_tensor* a = g.src(ha, false);
    struct ggml_tensor* b = g.src(hb, false);
    struct ggml_tensor* y = ggml_mul_mat(g.ctx, a, b);
    const bool qmm = c.K % 32 == 0 && c.b2 == 1 && c.b3 == 1; // the shapes of ameva_qmm; every other one: ameva_mm
    char detail[200];
    snprintf(detail, sizeof(detail), "K=%lld N=%lld M=%lld a=[..,%lld,%lld] b=[..,%lld,%lld] kernel=%s", (long long)c.K,
             (long long)c.N, (long long)c.M, (long long)c.a2, (long long)c.a3, (long long)c.b2, (long long)c.b3,
             qmm ? "ameva_qmm" : "ameva_mm");
    if (!ggml_backend_supports_op(be, y)) return report("MUL_MAT", c.name, "supports_op says no", detail);
    std::vector<uint8_t> got;
    const char* fail = g.run(be, y, got);
    double worst = 0, max_abs = 0;
    size_t nan = 0;
    if (!fail) {
        const float* gv = (const float*)got.data();
        for (int64_t i3 = 0; i3 < c.b3; ++i3)
            for (int64_t i2 = 0; i2 < c.b2; ++i2)
                for (int64_t m = 0; m < c.M; ++m)
                    for (int64_t n = 0; n < c.N; ++n) {
                        // ggml repeats each matrix of a over b2 / a2 consecutive matrices of b
                        const float* ar = &ha.v[ha.idx(0, n, i2 / (c.b2 / c.a2), i3 / (c.b3 / c.a3))];
                        const float* br = &hb.v[hb.idx(0, m, i2, i3)];
                        double s = 0, sa = 0;
                        for (int64_t k = 0; k < c.K; ++k) {
                            const double p = (double)ar[k] * (double)br[k];
                            s += p;
                            sa += std::fabs(p);
                        }
                        const float gg = gv[(size_t)(n + c.N * (m + c.M * (i2 + c.b2 * i3)))];
                        const double d = std::fabs((double)gg - s);
                        const double tol = ((double)c.K + 8.0) * (double)FLT_EPSILON * sa + 1e-30;
                        nan += gg != gg;
                        max_abs = std::fmax(max_abs, d);
                        worst = std::fmax(worst, d / tol);
                    }
        if (nan) fail = "NaN";
        else if (worst > 1.0) fail = "outside the rounding bound";
    }
    char d2[256];
    snprintf(d2, sizeof(d2), "%s err_over_bound=%.4f max_abs_err=%.2e", detail, worst, max_abs);
    report("MUL_MAT", c.name, fail, d2);
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    const UpCase up[] = {
        { "2x of 5x3x7x2", { 5, 3, 7, 2 }, { 10, 6, 7, 2 }, false, false },
        { "2x of 32x32x512 (decoder up.3)", { 32, 32, 512, 1 }, { 64, 64, 512, 1 }, false, false },
        { "2x of 128x128x256 (decoder up.1)", { 128, 128, 256, 1 }, { 256, 256, 256, 1 }, false, false },
        { "3x of a transposed view 4x6x3x1", { 4, 6, 3, 1 }, { 12, 18, 3, 1 }, true, false },
        { "3x2x2x1 on four axes", { 7, 5, 3, 2 }, { 21, 10, 6, 2 }, false, false },
        { "1.5x and 9/7 (fractional factors)", { 10, 7, 2, 1 }, { 15, 9, 2, 1 }, false, false },
        { "7/3 (fractional factor)", { 3, 3, 1, 1 }, { 7, 7, 1, 1 }, false, false },
        { "bilinear 2x (to be declined)", { 8, 8, 4, 1 }, { 16, 16, 4, 1 }, false, true },
    };
    for (const UpCase& c : up) check_upscale(be, c);
    const PadCase pad[] = {
        { "after the source on two axes", { 5, 4, 3, 2 }, { 0, 0, 0, 0 }, { 1, 2, 0, 0 }, false, false },
        { "before and after, four axes", { 4, 3, 2, 2 }, { 1, 0, 2, 1 }, { 0, 3, 1, 0 }, false, false },
        { "circular 1 on both sides of two axes", { 6, 5, 3, 1 }, { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, true, false },
        { "circular, padding as wide as the source", { 4, 3, 1, 1 }, { 4, 0, 0, 0 }, { 5, 2, 0, 0 }, true, false },
        { "64x64x128 by 1 on both sides", { 64, 64, 128, 1 }, { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, false, false },
        { "transposed view, before and after", { 7, 3, 2, 1 }, { 2, 1, 0, 0 }, { 1, 2, 1, 0 }, false, true },
        { "transposed view, circular", { 7, 3, 2, 1 }, { 2, 1, 1, 0 }, { 1, 2, 0, 0 }, true, true },
    };
    for (const PadCase& c : pad) check_pad(be, c);
    const CatCase cat[] = {
        { "dimension 0", { 3, 4, 2, 2 }, { 5, 4, 2, 2 }, 0, false, false },
        { "dimension 1", { 6, 2, 3, 1 }, { 6, 7, 3, 1 }, 1, false, false },
        { "dimension 2 (channels)", { 64, 64, 96, 1 }, { 64, 64, 160, 1 }, 2, false, false },
        { "dimension 3", { 5, 3, 2, 1 }, { 5, 3, 2, 4 }, 3, false, false },
        { "dimension 0, f16, odd lengths", { 3, 5, 2, 1 }, { 5, 5, 2, 1 }, 0, true, false },
        { "dimension 2, f16", { 9, 7, 3, 2 }, { 9, 7, 5, 2 }, 2, true, false },
        { "dimension 1, first source a transposed view", { 6, 4, 3, 1 }, { 6, 5, 3, 1 }, 1, false, true },
    };
    for (const CatCase& c : cat) check_concat(be, c);
    const SmCase sm[] = {
        { "1024x37", { 1024, 37, 1, 1 }, false, 1.0f, 4.0f, 0, 1, 0.0f },
        { "7x5x3x2", { 7, 5, 3, 2 }, false, 1.0f, 4.0f, 0, 1, 0.0f },
        { "rows of one element", { 1, 9, 1, 1 }, false, 1.0f, 4.0f, 0, 1, 0.0f },
        { "129x130 in place", { 129, 130, 1, 1 }, true, 1.0f, 8.0f, 0, 1, 0.0f },
        { "576x576 in place, spread 30 (VAE attention of a 24x24 latent)", { 576, 576, 1, 1 }, true, 1.0f, 30.0f, 0, 1, 0.0f },
        { "64x16x4 scale 0.5, f32 mask repeated over dimension 2", { 64, 16, 4, 1 }, false, 0.5f, 6.0f, 1, 1, 0.0f },
        { "64x64x2 scale 0.125, f16 causal mask", { 64, 64, 2, 1 }, false, 0.125f, 20.0f, 2, 2, 0.0f },
        { "ALiBi bias (to be declined)", { 32, 32, 4, 1 }, false, 1.0f, 4.0f, 1, 1, 8.0f },
    };
    for (const SmCase& c : sm) check_soft_max(be, c);
    const MmCase mm[] = {
        { "VAE attention k x q, 24x24 latent", 512, 576, 576, 1, 1, 1, 1 },
        { "VAE attention v x kq, 24x24 latent", 576, 512, 576, 1, 1, 1, 1 },
        { "VAE attention v x kq, 30x30 latent (K = 900)", 900, 512, 900, 1, 1, 1, 1 },
        { "K=7 N=5 M=3", 7, 5, 3, 1, 1, 1, 1 },
        { "K=33 N=17 M=9", 33, 17, 9, 1, 1, 1, 1 },
        { "3x2 matrices on both sides", 16, 9, 4, 3, 2, 3, 2 },
        { "one weight matrix for 3x2 activation matrices", 16, 9, 4, 1, 1, 3, 2 },
        { "K=64, 2 matrices", 64, 8, 5, 2, 1, 2, 1 },
        { "2x1 weight matrices, each for 2x3 of 4x3 activation matrices", 12, 7, 5, 2, 1, 4, 3 },
    };
    for (const MmCase& c : mm) check_mul_mat(be, c);
    printf("K3_CHECK %d/%d\n", g_passed, g_total);
    ggml_backend_free(be);
    return g_passed == g_total ? 0 : 1;
}
