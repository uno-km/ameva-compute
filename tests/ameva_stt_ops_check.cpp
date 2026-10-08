// SPDX-License-Identifier: Apache-2.0
// AMEVA check of the ops the encoder stem of whisper.cpp needs beyond those of llama.cpp and stable-diffusion.cpp:
// IM2COL and the MUL_MAT of two f16 matrices that follows it inside ggml_conv_1d / ggml_conv_2d, through the real
// ggml backend path (supports_op -> graph_compute) on the AMEVA backend.
//
//   IM2COL    columns of f32 data as f16 or f32, 1-D and 2-D, with stride, padding and dilation. Reference: the
//             index expression of ggml_compute_forward_im2col_f16 with ggml's own f32 -> f16 conversion. The result
//             must equal it in every bit.
//   CONV_1D   ggml_conv_1d(kernel f16, data f32): IM2COL, MUL_MAT f16 x f16 -> f32, RESHAPE. Every node must be
//             accepted. Reference in double from the f16 columns and the f16 kernel; the kernel sums in f32 in another
//             order than the CPU, so the bound is a rounding bound: |y - ref| <= 4e-6 * sum |column * kernel|.
//   CONV_2D   ggml_conv_2d(kernel f16, data f32): the same with 2-D columns and the final PERMUTE + CONT.
// The largest error over its bound is printed per case. Exit status 0 only when every case passes.
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

static uint32_t g_seed = 0x2545F491u;
static float rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f;
}

static int g_passed = 0, g_total = 0;

struct ConvCase {
    const char* name;
    int64_t iw, ih, ic, n; // data
    int64_t kw, kh, oc;    // kernel
    int s0, s1, p0, p1, d0, d1;
    bool two_d;
};

// column c = (in * OH + ioh) * OW + iow, value (iic * KH + ikh) * KW + ikw (ggml_compute_forward_im2col_f16)
static void ref_columns(const ConvCase& c, const std::vector<float>& x, int64_t ow, int64_t oh, std::vector<float>& col) {
    const int64_t ck = c.ic * c.kh * c.kw;
    col.assign((size_t)(c.n * oh * ow * ck), 0.0f);
    for (int64_t in = 0; in < c.n; ++in)
        for (int64_t ioh = 0; ioh < oh; ++ioh)
            for (int64_t iow = 0; iow < ow; ++iow)
                for (int64_t iic = 0; iic < c.ic; ++iic)
                    for (int64_t ikh = 0; ikh < c.kh; ++ikh)
                        for (int64_t ikw = 0; ikw < c.kw; ++ikw) {
                            const int64_t px = iow * c.s0 + ikw * c.d0 - c.p0, py = ioh * c.s1 + ikh * c.d1 - c.p1;
                            if (px < 0 || px >= c.iw || py < 0 || py >= c.ih) continue;
                            col[(size_t)(((in * oh + ioh) * ow + iow) * ck + (iic * c.kh + ikh) * c.kw + ikw)] =
                                x[(size_t)(px + c.iw * (py + c.ih * (iic + c.ic * in)))];
                        }
}

static struct ggml_tensor* new_data(struct ggml_context* ctx, const ConvCase& c) {
    return c.two_d ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.iw, c.ih, c.ic, c.n)
                   : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.iw, c.ic, c.n);
}
static struct ggml_tensor* new_kernel(struct ggml_context* ctx, const ConvCase& c, enum ggml_type type) {
    return c.two_d ? ggml_new_tensor_4d(ctx, type, c.kw, c.kh, c.ic, c.oc) : ggml_new_tensor_3d(ctx, type, c.kw, c.ic, c.oc);
}

static void check_im2col(ggml_backend_t be, const ConvCase& c, bool f16) {
    ++g_total;
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = new_data(ctx, c);
    struct ggml_tensor* w = new_kernel(ctx, c, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    struct ggml_tensor* y = ggml_im2col(ctx, w, x, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1, c.two_d, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t nx = (size_t)ggml_nelements(x), ny = (size_t)ggml_nelements(y);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    if (!fail) {
        std::vector<float> xv(nx), col;
        for (float& v : xv) v = rnd() * 3.0f;
        xv[nx / 2] = 3.0e-6f; // an f16 subnormal after the conversion
        xv[nx / 3] = 1.0e-9f; // rounds to zero
        ggml_backend_tensor_set(x, xv.data(), 0, nx * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail) {
            ref_columns(c, xv, y->ne[1], c.two_d ? y->ne[2] : 1, col);
            if (col.size() != ny) {
                fail = "result size differs from the reference";
            } else if (f16) {
                std::vector<ggml_fp16_t> got(ny);
                ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < ny; ++i) {
                    const ggml_fp16_t r = ggml_fp32_to_fp16(col[i]);
                    bad += memcmp(&got[i], &r, sizeof(r)) != 0;
                }
            } else {
                std::vector<float> got(ny);
                ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(float));
                for (size_t i = 0; i < ny; ++i) bad += memcmp(&got[i], &col[i], sizeof(float)) != 0;
            }
            if (!fail && bad) fail = "differs from the reference";
        }
    }
    printf("STT_OPS_CHECK op=IM2COL type=%s case=\"%s\" elements=%zu mismatches=%zu %s%s\n", f16 ? "f16" : "f32", c.name, ny,
           bad, fail ? "FAIL: " : "PASS", fail ? fail : "");
    if (!fail) ++g_passed;
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void check_conv(ggml_backend_t be, const ConvCase& c) {
    ++g_total;
    struct ggml_init_params ip = { 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = new_data(ctx, c);
    struct ggml_tensor* w = new_kernel(ctx, c, GGML_TYPE_F16);
    struct ggml_tensor* y = c.two_d ? ggml_conv_2d(ctx, w, x, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1)
                                    : ggml_conv_1d(ctx, w, x, c.s0, c.p0, c.d0);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t nx = (size_t)ggml_nelements(x), nw = (size_t)ggml_nelements(w), ny = (size_t)ggml_nelements(y);
    const int64_t ck = c.ic * c.kh * c.kw;
    const int64_t ow = y->ne[0], oh = c.two_d ? y->ne[1] : 1;
    const char* fail = buf ? nullptr : "allocation";
    double worst = 0.0;
    if (!fail) {
        std::vector<float> xv(nx), col, got(ny);
        std::vector<ggml_fp16_t> wv(nw);
        for (float& v : xv) v = rnd() * 2.0f;
        for (ggml_fp16_t& v : wv) v = ggml_fp32_to_fp16(rnd() * 0.25f);
        ggml_backend_tensor_set(x, xv.data(), 0, nx * sizeof(float));
        ggml_backend_tensor_set(w, wv.data(), 0, nw * sizeof(ggml_fp16_t));
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        for (int i = 0; i < ggml_graph_n_nodes(gf) && !fail; ++i) {
            if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) fail = "supports_op says no to a node of the convolution";
        }
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(float));
            ref_columns(c, xv, ow, oh, col);
            // kernel element (ikw, ikh, iic, ioc) lies at ioc * ck + (iic * KH + ikh) * KW + ikw: the order of a column
            for (int64_t in = 0; in < c.n; ++in)
                for (int64_t ioc = 0; ioc < c.oc; ++ioc)
                    for (int64_t p = 0; p < oh * ow; ++p) {
                        double ref = 0.0, mag = 0.0;
                        const size_t cb = (size_t)((in * oh * ow + p) * ck);
                        for (int64_t k = 0; k < ck; ++k) {
                            const double t = (double)ggml_fp16_to_fp32(ggml_fp32_to_fp16(col[cb + (size_t)k])) *
                                             (double)ggml_fp16_to_fp32(wv[(size_t)(ioc * ck + k)]);
                            ref += t;
                            mag += fabs(t);
                        }
                        // 1-D: [OW, OC, N]; 2-D: [OW, OH, OC, N]
                        const double v = got[(size_t)(p + oh * ow * (ioc + c.oc * in))];
                        const double r = fabs(v - ref) / (4e-6 * mag + 1e-30);
                        if (!(r <= worst)) worst = r; // also catches NaN
                    }
            if (!(worst <= 1.0)) fail = "above the rounding bound";
        }
    }
    printf("STT_OPS_CHECK op=%s case=\"%s\" elements=%zu err_over_bound=%.3g %s%s\n", c.two_d ? "CONV_2D" : "CONV_1D", c.name,
           ny, worst, fail ? "FAIL: " : "PASS", fail ? fail : "");
    if (!fail) ++g_passed;
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    const ConvCase cases[] = {
        { "whisper conv1: 3 x 128 -> 64 over 300 frames, pad 1", 300, 1, 128, 1, 3, 1, 64, 1, 1, 1, 0, 1, 1, false },
        { "whisper conv2: 3 x 96 -> 40 over 301 frames, stride 2, pad 1", 301, 1, 96, 1, 3, 1, 40, 2, 1, 1, 0, 1, 1, false },
        { "3 x 5 -> 7 over 17 frames, dilation 2 (odd column length)", 17, 1, 5, 1, 3, 1, 7, 1, 1, 0, 0, 2, 1, false },
        { "1 x 33 -> 9 over 50 frames (kernel of one tap)", 50, 1, 33, 1, 1, 1, 9, 1, 1, 0, 0, 1, 1, false },
        { "5 x 2 -> 3 over 9 frames, pad 4 (columns mostly padding)", 9, 1, 2, 1, 5, 1, 3, 1, 1, 4, 0, 1, 1, false },
        { "2-D: 3 x 2 kernel, 3 -> 6, 9 x 7, strides 2 and 1, pad 1 and 0, batch 2", 9, 7, 3, 2, 3, 2, 6, 2, 1, 1, 0, 1, 1, true },
        { "2-D: 3 x 3 kernel, 4 -> 8, 16 x 16, pad 1 (CLIP-like patch stem)", 16, 16, 4, 1, 3, 3, 8, 1, 1, 1, 1, 1, 1, true },
        { "2-D: 2 x 2 kernel, 5 -> 3, 11 x 6, dilation 2 and 3", 11, 6, 5, 1, 2, 2, 3, 1, 1, 0, 0, 2, 3, true },
    };
    for (const ConvCase& c : cases) {
        check_im2col(be, c, true);
        check_im2col(be, c, false);
    }
    for (const ConvCase& c : cases) check_conv(be, c);
    printf("STT_OPS_CHECK %d/%d passed\n", g_passed, g_total);
    ggml_backend_free(be);
    return g_passed == g_total ? 0 : 1;
}
