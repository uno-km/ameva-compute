// SPDX-License-Identifier: Apache-2.0
// AMEVA check of the two ops the UNet of stable-diffusion.cpp needs beyond those of the VAE: NORM (ggml_norm, the
// layer norm of the transformer blocks) and TIMESTEP_EMBEDDING, through the real ggml backend path (supports_op ->
// graph_compute) on the AMEVA backend.
//
//   NORM                 y = (x - mean) / sqrt(var + eps) per row of ne0. Reference in double. The kernel and ggml's
//                        CPU code both sum in f32, so the bound is a rounding bound, not equality:
//                        |y - ref| <= 1e-5 * (1 + |mean| / std) (the error of an f32 mean, relative to the spread).
//                        Cases: row lengths below, at and above one workgroup, 4-D, a mean of 30 against a spread of
//                        1, in place.
//   TIMESTEP_EMBEDDING   cos and sin of t * freq[j]. Reference: the expression of
//                        ggml_compute_forward_timestep_embedding_f32 with this device's libm. The kernel takes the
//                        frequencies from the host (same bits) and computes sine and cosine itself, so the bound is
//                        absolute: |y - ref| <= 6e-7. Timesteps 0 .. 1000, a negative one, fractions; even and odd
//                        dim. A ggml whose result is not [dim, n] is declined, which the test accepts for an odd dim.
//   REPEAT               (the LoRA merge of stable-diffusion.cpp) the source repeated over the result: equal to
//                        the reference in every bit.
// The largest error of each case is printed. Exit status 0 only when every case passes.
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

static uint32_t g_seed = 0x1B873593u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static int g_passed = 0, g_total = 0;

struct NormCase {
    const char* name;
    int64_t ne[4];
    float eps, offset, spread;
    bool inplace;
};

static void check_norm(ggml_backend_t be, const NormCase& c) {
    ++g_total;
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne[0], c.ne[1], c.ne[2], c.ne[3]);
    struct ggml_tensor* y = c.inplace ? ggml_norm_inplace(ctx, x, c.eps) : ggml_norm(ctx, x, c.eps);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t n = (size_t)ggml_nelements(x), n0 = (size_t)c.ne[0];
    const char* fail = buf ? nullptr : "allocation";
    double worst = 0; // largest error as a fraction of the bound
    size_t nan = 0;
    if (!fail) {
        std::vector<float> xv(n), got(n);
        for (float& v : xv) v = c.offset + c.spread * rnd();
        ggml_backend_tensor_set(x, xv.data(), 0, n * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, n * sizeof(float));
            for (size_t r = 0; r < n / n0; ++r) {
                const float* row = xv.data() + r * n0;
                double mean = 0, var = 0;
                for (size_t k = 0; k < n0; ++k) mean += row[k];
                mean /= (double)n0;
                for (size_t k = 0; k < n0; ++k) var += (row[k] - mean) * (row[k] - mean);
                var /= (double)n0;
                const double scale = 1.0 / std::sqrt(var + (double)c.eps);
                const double bound = 1e-5 * (1.0 + std::fabs(mean) * scale);
                for (size_t k = 0; k < n0; ++k) {
                    const float g = got[r * n0 + k];
                    nan += g != g;
                    worst = std::fmax(worst, std::fabs((double)g - (row[k] - mean) * scale) / bound);
                }
            }
            if (nan) fail = "NaN in the result";
            else if (worst > 1.0) fail = "above the rounding bound";
        }
    }
    printf("UNET_OPS_CHECK op=NORM case=\"%s\" shape=[%lld,%lld,%lld,%lld] err_over_bound=%.3f %s%s\n", c.name,
           (long long)c.ne[0], (long long)c.ne[1], (long long)c.ne[2], (long long)c.ne[3], worst, fail ? "FAIL: " : "PASS",
           fail ? fail : "");
    if (!fail) ++g_passed;
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

struct TsCase {
    const char* name;
    int dim, max_period;
    std::vector<float> t;
};

static void check_timestep(ggml_backend_t be, const TsCase& c) {
    ++g_total;
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    const int64_t nt = (int64_t)c.t.size();
    struct ggml_tensor* ts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nt);
    struct ggml_tensor* y = ggml_timestep_embedding(ctx, ts, c.dim, c.max_period);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const char* fail = buf ? nullptr : "allocation";
    double worst = 0;
    bool declined_shape = false;
    if (!fail) {
        ggml_backend_tensor_set(ts, c.t.data(), 0, c.t.size() * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) {
            // a ggml that pads an odd dim gives a result the kernel does not write; that has to be declined
            if (y->ne[0] != c.dim) declined_shape = true;
            else fail = "supports_op says no";
        }
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (!fail && !declined_shape && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail && !declined_shape) {
            std::vector<float> got((size_t)ggml_nelements(y));
            ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
            const int half = c.dim / 2;
            for (int64_t i = 0; i < nt; ++i) {
                const float* row = got.data() + (size_t)i * (size_t)c.dim;
                for (int64_t j = 0; j < half; ++j) {
                    const float freq = (float)expf(-logf(c.max_period) * j / half);
                    const float arg = c.t[(size_t)i] * freq;
                    worst = std::fmax(worst, std::fabs((double)row[j] - (double)cosf(arg)));
                    worst = std::fmax(worst, std::fabs((double)row[j + half] - (double)sinf(arg)));
                }
                if (c.dim % 2 != 0 && row[2 * half] != 0.0f) fail = "the last element of an odd dim is not zero";
            }
            if (!fail && !(worst <= 6e-7)) fail = "above 6e-7";
        }
    }
    printf("UNET_OPS_CHECK op=TIMESTEP_EMBEDDING case=\"%s\" dim=%d max_period=%d timesteps=%lld max_abs_diff=%.3e %s%s\n",
           c.name, c.dim, c.max_period, (long long)nt, worst,
           fail ? "FAIL: " : declined_shape ? "PASS (declined: this ggml pads the result)" : "PASS", fail ? fail : "");
    if (!fail) ++g_passed;
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void check_repeat(ggml_backend_t be, const char* name, const int64_t a[4], const int64_t d[4]) {
    ++g_total;
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a[0], a[1], a[2], a[3]);
    struct ggml_tensor* shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d[0], d[1], d[2], d[3]);
    struct ggml_tensor* y = ggml_repeat(ctx, x, shape);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t nx = (size_t)ggml_nelements(x), ny = (size_t)ggml_nelements(y);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    if (!fail) {
        std::vector<float> xv(nx), got(ny);
        for (float& v : xv) v = rnd();
        ggml_backend_tensor_set(x, xv.data(), 0, nx * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(float));
            size_t i = 0;
            for (int64_t i3 = 0; i3 < d[3]; ++i3)
                for (int64_t i2 = 0; i2 < d[2]; ++i2)
                    for (int64_t i1 = 0; i1 < d[1]; ++i1)
                        for (int64_t i0 = 0; i0 < d[0]; ++i0, ++i) {
                            const size_t s = (size_t)((i0 % a[0]) + a[0] * ((i1 % a[1]) + a[1] * ((i2 % a[2]) + a[2] * (i3 % a[3]))));
                            bad += memcmp(&got[i], &xv[s], sizeof(float)) != 0;
                        }
            if (bad) fail = "differs from the reference";
        }
    }
    printf("UNET_OPS_CHECK op=REPEAT case=\"%s\" elements=%zu mismatches=%zu %s%s\n", name, ny, bad, fail ? "FAIL: " : "PASS",
           fail ? fail : "");
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
    const NormCase norm[] = {
        { "320 x 77 (SD 1.x attention block)", { 320, 77, 1, 1 }, 1e-5f, 0.1f, 1.0f, false },
        { "1280 x 64 x 2 (rows above one workgroup, batch)", { 1280, 64, 2, 1 }, 1e-5f, -0.3f, 2.0f, false },
        { "768 x 77 in place (CLIP)", { 768, 77, 1, 1 }, 1e-5f, 0.0f, 1.0f, true },
        { "5 x 3 x 2 x 2 (rows below one workgroup, 4-D)", { 5, 3, 2, 2 }, 1e-6f, 0.5f, 1.0f, false },
        { "128 x 9 (one element per invocation)", { 128, 9, 1, 1 }, 1e-5f, 0.0f, 1.0f, false },
        { "640 x 256, mean 30 against spread 1", { 640, 256, 1, 1 }, 1e-5f, 30.0f, 1.0f, false },
        { "1 x 7 (one element per row: zero over sqrt(eps))", { 1, 7, 1, 1 }, 1e-5f, 0.5f, 1.0f, false },
    };
    for (const NormCase& c : norm) check_norm(be, c);
    const std::vector<float> steps = { 0.0f, 1.0f, 20.5f, 261.0f, 500.0f, 999.0f, 1000.0f, -3.25f, 0.001f, 7999.0f };
    const TsCase ts[] = {
        { "SD 1.x time embedding", 320, 10000, steps },
        { "256 (SDXL label embedding, MMDiT)", 256, 10000, steps },
        { "odd dim", 255, 10000, steps },
        { "one timestep, dim 2", 2, 10000, { 981.0f } },
        { "max_period 1000", 128, 1000, steps },
    };
    for (const TsCase& c : ts) check_timestep(be, c);
    {
        const int64_t a0[4] = { 1, 1, 1, 1 }, d0[4] = { 320, 4, 1, 1 };  // a scalar over a matrix (LoRA scale)
        const int64_t a1[4] = { 5, 1, 3, 1 }, d1[4] = { 10, 7, 3, 2 };   // every axis: 2, 7, 1 and 2 times
        const int64_t a2[4] = { 64, 9, 1, 1 }, d2[4] = { 64, 9, 1, 1 };  // the same shape
        check_repeat(be, "scalar over 320 x 4", a0, d0);
        check_repeat(be, "5x1x3x1 over 10x7x3x2", a1, d1);
        check_repeat(be, "same shape", a2, d2);
    }
    printf("UNET_OPS_CHECK %d/%d passed\n", g_passed, g_total);
    ggml_backend_free(be);
    return g_passed == g_total ? 0 : 1;
}
