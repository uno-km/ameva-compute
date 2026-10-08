// SPDX-License-Identifier: Apache-2.0
// AMEVA window check: UPSCALE (nearest) and the binary ops on tensors above one buffer binding, through the real ggml
// backend path (supports_op -> graph_compute) on the AMEVA backend.
//
// A device binds at most maxStorageBufferRange bytes of a buffer to a kernel (256 MiB on Mali-G68). The engine gives
// a kernel a tensor above that in windows of whole planes (index of dimension 2 at one index of dimension 3). This
// test sets AMEVA_BINDING_LIMIT=65536 before the backend starts, so that tensors of a few hundred KiB take those
// paths, and compares every result with a reference computed here:
//   UPSCALE   element (i0, i1, i2, i3) = source (i0 * a0 / d0, i1 * a1 / d1, i2, i3): equal in every bit.
//   ADD, SUB, MUL with ggml's broadcasting of src1 (a per-channel vector, a scalar, the shape of src0, one batch
//             element, a row per plane), in place and not: equal in every bit (one correctly rounded f32 operation).
//   DIV       within 4e-7 of the reference, relative (a driver's division need not be correctly rounded).
//   MUL_MAT   f32 x f32 with several matrices (the attention weights of a diffusion model: [keys, queries, heads]),
//             the result, or src1, above the limit and bound one matrix at a time. Reference in double; bound
//             1e-5 * sum |a| |b| per element (the kernel sums in f32 in its own order).
//   SOFT_MAX  in place on such a tensor, in windows of planes. Reference in double; bound 2e-6 absolute.
// Required as well: supports_op says yes, the backend's window counter advances by one per graph (so a result cannot
// come from the path of tensors inside one binding), and the two forms the windows cannot express are declined: an
// UPSCALE that also scales dimension 2, and a src1 repeated along dimension 2.
// CONV_2D and GROUP_NORM in windows: tests/ameva_conv_check.cpp and tests/ameva_cnorm_check.cpp under
// AMEVA_BINDING_LIMIT (scripts/termux/lab/conv_check.sh). Exit status 0 only when every case passes.
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

static uint32_t g_seed = 0x6D2B79F5u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static int g_passed = 0, g_total = 0;
static void report(const char* op, const char* name, const char* fail, long long windows, size_t bad, size_t n) {
    ++g_total;
    if (!fail) ++g_passed;
    printf("WINDOW_CHECK op=%s case=\"%s\" elements=%zu mismatches=%zu window_nodes=%lld %s%s\n", op, name, n, bad, windows,
           fail ? "FAIL: " : "PASS", fail ? fail : "");
}

struct UpCase {
    const char* name;
    int64_t a[4], d[4];
    bool declined; // the contract has to say no
};

static void check_upscale(ggml_backend_t be, const UpCase& c) {
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.a[0], c.a[1], c.a[2], c.a[3]);
    struct ggml_tensor* y = ggml_interpolate(ctx, x, c.d[0], c.d[1], c.d[2], c.d[3], GGML_SCALE_MODE_NEAREST);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t nx = (size_t)ggml_nelements(x), ny = (size_t)ggml_nelements(y);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    long long windows = 0;
    if (!fail && c.declined) {
        if (ggml_backend_supports_op(be, y)) fail = "supports_op says yes";
    } else if (!fail) {
        std::vector<float> xv(nx), got(ny);
        for (float& v : xv) v = rnd();
        ggml_backend_tensor_set(x, xv.data(), 0, nx * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        const int64_t w0 = ggml_backend_ameva_get_window_node_count();
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        windows = (long long)(ggml_backend_ameva_get_window_node_count() - w0);
        if (!fail && windows != 1) fail = "the window counter did not advance by one";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(float));
            size_t i = 0;
            for (int64_t i3 = 0; i3 < c.d[3]; ++i3)
                for (int64_t i2 = 0; i2 < c.d[2]; ++i2)
                    for (int64_t i1 = 0; i1 < c.d[1]; ++i1)
                        for (int64_t i0 = 0; i0 < c.d[0]; ++i0, ++i) {
                            const size_t s = (size_t)((i0 * c.a[0] / c.d[0]) +
                                                      c.a[0] * ((i1 * c.a[1] / c.d[1]) + c.a[1] * (i2 + c.a[2] * i3)));
                            bad += memcmp(&got[i], &xv[s], sizeof(float)) != 0;
                        }
            if (bad) fail = "differs from the reference";
        }
    }
    report("UPSCALE", c.name, fail, windows, bad, c.declined ? 0 : ny);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

struct BinCase {
    const char* name;
    int op; // 0 ADD, 1 SUB, 2 MUL, 3 DIV
    int64_t a[4], b[4];
    bool inplace, declined;
};

static void check_binary(ggml_backend_t be, const BinCase& c) {
    static const char* names[4] = { "ADD", "SUB", "MUL", "DIV" };
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.a[0], c.a[1], c.a[2], c.a[3]);
    struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.b[0], c.b[1], c.b[2], c.b[3]);
    struct ggml_tensor* y = nullptr;
    switch (c.op) {
        case 0: y = c.inplace ? ggml_add_inplace(ctx, a, b) : ggml_add(ctx, a, b); break;
        case 1: y = c.inplace ? ggml_sub_inplace(ctx, a, b) : ggml_sub(ctx, a, b); break;
        case 2: y = c.inplace ? ggml_mul_inplace(ctx, a, b) : ggml_mul(ctx, a, b); break;
        default: y = c.inplace ? ggml_div_inplace(ctx, a, b) : ggml_div(ctx, a, b); break;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t na = (size_t)ggml_nelements(a), nb = (size_t)ggml_nelements(b);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    long long windows = 0;
    if (!fail && c.declined) {
        if (ggml_backend_supports_op(be, y)) fail = "supports_op says yes";
    } else if (!fail) {
        std::vector<float> av(na), bv(nb), got(na);
        for (float& v : av) v = rnd();
        for (float& v : bv) v = c.op == 3 ? 1.5f + 0.5f * rnd() : rnd(); // divisors away from zero
        ggml_backend_tensor_set(a, av.data(), 0, na * sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, nb * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        const int64_t w0 = ggml_backend_ameva_get_window_node_count();
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        windows = (long long)(ggml_backend_ameva_get_window_node_count() - w0);
        if (!fail && windows != 1) fail = "the window counter did not advance by one";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, na * sizeof(float));
            size_t i = 0;
            for (int64_t i3 = 0; i3 < c.a[3]; ++i3)
                for (int64_t i2 = 0; i2 < c.a[2]; ++i2)
                    for (int64_t i1 = 0; i1 < c.a[1]; ++i1)
                        for (int64_t i0 = 0; i0 < c.a[0]; ++i0, ++i) {
                            const float x = av[i];
                            const float v = bv[(size_t)((i0 % c.b[0]) +
                                                        c.b[0] * ((i1 % c.b[1]) + c.b[1] * ((i2 % c.b[2]) + c.b[2] * (i3 % c.b[3]))))];
                            volatile float ref = c.op == 0 ? x + v : c.op == 1 ? x - v : c.op == 2 ? x * v : x / v;
                            const float r = ref;
                            if (c.op == 3) bad += !(std::fabs((double)got[i] - (double)r) <= 4e-7 * std::fabs((double)r));
                            else bad += memcmp(&got[i], &r, sizeof(float)) != 0 && !(got[i] == 0.0f && r == 0.0f);
                        }
            if (bad) fail = "differs from the reference";
        }
    }
    report(names[c.op], c.name, fail, windows, bad, c.declined ? 0 : na);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// dst[m][n] = sum over k of a[n][k] * b[m][k] per matrix (ggml MUL_MAT): a [K, N, H], b [K, M, H], dst [N, M, H]
struct MmCase {
    const char* name;
    int64_t K, N, M, H;
};

static void check_mul_mat(ggml_backend_t be, const MmCase& c) {
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.N, c.H);
    struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.M, c.H);
    struct ggml_tensor* y = ggml_mul_mat(ctx, a, b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t na = (size_t)ggml_nelements(a), nb = (size_t)ggml_nelements(b), ny = (size_t)ggml_nelements(y);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    long long windows = 0;
    if (!fail) {
        std::vector<float> av(na), bv(nb), got(ny);
        for (float& v : av) v = rnd();
        for (float& v : bv) v = rnd();
        ggml_backend_tensor_set(a, av.data(), 0, na * sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, nb * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        const int64_t w0 = ggml_backend_ameva_get_window_node_count();
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        windows = (long long)(ggml_backend_ameva_get_window_node_count() - w0);
        if (!fail && windows != 1) fail = "the window counter did not advance by one";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, ny * sizeof(float));
            for (int64_t h = 0; h < c.H; ++h)
                for (int64_t m = 0; m < c.M; ++m)
                    for (int64_t n = 0; n < c.N; ++n) {
                        const float* ar = av.data() + (size_t)((h * c.N + n) * c.K);
                        const float* br = bv.data() + (size_t)((h * c.M + m) * c.K);
                        double s = 0, mag = 0;
                        for (int64_t k = 0; k < c.K; ++k) {
                            s += (double)ar[k] * br[k];
                            mag += std::fabs((double)ar[k] * br[k]);
                        }
                        const float g = got[(size_t)((h * c.M + m) * c.N + n)];
                        bad += !(std::fabs((double)g - s) <= 1e-5 * mag);
                    }
            if (bad) fail = "above the rounding bound";
        }
    }
    report("MUL_MAT", c.name, fail, windows, bad, ny);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void check_soft_max(ggml_backend_t be, const char* name, int64_t n0, int64_t n1, int64_t n2) {
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    struct ggml_tensor* y = ggml_soft_max_inplace(ctx, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    const size_t n = (size_t)ggml_nelements(x);
    const char* fail = buf ? nullptr : "allocation";
    size_t bad = 0;
    long long windows = 0;
    if (!fail) {
        std::vector<float> xv(n), got(n);
        for (float& v : xv) v = 4.0f * rnd();
        ggml_backend_tensor_set(x, xv.data(), 0, n * sizeof(float));
        if (!ggml_backend_supports_op(be, y)) fail = "supports_op says no";
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        const int64_t w0 = ggml_backend_ameva_get_window_node_count();
        if (!fail && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail = "graph_compute failed";
        windows = (long long)(ggml_backend_ameva_get_window_node_count() - w0);
        if (!fail && windows != 1) fail = "the window counter did not advance by one";
        if (!fail) {
            ggml_backend_tensor_get(y, got.data(), 0, n * sizeof(float));
            for (size_t r = 0; r < n / (size_t)n0; ++r) {
                const float* row = xv.data() + r * (size_t)n0;
                double mx = row[0], sum = 0;
                for (int64_t k = 1; k < n0; ++k) mx = std::fmax(mx, (double)row[k]);
                for (int64_t k = 0; k < n0; ++k) sum += std::exp((double)row[k] - mx);
                for (int64_t k = 0; k < n0; ++k) {
                    bad += !(std::fabs((double)got[r * (size_t)n0 + (size_t)k] - std::exp((double)row[k] - mx) / sum) <= 2e-6);
                }
            }
            if (bad) fail = "above 2e-6";
        }
    }
    report("SOFT_MAX", name, fail, windows, bad, n);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    // read by the backend when it first asks for the span of a binding
    setenv("AMEVA_BINDING_LIMIT", "65536", 1);
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    const UpCase up[] = {
        { "2x, result in windows of 17 planes", { 20, 12, 40, 1 }, { 40, 24, 40, 1 }, false },
        { "2x batch 2, source and result in windows", { 40, 30, 24, 2 }, { 80, 60, 24, 2 }, false },
        { "3x along w only, 1 plane per window", { 30, 110, 5, 1 }, { 90, 110, 5, 1 }, false },
        { "dimension 2 scaled as well (to be declined)", { 40, 40, 20, 1 }, { 40, 40, 40, 1 }, true },
    };
    for (const UpCase& c : up) check_upscale(be, c);
    const BinCase bin[] = {
        { "per-channel vector", 0, { 24, 20, 50, 1 }, { 1, 1, 50, 1 }, false, false },
        { "per-channel vector, in place, batch 2", 2, { 24, 20, 50, 2 }, { 1, 1, 50, 1 }, true, false },
        { "same shape (both above the limit)", 1, { 24, 20, 50, 2 }, { 24, 20, 50, 2 }, false, false },
        { "one batch element above the limit added to two", 0, { 24, 20, 50, 2 }, { 24, 20, 50, 1 }, false, false },
        { "scalar", 2, { 24, 20, 50, 2 }, { 1, 1, 1, 1 }, false, false },
        { "a row per plane and batch element (inside the limit)", 0, { 24, 20, 50, 2 }, { 24, 1, 50, 2 }, true, false },
        { "per-channel and batch vector", 1, { 24, 20, 50, 2 }, { 1, 1, 50, 2 }, false, false },
        { "per-channel vector", 3, { 24, 20, 50, 1 }, { 1, 1, 50, 1 }, false, false },
        { "src1 repeated along dimension 2 (to be declined)", 0, { 24, 20, 50, 1 }, { 24, 20, 25, 1 }, false, true },
    };
    for (const BinCase& c : bin) check_binary(be, c);
    const MmCase mm[] = {
        { "keys x queries per head: result above the limit", 40, 96, 80, 6 },
        { "values x attention weights: src1 and result above the limit", 96, 40, 80, 6 },
        { "K not a multiple of 4, one matrix per window", 37, 150, 100, 3 },
    };
    for (const MmCase& c : mm) check_mul_mat(be, c);
    check_soft_max(be, "attention weights in place, windows of 2 planes", 96, 80, 6);
    check_soft_max(be, "rows above one workgroup, 1 plane per window", 300, 40, 3);
    printf("WINDOW_CHECK %d/%d passed, binding limit=%s, nodes in windows=%lld\n", g_passed, g_total,
           getenv("AMEVA_BINDING_LIMIT"), (long long)ggml_backend_ameva_get_window_node_count());
    ggml_backend_free(be);
    return g_passed == g_total ? 0 : 1;
}
