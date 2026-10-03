// SPDX-License-Identifier: Apache-2.0
// AMEVA quantized KV cache (q8_0 / q4_0, ggml layout) through the real ggml backend path.
//
// 1. SET_ROWS f32 -> q8_0 / q4_0 (the KV-cache write). Rows of 96 elements (3 blocks: every other row starts in the
//    middle of a 32-bit word), 128 and 256. Checks:
//    - rows that are not indexed keep their previous bytes (the 16-bit halves shared with a neighbour are written
//      with atomics on disjoint bits);
//    - every written block is a valid quantization of its 32 inputs: d is f16(amax / 127) within one f16 step (q8_0)
//      or exactly f16(max / -8) (q4_0), and every dequantized value is within the rounding bound of its input;
//    - at least 99% of the written blocks are byte-identical to ggml_quantize_chunk (the remaining ones differ by
//      the last bit of the division or by the tie rule, both allowed by the bound above).
// 2. CPY f32 -> quantized of the same 3-D shape into a strided view (the KV-cache shift writing K back): same checks.
// 3. CPY quantized -> f32 from a strided 3-D view (the KV-cache shift casting K to f32): bit-identical to ggml's
//    dequantize_row_<type> (type_traits.to_float).
// 4. FLASH_ATTN_EXT with q8_0 / q4_0 / f16 K and V in the KV-cache view layout of llama.cpp (2-D cache, 3-D view,
//    permuted), including mixed K/V types, grouped heads, a causal mask, head sizes 64 / 128 / 256 and a cache long
//    enough for key splits. The reference is computed in double from ggml's dequantization of the same bytes; the
//    relative error must stay below 1e-5 (fp32 accumulation only).
// Exit 0 only if every check passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0x3c6ef372u;
static uint32_t rnd_u32() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}
static float rnd() { return (float)(rnd_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f; }

static int g_failed = 0;
static void fail(const char* what, const char* detail) {
    printf("  FAIL %s: %s\n", what, detail);
    ++g_failed;
}

static uint16_t f16_bits(float f) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float f16_val(const uint8_t* p) {
    ggml_fp16_t h;
    memcpy(&h, p, 2);
    return ggml_fp16_to_fp32(h);
}

// Checks one block written by the GPU against its 32 inputs (see the header). Returns false and prints on failure.
static bool check_block(enum ggml_type t, const float* x, const uint8_t* blk, const char* what) {
    const struct ggml_type_traits* tt = ggml_get_type_traits(t);
    float deq[32];
    tt->to_float(blk, deq, 32);
    const float dgot = f16_val(blk);
    uint16_t hgot;
    memcpy(&hgot, blk, 2);
    float amax = 0.0f, mx = 0.0f;
    for (int j = 0; j < 32; ++j) {
        if (amax < fabsf(x[j])) {
            amax = fabsf(x[j]);
            mx = x[j];
        }
    }
    char msg[160];
    if (t == GGML_TYPE_Q8_0) {
        const int diff = (int)hgot - (int)f16_bits(amax / 127.0f);
        if (diff < -1 || diff > 1) {
            snprintf(msg, sizeof(msg), "q8_0 scale %g, expected f16(%g)", dgot, amax / 127.0f);
            fail(what, msg);
            return false;
        }
    } else if (hgot != f16_bits(mx / -8.0f)) {
        snprintf(msg, sizeof(msg), "q4_0 scale %g, expected f16(%g)", dgot, mx / -8.0f);
        fail(what, msg);
        return false;
    }
    // q8_0: half a step plus the f16 rounding of d over 127 steps; q4_0: one step (the top level clamps at 15)
    const float tol = fabsf(dgot) * (t == GGML_TYPE_Q8_0 ? 0.57f : 1.01f) + 1e-30f;
    for (int j = 0; j < 32; ++j) {
        if (!(fabsf(deq[j] - x[j]) <= tol)) {
            snprintf(msg, sizeof(msg), "element %d: x=%g dequantized=%g d=%g", j, x[j], deq[j], dgot);
            fail(what, msg);
            return false;
        }
    }
    return true;
}

struct Ctx {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~Ctx() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

static bool compute(ggml_backend_t be, ggml_context* ctx, ggml_tensor* out, const char* what) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor* n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            fail(what, ggml_op_desc(n));
            return false;
        }
    }
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fail(what, "graph_compute");
        return false;
    }
    return true;
}

// 1. SET_ROWS f32 -> quantized
static void test_set_rows(ggml_backend_t be, enum ggml_type t, int E) {
    const int n_cache = 37, n_src = 11;
    char what[96];
    snprintf(what, sizeof(what), "SET_ROWS f32->%s E=%d", ggml_type_name(t), E);
    Ctx c;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    c.ctx = ggml_init(ip);
    ggml_tensor* cache = ggml_new_tensor_2d(c.ctx, t, E, n_cache);
    ggml_tensor* src = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, E, n_src);
    ggml_tensor* idx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, n_src);
    ggml_tensor* out = ggml_set_rows(c.ctx, cache, src, idx);
    c.buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_get_default_buffer_type(be));
    if (!c.buf) return fail(what, "allocation");
    const size_t rb = ggml_row_size(t, E), cb = ggml_nbytes(cache);
    std::vector<uint8_t> before(cb);
    for (auto& b : before) b = (uint8_t)rnd_u32();
    ggml_backend_tensor_set(cache, before.data(), 0, cb);
    std::vector<float> x((size_t)E * n_src);
    for (auto& v : x) v = 4.0f * rnd();
    for (int r = 0; r < n_src; r += 4) x[(size_t)r * E + 5] = 0.0f; // a few zeros
    for (int j = 0; j < 32; ++j) x[(size_t)1 * E + j] = 0.0f;        // one all-zero block (d = 0)
    ggml_backend_tensor_set(src, x.data(), 0, x.size() * 4);
    std::vector<int64_t> rows(n_cache);
    for (int i = 0; i < n_cache; ++i) rows[i] = i;
    for (int i = n_cache - 1; i > 0; --i) std::swap(rows[i], rows[rnd_u32() % (i + 1)]);
    rows.resize(n_src);
    ggml_backend_tensor_set(idx, rows.data(), 0, n_src * sizeof(int64_t));
    if (!compute(be, c.ctx, out, what)) return;
    std::vector<uint8_t> after(cb);
    ggml_backend_tensor_get(cache, after.data(), 0, cb);
    std::vector<uint8_t> ref(rb);
    int blocks = 0, identical = 0;
    std::vector<bool> written(n_cache, false);
    for (int r = 0; r < n_src; ++r) {
        written[rows[r]] = true;
        ggml_quantize_chunk(t, x.data() + (size_t)r * E, ref.data(), 0, 1, E, nullptr);
        const uint8_t* got = after.data() + (size_t)rows[r] * rb;
        for (int b = 0; b < E / 32; ++b) {
            const size_t bs = ggml_type_size(t);
            if (!check_block(t, x.data() + (size_t)r * E + 32 * b, got + b * bs, what)) return;
            ++blocks;
            identical += memcmp(got + b * bs, ref.data() + b * bs, bs) == 0;
        }
    }
    for (int i = 0; i < n_cache; ++i) {
        if (!written[i] && memcmp(after.data() + (size_t)i * rb, before.data() + (size_t)i * rb, rb) != 0) {
            char m[64];
            snprintf(m, sizeof(m), "row %d was not indexed but changed", i);
            return fail(what, m);
        }
    }
    if (identical * 100 < blocks * 99) {
        char m[96];
        snprintf(m, sizeof(m), "only %d of %d blocks byte-identical to ggml_quantize_chunk", identical, blocks);
        return fail(what, m);
    }
    printf("KVQ %-28s blocks=%d identical=%d PASS\n", what, blocks, identical);
}

// 2. + 3. CPY into / from a strided 3-D view of a 2-D cache [D * H, n] (the KV-cache shift)
static void test_cpy(ggml_backend_t be, enum ggml_type t, int D, int H, int n) {
    char what[96];
    snprintf(what, sizeof(what), "CPY %s<->f32 D=%d H=%d n=%d", ggml_type_name(t), D, H, n);
    Ctx c;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    c.ctx = ggml_init(ip);
    ggml_tensor* cache = ggml_new_tensor_2d(c.ctx, t, (int64_t)D * H, n);
    ggml_tensor* view = ggml_view_3d(c.ctx, cache, D, H, n, ggml_row_size(t, D), ggml_row_size(t, (int64_t)D * H), 0);
    ggml_tensor* src = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, D, H, n);
    ggml_tensor* store = ggml_cpy(c.ctx, src, view);
    ggml_tensor* cast = ggml_cast(c.ctx, view, GGML_TYPE_F32);
    c.buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_get_default_buffer_type(be));
    if (!c.buf) return fail(what, "allocation");
    const size_t cb = ggml_nbytes(cache);
    std::vector<float> x((size_t)D * H * n);
    for (auto& v : x) v = 2.0f * rnd();
    ggml_backend_tensor_set(src, x.data(), 0, x.size() * 4);
    std::vector<uint8_t> zero(cb, 0);
    ggml_backend_tensor_set(cache, zero.data(), 0, cb);
    // f32 -> quantized into the view (the view covers the whole cache)
    if (!compute(be, c.ctx, store, what)) return;
    std::vector<uint8_t> q(cb);
    ggml_backend_tensor_get(cache, q.data(), 0, cb);
    const size_t bs = ggml_type_size(t);
    for (size_t b = 0; b < x.size() / 32; ++b) {
        if (!check_block(t, x.data() + 32 * b, q.data() + b * bs, what)) return;
    }
    // quantized -> f32 from the view: bit-identical to ggml's dequantization
    if (!compute(be, c.ctx, cast, what)) return;
    std::vector<float> got(x.size()), ref(x.size());
    ggml_backend_tensor_get(cast, got.data(), 0, got.size() * 4);
    ggml_get_type_traits(t)->to_float(q.data(), ref.data(), (int64_t)x.size());
    if (memcmp(got.data(), ref.data(), got.size() * 4) != 0) return fail(what, "cast to f32 differs from dequantize_row");
    printf("KVQ %-28s blocks=%zu PASS\n", what, x.size() / 32);
}

// 4. FLASH_ATTN_EXT with quantized K / V
struct FaCase {
    int D, DV, H, Hkv, n_kv, n_q;
    enum ggml_type kt, vt;
};

static void test_fa(ggml_backend_t be, const FaCase& fc) {
    char what[128];
    snprintf(what, sizeof(what), "FA K=%s V=%s D=%d H=%d/%d kv=%d q=%d", ggml_type_name(fc.kt), ggml_type_name(fc.vt),
             fc.D, fc.H, fc.Hkv, fc.n_kv, fc.n_q);
    Ctx c;
    ggml_init_params ip = { 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    c.ctx = ggml_init(ip);
    ggml_tensor* kc = ggml_new_tensor_2d(c.ctx, fc.kt, (int64_t)fc.D * fc.Hkv, fc.n_kv);
    ggml_tensor* vc = ggml_new_tensor_2d(c.ctx, fc.vt, (int64_t)fc.DV * fc.Hkv, fc.n_kv);
    ggml_tensor* k = ggml_permute(c.ctx, ggml_view_3d(c.ctx, kc, fc.D, fc.Hkv, fc.n_kv, ggml_row_size(fc.kt, fc.D),
                                                      ggml_row_size(fc.kt, (int64_t)fc.D * fc.Hkv), 0), 0, 2, 1, 3);
    ggml_tensor* v = ggml_permute(c.ctx, ggml_view_3d(c.ctx, vc, fc.DV, fc.Hkv, fc.n_kv, ggml_row_size(fc.vt, fc.DV),
                                                      ggml_row_size(fc.vt, (int64_t)fc.DV * fc.Hkv), 0), 0, 2, 1, 3);
    ggml_tensor* q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, fc.D, fc.n_q, fc.H);
    ggml_tensor* m = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, fc.n_kv, fc.n_q);
    const float scale = 1.0f / sqrtf((float)fc.D);
    ggml_tensor* out = ggml_flash_attn_ext(c.ctx, q, k, v, m, scale, 0.0f, 0.0f);
    c.buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_get_default_buffer_type(be));
    if (!c.buf) return fail(what, "allocation");
    // caches quantized by ggml on the CPU, dequantized by ggml for the reference
    std::vector<float> kf((size_t)fc.D * fc.Hkv * fc.n_kv), vf((size_t)fc.DV * fc.Hkv * fc.n_kv);
    for (auto& x : kf) x = rnd();
    for (auto& x : vf) x = rnd();
    std::vector<uint8_t> kb(ggml_nbytes(kc)), vb(ggml_nbytes(vc));
    ggml_quantize_chunk(fc.kt, kf.data(), kb.data(), 0, fc.n_kv, (int64_t)fc.D * fc.Hkv, nullptr);
    ggml_quantize_chunk(fc.vt, vf.data(), vb.data(), 0, fc.n_kv, (int64_t)fc.DV * fc.Hkv, nullptr);
    ggml_get_type_traits(fc.kt)->to_float(kb.data(), kf.data(), (int64_t)kf.size());
    ggml_get_type_traits(fc.vt)->to_float(vb.data(), vf.data(), (int64_t)vf.size());
    ggml_backend_tensor_set(kc, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(vc, vb.data(), 0, vb.size());
    std::vector<float> qf((size_t)fc.D * fc.n_q * fc.H);
    for (auto& x : qf) x = rnd();
    ggml_backend_tensor_set(q, qf.data(), 0, qf.size() * 4);
    // causal mask: query i sees keys j <= n_kv - n_q + i
    std::vector<ggml_fp16_t> mh((size_t)fc.n_kv * fc.n_q);
    std::vector<float> mf(mh.size());
    for (int i = 0; i < fc.n_q; ++i) {
        for (int j = 0; j < fc.n_kv; ++j) {
            const float val = j <= fc.n_kv - fc.n_q + i ? 0.0f : -INFINITY;
            mf[(size_t)i * fc.n_kv + j] = val;
            mh[(size_t)i * fc.n_kv + j] = ggml_fp32_to_fp16(val);
        }
    }
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * 2);
    if (!compute(be, c.ctx, out, what)) return;
    std::vector<float> got((size_t)fc.DV * fc.H * fc.n_q);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
    const int G = fc.H / fc.Hkv;
    double worst = 0.0;
    std::vector<double> s(fc.n_kv);
    for (int i = 0; i < fc.n_q; ++i) {
        for (int h = 0; h < fc.H; ++h) {
            const int hk = h / G;
            double mx = -INFINITY;
            for (int j = 0; j < fc.n_kv; ++j) {
                double dot = 0.0;
                for (int d = 0; d < fc.D; ++d) {
                    dot += (double)qf[((size_t)h * fc.n_q + i) * fc.D + d] *
                           (double)kf[((size_t)j * fc.Hkv + hk) * fc.D + d];
                }
                s[j] = dot * scale + mf[(size_t)i * fc.n_kv + j];
                mx = std::max(mx, s[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < fc.n_kv; ++j) sum += std::isinf(s[j]) ? 0.0 : exp(s[j] - mx);
            for (int d = 0; d < fc.DV; ++d) {
                double o = 0.0, mag = 0.0;
                for (int j = 0; j < fc.n_kv; ++j) {
                    if (std::isinf(s[j])) continue;
                    const double p = exp(s[j] - mx) / sum;
                    const double vv = vf[((size_t)j * fc.Hkv + hk) * fc.DV + d];
                    o += p * vv;
                    mag += p * fabs(vv);
                }
                const double g = got[((size_t)i * fc.H + h) * fc.DV + d];
                const double rel = fabs(g - o) / (mag + 1e-30);
                if (!std::isfinite(g) || rel > worst) worst = std::isfinite(g) ? rel : 1e30;
            }
        }
    }
    if (worst > 1e-5) {
        char m2[64];
        snprintf(m2, sizeof(m2), "max_rel=%.3g", worst);
        return fail(what, m2);
    }
    printf("KVQ %-44s max_rel=%.3g PASS\n", what, worst);
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    for (enum ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int E : { 96, 128, 256 }) test_set_rows(be, t, E);
        test_cpy(be, t, 64, 2, 37);
        test_cpy(be, t, 96, 3, 11); // 3 blocks per head row: heads start in the middle of words
    }
    const FaCase fa[] = {
        { 64, 64, 4, 2, 300, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
        { 64, 64, 4, 2, 300, 7, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 },
        { 128, 128, 8, 2, 700, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }, // long cache: key splits
        { 256, 256, 4, 4, 96, 3, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 },
        { 64, 64, 14, 2, 40, 5, GGML_TYPE_Q8_0, GGML_TYPE_F16 },
        { 96, 96, 6, 3, 130, 33, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 }, // 3 blocks per head row, prompt-sized tile
    };
    for (const FaCase& f : fa) test_fa(be, f);
    ggml_backend_free(be);
    printf("AMEVA_KVQ_CHECK=%s (%d failed checks)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
