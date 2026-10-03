// SPDX-License-Identifier: Apache-2.0
// AMEVA row-chunked element-wise ops: SCALE -> TANH -> SCALE (the final logit softcap of Gemma-2) on f32 tensors larger
// than one storage-buffer binding, which the engine processes in row chunks.
//
// Run with AMEVA_BINDING_LIMIT=<bytes> below the tensor size (e.g. 1048576) to force chunking on any device; without
// it the tensors are chunked only where they exceed maxStorageBufferRange. Each case checks that supports_op accepts
// the ops, that the graph completes, and that every output matches a double-precision reference of the same formulas
// to 1e-6 relative (ameva_tanh is the Cephes scheme, ~1e-7). Shapes: contiguous rows, rows with padding (a view of a
// wider tensor), and a 3-D tensor. Exit 0 only if every case passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0x6a09e667u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

struct Case {
    const char* name;
    int64_t ne0, ne1, ne2;
    int64_t pad0; // extra elements per row of the source tensor (the op reads a view of the first ne0)
};

static bool run_case(ggml_backend_t be, const Case& c) {
    const float s_in = 1.0f / 30.0f, s_out = 30.0f; // softcap 30
    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne0 + c.pad0, c.ne1, c.ne2);
    struct ggml_tensor* x = c.pad0 ? ggml_view_3d(ctx, src, c.ne0, c.ne1, c.ne2, src->nb[1], src->nb[2], 0) : src;
    struct ggml_tensor* y = ggml_scale(ctx, x, s_in);
    y = ggml_tanh(ctx, y);
    y = ggml_scale(ctx, y, s_out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(be));
    if (!buf) {
        printf("  FAIL %s: allocation\n", c.name);
        ggml_free(ctx);
        return false;
    }
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        struct ggml_tensor* n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            printf("  FAIL %s: supports_op declined %s\n", c.name, ggml_op_desc(n));
            ok = false;
        }
    }
    const size_t n_src = (size_t)ggml_nelements(src);
    std::vector<float> h(n_src);
    for (float& v : h) v = 60.0f * rnd();
    ggml_backend_tensor_set(src, h.data(), 0, n_src * sizeof(float));
    if (ok && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  FAIL %s: graph_compute\n", c.name);
        ok = false;
    }
    double max_rel = 0.0;
    if (ok) {
        std::vector<float> out((size_t)ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
        for (int64_t i2 = 0; i2 < c.ne2; ++i2) {
            for (int64_t i1 = 0; i1 < c.ne1; ++i1) {
                for (int64_t i0 = 0; i0 < c.ne0; ++i0) {
                    const double xv = h[(size_t)((i2 * c.ne1 + i1) * (c.ne0 + c.pad0) + i0)];
                    const double ref = (double)s_out * std::tanh((double)s_in * xv);
                    const double got = out[(size_t)((i2 * c.ne1 + i1) * c.ne0 + i0)];
                    const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
                    if (!std::isfinite(got) || rel > max_rel) max_rel = std::isfinite(got) ? rel : 1e30;
                }
            }
        }
        if (max_rel > 1e-6) ok = false;
    }
    const size_t bytes = (size_t)ggml_nbytes(y);
    printf("CHUNK case=%-10s shape=[%lld,%lld,%lld] pad=%lld bytes=%zu max_rel=%.3g status=%s\n", c.name,
           (long long)c.ne0, (long long)c.ne1, (long long)c.ne2, (long long)c.pad0, bytes, max_rel, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    // logits-like rows (vocab x tokens), rows with padding, and a 3-D tensor; each about 4 to 10 MB
    const Case cases[] = {
        { "logits", 32000, 64, 1, 0 },
        { "padded", 4093, 300, 1, 7 },
        { "3d", 1024, 40, 37, 0 },
    };
    int failed = 0;
    for (const Case& c : cases) failed += run_case(be, c) ? 0 : 1;
    ggml_backend_free(be);
    printf("AMEVA_CHUNK_CHECK=%s (%d/%d cases)\n", failed ? "FAIL" : "PASS", (int)(sizeof(cases) / sizeof(cases[0])) - failed,
           (int)(sizeof(cases) / sizeof(cases[0])));
    return failed ? 1 : 0;
}
