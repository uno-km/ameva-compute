// SPDX-License-Identifier: Apache-2.0
// GPU vs CPU parity for the whisper AMEVA matmul kernels (src/shaders/stt/matmul_{f16,q5_0}_f32.comp) through the
// backend's public entry point ameva_backend_run_cached_gpu_matmul. The CPU reference uses the weights exactly as
// the GPU sees them: F16 rounded by ggml, Q5_0 quantized by ggml_quantize_chunk and dequantized by ggml's own
// to_float, accumulated in double. Shapes include M/N that are not multiples of the 16x16 tile.
//
// Output: one "PARITY type=... M= N= K= max_abs_err= max_rel_err= status=PASS|FAIL" line per case, exit 0 only if
// every case passes. Needs the two .spv files (AMEVA_SPIRV_F16_PATH / AMEVA_SPIRV_Q5_0_PATH or the defaults).
#include "ggml.h"
#include "ggml-backend-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static uint32_t g_seed = 0x2468ace1u;
static float rnd() { // xorshift32 in [-1, 1)
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static bool run_case(enum ggml_type type, uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> w((size_t)N * K), a((size_t)M * K), w_ref((size_t)N * K);
    for (float& v : w) v = rnd();
    for (float& v : a) v = rnd();

    std::vector<uint8_t> wbytes;
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(w.size());
        ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t)w.size());
        ggml_fp16_to_fp32_row(h.data(), w_ref.data(), (int64_t)w.size());
        wbytes.assign((const uint8_t*)h.data(), (const uint8_t*)h.data() + h.size() * sizeof(ggml_fp16_t));
    } else {
        const size_t row = ggml_row_size(type, K);
        wbytes.resize(row * N);
        ggml_quantize_chunk(type, w.data(), wbytes.data(), 0, N, K, nullptr);
        const struct ggml_type_traits* tt = ggml_get_type_traits(type);
        for (uint32_t n = 0; n < N; ++n) tt->to_float(wbytes.data() + row * n, w_ref.data() + (size_t)n * K, K);
    }

    std::vector<float> out((size_t)M * N, NAN);
    const bool ok = ameva_backend_run_cached_gpu_matmul(wbytes.data(), wbytes.size(), a.data(), out.data(), M, N, K, type);
    if (!ok) {
        printf("PARITY type=%s M=%u N=%u K=%u status=FAIL reason=backend_returned_false\n", ggml_type_name(type), M, N, K);
        return false;
    }

    double max_abs = 0.0, max_rel = 0.0;
    bool pass = true;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                const double p = (double)w_ref[(size_t)n * K + k] * (double)a[(size_t)m * K + k];
                ref += p;
                mag += std::fabs(p);
            }
            const double g = out[(size_t)m * N + n];
            const double err = std::fabs(g - ref);
            // fp32 accumulation of K products: error bound proportional to the sum of magnitudes
            const double tol = 1e-5 * mag + 1e-6;
            if (!std::isfinite(g) || err > tol) pass = false;
            if (err > max_abs) max_abs = err;
            if (mag > 0 && err / mag > max_rel) max_rel = err / mag;
        }
    }
    printf("PARITY type=%s M=%u N=%u K=%u max_abs_err=%.3g max_rel_err=%.3g status=%s\n", ggml_type_name(type), M, N, K,
           max_abs, max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    struct Shape { uint32_t M, N, K; };
    const Shape shapes[] = {{1, 64, 64}, {3, 5, 32}, {17, 33, 96}, {64, 96, 320}, {40, 48, 1280}};
    int failed = 0, total = 0;
    for (enum ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q5_0}) {
        for (const Shape& s : shapes) {
            ++total;
            if (!run_case(type, s.M, s.N, s.K)) ++failed;
        }
    }
    ameva_backend_teardown_hardware();
    printf("STT_MATMUL_PARITY=%s (%d/%d cases passed)\n", failed ? "FAIL" : "PASS", total - failed, total);
    return failed ? 1 : 0;
}
