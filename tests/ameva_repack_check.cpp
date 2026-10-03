// SPDX-License-Identifier: Apache-2.0
// AMEVA repacked weight layout: structural checks on the host.
//   * the three regions of a tensor add up to ggml_nbytes for every repackable type and shape
//   * repack followed by unrepack restores every byte, for random bytes (all bit patterns of every field) and for
//     real quantized data produced by ggml_quantize_chunk
//   * dequantizing the restored data with ggml gives the original values
// Built from ggml-ameva-repack.cpp and libggml-base only; the GPU side is covered by ameva_qmm_check.
#include "ggml.h"
#include "ggml-ameva-repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t g_seed = 0x2545F491u;
static uint32_t next_u32() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}

int main() {
    const enum ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                                    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS};
    const int64_t shapes[][2] = {{256, 1}, {512, 3}, {768, 7}, {4864, 5}};
    int failures = 0, cases = 0;
    for (enum ggml_type t : types) {
        AmevaRepackLayout L;
        if (!ameva_repack_layout(t, &L)) {
            printf("FAIL %s: no layout\n", ggml_type_name(t));
            ++failures;
            continue;
        }
        const uint32_t upb = L.block_elems / 32;
        if (upb * (L.qb + L.hb) + L.sb != L.block_bytes || L.block_bytes != ggml_type_size(t) ||
            L.block_elems != (uint32_t)ggml_blck_size(t)) {
            printf("FAIL %s: layout does not add up to the ggml block (%u units * (%u + %u) + %u vs %zu)\n",
                   ggml_type_name(t), upb, L.qb, L.hb, L.sb, ggml_type_size(t));
            ++failures;
            continue;
        }
        for (const auto& sh : shapes) {
            const int64_t K = sh[0], rows = sh[1];
            if (K % L.block_elems) continue;
            ++cases;
            const size_t bytes = ggml_row_size(t, K) * rows;
            AmevaRepackRegions r;
            if (!ameva_repack_regions(t, K, rows, &r) || r.s + rows * r.s_row != bytes) {
                printf("FAIL %s K=%lld rows=%lld: regions do not cover the tensor\n", ggml_type_name(t),
                       (long long)K, (long long)rows);
                ++failures;
                continue;
            }
            // 1. random bytes: every field takes arbitrary bit patterns
            std::vector<uint8_t> a(bytes), p(bytes), b(bytes);
            for (auto& x : a) x = (uint8_t)next_u32();
            const bool ok1 = ameva_repack(t, K, rows, a.data(), p.data()) &&
                             ameva_unrepack(t, K, rows, p.data(), b.data()) && memcmp(a.data(), b.data(), bytes) == 0;
            // 2. real quantized data, and its dequantized values after the round trip
            std::vector<float> f((size_t)K * rows), fa((size_t)K * rows), fb((size_t)K * rows);
            for (auto& x : f) x = (float)((int32_t)(next_u32() >> 8) - (1 << 23)) / (float)(1 << 23);
            ggml_quantize_chunk(t, f.data(), a.data(), 0, rows, K, nullptr);
            const bool ok2 = ameva_repack(t, K, rows, a.data(), p.data()) &&
                             ameva_unrepack(t, K, rows, p.data(), b.data()) && memcmp(a.data(), b.data(), bytes) == 0;
            const struct ggml_type_traits* tt = ggml_get_type_traits(t);
            tt->to_float(a.data(), fa.data(), K * rows);
            tt->to_float(b.data(), fb.data(), K * rows);
            const bool ok3 = memcmp(fa.data(), fb.data(), fa.size() * sizeof(float)) == 0;
            // 3. row ranges: repacking in two pieces gives the whole-tensor result, and a range unpacks to its rows
            const int64_t cut = rows / 2;
            const size_t rb = ggml_row_size(t, K);
            std::vector<uint8_t> p2(bytes, 0), sub((size_t)(rows - cut) * rb);
            const bool ok4 = ameva_repack_rows(t, K, rows, 0, cut, a.data(), p2.data()) &&
                             ameva_repack_rows(t, K, rows, cut, rows - cut, a.data() + cut * rb, p2.data()) &&
                             memcmp(p2.data(), p.data(), bytes) == 0;
            const bool ok5 = ameva_unrepack_rows(t, K, rows, cut, rows - cut, p.data(), sub.data()) &&
                             memcmp(sub.data(), a.data() + cut * rb, sub.size()) == 0;
            if (!ok1 || !ok2 || !ok3 || !ok4 || !ok5) {
                printf("FAIL %s K=%lld rows=%lld random_bytes=%d quantized=%d dequant=%d rows_repack=%d rows_unpack=%d\n",
                       ggml_type_name(t), (long long)K, (long long)rows, (int)ok1, (int)ok2, (int)ok3, (int)ok4,
                       (int)ok5);
                ++failures;
            }
        }
        printf("REPACK type=%-6s block=%u/%u q=%u h=%u s=%u\n", ggml_type_name(t), L.block_elems, L.block_bytes, L.qb,
               L.hb, L.sb);
    }
    printf("AMEVA_REPACK_CHECK=%s (%d cases, %d failed)\n", failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}
