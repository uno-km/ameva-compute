/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// AMEVA repacked weight layout: see ggml-ameva-repack.h. Block field offsets follow ggml-common.h.
#include "ggml-ameva-repack.h"

#include <cstring>

namespace {

enum Kind { K_Q4_0, K_Q4_1, K_Q5_0, K_Q5_1, K_Q8_0, K_Q2_K, K_Q3_K, K_Q4_K, K_Q5_K, K_Q6_K, K_IQ4_NL, K_IQ4_XS };

struct TypeInfo {
    const char* name; // ggml_type_name(): enum values are not stable across ggml versions, names are
    Kind kind;
    AmevaRepackLayout layout;
};

// {block_elems, block_bytes, qb, hb, sb}; per block: units * (qb + hb) + sb == block_bytes
const TypeInfo kTypes[] = {
    {"q4_0", K_Q4_0, {32, 18, 16, 0, 2}},     {"q4_1", K_Q4_1, {32, 20, 16, 0, 4}},
    {"q5_0", K_Q5_0, {32, 22, 16, 4, 2}},     {"q5_1", K_Q5_1, {32, 24, 16, 4, 4}},
    {"q8_0", K_Q8_0, {32, 34, 32, 0, 2}},     {"q2_K", K_Q2_K, {256, 84, 8, 0, 20}},
    {"q3_K", K_Q3_K, {256, 110, 8, 4, 14}},   {"q4_K", K_Q4_K, {256, 144, 16, 0, 16}},
    {"q5_K", K_Q5_K, {256, 176, 16, 4, 16}},  {"q6_K", K_Q6_K, {256, 210, 16, 8, 18}},
    {"iq4_nl", K_IQ4_NL, {32, 18, 16, 0, 2}}, {"iq4_xs", K_IQ4_XS, {256, 136, 16, 0, 8}},
};

const TypeInfo* find(enum ggml_type type) {
    const char* n = ggml_type_name(type);
    if (!n) return nullptr;
    for (const TypeInfo& t : kTypes) {
        if (strcmp(t.name, n) == 0) return &t;
    }
    return nullptr;
}

// ---- canonical unit codecs (v: the 32 values of one unit, in weight order) ----
void put4(const uint8_t v[32], uint8_t* q) {
    for (int b = 0; b < 16; ++b) q[b] = (uint8_t)((v[b] & 0xF) | ((v[b + 16] & 0xF) << 4));
}
void get4(const uint8_t* q, uint8_t v[32]) {
    for (int b = 0; b < 16; ++b) {
        v[b] = q[b] & 0xF;
        v[b + 16] = q[b] >> 4;
    }
}
void put2(const uint8_t v[32], uint8_t* q) { // two little-endian words, weight k at bits 2*(k%16) of word k/16
    memset(q, 0, 8);
    for (int k = 0; k < 32; ++k) {
        const int bit = 2 * (k % 16) + 32 * (k / 16);
        q[bit / 8] |= (uint8_t)((v[k] & 3) << (bit % 8));
    }
}
void get2(const uint8_t* q, uint8_t v[32]) {
    for (int k = 0; k < 32; ++k) {
        const int bit = 2 * (k % 16) + 32 * (k / 16);
        v[k] = (q[bit / 8] >> (bit % 8)) & 3;
    }
}
void put1(const uint8_t v[32], uint8_t* h) { // one little-endian word, weight k at bit k
    memset(h, 0, 4);
    for (int k = 0; k < 32; ++k) h[k / 8] |= (uint8_t)((v[k] & 1) << (k % 8));
}
void get1(const uint8_t* h, uint8_t v[32]) {
    for (int k = 0; k < 32; ++k) v[k] = (h[k / 8] >> (k % 8)) & 1;
}

// One block: ggml bytes <-> (Q units, H units, S bytes). `to_repacked` selects the direction; the same code walks
// the fields both ways so the two directions cannot disagree about the layout.
void block_codec(Kind kind, bool to_repacked, uint8_t* blk, uint8_t* q, uint8_t* h, uint8_t* s) {
    uint8_t v[32], hv[32];
    auto field = [&](uint8_t* blk_ptr, uint8_t* s_ptr, size_t n) { // copy a scale field
        if (to_repacked) memcpy(s_ptr, blk_ptr, n);
        else memcpy(blk_ptr, s_ptr, n);
    };
    switch (kind) {
        case K_Q4_0:
        case K_IQ4_NL: // d@0 qs[16]@2: qs is already canonical
            field(blk, s, 2);
            if (to_repacked) memcpy(q, blk + 2, 16);
            else memcpy(blk + 2, q, 16);
            break;
        case K_Q4_1: // d@0 m@2 qs@4
            field(blk, s, 4);
            if (to_repacked) memcpy(q, blk + 4, 16);
            else memcpy(blk + 4, q, 16);
            break;
        case K_Q5_0: // d@0 qh[4]@2 qs@6: qh bit j is the fifth bit of weight j (canonical)
            field(blk, s, 2);
            if (to_repacked) {
                memcpy(h, blk + 2, 4);
                memcpy(q, blk + 6, 16);
            } else {
                memcpy(blk + 2, h, 4);
                memcpy(blk + 6, q, 16);
            }
            break;
        case K_Q5_1: // d@0 m@2 qh@4 qs@8
            field(blk, s, 4);
            if (to_repacked) {
                memcpy(h, blk + 4, 4);
                memcpy(q, blk + 8, 16);
            } else {
                memcpy(blk + 4, h, 4);
                memcpy(blk + 8, q, 16);
            }
            break;
        case K_Q8_0: // d@0 qs[32]@2
            field(blk, s, 2);
            if (to_repacked) memcpy(q, blk + 2, 32);
            else memcpy(blk + 2, q, 32);
            break;
        case K_Q2_K: { // scales[16]@0 qs[64]@16 d@80 dmin@82; unit 4n+j: (qs[32n+l] >> 2j) & 3
            field(blk, s, 16);
            field(blk + 80, s + 16, 4);
            uint8_t* qs = blk + 16;
            if (!to_repacked) memset(qs, 0, 64);
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    uint8_t* qu = q + 8 * (4 * n + j);
                    if (to_repacked) {
                        for (int l = 0; l < 32; ++l) v[l] = (qs[32 * n + l] >> (2 * j)) & 3;
                        put2(v, qu);
                    } else {
                        get2(qu, v);
                        for (int l = 0; l < 32; ++l) qs[32 * n + l] |= (uint8_t)(v[l] << (2 * j));
                    }
                }
            }
            break;
        }
        case K_Q3_K: { // hmask[32]@0 qs[64]@32 scales[12]@96 d@108; unit 4n+j: low (qs[32n+l] >> 2j) & 3,
                       // high bit (hmask[l] >> (4n+j)) & 1
            field(blk + 96, s, 14);
            uint8_t* hm = blk;
            uint8_t* qs = blk + 32;
            if (!to_repacked) {
                memset(hm, 0, 32);
                memset(qs, 0, 64);
            }
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    const int u = 4 * n + j;
                    if (to_repacked) {
                        for (int l = 0; l < 32; ++l) {
                            v[l] = (qs[32 * n + l] >> (2 * j)) & 3;
                            hv[l] = (hm[l] >> u) & 1;
                        }
                        put2(v, q + 8 * u);
                        put1(hv, h + 4 * u);
                    } else {
                        get2(q + 8 * u, v);
                        get1(h + 4 * u, hv);
                        for (int l = 0; l < 32; ++l) {
                            qs[32 * n + l] |= (uint8_t)(v[l] << (2 * j));
                            hm[l] |= (uint8_t)(hv[l] << u);
                        }
                    }
                }
            }
            break;
        }
        case K_Q4_K:
        case K_Q5_K: { // d@0 dmin@2 scales[12]@4 [qh[32]@16] qs[128]@16|48; unit 2k+s: (qs[32k+l] >> 4s) & 0xF,
                       // Q5_K fifth bit (qh[l] >> u) & 1
            field(blk, s, 16);
            const bool q5 = kind == K_Q5_K;
            uint8_t* qh = blk + 16;
            uint8_t* qs = blk + (q5 ? 48 : 16);
            if (!to_repacked) {
                memset(qs, 0, 128);
                if (q5) memset(qh, 0, 32);
            }
            for (int k = 0; k < 4; ++k) {
                for (int sh = 0; sh < 2; ++sh) {
                    const int u = 2 * k + sh;
                    if (to_repacked) {
                        for (int l = 0; l < 32; ++l) {
                            v[l] = (qs[32 * k + l] >> (4 * sh)) & 0xF;
                            if (q5) hv[l] = (qh[l] >> u) & 1;
                        }
                        put4(v, q + 16 * u);
                        if (q5) put1(hv, h + 4 * u);
                    } else {
                        get4(q + 16 * u, v);
                        if (q5) get1(h + 4 * u, hv);
                        for (int l = 0; l < 32; ++l) {
                            qs[32 * k + l] |= (uint8_t)(v[l] << (4 * sh));
                            if (q5) qh[l] |= (uint8_t)(hv[l] << u);
                        }
                    }
                }
            }
            break;
        }
        case K_Q6_K: { // ql[128]@0 qh[64]@128 scales[16]@192 d@208; unit 4h+t: low nibble (t>>1) of
                       // ql[64h + 32(t&1) + l], high two bits (qh[32h + l] >> 2t) & 3
            field(blk + 192, s, 18);
            uint8_t* ql = blk;
            uint8_t* qh = blk + 128;
            if (!to_repacked) {
                memset(ql, 0, 128);
                memset(qh, 0, 64);
            }
            for (int hh = 0; hh < 2; ++hh) {
                for (int t = 0; t < 4; ++t) {
                    const int u = 4 * hh + t;
                    uint8_t* lq = ql + 64 * hh + 32 * (t & 1);
                    const int lsh = 4 * (t >> 1);
                    if (to_repacked) {
                        for (int l = 0; l < 32; ++l) {
                            v[l] = (lq[l] >> lsh) & 0xF;
                            hv[l] = (qh[32 * hh + l] >> (2 * t)) & 3;
                        }
                        put4(v, q + 16 * u);
                        put2(hv, h + 8 * u);
                    } else {
                        get4(q + 16 * u, v);
                        get2(h + 8 * u, hv);
                        for (int l = 0; l < 32; ++l) {
                            lq[l] |= (uint8_t)(v[l] << lsh);
                            qh[32 * hh + l] |= (uint8_t)(hv[l] << (2 * t));
                        }
                    }
                }
            }
            break;
        }
        case K_IQ4_XS: // d@0 scales_h@2 scales_l[4]@4 qs[128]@8: unit ib is qs[16ib .. 16ib+15], already canonical
            field(blk, s, 8);
            if (to_repacked) memcpy(q, blk + 8, 128);
            else memcpy(blk + 8, q, 128);
            break;
    }
}

// Rows [r0, r0 + nr) between `ggml` (those rows, ggml layout) and `tensor` (the whole repacked tensor of `rows` rows).
bool codec(enum ggml_type type, int64_t K, int64_t rows, int64_t r0, int64_t nr, uint8_t* ggml, uint8_t* tensor,
           bool to_repacked) {
    const TypeInfo* ti = find(type);
    if (!ti || K <= 0 || rows < 0 || K % ti->layout.block_elems || r0 < 0 || nr < 0 || r0 > rows || nr > rows - r0)
        return false;
    AmevaRepackRegions r;
    if (!ameva_repack_regions(type, K, rows, &r)) return false;
    const AmevaRepackLayout& L = ti->layout;
    const int64_t nb = K / L.block_elems;
    const uint32_t upb = L.block_elems / 32;
    uint8_t blk[256];
    for (int64_t row = r0; row < r0 + nr; ++row) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint64_t u0 = (uint64_t)b * upb;
            uint8_t* g = ggml + (uint64_t)(row - r0) * nb * L.block_bytes + (uint64_t)b * L.block_bytes;
            uint8_t* q = tensor + r.q + (uint64_t)row * r.q_row + u0 * L.qb;
            uint8_t* h = tensor + r.h + (uint64_t)row * r.h_row + u0 * L.hb;
            uint8_t* s = tensor + r.s + (uint64_t)row * r.s_row + (uint64_t)b * L.sb;
            if (to_repacked) {
                memcpy(blk, g, L.block_bytes);
                block_codec(ti->kind, true, blk, q, h, s);
            } else {
                // the codec writes the ggml block from the regions; it never modifies the region bytes
                block_codec(ti->kind, false, blk, q, h, s);
                memcpy(g, blk, L.block_bytes);
            }
        }
    }
    return true;
}

} // namespace

bool ameva_repack_layout(enum ggml_type type, AmevaRepackLayout* out) {
    const TypeInfo* ti = find(type);
    if (!ti) return false;
    if (out) *out = ti->layout;
    return true;
}

bool ameva_repack_regions(enum ggml_type type, int64_t K, int64_t rows, AmevaRepackRegions* out) {
    const TypeInfo* ti = find(type);
    if (!ti || !out || K <= 0 || rows < 0 || K % ti->layout.block_elems) return false;
    const AmevaRepackLayout& L = ti->layout;
    const uint64_t units = (uint64_t)K / 32, blocks = (uint64_t)K / L.block_elems;
    out->q_row = units * L.qb;
    out->h_row = units * L.hb;
    out->s_row = blocks * L.sb;
    out->q = 0;
    out->h = (uint64_t)rows * out->q_row;
    out->s = out->h + (uint64_t)rows * out->h_row;
    return true;
}

bool ameva_repack(enum ggml_type type, int64_t K, int64_t rows, const void* src, void* dst) {
    return codec(type, K, rows, 0, rows, (uint8_t*)const_cast<void*>(src), (uint8_t*)dst, true);
}

bool ameva_unrepack(enum ggml_type type, int64_t K, int64_t rows, const void* src, void* dst) {
    return codec(type, K, rows, 0, rows, (uint8_t*)dst, (uint8_t*)const_cast<void*>(src), false);
}

bool ameva_repack_rows(enum ggml_type type, int64_t K, int64_t rows, int64_t r0, int64_t nr, const void* ggml,
                       void* tensor) {
    return codec(type, K, rows, r0, nr, (uint8_t*)const_cast<void*>(ggml), (uint8_t*)tensor, true);
}

bool ameva_unrepack_rows(enum ggml_type type, int64_t K, int64_t rows, int64_t r0, int64_t nr, const void* tensor,
                         void* ggml) {
    return codec(type, K, rows, r0, nr, (uint8_t*)ggml, (uint8_t*)const_cast<void*>(tensor), false);
}
