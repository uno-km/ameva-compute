/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#pragma once
// AMEVA weight layout for the GPU ("repacked").
//
// ggml stores a quantized row as an array of blocks that interleave scales and quants (18, 22, 84, 210 ... bytes),
// so a GPU thread reading one 32-weight unit issues several unaligned 32-bit loads. The repacked layout separates the
// fields of every block of a tensor into three regions, each row-major and aligned:
//   Q  low quant bits of every 32-weight unit, QB bytes per unit (16: 4-bit, 8: 2-bit, 32: 8-bit)
//   H  high quant bits per unit, HB bytes (0, 4 for one bit, 8 for two bits per weight)
//   S  the block's scale fields, SB bytes per block, byte-for-byte as in ggml (d, m, dmin, packed scales)
// Q and H hold a canonical bit order: in a 16-byte Q unit, byte b carries weight b (low nibble) and weight b + 16
// (high nibble); 2-bit units and H units carry weight k in bits [w*k, w*k + w) of word k / (32 / w). Quant values and
// scale bytes are unchanged, so dequantization gives exactly ggml's numbers. The three regions together take exactly
// ggml_nbytes(tensor): sizes and offsets seen by ggml do not change.
//
//   tensor of R rows, U = K/32 units and B = K/block_elems blocks per row
//   Q region at 0                      row r at r*U*QB
//   H region at R*U*QB                 row r at H + r*U*HB
//   S region at R*U*(QB+HB)            row r at S + r*B*SB
#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct AmevaRepackLayout {
    uint32_t block_elems; // weights per ggml block
    uint32_t block_bytes; // bytes per ggml block
    uint32_t qb;          // Q bytes per 32-weight unit
    uint32_t hb;          // H bytes per unit
    uint32_t sb;          // S bytes per block
};

// Layout of a repackable type, or false (F32/F16/BF16 and types without an AMEVA kernel are never repacked).
bool ameva_repack_layout(enum ggml_type type, AmevaRepackLayout* out);

// Byte offsets of the three regions of a tensor with `rows` rows of K weights.
struct AmevaRepackRegions {
    uint64_t q, h, s;       // region starts
    uint64_t q_row, h_row;  // bytes per row in Q and H
    uint64_t s_row;         // bytes per row in S
};
bool ameva_repack_regions(enum ggml_type type, int64_t K, int64_t rows, AmevaRepackRegions* out);

// ggml layout -> repacked layout (and back) for a whole tensor of `rows` rows of K weights; src and dst must not
// overlap. Return false for a type that has no repacked layout or a K that is not a whole number of blocks.
bool ameva_repack(enum ggml_type type, int64_t K, int64_t rows, const void* src, void* dst);
bool ameva_unrepack(enum ggml_type type, int64_t K, int64_t rows, const void* src, void* dst);

// The same for rows [r0, r0 + nr) of a tensor of `rows` rows: `ggml` holds exactly those nr rows in the ggml layout,
// `tensor` is the start of the whole repacked tensor (only the bytes of those rows are read or written).
bool ameva_repack_rows(enum ggml_type type, int64_t K, int64_t rows, int64_t r0, int64_t nr, const void* ggml,
                       void* tensor);
bool ameva_unrepack_rows(enum ggml_type type, int64_t K, int64_t rows, int64_t r0, int64_t nr, const void* tensor,
                         void* ggml);
