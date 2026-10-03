// AMEVA repacked-weight decoder (ggml-ameva-repack.h), shared by ameva_qmv and ameva_qgemm. Included after
// ameva_common.glsl, with exactly one -DTYPE_<T> defined. Declares bindings 0..2:
//   binding 0: Q (QVEC elements), binding 1: H (HVEC elements; any range when the type has no high bits),
//   binding 2: S as 32-bit words
// decode() turns one 32-weight unit into 8 vec4 with exactly the arithmetic of ggml's dequantize_row_<type>.
// qdecode_init() must be called by every invocation at the start of main (it contains a barrier for the IQ4 table).

#if defined(TYPE_Q8_0)
#define QVEC uvec4
#define QPER 2u // Q vector elements per unit
#elif defined(TYPE_Q2_K) || defined(TYPE_Q3_K)
#define QVEC uvec2
#define QPER 1u
#else
#define QVEC uvec4
#define QPER 1u
#endif
#if defined(TYPE_Q6_K)
#define HVEC uvec2
#else
#define HVEC uint
#endif

layout(std430, binding = 0) readonly buffer BufQ { QVEC qv[]; };
layout(std430, binding = 1) readonly buffer BufH { HVEC hv[]; };
layout(std430, binding = 2) readonly buffer BufS { uint sw[]; };

uint s_last; // last word of the S binding: a clamped word only feeds bytes outside the field being read
uint sw_at(uint wi) { return sw[min(wi, s_last)]; }
// little-endian loads at even byte addresses of S (every scale field starts at an even offset)
uint sld32(uint a) {
    const uint wi = a >> 2u;
    if ((a & 2u) == 0u) return sw_at(wi);
    return (sw_at(wi) >> 16u) | (sw_at(wi + 1u) << 16u);
}
uint sld16(uint a) { return (sw_at(a >> 2u) >> ((a & 2u) * 8u)) & 0xFFFFu; }
float sldh(uint a) { return ameva_h2f(sld16(a)); }

vec4 u8x4(uint w) { return vec4(uvec4(w, w >> 8u, w >> 16u, w >> 24u) & 0xFFu); }
vec4 s8x4(uint w) {
    const int x = int(w);
    return vec4(bitfieldExtract(x, 0, 8), bitfieldExtract(x, 8, 8), bitfieldExtract(x, 16, 8), bitfieldExtract(x, 24, 8));
}
// bit k (k = 0..3) of x -> bit 4 of byte k
uint spread4(uint x) { return ((x & 1u) << 4u) | ((x & 2u) << 11u) | ((x & 4u) << 18u) | ((x & 8u) << 25u); }
// 2-bit field k (k = 0..3) of x -> bits 0..1 of byte k
uint spread2(uint x) { return (x & 3u) | ((x & 0xCu) << 6u) | ((x & 0x30u) << 12u) | ((x & 0xC0u) << 18u); }

#if defined(TYPE_IQ4_NL) || defined(TYPE_IQ4_XS)
#define KV_LUT
shared float s_kv[16];
// kvalues_iq4nl (ggml-common.h) as signed bytes
float kv_init(uint i) {
    const uint w = (i < 8u) ? ((i < 4u) ? 0xBFAD9881u : 0xF6EADDCFu) : ((i < 12u) ? 0x26190D01u : 0x71594535u);
    return float(bitfieldExtract(int(w), int(8u * (i & 3u)), 8));
}
vec4 kv4(uint q) { return vec4(s_kv[q & 0xFu], s_kv[(q >> 8u) & 0xFu], s_kv[(q >> 16u) & 0xFu], s_kv[(q >> 24u) & 0xFu]); }
#endif

#if defined(TYPE_Q4_0) || defined(TYPE_Q4_1) || defined(TYPE_Q5_0) || defined(TYPE_Q5_1) || defined(TYPE_Q8_0) || \
    defined(TYPE_IQ4_NL)
#define BLOCK_ELEMS 32u
#else
#define BLOCK_ELEMS 256u
#endif
// S bytes per block (ggml-ameva-repack.cpp kTypes)
#if defined(TYPE_Q4_0) || defined(TYPE_Q5_0) || defined(TYPE_Q8_0) || defined(TYPE_IQ4_NL)
#define SB 2u
#elif defined(TYPE_Q4_1) || defined(TYPE_Q5_1)
#define SB 4u
#elif defined(TYPE_Q2_K)
#define SB 20u
#elif defined(TYPE_Q3_K)
#define SB 14u
#elif defined(TYPE_Q4_K) || defined(TYPE_Q5_K)
#define SB 16u
#elif defined(TYPE_Q6_K)
#define SB 18u
#elif defined(TYPE_IQ4_XS)
#define SB 8u
#else
#error "define one TYPE_<T>"
#endif
#define UPB (BLOCK_ELEMS / 32u)

// One unit: qi / hi = its Q / H vector index, sb = byte address of its block's S fields, j = unit inside the block.
// w[i] = weights 4i..4i+3, w[i + 4] = weights 16+4i..16+4i+3 (i = 0..3).
void decode(uint qi, uint hi, uint sb, uint j, out vec4 w[8]) {
#if defined(TYPE_Q4_0)
    // S: d@0. y = (q - 8) * d
    const uvec4 q = qv[qi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = (u8x4(q[i] & 0x0F0F0F0Fu) - 8.0) * d;
        w[i + 4u] = (u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) - 8.0) * d;
    }
#elif defined(TYPE_Q4_1)
    // S: d@0 m@2. y = q * d + m
    const uvec4 q = qv[qi];
    const float d = sldh(sb), m = sldh(sb + 2u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = u8x4(q[i] & 0x0F0F0F0Fu) * d + m;
        w[i + 4u] = u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) * d + m;
    }
#elif defined(TYPE_Q5_0)
    // S: d@0, H: bit k = fifth bit of weight k. y = (q - 16) * d
    const uvec4 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = (u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) - 16.0) * d;
        w[i + 4u] = (u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) - 16.0) * d;
    }
#elif defined(TYPE_Q5_1)
    // S: d@0 m@2. y = q * d + m
    const uvec4 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb), m = sldh(sb + 2u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) * d + m;
        w[i + 4u] = u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) * d + m;
    }
#elif defined(TYPE_Q8_0)
    // S: d@0, Q: 32 int8. y = q * d
    const uvec4 q0 = qv[qi], q1 = qv[qi + 1u];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = s8x4(q0[i]) * d;
        w[i + 4u] = s8x4(q1[i]) * d;
    }
#elif defined(TYPE_IQ4_NL)
    // S: d@0. y = kvalues[q] * d
    const uvec4 q = qv[qi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = kv4(q[i]) * d;
        w[i + 4u] = kv4(q[i] >> 4u) * d;
    }
#elif defined(TYPE_IQ4_XS)
    // S: d@0 scales_h@2 scales_l[4]@4. ls = ((scales_l[j/2] >> 4(j%2)) & 0xF) | (((scales_h >> 2j) & 3) << 4),
    // y = kvalues[q] * (d * (ls - 32))
    const uvec4 q = qv[qi];
    const uint ls = ((sld32(sb + 4u) >> (4u * j)) & 0xFu) | (((sld16(sb + 2u) >> (2u * j)) & 3u) << 4u);
    const float dl = sldh(sb) * float(int(ls) - 32);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = kv4(q[i]) * dl;
        w[i + 4u] = kv4(q[i] >> 4u) * dl;
    }
#elif defined(TYPE_Q2_K)
    // S: scales[16]@0 d@16 dmin@18. Unit j = 4n + jj: weights 0..15 use scales[8n + 2jj], 16..31 scales[8n + 2jj + 1];
    // y = d * (sc & 0xF) * q - dmin * (sc >> 4). Q: weight k at bits 2(k%16) of word k/16.
    const uvec2 q = qv[qi];
    const float d = sldh(sb + 16u), dmin = sldh(sb + 18u);
    const uint sc2 = sld16(sb + 8u * (j >> 2u) + 2u * (j & 3u));
    const uint sa = sc2 & 0xFFu, sbh = sc2 >> 8u;
    const float dl0 = d * float(sa & 0xFu), ml0 = dmin * float(sa >> 4u);
    const float dl1 = d * float(sbh & 0xFu), ml1 = dmin * float(sbh >> 4u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = u8x4(spread2((q.x >> (8u * i)) & 0xFFu)) * dl0 - ml0;
        w[i + 4u] = u8x4(spread2((q.y >> (8u * i)) & 0xFFu)) * dl1 - ml1;
    }
#elif defined(TYPE_Q3_K)
    // S: scales[12]@0 d@12, unpacked with the kmask shuffle of dequantize_row_q3_K; H: bit k = third bit of weight k.
    // y = d * (scale - 32) * ((low2 | high << 2) - 4)
    const uvec2 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb + 12u);
    const uint a0 = sld32(sb), a1 = sld32(sb + 4u), t = sld32(sb + 8u);
    const uint si = 8u * (j >> 2u) + 2u * (j & 3u);
    // word si / 4 of the unpacked scales, without a branch (the lanes of a row hold different j): words 0..3 take the
    // low nibbles of a0, a1, a0 >> 4, a1 >> 4 and two high bits from t >> 0, 2, 4, 6
    const uint qsel = si >> 2u;
    const uint aux = ((((qsel & 1u) != 0u ? a1 : a0) >> (4u * (qsel >> 1u))) & 0x0F0F0F0Fu) |
                     (((t >> (2u * qsel)) & 0x03030303u) << 4u);
    const float dl0 = d * float(int((aux >> (8u * (si & 3u))) & 0xFFu) - 32);
    const float dl1 = d * float(int((aux >> (8u * ((si & 3u) + 1u))) & 0xFFu) - 32);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = (u8x4(spread2((q.x >> (8u * i)) & 0xFFu) | (spread4(h >> (4u * i)) >> 2u)) - 4.0) * dl0;
        w[i + 4u] = (u8x4(spread2((q.y >> (8u * i)) & 0xFFu) | (spread4(h >> (16u + 4u * i)) >> 2u)) - 4.0) * dl1;
    }
#elif defined(TYPE_Q4_K) || defined(TYPE_Q5_K)
    // S: d@0 dmin@2 scales[12]@4 (get_scale_min_k4(j)). y = d * sc * q - dmin * m
    const uvec4 q = qv[qi];
    const float d = sldh(sb), dmin = sldh(sb + 2u);
    const uint s0 = sld32(sb + 4u), s1 = sld32(sb + 8u), s2 = sld32(sb + 12u);
    // get_scale_min_k4 without a branch (the lanes of a row hold different j)
    const uint k = 8u * (j & 3u);
    const uint b0 = (s0 >> k) & 0xFFu, b1 = (s1 >> k) & 0xFFu, b2 = (s2 >> k) & 0xFFu;
    const bool lo = j < 4u;
    const uint sc = lo ? (b0 & 63u) : ((b2 & 0xFu) | ((b0 >> 6u) << 4u));
    const uint mn = lo ? (b1 & 63u) : ((b2 >> 4u) | ((b1 >> 6u) << 4u));
    const float dl = d * float(sc), ml = dmin * float(mn);
#if defined(TYPE_Q5_K)
    const uint h = hv[hi];
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) * dl - ml;
        w[i + 4u] = u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) * dl - ml;
    }
#else
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = u8x4(q[i] & 0x0F0F0F0Fu) * dl - ml;
        w[i + 4u] = u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) * dl - ml;
    }
#endif
#elif defined(TYPE_Q6_K)
    // S: scales[16] int8 @0 d@16. Unit j = 4n + t: weights 0..15 use scales[8n + 2t], 16..31 scales[8n + 2t + 1];
    // H: weight k at bits 2(k%16) of word k/16. y = d * scale * ((low4 | high2 << 4) - 32)
    const uvec4 q = qv[qi];
    const uvec2 h = hv[hi];
    const float d = sldh(sb + 16u);
    const int sc2 = int(sld16(sb + 8u * (j >> 2u) + 2u * (j & 3u)));
    const float dl0 = d * float(bitfieldExtract(sc2, 0, 8));
    const float dl1 = d * float(bitfieldExtract(sc2, 8, 8));
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        w[i] = (u8x4((q[i] & 0x0F0F0F0Fu) | (spread2((h.x >> (8u * i)) & 0xFFu) << 4u)) - 32.0) * dl0;
        w[i + 4u] = (u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | (spread2((h.y >> (8u * i)) & 0xFFu) << 4u)) - 32.0) * dl1;
    }
#endif
}

void qdecode_init(uint tid) {
    s_last = uint(sw.length()) - 1u;
#if defined(KV_LUT)
    if (tid < 16u) {
        s_kv[tid] = kv_init(tid);
    }
    barrier();
#endif
}
