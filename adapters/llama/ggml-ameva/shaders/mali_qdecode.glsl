// AMEVA Mali-Optimized Repacked-Weight Stream Decoder
// SPDX-License-Identifier: Apache-2.0
//
// Designed for ARM Mali (Valhall/Bifrost) architecture:
// Eliminates register spill by streaming weights into dot products immediately (no w[8] array).
// Peak register usage drops from ~90 registers down to < 20 registers per thread.

#if defined(TYPE_Q8_0)
#define QVEC uvec4
#define QPER 2u
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
uint spread4(uint x) { return ((x & 1u) << 4u) | ((x & 2u) << 11u) | ((x & 4u) << 18u) | ((x & 8u) << 25u); }
uint spread2(uint x) { return (x & 3u) | ((x & 0xCu) << 6u) | ((x & 0x30u) << 12u) | ((x & 0xC0u) << 18u); }

#if defined(TYPE_IQ4_NL) || defined(TYPE_IQ4_XS)
#define KV_LUT
shared float s_kv[16];
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

void qdecode_init(uint tid) {
    // The bound is the end of the binding, as in ameva_qdecode.glsl. A bound computed from this dispatch's rows
    // ((s_off >> 2) + (N * s_row >> 2) - 1) is one word short whenever s_off or N * s_row is 2 modulo 4: the last
    // scale field of the last row was then read from the word before it.
    s_last = uint(sw.length()) - 1u;
#ifdef KV_LUT
    if (tid < 16u) s_kv[tid] = kv_init(tid);
    barrier();
#endif
}

// Decode one 32-weight unit in 4-weight pairs and immediately dot with activations at ab, accumulating into s.
void decode_dot(uint qi, uint hi, uint sb, uint j, uint ab, inout float s) {
#if defined(TYPE_Q4_0)
    const uvec4 q = qv[qi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = (u8x4(q[i] & 0x0F0F0F0Fu) - 8.0) * d;
        vec4 w1 = (u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) - 8.0) * d;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q4_1)
    const uvec4 q = qv[qi];
    const float d = sldh(sb);
    const float m = sldh(sb + 2u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = u8x4(q[i] & 0x0F0F0F0Fu) * d + m;
        vec4 w1 = u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) * d + m;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q5_0)
    const uvec4 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = (u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) - 16.0) * d;
        vec4 w1 = (u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) - 16.0) * d;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q5_1)
    const uvec4 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb);
    const float m = sldh(sb + 2u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) * d + m;
        vec4 w1 = u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) * d + m;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q8_0)
    const uvec4 q0 = qv[qi];
    const uvec4 q1 = qv[qi + 1u];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = s8x4(q0[i]) * d;
        vec4 w1 = s8x4(q1[i]) * d;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_IQ4_NL)
    const uvec4 q = qv[qi];
    const float d = sldh(sb);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = kv4(q[i]) * d;
        vec4 w1 = kv4(q[i] >> 4u) * d;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_IQ4_XS)
    const uvec4 q = qv[qi];
    const uint ls = ((sld32(sb + 4u) >> (4u * j)) & 0xFu) | (((sld16(sb + 2u) >> (2u * j)) & 3u) << 4u);
    const float dl = sldh(sb) * float(int(ls) - 32);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = kv4(q[i]) * dl;
        vec4 w1 = kv4(q[i] >> 4u) * dl;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q2_K)
    const uvec2 q = qv[qi];
    const float d = sldh(sb + 16u), dmin = sldh(sb + 18u);
    const uint sc2 = sld16(sb + 8u * (j >> 2u) + 2u * (j & 3u));
    const uint sa = sc2 & 0xFFu, sbh = sc2 >> 8u;
    const float dl0 = d * float(sa & 0xFu), ml0 = dmin * float(sa >> 4u);
    const float dl1 = d * float(sbh & 0xFu), ml1 = dmin * float(sbh >> 4u);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = u8x4(spread2((q.x >> (8u * i)) & 0xFFu)) * dl0 - ml0;
        vec4 w1 = u8x4(spread2((q.y >> (8u * i)) & 0xFFu)) * dl1 - ml1;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q3_K)
    const uvec2 q = qv[qi];
    const uint h = hv[hi];
    const float d = sldh(sb + 12u);
    const uint a0 = sld32(sb), a1 = sld32(sb + 4u), t = sld32(sb + 8u);
    const uint si = 8u * (j >> 2u) + 2u * (j & 3u);
    const uint qsel = si >> 2u;
    const uint aux = ((((qsel & 1u) != 0u ? a1 : a0) >> (4u * (qsel >> 1u))) & 0x0F0F0F0Fu) |
                     (((t >> (2u * qsel)) & 0x03030303u) << 4u);
    const float dl0 = d * float(int((aux >> (8u * (si & 3u))) & 0xFFu) - 32);
    const float dl1 = d * float(int((aux >> (8u * ((si & 3u) + 1u))) & 0xFFu) - 32);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = (u8x4(spread2((q.x >> (8u * i)) & 0xFFu) | (spread4(h >> (4u * i)) >> 2u)) - 4.0) * dl0;
        vec4 w1 = (u8x4(spread2((q.y >> (8u * i)) & 0xFFu) | (spread4(h >> (16u + 4u * i)) >> 2u)) - 4.0) * dl1;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#elif defined(TYPE_Q4_K) || defined(TYPE_Q5_K)
    const uvec4 q = qv[qi];
    const float d = sldh(sb), dmin = sldh(sb + 2u);
    const uint s0 = sld32(sb + 4u), s1 = sld32(sb + 8u), s2 = sld32(sb + 12u);
    const uint k = 8u * (j & 3u);
    const uint b0 = (s0 >> k) & 0xFFu, b1 = (s1 >> k) & 0xFFu, b2 = (s2 >> k) & 0xFFu;
    const bool lo = j < 4u;
    const uint sc = lo ? (b0 & 63u) : ((b2 & 0xFu) | ((b0 >> 6u) << 4u));
    const uint mn = lo ? (b1 & 63u) : ((b2 >> 4u) | ((b1 >> 6u) << 4u));
    const float dl = d * float(sc), ml = dmin * float(mn);
#if defined(TYPE_Q5_K)
    const uint h = hv[hi];
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = u8x4((q[i] & 0x0F0F0F0Fu) | spread4(h >> (4u * i))) * dl - ml;
        vec4 w1 = u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | spread4(h >> (16u + 4u * i))) * dl - ml;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#else
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = u8x4(q[i] & 0x0F0F0F0Fu) * dl - ml;
        vec4 w1 = u8x4((q[i] >> 4u) & 0x0F0F0F0Fu) * dl - ml;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#endif
#elif defined(TYPE_Q6_K)
    const uvec4 q = qv[qi];
    const uvec2 h = hv[hi];
    const float d = sldh(sb + 16u);
    const int sc2 = int(sld16(sb + 8u * (j >> 2u) + 2u * (j & 3u)));
    const float dl0 = d * float(bitfieldExtract(sc2, 0, 8));
    const float dl1 = d * float(bitfieldExtract(sc2, 8, 8));
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        vec4 w0 = (u8x4((q[i] & 0x0F0F0F0Fu) | (spread2((h.x >> (8u * i)) & 0xFFu) << 4u)) - 32.0) * dl0;
        vec4 w1 = (u8x4(((q[i] >> 4u) & 0x0F0F0F0Fu) | (spread2((h.y >> (8u * i)) & 0xFFu) << 4u)) - 32.0) * dl1;
        s += dot(w0, a4[ab + i]);
        s += dot(w1, a4[ab + i + 4u]);
    }
#endif
}
