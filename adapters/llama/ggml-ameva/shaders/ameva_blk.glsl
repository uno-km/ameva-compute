// AMEVA block-integer kernels (ameva_mmqb.comp, ameva_mmvqb.comp): what they share. Included after
// ameva_qdecode.glsl with one of TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K defined and GL_EXT_integer_dot_product enabled.
// The weights of a 32-weight unit are taken as int8 and multiplied with activations quantized to int8 with one scale
// per 256-element block (ameva_quant_a8b); the weight scales of these types are small integers per unit (or per
// 16-weight half) times one f16 per block, so the sums over a block stay in 32-bit integers.

#if defined(TYPE_Q6_K)
#define HALF_SCALES // an integer scale per 16-weight half; the weights are taken signed (v - 32), so no offset term
#elif !defined(TYPE_Q4_K) && !defined(TYPE_Q5_K)
#error "block-integer kernels: TYPE_Q4_K, TYPE_Q5_K or TYPE_Q6_K"
#endif

// the four bytes of a word as integers
ivec4 ub4(uint w) { return ivec4(uvec4(w, w >> 8u, w >> 16u, w >> 24u) & 0xFFu); }
// the low and the high 16 bits of each component, sign-extended
ivec4 lo16(ivec4 p) { return (p << 16) >> 16; }
ivec4 hi16(ivec4 p) { return p >> 16; }

// the 16 bytes at the even byte address a of S as four little-endian words
void sld4(uint a, out uint w[4]) {
    const uint wi = a >> 2u;
    const uint x0 = sw_at(wi), x1 = sw_at(wi + 1u), x2 = sw_at(wi + 2u), x3 = sw_at(wi + 3u);
    if ((a & 2u) == 0u) {
        w[0] = x0;
        w[1] = x1;
        w[2] = x2;
        w[3] = x3;
    } else {
        const uint x4 = sw_at(wi + 4u);
        w[0] = (x0 >> 16u) | (x1 << 16u);
        w[1] = (x1 >> 16u) | (x2 << 16u);
        w[2] = (x2 >> 16u) | (x3 << 16u);
        w[3] = (x3 >> 16u) | (x4 << 16u);
    }
}

// ameva_qdecode.glsl's spread4 and spread2 on four words at once: bit k (k = 0..3) of each component -> bit 4 of its
// byte k; 2-bit field k of each component -> bits 0..1 of its byte k
uvec4 spread4v(uvec4 x) { return ((x & 1u) << 4u) | ((x & 2u) << 11u) | ((x & 4u) << 18u) | ((x & 8u) << 25u); }
uvec4 spread2v(uvec4 x) { return (x & 3u) | ((x & 0xCu) << 6u) | ((x & 0x30u) << 12u) | ((x & 0xC0u) << 18u); }

// The 32 weights of one unit as int8: byte k of component c of lo is the weight 4c + k, of hi the weight 16 + 4c + k.
// q4_K, q5_K: v (0..15, 0..31) as ameva_qdecode.glsl's decode_i8 gives it; q6_K: v - 32 (-32..31).
void unit_i8(uint qi, uint hi, out uvec4 lo_out, out uvec4 hi_out) {
    const uvec4 q = qv[qi];
    uvec4 l = q & 0x0F0F0F0Fu;
    uvec4 h = (q >> 4u) & 0x0F0F0F0Fu;
#if defined(TYPE_Q5_K)
    const uvec4 b = uvec4(hv[hi]); // bit k: the fifth bit of weight k
    l = l | spread4v(b >> uvec4(0u, 4u, 8u, 12u));
    h = h | spread4v(b >> uvec4(16u, 20u, 24u, 28u));
#elif defined(TYPE_Q6_K)
    const uvec2 b = hv[hi]; // 2-bit field k of x: bits 4..5 of weight k, of y: of weight 16 + k
    l = l | (spread2v(uvec4(b.x) >> uvec4(0u, 8u, 16u, 24u)) << 4u);
    h = h | (spread2v(uvec4(b.y) >> uvec4(0u, 8u, 16u, 24u)) << 4u);
    // v - 32 as a signed byte: v & 31 when bit 5 is set, otherwise v | 0xE0
    l = (l & 0x1F1F1F1Fu) | (((~l >> 5u) & 0x01010101u) * 0xE0u);
    h = (h & 0x1F1F1F1Fu) | (((~h >> 5u) & 0x01010101u) * 0xE0u);
#endif
    lo_out = l;
    hi_out = h;
}

// sum of the sixteen products of one half of a unit
int dot16(uvec4 w, uvec4 a) {
    return dotPacked4x8EXT(int(w.x), int(a.x)) + dotPacked4x8EXT(int(w.y), int(a.y)) +
           dotPacked4x8EXT(int(w.z), int(a.z)) + dotPacked4x8EXT(int(w.w), int(a.w));
}

// Weight row i of the invocation in unit u = 8 b + j8: its 32 weights as int8 (wlo, whi) and its integer scales (sc0,
// and sc1 with HALF_SCALES), from the block's scale words scw (four per weight row, as sld4 loads the first 16 bytes
// of the block's scales). The names n0, U, u, j8, scw, wlo, whi, sc0, sc1 are the caller's.
// A macro and not the body of a loop over the weight rows: as a loop over i the Mali OpenCL compiler keeps it a loop
// once the body holds the high-bit decode of q5_K or q6_K, and the kernel of two weight rows (ameva_mmqb, 8 activation
// rows) then takes 67 ms (q5_K) and 62 ms (q6_K) instead of 24 and 28 (Mali-G68 r32p1, OpenCL, 352x2048x2048,
// 2026-10-07; q4_K is 19 ms either way). Writing out only this loop is what moved the time; with the loop over the
// activation rows written out as well the same kernels take 33 and 46 ms.
#if defined(HALF_SCALES)
#define UNIT_SCALES(i) \
    const int pr = int(scw[4u * (i) + (j8 >> 1u)] >> (16u * (j8 & 1u))); \
    sc0[i] = (pr << 24) >> 24; \
    sc1[i] = (pr << 16) >> 24;
#else
#define UNIT_SCALES(i) \
    const uint k = 8u * (j8 & 3u); \
    const uint b0 = (scw[4u * (i) + 1u] >> k) & 0xFFu, b2 = (scw[4u * (i) + 3u] >> k) & 0xFFu; \
    sc0[i] = int(j8 < 4u ? (b0 & 63u) : ((b2 & 0xFu) | ((b0 >> 6u) << 4u)));
#endif
#define UNIT_ROW(i) { \
    const uint row = min(n0 + (i), pc.N - 1u); \
    uvec4 l, h; \
    unit_i8(pc.q_off + (row * U + u) * QPER, pc.h_off + row * U + u, l, h); \
    wlo[i] = l; \
    whi[i] = h; \
    UNIT_SCALES(i) \
}
