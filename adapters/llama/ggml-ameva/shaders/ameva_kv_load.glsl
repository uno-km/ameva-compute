// AMEVA KV-cache loads, shared by the FLASH_ATTN_EXT kernels (included, not compiled on its own). Included after
// ameva_common.glsl and after the including kernel has declared
//   uint k[] and / or uint v[]   the K and V bindings as 32-bit words
//   pc.kv_types                  K type in bits 0..3, V type in bits 4..7 (KV_* below)
// A kernel that binds only one of the two defines KV_ONLY_K or KV_ONLY_V before the include.
// K and V rows are addressed by an element index (f32, f16) or a byte address (q8_0, q4_0).

// K / V element types (pc.kv_types)
#define KV_F32 0u
#define KV_F16 1u
#define KV_Q8_0 2u
#define KV_Q4_0 3u

// four f16 elements from the odd element index i of binding arr
#define KV_F16_VEC(arr, i)                                                                                          \
    vec4(AMEVA_LOAD_F16(arr, i), AMEVA_LOAD_F16(arr, (i) + 1u), AMEVA_LOAD_F16(arr, (i) + 2u),                       \
         AMEVA_LOAD_F16(arr, (i) + 3u))
#if defined(KV_ONLY_K)
uint kv_word(bool is_k, uint wi) { return k[wi]; }
#define KV_LOAD_F16_VEC(is_k, i) KV_F16_VEC(k, i)
#elif defined(KV_ONLY_V)
uint kv_word(bool is_k, uint wi) { return v[wi]; }
#define KV_LOAD_F16_VEC(is_k, i) KV_F16_VEC(v, i)
#else
uint kv_word(bool is_k, uint wi) { return is_k ? k[wi] : v[wi]; }
#define KV_LOAD_F16_VEC(is_k, i) ((is_k) ? KV_F16_VEC(k, i) : KV_F16_VEC(v, i))
#endif
// 4 bytes starting at the even byte address p (ggml blocks are 34 / 18 bytes: fields start at even addresses)
uint kv_bytes4(bool is_k, uint p) {
    const uint w0 = kv_word(is_k, p >> 2u);
    if ((p & 2u) == 0u) return w0;
    return (w0 >> 16u) | (kv_word(is_k, (p >> 2u) + 1u) << 16u);
}
float kv_half(bool is_k, uint p) { return ameva_h2f((kv_word(is_k, p >> 2u) >> ((p & 2u) * 8u)) & 0xFFFFu); }

// elements e .. e + 3 (e a multiple of 4) of a K (is_k) or V row that starts at `row`: an element index (f32, f16)
// or a byte address (q8_0, q4_0). Uniform branch: the type is the same for the whole dispatch.
vec4 load4(bool is_k, uint row, uint e) {
    const uint t = is_k ? (pc.kv_types & 15u) : ((pc.kv_types >> 4u) & 15u);
    if (t == KV_F32) {
        const uint idx = row + e;
        return uintBitsToFloat(uvec4(kv_word(is_k, idx), kv_word(is_k, idx + 1u), kv_word(is_k, idx + 2u),
                                     kv_word(is_k, idx + 3u)));
    }
    if (t == KV_F16) {
        const uint idx = row + e;
        if ((idx & 1u) == 0u) {
            const uint w0 = kv_word(is_k, idx >> 1u), w1 = kv_word(is_k, (idx >> 1u) + 1u);
            return vec4(ameva_h2f(w0 & 0xFFFFu), ameva_h2f(w0 >> 16u), ameva_h2f(w1 & 0xFFFFu), ameva_h2f(w1 >> 16u));
        }
        return KV_LOAD_F16_VEC(is_k, idx);
    }
    const uint b = e >> 5u, j = e & 31u;
    if (t == KV_Q8_0) {
        // block: d f16 @0, qs int8[32] @2. y = qs[j] * d
        const uint bb = row + 34u * b;
        const int x = int(kv_bytes4(is_k, bb + 2u + j));
        return vec4(bitfieldExtract(x, 0, 8), bitfieldExtract(x, 8, 8), bitfieldExtract(x, 16, 8),
                    bitfieldExtract(x, 24, 8)) * kv_half(is_k, bb);
    }
    // KV_Q4_0 block: d f16 @0, qs[16] @2; element j < 16 is the low nibble of qs[j], j >= 16 the high nibble of
    // qs[j - 16]. y = (nibble - 8) * d
    const uint bb = row + 18u * b;
    const uint w = kv_bytes4(is_k, bb + 2u + (j & 15u));
    const uint n = j < 16u ? (w & 0x0F0F0F0Fu) : ((w >> 4u) & 0x0F0F0F0Fu);
    return (vec4(uvec4(n, n >> 8u, n >> 16u, n >> 24u) & 0xFFu) - 8.0) * kv_half(is_k, bb);
}
