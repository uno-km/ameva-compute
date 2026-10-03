// AMEVA op kernels: helpers shared by every kernel family (included, not compiled on its own).
//
// Conventions of every op kernel:
//   * each tensor is one storage-buffer binding: the host binds the buffer range that starts at the tensor's offset
//     rounded down to minStorageBufferOffsetAlignment and passes the remainder (in elements) in the push constants;
//   * all buffers are declared as uint words; f32 is read with uintBitsToFloat, f16 with unpackHalf2x16;
//   * strides in the push constants are in elements of the tensor's own type (ggml nb[] / type size);
//   * a 1-D index space is spread over a 2-D grid (65535 workgroups per dimension).

uint ameva_linear_id() {
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
}

uint ameva_workgroup_id() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// IEEE binary16 -> binary32, exact for every input including subnormals (GLSL's unpackHalf2x16 may flush f16
// subnormals to zero; ggml's CPU conversion does not).
float ameva_h2f(uint h) {
    const uint s = (h & 0x8000u) << 16u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 0u) {
        const float v = float(m) * 5.9604644775390625e-8; // m * 2^-24, exact
        return s != 0u ? -v : v;
    }
    if (e == 31u) {
        return uintBitsToFloat(s | 0x7F800000u | (m << 13u));
    }
    return uintBitsToFloat(s | ((e + 112u) << 23u) | (m << 13u));
}

// binary32 -> binary16 with round-to-nearest-even, the conversion ggml's CPU path uses (GLSL leaves the rounding of
// packHalf2x16 to the implementation).
uint ameva_f2h(float f) {
    const uint x = floatBitsToUint(f);
    const uint sign = (x >> 16u) & 0x8000u;
    const uint ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return sign | (ax > 0x7F800000u ? 0x7E00u : 0x7C00u); // NaN stays NaN, inf stays inf
    if (ax >= 0x477FF000u) return sign | 0x7C00u;                                 // rounds to inf
    if (ax < 0x38800000u) {                                                       // f16 subnormal or zero
        if (ax < 0x33000001u) return sign;                                        // <= 2^-25 rounds to 0
        const uint e = ax >> 23u;
        const uint m = (ax & 0x7FFFFFu) | 0x800000u;
        const uint shift = 126u - e;                                              // 14..24
        uint h = m >> shift;
        const uint rem = m & ((1u << shift) - 1u);
        const uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h += 1u;
        return sign | h;
    }
    uint h = (ax >> 13u) - (112u << 10u);
    const uint rem = ax & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;
    return sign | h;
}

// f16 element i of a word array (i counts halves from the start of the binding)
#define AMEVA_LOAD_F16(arr, i) ameva_h2f(((arr)[(i) >> 1u] >> (((i) & 1u) * 16u)) & 0xFFFFu)

// Stores f16 element i without touching the other half of its word: the two halves of one word may be written by
// different invocations, so the update is two atomics on disjoint bit ranges (clear own half, then set it).
#define AMEVA_STORE_F16(arr, i, v)                                                                                  \
    {                                                                                                                \
        const uint _sh = ((i) & 1u) * 16u;                                                                          \
        const uint _h = ameva_f2h(v);                                                                               \
        atomicAnd((arr)[(i) >> 1u], ~(0xFFFFu << _sh));                                                              \
        atomicOr((arr)[(i) >> 1u], _h << _sh);                                                                       \
    }

// tanh with the Cephes tanhf scheme: odd polynomial below 0.625, exp form above (relative error ~1e-7, no
// cancellation for small |x|, exact +-1 for large |x|).
float ameva_tanh(float x) {
    const float z = abs(x);
    if (z < 0.625) {
        const float s = x * x;
        return ((((-5.70498872745e-3 * s + 2.06390887954e-2) * s - 5.37397155531e-2) * s + 1.33314422036e-1) * s -
                3.33332819422e-1) * s * x + x;
    }
    const float e = exp(2.0 * z);
    const float t = 1.0 - 2.0 / (e + 1.0);
    return x < 0.0 ? -t : t;
}

// erf with the Numerical Recipes erfc Chebyshev fit (fractional error < 1.2e-7).
float ameva_erf(float x) {
    const float z = abs(x);
    const float t = 1.0 / (1.0 + 0.5 * z);
    const float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                    t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? 1.0 - r : r - 1.0;
}

// sin and cos of x for |x| < 2^24 * pi/2: Cody-Waite reduction by pi/2 with fma (the product q*C is exact inside
// fma), then the Cephes sinf/cosf minimax polynomials on [-pi/4, pi/4] (~1 ulp). GLSL's own sin/cos only promise an
// absolute error of 2^-11 on [-pi, pi], and RoPE angles reach the context length in radians.
vec2 ameva_sincos(float x) {
    const float q = round(x * 0.63661977236758134308); // 2/pi
    float r = fma(-q, 1.57079637050628662109375, x);    // float(pi/2)
    r = fma(-q, -4.37113900018624283e-8, r);              // pi/2 - float(pi/2), split in two floats
    r = fma(-q, -1.71512451e-15, r);
    const float r2 = r * r;
    const float s = fma(fma(fma(-1.9515295891e-4, r2, 8.3321608736e-3), r2, -1.6666654611e-1) * r2, r, r);
    const float c = fma(fma(fma(fma(2.443315711809948e-5, r2, -1.388731625493765e-3), r2, 4.166664568298827e-2), r2, -0.5),
                        r2, 1.0);
    const int k = int(q) & 3;
    if (k == 0) return vec2(s, c);
    if (k == 1) return vec2(c, -s);
    if (k == 2) return vec2(-s, -c);
    return vec2(-c, s);
}
