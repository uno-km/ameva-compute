// SPDX-License-Identifier: Apache-2.0
// AMEVA OpenCL prototype for Arm Mali: q4_K MUL_MAT (dst = W^T x A) with packed int8 dot products.
//
// Why OpenCL: the Mali DDKs r32p1 (Galaxy A53) and r38p0 (Galaxy S21) do not expose
// VK_KHR_shader_integer_dot_product in Vulkan, so ggml-ameva's integer kernels (ameva_mmq, ameva_mmvq) cannot use the
// multiplier that the same drivers expose in OpenCL as cl_arm_integer_dot_product_accumulate_int8 (arm_dot_acc).
// These kernels are the arithmetic of ggml-ameva's verified Vulkan kernels written in OpenCL C:
//   weights      the AMEVA repacked q4_K layout (ggml-ameva-repack.h): per 32-weight unit 16 bytes of nibbles
//                (weight k < 16: low nibble of byte k, weight 16 + k: high nibble of byte k), per 256-weight block
//                16 bytes of scales (d f16, dmin f16, the 12 bytes of ggml's get_scale_min_k4). Weight value
//                = d * sc_j * q - dmin * m_j with (sc_j, m_j) of unit j of its block: shaders/ameva_qdecode.glsl.
//   activations  int8 per 32-element unit (d = amax / 127, q = round-to-even(x / d)) with the unit's d and the sum of
//                q: shaders/ameva_quant_a8.comp.
//   product      sum over units of d_a * (d sc * dot(q_w, q_a) - dmin m * sum(q_a)): shaders/ameva_mmq.comp.
// Kernels: quant_a8 (activations), gemv_q4k_i8 (one activation row: token generation), gemv_q4k_f32 (the same row
// with float weights and float activations, for comparison inside one API), gemm_q4k_i8 (many activation rows:
// prompt processing, a tile of 64 weight rows x 32 activation rows in local memory).
// Every buffer is read through 16-byte vector elements (uint4, int4, float4): a vload16 from a byte pointer has no
// known alignment, and the first form of these kernels, written that way, ran the tile at 14 GMAC/s on Mali-G78
// where ggml-ameva's f32 tile kernel in Vulkan reaches 48.
// Prototype: one weight type, sizes that are multiples of the tile, no row ranges. Bench: tests/ameva_cl_bench.c.
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable

#ifndef GEMV_ROWS
// Weight rows per work-item of the token-generation kernels. Mali-G78 (Galaxy S21, 2026-10-05), q4_K, ms per kernel
// with 1 / 2 / 4 / 8 rows: 8192x2048 1.21 / 1.15 / 1.44 / 2.0, 2048x8192 1.03 / 1.58 / 4.0 / 8.5, 128256x2048
// 24.3 / 29.9 / 43.3 / 44.5: sharing the activation loads between rows does not pay here, one row per item it is.
#define GEMV_ROWS 1
#endif
#ifndef UPR
#define UPR 1 // units staged per barrier round of the tile kernel (K / 32 a multiple of it)
#endif

float h2f(uint h) {
    const ushort t = (ushort)h;
    return vload_half(0, (const half*)&t);
}

// dl = d * sc_j and ml = dmin * m_j of unit u of a weight row whose scale blocks start at ws: ggml's
// get_scale_min_k4 without a branch, as in ameva_qdecode.glsl
float2 unit_scale(__global const uint4* ws, int u) {
    const uint4 sw = ws[u >> 3];
    const uint j = (uint)u & 7u, k = 8u * (j & 3u);
    const uint b0 = (sw.y >> k) & 0xFFu, b1 = (sw.z >> k) & 0xFFu, b2 = (sw.w >> k) & 0xFFu;
    const uint sc = j < 4u ? (b0 & 63u) : ((b2 & 0xFu) | ((b0 >> 6) << 4));
    const uint mn = j < 4u ? (b1 & 63u) : ((b2 >> 4) | ((b1 >> 6) << 4));
    return (float2)(h2f(sw.x & 0xFFFFu) * (float)sc, h2f(sw.x >> 16) * (float)mn);
}

// the 32 products of one unit: unpacked weights (lo: weights 0..15, hi: 16..31, one byte each) against 32 int8
int dot32(uint4 lo, uint4 hi, int4 a0, int4 a1) {
    int s = arm_dot_acc(as_char4(lo.x), as_char4(a0.x), 0);
    s = arm_dot_acc(as_char4(lo.y), as_char4(a0.y), s);
    s = arm_dot_acc(as_char4(lo.z), as_char4(a0.z), s);
    s = arm_dot_acc(as_char4(lo.w), as_char4(a0.w), s);
    s = arm_dot_acc(as_char4(hi.x), as_char4(a1.x), s);
    s = arm_dot_acc(as_char4(hi.y), as_char4(a1.y), s);
    s = arm_dot_acc(as_char4(hi.z), as_char4(a1.z), s);
    s = arm_dot_acc(as_char4(hi.w), as_char4(a1.w), s);
    return s;
}

// Activations: one work-item per (row, unit). aq: 32 int8 per unit (two int4), ad: d, asum: sum of q (as float)
__kernel void quant_a8(__global const float4* a, __global int4* aq, __global float* ad, __global float* asum, int M,
                       int K) {
    const int U = K >> 5;
    const int g = get_global_id(0);
    if (g >= M * U) return;
    float4 x[8];
    float4 m4 = (float4)(0.0f);
    for (int i = 0; i < 8; ++i) {
        x[i] = a[(size_t)g * 8 + i];
        m4 = fmax(m4, fabs(x[i]));
    }
    const float amax = fmax(fmax(m4.x, m4.y), fmax(m4.z, m4.w));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    int4 q[8];
    int4 s4 = (int4)(0);
    for (int i = 0; i < 8; ++i) {
        q[i] = convert_int4(clamp(rint(x[i] * id), -127.0f, 127.0f));
        s4 += q[i];
    }
    // four int8 per 32-bit word, little endian
    int4 p[2];
    for (int h = 0; h < 2; ++h) {
        const int4 b0 = (int4)(q[4 * h].x, q[4 * h + 1].x, q[4 * h + 2].x, q[4 * h + 3].x) & 0xFF;
        const int4 b1 = (int4)(q[4 * h].y, q[4 * h + 1].y, q[4 * h + 2].y, q[4 * h + 3].y) & 0xFF;
        const int4 b2 = (int4)(q[4 * h].z, q[4 * h + 1].z, q[4 * h + 2].z, q[4 * h + 3].z) & 0xFF;
        const int4 b3 = (int4)(q[4 * h].w, q[4 * h + 1].w, q[4 * h + 2].w, q[4 * h + 3].w);
        p[h] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    aq[(size_t)g * 2] = p[0];
    aq[(size_t)g * 2 + 1] = p[1];
    ad[g] = d;
    asum[g] = (float)(s4.x + s4.y + s4.z + s4.w);
}

// Token generation: one work-item per GEMV_ROWS weight rows, activation row 0. The rows of a work-item share the
// loads of the activations.
__kernel void gemv_q4k_i8(__global const uint4* wq, __global const uint4* ws, __global const int4* aq,
                          __global const float* ad, __global const float* asum, __global float* dst, int N, int K) {
    const int n0 = get_global_id(0) * GEMV_ROWS;
    if (n0 >= N) return;
    const int U = K >> 5, SBR = U >> 3;
    float acc[GEMV_ROWS];
    for (int r = 0; r < GEMV_ROWS; ++r) acc[r] = 0.0f;
    for (int u = 0; u < U; ++u) {
        const int4 a0 = aq[2 * u], a1 = aq[2 * u + 1];
        const float da = ad[u], sa = asum[u];
        for (int r = 0; r < GEMV_ROWS; ++r) {
            const int n = min(n0 + r, N - 1);
            const uint4 qw = wq[(size_t)n * U + u];
            const float2 sm = unit_scale(ws + (size_t)n * SBR, u);
            const int s = dot32(qw & (uint4)(0x0F0F0F0Fu), (qw >> 4) & (uint4)(0x0F0F0F0Fu), a0, a1);
            acc[r] += da * (sm.x * (float)s - sm.y * sa);
        }
    }
    for (int r = 0; r < GEMV_ROWS; ++r) {
        if (n0 + r < N) dst[n0 + r] = acc[r];
    }
}

// The same product with float weights and float activations (what ameva_qmv computes in Vulkan)
__kernel void gemv_q4k_f32(__global const uint4* wq, __global const uint4* ws, __global const float4* a,
                           __global float* dst, int N, int K) {
    const int n0 = get_global_id(0) * GEMV_ROWS;
    if (n0 >= N) return;
    const int U = K >> 5, SBR = U >> 3;
    float acc[GEMV_ROWS];
    for (int r = 0; r < GEMV_ROWS; ++r) acc[r] = 0.0f;
    for (int u = 0; u < U; ++u) {
        float4 x[8];
        for (int i = 0; i < 8; ++i) x[i] = a[8 * u + i];
        for (int r = 0; r < GEMV_ROWS; ++r) {
            const int n = min(n0 + r, N - 1);
            const uint4 qw = wq[(size_t)n * U + u];
            const float2 sm = unit_scale(ws + (size_t)n * SBR, u);
            const uint4 lo = qw & (uint4)(0x0F0F0F0Fu), hi = (qw >> 4) & (uint4)(0x0F0F0F0Fu);
            float4 p = (float4)(0.0f);
            // word i of lo holds weights 4i .. 4i + 3, word i of hi weights 16 + 4i ..
            p += (convert_float4(as_uchar4(lo.x)) * sm.x - sm.y) * x[0];
            p += (convert_float4(as_uchar4(lo.y)) * sm.x - sm.y) * x[1];
            p += (convert_float4(as_uchar4(lo.z)) * sm.x - sm.y) * x[2];
            p += (convert_float4(as_uchar4(lo.w)) * sm.x - sm.y) * x[3];
            p += (convert_float4(as_uchar4(hi.x)) * sm.x - sm.y) * x[4];
            p += (convert_float4(as_uchar4(hi.y)) * sm.x - sm.y) * x[5];
            p += (convert_float4(as_uchar4(hi.z)) * sm.x - sm.y) * x[6];
            p += (convert_float4(as_uchar4(hi.w)) * sm.x - sm.y) * x[7];
            acc[r] += p.x + p.y + p.z + p.w;
        }
    }
    for (int r = 0; r < GEMV_ROWS; ++r) {
        if (n0 + r < N) dst[n0 + r] = acc[r];
    }
}

// Prompt processing: a workgroup of 128 work-items owns a tile of 64 weight rows x 32 activation rows and walks K
// UPR units at a time (ameva_mmq's work split, which stages one unit per round):
//   1. items 0..63 stage UPR units of one weight row each (unpacked: lo and hi, then dl, ml), items 64..95 the units
//      of one activation row each (32 int8, d, d * sum);
//   2. every item accumulates its 4 x 4 block: 16 dot products of 32 values per unit.
// Two barriers per round, so UPR divides the barriers per multiply-accumulate. N a multiple of 64, M a multiple of 32
// (prototype). Global size (N / 64 * 128, M / 32), local size (128, 1). Local memory per unit of UPR: lw 2 KiB,
// lws 512 B, la 1 KiB, las 256 B.
__kernel void gemm_q4k_i8(__global const uint4* wq, __global const uint4* ws, __global const int4* aq,
                          __global const float* ad, __global const float* asum, __global float* dst, int M, int N,
                          int K, __local uint4* lw, __local float2* lws, __local int4* la, __local float2* las) {
    const int t = get_local_id(0);
    const int n0 = get_group_id(0) * 64, m0 = get_group_id(1) * 32;
    const int U = K >> 5, SBR = U >> 3;
    const int tn = t & 15, tm = t >> 4;
    float4 acc[4]; // acc[j][i]: activation row 4 tm + j, weight row 4 tn + i
    for (int j = 0; j < 4; ++j) acc[j] = (float4)(0.0f);
    for (int u0 = 0; u0 < U; u0 += UPR) {
        if (t < 64) {
            const int n = n0 + t;
            for (int p = 0; p < UPR; ++p) {
                const uint4 qw = wq[(size_t)n * U + u0 + p];
                lw[2 * (p * 64 + t)] = qw & (uint4)(0x0F0F0F0Fu);
                lw[2 * (p * 64 + t) + 1] = (qw >> 4) & (uint4)(0x0F0F0F0Fu);
                lws[p * 64 + t] = unit_scale(ws + (size_t)n * SBR, u0 + p);
            }
        } else if (t < 96) {
            const int r = t - 64;
            for (int p = 0; p < UPR; ++p) {
                const size_t g = (size_t)(m0 + r) * U + u0 + p;
                la[2 * (p * 32 + r)] = aq[2 * g];
                la[2 * (p * 32 + r) + 1] = aq[2 * g + 1];
                las[p * 32 + r] = (float2)(ad[g], ad[g] * asum[g]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int p = 0; p < UPR; ++p) {
            const int wb = p * 64 + 4 * tn, ab = p * 32 + 4 * tm;
            uint4 lo[4], hi[4];
            for (int i = 0; i < 4; ++i) {
                lo[i] = lw[2 * (wb + i)];
                hi[i] = lw[2 * (wb + i) + 1];
            }
            const float4 dl = (float4)(lws[wb].x, lws[wb + 1].x, lws[wb + 2].x, lws[wb + 3].x);
            const float4 ml = (float4)(lws[wb].y, lws[wb + 1].y, lws[wb + 2].y, lws[wb + 3].y);
            for (int j = 0; j < 4; ++j) {
                const int4 a0 = la[2 * (ab + j)], a1 = la[2 * (ab + j) + 1];
                const float2 ds = las[ab + j];
                const int4 s = (int4)(dot32(lo[0], hi[0], a0, a1), dot32(lo[1], hi[1], a0, a1),
                                      dot32(lo[2], hi[2], a0, a1), dot32(lo[3], hi[3], a0, a1));
                acc[j] += ds.x * (dl * convert_float4(s)) - ml * ds.y;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = 0; j < 4; ++j) {
        const size_t o = (size_t)(m0 + 4 * tm + j) * N + n0 + 4 * tn;
        dst[o] = acc[j].x;
        dst[o + 1] = acc[j].y;
        dst[o + 2] = acc[j].z;
        dst[o + 3] = acc[j].w;
    }
}
