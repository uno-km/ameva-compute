// Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
//
// OpenCL C form of ameva_mmq.comp -DTYPE_Q5_0, written by hand, used instead of the translation of the GLSL
// (scripts/build_ameva_kernels.py, OpenCL tables only; the SPIR-V is still compiled from the GLSL). The algorithm and
// the arithmetic expressions are those of the GLSL; the structure differs where the Mali OpenCL compiler
// loses: no helper functions taking the context by pointer, no private arrays, plain scalar stores into
// shared memory, one weight-offset tile and one activation-sum tile fewer (q5_0/q8_0 carry a scale per 32
// weights, so they have no block-integer form). Interface of every ameva_mmq kernel: bindings 0..5, the push
// struct below (11 words), local size 128, grid (ceil(N/64), ceil(M/32)). Measured: docs/design/opencl-prompt-path.md.
#define gl_WorkGroupSize ((uint3)(128u, 1u, 1u))
typedef struct { uint M; uint K; uint N; uint q_off; uint h_off; uint s_off; uint aq_off; uint ad_off; uint d_off; uint sd; uint qt; } ameva_pc_t;
typedef struct { uint bytes[6]; } ameva_ranges_t;

float q50_h2f(uint h) {
    const uint s = (h & 0x8000u) << 16u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 0u) {
        const float v = (float)m * 5.9604644775390625e-8f;
        return s != 0u ? -v : v;
    }
    if (e == 31u) return as_float(s | 0x7F800000u | (m << 13u));
    return as_float(s | ((e + 112u) << 23u) | (m << 13u));
}
#define SPREAD4(x) ((((x) & 1u) << 4u) | (((x) & 2u) << 11u) | (((x) & 4u) << 18u) | (((x) & 8u) << 25u))
#define DOT(a, b) arm_dot(as_char4(a), as_char4(b))

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void ameva_main(__global const uint4* qv, __global const uint* hv, __global const uint* sw, __global const uint4* aq,
                __global const uint2* ad, __global float* dst, ameva_pc_t pc, ameva_ranges_t rg) {
    __local uint4 s_wq[128];  // uint index w * 64 + n: word w of weight row n
    __local float4 s_wsc[16]; // index n
    __local uint4 s_aq[64];   // uint index w * 32 + m: word w of activation row m
    __local float4 s_ad[8];   // index m
    __local float4 s_as[8];   // index m: sum of both halves
    const uint s_last = (uint)((int)(rg.bytes[2] / 4u)) - 1u;
    const uint tid = get_local_id(0);
    const uint n0 = get_group_id(0) * 64u;
    const uint m0 = get_group_id(1) * 32u;
    const uint U = pc.K >> 5u;
    const uint s_row = (pc.K / 32u) * 2u;
    const uint t = tid & 63u;
    const uint drow = min(n0 + t, pc.N - 1u);
    const uint qrow = min(m0 + (t >> 1u), pc.M - 1u);
    const uint drow_a = min(m0 + (t & 31u), pc.M - 1u);
    const uint tn = tid & 15u, tm = tid >> 4u;
    __local uint* wq = (__local uint*)s_wq;
    __local uint* aqs = (__local uint*)s_aq;
    float4 acc0 = (float4)(0.0f), acc1 = (float4)(0.0f), acc2 = (float4)(0.0f), acc3 = (float4)(0.0f);
    for (uint u = 0u; u < U; ++u) {
        if (tid < 64u) {
            const uint4 q = qv[pc.q_off + drow * U + u];
            const uint h = hv[pc.h_off + drow * U + u];
            const uint a = pc.s_off + drow * s_row + u * 2u;
            const float d = q50_h2f((sw[min(a >> 2u, s_last)] >> ((a & 2u) * 8u)) & 0xFFFFu);
            wq[0u * 64u + t] = (q.x & 0x0F0F0F0Fu) | SPREAD4(h);
            wq[1u * 64u + t] = (q.y & 0x0F0F0F0Fu) | SPREAD4(h >> 4u);
            wq[2u * 64u + t] = (q.z & 0x0F0F0F0Fu) | SPREAD4(h >> 8u);
            wq[3u * 64u + t] = (q.w & 0x0F0F0F0Fu) | SPREAD4(h >> 12u);
            wq[4u * 64u + t] = ((q.x >> 4u) & 0x0F0F0F0Fu) | SPREAD4(h >> 16u);
            wq[5u * 64u + t] = ((q.y >> 4u) & 0x0F0F0F0Fu) | SPREAD4(h >> 20u);
            wq[6u * 64u + t] = ((q.z >> 4u) & 0x0F0F0F0Fu) | SPREAD4(h >> 24u);
            wq[7u * 64u + t] = ((q.w >> 4u) & 0x0F0F0F0Fu) | SPREAD4(h >> 28u);
            ((__local float*)s_wsc)[t] = d;
        } else {
            const uint r = t >> 1u, hh = t & 1u;
            const uint4 x = aq[pc.aq_off + (qrow * U + u) * 2u + hh];
            aqs[(4u * hh + 0u) * 32u + r] = x.x;
            aqs[(4u * hh + 1u) * 32u + r] = x.y;
            aqs[(4u * hh + 2u) * 32u + r] = x.z;
            aqs[(4u * hh + 3u) * 32u + r] = x.w;
            if (t < 32u) {
                const uint2 e = ad[pc.ad_off + drow_a * U + u];
                const float da = as_float(e.x);
                ((__local float*)s_ad)[t] = da;
                const float p0 = da * (float)(((int)(e.y << 16u)) >> 16); // separate statements: no contraction into an fma
                const float p1 = da * (float)(((int)e.y) >> 16);
                ((__local float*)s_as)[t] = p0 + p1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const float4 da = s_ad[tm];
        int4 i0 = (int4)(0), i1 = (int4)(0), i2 = (int4)(0), i3 = (int4)(0);
        for (uint w = 0u; w < 8u; ++w) {
            const uint4 W = s_wq[w * 16u + tn];
            const uint4 A = s_aq[w * 8u + tm];
            i0 += (int4)(DOT(W.x, A.x), DOT(W.y, A.x), DOT(W.z, A.x), DOT(W.w, A.x));
            i1 += (int4)(DOT(W.x, A.y), DOT(W.y, A.y), DOT(W.z, A.y), DOT(W.w, A.y));
            i2 += (int4)(DOT(W.x, A.z), DOT(W.y, A.z), DOT(W.z, A.z), DOT(W.w, A.z));
            i3 += (int4)(DOT(W.x, A.w), DOT(W.y, A.w), DOT(W.z, A.w), DOT(W.w, A.w));
        }
        const float4 sc = s_wsc[tn], mc = -16.0f * sc;
        const float4 sa = s_as[tm];
        acc0 += da.x * (sc * convert_float4(i0)) + mc * sa.x;
        acc1 += da.y * (sc * convert_float4(i1)) + mc * sa.y;
        acc2 += da.z * (sc * convert_float4(i2)) + mc * sa.z;
        acc3 += da.w * (sc * convert_float4(i3)) + mc * sa.w;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint nb = n0 + 4u * tn;
    const uint mb = m0 + 4u * tm;
    for (uint j = 0u; j < 4u; ++j) {
        const uint m = mb + j;
        if (m < pc.M) {
            const float4 a = j == 0u ? acc0 : (j == 1u ? acc1 : (j == 2u ? acc2 : acc3));
            const uint o = pc.d_off + m * pc.sd + nb;
            if (nb + 0u < pc.N) dst[o + 0u] = a.x;
            if (nb + 1u < pc.N) dst[o + 1u] = a.y;
            if (nb + 2u < pc.N) dst[o + 2u] = a.z;
            if (nb + 3u < pc.N) dst[o + 3u] = a.w;
        }
    }
}
