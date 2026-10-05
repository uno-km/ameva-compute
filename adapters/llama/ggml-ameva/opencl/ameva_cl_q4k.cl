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
// Further down, from the measurements of docs/design/opencl-mali-int8.md (section 4): gemm_q4k_i8_direct (the prompt
// kernel without local memory), the peak_* kernels (what this GPU does per second: loads are the scarce resource, a
// packed dot product costs a tenth of a load), and the kernels that follow from it, quant_a8b, gemm_q4k_i8_blk and
// gemv_q4k_i8_blk: one activation scale per 256-element block (ggml's Q8_K) and integer accumulation inside the block.
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
// Where unit u of activation row m is in aq / ad / asum. Row-major is the layout of the float activations. ATRANS
// stores them unit-major (unit u of every row side by side), so that a work-item that walks the activation rows of
// one unit reads consecutive memory instead of one stream per row.
#ifdef ATRANS
#define AIDX(m, u, M, U) ((size_t)(u) * (size_t)(M) + (size_t)(m))
#else
#define AIDX(m, u, M, U) ((size_t)(m) * (size_t)(U) + (size_t)(u))
#endif
// The same for the per-block values (d and unit sums) of quant_a8b: block b of activation row m
#ifdef ATRANS
#define ABIDX(m, b, M, B) ((size_t)(b) * (size_t)(M) + (size_t)(m))
#else
#define ABIDX(m, b, M, B) ((size_t)(m) * (size_t)(B) + (size_t)(b))
#endif
#ifndef DRW
#define DRW 4 // weight rows per work-item of gemm_q4k_i8_direct (a divisor of N)
#endif
#ifndef DRA
#define DRA 8 // activation rows per work-item of gemm_q4k_i8_direct (a divisor of M)
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

// One packed dot product of four int8 pairs added to s. DOT_KHR (kernel tuning) takes the OpenCL 3.0 built-in on
// packed 32-bit words (cl_khr_integer_dot_product; the Galaxy A53's driver does not have it) instead of arm_dot_acc.
#ifdef DOT_KHR
#define DOT4(w, a, s) ((s) + dot_4x8packed_ss_int((uint)(w), (uint)(a)))
#else
#define DOT4(w, a, s) arm_dot_acc(as_char4(w), as_char4(a), (s))
#endif

// the 32 products of one unit: unpacked weights (lo: weights 0..15, hi: 16..31, one byte each) against 32 int8
int dot32(uint4 lo, uint4 hi, int4 a0, int4 a1) {
    int s = DOT4(lo.x, a0.x, 0);
    s = DOT4(lo.y, a0.y, s);
    s = DOT4(lo.z, a0.z, s);
    s = DOT4(lo.w, a0.w, s);
    s = DOT4(hi.x, a1.x, s);
    s = DOT4(hi.y, a1.y, s);
    s = DOT4(hi.z, a1.z, s);
    s = DOT4(hi.w, a1.w, s);
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
    const size_t o = AIDX(g / U, g % U, M, U);
    aq[o * 2] = p[0];
    aq[o * 2 + 1] = p[1];
    ad[o] = d;
    asum[o] = (float)(s4.x + s4.y + s4.z + s4.w);
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
                const size_t g = AIDX(m0 + r, u0 + p, M, U);
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
#if defined(ABL_NODOT) // kernel tuning (wrong results): everything but the dot products
                const int4 s = (int4)((int)(lo[0].x ^ (uint)a0.x), (int)(lo[1].y ^ (uint)a0.y),
                                      (int)(lo[2].z ^ (uint)a1.z), (int)(lo[3].w ^ (uint)a1.w)) & 0xFF;
#else
                const int4 s = (int4)(dot32(lo[0], hi[0], a0, a1), dot32(lo[1], hi[1], a0, a1),
                                      dot32(lo[2], hi[2], a0, a1), dot32(lo[3], hi[3], a0, a1));
#endif
#if defined(ABL_NOSCALE) // kernel tuning (wrong results): the sums without the scales
                acc[j] += convert_float4(s);
#else
                acc[j] += ds.x * (dl * convert_float4(s)) - ml * ds.y;
#endif
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

// Prompt processing without local memory: a work-item owns DRW weight rows x DRA activation rows and reads its units
// straight from the buffers: no staging, no barrier. Against the tile kernel it reads every weight unit M / DRA times
// (the tile: M / 32 times) and decodes its scale as often, and it has no synchronization at all. Global size
// (N / DRW, M / DRA), any local size.
__kernel void gemm_q4k_i8_direct(__global const uint4* wq, __global const uint4* ws, __global const int4* aq,
                                 __global const float* ad, __global const float* asum, __global float* dst, int M,
                                 int N, int K) {
    const int n0 = get_global_id(0) * DRW, m0 = get_global_id(1) * DRA;
    if (n0 >= N || m0 >= M) return;
    const int U = K >> 5, SBR = U >> 3;
    float acc[DRA][DRW];
    for (int j = 0; j < DRA; ++j) {
        for (int i = 0; i < DRW; ++i) acc[j][i] = 0.0f;
    }
    for (int u = 0; u < U; ++u) {
        uint4 lo[DRW], hi[DRW];
        float2 sm[DRW];
        for (int i = 0; i < DRW; ++i) {
            const uint4 qw = wq[(size_t)(n0 + i) * U + u];
            lo[i] = qw & (uint4)(0x0F0F0F0Fu);
            hi[i] = (qw >> 4) & (uint4)(0x0F0F0F0Fu);
            sm[i] = unit_scale(ws + (size_t)(n0 + i) * SBR, u);
        }
        // Kernel tuning switches (wrong results; they show what a work-item's time goes to): ABL_NODOT one exclusive
        // or in place of the eight dot products, ABL_NOSCALE the sums without the scales, ABL_NOAD the scales without
        // the two loads of the activation unit's d and sum, ABL_NOACT one load of the activations per unit in place
        // of one per activation row.
        for (int j = 0; j < DRA; ++j) {
#if defined(ABL_NOACT)
            const size_t g = AIDX(m0, u, M, U);
#else
            const size_t g = AIDX(m0 + j, u, M, U);
#endif
            const int4 a0 = aq[2 * g], a1 = aq[2 * g + 1];
#if defined(ABL_NOAD)
            const float da = 0.5f + (float)j, ds = 3.0f;
#else
            const float da = ad[g], ds = da * asum[g];
#endif
            for (int i = 0; i < DRW; ++i) {
#if defined(ABL_NODOT)
                const int s = (int)((lo[i].x ^ (uint)a0.x) + (hi[i].w ^ (uint)a1.w)) & 0xFFFF;
#else
                const int s = dot32(lo[i], hi[i], a0, a1);
#endif
#if defined(ABL_NOSCALE)
                acc[j][i] += (float)s;
#else
                acc[j][i] += da * (sm[i].x * (float)s) - sm[i].y * ds;
#endif
            }
        }
    }
    for (int j = 0; j < DRA; ++j) {
        for (int i = 0; i < DRW; ++i) dst[(size_t)(m0 + j) * N + n0 + i] = acc[j][i];
    }
}

// Hardware limits (ameva_cl_bench with AMEVA_CL_PEAK=1): what the work-items of this GPU do per second when nothing
// else is in the way, so that a kernel's distance from the limit that binds it can be read off. One "rep" is the work
// of one activation unit against one weight unit (32 multiply-accumulates). Every kernel writes one value per item.
//   peak_dot        the eight packed dot products of a unit on registers
//   peak_fma        eight float4 fused multiply-adds (the 32 multiply-accumulates of the f32 kernels)
//   peak_load       two 16-byte loads per rep from a window of `mask + 1` vectors that starts at the item's own offset
//   peak_load_dot   those two loads and the eight dot products with them: a tile kernel's inner loop without scales
__kernel void peak_dot(__global int* out, int reps) {
    const int g = get_global_id(0);
    const uint4 w = (uint4)(0x01020304u, 0x05060708u, 0x090A0B0Cu, 0x0D0E0F01u) + (uint4)((uint)g & 3u);
    const uint4 lo = w & (uint4)(0x0F0F0F0Fu), hi = (w >> 4) & (uint4)(0x0F0F0F0Fu);
    int4 a = (int4)(0x7F01FE02, 0x03FD04FC, 0x05FB06FA, 0x07F908F8) ^ (int4)(g);
    int s = 0;
    for (int r = 0; r < reps; ++r) {
        a += (int4)(0x01010101);
        s += dot32(lo, hi, a, a);
    }
    out[g] = s;
}

__kernel void peak_fma(__global int* out, int reps) {
    const int g = get_global_id(0);
    float4 x = (float4)(1.0f, 0.5f, 0.25f, 0.125f) * (float)(g & 7);
    const float4 y = (float4)(1.0e-3f, 2.0e-3f, 3.0e-3f, 4.0e-3f);
    float4 acc0 = (float4)(0.0f), acc1 = (float4)(0.0f), acc2 = (float4)(0.0f), acc3 = (float4)(0.0f);
    for (int r = 0; r < reps; ++r) {
        x += (float4)(1.0e-6f);
        acc0 = fma(x, y.xxxx, acc0);
        acc1 = fma(x, y.yyyy, acc1);
        acc2 = fma(x, y.zzzz, acc2);
        acc3 = fma(x, y.wwww, acc3);
        acc0 = fma(x.yzwx, y.xxxx, acc0);
        acc1 = fma(x.yzwx, y.yyyy, acc1);
        acc2 = fma(x.yzwx, y.zzzz, acc2);
        acc3 = fma(x.yzwx, y.wwww, acc3);
    }
    const float4 t = acc0 + acc1 + acc2 + acc3;
    out[g] = (int)(t.x + t.y + t.z + t.w);
}

__kernel void peak_load(__global const int4* buf, __global int* out, int reps, int mask, int stride) {
    const int g = get_global_id(0);
    const size_t base = (size_t)g * (size_t)stride;
    int4 x = (int4)(0);
    for (int r = 0; r < reps; ++r) {
        const size_t o = base + (size_t)((2 * r) & mask);
        x ^= buf[o];
        x += buf[o + 1];
    }
    out[g] = x.x + x.y + x.z + x.w;
}

__kernel void peak_load_dot(__global const int4* buf, __global int* out, int reps, int mask, int stride) {
    const int g = get_global_id(0);
    const size_t base = (size_t)g * (size_t)stride;
    const uint4 w = (uint4)(0x01020304u, 0x05060708u, 0x090A0B0Cu, 0x0D0E0F01u) + (uint4)((uint)g & 3u);
    const uint4 lo = w & (uint4)(0x0F0F0F0Fu), hi = (w >> 4) & (uint4)(0x0F0F0F0Fu);
    int s = 0;
    for (int r = 0; r < reps; ++r) {
        const size_t o = base + (size_t)((2 * r) & mask);
        s += dot32(lo, hi, buf[o], buf[o + 1]);
    }
    out[g] = s;
}

// ---- Integer accumulation over a 256-weight block -------------------------------------------------------------------
// What limits these kernels on Mali is the number of loads (peak kernels above: about 23 G 16-byte loads per second on
// Mali-G78 from memory that is in the first-level cache, against 258 G packed dot products per second), and the
// kernels above load four values for every 32 multiply-accumulates: the two halves of the activation unit, its d and
// its sum. With one d per 256-element block of an activation row (ggml's Q8_K, which is what the CPU backend
// multiplies q4_K weights with) the scales leave the inner loop:
//   sum over blocks of d_a * (d_w * sum_u(sc_u * dot(q_w, q_a)) - dmin_w * sum_u(m_u * sum(q_a)))
// with sc_u * dot summed as integers over the 8 units of the block (at most 8 * 63 * 32 * 15 * 127 < 2^25).

// 32 values in -127..127 (q[i]: values 4 i .. 4 i + 3) as 32 int8, four per 32-bit word, little endian
void pack_i8(const int4* q, int4* p) {
    for (int h = 0; h < 2; ++h) {
        const int4 b0 = (int4)(q[4 * h].x, q[4 * h + 1].x, q[4 * h + 2].x, q[4 * h + 3].x) & 0xFF;
        const int4 b1 = (int4)(q[4 * h].y, q[4 * h + 1].y, q[4 * h + 2].y, q[4 * h + 3].y) & 0xFF;
        const int4 b2 = (int4)(q[4 * h].z, q[4 * h + 1].z, q[4 * h + 2].z, q[4 * h + 3].z) & 0xFF;
        const int4 b3 = (int4)(q[4 * h].w, q[4 * h + 1].w, q[4 * h + 2].w, q[4 * h + 3].w);
        p[h] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

// Activations per block: one work-item per (row, block). aq as quant_a8 writes it (two int4 per unit), adb the
// block's d, absum the sums of q of its 8 units (two int4 per block).
__kernel void quant_a8b(__global const float4* a, __global int4* aq, __global float* adb, __global int4* absum, int M,
                        int K) {
    const int U = K >> 5, B = K >> 8;
    const int g = get_global_id(0);
    if (g >= M * B) return;
    const int m = g / B, b = g % B;
    float4 m4 = (float4)(0.0f);
    for (int i = 0; i < 64; ++i) m4 = fmax(m4, fabs(a[(size_t)g * 64 + i]));
    const float amax = fmax(fmax(m4.x, m4.y), fmax(m4.z, m4.w));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    int su[8];
    for (int u = 0; u < 8; ++u) {
        int4 q[8];
        int4 s4 = (int4)(0);
        for (int i = 0; i < 8; ++i) {
            q[i] = convert_int4(clamp(rint(a[(size_t)g * 64 + 8 * u + i] * id), -127.0f, 127.0f));
            s4 += q[i];
        }
        int4 p[2];
        pack_i8(q, p);
        const size_t o = AIDX(m, 8 * b + u, M, U);
        aq[o * 2] = p[0];
        aq[o * 2 + 1] = p[1];
        su[u] = s4.x + s4.y + s4.z + s4.w;
    }
    const size_t ob = ABIDX(m, b, M, B);
    adb[ob] = d;
    absum[ob * 2] = (int4)(su[0], su[1], su[2], su[3]);
    absum[ob * 2 + 1] = (int4)(su[4], su[5], su[6], su[7]);
}

// Prompt processing, no local memory: a work-item owns DRW weight rows x DRA activation rows (gemm_q4k_i8_direct's
// split). Per unit and pair of rows: two loads, eight dot products and one integer multiply-add; the floating-point
// work is per block. Global size (N / DRW, M / DRA).
__kernel void gemm_q4k_i8_blk(__global const uint4* wq, __global const uint4* ws, __global const int4* aq,
                              __global const float* adb, __global const int4* absum, __global float* dst, int M, int N,
                              int K) {
    const int n0 = get_global_id(0) * DRW, m0 = get_global_id(1) * DRA;
    if (n0 >= N || m0 >= M) return;
    const int U = K >> 5, B = K >> 8;
    float acc[DRA][DRW];
    for (int j = 0; j < DRA; ++j) {
        for (int i = 0; i < DRW; ++i) acc[j][i] = 0.0f;
    }
    for (int b = 0; b < B; ++b) {
        // the block's scales per weight row: sc and m of its 8 units (ggml's get_scale_min_k4), d and dmin
        int4 sc0[DRW], sc1[DRW], mn0[DRW], mn1[DRW];
        float dw[DRW], dm[DRW];
        for (int i = 0; i < DRW; ++i) {
            const uint4 sw = ws[(size_t)(n0 + i) * B + b];
            const uchar4 y = as_uchar4(sw.y), z = as_uchar4(sw.z), w = as_uchar4(sw.w);
            sc0[i] = convert_int4(y & (uchar4)(63));
            mn0[i] = convert_int4(z & (uchar4)(63));
            sc1[i] = convert_int4((w & (uchar4)(15)) | ((y >> (uchar4)(6)) << (uchar4)(4)));
            mn1[i] = convert_int4((w >> (uchar4)(4)) | ((z >> (uchar4)(6)) << (uchar4)(4)));
            dw[i] = h2f(sw.x & 0xFFFFu);
            dm[i] = h2f(sw.x >> 16);
        }
        int isum[DRA][DRW];
        for (int j = 0; j < DRA; ++j) {
            for (int i = 0; i < DRW; ++i) isum[j][i] = 0;
        }
        for (int u8 = 0; u8 < 8; ++u8) {
            const int u = 8 * b + u8;
            uint4 lo[DRW], hi[DRW];
            int sc[DRW];
            for (int i = 0; i < DRW; ++i) {
                const uint4 qw = wq[(size_t)(n0 + i) * U + u];
                lo[i] = qw & (uint4)(0x0F0F0F0Fu);
                hi[i] = (qw >> 4) & (uint4)(0x0F0F0F0Fu);
                const int4 s4 = u8 < 4 ? sc0[i] : sc1[i];
                const int e = u8 & 3;
                sc[i] = e == 0 ? s4.x : (e == 1 ? s4.y : (e == 2 ? s4.z : s4.w));
            }
            for (int j = 0; j < DRA; ++j) {
                const size_t g = AIDX(m0 + j, u, M, U);
                const int4 a0 = aq[2 * g], a1 = aq[2 * g + 1];
                for (int i = 0; i < DRW; ++i) isum[j][i] += sc[i] * dot32(lo[i], hi[i], a0, a1);
            }
        }
        for (int j = 0; j < DRA; ++j) {
            const size_t gb = ABIDX(m0 + j, b, M, B);
            const float da = adb[gb];
            const int4 s0 = absum[2 * gb], s1 = absum[2 * gb + 1];
            for (int i = 0; i < DRW; ++i) {
                const int4 t = mn0[i] * s0 + mn1[i] * s1;
                acc[j][i] += da * (dw[i] * (float)isum[j][i] - dm[i] * (float)(t.x + t.y + t.z + t.w));
            }
        }
    }
    for (int j = 0; j < DRA; ++j) {
        for (int i = 0; i < DRW; ++i) dst[(size_t)(m0 + j) * N + n0 + i] = acc[j][i];
    }
}

// Token generation with quant_a8b's activations: one work-item per weight row, activation row 0. Per unit three loads
// (the weights and the two halves of the activation unit), eight dot products and one integer multiply-add; the
// scales, d and the sums are per block.
__kernel void gemv_q4k_i8_blk(__global const uint4* wq, __global const uint4* ws, __global const int4* aq,
                              __global const float* adb, __global const int4* absum, __global float* dst, int N, int K) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int U = K >> 5, B = K >> 8;
    const uint4 nib = (uint4)(0x0F0F0F0Fu);
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        const uint4 sw = ws[(size_t)n * B + b];
        const uchar4 y = as_uchar4(sw.y), z = as_uchar4(sw.z), w = as_uchar4(sw.w);
        const int4 sc0 = convert_int4(y & (uchar4)(63));
        const int4 mn0 = convert_int4(z & (uchar4)(63));
        const int4 sc1 = convert_int4((w & (uchar4)(15)) | ((y >> (uchar4)(6)) << (uchar4)(4)));
        const int4 mn1 = convert_int4((w >> (uchar4)(4)) | ((z >> (uchar4)(6)) << (uchar4)(4)));
        const __global uint4* q = wq + (size_t)n * U + 8 * b;
        const __global int4* a = aq + 16 * b;
        int isum = sc0.x * dot32(q[0] & nib, (q[0] >> 4) & nib, a[0], a[1]);
        isum += sc0.y * dot32(q[1] & nib, (q[1] >> 4) & nib, a[2], a[3]);
        isum += sc0.z * dot32(q[2] & nib, (q[2] >> 4) & nib, a[4], a[5]);
        isum += sc0.w * dot32(q[3] & nib, (q[3] >> 4) & nib, a[6], a[7]);
        isum += sc1.x * dot32(q[4] & nib, (q[4] >> 4) & nib, a[8], a[9]);
        isum += sc1.y * dot32(q[5] & nib, (q[5] >> 4) & nib, a[10], a[11]);
        isum += sc1.z * dot32(q[6] & nib, (q[6] >> 4) & nib, a[12], a[13]);
        isum += sc1.w * dot32(q[7] & nib, (q[7] >> 4) & nib, a[14], a[15]);
        const int4 t = mn0 * absum[2 * b] + mn1 * absum[2 * b + 1];
        acc += adb[b] * (h2f(sw.x & 0xFFFFu) * (float)isum - h2f(sw.x >> 16) * (float)(t.x + t.y + t.z + t.w));
    }
    dst[n] = acc;
}
