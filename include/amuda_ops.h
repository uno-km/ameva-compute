/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file amuda_ops.h
 * @brief High-Performance Deep Learning and BLAS Operations for AMUDA
 *
 * Provides accelerated GEMM, BitNet 1.58b GEMV, RMSNorm, SwiGLU, and RoPE
 * with zero-copy UMA memory coherency and hardware quirk protection.
 */

#ifndef AMUDA_OPS_H
#define AMUDA_OPS_H

#include "amuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 2D Tiled General Matrix Multiplication: C = alpha * (A @ B) + beta * C
 *
 * @param ctx   Active AMUDA context
 * @param M     Rows of matrix A and C
 * @param N     Columns of matrix B and C
 * @param K     Columns of matrix A, Rows of matrix B
 * @param alpha Scalar multiplier for A @ B
 * @param A     Buffer containing M x K row-major matrix
 * @param B     Buffer containing K x N row-major matrix
 * @param beta  Scalar multiplier for C (pass 0.0f to overwrite C)
 * @param C     Buffer for M x N output matrix
 * @param out_event Optional event handle for microsecond profiling
 */
AcResult ac_ops_gemm_fp32(
    AcContext* ctx,
    int M, int N, int K,
    float alpha,
    AcBuffer* A,
    AcBuffer* B,
    float beta,
    AcBuffer* C,
    AcEvent** out_event
);

/**
 * @brief BitNet 1.58-bit (Ternary {-1, 0, +1}) Packed GEMV: y = scales * (W @ x)
 *
 * @param ctx       Active AMUDA context
 * @param M         Number of output rows (features)
 * @param K         Number of input features (must be multiple of 4)
 * @param x         Buffer containing K elements (FP32 activation vector)
 * @param W_packed  Buffer containing M x (K/4) bytes (2-bit packed ternary weights)
 * @param scales    Buffer containing M elements (FP32 per-row scaling factors)
 * @param y         Buffer for M elements (FP32 output vector)
 * @param out_event Optional profiling event
 */
AcResult ac_ops_bitnet_gemv_i2(
    AcContext* ctx,
    int M, int K,
    AcBuffer* x,
    AcBuffer* W_packed,
    AcBuffer* scales,
    AcBuffer* y,
    AcEvent** out_event
);

/**
 * @brief Official BitNet I2_S GEMV: y = tensor_scale * (W_i2_s * x)
 * Exact 32-stride 4-way interleaved SIMT dequantization matching llama.cpp.
 */
AcResult ac_ops_bitnet_gemv_i2_s_exact(
    AcContext* ctx,
    int M, int K,
    AcBuffer* x,
    AcBuffer* W_packed,
    float tensor_scale,
    AcBuffer* y,
    AcEvent** out_event
);

/**
 * @brief Transformer RMSNorm: y = (x / RMS(x)) * weight
 *
 * @param ctx     Active AMUDA context
 * @param num_rows Number of tokens / rows
 * @param hidden_dim Dimension of hidden features
 * @param eps     Epsilon constant (e.g. 1e-6)
 * @param x       Input buffer [num_rows x hidden_dim]
 * @param weight  Scale weight buffer [hidden_dim]
 * @param y       Output buffer [num_rows x hidden_dim]
 * @param out_event Optional profiling event
 */
AcResult ac_ops_rmsnorm(
    AcContext* ctx,
    int num_rows,
    int hidden_dim,
    float eps,
    AcBuffer* x,
    AcBuffer* weight,
    AcBuffer* y,
    AcEvent** out_event
);

/**
 * @brief Fused SwiGLU Activation: y = Swish(gate) * up
 *
 * @param ctx       Active AMUDA context
 * @param num_elements Total elements (num_rows * intermediate_dim)
 * @param gate      Input gate buffer
 * @param up        Input up projection buffer
 * @param y         Output buffer
 * @param out_event Optional profiling event
 */
AcResult ac_ops_swiglu(
    AcContext* ctx,
    int num_elements,
    AcBuffer* gate,
    AcBuffer* up,
    AcBuffer* y,
    AcEvent** out_event
);

/**
 * @brief Rotary Position Embedding (RoPE) In-Place:
 *
 * Rotates adjacent pairs in Q and K tensors according to frequency tables.
 *
 * @param ctx       Active AMUDA context
 * @param seq_len   Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head (must be even)
 * @param qk        Buffer [seq_len x num_heads x head_dim] modified in-place
 * @param cos_sin   Buffer [seq_len x head_dim] with precomputed [cos, sin]
 * @param out_event Optional profiling event
 */
AcResult ac_ops_rope(
    AcContext* ctx,
    int seq_len,
    int num_heads,
    int head_dim,
    AcBuffer* qk,
    AcBuffer* cos_sin,
    AcEvent** out_event
);

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_OPS_H */
