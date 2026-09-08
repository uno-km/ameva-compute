/**
 * @file transformer_ops.cl
 * @brief Accelerated Transformer Activation and Normalization Primitives
 *
 * Implements:
 * 1. RMSNorm (Root Mean Square Normalization with Scale Weight)
 * 2. SwiGLU (Fused SiLU-Gated Linear Unit)
 * 3. RoPE (In-Place Rotary Position Embedding for Attention Heads)
 */

/* ========================================================================= */
/* 1. Transformer RMSNorm Kernel                                             */
/* ========================================================================= */
__kernel void rmsnorm_forward(
    const int num_rows,
    const int hidden_dim,
    const float eps,
    const __global float* x,
    const __global float* weight,
    __global float* y
) {
    int row = get_global_id(0); // 1 thread per token/row
    if (row >= num_rows) return;

    int offset = row * hidden_dim;

    // 1. Calculate sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float val = x[offset + i];
        sum_sq += val * val;
    }

    // 2. Inverse Root Mean Square
    float inv_rms = rsqrt((sum_sq / (float)hidden_dim) + eps);

    // 3. Normalize and scale
    for (int i = 0; i < hidden_dim; ++i) {
        y[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}

/* ========================================================================= */
/* 2. Fused SwiGLU Kernel: y = Swish(gate) * up                              */
/* ========================================================================= */
__kernel void swiglu_forward(
    const int num_elements,
    const __global float* gate,
    const __global float* up,
    __global float* y
) {
    int idx = get_global_id(0);
    if (idx >= num_elements) return;

    float g = gate[idx];
    float silu = g / (1.0f + exp(-g));
    y[idx] = silu * up[idx];
}

/* ========================================================================= */
/* 3. Rotary Position Embedding (RoPE) In-Place Kernel                       */
/* ========================================================================= */
__kernel void rope_forward_inplace(
    const int total_pairs, // seq_len * num_heads * (head_dim / 2)
    const int num_heads,
    const int half_dim,
    __global float* qk,
    const __global float* cos_sin // [seq_len, half_dim * 2] where [cos, sin]
) {
    int pair_idx = get_global_id(0);
    if (pair_idx >= total_pairs) return;

    // Deconstruct pair coordinate
    int d = pair_idx % half_dim;
    int temp = pair_idx / half_dim;
    int h = temp % num_heads;
    int s = temp / num_heads;

    int head_dim = half_dim * 2;
    int qk_idx0 = s * (num_heads * head_dim) + h * head_dim + (d * 2);
    int qk_idx1 = qk_idx0 + 1;

    int cs_idx_cos = s * head_dim + d;
    int cs_idx_sin = s * head_dim + half_dim + d;

    float cos_val = cos_sin[cs_idx_cos];
    float sin_val = cos_sin[cs_idx_sin];

    float x0 = qk[qk_idx0];
    float x1 = qk[qk_idx1];

    // Standard 2D rotation
    qk[qk_idx0] = x0 * cos_val - x1 * sin_val;
    qk[qk_idx1] = x0 * sin_val + x1 * cos_val;
}
