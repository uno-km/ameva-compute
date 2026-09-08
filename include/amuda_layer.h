/**
 * @file amuda_layer.h
 * @brief BitNet 1.58b Single Transformer Block Forward Engine
 *
 * Implements:
 * 1. attn_norm (RMSNorm)
 * 2. QKV Projections (I2_S GEMV)
 * 3. RoPE Positional Embeddings
 * 4. GQA Attention with KV-Cache
 * 5. attn_sub_norm (RMSNorm)
 * 6. attn_output (I2_S GEMV)
 * 7. Residual Add 1
 * 8. ffn_norm (RMSNorm)
 * 9. ffn_gate & ffn_up (I2_S GEMV)
 * 10. SwiGLU Activation
 * 11. ffn_sub_norm (RMSNorm)
 * 12. ffn_down (I2_S GEMV)
 * 13. Residual Add 2
 */

#ifndef AMUDA_LAYER_H
#define AMUDA_LAYER_H

#include "amuda.h"
#include "amuda_ops.h"
#include "amuda_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int layer_idx;

    // Attention Tensors
    GgufTensor* attn_norm;
    GgufTensor* attn_q;
    GgufTensor* attn_k;
    GgufTensor* attn_v;
    GgufTensor* attn_sub_norm;
    GgufTensor* attn_output;

    float scale_q;
    float scale_k;
    float scale_v;
    float scale_output;

    // FFN Tensors
    GgufTensor* ffn_norm;
    GgufTensor* ffn_gate;
    GgufTensor* ffn_up;
    GgufTensor* ffn_sub_norm;
    GgufTensor* ffn_down;

    float scale_gate;
    float scale_up;
    float scale_down;

    // Bound GPU Buffers for Weights
    AcBuffer* buf_w_q;
    AcBuffer* buf_w_k;
    AcBuffer* buf_w_v;
    AcBuffer* buf_w_output;

    AcBuffer* buf_w_gate;
    AcBuffer* buf_w_up;
    AcBuffer* buf_w_down;

    AcBuffer* buf_w_attn_norm;
    AcBuffer* buf_w_attn_sub_norm;
    AcBuffer* buf_w_ffn_norm;
    AcBuffer* buf_w_ffn_sub_norm;
} BitNetLayer;

typedef struct {
    // Intermediate activation buffers
    AcBuffer* buf_norm;      // [hidden_dim] (2560)
    AcBuffer* buf_q;         // [num_heads * head_dim] (2560)
    AcBuffer* buf_k;         // [num_kv_heads * head_dim] (640)
    AcBuffer* buf_v;         // [num_kv_heads * head_dim] (640)
    AcBuffer* buf_attn_out;  // [hidden_dim] (2560)
    AcBuffer* buf_attn_sub;  // [hidden_dim] (2560)
    AcBuffer* buf_ffn_inp;   // [hidden_dim] (2560)
    AcBuffer* buf_gate;      // [ffn_dim] (6912)
    AcBuffer* buf_up;        // [ffn_dim] (6912)
    AcBuffer* buf_swiglu;    // [ffn_dim] (6912)
    AcBuffer* buf_ffn_sub;   // [ffn_dim] (6912)
    AcBuffer* buf_ffn_down;  // [hidden_dim] (2560)
} BitNetScratch;

/**
 * @brief Initialize a BitNet layer by binding GGUF tensors and allocating GPU weight buffers
 */
AcResult amuda_layer_init(
    AcContext* ctx,
    GgufModel* model,
    int layer_idx,
    BitNetLayer** out_layer
);

/**
 * @brief Free a BitNet layer and its GPU buffers
 */
void amuda_layer_free(AcContext* ctx, BitNetLayer* layer);

/**
 * @brief Allocate scratch buffers shared across layers
 */
AcResult amuda_scratch_alloc(
    AcContext* ctx,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    BitNetScratch** out_scratch
);

/**
 * @brief Free scratch buffers
 */
void amuda_scratch_free(AcContext* ctx, BitNetScratch* scratch);

/**
 * @brief Execute forward pass for a single BitNet Transformer Block
 *
 * @param ctx       AMUDA Context
 * @param layer     Initialized layer weights and scales
 * @param scratch   Working scratch buffers
 * @param buf_x_in  Input hidden state [hidden_dim]
 * @param buf_x_out Output hidden state [hidden_dim]
 * @param pos       Current token sequence position (e.g. 0, 1, 2...)
 * @param eps       RMSNorm epsilon (e.g. 1e-5)
 */
AcResult amuda_layer_forward(
    AcContext* ctx,
    BitNetLayer* layer,
    BitNetScratch* scratch,
    AcBuffer* buf_x_in,
    AcBuffer* buf_x_out,
    int pos,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_LAYER_H */
