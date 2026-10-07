/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file amuda_layer.c
 * @brief BitNet 1.58b Transformer Block Implementation
 */

#include "amuda_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static AcResult bind_weight_buffer(AcContext* ctx, GgufTensor* tensor, AcBuffer** out_buf) {
    if (!tensor || !out_buf) return AC_ERR_NULL_POINTER;
    AcResult res = ac_buffer_alloc(ctx, tensor->size_bytes, out_buf);
    if (res != AC_SUCCESS) return res;
    memcpy((*out_buf)->host_ptr, tensor->data_ptr, tensor->size_bytes);
    ac_buffer_sync_to_device(ctx, *out_buf);
    return AC_SUCCESS;
}

AcResult amuda_layer_init(
    AcContext* ctx,
    GgufModel* model,
    int layer_idx,
    BitNetLayer** out_layer
) {
    if (!ctx || !model || !out_layer) return AC_ERR_NULL_POINTER;

    BitNetLayer* layer = (BitNetLayer*)calloc(1, sizeof(BitNetLayer));
    if (!layer) return AC_ERR_OUT_OF_HOST_MEMORY;

    layer->layer_idx = layer_idx;
    char name[128];

    #define BIND_TENSOR(field, suffix) do { \
        snprintf(name, sizeof(name), "blk.%d.%s.weight", layer_idx, suffix); \
        layer->field = ac_gguf_get_tensor(model, name); \
        if (!layer->field) { \
            fprintf(stderr, "[AMUDA-LAYER] Tensor %s not found!\n", name); \
            amuda_layer_free(ctx, layer); \
            return AC_ERR_KERNEL_NOT_FOUND; \
        } \
    } while (0)

    BIND_TENSOR(attn_norm, "attn_norm");
    BIND_TENSOR(attn_q, "attn_q");
    BIND_TENSOR(attn_k, "attn_k");
    BIND_TENSOR(attn_v, "attn_v");
    BIND_TENSOR(attn_sub_norm, "attn_sub_norm");
    BIND_TENSOR(attn_output, "attn_output");

    BIND_TENSOR(ffn_norm, "ffn_norm");
    BIND_TENSOR(ffn_gate, "ffn_gate");
    BIND_TENSOR(ffn_up, "ffn_up");
    BIND_TENSOR(ffn_sub_norm, "ffn_sub_norm");
    BIND_TENSOR(ffn_down, "ffn_down");

    #undef BIND_TENSOR

    // Read official scales from GGUF
    ac_gguf_get_tensor_scale(layer->attn_q, &layer->scale_q);
    ac_gguf_get_tensor_scale(layer->attn_k, &layer->scale_k);
    ac_gguf_get_tensor_scale(layer->attn_v, &layer->scale_v);
    ac_gguf_get_tensor_scale(layer->attn_output, &layer->scale_output);

    ac_gguf_get_tensor_scale(layer->ffn_gate, &layer->scale_gate);
    ac_gguf_get_tensor_scale(layer->ffn_up, &layer->scale_up);
    ac_gguf_get_tensor_scale(layer->ffn_down, &layer->scale_down);

    // Bind weights to GPU memory
    bind_weight_buffer(ctx, layer->attn_norm, &layer->buf_w_attn_norm);
    bind_weight_buffer(ctx, layer->attn_q, &layer->buf_w_q);
    bind_weight_buffer(ctx, layer->attn_k, &layer->buf_w_k);
    bind_weight_buffer(ctx, layer->attn_v, &layer->buf_w_v);
    bind_weight_buffer(ctx, layer->attn_sub_norm, &layer->buf_w_attn_sub_norm);
    bind_weight_buffer(ctx, layer->attn_output, &layer->buf_w_output);

    bind_weight_buffer(ctx, layer->ffn_norm, &layer->buf_w_ffn_norm);
    bind_weight_buffer(ctx, layer->ffn_gate, &layer->buf_w_gate);
    bind_weight_buffer(ctx, layer->ffn_up, &layer->buf_w_up);
    bind_weight_buffer(ctx, layer->ffn_sub_norm, &layer->buf_w_ffn_sub_norm);
    bind_weight_buffer(ctx, layer->ffn_down, &layer->buf_w_down);

    *out_layer = layer;
    return AC_SUCCESS;
}

void amuda_layer_free(AcContext* ctx, BitNetLayer* layer) {
    if (!layer) return;
    if (ctx) {
        if (layer->buf_w_attn_norm) ac_buffer_free(ctx, layer->buf_w_attn_norm);
        if (layer->buf_w_q) ac_buffer_free(ctx, layer->buf_w_q);
        if (layer->buf_w_k) ac_buffer_free(ctx, layer->buf_w_k);
        if (layer->buf_w_v) ac_buffer_free(ctx, layer->buf_w_v);
        if (layer->buf_w_attn_sub_norm) ac_buffer_free(ctx, layer->buf_w_attn_sub_norm);
        if (layer->buf_w_output) ac_buffer_free(ctx, layer->buf_w_output);

        if (layer->buf_w_ffn_norm) ac_buffer_free(ctx, layer->buf_w_ffn_norm);
        if (layer->buf_w_gate) ac_buffer_free(ctx, layer->buf_w_gate);
        if (layer->buf_w_up) ac_buffer_free(ctx, layer->buf_w_up);
        if (layer->buf_w_ffn_sub_norm) ac_buffer_free(ctx, layer->buf_w_ffn_sub_norm);
        if (layer->buf_w_down) ac_buffer_free(ctx, layer->buf_w_down);
    }
    free(layer);
}

AcResult amuda_scratch_alloc(
    AcContext* ctx,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    BitNetScratch** out_scratch
) {
    if (!ctx || !out_scratch) return AC_ERR_NULL_POINTER;

    BitNetScratch* s = (BitNetScratch*)calloc(1, sizeof(BitNetScratch));
    if (!s) return AC_ERR_OUT_OF_HOST_MEMORY;

    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &s->buf_norm);
    ac_buffer_alloc(ctx, num_heads * head_dim * sizeof(float), &s->buf_q);
    ac_buffer_alloc(ctx, num_kv_heads * head_dim * sizeof(float), &s->buf_k);
    ac_buffer_alloc(ctx, num_kv_heads * head_dim * sizeof(float), &s->buf_v);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &s->buf_attn_out);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &s->buf_attn_sub);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &s->buf_ffn_inp);

    ac_buffer_alloc(ctx, ffn_dim * sizeof(float), &s->buf_gate);
    ac_buffer_alloc(ctx, ffn_dim * sizeof(float), &s->buf_up);
    ac_buffer_alloc(ctx, ffn_dim * sizeof(float), &s->buf_swiglu);
    ac_buffer_alloc(ctx, ffn_dim * sizeof(float), &s->buf_ffn_sub);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &s->buf_ffn_down);

    *out_scratch = s;
    return AC_SUCCESS;
}

void amuda_scratch_free(AcContext* ctx, BitNetScratch* s) {
    if (!s) return;
    if (ctx) {
        if (s->buf_norm) ac_buffer_free(ctx, s->buf_norm);
        if (s->buf_q) ac_buffer_free(ctx, s->buf_q);
        if (s->buf_k) ac_buffer_free(ctx, s->buf_k);
        if (s->buf_v) ac_buffer_free(ctx, s->buf_v);
        if (s->buf_attn_out) ac_buffer_free(ctx, s->buf_attn_out);
        if (s->buf_attn_sub) ac_buffer_free(ctx, s->buf_attn_sub);
        if (s->buf_ffn_inp) ac_buffer_free(ctx, s->buf_ffn_inp);

        if (s->buf_gate) ac_buffer_free(ctx, s->buf_gate);
        if (s->buf_up) ac_buffer_free(ctx, s->buf_up);
        if (s->buf_swiglu) ac_buffer_free(ctx, s->buf_swiglu);
        if (s->buf_ffn_sub) ac_buffer_free(ctx, s->buf_ffn_sub);
        if (s->buf_ffn_down) ac_buffer_free(ctx, s->buf_ffn_down);
    }
    free(s);
}

AcResult amuda_layer_forward(
    AcContext* ctx,
    BitNetLayer* layer,
    BitNetScratch* s,
    AcBuffer* buf_x_in,
    AcBuffer* buf_x_out,
    int pos,
    float eps
) {
    (void)pos; // RoPE position
    int hidden_dim = (int)layer->attn_norm->ne[0]; // 2560
    int ffn_dim = (int)layer->ffn_norm->ne[0];    // 6912
    int kv_dim = (int)layer->attn_k->ne[1];       // 640

    // 1. Attention Pre-Norm
    ac_ops_rmsnorm(ctx, 1, hidden_dim, eps, buf_x_in, layer->buf_w_attn_norm, s->buf_norm, NULL);

    // 2. Q, K, V Projections (Exact I2_S GEMV)
    ac_ops_bitnet_gemv_i2_s_exact(ctx, hidden_dim, hidden_dim, s->buf_norm, layer->buf_w_q, layer->scale_q, s->buf_q, NULL);
    ac_ops_bitnet_gemv_i2_s_exact(ctx, kv_dim, hidden_dim, s->buf_norm, layer->buf_w_k, layer->scale_k, s->buf_k, NULL);
    ac_ops_bitnet_gemv_i2_s_exact(ctx, kv_dim, hidden_dim, s->buf_norm, layer->buf_w_v, layer->scale_v, s->buf_v, NULL);

    // 3. Attention Forward (Single Token Greedy: GQA broadcast V to heads)
    // 20 heads / 5 kv heads = 4 queries per kv head.
    ac_buffer_sync_to_host(ctx, s->buf_v);
    ac_buffer_sync_to_host(ctx, s->buf_attn_out);
    float* v_ptr = (float*)s->buf_v->host_ptr;
    float* attn_out_ptr = (float*)s->buf_attn_out->host_ptr;

    int head_dim = 128;
    for (int h = 0; h < 20; ++h) {
        int kv_h = h / 4;
        memcpy(attn_out_ptr + h * head_dim, v_ptr + kv_h * head_dim, head_dim * sizeof(float));
    }
    ac_buffer_sync_to_device(ctx, s->buf_attn_out);

    // 4. Attention Sub-Norm
    ac_ops_rmsnorm(ctx, 1, hidden_dim, eps, s->buf_attn_out, layer->buf_w_attn_sub_norm, s->buf_attn_sub, NULL);

    // 5. Attention Output Projection
    ac_ops_bitnet_gemv_i2_s_exact(ctx, hidden_dim, hidden_dim, s->buf_attn_sub, layer->buf_w_output, layer->scale_output, s->buf_attn_out, NULL);

    // 6. Residual 1: ffn_inp = attn_out + x_in
    ac_buffer_sync_to_host(ctx, s->buf_attn_out);
    ac_buffer_sync_to_host(ctx, buf_x_in);
    ac_buffer_sync_to_host(ctx, s->buf_ffn_inp);
    float* aout = (float*)s->buf_attn_out->host_ptr;
    float* xin = (float*)buf_x_in->host_ptr;
    float* finp = (float*)s->buf_ffn_inp->host_ptr;
    for (int i = 0; i < hidden_dim; ++i) {
        finp[i] = aout[i] + xin[i];
    }
    ac_buffer_sync_to_device(ctx, s->buf_ffn_inp);

    // 7. FFN Pre-Norm
    ac_ops_rmsnorm(ctx, 1, hidden_dim, eps, s->buf_ffn_inp, layer->buf_w_ffn_norm, s->buf_norm, NULL);

    // 8. FFN Gate & Up Projections
    ac_ops_bitnet_gemv_i2_s_exact(ctx, ffn_dim, hidden_dim, s->buf_norm, layer->buf_w_gate, layer->scale_gate, s->buf_gate, NULL);
    ac_ops_bitnet_gemv_i2_s_exact(ctx, ffn_dim, hidden_dim, s->buf_norm, layer->buf_w_up, layer->scale_up, s->buf_up, NULL);

    // 9. SwiGLU Activation
    ac_ops_swiglu(ctx, ffn_dim, s->buf_gate, s->buf_up, s->buf_swiglu, NULL);

    // 10. FFN Sub-Norm
    ac_ops_rmsnorm(ctx, 1, ffn_dim, eps, s->buf_swiglu, layer->buf_w_ffn_sub_norm, s->buf_ffn_sub, NULL);

    // 11. FFN Down Projection
    ac_ops_bitnet_gemv_i2_s_exact(ctx, hidden_dim, ffn_dim, s->buf_ffn_sub, layer->buf_w_down, layer->scale_down, s->buf_ffn_down, NULL);

    // 12. Residual 2: x_out = ffn_down + ffn_inp
    ac_buffer_sync_to_host(ctx, s->buf_ffn_down);
    ac_buffer_sync_to_host(ctx, buf_x_out);
    float* fdown = (float*)s->buf_ffn_down->host_ptr;
    float* xout = (float*)buf_x_out->host_ptr;
    for (int i = 0; i < hidden_dim; ++i) {
        xout[i] = fdown[i] + finp[i];
    }
    ac_buffer_sync_to_device(ctx, buf_x_out);

    return AC_SUCCESS;
}
