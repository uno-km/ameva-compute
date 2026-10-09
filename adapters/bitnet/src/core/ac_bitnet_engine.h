/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#pragma once
// BitNet b1.58 token execution on the AMUDA runtime (ac_runtime, Vulkan backend). Replaces the raw-Vulkan
// VulkanBitNetEngine: no instance, device, pipeline, descriptor or command-buffer code of its own; every buffer,
// kernel (SPIR-V pinned by SHA-256, bitnet_kernels_spv.h) and submission goes through ac_runtime.
//
// One token of the offloaded layers is one command batch: per layer
//   norm_q8(x, attn_norm) -> Q, K, V projections (V into its cache row) -> RoPE (K into its cache row) -> attention
//   -> norm_q8(attention, attn_sub_norm) -> output projection added to x -> norm_q8(x, ffn_norm) -> gate, up
//   -> norm_q8(relu^2(gate) * up or silu(gate) * up, ffn_sub_norm) -> down projection added to x
// followed, when the logits are asked for, by norm(x, output_norm) and the f16 LM head. The arithmetic follows the CPU
// path of llama_bitnet_core.cpp (int8 activations, integer i2_s sums, the same scale products), so the GPU result is
// compared with the CPU result of the same build (tools: ac_bitnet_parity).
//
// Every failure throws AcBitNetError naming the step and the ac_runtime result; there is no CPU fallback.

#include "ac_runtime.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ameva {
namespace core {

class AcBitNetError : public std::runtime_error {
public:
    AcBitNetError(const std::string& step, int result);
    int result() const { return result_; }

private:
    int result_;
};

struct BitNetGpuConfig {
    uint32_t n_embd = 0;
    uint32_t n_ffn = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t n_ctx = 0;
    uint32_t n_layers = 0;     // layers on the GPU (the first n_layers of the model)
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool relu2_ffn = false;    // FFN activation relu(gate)^2 * up (BitNet b1.58) instead of silu(gate) * up
};

// One layer on the host: norm weights as f32 (nullptr: the norm is not applied), i2_s projections with their
// packed bytes (out_rows * in_dim / 4) and per-tensor scales.
struct BitNetGpuLayerHost {
    const float* attn_norm = nullptr;
    const float* attn_sub_norm = nullptr;
    const float* ffn_norm = nullptr;
    const float* ffn_sub_norm = nullptr;
    const void* wq = nullptr;
    const void* wk = nullptr;
    const void* wv = nullptr;
    const void* wo = nullptr;
    const void* w_gate = nullptr;
    const void* w_up = nullptr;
    const void* w_down = nullptr;
    float scale_wq = 1.0f, scale_wk = 1.0f, scale_wv = 1.0f, scale_wo = 1.0f;
    float scale_w_gate = 1.0f, scale_w_up = 1.0f, scale_w_down = 1.0f;
};

class AcBitNetEngine {
public:
    AcBitNetEngine() = default;
    ~AcBitNetEngine();
    AcBitNetEngine(const AcBitNetEngine&) = delete;
    AcBitNetEngine& operator=(const AcBitNetEngine&) = delete;

    // Creates the ac_runtime context (Vulkan) and the kernels.
    void Initialize();
    // Allocates the per-layer weight, KV-cache and scratch buffers for cfg.
    void Configure(const BitNetGpuConfig& cfg);
    // Uploads layer l (0 <= l < cfg.n_layers).
    void UploadLayer(uint32_t l, const BitNetGpuLayerHost& lay);
    // The final norm (f32, n_embd) and the f16 LM head ([n_vocab][n_embd]); the head is split into slices that each fit
    // one binding.
    void UploadOutput(const float* output_norm, const uint16_t* lm_head_f16, uint32_t n_vocab);

    // Runs the GPU layers for the token at pos on x_in (n_embd). x_out (n_embd): the residual stream after the last GPU
    // layer, read back. logits (n_vocab): the final norm and LM head run in the same batch (all layers on the GPU).
    // chunk_layers > 0: the batch is submitted every chunk_layers layers (bounds one submission's GPU time).
    void EvalToken(uint32_t pos, const float* x_in, float* x_out, float* logits, uint32_t chunk_layers);
    // The LM head alone on a normalized x (n_embd), for runs whose last layers are on the CPU.
    void LMHead(const float* x_norm, float* logits);

    bool HasLMHead() const { return !lm_slices_.empty(); }
    const std::string& DeviceName() const { return device_name_; }
    uint64_t WeightBytes() const { return weight_bytes_; }
    uint64_t LMHeadBytes() const { return lm_head_bytes_; }
    uint64_t KVCacheBytes() const { return kv_bytes_; }

private:
    struct Layer {
        AcBuffer* weights = nullptr;
        uint64_t off_attn_norm = 0, off_attn_sub_norm = 0, off_ffn_norm = 0, off_ffn_sub_norm = 0;
        uint64_t off_wq = 0, off_wk = 0, off_wv = 0, off_wo = 0, off_gate = 0, off_up = 0, off_down = 0;
        bool has_attn_sub_norm = false, has_ffn_sub_norm = false;
        float scale_wq = 1.0f, scale_wk = 1.0f, scale_wv = 1.0f, scale_wo = 1.0f;
        float scale_gate = 1.0f, scale_up = 1.0f, scale_down = 1.0f;
        AcBuffer* kcache = nullptr;
        AcBuffer* vcache = nullptr;
    };
    struct LMSlice {
        AcBuffer* buf = nullptr;
        uint32_t row0 = 0;
        uint32_t rows = 0;
    };

    AcBuffer* NewBuffer(uint64_t bytes, const char* what);
    void Write(AcBuffer* buf, uint64_t off, const void* src, uint64_t bytes, const char* what);
    AcBufferBinding Bind(AcBuffer* buf, uint64_t off, uint64_t bytes) const;
    uint64_t Align(uint64_t v) const { return (v + align_ - 1) / align_ * align_; }
    void Dispatch(AcCommandBatch* batch, uint32_t kernel, const AcBufferBinding* b, uint32_t nb, const void* pc,
                  uint32_t pc_bytes, uint32_t groups);
    void Barrier(AcCommandBatch* batch);
    void RecordNormQ8(AcCommandBatch* batch, AcBufferBinding in_a, AcBufferBinding in_b, AcBufferBinding w,
                      uint32_t dim, uint32_t mode, bool do_norm);
    uint32_t GemvKernel(bool accum) const;
    void RecordGemv(AcCommandBatch* batch, bool accum, AcBufferBinding weights, uint32_t in_dim, uint32_t out_dim,
                    float scale, AcBufferBinding out, uint32_t out_offset_words);
    void RecordLayer(AcCommandBatch* batch, const Layer& L, uint32_t pos);
    void RecordLMHead(AcCommandBatch* batch, AcBufferBinding x_norm);
    void SubmitAndWait(AcCommandBatch* batch, const char* what, uint32_t pos);
    void Release();

    // AC_BITNET_PROFILE=1: every token batch records a GPU timestamp per dispatch; the time per kernel is printed to
    // stderr for each token (kernel order of the batch in recorded_)
    bool profile_ = false;
    std::vector<uint32_t> recorded_;

    AcContext* ctx_ = nullptr;
    std::vector<AcKernel*> kernels_;
    std::string device_name_;
    uint64_t align_ = 256;
    uint64_t max_range_ = 0;
    uint32_t max_groups_x_ = 65535;
    bool configured_ = false;
    bool use_dot_ = false;         // i2_s GEMV through VK_KHR_shader_integer_dot_product
    uint32_t gemv_shape_ = 0;      // index into kAcBitnetGemvShapes (AC_BITNET_GEMV_SHAPE)
    uint32_t lmhead_kernel_ = 0;   // AC_BITNET_K_GEMV_F16_W1 / W2 / W4 (AC_BITNET_LMHEAD_SHAPE)
    BitNetGpuConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<LMSlice> lm_slices_;
    uint32_t n_vocab_ = 0;
    uint64_t weight_bytes_ = 0, lm_head_bytes_ = 0, kv_bytes_ = 0;

    // scratch
    AcBuffer* x_ = nullptr;        // residual stream, n_embd f32
    AcBuffer* y_ = nullptr;        // f32 activations of norm_q8, max(n_embd, n_ffn)
    AcBuffer* q8_ = nullptr;       // int8 activations, max(n_embd, n_ffn) / 4 words
    AcBuffer* qeo_ = nullptr;      // q8_ + 128 split into even / odd bytes (emulated dot product)
    AcBuffer* act_ = nullptr;      // dequantization factor and sum of q8_
    AcBuffer* q_ = nullptr;        // Q, n_heads * head_dim
    AcBuffer* k_ = nullptr;        // K before RoPE, n_kv_heads * head_dim
    AcBuffer* att_ = nullptr;      // attention output, n_heads * head_dim
    AcBuffer* scores_ = nullptr;   // n_heads * n_ctx
    AcBuffer* gate_ = nullptr;     // n_ffn
    AcBuffer* up_ = nullptr;       // n_ffn
    AcBuffer* cs_ = nullptr;       // RoPE cos / sin of the token's position, head_dim
    AcBuffer* out_norm_ = nullptr; // final norm weight, n_embd
    AcBuffer* logits_ = nullptr;   // n_vocab
    AcBuffer* dummy_ = nullptr;    // bound where a kernel binding is unused
    std::vector<float> cs_host_;
};

} // namespace core
} // namespace ameva
