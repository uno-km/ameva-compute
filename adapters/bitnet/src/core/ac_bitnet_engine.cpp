/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ac_bitnet_engine.h"
#include "bitnet_kernels_spv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ameva {
namespace core {

namespace {

constexpr uint64_t kWaitNs = 60ull * 1000000000ull;
// LM-head slice: at most this many bytes per buffer, below every binding limit the fleet reports
constexpr uint64_t kSliceBytes = 128ull << 20;

uint32_t fbits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

std::string step_text(const std::string& step, int result) {
    return "[ac_bitnet] " + step + " failed (AcResult " + std::to_string(result) + ")";
}

void check(AcResult r, const char* step) {
    if (r != AC_SUCCESS) throw AcBitNetError(step, (int)r);
}

} // namespace

AcBitNetError::AcBitNetError(const std::string& step, int result) : std::runtime_error(step_text(step, result)), result_(result) {}

AcBitNetEngine::~AcBitNetEngine() { Release(); }

void AcBitNetEngine::Initialize() {
    if (ctx_) return;
    AcContextDesc desc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    check(ac_runtime_context_create(&desc, &ctx_), "ac_runtime_context_create (Vulkan)");
    char name[256] = {};
    check(ac_runtime_context_get_device_name(ctx_, name, sizeof(name)), "ac_runtime_context_get_device_name");
    device_name_ = name;
    AcDeviceLimits lim = {};
    lim.abi_version = AC_ABI_VERSION;
    lim.struct_size = sizeof(AcDeviceLimits);
    check(ac_runtime_context_get_limits(ctx_, &lim), "ac_runtime_context_get_limits");
    align_ = std::max<uint64_t>(256, lim.min_storage_buffer_offset_alignment);
    max_range_ = lim.max_storage_buffer_range;
    max_groups_x_ = lim.max_compute_workgroup_count[0];
    if (max_range_ == 0) throw AcBitNetError("device reports maxStorageBufferRange 0", (int)AC_ERR_INVALID_ARGUMENT);
    // AC_BITNET_GEMV: auto (default) takes the integer dot product where the device enabled it; dot requires it; emu
    // takes the emulated form (bitnet_gemv_i2_s.comp -DEMU_DOT). Both give the same integers.
    {
        const char* e = getenv("AC_BITNET_GEMV");
        const std::string mode = (e && *e) ? e : "auto";
        if (mode != "auto" && mode != "dot" && mode != "emu") {
            throw AcBitNetError("AC_BITNET_GEMV must be auto, dot or emu, got " + mode, (int)AC_ERR_INVALID_ARGUMENT);
        }
        if (mode == "dot" && !lim.integer_dot_product) {
            throw AcBitNetError("AC_BITNET_GEMV=dot on a device without VK_KHR_shader_integer_dot_product",
                                (int)AC_ERR_FEATURE_NOT_SUPPORTED);
        }
        use_dot_ = mode == "dot" || (mode == "auto" && lim.integer_dot_product);
        // AC_BITNET_GEMV_SHAPE: the work split of the GEMV (lanes per row x words per load, kAcBitnetGemvShapes)
        const char* sh = getenv("AC_BITNET_GEMV_SHAPE");
        const std::string shape = (sh && *sh) ? sh : "l32w1";
        gemv_shape_ = AC_BITNET_GEMV_SHAPES;
        for (uint32_t i = 0; i < AC_BITNET_GEMV_SHAPES; ++i) {
            if (shape == kAcBitnetGemvShapes[i]) gemv_shape_ = i;
        }
        if (gemv_shape_ == AC_BITNET_GEMV_SHAPES) {
            throw AcBitNetError("AC_BITNET_GEMV_SHAPE " + shape + " is not a built GEMV shape", (int)AC_ERR_INVALID_ARGUMENT);
        }
        // AC_BITNET_LMHEAD_SHAPE: words per load of the f16 LM head (w1, w2, w4)
        const char* lh = getenv("AC_BITNET_LMHEAD_SHAPE");
        const std::string lshape = (lh && *lh) ? lh : "w1";
        if (lshape == "w1") lmhead_kernel_ = AC_BITNET_K_GEMV_F16_W1;
        else if (lshape == "w2") lmhead_kernel_ = AC_BITNET_K_GEMV_F16_W2;
        else if (lshape == "w4") lmhead_kernel_ = AC_BITNET_K_GEMV_F16_W4;
        else throw AcBitNetError("AC_BITNET_LMHEAD_SHAPE must be w1, w2 or w4, got " + lshape, (int)AC_ERR_INVALID_ARGUMENT);
        fprintf(stderr, "[ac_bitnet] %s: i2_s GEMV through %s (AC_BITNET_GEMV=%s), work split %s, LM head %s\n",
                device_name_.c_str(),
                use_dot_ ? "the integer dot product (VK_KHR_shader_integer_dot_product)" : "the emulated dot product",
                mode.c_str(), shape.c_str(), lshape.c_str());
    }
    if (const char* e = getenv("AC_BITNET_PROFILE")) {
        if (strcmp(e, "1") == 0) profile_ = true;
        else if (*e && strcmp(e, "0") != 0) throw AcBitNetError(std::string("AC_BITNET_PROFILE must be 0 or 1, got ") + e, (int)AC_ERR_INVALID_ARGUMENT);
    }

    static const AcRuntimeBindingDesc kBindings[8] = {
        { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 3, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 4, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 5, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
        { 6, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 7, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    };
    kernels_.assign(AC_BITNET_K_COUNT, nullptr);
    for (uint32_t i = 0; i < AC_BITNET_K_COUNT; ++i) {
        const AcBitnetKernel& k = kAcBitnetKernels[i];
        if (i < AC_BITNET_K_NORM_Q8 && i != GemvKernel(false) && i != GemvKernel(true)) continue; // not dispatched
        if (i >= AC_BITNET_K_GEMV_F16_W1 && i <= AC_BITNET_K_GEMV_F16_W4 && i != lmhead_kernel_) continue;
        AcKernelRuntimeMetadata meta = { k.name, "main", k.bindings, kBindings, k.push_constant_bytes,
                                         { k.local_size[0], k.local_size[1], k.local_size[2] }, 0, k.spirv_sha256, 0,
                                         nullptr };
        AcKernelDesc kd = { AC_ABI_VERSION, sizeof(AcKernelDesc), k.spirv, k.spirv_bytes, &meta, 0, nullptr };
        check(ac_runtime_kernel_create(ctx_, &kd, &kernels_[i]), (std::string("ac_runtime_kernel_create ") + k.name).c_str());
    }
}

AcBuffer* AcBitNetEngine::NewBuffer(uint64_t bytes, const char* what) {
    AcBufferDesc d = { AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)std::max<uint64_t>(bytes, 16), AC_BUFFER_USAGE_STORAGE, 0 };
    AcBuffer* b = nullptr;
    const AcResult r = ac_runtime_buffer_create(ctx_, &d, &b);
    if (r != AC_SUCCESS) {
        throw AcBitNetError(std::string("ac_runtime_buffer_create ") + what + " (" + std::to_string(bytes) + " bytes)", (int)r);
    }
    return b;
}

void AcBitNetEngine::Write(AcBuffer* buf, uint64_t off, const void* src, uint64_t bytes, const char* what) {
    check(ac_runtime_buffer_write(buf, (size_t)off, src, (size_t)bytes), (std::string("ac_runtime_buffer_write ") + what).c_str());
}

AcBufferBinding AcBitNetEngine::Bind(AcBuffer* buf, uint64_t off, uint64_t bytes) const {
    return { buf, off, bytes };
}

void AcBitNetEngine::Configure(const BitNetGpuConfig& cfg) {
    if (!ctx_) throw AcBitNetError("Configure before Initialize", (int)AC_ERR_INVALID_ARGUMENT);
    if (configured_) throw AcBitNetError("Configure called twice", (int)AC_ERR_INVALID_ARGUMENT);
    // the shapes the kernels are written for
    const bool shapes_ok = cfg.n_embd % 128 == 0 && cfg.n_ffn % 128 == 0 && cfg.n_layers > 0 && cfg.n_ctx > 0 &&
                           cfg.n_kv_heads > 0 && cfg.n_heads % cfg.n_kv_heads == 0 && cfg.head_dim % 2 == 0 &&
                           cfg.head_dim <= 256 && cfg.n_heads * cfg.head_dim == cfg.n_embd;
    if (!shapes_ok) {
        throw AcBitNetError("unsupported model shape (n_embd " + std::to_string(cfg.n_embd) + ", n_ffn " +
                            std::to_string(cfg.n_ffn) + ", heads " + std::to_string(cfg.n_heads) + "/" +
                            std::to_string(cfg.n_kv_heads) + ", head_dim " + std::to_string(cfg.head_dim) + ")",
                            (int)AC_ERR_INVALID_ARGUMENT);
    }
    cfg_ = cfg;
    const uint32_t kv_dim = cfg.n_kv_heads * cfg.head_dim;
    const uint32_t q_dim = cfg.n_heads * cfg.head_dim;
    const uint32_t vmax = std::max(cfg.n_embd, cfg.n_ffn);

    // per layer: norm weights, then the seven projections, each at an aligned offset
    const uint64_t embd_row = cfg.n_embd / 4, ffn_row = cfg.n_ffn / 4;
    layers_.assign(cfg.n_layers, Layer());
    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        Layer& L = layers_[l];
        uint64_t off = 0;
        auto take = [&](uint64_t bytes) { const uint64_t o = off; off = Align(off + bytes); return o; };
        L.off_attn_norm = take(4ull * cfg.n_embd);
        L.off_attn_sub_norm = take(4ull * cfg.n_embd);
        L.off_ffn_norm = take(4ull * cfg.n_embd);
        L.off_ffn_sub_norm = take(4ull * cfg.n_ffn);
        L.off_wq = take(q_dim * embd_row);
        L.off_wk = take(kv_dim * embd_row);
        L.off_wv = take(kv_dim * embd_row);
        L.off_wo = take(cfg.n_embd * embd_row);
        L.off_gate = take(cfg.n_ffn * embd_row);
        L.off_up = take(cfg.n_ffn * embd_row);
        L.off_down = take(cfg.n_embd * ffn_row);
        L.weights = NewBuffer(off, "layer weights");
        weight_bytes_ += off;
        const uint64_t kv = 4ull * cfg.n_ctx * kv_dim;
        L.kcache = NewBuffer(kv, "K cache");
        L.vcache = NewBuffer(kv, "V cache");
        kv_bytes_ += 2 * kv;
    }
    x_ = NewBuffer(4ull * cfg.n_embd, "x");
    y_ = NewBuffer(4ull * vmax, "y");
    q8_ = NewBuffer(vmax, "q8");
    qeo_ = NewBuffer(2ull * vmax, "q8 even / odd");
    act_ = NewBuffer(16, "act");
    q_ = NewBuffer(4ull * q_dim, "q");
    k_ = NewBuffer(4ull * kv_dim, "k");
    att_ = NewBuffer(4ull * q_dim, "attention");
    scores_ = NewBuffer(4ull * cfg.n_heads * cfg.n_ctx, "scores");
    gate_ = NewBuffer(4ull * cfg.n_ffn, "gate");
    up_ = NewBuffer(4ull * cfg.n_ffn, "up");
    cs_ = NewBuffer(4ull * cfg.head_dim, "rope table");
    dummy_ = NewBuffer(256, "dummy");
    cs_host_.assign(cfg.head_dim, 0.0f);
    configured_ = true;
}

void AcBitNetEngine::UploadLayer(uint32_t l, const BitNetGpuLayerHost& lay) {
    if (!configured_ || l >= cfg_.n_layers) throw AcBitNetError("UploadLayer: layer out of range", (int)AC_ERR_INVALID_ARGUMENT);
    Layer& L = layers_[l];
    if (!lay.attn_norm || !lay.ffn_norm || !lay.wq || !lay.wk || !lay.wv || !lay.wo || !lay.w_gate || !lay.w_up || !lay.w_down) {
        throw AcBitNetError("layer " + std::to_string(l) + " lacks a norm or projection tensor the GPU chain needs",
                            (int)AC_ERR_INVALID_ARGUMENT);
    }
    const uint32_t kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    const uint64_t embd_row = cfg_.n_embd / 4, ffn_row = cfg_.n_ffn / 4;
    Write(L.weights, L.off_attn_norm, lay.attn_norm, 4ull * cfg_.n_embd, "attn_norm");
    Write(L.weights, L.off_ffn_norm, lay.ffn_norm, 4ull * cfg_.n_embd, "ffn_norm");
    L.has_attn_sub_norm = lay.attn_sub_norm != nullptr;
    L.has_ffn_sub_norm = lay.ffn_sub_norm != nullptr;
    if (L.has_attn_sub_norm) Write(L.weights, L.off_attn_sub_norm, lay.attn_sub_norm, 4ull * cfg_.n_embd, "attn_sub_norm");
    if (L.has_ffn_sub_norm) Write(L.weights, L.off_ffn_sub_norm, lay.ffn_sub_norm, 4ull * cfg_.n_ffn, "ffn_sub_norm");
    if (L.has_ffn_sub_norm != cfg_.relu2_ffn) {
        throw AcBitNetError("layer " + std::to_string(l) + ": ffn_sub_norm presence differs from the model's FFN form",
                            (int)AC_ERR_INVALID_ARGUMENT);
    }
    Write(L.weights, L.off_wq, lay.wq, (uint64_t)cfg_.n_heads * cfg_.head_dim * embd_row, "attn_q");
    Write(L.weights, L.off_wk, lay.wk, (uint64_t)kv_dim * embd_row, "attn_k");
    Write(L.weights, L.off_wv, lay.wv, (uint64_t)kv_dim * embd_row, "attn_v");
    Write(L.weights, L.off_wo, lay.wo, (uint64_t)cfg_.n_embd * embd_row, "attn_output");
    Write(L.weights, L.off_gate, lay.w_gate, (uint64_t)cfg_.n_ffn * embd_row, "ffn_gate");
    Write(L.weights, L.off_up, lay.w_up, (uint64_t)cfg_.n_ffn * embd_row, "ffn_up");
    Write(L.weights, L.off_down, lay.w_down, (uint64_t)cfg_.n_embd * ffn_row, "ffn_down");
    L.scale_wq = lay.scale_wq;
    L.scale_wk = lay.scale_wk;
    L.scale_wv = lay.scale_wv;
    L.scale_wo = lay.scale_wo;
    L.scale_gate = lay.scale_w_gate;
    L.scale_up = lay.scale_w_up;
    L.scale_down = lay.scale_w_down;
}

void AcBitNetEngine::UploadOutput(const float* output_norm, const uint16_t* lm_head_f16, uint32_t n_vocab) {
    if (!configured_ || !output_norm || !lm_head_f16 || n_vocab == 0 || cfg_.n_embd % 8 != 0) {
        throw AcBitNetError("UploadOutput: missing output norm or LM head", (int)AC_ERR_INVALID_ARGUMENT);
    }
    if (!lm_slices_.empty()) throw AcBitNetError("UploadOutput called twice", (int)AC_ERR_INVALID_ARGUMENT);
    out_norm_ = NewBuffer(4ull * cfg_.n_embd, "output_norm");
    Write(out_norm_, 0, output_norm, 4ull * cfg_.n_embd, "output_norm");
    const uint64_t row_bytes = 2ull * cfg_.n_embd;
    uint64_t rows_per = std::min<uint64_t>(std::min(kSliceBytes, max_range_) / row_bytes, 4ull * max_groups_x_);
    if (rows_per == 0) throw AcBitNetError("LM head row does not fit one binding", (int)AC_ERR_INVALID_ARGUMENT);
    for (uint32_t r0 = 0; r0 < n_vocab; r0 += (uint32_t)rows_per) {
        LMSlice s;
        s.row0 = r0;
        s.rows = (uint32_t)std::min<uint64_t>(rows_per, n_vocab - r0);
        s.buf = NewBuffer(s.rows * row_bytes, "LM head slice");
        lm_slices_.push_back(s);
        Write(s.buf, 0, lm_head_f16 + (uint64_t)r0 * cfg_.n_embd, s.rows * row_bytes, "LM head slice");
        lm_head_bytes_ += s.rows * row_bytes;
    }
    n_vocab_ = n_vocab;
    logits_ = NewBuffer(4ull * n_vocab, "logits");
}

void AcBitNetEngine::Dispatch(AcCommandBatch* batch, uint32_t kernel, const AcBufferBinding* b, uint32_t nb,
                              const void* pc, uint32_t pc_bytes, uint32_t groups) {
    if (groups == 0 || groups > max_groups_x_) {
        throw AcBitNetError(std::string("dispatch of ") + kAcBitnetKernels[kernel].name + " with " +
                            std::to_string(groups) + " workgroups", (int)AC_ERR_INVALID_ARGUMENT);
    }
    const AcResult r = ac_runtime_batch_dispatch_bindings(batch, kernels_[kernel], b, nb, pc, pc_bytes, { groups, 1, 1 });
    if (r != AC_SUCCESS) {
        throw AcBitNetError(std::string("ac_runtime_batch_dispatch_bindings ") + kAcBitnetKernels[kernel].name, (int)r);
    }
    recorded_.push_back(kernel);
}

void AcBitNetEngine::Barrier(AcCommandBatch* batch) { check(ac_runtime_batch_barrier(batch), "ac_runtime_batch_barrier"); }

void AcBitNetEngine::RecordNormQ8(AcCommandBatch* batch, AcBufferBinding in_a, AcBufferBinding in_b, AcBufferBinding w,
                                  uint32_t dim, uint32_t mode, bool do_norm) {
    const AcBufferBinding b[7] = { in_a, in_b, w, Bind(y_, 0, 4ull * dim), Bind(q8_, 0, dim), Bind(act_, 0, 16),
                                   Bind(qeo_, 0, 2ull * dim) };
    const uint32_t pc[4] = { dim, fbits(cfg_.norm_eps), mode, do_norm ? 1u : 0u };
    Dispatch(batch, AC_BITNET_K_NORM_Q8, b, 7, pc, sizeof(pc), 1);
}

uint32_t AcBitNetEngine::GemvKernel(bool accum) const {
    return AC_BITNET_K_GEMV + 4 * gemv_shape_ + (use_dot_ ? 0 : 2) + (accum ? 1 : 0);
}

void AcBitNetEngine::RecordGemv(AcCommandBatch* batch, bool accum, AcBufferBinding weights, uint32_t in_dim,
                                uint32_t out_dim, float scale, AcBufferBinding out, uint32_t out_offset_words) {
    const AcBufferBinding b[4] = { weights, use_dot_ ? Bind(q8_, 0, in_dim) : Bind(qeo_, 0, 2ull * in_dim), out,
                                   Bind(act_, 0, 16) };
    const uint32_t pc[4] = { in_dim, out_dim, fbits(scale), out_offset_words };
    const uint32_t k = GemvKernel(accum);
    const uint32_t rows_per_group = kAcBitnetKernels[k].local_size[1];
    Dispatch(batch, k, b, 4, pc, sizeof(pc), (out_dim + rows_per_group - 1) / rows_per_group);
}

void AcBitNetEngine::RecordLayer(AcCommandBatch* batch, const Layer& L, uint32_t pos) {
    const uint32_t E = cfg_.n_embd, F = cfg_.n_ffn, kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    const uint32_t q_dim = cfg_.n_heads * cfg_.head_dim;
    const uint64_t embd_row = E / 4, ffn_row = F / 4;
    const uint64_t kv_bytes = 4ull * cfg_.n_ctx * kv_dim;
    const AcBufferBinding x = Bind(x_, 0, 4ull * E);
    const AcBufferBinding none = Bind(dummy_, 0, 256);

    // attention
    RecordNormQ8(batch, x, x, Bind(L.weights, L.off_attn_norm, 4ull * E), E, 0, true);
    Barrier(batch);
    RecordGemv(batch, false, Bind(L.weights, L.off_wq, q_dim * embd_row), E, q_dim, L.scale_wq, Bind(q_, 0, 4ull * q_dim), 0);
    RecordGemv(batch, false, Bind(L.weights, L.off_wk, kv_dim * embd_row), E, kv_dim, L.scale_wk, Bind(k_, 0, 4ull * kv_dim), 0);
    RecordGemv(batch, false, Bind(L.weights, L.off_wv, kv_dim * embd_row), E, kv_dim, L.scale_wv, Bind(L.vcache, 0, kv_bytes),
               pos * kv_dim);
    Barrier(batch);
    {
        const AcBufferBinding b[4] = { Bind(q_, 0, 4ull * q_dim), Bind(k_, 0, 4ull * kv_dim), Bind(L.kcache, 0, kv_bytes),
                                       Bind(cs_, 0, 4ull * cfg_.head_dim) };
        const uint32_t pc[4] = { cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim, pos * kv_dim };
        const uint32_t threads = (cfg_.n_heads + cfg_.n_kv_heads) * (cfg_.head_dim / 2);
        Dispatch(batch, AC_BITNET_K_ROPE, b, 4, pc, sizeof(pc), (threads + 63) / 64);
    }
    Barrier(batch);
    {
        const AcBufferBinding b[5] = { Bind(q_, 0, 4ull * q_dim), Bind(L.kcache, 0, kv_bytes), Bind(L.vcache, 0, kv_bytes),
                                       Bind(scores_, 0, 4ull * cfg_.n_heads * cfg_.n_ctx), Bind(att_, 0, 4ull * q_dim) };
        const float inv_sqrt = 1.0f / std::sqrt((float)cfg_.head_dim); // as multi_head_attention of the CPU path
        const uint32_t pc[7] = { cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim, pos + 1, kv_dim, cfg_.n_ctx, fbits(inv_sqrt) };
        Dispatch(batch, AC_BITNET_K_ATTN_DECODE, b, 5, pc, sizeof(pc), cfg_.n_heads);
    }
    Barrier(batch);
    const AcBufferBinding att = Bind(att_, 0, 4ull * q_dim);
    RecordNormQ8(batch, att, att, L.has_attn_sub_norm ? Bind(L.weights, L.off_attn_sub_norm, 4ull * E) : none, E, 0,
                 L.has_attn_sub_norm);
    Barrier(batch);
    RecordGemv(batch, true, Bind(L.weights, L.off_wo, E * embd_row), E, E, L.scale_wo, x, 0);
    Barrier(batch);

    // feed-forward
    RecordNormQ8(batch, x, x, Bind(L.weights, L.off_ffn_norm, 4ull * E), E, 0, true);
    Barrier(batch);
    RecordGemv(batch, false, Bind(L.weights, L.off_gate, F * embd_row), E, F, L.scale_gate, Bind(gate_, 0, 4ull * F), 0);
    RecordGemv(batch, false, Bind(L.weights, L.off_up, F * embd_row), E, F, L.scale_up, Bind(up_, 0, 4ull * F), 0);
    Barrier(batch);
    RecordNormQ8(batch, Bind(gate_, 0, 4ull * F), Bind(up_, 0, 4ull * F),
                 L.has_ffn_sub_norm ? Bind(L.weights, L.off_ffn_sub_norm, 4ull * F) : none, F, cfg_.relu2_ffn ? 1 : 2,
                 L.has_ffn_sub_norm);
    Barrier(batch);
    RecordGemv(batch, true, Bind(L.weights, L.off_down, E * ffn_row), F, E, L.scale_down, x, 0);
    Barrier(batch);
}

void AcBitNetEngine::RecordLMHead(AcCommandBatch* batch, AcBufferBinding x_norm) {
    for (const LMSlice& s : lm_slices_) {
        const AcBufferBinding b[3] = { Bind(s.buf, 0, 2ull * s.rows * cfg_.n_embd), x_norm, Bind(logits_, 0, 4ull * n_vocab_) };
        const uint32_t pc[3] = { cfg_.n_embd, s.rows, s.row0 };
        Dispatch(batch, lmhead_kernel_, b, 3, pc, sizeof(pc), (s.rows + 3) / 4);
    }
}

void AcBitNetEngine::SubmitAndWait(AcCommandBatch* batch, const char* what, uint32_t pos) {
    AcEvent* ev = nullptr;
    AcResult r = ac_runtime_batch_submit(batch, &ev);
    if (r == AC_SUCCESS) r = ac_runtime_event_wait(ev, kWaitNs);
    if (r == AC_SUCCESS && profile_) {
        std::vector<uint64_t> ns(recorded_.size());
        uint32_t n = 0;
        r = ac_runtime_batch_get_dispatch_times(batch, ns.data(), (uint32_t)ns.size(), &n);
        if (r == AC_SUCCESS && n == recorded_.size()) {
            double per[AC_BITNET_K_COUNT] = {};
            uint32_t cnt[AC_BITNET_K_COUNT] = {};
            double total = 0;
            for (uint32_t i = 0; i < n; ++i) {
                per[recorded_[i]] += ns[i] * 1e-6;
                cnt[recorded_[i]]++;
                total += ns[i] * 1e-6;
            }
            fprintf(stderr, "[ac_bitnet] profile %s pos %u: GPU %.2f ms", what, pos, total);
            for (uint32_t k = 0; k < AC_BITNET_K_COUNT; ++k) {
                if (cnt[k]) fprintf(stderr, " | %s %.2f ms (%u)", kAcBitnetKernels[k].name, per[k], cnt[k]);
            }
            fprintf(stderr, "\n");
        } else if (r == AC_SUCCESS) {
            r = AC_ERR_INVALID_ARGUMENT; // timestamps of another number of dispatches than were recorded
        }
    }
    if (ev) ac_runtime_event_destroy(ev);
    ac_runtime_batch_destroy(batch);
    recorded_.clear();
    if (r != AC_SUCCESS) throw AcBitNetError(std::string("submit / wait / profile of ") + what, (int)r);
}

void AcBitNetEngine::EvalToken(uint32_t pos, const float* x_in, float* x_out, float* logits, uint32_t chunk_layers) {
    if (!configured_) throw AcBitNetError("EvalToken before Configure", (int)AC_ERR_INVALID_ARGUMENT);
    if (pos >= cfg_.n_ctx) {
        throw AcBitNetError("position " + std::to_string(pos) + " outside the KV cache of " + std::to_string(cfg_.n_ctx),
                            (int)AC_ERR_INVALID_ARGUMENT);
    }
    if (logits && lm_slices_.empty()) throw AcBitNetError("logits requested without an LM head", (int)AC_ERR_INVALID_ARGUMENT);
    // RoPE factors of this position, computed as apply_rope of the CPU path does
    const uint32_t half = cfg_.head_dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        const float theta = (float)pos / std::pow(cfg_.rope_theta, (float)(2 * i) / (float)cfg_.head_dim);
        cs_host_[i] = std::cos(theta);
        cs_host_[half + i] = std::sin(theta);
    }
    Write(cs_, 0, cs_host_.data(), 4ull * cfg_.head_dim, "rope table");
    Write(x_, 0, x_in, 4ull * cfg_.n_embd, "x");

    const uint32_t capacity = cfg_.n_layers * 16 + (uint32_t)lm_slices_.size() + 8;
    const AcBatchDesc bd = { AC_ABI_VERSION, sizeof(AcBatchDesc), capacity,
                             AC_BATCH_FLAG_EXPLICIT_BARRIERS | (profile_ ? AC_BATCH_FLAG_DISPATCH_TIMESTAMPS : 0u) };
    AcCommandBatch* batch = nullptr;
    recorded_.clear();
    check(ac_runtime_batch_begin_ex(ctx_, &bd, &batch), "ac_runtime_batch_begin_ex");
    try {
        for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
            RecordLayer(batch, layers_[l], pos);
            if (chunk_layers > 0 && (l + 1) % chunk_layers == 0 && l + 1 < cfg_.n_layers) {
                check(ac_runtime_batch_flush(batch), "ac_runtime_batch_flush");
            }
        }
        if (logits) {
            const AcBufferBinding x = Bind(x_, 0, 4ull * cfg_.n_embd);
            RecordNormQ8(batch, x, x, Bind(out_norm_, 0, 4ull * cfg_.n_embd), cfg_.n_embd, 0, true);
            Barrier(batch);
            RecordLMHead(batch, Bind(y_, 0, 4ull * cfg_.n_embd));
        }
    } catch (...) {
        ac_runtime_batch_destroy(batch);
        throw;
    }
    SubmitAndWait(batch, "token", pos);
    if (x_out) check(ac_runtime_buffer_read(x_, 0, x_out, 4ull * cfg_.n_embd), "ac_runtime_buffer_read x");
    if (logits) check(ac_runtime_buffer_read(logits_, 0, logits, 4ull * n_vocab_), "ac_runtime_buffer_read logits");
}

void AcBitNetEngine::LMHead(const float* x_norm, float* logits) {
    if (lm_slices_.empty()) throw AcBitNetError("LMHead without an uploaded LM head", (int)AC_ERR_INVALID_ARGUMENT);
    Write(y_, 0, x_norm, 4ull * cfg_.n_embd, "x_norm");
    const AcBatchDesc bd = { AC_ABI_VERSION, sizeof(AcBatchDesc), (uint32_t)lm_slices_.size() + 1,
                             AC_BATCH_FLAG_EXPLICIT_BARRIERS | (profile_ ? AC_BATCH_FLAG_DISPATCH_TIMESTAMPS : 0u) };
    AcCommandBatch* batch = nullptr;
    recorded_.clear();
    check(ac_runtime_batch_begin_ex(ctx_, &bd, &batch), "ac_runtime_batch_begin_ex");
    try {
        RecordLMHead(batch, Bind(y_, 0, 4ull * cfg_.n_embd));
    } catch (...) {
        ac_runtime_batch_destroy(batch);
        throw;
    }
    SubmitAndWait(batch, "LM head", 0);
    check(ac_runtime_buffer_read(logits_, 0, logits, 4ull * n_vocab_), "ac_runtime_buffer_read logits");
}

void AcBitNetEngine::Release() {
    auto drop = [](AcBuffer*& b) {
        if (b) ac_runtime_buffer_destroy(b);
        b = nullptr;
    };
    for (Layer& L : layers_) {
        drop(L.weights);
        drop(L.kcache);
        drop(L.vcache);
    }
    layers_.clear();
    for (LMSlice& s : lm_slices_) drop(s.buf);
    lm_slices_.clear();
    for (AcBuffer** b : { &x_, &y_, &q8_, &qeo_, &act_, &q_, &k_, &att_, &scores_, &gate_, &up_, &cs_, &out_norm_, &logits_, &dummy_ }) {
        drop(*b);
    }
    for (AcKernel*& k : kernels_) {
        if (k) ac_runtime_kernel_destroy(k);
        k = nullptr;
    }
    kernels_.clear();
    if (ctx_) {
        const AcResult r = ac_runtime_context_destroy(ctx_);
        if (r != AC_SUCCESS) fprintf(stderr, "[ac_bitnet] ac_runtime_context_destroy failed (AcResult %d)\n", (int)r);
        ctx_ = nullptr;
    }
    configured_ = false;
}

} // namespace core
} // namespace ameva
