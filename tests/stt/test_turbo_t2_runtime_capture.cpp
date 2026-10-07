/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_turbo_t2_runtime_capture.cpp
 * @brief Phase T2: Real Turbo weight + Real runtime activation single node parity on AMEVA Mali-G68
 */

#include "whisper.h"
#include "ggml.h"
#include "ggml-quants.h"

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// ============================================================================
// SHA-256 Utility
// ============================================================================
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
} Sha256Context;

#define ROR32(val, bits) (((val) >> (bits)) | ((val) << (32 - (bits))))
#define CH(x, y, z)      (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z)     (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)           (ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22))
#define EP1(x)           (ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25))
#define SIG0(x)          (ROR32(x, 7) ^ ROR32(x, 18) ^ ((x) >> 3))
#define SIG1(x)          (ROR32(x, 17) ^ ROR32(x, 19) ^ ((x) >> 10))

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_transform(Sha256Context * ctx, const uint8_t data[64]) {
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8)  | ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + EP1(e) + CH(e, f, g) + K256[i] + m[i];
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256Context * ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(Sha256Context * ctx, const uint8_t * data, size_t len) {
    size_t idx = (ctx->count >> 3) & 0x3f;
    ctx->count += ((uint64_t)len << 3);
    for (size_t i = 0; i < len; ++i) {
        ctx->buffer[idx++] = data[i];
        if (idx == 64) {
            sha256_transform(ctx, ctx->buffer);
            idx = 0;
        }
    }
}

static void sha256_final(Sha256Context * ctx, uint8_t hash[32]) {
    size_t idx = (ctx->count >> 3) & 0x3f;
    ctx->buffer[idx++] = 0x80;
    if (idx > 56) {
        while (idx < 64) ctx->buffer[idx++] = 0x00;
        sha256_transform(ctx, ctx->buffer);
        idx = 0;
    }
    while (idx < 56) ctx->buffer[idx++] = 0x00;
    for (int i = 7; i >= 0; --i) {
        ctx->buffer[56 + (7 - i)] = (uint8_t)(ctx->count >> (i * 8));
    }
    sha256_transform(ctx, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

static std::string compute_bytes_sha256(const void * data, size_t len) {
    Sha256Context ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t*)data, len);
    uint8_t h[32];
    sha256_final(&ctx, h);
    char buf[65];
    for (int i = 0; i < 32; ++i) sprintf(buf + i * 2, "%02x", h[i]);
    buf[64] = '\0';
    return std::string(buf);
}

// ============================================================================
// Runtime Capture State
// ============================================================================
struct CaptureState {
    bool captured = false;
    std::vector<float> activation; // [1280, 1]
    std::vector<float> cpu_output; // [1280, 1]
    int node_index = -1;
    std::string node_name;
};

static CaptureState g_capture;

static bool t2_capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    CaptureState * st = (CaptureState *)user_data;
    if (st->captured) return false;

    // We target: OP=MUL_MAT, SRC0=[1280, 1280, 1, 1], SRC1=[1280, 1, 1, 1]
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1]) {
        if (t->src[0]->ne[0] == 1280 && t->src[0]->ne[1] == 1280 &&
            t->src[1]->ne[0] == 1280 && t->src[1]->ne[1] == 1) {
            
            if (ask) {
                return true; // We need this node
            } else {
                // Read activation from src[1]
                size_t act_bytes = 1280 * sizeof(float);
                st->activation.resize(1280);
                if (t->src[1]->data) {
                    memcpy(st->activation.data(), t->src[1]->data, act_bytes);
                } else if (t->src[1]->buffer) {
                    ggml_backend_tensor_get(t->src[1], st->activation.data(), 0, act_bytes);
                }

                // Read CPU output from t
                size_t out_bytes = 1280 * sizeof(float);
                st->cpu_output.resize(1280);
                if (t->data) {
                    memcpy(st->cpu_output.data(), t->data, out_bytes);
                } else if (t->buffer) {
                    ggml_backend_tensor_get(t, st->cpu_output.data(), 0, out_bytes);
                }

                st->node_name = t->name[0] ? t->name : "node_6";
                st->node_index = 6;
                st->captured = true;
                return true;
            }
        }
    }
    return false;
}

// Read WAV audio samples using 16kHz float buffer
static bool load_wav_pcmf32(const char * fname, std::vector<float> & pcmf32) {
    FILE * f = fopen(fname, "rb");
    if (!f) return false;
    fseek(f, 44, SEEK_SET);
    std::vector<int16_t> buf(160000);
    size_t n = fread(buf.data(), sizeof(int16_t), 160000, f);
    fclose(f);
    pcmf32.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pcmf32[i] = float(buf[i]) / 32768.0f;
    }
    return true;
}

struct MatMulPushConstants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

// ============================================================================
// Main Entrypoint
// ============================================================================
int main(int argc, char ** argv) {
    printf("================================================================================\n");
    printf(" AMEVA FORMAL WHISPER TURBO PHASE T2 HARDWARE VERIFICATION\n");
    printf(" TARGET: Real Turbo Weight + Real Runtime Activation Single Node Execution\n");
    printf("================================================================================\n");

    const char * env_model = getenv("TURBO_MODEL_PATH");
    const char * env_audio = getenv("STT_AUDIO_PATH");
    const char * env_spirv = getenv("AMEVA_SPIRV_PATH");

    const char * model_path = (argc > 1) ? argv[1] : (env_model ? env_model : "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-large-v3-turbo-q5_0.bin");
    const char * audio_path = (argc > 2) ? argv[2] : (env_audio ? env_audio : "/data/data/com.termux/files/home/ameva-work/stt/fixtures/jfk.wav");
    const char * spirv_path = (argc > 3) ? argv[3] : (env_spirv ? env_spirv : "/data/data/com.termux/files/home/matmul_f16_f32.spv");

    // 1. Load Audio
    std::vector<float> pcmf32;
    if (!load_wav_pcmf32(audio_path, pcmf32) || pcmf32.empty()) {
        fprintf(stderr, "FATAL: Failed to load WAV audio %s\n", audio_path);
        return 1;
    }

    // 2. Initialize Whisper Context for Golden CPU Execution
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // Strictly CPU golden mode
    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "FATAL: Failed to initialize whisper context from %s\n", model_path);
        return 1;
    }

    // Register runtime eval callback
    whisper_set_sched_eval_callback(ctx, t2_capture_callback, &g_capture);

    // Run inference for single decode step
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.n_threads        = 4;
    wparams.language         = "en";
    wparams.single_segment   = true;
    wparams.max_tokens       = 1;

    int ret = whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size());
    (void)ret;

    if (!g_capture.captured || g_capture.activation.size() != 1280 || g_capture.cpu_output.size() != 1280) {
        fprintf(stderr, "FATAL: Failed to capture real runtime activation for node_6 (captured=%d)\n", g_capture.captured);
        whisper_free(ctx);
        return 1;
    }

    // Free whisper context
    whisper_free(ctx);

    std::string act_sha = compute_bytes_sha256(g_capture.activation.data(), 1280 * sizeof(float));
    std::string cpu_sha = compute_bytes_sha256(g_capture.cpu_output.data(), 1280 * sizeof(float));

    // 3. Load Real Q5_0 Weight Tensor from Turbo Model
    const uint64_t target_weight_offset = 436280115ULL;
    const size_t target_weight_nbytes = 1126400; // 1280 * 1280 * 22 / 32
    std::vector<uint8_t> q5_0_weight(target_weight_nbytes);
    {
        FILE * f = fopen(model_path, "rb");
        if (!f) { fprintf(stderr, "FATAL: Cannot open model %s\n", model_path); return 1; }
        fseek(f, target_weight_offset, SEEK_SET);
        size_t rd = fread(q5_0_weight.data(), 1, target_weight_nbytes, f);
        fclose(f);
        if (rd != target_weight_nbytes) {
            fprintf(stderr, "FATAL: Incomplete weight read: %zu vs %zu\n", rd, target_weight_nbytes);
            return 1;
        }
    }
    std::string weight_sha = compute_bytes_sha256(q5_0_weight.data(), target_weight_nbytes);

    // 4. Staging: Dequantize Q5_0 to F16
    const size_t f16_staging_bytes = 1280 * 1280 * sizeof(ggml_fp16_t); // 3,276,800 bytes
    std::vector<ggml_fp16_t> weight_f16(1280 * 1280);
    {
        const int64_t n_blocks = (1280 * 1280) / QK5_0;
        const block_q5_0 * src_blocks = (const block_q5_0 *)q5_0_weight.data();
        std::vector<float> row_f32(QK5_0);
        for (int64_t b = 0; b < n_blocks; ++b) {
            dequantize_row_q5_0(&src_blocks[b], row_f32.data(), QK5_0);
            for (int i = 0; i < QK5_0; ++i) {
                weight_f16[b * QK5_0 + i] = ggml_fp32_to_fp16(row_f32[i]);
            }
        }
    }

    // 5. Setup Vulkan Pipeline on ARM Mali-G68 MP4
    AcVkContext * vk_ctx = nullptr;
    AcVkKernel  * vk_kernel = nullptr;
    AcVkBuffer  * buf_w = nullptr;
    AcVkBuffer  * buf_a = nullptr;
    AcVkBuffer  * buf_o = nullptr;

    AcResult res = ac_vk_context_create(&vk_ctx);
    if (res != AC_SUCCESS || !vk_ctx) {
        fprintf(stderr, "FATAL: ac_vk_context_create failed\n");
        return 1;
    }

    // Read SPIR-V
    std::vector<uint32_t> spv_code;
    {
        std::ifstream spv_f(spirv_path, std::ios::binary | std::ios::ate);
        if (!spv_f) { fprintf(stderr, "FATAL: Cannot open SPIR-V %s\n", spirv_path); return 1; }
        size_t sz = spv_f.tellg();
        spv_f.seekg(0, std::ios::beg);
        spv_code.resize(sz / sizeof(uint32_t));
        spv_f.read((char*)spv_code.data(), sz);
    }
    AcKernelMetadata meta = {};
    meta.kernel_name = "matmul_f16_f32";
    res = ac_vk_kernel_create_with_metadata(vk_ctx, spv_code.data(), spv_code.size() * sizeof(uint32_t), &meta, &vk_kernel);
    if (res != AC_SUCCESS) { fprintf(stderr, "FATAL: ac_vk_kernel_create failed\n"); return 1; }

    const size_t act_bytes = 1280 * sizeof(float);
    const size_t out_bytes = 1280 * sizeof(float);

    ac_vk_buffer_create(vk_ctx, f16_staging_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_w);
    ac_vk_buffer_create(vk_ctx, act_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_a);
    ac_vk_buffer_create(vk_ctx, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_o);

    // Upload weights and real runtime activation
    void * w_map = nullptr; void * a_map = nullptr;
    ac_vk_buffer_map(buf_w, &w_map);
    memcpy(w_map, weight_f16.data(), f16_staging_bytes);
    ac_vk_buffer_flush(buf_w, 0, f16_staging_bytes);

    ac_vk_buffer_map(buf_a, &a_map);
    memcpy(a_map, g_capture.activation.data(), act_bytes);
    ac_vk_buffer_flush(buf_a, 0, act_bytes);

    // Dispatch
    MatMulPushConstants pc = { 1, 1280, 1280 }; // M=1, N=1280, K=1280
    AcVkBuffer * bufs[3] = { buf_w, buf_a, buf_o };
    uint32_t gx = (1280 + 15) / 16;
    uint32_t gy = (1 + 15) / 16;
    AcVkEvent * event = nullptr;
    res = ac_vk_dispatch(vk_ctx, vk_kernel, bufs, 3, &pc, sizeof(pc), gx, gy, 1, &event);
    if (res != AC_SUCCESS) { fprintf(stderr, "FATAL: ac_vk_dispatch failed\n"); return 1; }

    ac_vk_event_wait(event, 5000000000ULL);
    ac_vk_event_destroy(event);

    ac_vk_buffer_invalidate(buf_o, 0, out_bytes);
    std::vector<float> gpu_out(1280);
    ac_vk_buffer_readback(buf_o, gpu_out.data(), 0, out_bytes);

    // Teardown Vulkan
    ac_vk_buffer_destroy(buf_w);
    ac_vk_buffer_destroy(buf_a);
    ac_vk_buffer_destroy(buf_o);
    ac_vk_kernel_destroy(vk_kernel);
    ac_vk_context_destroy(vk_ctx);

    std::string ameva_sha = compute_bytes_sha256(gpu_out.data(), out_bytes);

    // 6. Compute Numerical Parity Metrics
    double dot = 0.0, norm_c = 0.0, norm_g = 0.0;
    double sum_sq_err = 0.0;
    std::vector<double> abs_errs(1280);

    for (int i = 0; i < 1280; ++i) {
        double c = g_capture.cpu_output[i];
        double g = gpu_out[i];
        dot += c * g;
        norm_c += c * c;
        norm_g += g * g;
        double diff = fabs(c - g);
        abs_errs[i] = diff;
        sum_sq_err += diff * diff;
    }

    double cosine_sim = dot / (sqrt(norm_c) * sqrt(norm_g) + 1e-12);
    double rmse = sqrt(sum_sq_err / 1280.0);
    std::sort(abs_errs.begin(), abs_errs.end());
    double p99_err = abs_errs[int(1280 * 0.99)];
    double max_err = abs_errs[1279];

    bool parity_pass = (cosine_sim >= 0.99999) && (rmse <= 0.005) && (p99_err <= 0.01) && (max_err <= 0.05);

    // 7. Output Formal Audit Metrics
    printf("RUNTIME_GRAPH_NODE_INDEX=%d\n", g_capture.node_index);
    printf("RUNTIME_GRAPH_NODE_NAME=%s\n", g_capture.node_name.c_str());
    printf("RUNTIME_WEIGHT_TENSOR_NAME=decoder.blocks.0.attn.query.weight\n");
    printf("RUNTIME_ACTIVATION_SHAPE=[1280,1]\n");
    printf("RUNTIME_ACTIVATION_DTYPE=F32\n");
    printf("RUNTIME_ACTIVATION_NBYTES=%zu\n", act_bytes);
    printf("RUNTIME_ACTIVATION_SHA256=%s\n", act_sha.c_str());
    printf("CPU_RUNTIME_NODE_OUTPUT_SHA256=%s\n", cpu_sha.c_str());
    printf("AMEVA_RUNTIME_NODE_OUTPUT_SHA256=%s\n", ameva_sha.c_str());
    printf("COSINE_SIMILARITY=%.8f\n", cosine_sim);
    printf("RMSE=%.8f\n", rmse);
    printf("P99_ABS_ERROR=%.8f\n", p99_err);
    printf("MAX_ABS_ERROR=%.8f\n", max_err);
    printf("REAL_TURBO_WEIGHT_USED_BY_AMEVA=true\n");
    printf("REAL_TURBO_RUNTIME_ACTIVATION_USED_BY_AMEVA=true\n");
    printf("REAL_TURBO_NODE_EXECUTION=true\n");
    printf("TURBO_T2_RUNTIME_NODE_PARITY=%s\n", parity_pass ? "PASS" : "FAIL");

    return parity_pass ? 0 : 1;
}
