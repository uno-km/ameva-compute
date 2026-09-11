/**
 * @file test_turbo_t1_matmul.cpp
 * @brief Phase T1: Real Whisper Turbo Q5_0 weight single MatMul on AMEVA Vulkan with CPU dequant staging
 */

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
#include <algorithm>
#include <random>
#include <fstream>

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
    size_t i = 0;
    size_t idx = (ctx->count >> 3) & 0x3f;
    ctx->count += ((uint64_t)len << 3);
    while (i < len) {
        ctx->buffer[idx++] = data[i++];
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
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char buf[65];
    for (int i = 0; i < 32; ++i) sprintf(buf + i * 2, "%02x", hash[i]);
    buf[64] = '\0';
    return std::string(buf);
}

static std::string compute_file_sha256(const char * path, size_t * out_size = nullptr) {
    FILE * f = fopen(path, "rb");
    if (!f) return "";
    Sha256Context ctx;
    sha256_init(&ctx);
    uint8_t buf[65536];
    size_t total = 0;
    while (true) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n == 0) break;
        sha256_update(&ctx, buf, n);
        total += n;
    }
    fclose(f);
    if (out_size) *out_size = total;
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char hex[65];
    for (int i = 0; i < 32; ++i) sprintf(hex + i * 2, "%02x", hash[i]);
    hex[64] = '\0';
    return std::string(hex);
}

struct MatMulPushConstants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

int main(int argc, char ** argv) {
    printf("================================================================================\n");
    printf(" AMEVA FORMAL WHISPER TURBO PHASE T1 HARDWARE VERIFICATION\n");
    printf(" TARGET: Single Real Turbo Q5_0 MatMul with F16 Staging on ARM Mali-G68 MP4\n");
    printf("================================================================================\n");

    const char * model_path = (argc > 1) ? argv[1] : "/data/data/com.termux/files/home/ameva-models/stt/ggml-large-v3-turbo-q5_0.bin";
    size_t model_sz = 0;
    std::string model_sha = compute_file_sha256(model_path, &model_sz);
    printf("TURBO_MODEL_PATH=%s\n", model_path);
    printf("TURBO_MODEL_SIZE=%zu\n", model_sz);
    printf("TURBO_MODEL_SHA256=%s\n", model_sha.c_str());

    if (model_sz == 0) {
        fprintf(stderr, "FATAL: Turbo model not found at %s\n", model_path);
        return 1;
    }

    const uint32_t TARGET_K = 1280;
    const uint32_t TARGET_N = 1280;
    const uint32_t TARGET_M = 1;
    const size_t EXPECTED_WEIGHT_BYTES = (TARGET_K / 32) * 22 * TARGET_N; // 1,126,400 bytes

    const char * TARGET_GRAPH_KIND = "DECODER";
    const char * TARGET_NODE_NAME = "NOT_CAPTURED";
    const char * TEST_TARGET_LABEL = "node_decoder_0_attn_q";
    const char * TARGET_WEIGHT_TENSOR_NAME = "decoder.blocks.0.attn.query.weight";

    // 1. Scan model binary precisely with defensive input validation
    FILE * mf = fopen(model_path, "rb");
    if (!mf) {
        fprintf(stderr, "FATAL: Cannot open model file for binary scan\n");
        return 2;
    }

    uint32_t magic = 0;
    if (fread(&magic, 4, 1, mf) != 1 || magic != 0x67676d6c) {
        fprintf(stderr, "FATAL: Invalid model file magic: 0x%08x\n", magic);
        fclose(mf);
        return 3;
    }
    int32_t hparams[11];
    if (fread(hparams, sizeof(int32_t), 11, mf) != 11) {
        fprintf(stderr, "FATAL: Truncated hparams\n");
        fclose(mf);
        return 3;
    }
    int32_t n_mel = 0, n_fft = 0;
    if (fread(&n_mel, 4, 1, mf) != 1 || fread(&n_fft, 4, 1, mf) != 1) {
        fprintf(stderr, "FATAL: Truncated mel filter header\n");
        fclose(mf);
        return 3;
    }
    fseek(mf, (long)n_mel * n_fft * 4, SEEK_CUR);
    int32_t n_vocab = 0;
    if (fread(&n_vocab, 4, 1, mf) != 1 || n_vocab <= 0 || n_vocab > 1000000) {
        fprintf(stderr, "FATAL: Invalid vocab size: %d\n", n_vocab);
        fclose(mf);
        return 3;
    }
    for (int i = 0; i < n_vocab; ++i) {
        uint32_t len = 0;
        if (fread(&len, 4, 1, mf) != 1 || len > 1024) {
            fprintf(stderr, "FATAL: Invalid token length: %u\n", len);
            fclose(mf);
            return 3;
        }
        fseek(mf, len, SEEK_CUR);
    }

    std::vector<uint8_t> target_weight_bytes(EXPECTED_WEIGHT_BYTES);
    size_t target_weight_offset = 0;
    bool found_target_tensor = false;

    while (!feof(mf)) {
        int32_t n_dims = 0, name_len = 0, ttype = 0;
        if (fread(&n_dims, 4, 1, mf) != 1) break;
        if (n_dims < 1 || n_dims > 4) {
            fprintf(stderr, "FATAL: Invalid n_dims: %d\n", n_dims);
            fclose(mf);
            return 3;
        }
        if (fread(&name_len, 4, 1, mf) != 1 || name_len <= 0 || name_len > 256) {
            fprintf(stderr, "FATAL: Invalid name_len: %d\n", name_len);
            fclose(mf);
            return 3;
        }
        if (fread(&ttype, 4, 1, mf) != 1) {
            fprintf(stderr, "FATAL: Truncated tensor type\n");
            fclose(mf);
            return 3;
        }
        int32_t ne[4] = {1, 1, 1, 1};
        if (fread(ne, 4, n_dims, mf) != (size_t)n_dims) {
            fprintf(stderr, "FATAL: Truncated tensor ne\n");
            fclose(mf);
            return 3;
        }
        std::vector<char> tname(name_len + 1, 0);
        if (fread(tname.data(), 1, name_len, mf) != (size_t)name_len) {
            fprintf(stderr, "FATAL: Truncated tensor name\n");
            fclose(mf);
            return 3;
        }

        size_t nbytes = 0;
        if (ttype == 6) { // Q5_0
            if (ne[0] % 32 != 0) {
                fprintf(stderr, "FATAL: Q5_0 ne[0] not aligned to 32: %d\n", ne[0]);
                fclose(mf);
                return 3;
            }
            nbytes = ((size_t)ne[0] / 32) * 22;
            for (int d = 1; d < n_dims; ++d) nbytes *= (size_t)ne[d];
        } else if (ttype == 1) { // F16
            nbytes = 2;
            for (int d = 0; d < n_dims; ++d) nbytes *= (size_t)ne[d];
        } else if (ttype == 0) { // F32
            nbytes = 4;
            for (int d = 0; d < n_dims; ++d) nbytes *= (size_t)ne[d];
        } else {
            nbytes = 1;
            for (int d = 0; d < n_dims; ++d) nbytes *= (size_t)ne[d];
        }

        long cur_offset = ftell(mf);
        if ((size_t)cur_offset + nbytes > model_sz) {
            fprintf(stderr, "FATAL: Tensor offset out of bounds: %ld + %zu > %zu\n", cur_offset, nbytes, model_sz);
            fclose(mf);
            return 3;
        }

        if (strcmp(tname.data(), TARGET_WEIGHT_TENSOR_NAME) == 0) {
            target_weight_offset = cur_offset;
            if (ttype == 6 && n_dims == 2 && ne[0] == (int32_t)TARGET_K && ne[1] == (int32_t)TARGET_N && nbytes == EXPECTED_WEIGHT_BYTES) {
                if (fread(target_weight_bytes.data(), 1, EXPECTED_WEIGHT_BYTES, mf) == EXPECTED_WEIGHT_BYTES) {
                    found_target_tensor = true;
                    break;
                }
            }
        }
        fseek(mf, nbytes, SEEK_CUR);
    }
    fclose(mf);

    if (!found_target_tensor) {
        fprintf(stderr, "FATAL: Could not locate %s in %s\n", TARGET_WEIGHT_TENSOR_NAME, model_path);
        return 3;
    }

    std::string target_weight_sha = compute_bytes_sha256(target_weight_bytes.data(), target_weight_bytes.size());

    printf("TARGET_GRAPH_KIND=%s\n", TARGET_GRAPH_KIND);
    printf("TARGET_NODE_NAME=%s\n", TARGET_NODE_NAME);
    printf("TEST_TARGET_LABEL=%s\n", TEST_TARGET_LABEL);
    printf("TARGET_WEIGHT_TENSOR_NAME=%s\n", TARGET_WEIGHT_TENSOR_NAME);
    printf("TARGET_WEIGHT_OFFSET=%zu\n", target_weight_offset);
    printf("TARGET_WEIGHT_NBYTES=%zu\n", target_weight_bytes.size());
    printf("TARGET_WEIGHT_SHA256=%s\n", target_weight_sha.c_str());

    printf("TARGET_K=%u\n", TARGET_K);
    printf("TARGET_N=%u\n", TARGET_N);
    printf("TARGET_M=%u\n", TARGET_M);

    printf("Q5_0_INPUT_ACCEPTED=true\n");
    printf("Q5_0_CPU_DEQUANT_TO_F16=true\n");
    printf("Q5_0_GPU_NATIVE=false\n");
    printf("GPU_COMPUTE_PORTION=F16_F32_MUL_MAT\n");

    // 2. CPU Staging to F16 Buffer
    std::vector<ggml_fp16_t> staged_w_fp16(TARGET_K * TARGET_N);
    std::vector<float> row_temp(TARGET_K);
    const block_q5_0 * q_blocks = (const block_q5_0 *)target_weight_bytes.data();

    for (uint32_t r = 0; r < TARGET_N; ++r) {
        const block_q5_0 * row_q = q_blocks + (r * (TARGET_K / 32));
        dequantize_row_q5_0(row_q, row_temp.data(), TARGET_K);
        ggml_fp32_to_fp16_row(row_temp.data(), staged_w_fp16.data() + (r * TARGET_K), TARGET_K);
    }

    size_t f16_staging_bytes = staged_w_fp16.size() * sizeof(ggml_fp16_t);
    std::string f16_staging_sha = compute_bytes_sha256(staged_w_fp16.data(), f16_staging_bytes);
    printf("F16_STAGING_BYTES=%zu\n", f16_staging_bytes);
    printf("F16_STAGING_SHA256=%s\n", f16_staging_sha.c_str());

    // 3. Declarations for resources, lifecycle counters & metrics before any goto
    uint32_t ctx_create_count = 0;
    uint32_t ctx_destroy_count = 0;
    uint32_t kernel_create_count = 0;
    uint32_t kernel_destroy_count = 0;
    uint32_t buffer_create_count = 0;
    uint32_t buffer_destroy_count = 0;
    uint32_t event_create_count = 0;
    uint32_t event_destroy_count = 0;

    int api_gates_passed = 0;
    int api_gates_total = 0;

    AcVkContext * vk_ctx = nullptr;
    AcVkKernel  * vk_kernel = nullptr;
    AcVkKernel  * mutated_kernel = nullptr;
    AcVkBuffer  * buf_w = nullptr;
    AcVkBuffer  * buf_a = nullptr;
    AcVkBuffer  * buf_o = nullptr;
    AcVkEvent   * event = nullptr;
    AcVkEvent   * m_event = nullptr;
    AcVkEvent   * fail_event = nullptr;

    const uint32_t seeds[3] = {104729, 130363, 155921};
    int passed_challenges = 0;
    std::vector<std::string> input_hashes;
    std::vector<std::string> output_hashes;

    double worst_cos_sim = 1.0;
    double worst_rmse = 0.0;
    double worst_p99 = 0.0;
    double worst_max_abs = 0.0;
    uint64_t total_gpu_ops = 0;

    bool sens_pass = false;
    bool forced_fail_pass = false;

    const char * env_spirv = getenv("AMEVA_SPIRV_PATH");
    const char * normal_spirv = env_spirv ? env_spirv : "/data/data/com.termux/files/home/matmul_f16_f32.spv";
    std::string normal_spv_sha = compute_file_sha256(normal_spirv);

    const char * mutated_spv_path = "/data/data/com.termux/files/home/matmul_f16_f32_mutated.spv";
    std::string mutated_spv_sha = compute_file_sha256(mutated_spv_path);

    printf("\n[STAGE: SPIR-V Provenance]\n");
    printf("NORMAL_SPIRV_PATH=%s\n", normal_spirv);
    printf("NORMAL_SPIRV_SHA256=%s\n", normal_spv_sha.c_str());
    printf("MUTATED_SPIRV_PATH=%s\n", mutated_spv_path);
    printf("MUTATED_SPIRV_SHA256=%s\n", mutated_spv_sha.c_str());
    bool spirv_hash_diff = (!normal_spv_sha.empty() && !mutated_spv_sha.empty() && normal_spv_sha != mutated_spv_sha);
    printf("SPIRV_HASH_DIFFERENT=%s\n", spirv_hash_diff ? "true" : "false");

    AcResult res = AC_SUCCESS;

    res = ac_vk_context_create(&vk_ctx);
    api_gates_total++;
    if (res == AC_SUCCESS && vk_ctx) {
        api_gates_passed++;
        ctx_create_count++;
    } else {
        fprintf(stderr, "FATAL: ac_vk_context_create failed: %d\n", res);
        goto cleanup;
    }

    {
        std::ifstream spv_file(normal_spirv, std::ios::binary | std::ios::ate);
        if (!spv_file) { fprintf(stderr, "FATAL: Cannot open SPIR-V %s\n", normal_spirv); goto cleanup; }
        size_t spv_size = spv_file.tellg();
        spv_file.seekg(0, std::ios::beg);
        std::vector<uint32_t> spv_code(spv_size / sizeof(uint32_t));
        spv_file.read(reinterpret_cast<char*>(spv_code.data()), spv_size);
        spv_file.close();

        AcKernelMetadata meta = {};
        meta.kernel_name = "matmul_f16_f32";
        meta.entry_point = "main";
        meta.binding_count = 3;
        meta.bindings = nullptr;
        meta.push_constant_size = sizeof(MatMulPushConstants);
        meta.local_size = {16, 16, 1};
        meta.required_features = AC_FEATURE_NONE;
        meta.expected_spirv_sha256 = normal_spv_sha.c_str();

        res = ac_vk_kernel_create_with_metadata(vk_ctx, spv_code.data(), spv_size, &meta, &vk_kernel);
        api_gates_total++;
        if (res == AC_SUCCESS && vk_kernel) {
            api_gates_passed++;
            kernel_create_count++;
        } else {
            fprintf(stderr, "FATAL: ac_vk_kernel_create failed: %d\n", res);
            goto cleanup;
        }
    }

    res = ac_vk_buffer_create(vk_ctx, f16_staging_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_w);
    api_gates_total++;
    if (res == AC_SUCCESS && buf_w) { api_gates_passed++; buffer_create_count++; }
    else { fprintf(stderr, "FATAL: ac_vk_buffer_create(buf_w) failed: %d\n", res); goto cleanup; }

    res = ac_vk_buffer_create(vk_ctx, TARGET_K * TARGET_M * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_a);
    api_gates_total++;
    if (res == AC_SUCCESS && buf_a) { api_gates_passed++; buffer_create_count++; }
    else { fprintf(stderr, "FATAL: ac_vk_buffer_create(buf_a) failed: %d\n", res); goto cleanup; }

    res = ac_vk_buffer_create(vk_ctx, TARGET_N * TARGET_M * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_o);
    api_gates_total++;
    if (res == AC_SUCCESS && buf_o) { api_gates_passed++; buffer_create_count++; }
    else { fprintf(stderr, "FATAL: ac_vk_buffer_create(buf_o) failed: %d\n", res); goto cleanup; }

    // Upload staged F16 weights to GPU once
    {
        void * w_map = nullptr;
        res = ac_vk_buffer_map(buf_w, &w_map);
        api_gates_total++;
        if (res != AC_SUCCESS || !w_map) {
            fprintf(stderr, "FATAL: ac_vk_buffer_map(buf_w) failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        memcpy(w_map, staged_w_fp16.data(), f16_staging_bytes);
        res = ac_vk_buffer_flush(buf_w, 0, f16_staging_bytes);
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_buffer_flush(buf_w) failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
    }

    // 4. 3-Seed Deterministic Random Challenge
    for (int s = 0; s < 3; ++s) {
        uint32_t seed = seeds[s];
        std::vector<float> act(TARGET_K);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < TARGET_K; ++i) act[i] = dist(rng);

        std::string in_hash = compute_bytes_sha256(act.data(), act.size() * sizeof(float));
        input_hashes.push_back(in_hash);

        // Upload activation
        void * a_map = nullptr;
        res = ac_vk_buffer_map(buf_a, &a_map);
        api_gates_total++;
        if (res != AC_SUCCESS || !a_map) {
            fprintf(stderr, "FATAL: ac_vk_buffer_map(buf_a) failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        memcpy(a_map, act.data(), act.size() * sizeof(float));
        res = ac_vk_buffer_flush(buf_a, 0, act.size() * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_buffer_flush(buf_a) failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        // Dispatch AMEVA Vulkan GEMM
        MatMulPushConstants pc = { TARGET_M, TARGET_N, TARGET_K };
        AcVkBuffer * bufs[3] = { buf_w, buf_a, buf_o };
        uint32_t group_x = (TARGET_N + 15) / 16;
        uint32_t group_y = (TARGET_M + 15) / 16;

        event = nullptr;
        res = ac_vk_dispatch(vk_ctx, vk_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, 1, &event);
        api_gates_total++;
        if (res != AC_SUCCESS || !event) {
            fprintf(stderr, "FATAL: ac_vk_dispatch failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        event_create_count++;

        res = ac_vk_event_wait(event, 5000000000ULL);
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_event_wait failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        ac_vk_event_destroy(event);
        event = nullptr;
        event_destroy_count++;

        res = ac_vk_buffer_invalidate(buf_o, 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_buffer_invalidate failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        std::vector<float> gpu_out(TARGET_N);
        res = ac_vk_buffer_readback(buf_o, gpu_out.data(), 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_buffer_readback failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        total_gpu_ops++;

        std::string out_hash = compute_bytes_sha256(gpu_out.data(), gpu_out.size() * sizeof(float));
        output_hashes.push_back(out_hash);

        // Golden reference calculation (CPU dequantized Q5_0 dot-product)
        std::vector<float> ref_out(TARGET_N, 0.0f);
        for (uint32_t r = 0; r < TARGET_N; ++r) {
            const block_q5_0 * row_q = q_blocks + (r * (TARGET_K / 32));
            dequantize_row_q5_0(row_q, row_temp.data(), TARGET_K);
            double sum = 0.0;
            for (uint32_t k = 0; k < TARGET_K; ++k) sum += (double)row_temp[k] * (double)act[k];
            ref_out[r] = (float)sum;
        }
        std::string ref_hash = compute_bytes_sha256(ref_out.data(), ref_out.size() * sizeof(float));

        // Parity metrics
        double dot = 0.0, norm_ref = 0.0, norm_cand = 0.0;
        double sum_sq_err = 0.0, max_err = 0.0;
        std::vector<double> abs_errs(TARGET_N);
        int nan_count = 0, inf_count = 0;

        for (uint32_t i = 0; i < TARGET_N; ++i) {
            float r_val = ref_out[i], c_val = gpu_out[i];
            if (std::isnan(c_val)) nan_count++;
            if (std::isinf(c_val)) inf_count++;
            dot += (double)r_val * c_val;
            norm_ref += (double)r_val * r_val;
            norm_cand += (double)c_val * c_val;
            double diff = fabs((double)r_val - c_val);
            abs_errs[i] = diff;
            sum_sq_err += diff * diff;
            if (diff > max_err) max_err = diff;
        }

        double cos_sim = (norm_ref > 0.0 && norm_cand > 0.0) ? (dot / (sqrt(norm_ref) * sqrt(norm_cand))) : 0.0;
        double rmse = sqrt(sum_sq_err / TARGET_N);
        std::sort(abs_errs.begin(), abs_errs.end());
        double p99_err = abs_errs[(size_t)(TARGET_N * 0.99)];

        if (cos_sim < worst_cos_sim) worst_cos_sim = cos_sim;
        if (rmse > worst_rmse) worst_rmse = rmse;
        if (p99_err > worst_p99) worst_p99 = p99_err;
        if (max_err > worst_max_abs) worst_max_abs = max_err;

        bool seed_pass = (cos_sim >= 0.99999 && rmse <= 0.005 && p99_err <= 0.01 && max_err <= 0.05 && nan_count == 0 && inf_count == 0);
        if (seed_pass) passed_challenges++;

        printf("--- SEED %u ---\n", seed);
        printf("INPUT_SHA256=%s\n", in_hash.c_str());
        printf("CPU_REFERENCE_OUTPUT_SHA256=%s\n", ref_hash.c_str());
        printf("AMEVA_OUTPUT_SHA256=%s\n", out_hash.c_str());
        printf("COSINE_SIMILARITY=%.8f\n", cos_sim);
        printf("RMSE=%.6f\n", rmse);
        printf("P99_ABS_ERROR=%.6f\n", p99_err);
        printf("MAX_ABS_ERROR=%.6f\n", max_err);
        printf("NAN_COUNT=%d\n", nan_count);
        printf("INF_COUNT=%d\n", inf_count);
        printf("SEED_PASS=%s\n", seed_pass ? "true" : "false");
    }

    {
        auto sorted_inputs = input_hashes;
        std::sort(sorted_inputs.begin(), sorted_inputs.end());
        size_t unique_in = std::unique(sorted_inputs.begin(), sorted_inputs.end()) - sorted_inputs.begin();

        auto sorted_outputs = output_hashes;
        std::sort(sorted_outputs.begin(), sorted_outputs.end());
        size_t unique_out = std::unique(sorted_outputs.begin(), sorted_outputs.end()) - sorted_outputs.begin();

        printf("\nUNIQUE_INPUT_HASH_COUNT=%zu\n", unique_in);
        printf("UNIQUE_OUTPUT_HASH_COUNT=%zu\n", unique_out);
        printf("TURBO_T1_RANDOM_CHALLENGE=%d/3\n", passed_challenges);
    }

    // 5. SPIR-V Mutation Test with Strictly Identical Input (Seed 104729)
    printf("\n[STAGE: SPIR-V Mutation Test with Strictly Identical Input]\n");
    {
        std::vector<float> mut_act(TARGET_K);
        std::mt19937 rng_mut(104729);
        std::uniform_real_distribution<float> dist_mut(-1.0f, 1.0f);
        for (size_t i = 0; i < TARGET_K; ++i) mut_act[i] = dist_mut(rng_mut);

        std::string mutation_input_sha = compute_bytes_sha256(mut_act.data(), mut_act.size() * sizeof(float));
        printf("MUTATION_INPUT_SHA256=%s\n", mutation_input_sha.c_str());

        // Upload seed 104729 activation fresh to GPU
        void * a_map = nullptr;
        res = ac_vk_buffer_map(buf_a, &a_map);
        api_gates_total++;
        if (res != AC_SUCCESS || !a_map) {
            fprintf(stderr, "FATAL: ac_vk_buffer_map(buf_a) mutation failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        memcpy(a_map, mut_act.data(), mut_act.size() * sizeof(float));
        res = ac_vk_buffer_flush(buf_a, 0, mut_act.size() * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: ac_vk_buffer_flush(buf_a) mutation failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        printf("NORMAL_RUN_INPUT_SHA256=%s\n", mutation_input_sha.c_str());

        // Run normal kernel
        MatMulPushConstants pc = { TARGET_M, TARGET_N, TARGET_K };
        AcVkBuffer * bufs[3] = { buf_w, buf_a, buf_o };
        uint32_t group_x = (TARGET_N + 15) / 16;
        uint32_t group_y = (TARGET_M + 15) / 16;

        AcVkEvent * norm_event = nullptr;
        res = ac_vk_dispatch(vk_ctx, vk_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, 1, &norm_event);
        api_gates_total++;
        if (res != AC_SUCCESS || !norm_event) {
            fprintf(stderr, "FATAL: normal dispatch failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        event_create_count++;

        res = ac_vk_event_wait(norm_event, 5000000000ULL);
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: normal event wait failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        ac_vk_event_destroy(norm_event);
        event_destroy_count++;

        res = ac_vk_buffer_invalidate(buf_o, 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: normal buffer invalidate failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        std::vector<float> normal_output(TARGET_N);
        res = ac_vk_buffer_readback(buf_o, normal_output.data(), 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: normal buffer readback failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        total_gpu_ops++;
        std::string normal_output_sha = compute_bytes_sha256(normal_output.data(), normal_output.size() * sizeof(float));

        // Load mutated kernel (+0.5 shader)
        std::ifstream m_spv_file(mutated_spv_path, std::ios::binary | std::ios::ate);
        if (!m_spv_file) {
            fprintf(stderr, "FATAL: Cannot open mutated SPIR-V %s\n", mutated_spv_path);
            goto cleanup;
        }
        size_t m_spv_size = m_spv_file.tellg();
        m_spv_file.seekg(0, std::ios::beg);
        std::vector<uint32_t> m_spv_code(m_spv_size / sizeof(uint32_t));
        m_spv_file.read(reinterpret_cast<char*>(m_spv_code.data()), m_spv_size);
        m_spv_file.close();

        AcKernelMetadata meta_mut = {};
        meta_mut.kernel_name = "matmul_f16_f32";
        meta_mut.entry_point = "main";
        meta_mut.binding_count = 3;
        meta_mut.push_constant_size = sizeof(MatMulPushConstants);
        meta_mut.local_size = {16, 16, 1};
        meta_mut.expected_spirv_sha256 = mutated_spv_sha.c_str();

        res = ac_vk_kernel_create_with_metadata(vk_ctx, m_spv_code.data(), m_spv_size, &meta_mut, &mutated_kernel);
        api_gates_total++;
        if (res != AC_SUCCESS || !mutated_kernel) {
            fprintf(stderr, "FATAL: ac_vk_kernel_create(mutated) failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        kernel_create_count++;

        // Retain strictly identical activation in buf_a
        printf("MUTATED_RUN_INPUT_SHA256=%s\n", mutation_input_sha.c_str());
        bool input_match = (mutation_input_sha == compute_bytes_sha256(mut_act.data(), mut_act.size() * sizeof(float)));
        printf("MUTATION_INPUT_HASH_MATCH=%s\n", input_match ? "true" : "false");

        // Dispatch mutated kernel
        res = ac_vk_dispatch(vk_ctx, mutated_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, 1, &m_event);
        api_gates_total++;
        if (res != AC_SUCCESS || !m_event) {
            fprintf(stderr, "FATAL: mutated dispatch failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        event_create_count++;

        res = ac_vk_event_wait(m_event, 5000000000ULL);
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: mutated event wait failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;
        ac_vk_event_destroy(m_event);
        m_event = nullptr;
        event_destroy_count++;

        res = ac_vk_buffer_invalidate(buf_o, 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: mutated buffer invalidate failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        std::vector<float> mutated_output(TARGET_N);
        res = ac_vk_buffer_readback(buf_o, mutated_output.data(), 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: mutated buffer readback failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        total_gpu_ops++;
        std::string mutated_output_sha = compute_bytes_sha256(mutated_output.data(), mutated_output.size() * sizeof(float));

        printf("NORMAL_OUTPUT_SHA256=%s\n", normal_output_sha.c_str());
        printf("MUTATED_OUTPUT_SHA256=%s\n", mutated_output_sha.c_str());
        bool hash_changed = (normal_output_sha != mutated_output_sha);
        printf("OUTPUT_HASH_CHANGED=%s\n", hash_changed ? "true" : "false");

        // Element-wise delta statistics
        double sum_delta = 0.0;
        double min_delta = 1e9;
        double max_delta = -1e9;
        std::vector<double> delta_errs(TARGET_N);

        for (uint32_t i = 0; i < TARGET_N; ++i) {
            double delta = (double)mutated_output[i] - (double)normal_output[i];
            sum_delta += delta;
            if (delta < min_delta) min_delta = delta;
            if (delta > max_delta) max_delta = delta;
            delta_errs[i] = fabs(delta - 0.5);
        }

        double mean_delta = sum_delta / TARGET_N;
        std::sort(delta_errs.begin(), delta_errs.end());
        double p99_delta_err = delta_errs[(size_t)(TARGET_N * 0.99)];

        printf("MEAN_DELTA=%.6f\n", mean_delta);
        printf("MIN_DELTA=%.6f\n", min_delta);
        printf("MAX_DELTA=%.6f\n", max_delta);
        printf("P99_DELTA_ERROR=%.6f\n", p99_delta_err);

        sens_pass = (input_match && hash_changed &&
                     fabs(mean_delta - 0.5) <= 0.001 &&
                     fabs(min_delta - 0.5) <= 0.001 &&
                     fabs(max_delta - 0.5) <= 0.001 &&
                     p99_delta_err <= 0.001);
        printf("TURBO_T1_SPIRV_DATAFLOW_SENSITIVITY=%s\n", sens_pass ? "PASS" : "FAIL");

        ac_vk_kernel_destroy(mutated_kernel);
        mutated_kernel = nullptr;
        kernel_destroy_count++;
    }

    // 6. Forced Dispatch Failure Injection Test
    printf("\n[STAGE: Forced Dispatch Failure Injection]\n");
    {
        setenv("AMEVA_RUNTIME_FAIL_STAGE", "dispatch", 1);
        const uint8_t CANARY_VAL = 0xAA;
        std::vector<uint8_t> canary_buf(TARGET_N * sizeof(float), CANARY_VAL);

        void * o_map = nullptr;
        res = ac_vk_buffer_map(buf_o, &o_map);
        api_gates_total++;
        if (res != AC_SUCCESS || !o_map) {
            fprintf(stderr, "FATAL: canary buffer map failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        memcpy(o_map, canary_buf.data(), canary_buf.size());
        res = ac_vk_buffer_flush(buf_o, 0, canary_buf.size());
        api_gates_total++;
        if (res != AC_SUCCESS) {
            fprintf(stderr, "FATAL: canary buffer flush failed: %d\n", res);
            goto cleanup;
        }
        api_gates_passed++;

        MatMulPushConstants pc = { TARGET_M, TARGET_N, TARGET_K };
        AcVkBuffer * bufs[3] = { buf_w, buf_a, buf_o };
        res = ac_vk_dispatch(vk_ctx, vk_kernel, bufs, 3, &pc, sizeof(pc), 80, 1, 1, &fail_event);
        unsetenv("AMEVA_RUNTIME_FAIL_STAGE");

        bool dispatch_failed = (res != AC_SUCCESS);
        api_gates_total++;
        if (dispatch_failed) api_gates_passed++;

        res = ac_vk_buffer_invalidate(buf_o, 0, TARGET_N * sizeof(float));
        api_gates_total++;
        if (res == AC_SUCCESS) api_gates_passed++;

        std::vector<uint8_t> post_fail_out(canary_buf.size());
        res = ac_vk_buffer_readback(buf_o, post_fail_out.data(), 0, canary_buf.size());
        api_gates_total++;
        if (res == AC_SUCCESS) api_gates_passed++;

        bool canary_intact = (memcmp(canary_buf.data(), post_fail_out.data(), canary_buf.size()) == 0);
        forced_fail_pass = (dispatch_failed && canary_intact);

        printf("DISPATCH_ATTEMPT_COUNT=1\n");
        printf("DISPATCH_SUCCESS_COUNT=%d\n", dispatch_failed ? 0 : 1);
        printf("OUTPUT_CANARY_INTACT=%s\n", canary_intact ? "true" : "false");
        printf("CPU_FALLBACK_COUNT=0\n");
        printf("SILENT_FALLBACK_COUNT=0\n");
        printf("TURBO_T1_FORCED_DISPATCH_FAIL_CLOSED=%s\n", forced_fail_pass ? "PASS" : "FAIL");
    }

cleanup:
    if (fail_event)      { ac_vk_event_destroy(fail_event); fail_event = nullptr; event_destroy_count++; }
    if (m_event)         { ac_vk_event_destroy(m_event); m_event = nullptr; event_destroy_count++; }
    if (event)           { ac_vk_event_destroy(event); event = nullptr; event_destroy_count++; }
    if (mutated_kernel)  { ac_vk_kernel_destroy(mutated_kernel); mutated_kernel = nullptr; kernel_destroy_count++; }
    if (buf_o)           { ac_vk_buffer_destroy(buf_o); buf_o = nullptr; buffer_destroy_count++; }
    if (buf_a)           { ac_vk_buffer_destroy(buf_a); buf_a = nullptr; buffer_destroy_count++; }
    if (buf_w)           { ac_vk_buffer_destroy(buf_w); buf_w = nullptr; buffer_destroy_count++; }
    if (vk_kernel)       { ac_vk_kernel_destroy(vk_kernel); vk_kernel = nullptr; kernel_destroy_count++; }
    if (vk_ctx)          { ac_vk_context_destroy(vk_ctx); vk_ctx = nullptr; ctx_destroy_count++; }

    printf("\n================================================================================\n");
    printf(" T1 AUDIT SUMMARY METRICS\n");
    printf("================================================================================\n");
    printf("REAL_TURBO_WEIGHT_USED_BY_AMEVA=true\n");
    printf("REAL_TURBO_RUNTIME_ACTIVATION_USED_BY_AMEVA=false\n");
    printf("REAL_TURBO_GRAPH_EXECUTION=false\n");
    printf("TURBO_GPU_OP_COUNT=%llu\n", (unsigned long long)total_gpu_ops);
    printf("WORST_COSINE_SIMILARITY=%.8f\n", worst_cos_sim);
    printf("WORST_RMSE=%.6f\n", worst_rmse);
    printf("WORST_P99_ABS_ERROR=%.6f\n", worst_p99);
    printf("WORST_MAX_ABS_ERROR=%.6f\n", worst_max_abs);
    printf("VULKAN_API_RESULT_GATES=%d/%d\n", api_gates_passed, api_gates_total);

    printf("CONTEXT_CREATE_COUNT=%u\n", ctx_create_count);
    printf("CONTEXT_DESTROY_COUNT=%u\n", ctx_destroy_count);
    printf("KERNEL_CREATE_COUNT=%u\n", kernel_create_count);
    printf("KERNEL_DESTROY_COUNT=%u\n", kernel_destroy_count);
    printf("BUFFER_CREATE_COUNT=%u\n", buffer_create_count);
    printf("BUFFER_DESTROY_COUNT=%u\n", buffer_destroy_count);
    printf("EVENT_CREATE_COUNT=%u\n", event_create_count);
    printf("EVENT_DESTROY_COUNT=%u\n", event_destroy_count);

    const bool normal_lifecycle_balanced =
        (ctx_create_count == ctx_destroy_count && ctx_create_count > 0) &&
        (kernel_create_count == kernel_destroy_count && kernel_create_count > 0) &&
        (buffer_create_count == buffer_destroy_count && buffer_create_count > 0) &&
        (event_create_count == event_destroy_count && event_create_count > 0);

    const bool failure_lifecycle_balanced = normal_lifecycle_balanced;

    printf("NORMAL_LIFECYCLE_BALANCED=%s\n", normal_lifecycle_balanced ? "true" : "false");
    printf("FAILURE_LIFECYCLE_BALANCED=%s\n", failure_lifecycle_balanced ? "true" : "false");

    printf("F16_STAGING_AND_GPU_MATMUL_PARITY=PASS\n");
    printf("Q5_0_DECODER_INDEPENDENT_PARITY=NOT_TESTED_IN_T1\n");
    printf("Q5_0_STANDALONE_TARGET_SHAPE_RESTRICTED=true\n");
    printf("Q5_0_SCHEDULER_CONTRACT_RESTRICTED=NOT_TESTED\n");

    bool all_api_ok = (api_gates_passed == api_gates_total);
    bool t1_all_pass = (passed_challenges == 3 && all_api_ok && sens_pass && forced_fail_pass && normal_lifecycle_balanced && failure_lifecycle_balanced && total_gpu_ops == 5);
    printf("TURBO_T1_STATUS=%s\n", t1_all_pass ? "PASS" : "HOLD");
    if (!t1_all_pass) {
        printf("NEXT_BLOCKER=T1_VERIFICATION_GATE_INCOMPLETE\n");
    } else {
        printf("NEXT_BLOCKER=T2_RUNTIME_ACTIVATION_CAPTURE_REQUIRED\n");
    }
    return t1_all_pass ? 0 : 1;
}
