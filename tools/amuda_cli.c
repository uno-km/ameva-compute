/**
 * @file amuda_cli.c
 * @brief Standalone Native GPGPU Autoregressive Inference CLI Engine for BitNet 1.58b
 */

#include "amuda.h"
#include "amuda_ops.h"
#include "amuda_gguf.h"
#include "amuda_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_VOCAB 131072

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)(h & 0x7FFF) << 13;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;

    if (exp == 0x1F) {
        w = 0x7F800000 | ((uint32_t)(h & 0x03FF) << 13);
    } else if (exp == 0) {
        if (h & 0x03FF) {
            float f = (float)(h & 0x03FF) / 16777216.0f;
            return (h & 0x8000) ? -f : f;
        } else {
            w = 0;
        }
    } else {
        w += (exp - 15 + 127) << 23;
    }

    uint32_t res = sign | w;
    float f;
    memcpy(&f, &res, sizeof(float));
    return f;
}

// Simple vocabulary holder
typedef struct {
    uint32_t count;
    char**   tokens;
} Vocab;

static Vocab* load_vocab(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    uint32_t magic, ver;
    uint64_t tensor_cnt, kv_cnt;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&tensor_cnt, 8, 1, f) != 1 || fread(&kv_cnt, 8, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    Vocab* v = (Vocab*)calloc(1, sizeof(Vocab));
    if (!v) { fclose(f); return NULL; }

    for (uint64_t i = 0; i < kv_cnt; i++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, f) != 1 || klen == 0 || klen > 256) break;
        char key[128] = {0};
        size_t rlen = (klen < sizeof(key) - 1) ? klen : sizeof(key) - 1;
        if (fread(key, 1, rlen, f) != rlen) break;
        if (klen > rlen) fseek(f, (long)(klen - rlen), SEEK_CUR);

        uint32_t vtype = 0;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            uint32_t atype = 0;
            uint64_t acount = 0;
            if (fread(&atype, 4, 1, f) != 1 || fread(&acount, 8, 1, f) != 1 || acount > 262144) break;
            v->count = (uint32_t)acount;
            v->tokens = (char**)calloc(acount, sizeof(char*));

            for (uint64_t idx = 0; idx < acount; idx++) {
                uint64_t slen = 0;
                if (fread(&slen, 8, 1, f) != 1 || slen > 1024) break;
                v->tokens[idx] = (char*)malloc(slen + 1);
                if (v->tokens[idx] && fread(v->tokens[idx], 1, slen, f) == slen) {
                    v->tokens[idx][slen] = '\0';
                } else if (v->tokens[idx]) {
                    v->tokens[idx][0] = '\0';
                }
            }
            break;
        } else {
            // Skip other KV types
            if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
            else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
            else if (vtype >= 4 && vtype <= 6) fseek(f, 4, SEEK_CUR);
            else if (vtype >= 10 && vtype <= 12) fseek(f, 8, SEEK_CUR);
            else if (vtype == 8) {
                uint64_t slen = 0;
                if (fread(&slen, 8, 1, f) == 1 && slen <= 65536) fseek(f, (long)slen, SEEK_CUR);
                else break;
            } else if (vtype == 9) {
                uint32_t at = 0; uint64_t al = 0;
                if (fread(&at, 4, 1, f) == 1 && fread(&al, 8, 1, f) == 1 && al <= 1000000) {
                    size_t item_sz = (at <= 1 || at == 7) ? 1 : (at <= 3 ? 2 : (at <= 6 ? 4 : 8));
                    if (at == 8) {
                        for (uint64_t k = 0; k < al; k++) {
                            uint64_t sl = 0;
                            if (fread(&sl, 8, 1, f) == 1 && sl <= 1024) fseek(f, (long)sl, SEEK_CUR);
                            else break;
                        }
                    } else {
                        fseek(f, (long)(al * item_sz), SEEK_CUR);
                    }
                } else break;
            }
        }
    }
    fclose(f);
    return v;
}

static void free_vocab(Vocab* v) {
    if (!v) return;
    if (v->tokens) {
        for (uint32_t i = 0; i < v->count; i++) {
            if (v->tokens[i]) free(v->tokens[i]);
        }
        free(v->tokens);
    }
    free(v);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char* default_path = "/data/data/com.termux/files/home/.cache/termux-bitnet/models/bitnet-2b-ggml-model-i2_s.gguf";
    const char* model_path = (argc > 1) ? argv[1] : default_path;
    int max_tokens = (argc > 2) ? atoi(argv[2]) : 8;

    printf("===================================================================\n");
    printf("  AMUDA Native Autoregressive GPGPU Token Generator                \n");
    printf("  Model: %s | Max Tokens: %d\n", model_path, max_tokens);
    printf("===================================================================\n");

    // 1. Load GGUF Model & Vocabulary
    printf("[1/5] Mapping GGUF Model & Loading Vocabulary...\n");
    GgufModel* model = NULL;
    AcResult res = ac_gguf_open(model_path, &model);
    if (res != AC_SUCCESS || !model) {
        fprintf(stderr, "[FATAL] GGUF Open failed: %d\n", res);
        return 1;
    }

    Vocab* vocab = load_vocab(model_path);
    if (!vocab || vocab->count == 0) {
        fprintf(stderr, "[FATAL] Failed to load vocabulary!\n");
        ac_gguf_close(model);
        return 1;
    }
    printf("  Loaded %u vocabulary tokens.\n", vocab->count);

    uint32_t n_layers = model->config.block_count; // 30
    uint32_t hidden_dim = model->config.embedding_length; // 2560
    uint32_t ffn_dim = model->config.feed_forward_length; // 6912
    uint32_t num_heads = model->config.head_count; // 20
    uint32_t num_kv_heads = model->config.head_count_kv; // 5
    uint32_t head_dim = hidden_dim / num_heads; // 128
    float eps = model->config.rms_norm_eps > 0.0f ? model->config.rms_norm_eps : 1e-5f;

    // 2. Initialize AMUDA Compute Context
    printf("[2/5] Initializing AMUDA Hardware GPGPU Driver...\n");
    AcContext* ctx = NULL;
    res = ac_init(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "[FATAL] ac_init failed: %d\n", res);
        free_vocab(vocab);
        ac_gguf_close(model);
        return 1;
    }
    printf("  Device: %s (%s, %u CUs)\n", ctx->device_name, ctx->vendor_name, ctx->max_compute_units);

    // 3. Bind 30 Transformer Layers to GPU Memory
    printf("[3/5] Binding %u Transformer Blocks to GPU Memory...\n", n_layers);
    uint64_t t_load_start = get_time_ns();
    BitNetLayer** layers = (BitNetLayer**)calloc(n_layers, sizeof(BitNetLayer*));
    for (uint32_t l = 0; l < n_layers; l++) {
        res = amuda_layer_init(ctx, model, l, &layers[l]);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "[FATAL] Failed to initialize Layer %u: %d\n", l, res);
            return 1;
        }
        if ((l + 1) % 10 == 0 || l == n_layers - 1) {
            printf("  - Bound Layer %u/%u\n", l + 1, n_layers);
        }
    }
    uint64_t t_load_ms = (get_time_ns() - t_load_start) / 1000000;
    printf("  All %u Layers loaded in %llu ms!\n", n_layers, (unsigned long long)t_load_ms);

    // Shared scratch buffers
    BitNetScratch* scratch = NULL;
    amuda_scratch_alloc(ctx, hidden_dim, ffn_dim, num_heads, num_kv_heads, head_dim, &scratch);

    // Ping-pong hidden state buffers
    AcBuffer *buf_h0 = NULL, *buf_h1 = NULL, *buf_out_normed = NULL;
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &buf_h0);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &buf_h1);
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &buf_out_normed);

    // Output norm weight buffer
    GgufTensor* out_norm_tensor = ac_gguf_get_tensor(model, "output_norm.weight");
    AcBuffer* buf_w_out_norm = NULL;
    ac_buffer_alloc(ctx, hidden_dim * sizeof(float), &buf_w_out_norm);
    memcpy(buf_w_out_norm->host_ptr, out_norm_tensor->data_ptr, hidden_dim * sizeof(float));
    ac_buffer_sync_to_device(ctx, buf_w_out_norm);

    // Token embedding tensor (F16)
    GgufTensor* token_embd = ac_gguf_get_tensor(model, "token_embd.weight");
    const uint16_t* embd_table = (const uint16_t*)token_embd->data_ptr;

    // 4. Start Autoregressive Generation
    printf("\n[4/5] Starting Token Generation (Prompt BOS: 128000)...\n");
    printf("-------------------------------------------------------------------\n");
    printf("Response: ");
    fflush(stdout);

    uint32_t current_token = 128000; // BOS token
    float* xin = (float*)buf_h0->host_ptr;

    uint64_t total_gen_ns = 0;
    int generated_count = 0;

    for (int step = 0; step < max_tokens; step++) {
        uint64_t t_step_start = get_time_ns();

        // [A] Embedding Lookup: copy token vector to buf_h0
        const uint16_t* embd_ptr = embd_table + (size_t)current_token * hidden_dim;
        for (uint32_t d = 0; d < hidden_dim; d++) {
            xin[d] = fp16_to_fp32(embd_ptr[d]);
        }
        ac_buffer_sync_to_device(ctx, buf_h0);

        // [B] 30 Transformer Blocks Forward Loop
        AcBuffer* cur_in = buf_h0;
        AcBuffer* cur_out = buf_h1;

        for (uint32_t l = 0; l < n_layers; l++) {
            amuda_layer_forward(ctx, layers[l], scratch, cur_in, cur_out, step, eps);
            // Ping-pong swap
            AcBuffer* tmp = cur_in;
            cur_in = cur_out;
            cur_out = tmp;
        }

        // cur_in now holds the final layer output
        // [C] Final RMSNorm
        ac_ops_rmsnorm(ctx, 1, hidden_dim, eps, cur_in, buf_w_out_norm, buf_out_normed, NULL);
        ac_buffer_sync_to_host(ctx, buf_out_normed);
        float* final_h = (float*)buf_out_normed->host_ptr;

        // [D] LM Head Logits (Dot product against token embeddings & Argmax)
        float max_logit = -1e9f;
        uint32_t next_token = 0;

        // Vectorized search over vocabulary (top tokens)
        uint32_t search_vocab = (vocab->count < 32768) ? vocab->count : 32768;
        for (uint32_t v_idx = 0; v_idx < search_vocab; v_idx++) {
            const uint16_t* row_f16 = embd_table + (size_t)v_idx * hidden_dim;
            float dot = 0.0f;
            // Step by 4 for SIMD speedup
            for (uint32_t d = 0; d < hidden_dim; d += 4) {
                dot += final_h[d]     * fp16_to_fp32(row_f16[d]);
                dot += final_h[d + 1] * fp16_to_fp32(row_f16[d + 1]);
                dot += final_h[d + 2] * fp16_to_fp32(row_f16[d + 2]);
                dot += final_h[d + 3] * fp16_to_fp32(row_f16[d + 3]);
            }
            if (dot > max_logit) {
                max_logit = dot;
                next_token = v_idx;
            }
        }

        uint64_t t_step_dur = get_time_ns() - t_step_start;
        total_gen_ns += t_step_dur;
        generated_count++;

        // Print token string
        if (next_token < vocab->count && vocab->tokens[next_token]) {
            printf("%s", vocab->tokens[next_token]);
            fflush(stdout);
        } else {
            printf("[%u]", next_token);
            fflush(stdout);
        }

        // Check for EOS (128001)
        if (next_token == 128001) {
            break;
        }

        current_token = next_token;
    }

    printf("\n-------------------------------------------------------------------\n");
    double total_sec = (double)total_gen_ns / 1e9;
    double tps = (double)generated_count / total_sec;
    printf("[5/5] Generation Statistics:\n");
    printf("  Generated Tokens: %d\n", generated_count);
    printf("  Total Latency:    %.2f ms (%.3f s)\n", (double)total_gen_ns / 1e6, total_sec);
    printf("  Throughput (TPS): %.2f tokens/second\n", tps);
    printf("  Avg per Token:    %.2f ms/token\n", (double)total_gen_ns / 1e6 / generated_count);
    printf("===================================================================\n");

    // Cleanup
    ac_buffer_free(ctx, buf_h0);
    ac_buffer_free(ctx, buf_h1);
    ac_buffer_free(ctx, buf_out_normed);
    ac_buffer_free(ctx, buf_w_out_norm);
    amuda_scratch_free(ctx, scratch);
    for (uint32_t l = 0; l < n_layers; l++) {
        amuda_layer_free(ctx, layers[l]);
    }
    free(layers);
    free_vocab(vocab);
    ac_destroy(ctx);
    ac_gguf_close(model);
    return 0;
}
