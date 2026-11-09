// Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 OR Commercial-License
//
// ameva_split_gen: prompt on the GPU, generation on the CPU (research question RQ-25), through the public llama.h state
// API: the sequence state of the context that ran the prompt is copied into a context of a second, CPU-only model.
//   ameva_split_gen MODEL.gguf PROMPT_FILE MODE N THREADS [GPU_DEVICE]
//   MODE split: prompt on GPU_DEVICE (default AMEVA0), state transfer, N tokens on the CPU with THREADS threads
//   MODE gpu:   prompt and generation on GPU_DEVICE
//   MODE cpu:   prompt and generation on the CPU
//   MODE xfer:  prompt on the CPU, state transfer to a second CPU context, generation there (the transfer itself must
//               be lossless: the generated tokens equal those of MODE cpu)
// Greedy decoding, context 2048, batch 512, flash attention on. One result line:
//   SPLIT mode= prompt_tokens= pp_tps= state_bytes= state_read_ms= state_write_ms= tg_tps= gen_hash= vmhwm_mb=
// tg_tps counts the N-1 tokens decoded after the first sampled token (the same convention as llama-completion). Any
// failure (device missing, decode error, state not accepted) prints the reason on stderr and exits with 2.
#include "ggml-backend.h"
#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

[[noreturn]] static void die(const char* what, long code = 0) {
    fprintf(stderr, "ameva_split_gen: %s (%ld)\n", what, code);
    exit(2);
}

static long vmhwm_mb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmHWM: %ld", &kb) == 1) break;
    }
    fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

static llama_context* make_ctx(llama_model* m, int threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* c = llama_init_from_model(m, cp);
    if (!c) die("llama_init_from_model failed");
    return c;
}

static llama_token argmax(llama_context* c, int n_vocab) {
    const float* lg = llama_get_logits_ith(c, -1);
    if (!lg) die("no logits");
    llama_token best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (lg[i] > lg[best]) best = i;
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 6) die("usage: MODEL PROMPT_FILE split|gpu|cpu N THREADS [GPU_DEVICE]");
    const char* model_path = argv[1];
    const std::string mode = argv[3];
    const int n_gen = atoi(argv[4]);
    const int threads = atoi(argv[5]);
    const char* gpu_name = argc > 6 ? argv[6] : "AMEVA0";
    const bool prompt_gpu = mode == "split" || mode == "gpu";
    const bool gen_gpu = mode == "gpu";
    const bool transfer = mode == "split" || mode == "xfer";
    if (!prompt_gpu && mode != "cpu" && mode != "xfer") die("mode must be split, gpu, cpu or xfer");

    FILE* pf = fopen(argv[2], "rb");
    if (!pf) die("cannot open the prompt file");
    std::string prompt;
    char buf[4096];
    size_t got;
    while ((got = fread(buf, 1, sizeof(buf), pf)) > 0) prompt.append(buf, got);
    fclose(pf);

    llama_backend_init();
    ggml_backend_load_all();

    ggml_backend_dev_t gpu = nullptr;
    if (prompt_gpu) {
        gpu = ggml_backend_dev_by_name(gpu_name);
        if (!gpu) die("GPU device not found");
    }
    ggml_backend_dev_t gpu_devs[2] = {gpu, nullptr};
    ggml_backend_dev_t no_devs[1] = {nullptr};

    llama_model* model_gpu = nullptr;
    llama_model* model_cpu = nullptr;
    if (prompt_gpu) {
        llama_model_params mp = llama_model_default_params();
        mp.devices = gpu_devs;
        mp.n_gpu_layers = 999;
        model_gpu = llama_model_load_from_file(model_path, mp);
        if (!model_gpu) die("GPU model load failed");
    }
    if (!gen_gpu) {
        llama_model_params mp = llama_model_default_params();
        mp.devices = no_devs;
        mp.n_gpu_layers = 0;
        model_cpu = llama_model_load_from_file(model_path, mp);
        if (!model_cpu) die("CPU model load failed");
    }
    llama_model* first = prompt_gpu ? model_gpu : model_cpu;
    const llama_vocab* vocab = llama_model_get_vocab(first);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    const int n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), toks.data(), (int32_t)toks.size(), true, true);
    if (n <= 0) die("tokenize failed", n);
    toks.resize((size_t)n);

    llama_context* cp = make_ctx(first, threads);
    double t0 = now_ms();
    for (int i = 0; i < n; i += 512) {
        const int c = n - i < 512 ? n - i : 512;
        const int32_t e = llama_decode(cp, llama_batch_get_one(toks.data() + i, c));
        if (e) die("prompt decode failed", e);
    }
    const double prompt_ms = now_ms() - t0;
    llama_token tok = argmax(cp, n_vocab);

    llama_context* cg = cp;
    size_t state_bytes = 0;
    double read_ms = 0, write_ms = 0;
    if (transfer) {
        cg = make_ctx(model_cpu, threads);
        const size_t sz = llama_state_seq_get_size(cp, 0);
        if (!sz) die("state size is zero");
        std::vector<uint8_t> st(sz);
        t0 = now_ms();
        state_bytes = llama_state_seq_get_data(cp, st.data(), st.size(), 0);
        read_ms = now_ms() - t0;
        if (!state_bytes) die("state read failed");
        t0 = now_ms();
        if (llama_state_seq_set_data(cg, st.data(), state_bytes, 0) != state_bytes) die("state write failed");
        write_ms = now_ms() - t0;
        llama_free(cp);
    }

    uint64_t h = 1469598103934665603ull;
    int produced = 0;
    std::vector<llama_token> generated;
    t0 = now_ms();
    for (int i = 1; i < n_gen; ++i) {
        h = (h ^ (uint64_t)(uint32_t)tok) * 1099511628211ull;
        generated.push_back(tok);
        if (llama_vocab_is_eog(vocab, tok)) break;
        const int32_t e = llama_decode(cg, llama_batch_get_one(&tok, 1));
        if (e) die("generation decode failed", e);
        tok = argmax(cg, n_vocab);
        ++produced;
    }
    const double gen_ms = now_ms() - t0;
    // SPLIT_TOKENS=<file>: the generated token ids, one per line, and the text after them (first divergence between modes)
    if (const char* tf = getenv("SPLIT_TOKENS")) {
        FILE* f = fopen(tf, "w");
        if (!f) die("cannot write SPLIT_TOKENS");
        for (llama_token t : generated) {
            char piece[64];
            const int len = llama_token_to_piece(vocab, t, piece, sizeof(piece) - 1, 0, true);
            if (len < 0) die("llama_token_to_piece failed", len);
            piece[len] = 0;
            for (int j = 0; j < len; ++j) if (piece[j] == '\n') piece[j] = ' ';
            fprintf(f, "%d\t%s\n", t, piece);
        }
        fclose(f);
    }
    printf("SPLIT mode=%s prompt_tokens=%d pp_tps=%.2f state_bytes=%zu state_read_ms=%.1f state_write_ms=%.1f tg_tps=%.2f produced=%d gen_hash=%016llx vmhwm_mb=%ld\n",
           mode.c_str(), n, n * 1000.0 / prompt_ms, state_bytes, read_ms, write_ms, produced ? produced * 1000.0 / gen_ms : 0.0,
           produced, (unsigned long long)h, vmhwm_mb());
    return 0;
}
