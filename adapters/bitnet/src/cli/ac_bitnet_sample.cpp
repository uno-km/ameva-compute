/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// ac_bitnet_sample: what one device answers to one prompt, with random sampling instead of the greedy choice.
//
//   ac_bitnet_sample -m MODEL [-p PROMPT] [-n GEN] [-k SAMPLES] [--temp T] [--top-k K] [--top-p P] [--seed S]
//                    [--backend cpu|gpu] [-c N_CTX] [-t THREADS]
//
// Greedy decoding (temperature 0) takes the arg-max token at every step, so every device that computes the same
// logits to within its rounding prints the same text; that is what ac_bitnet_parity checks. This tool shows the
// distribution instead: the top-5 next tokens after the prompt with their probabilities (at temperature 1 and at --temp;
// the logits come from this device's own computation, so the digits differ slightly between devices), then SAMPLES
// continuations drawn with seeds S, S+1, ... (--seed 0, the default, takes S from std::random_device and prints it, so
// any line can be reproduced with --seed). The same seed on two devices draws the same random numbers, but the
// continuations can still part where two tokens are nearly tied and the devices' logits differ in the last digits.
// Sampling: temperature 1.0, top-k 40, top-p 0.95 by default (ac-bitnet-cli: temperature 0.7). The library's sampler
// applies temperature, top-k (0 means 40) and top-p only; the repetition penalty keeps the library default. Exit status 0 when every sample produced at least one token.
#include "termux_bitnet.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::string piece(bitnet_context_t ctx, int tok) {
    char buf[256] = {};
    bitnet_token_to_str(ctx, tok, buf, sizeof(buf));
    std::string s = buf;
    for (char& c : s)
        if (c == '\n') c = ' ';
    return s;
}

bool collect(const char* token_str, int32_t token_id, void* user_data) {
    (void)token_id;
    static_cast<std::string*>(user_data)->append(token_str ? token_str : "");
    return true;
}

// probabilities of softmax(logits / temp), in double
std::vector<double> softmax(const float* logits, int n, double temp) {
    std::vector<double> p(n);
    const double m = *std::max_element(logits, logits + n);
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[i] = std::exp(((double)logits[i] - m) / temp);
    for (double& v : p) v /= z;
    return p;
}

} // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    std::string prompt = "The capital of France is";
    std::string backend = "gpu";
    int gen = 24, samples = 3, n_ctx = 512, threads = 4, top_k = 40;
    float temp = 1.0f, top_p = 0.95f;
    uint32_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") gen = atoi(next());
        else if (a == "-k") samples = atoi(next());
        else if (a == "-c") n_ctx = atoi(next());
        else if (a == "-t") threads = atoi(next());
        else if (a == "--temp") temp = (float)atof(next());
        else if (a == "--top-k") top_k = atoi(next());
        else if (a == "--top-p") top_p = (float)atof(next());
        else if (a == "--seed") seed = (uint32_t)strtoul(next(), nullptr, 10);
        else if (a == "--backend") backend = next();
        else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (!model || (backend != "cpu" && backend != "gpu") || temp <= 0.0f || samples < 1 || gen < 1) {
        fprintf(stderr, "usage: ac_bitnet_sample -m MODEL [-p PROMPT] [-n GEN] [-k SAMPLES] [--temp T>0] [--top-k K]\n"
                        "       [--top-p P] [--seed S] [--backend cpu|gpu] [-c N_CTX] [-t THREADS]\n");
        return 2;
    }
    if (seed == 0) {
        std::random_device rd;
        do seed = rd(); while (seed == 0);
    }

    bitnet_params_t p = bitnet_default_params();
    p.model_path = model;
    p.n_ctx = n_ctx;
    p.n_threads = threads;
    p.n_gpu_layers = backend == "gpu" ? 9999 : 0;
    p.temperature = temp;
    p.top_k = top_k;
    p.top_p = top_p;
    p.seed = seed;
    p.verbose = false;
    bitnet_context_t ctx = bitnet_init(&p);
    if (!ctx) {
        fprintf(stderr, "bitnet_init failed for %s (backend %s)\n", model, backend.c_str());
        return 1;
    }
    char hw[512] = {};
    bitnet_get_hardware_info(hw, sizeof(hw));
    printf("backend %s | %s\n", backend.c_str(), hw);
    printf("prompt \"%s\" | temp %.2f top-k %d top-p %.2f repeat-penalty %.2f | %d samples of %d tokens\n",
           prompt.c_str(), temp, top_k, top_p, p.repeat_penalty, samples, gen);

    // the next-token distribution after the prompt
    std::vector<int32_t> toks(n_ctx);
    const int n = bitnet_tokenize(ctx, prompt.c_str(), toks.data(), (int32_t)toks.size());
    if (n <= 0 || bitnet_eval(ctx, toks.data(), n) != 0) {
        fprintf(stderr, "prompt evaluation failed (%d tokens)\n", n);
        bitnet_free(ctx);
        return 1;
    }
    int32_t n_vocab = 0;
    const float* lg = bitnet_get_logits(ctx, &n_vocab);
    if (!lg || n_vocab <= 0) {
        fprintf(stderr, "no logits after the prompt\n");
        bitnet_free(ctx);
        return 1;
    }
    const std::vector<double> p1 = softmax(lg, n_vocab, 1.0), pt = softmax(lg, n_vocab, temp);
    std::vector<int> order(n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    printf("next token after the prompt (%d prompt tokens):\n", n);
    for (int r = 0; r < 5; ++r) {
        const int t = order[r];
        printf("  %d. %-16s logit %9.5f  p(T=1) %.6f  p(T=%.2f) %.6f\n", r + 1, ("\"" + piece(ctx, t) + "\"").c_str(),
               lg[t], p1[t], temp, pt[t]);
    }

    int bad = 0;
    for (int s = 0; s < samples; ++s) {
        bitnet_reset(ctx);
        bitnet_params_t q = p;
        q.seed = seed + (uint32_t)s;
        if (q.seed == 0) q.seed = 1;
        bitnet_set_params(ctx, &q);
        std::string text;
        const int got = bitnet_generate_stream(ctx, prompt.c_str(), gen, collect, &text);
        double pe = 0, ev = 0, tps = 0;
        bitnet_get_perf_stats(ctx, &pe, &ev, &tps);
        for (char& c : text)
            if (c == '\n') c = ' ';
        printf("sample %d seed %u (%d tokens, %.1f t/s): %s%s\n", s + 1, q.seed, got, tps, prompt.c_str(), text.c_str());
        bad += got <= 0;
    }
    bitnet_free(ctx);
    printf("RESULT %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
