/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// ac_bitnet_parity: the AMUDA GPU engine against the CPU path of the same library, on one model.
//
//   ac_bitnet_parity -m MODEL [-p PROMPT] [-n GEN] [-c N_CTX] [-t THREADS] [--kld-max X] [--top1-min F]
//   ac_bitnet_parity -m MODEL --save FILE --backend cpu|gpu [-p PROMPT] [-n GEN] ...
//   ac_bitnet_parity -m MODEL --load FILE --backend cpu|gpu ...
//
// --save: one context evaluates the prompt and GEN greedy tokens of its own; the tokens and every position's logits
// are written to FILE. --load: one context evaluates the tokens of FILE and its logits are compared with FILE's
// (another build, another device or the other backend: e.g. the float-order noise floor of the CPU path, NEON build
// against scalar build).
//
// Two contexts (CPU: -ngl 0, GPU: every layer) evaluate the same tokens one at a time: the prompt, then GEN tokens
// chosen greedily by the CPU (both contexts are fed the CPU's choice, so one disagreement does not change the inputs
// of the later positions). For every position: KL(CPU || GPU) of the softmax of the logits, whether the arg-max agrees,
// and the largest absolute logit difference. Then each context generates GEN tokens greedily on its own and the two
// token sequences are compared (length of the common prefix).
// Exit status 0 only when max KLD <= --kld-max (default 1e-3) and the arg-max agreement >= --top1-min (default 0.99).
#include "termux_bitnet.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int argmax(const float* v, int n) {
    return (int)(std::max_element(v, v + n) - v);
}

// KL(p || q) of softmax(a), softmax(b), in double
double kld(const float* a, const float* b, int n) {
    const double ma = *std::max_element(a, a + n), mb = *std::max_element(b, b + n);
    double za = 0, zb = 0;
    for (int i = 0; i < n; ++i) {
        za += std::exp((double)a[i] - ma);
        zb += std::exp((double)b[i] - mb);
    }
    const double la = std::log(za), lb = std::log(zb);
    double k = 0;
    for (int i = 0; i < n; ++i) {
        const double lpa = (double)a[i] - ma - la, lpb = (double)b[i] - mb - lb;
        k += std::exp(lpa) * (lpa - lpb);
    }
    return k;
}

bitnet_context_t open_ctx(const char* model, int n_ctx, int threads, int ngl) {
    bitnet_params_t p = bitnet_default_params();
    p.model_path = model;
    p.n_ctx = n_ctx;
    p.n_threads = threads;
    p.n_gpu_layers = ngl;
    p.temperature = 0.0f;
    p.verbose = false;
    return bitnet_init(&p);
}

std::string piece(bitnet_context_t ctx, int tok) {
    char buf[256] = {};
    bitnet_token_to_str(ctx, tok, buf, sizeof(buf));
    return buf;
}

struct Stats {
    double kmax = 0, ksum = 0, dmax = 0;
    int agree = 0, total = 0, kmax_pos = -1;
    void add(int pos, int tok, const float* ref, const float* got, int n_vocab, bool always_print) {
        const double k = kld(ref, got, n_vocab);
        const int ar = argmax(ref, n_vocab), ag = argmax(got, n_vocab);
        double d = 0;
        for (int i = 0; i < n_vocab; ++i) d = std::max(d, (double)std::fabs(ref[i] - got[i]));
        ksum += k;
        if (k > kmax) {
            kmax = k;
            kmax_pos = pos;
        }
        dmax = std::max(dmax, d);
        agree += ar == ag;
        ++total;
        if (always_print || ar != ag) {
            printf("pos %3d tok %6d  kld %.3e  maxdiff %.4f  argmax ref %6d got %6d%s\n", pos, tok, k, d, ar, ag,
                   ar == ag ? "" : "  DIFFER");
        }
    }
    void print(const char* what) const {
        printf("%s positions %d: kld mean %.3e max %.3e (pos %d), max |logit diff| %.4f, argmax agree %d/%d\n", what,
               total, ksum / total, kmax, kmax_pos, dmax, agree, total);
    }
};

} // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    std::string prompt = "The capital of France is";
    int gen = 32, n_ctx = 512, threads = 4;
    double kld_max = 1e-3, top1_min = 0.99;
    const char* save = nullptr;
    const char* load = nullptr;
    std::string backend;
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
        else if (a == "-c") n_ctx = atoi(next());
        else if (a == "-t") threads = atoi(next());
        else if (a == "--kld-max") kld_max = atof(next());
        else if (a == "--top1-min") top1_min = atof(next());
        else if (a == "--save") save = next();
        else if (a == "--load") load = next();
        else if (a == "--backend") backend = next();
        else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (!model) {
        fprintf(stderr, "usage: ac_bitnet_parity -m MODEL [-p PROMPT] [-n GEN] [-c N_CTX] [-t THREADS]\n");
        return 2;
    }

    if (save || load) {
        if ((save && load) || (backend != "cpu" && backend != "gpu")) {
            fprintf(stderr, "--save or --load takes --backend cpu|gpu\n");
            return 2;
        }
        bitnet_context_t ctx = open_ctx(model, n_ctx, threads, backend == "gpu" ? 9999 : 0);
        if (!ctx) {
            fprintf(stderr, "PARITY FAIL: %s context\n", backend.c_str());
            return 3;
        }
        int n_vocab = 0;
        std::vector<int32_t> seq;
        std::vector<float> ref;
        if (load) {
            FILE* f = fopen(load, "rb");
            int32_t hdr[2] = {};
            if (!f || fread(hdr, 4, 2, f) != 2) {
                fprintf(stderr, "PARITY FAIL: cannot read %s\n", load);
                return 3;
            }
            seq.resize(hdr[0]);
            ref.resize((size_t)hdr[0] * hdr[1]);
            if (fread(seq.data(), 4, seq.size(), f) != seq.size() || fread(ref.data(), 4, ref.size(), f) != ref.size()) {
                fprintf(stderr, "PARITY FAIL: %s is truncated\n", load);
                return 3;
            }
            fclose(f);
            n_vocab = hdr[1];
        } else {
            seq.resize(n_ctx);
            const int np = bitnet_tokenize(ctx, prompt.c_str(), seq.data(), n_ctx);
            if (np <= 0 || np + gen > n_ctx) {
                fprintf(stderr, "PARITY FAIL: prompt tokens %d with %d generated do not fit n_ctx %d\n", np, gen, n_ctx);
                return 3;
            }
            seq.resize(np);
        }
        Stats st;
        std::vector<float> out;
        double t_eval = 0;
        const int n_pos = load ? (int)seq.size() : (int)seq.size() + gen;
        for (int pos = 0; pos < n_pos; ++pos) {
            const int32_t tok = seq[pos];
            const double t0 = now_ms();
            if (bitnet_eval(ctx, &tok, 1) != 0) {
                fprintf(stderr, "PARITY FAIL: eval at position %d\n", pos);
                return 4;
            }
            t_eval += now_ms() - t0;
            int nv = 0;
            const float* l = bitnet_get_logits(ctx, &nv);
            if (load) {
                if (nv != n_vocab) {
                    fprintf(stderr, "PARITY FAIL: vocab %d, file %d\n", nv, n_vocab);
                    return 4;
                }
                st.add(pos, tok, &ref[(size_t)pos * n_vocab], l, n_vocab, pos < 4 || pos == n_pos - 1);
            } else {
                n_vocab = nv;
                out.insert(out.end(), l, l + nv);
                if (pos + 1 < n_pos && pos + 1 >= (int)seq.size()) seq.push_back(argmax(l, nv));
            }
        }
        printf("TIME per token (%s): %.1f ms\n", backend.c_str(), t_eval / n_pos);
        bitnet_free(ctx);
        if (save) {
            FILE* f = fopen(save, "wb");
            const int32_t hdr[2] = { (int32_t)seq.size(), n_vocab };
            if (!f || fwrite(hdr, 4, 2, f) != 2 || fwrite(seq.data(), 4, seq.size(), f) != seq.size() ||
                fwrite(out.data(), 4, out.size(), f) != out.size() || fclose(f) != 0) {
                fprintf(stderr, "PARITY FAIL: cannot write %s\n", save);
                return 3;
            }
            printf("SAVED %d positions x %d logits (%s) to %s\n", (int)seq.size(), n_vocab, backend.c_str(), save);
            return 0;
        }
        st.print("COMPARED");
        const bool pass = st.kmax <= kld_max && (double)st.agree / st.total >= top1_min;
        printf("PARITY %s (gate: max kld <= %.1e, argmax agreement >= %.3f)\n", pass ? "PASS" : "FAIL", kld_max, top1_min);
        return pass ? 0 : 1;
    }

    bitnet_context_t cpu = open_ctx(model, n_ctx, threads, 0);
    if (!cpu) {
        fprintf(stderr, "PARITY FAIL: CPU context\n");
        return 3;
    }
    bitnet_context_t gpu = open_ctx(model, n_ctx, threads, 9999);
    if (!gpu) {
        fprintf(stderr, "PARITY FAIL: GPU context\n");
        bitnet_free(cpu);
        return 3;
    }
    std::vector<int32_t> toks(n_ctx);
    const int np = bitnet_tokenize(cpu, prompt.c_str(), toks.data(), n_ctx);
    if (np <= 0 || np + 2 * gen > n_ctx) {
        fprintf(stderr, "PARITY FAIL: prompt tokens %d with %d generated do not fit n_ctx %d\n", np, gen, n_ctx);
        return 3;
    }
    toks.resize(np);

    // teacher-forced comparison
    int n_vocab = 0;
    double kmax = 0, ksum = 0, dmax = 0, t_cpu = 0, t_gpu = 0;
    int agree = 0, total = 0, kmax_pos = -1;
    std::vector<float> lc;
    int32_t tok = toks[0];
    for (int pos = 0; pos < np + gen; ++pos) {
        if (pos < np) tok = toks[pos];
        double t0 = now_ms();
        if (bitnet_eval(cpu, &tok, 1) != 0) {
            fprintf(stderr, "PARITY FAIL: CPU eval at position %d\n", pos);
            return 4;
        }
        double t1 = now_ms();
        if (bitnet_eval(gpu, &tok, 1) != 0) {
            fprintf(stderr, "PARITY FAIL: GPU eval at position %d\n", pos);
            return 4;
        }
        double t2 = now_ms();
        t_cpu += t1 - t0;
        t_gpu += t2 - t1;
        const float* c = bitnet_get_logits(cpu, &n_vocab);
        lc.assign(c, c + n_vocab);
        int nv2 = 0;
        const float* g = bitnet_get_logits(gpu, &nv2);
        if (nv2 != n_vocab) {
            fprintf(stderr, "PARITY FAIL: vocab %d vs %d\n", n_vocab, nv2);
            return 4;
        }
        const double k = kld(lc.data(), g, n_vocab);
        const int ac = argmax(lc.data(), n_vocab), ag = argmax(g, n_vocab);
        double d = 0;
        for (int i = 0; i < n_vocab; ++i) d = std::max(d, (double)std::fabs(lc[i] - g[i]));
        ksum += k;
        if (k > kmax) {
            kmax = k;
            kmax_pos = pos;
        }
        dmax = std::max(dmax, d);
        agree += ac == ag;
        ++total;
        if (pos < 4 || ac != ag || pos == np + gen - 1) {
            printf("pos %3d tok %6d  kld %.3e  maxdiff %.4f  argmax cpu %6d gpu %6d%s\n", pos, tok, k, d, ac, ag,
                   ac == ag ? "" : "  DIFFER");
        }
        if (pos >= np - 1) tok = ac; // the CPU's greedy choice feeds both
    }
    const double top1 = (double)agree / total;
    printf("TEACHER-FORCED positions %d: kld mean %.3e max %.3e (pos %d), max |logit diff| %.4f, argmax agree %d/%d\n",
           total, ksum / total, kmax, kmax_pos, dmax, agree, total);
    printf("TIME per token: CPU %.1f ms, GPU %.1f ms\n", t_cpu / total, t_gpu / total);

    // free-running greedy generation of each context
    std::vector<int> seq[2];
    bitnet_context_t cs[2] = { cpu, gpu };
    for (int w = 0; w < 2; ++w) {
        bitnet_reset(cs[w]);
        if (bitnet_eval(cs[w], toks.data(), np) != 0) {
            fprintf(stderr, "PARITY FAIL: %s prompt eval\n", w ? "GPU" : "CPU");
            return 4;
        }
        for (int i = 0; i < gen; ++i) {
            int nv = 0;
            const float* l = bitnet_get_logits(cs[w], &nv);
            int32_t nt = argmax(l, nv);
            seq[w].push_back(nt);
            if (bitnet_eval(cs[w], &nt, 1) != 0) {
                fprintf(stderr, "PARITY FAIL: %s eval of generated token %d\n", w ? "GPU" : "CPU", i);
                return 4;
            }
        }
    }
    int common = 0;
    while (common < gen && seq[0][common] == seq[1][common]) ++common;
    for (int w = 0; w < 2; ++w) {
        std::string text;
        for (int t : seq[w]) text += piece(cs[w], t);
        printf("%s greedy: %s\n", w ? "GPU" : "CPU", text.c_str());
    }
    printf("GREEDY common prefix %d/%d tokens\n", common, gen);

    const bool pass = kmax <= kld_max && top1 >= top1_min;
    printf("PARITY %s (gate: max kld <= %.1e, argmax agreement >= %.3f)\n", pass ? "PASS" : "FAIL", kld_max, top1_min);
    bitnet_free(gpu);
    bitnet_free(cpu);
    return pass ? 0 : 1;
}
