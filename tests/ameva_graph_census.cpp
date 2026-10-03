// SPDX-License-Identifier: Apache-2.0
// Graph census: the exact ggml nodes llama.cpp builds for a model, as input for the AMEVA op coverage.
//
// The model runs on the CPU backend only; the scheduler's eval callback sees every node before it is computed.
// Two graphs are recorded: a prompt batch of --batch tokens and a single-token decode step after it. For each
// distinct node signature (op, result type, source types, contiguity and broadcast pattern) the tool prints how
// often it occurs per graph and one example with its shapes, so kernel work can be derived from facts rather than
// from reading the model code.
//
// Build against the llama.h of the libllama that is loaded at run time (llama.cpp b8471 on the fleet).
#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct Census {
    std::map<std::string, std::pair<int, std::string>> sig; // signature -> (count, example)
    std::vector<std::string> order;                         // graph order of first appearance
    int nodes = 0;
};

static Census* g_census = nullptr;

static std::string shape(const struct ggml_tensor* t) {
    char b[160];
    snprintf(b, sizeof(b), "%s[%lld,%lld,%lld,%lld]%s", ggml_type_name(t->type), (long long)t->ne[0],
             (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], ggml_is_contiguous(t) ? "" : "nc");
    return b;
}

static std::string signature(const struct ggml_tensor* t) {
    std::string s = ggml_op_desc(t);
    s += " -> ";
    s += ggml_type_name(t->type);
    s += ggml_is_contiguous(t) ? "" : "(nc)";
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i) {
        const struct ggml_tensor* x = t->src[i];
        s += i ? ", " : " <- ";
        s += ggml_type_name(x->type);
        if (!ggml_is_contiguous(x)) s += "(nc)";
        if (i > 0 && t->src[0] && !ggml_are_same_shape(x, t->src[0])) s += "(bcast)";
    }
    return s;
}

static bool census_cb(struct ggml_tensor* t, bool ask, void* user) {
    (void)user;
    if (!ask || !g_census) return false;
    if (t->op == GGML_OP_NONE) return false;
    g_census->nodes++;
    const std::string s = signature(t);
    auto it = g_census->sig.find(s);
    if (it == g_census->sig.end()) {
        std::string ex = std::string(t->name) + " " + shape(t);
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i) ex += (i ? ", " : " <- ") + shape(t->src[i]);
        const int32_t* p = (const int32_t*)t->op_params;
        char ps[96];
        snprintf(ps, sizeof(ps), " op_params=[%d,%d,%d,%d,%d,%d]", p[0], p[1], p[2], p[3], p[4], p[5]);
        ex += ps;
        g_census->sig.emplace(s, std::make_pair(1, ex));
        g_census->order.push_back(s);
    } else {
        it->second.first++;
    }
    return false; // observe only, never request the data
}

static void report(const char* what, const Census& c) {
    printf("== %s: %d nodes, %zu distinct signatures\n", what, c.nodes, c.sig.size());
    for (const std::string& s : c.order) {
        const auto& v = c.sig.at(s);
        printf("  %4d x %s\n         e.g. %s\n", v.first, s.c_str(), v.second.c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--batch N] [--threads N]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    int n_batch = 8, threads = 4;
    bool fa = false;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--batch") && i + 1 < argc) n_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fa")) fa = true;
    }
    llama_log_set([](enum ggml_log_level l, const char* t, void*) { if (l == GGML_LOG_LEVEL_ERROR) fputs(t, stderr); },
                  nullptr);
    llama_backend_init();
    static ggml_backend_dev_t no_devices[1] = {nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = no_devices;
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 2;
    }
    Census prompt, decode;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = census_cb;
    cp.cb_eval_user_data = nullptr;
    cp.no_perf = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 2;
    }
    char desc[128];
    llama_model_desc(model, desc, sizeof(desc));
    printf("GRAPH_CENSUS model=\"%s\" layers=%d batch=%d flash_attn=%s\n", desc, llama_model_n_layer(model), n_batch,
           fa ? "enabled" : "disabled");

    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(n_batch);
    for (int i = 0; i < n_batch; ++i) toks[i] = (llama_token)(100 + 7 * i) % llama_vocab_n_tokens(vocab);
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    for (int i = 0; i < n_batch; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n_batch - 1;
    }
    b.n_tokens = n_batch;
    g_census = &prompt;
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }
    b.token[0] = toks[0];
    b.pos[0] = n_batch;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    b.n_tokens = 1;
    g_census = &decode;
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    g_census = nullptr;
    llama_batch_free(b);
    report("prompt batch", prompt);
    report("single-token decode", decode);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
