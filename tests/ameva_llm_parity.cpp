// SPDX-License-Identifier: Apache-2.0
// AMEVA real-model parity: one GGUF model evaluated by the CPU backend and by AMEVA (every weight MUL_MAT through
// ameva_qmm on the GPU), compared position by position on real text.
//
// Build against the llama.h of the libllama that is loaded at run time (llama.cpp b8471 on the fleet). The parameter
// structs are passed by value and change layout between builds; a newer header corrupts them without any error.
//
// Protocol (the one of llama-perplexity): the text is cut into chunks of n_ctx tokens, each chunk is evaluated from
// an empty KV cache with BOS as its first token (when the model adds one), and the second half of each chunk is
// scored.
//   cpu-batch     reference: the whole chunk in one llama_decode
//   cpu-decode    one token per llama_decode on the CPU: the distance between two CPU kernel paths (noise floor)
//   ameva-batch   the whole chunk in one llama_decode: MUL_MAT with M = n_ctx (ncols=8 kernels, prompt processing)
//   ameva-decode  one token per llama_decode: MUL_MAT with M = 1 (ncols=1 kernels, token generation)
// For every run against cpu-batch: perplexity, KL divergence per scored position (mean, p99, max), top-1 agreement,
// RMS difference of log p(actual next token) and decode throughput. For AMEVA runs the completed GPU MUL_MAT
// dispatches are read from the plugin's own counters. Then greedy generation from a fixed prompt on both backends.
//
// Two reference modes:
//   default        the reference is the CPU backend on the same model. For quantized weights the CPU quantizes the
//                  activations too (Q8_0/Q8_K vec_dot, F16 for f16 weights) while AMEVA multiplies fp32 activations,
//                  so the KL divergence measures the CPU's approximation as much as AMEVA. Gate: kld_mean <= 0.01,
//                  two orders below what a wrong decoder produces.
//   --ref-model R  the reference is the CPU backend on R, the same model with every weight dequantized to F32
//                  (tools/ameva_quantize IN R F32): identical weight values, no activation quantization anywhere.
//                  AMEVA on the quantized model must then agree to fp32 accumulation-order noise; a "cpu-model" run
//                  shows the CPU backend's own distance on the quantized model for comparison. Gate: kld_mean
//                  <= 1e-4 unless --gate-kld says otherwise.
// Top-1 agreement is reported, not gated: on a few hundred positions near-ties make it move by whole percents
// between two CPU kernel paths of the same backend.
// Exit status 0 only if both AMEVA runs meet the gate and every AMEVA phase dispatched GPU work.
#include "llama.h"
#include "ggml-backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Options {
    const char* model = nullptr;
    const char* ref_model = nullptr;
    const char* plugin = nullptr;
    const char* text = nullptr;
    const char* prompt = "The three primary colors are";
    int ctx = 256;
    int chunks = 2;
    int threads = 4;
    int gen = 32;
    bool cpu_decode = true;
    bool fa = true;      // flash attention on both backends (the AMEVA graph engine runs it on the GPU)
    bool kv_f32 = false; // f32 KV cache instead of f16
    double gate_kld = -1.0; // < 0: mode default
};

static int g_log_level = GGML_LOG_LEVEL_NONE;
static bool g_log_line_kept = false;
// warnings and errors, plus the loader's placement summary (buffer sizes, offload count, graph splits)
static void log_filter(enum ggml_log_level level, const char* text, void* user) {
    (void)user;
    if (level != GGML_LOG_LEVEL_CONT) {
        g_log_level = level;
        g_log_line_kept = level >= GGML_LOG_LEVEL_WARN ||
                          (level == GGML_LOG_LEVEL_INFO &&
                           (strstr(text, "buffer size") || strstr(text, "offloaded") || strstr(text, "graph splits")));
    }
    if (g_log_line_kept) fputs(text, stderr);
}

// weight types present in a GGUF file, e.g. "f32=121 q4_K=96 q6_K=13"
static std::string tensor_types(const char* path) {
    struct gguf_init_params gp = {true, nullptr};
    struct gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) return "unreadable";
    int counts[GGML_TYPE_COUNT] = {0};
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const enum ggml_type t = gguf_get_tensor_type(g, i);
        if ((int)t >= 0 && (int)t < GGML_TYPE_COUNT) counts[t]++;
    }
    gguf_free(g);
    std::string s;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (!counts[t]) continue;
        if (!s.empty()) s += ' ';
        s += ggml_type_name((enum ggml_type)t);
        s += '=';
        s += std::to_string(counts[t]);
    }
    return s;
}

typedef int64_t (*counter_fn)(void);
static counter_fn g_gpu_ops = nullptr;   // MUL_MAT nodes completed on the GPU
static counter_fn g_gpu_nodes = nullptr; // graph nodes of any op completed on the GPU

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// log-softmax in double precision; returns the argmax, or -1 when a logit is not finite
static int log_softmax(const float* logits, int n, float* out) {
    int am = 0;
    float mx = logits[0];
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) return -1;
        if (logits[i] > mx) {
            mx = logits[i];
            am = i;
        }
    }
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double)logits[i] - (double)mx);
    const double lz = (double)mx + std::log(s);
    for (int i = 0; i < n; ++i) out[i] = (float)((double)logits[i] - lz);
    return am;
}

struct Run {
    std::string name;
    bool ok = true;           // every llama_decode succeeded and every logit was finite
    int scored = 0;
    int nonfinite = 0;
    double nll = 0.0;         // sum of -log p(actual next token)
    std::vector<double> kld;  // KL(P_ref || P_run) per scored position
    int top1_same = 0;
    double dlogp_sq = 0.0;
    double decode_s = 0.0;    // time inside llama_decode
    int64_t evaluated = 0;    // tokens fed to llama_decode
    int64_t gpu_ops = -1;     // AMEVA MUL_MAT nodes completed during the run (-1: CPU run)
    int64_t gpu_nodes = -1;   // AMEVA nodes of any op completed during the run
};

struct Reference {
    int n_vocab = 0;
    std::vector<float> logp;  // scored x n_vocab
    std::vector<int> top1;
    std::vector<float> lp_next;
};

// Evaluates every chunk in batch or decode mode; on_pos(scored_index, logits, next_token) per scored position.
template <class F>
static bool eval_chunks(llama_context* ctx, const std::vector<llama_token>& toks, const Options& o, bool add_bos,
                        llama_token bos, bool decode_mode, Run& run, F on_pos) {
    const int n_ctx = o.ctx, first = n_ctx / 2;
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    int si = 0;
    bool ok = true;
    for (int c = 0; c < o.chunks && ok; ++c) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> ch(toks.begin() + (size_t)c * n_ctx, toks.begin() + (size_t)(c + 1) * n_ctx);
        if (add_bos) ch[0] = bos;
        if (!decode_mode) {
            for (int j = 0; j < n_ctx; ++j) {
                batch.token[j] = ch[j];
                batch.pos[j] = j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (j >= first && j < n_ctx - 1) ? 1 : 0;
            }
            batch.n_tokens = n_ctx;
            const double t0 = now_s();
            const int rc = llama_decode(ctx, batch);
            run.decode_s += now_s() - t0;
            run.evaluated += n_ctx;
            if (rc != 0) {
                fprintf(stderr, "%s: llama_decode failed (%d) in chunk %d\n", run.name.c_str(), rc, c);
                ok = false;
                break;
            }
            for (int j = first; j < n_ctx - 1; ++j) {
                const float* lg = llama_get_logits_ith(ctx, j);
                if (!lg) {
                    fprintf(stderr, "%s: no logits for position %d\n", run.name.c_str(), j);
                    ok = false;
                    break;
                }
                on_pos(si++, lg, ch[j + 1]);
            }
        } else {
            for (int j = 0; j < n_ctx - 1; ++j) {
                batch.token[0] = ch[j];
                batch.pos[0] = j;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = j >= first ? 1 : 0;
                batch.n_tokens = 1;
                const double t0 = now_s();
                const int rc = llama_decode(ctx, batch);
                run.decode_s += now_s() - t0;
                run.evaluated += 1;
                if (rc != 0) {
                    fprintf(stderr, "%s: llama_decode failed (%d) at chunk %d position %d\n", run.name.c_str(), rc, c,
                            j);
                    ok = false;
                    break;
                }
                if (j >= first) {
                    const float* lg = llama_get_logits_ith(ctx, 0);
                    if (!lg) {
                        fprintf(stderr, "%s: no logits for position %d\n", run.name.c_str(), j);
                        ok = false;
                        break;
                    }
                    on_pos(si++, lg, ch[j + 1]);
                }
            }
        }
    }
    llama_batch_free(batch);
    run.ok = run.ok && ok;
    return ok;
}

static void run_reference(llama_context* ctx, const std::vector<llama_token>& toks, const Options& o, bool add_bos,
                          llama_token bos, Reference& ref, Run& run) {
    const int n_vocab = ref.n_vocab;
    eval_chunks(ctx, toks, o, add_bos, bos, false, run, [&](int si, const float* lg, llama_token next) {
        float* lp = &ref.logp[(size_t)si * n_vocab];
        const int am = log_softmax(lg, n_vocab, lp);
        if (am < 0) {
            run.nonfinite++;
            run.ok = false;
            ref.top1[si] = -1;
            ref.lp_next[si] = 0.0f;
            return;
        }
        ref.top1[si] = am;
        ref.lp_next[si] = lp[next];
        run.nll -= lp[next];
        run.scored++;
    });
}

static void run_compare(llama_context* ctx, const std::vector<llama_token>& toks, const Options& o, bool add_bos,
                        llama_token bos, bool decode_mode, const Reference& ref, Run& run) {
    const int n_vocab = ref.n_vocab;
    std::vector<float> lp(n_vocab);
    const int64_t ops0 = g_gpu_ops && run.gpu_ops >= 0 ? g_gpu_ops() : 0;
    const int64_t nodes0 = g_gpu_nodes && run.gpu_ops >= 0 ? g_gpu_nodes() : 0;
    eval_chunks(ctx, toks, o, add_bos, bos, decode_mode, run, [&](int si, const float* lg, llama_token next) {
        const int am = log_softmax(lg, n_vocab, lp.data());
        if (am < 0 || ref.top1[si] < 0) {
            run.nonfinite++;
            run.ok = false;
            return;
        }
        const float* rp = &ref.logp[(size_t)si * n_vocab];
        double kl = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const double p = std::exp((double)rp[i]);
            kl += p * ((double)rp[i] - (double)lp[i]);
        }
        run.kld.push_back(kl);
        if (am == ref.top1[si]) run.top1_same++;
        const double d = (double)lp[next] - (double)ref.lp_next[si];
        run.dlogp_sq += d * d;
        run.nll -= lp[next];
        run.scored++;
    });
    if (g_gpu_ops && run.gpu_ops >= 0) {
        run.gpu_ops = g_gpu_ops() - ops0;
        run.gpu_nodes = g_gpu_nodes() - nodes0;
    }
}

static std::string piece(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    if (n < 0) return "<?>";
    std::string s(buf, (size_t)n);
    std::string out;
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

// Greedy continuation of the prompt. gen_ops/gen_steps: GPU MUL_MAT dispatches over the single-token decodes only.
static bool greedy(llama_context* ctx, const llama_vocab* vocab, const std::vector<llama_token>& prompt, int n_gen,
                   std::vector<llama_token>& out, double* step_s, int64_t* gen_ops, int* gen_steps) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(std::max<int>((int)prompt.size(), 1), 0, 1);
    for (size_t j = 0; j < prompt.size(); ++j) {
        batch.token[j] = prompt[j];
        batch.pos[j] = (llama_pos)j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = j + 1 == prompt.size() ? 1 : 0;
    }
    batch.n_tokens = (int)prompt.size();
    bool ok = llama_decode(ctx, batch) == 0;
    int pos = (int)prompt.size();
    *step_s = 0.0;
    *gen_steps = 0;
    const int64_t ops0 = g_gpu_ops ? g_gpu_ops() : 0;
    for (int i = 0; ok && i < n_gen; ++i) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        if (!lg) {
            ok = false;
            break;
        }
        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (lg[v] > lg[best]) best = v;
        }
        out.push_back(best);
        if (llama_vocab_is_eog(vocab, best) || i + 1 == n_gen) break;
        batch.token[0] = best;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        const double t0 = now_s();
        ok = llama_decode(ctx, batch) == 0;
        *step_s += now_s() - t0;
        (*gen_steps)++;
    }
    *gen_ops = g_gpu_ops ? g_gpu_ops() - ops0 : -1;
    llama_batch_free(batch);
    if (!ok) fprintf(stderr, "greedy generation failed\n");
    return ok;
}

static llama_model* load_model(const Options& o, const char* path, ggml_backend_dev_t dev) {
    // an explicit device list: an empty list keeps every layer and every op on the CPU backend
    static ggml_backend_dev_t devs_cpu[1] = {nullptr};
    static ggml_backend_dev_t devs_gpu[2] = {nullptr, nullptr};
    llama_model_params mp = llama_model_default_params();
    if (dev) {
        devs_gpu[0] = dev;
        mp.devices = devs_gpu;
        mp.n_gpu_layers = 999;
    } else {
        mp.devices = devs_cpu;
        mp.n_gpu_layers = 0;
    }
    llama_model* m = llama_model_load_from_file(path, mp);
    if (!m) fprintf(stderr, "cannot load %s\n", path);
    (void)o;
    return m;
}

static llama_context* make_ctx(llama_model* m, const Options& o) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)o.ctx;
    cp.n_batch = (uint32_t)o.ctx;
    cp.n_ubatch = (uint32_t)o.ctx;
    cp.n_seq_max = 1;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    // the same attention graph on both backends ("auto" would decide per device)
    cp.flash_attn_type = o.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (o.kv_f32) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
    }
    cp.no_perf = true;
    llama_context* c = llama_init_from_model(m, cp);
    if (!c) fprintf(stderr, "cannot create a context\n");
    return c;
}

static double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, (size_t)std::ceil(q * (double)v.size()) - 1);
    return v[i];
}

static void print_run(const Run& r, const Options& o, bool is_ref) {
    const double ppl = r.scored ? std::exp(r.nll / r.scored) : 0.0;
    printf("run=%-12s ok=%d scored=%d ppl=%.4f", r.name.c_str(), (int)r.ok, r.scored, ppl);
    if (!is_ref) {
        double mean = 0.0, mx = 0.0;
        for (double k : r.kld) {
            mean += k;
            mx = std::max(mx, k);
        }
        mean = r.kld.empty() ? 0.0 : mean / (double)r.kld.size();
        printf(" kld_mean=%.3e kld_p99=%.3e kld_max=%.3e top1=%.2f%% dlogp_rms=%.4f", mean, percentile(r.kld, 0.99),
               mx, r.scored ? 100.0 * r.top1_same / r.scored : 0.0, r.scored ? std::sqrt(r.dlogp_sq / r.scored) : 0.0);
    }
    printf(" decode_tok_s=%.2f", r.decode_s > 0 ? (double)r.evaluated / r.decode_s : 0.0);
    if (r.gpu_ops >= 0) printf(" gpu_mul_mat=%lld gpu_nodes=%lld", (long long)r.gpu_ops, (long long)r.gpu_nodes);
    if (r.nonfinite) printf(" nonfinite=%d", r.nonfinite);
    printf("\n");
    (void)o;
}

static bool gate(const Run& r, const Options& o) {
    if (!r.ok || r.scored == 0 || r.gpu_ops <= 0 || r.kld.empty()) return false;
    double mean = 0.0;
    for (double k : r.kld) mean += k;
    mean /= (double)r.kld.size();
    return mean <= o.gate_kld;
}

static void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s --model M.gguf --plugin libggml-ameva.so --text corpus.txt [--ref-model M_f32.gguf]\n"
            "          [--ctx 256] [--chunks 2] [--threads 4] [--gen 32] [--prompt TEXT] [--no-cpu-decode]\n"
            "          [--fa on|off] [--kv f16|f32] [--gate-kld X]\n",
            argv0);
}

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const bool has = i + 1 < argc;
        if (!strcmp(a, "--model") && has) o.model = argv[++i];
        else if (!strcmp(a, "--ref-model") && has) o.ref_model = argv[++i];
        else if (!strcmp(a, "--plugin") && has) o.plugin = argv[++i];
        else if (!strcmp(a, "--text") && has) o.text = argv[++i];
        else if (!strcmp(a, "--prompt") && has) o.prompt = argv[++i];
        else if (!strcmp(a, "--ctx") && has) o.ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--chunks") && has) o.chunks = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has) o.threads = atoi(argv[++i]);
        else if (!strcmp(a, "--gen") && has) o.gen = atoi(argv[++i]);
        else if (!strcmp(a, "--gate-kld") && has) o.gate_kld = atof(argv[++i]);
        else if (!strcmp(a, "--no-cpu-decode")) o.cpu_decode = false;
        else if (!strcmp(a, "--fa") && has) o.fa = !strcmp(argv[++i], "on");
        else if (!strcmp(a, "--kv") && has) o.kv_f32 = !strcmp(argv[++i], "f32");
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!o.model || !o.plugin || !o.text || o.ctx < 8 || o.ctx % 2 || o.chunks < 1 || o.gen < 1) {
        usage(argv[0]);
        return 2;
    }
    if (o.gate_kld < 0) o.gate_kld = o.ref_model ? 1e-4 : 1e-2;
    const char* ref_path = o.ref_model ? o.ref_model : o.model;

    llama_log_set(log_filter, nullptr);
    llama_backend_init();

    ggml_backend_reg_t reg = ggml_backend_load(o.plugin);
    if (!reg) {
        fprintf(stderr, "ggml_backend_load(%s) failed\n", o.plugin);
        return 2;
    }
    ggml_backend_dev_t ameva = ggml_backend_dev_by_name("AMEVA0");
    if (!ameva) {
        fprintf(stderr, "device AMEVA0 is not registered after loading %s\n", o.plugin);
        return 2;
    }
    void* handle = dlopen(o.plugin, RTLD_NOW | RTLD_NOLOAD);
    g_gpu_ops = handle ? (counter_fn)dlsym(handle, "ggml_backend_ameva_get_f32_gpu_sgemm_dispatch_count") : nullptr;
    g_gpu_nodes = handle ? (counter_fn)dlsym(handle, "ggml_backend_ameva_get_gpu_op_count") : nullptr;
    if (!g_gpu_ops || !g_gpu_nodes) {
        fprintf(stderr, "AMEVA counters not found in %s: GPU work cannot be proven\n", o.plugin);
        return 2;
    }

    std::string text;
    {
        FILE* f = fopen(o.text, "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", o.text);
            return 2;
        }
        std::vector<char> buf(1 << 20);
        const size_t n = fread(buf.data(), 1, buf.size(), f);
        fclose(f);
        text.assign(buf.data(), n);
    }

    // ---- CPU reference ----
    llama_model* model = load_model(o, ref_path, nullptr);
    if (!model) return 2;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    Reference ref;
    ref.n_vocab = llama_vocab_n_tokens(vocab);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    std::vector<llama_token> toks(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.data(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), add_bos,
                                  false);
    if (nt < o.ctx * o.chunks) {
        fprintf(stderr, "the text has %d tokens, %d are needed\n", nt, o.ctx * o.chunks);
        return 2;
    }
    toks.resize((size_t)nt);
    std::vector<llama_token> prompt(256);
    const int np = llama_tokenize(vocab, o.prompt, (int32_t)strlen(o.prompt), prompt.data(), (int32_t)prompt.size(),
                                  add_bos, false);
    if (np <= 0) {
        fprintf(stderr, "cannot tokenize the prompt\n");
        return 2;
    }
    prompt.resize((size_t)np);

    char desc[128];
    llama_model_desc(model, desc, sizeof(desc));
    const char* base = strrchr(o.model, '/');
    const int scored_total = o.chunks * (o.ctx - 1 - o.ctx / 2);
    printf("AMEVA_LLM_PARITY model=\"%s\" file=%s params=%.2fM n_vocab=%d layers=%d ctx=%d chunks=%d scored=%d "
           "threads=%d flash_attn=%s kv=%s reference=%s\n",
           desc, base ? base + 1 : o.model, llama_model_n_params(model) / 1e6, ref.n_vocab, llama_model_n_layer(model),
           o.ctx, o.chunks, scored_total, o.threads, o.fa ? "on" : "off", o.kv_f32 ? "f32" : "f16",
           o.ref_model ? "cpu-on-f32-dequantized-model" : "cpu-on-same-model");
    printf("tensor_types model: %s\n", tensor_types(o.model).c_str());
    if (o.ref_model) printf("tensor_types reference: %s\n", tensor_types(o.ref_model).c_str());
    fflush(stdout);

    ref.logp.assign((size_t)scored_total * ref.n_vocab, 0.0f);
    ref.top1.assign((size_t)scored_total, -1);
    ref.lp_next.assign((size_t)scored_total, 0.0f);

    std::vector<Run> runs;
    std::vector<llama_token> gen_cpu, gen_gpu;
    std::string text_cpu, text_gpu; // rendered while the model owning the vocabulary is alive
    double cpu_step_s = 0.0, gpu_step_s = 0.0;
    int cpu_steps = 0, gpu_steps = 0;
    int64_t gpu_gen_ops = -1, unused_ops = 0;
    {
        llama_context* ctx = make_ctx(model, o);
        if (!ctx) return 2;
        Run r;
        r.name = o.ref_model ? "cpu-ref-f32" : "cpu-batch";
        run_reference(ctx, toks, o, add_bos, bos, ref, r);
        runs.push_back(r);
        print_run(runs.back(), o, true);
        fflush(stdout);
        if (o.cpu_decode) {
            Run d;
            d.name = "cpu-decode";
            run_compare(ctx, toks, o, add_bos, bos, true, ref, d);
            runs.push_back(d);
            print_run(runs.back(), o, false);
            fflush(stdout);
        }
        greedy(ctx, vocab, prompt, o.gen, gen_cpu, &cpu_step_s, &unused_ops, &cpu_steps);
        for (llama_token t : gen_cpu) text_cpu += piece(vocab, t);
        llama_free(ctx);
    }
    llama_model_free(model);
    vocab = nullptr;
    if (!runs[0].ok) {
        printf("AMEVA_LLM_PARITY=FAIL (CPU reference failed)\n");
        return 1;
    }

    // ---- CPU on the tested model, against the f32 reference: the CPU backend's own deviation ----
    if (o.ref_model) {
        model = load_model(o, o.model, nullptr);
        if (!model) return 2;
        if (llama_vocab_n_tokens(llama_model_get_vocab(model)) != ref.n_vocab) {
            fprintf(stderr, "%s and %s have different vocabularies\n", o.model, o.ref_model);
            return 2;
        }
        llama_context* ctx = make_ctx(model, o);
        if (!ctx) return 2;
        Run r;
        r.name = "cpu-model";
        run_compare(ctx, toks, o, add_bos, bos, false, ref, r);
        runs.push_back(r);
        print_run(runs.back(), o, false);
        fflush(stdout);
        llama_free(ctx);
        llama_model_free(model);
    }

    // ---- AMEVA ----
    model = load_model(o, o.model, ameva);
    if (!model) return 2;
    bool pass = true;
    {
        llama_context* ctx = make_ctx(model, o);
        if (!ctx) return 2;
        for (int mode = 0; mode < 2; ++mode) {
            Run r;
            r.name = mode == 0 ? "ameva-batch" : "ameva-decode";
            r.gpu_ops = 0;
            run_compare(ctx, toks, o, add_bos, bos, mode == 1, ref, r);
            runs.push_back(r);
            print_run(runs.back(), o, false);
            fflush(stdout);
            if (!gate(r, o)) pass = false;
        }
        const llama_vocab* gvocab = llama_model_get_vocab(model);
        if (!greedy(ctx, gvocab, prompt, o.gen, gen_gpu, &gpu_step_s, &gpu_gen_ops, &gpu_steps)) pass = false;
        for (llama_token t : gen_gpu) text_gpu += piece(gvocab, t);
        llama_free(ctx);
    }
    llama_model_free(model);

    size_t div = 0;
    while (div < gen_cpu.size() && div < gen_gpu.size() && gen_cpu[div] == gen_gpu[div]) ++div;
    printf("gen prompt=\"%s\" tokens cpu=%zu ameva=%zu identical_prefix=%zu\n", o.prompt, gen_cpu.size(),
           gen_gpu.size(), div);
    printf("gen cpu  : \"%s\"\n", text_cpu.c_str());
    printf("gen ameva: \"%s\"\n", text_gpu.c_str());
    printf("gen decode_tok_s cpu=%.2f ameva=%.2f ameva_gpu_mul_mat_per_token=%.1f\n",
           cpu_step_s > 0 ? cpu_steps / cpu_step_s : 0.0, gpu_step_s > 0 ? gpu_steps / gpu_step_s : 0.0,
           gpu_steps ? (double)gpu_gen_ops / gpu_steps : 0.0);
    printf("AMEVA_LLM_PARITY=%s (gate: kld_mean<=%.3g against %s, gpu_mul_mat>0, both AMEVA runs)\n",
           pass ? "PASS" : "FAIL", o.gate_kld, o.ref_model ? "the f32-dequantized model" : "the CPU backend");
    llama_backend_free();
    return pass ? 0 : 1;
}
