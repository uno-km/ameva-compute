// SPDX-License-Identifier: Apache-2.0
// AMEVA vision parity: the vision encoder and projector of a multimodal GGUF pair (mmproj + text model) run through
// libmtmd (clip.cpp) on the CPU backend and on AMEVA0, compared embedding by embedding on one image.
//
// Build against the mtmd.h / llama.h of the libmtmd and libllama that are loaded at run time (llama.cpp b8471 on the
// fleet). mtmd_context_params and llama_*_params are passed by value; a header of another build corrupts them silently.
//
// What is proven, per backend:
//   - the backend clip.cpp actually opened ("CLIP using <name> backend" in its own log). clip.cpp falls back to the
//     default GPU and then to the CPU when MTMD_BACKEND_DEVICE cannot be opened; that is a failure here.
//   - the operators clip.cpp lists as not supported by the backend and the scheduler's graph splits; with AMEVA0 any
//     unsupported operator fails the run unless --allow-split is given (the scheduler would compute it on the CPU).
//   - the GPU operations AMEVA completed during the encode (plugin counter), which must be > 0.
// Compared against the reference (CPU on --ref-mmproj, default the same mmproj): cosine similarity per output token
// (min, mean), relative L2 distance of the whole output, largest absolute difference. Encode time: --reps encodes of
// every image chunk after clip.cpp's warm-up, min and median per image.
// --caption N: greedy decoding of N tokens by the text model on the CPU from each backend's image embeddings (the
// text model is the same in both runs, so a divergence comes from the image embeddings).
// Gate. With --ref-mmproj R (the mmproj with every weight dequantized to F32, e.g. by gguf-py; then the reference has
// no quantization anywhere) the CPU backend also runs the tested mmproj ("cpu-model"), whose distance from R is the
// floor: the CPU's own activation quantization (q8_0 activations for q8_0 weights). AMEVA's relative L2 distance from
// R must stay within --gate-ratio (1.5) times that floor. Without R the smallest per-token cosine against the CPU on
// the same mmproj must reach --gate-cos (0.999), a weaker check: two quantized paths are compared with each other.
// The caption is reported, not gated: greedy decoding flips at near-ties (the logit lead is printed).
// Exit status 0 only if AMEVA0 ran the encoder with GPU work, no unsupported operator (unless allowed), and the gate
// holds.
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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

typedef int64_t (*counter_fn)(void);

namespace {

struct Options {
    const char* model = nullptr;
    const char* mmproj = nullptr;
    const char* ref_mmproj = nullptr;
    const char* image = nullptr;
    const char* plugin = nullptr;
    const char* prompt = "Describe this image in one sentence.";
    int threads = 4;
    int reps = 3;
    int caption = 0;
    int fa = -1;  // llama_flash_attn_type: -1 auto, 0 off, 1 on
    double gate_cos = 0.999;   // without --ref-mmproj
    double gate_ratio = 1.5;   // with --ref-mmproj: AMEVA rel_l2 / CPU-model rel_l2
    bool allow_split = false;
};

// clip.cpp log lines that say where the encoder runs
struct ClipLog {
    std::string backend;
    int splits = -1;
    int unsupported = 0;
    bool in_unsupported_list = false;
    std::string fa;
};
ClipLog g_clip;

void clip_log(enum ggml_log_level level, const char* text, void* /*ud*/) {
    if (!text) return;
    const std::string s(text);
    size_t p;
    if ((p = s.find("CLIP using ")) != std::string::npos) {
        const size_t b = p + strlen("CLIP using ");
        const size_t e = s.find(" backend", b);
        g_clip.backend = s.substr(b, e == std::string::npos ? std::string::npos : e - b);
    }
    if ((p = s.find("graph splits = ")) != std::string::npos) g_clip.splits = atoi(s.c_str() + p + 15);
    if ((p = s.find("flash attention is ")) != std::string::npos) {
        g_clip.fa = s.substr(p + 19);
        while (!g_clip.fa.empty() && (g_clip.fa.back() == '\n' || g_clip.fa.back() == '\r')) g_clip.fa.pop_back();
        g_clip.in_unsupported_list = false;
    }
    if (s.find("list of unsupported ops") != std::string::npos) {
        g_clip.in_unsupported_list = true;
    } else if (g_clip.in_unsupported_list && s.find(": type = ") != std::string::npos) {
        ++g_clip.unsupported;
    }
    if (level >= GGML_LOG_LEVEL_WARN || s.find("CLIP using") != std::string::npos ||
        s.find("graph splits") != std::string::npos || s.find("flash attention is") != std::string::npos) {
        fputs(text, stderr);
    }
}

void llama_log_filter(enum ggml_log_level level, const char* text, void* /*ud*/) {
    if (level >= GGML_LOG_LEVEL_ERROR && text) fputs(text, stderr);
}

counter_fn g_gpu_ops = nullptr;
std::string g_ameva_name;  // backend name clip.cpp logs for AMEVA0 (the plugin registry's name)

struct Encoded {
    std::string label;
    std::string backend;
    int splits = -1;
    int unsupported = 0;
    std::string fa;
    std::vector<float> embd;  // all image chunks, n_tokens x n_embd
    size_t n_tokens = 0;
    size_t n_chunks = 0;
    std::vector<double> ms;   // per image (all chunks), one per rep
    int64_t gpu_ops = 0;
    std::vector<llama_token> caption;
    std::vector<float> margin;  // per caption step: logit of the chosen token minus the runner-up
    std::string caption_text;
    bool ok = false;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

bool greedy_caption(const Options& o, llama_model* model, mtmd_context* mctx, const mtmd_input_chunks* chunks,
                    Encoded& r) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_pos n_pos = mtmd_helper_get_n_pos(chunks);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n_pos + o.caption + 16);
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        fprintf(stderr, "[%s] llama_init_from_model failed\n", r.label.c_str());
        return false;
    }
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, (int32_t)cp.n_batch, true, &n_past) != 0) {
        fprintf(stderr, "[%s] mtmd_helper_eval_chunks failed\n", r.label.c_str());
        llama_free(lctx);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < o.caption; ++i) {
        const float* logits = llama_get_logits_ith(lctx, -1);
        if (!logits) {
            fprintf(stderr, "[%s] no logits at step %d\n", r.label.c_str(), i);
            llama_free(lctx);
            return false;
        }
        llama_token best = 0;
        for (llama_token t = 1; t < n_vocab; ++t)
            if (logits[t] > logits[best]) best = t;
        float second = -INFINITY;
        for (llama_token t = 0; t < n_vocab; ++t)
            if (t != best && logits[t] > second) second = logits[t];
        r.margin.push_back(logits[best] - second);
        if (llama_vocab_is_eog(vocab, best)) break;
        r.caption.push_back(best);
        char buf[256];
        const int n = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        if (n > 0) r.caption_text.append(buf, n);
        llama_batch b = llama_batch_get_one(&best, 1);
        if (llama_decode(lctx, b) != 0) {
            fprintf(stderr, "[%s] llama_decode failed at step %d\n", r.label.c_str(), i);
            llama_free(lctx);
            return false;
        }
    }
    llama_free(lctx);
    return true;
}

Encoded run(const Options& o, llama_model* model, const char* mmproj, bool gpu, const char* label) {
    Encoded r;
    r.label = label;
    g_clip = ClipLog();
    if (gpu) setenv("MTMD_BACKEND_DEVICE", "AMEVA0", 1);
    else unsetenv("MTMD_BACKEND_DEVICE");

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = gpu;
    mp.print_timings = false;
    mp.n_threads = o.threads;
    mp.flash_attn_type = (enum llama_flash_attn_type)o.fa;
    mp.warmup = true;
    mtmd_context* mctx = mtmd_init_from_file(mmproj, model, mp);
    if (!mctx) {
        fprintf(stderr, "[%s] mtmd_init_from_file(%s) failed\n", label, mmproj);
        return r;
    }
    r.backend = g_clip.backend;
    r.splits = g_clip.splits;
    r.unsupported = g_clip.unsupported;
    r.fa = g_clip.fa;
    const std::string want = gpu ? g_ameva_name : "CPU";
    if (r.backend != want) {
        fprintf(stderr, "[%s] clip.cpp opened backend \"%s\", requested %s (FAIL: no silent fallback)\n", label,
                r.backend.c_str(), want.c_str());
        mtmd_free(mctx);
        return r;
    }

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mctx, o.image);
    if (!bmp) {
        fprintf(stderr, "[%s] cannot load image %s\n", label, o.image);
        mtmd_free(mctx);
        return r;
    }
    const std::string text = std::string("<|im_start|>User:") + mtmd_default_marker() + o.prompt +
                             "<end_of_utterance>\nAssistant:";
    mtmd_input_text in = {text.c_str(), true, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = {bmp};
    if (mtmd_tokenize(mctx, chunks, &in, bmps, 1) != 0) {
        fprintf(stderr, "[%s] mtmd_tokenize failed\n", label);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
        return r;
    }
    const size_t n_embd = (size_t)llama_model_n_embd(model);
    for (int rep = 0; rep < o.reps; ++rep) {
        const int64_t g0 = g_gpu_ops ? g_gpu_ops() : 0;
        double ms = 0.0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            const auto t0 = std::chrono::steady_clock::now();
            if (mtmd_encode_chunk(mctx, c) != 0) {
                fprintf(stderr, "[%s] mtmd_encode_chunk failed (chunk %zu)\n", label, i);
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                mtmd_free(mctx);
                return r;
            }
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep == 0) {
                const size_t nt = mtmd_input_chunk_get_n_tokens(c);
                const float* e = mtmd_get_output_embd(mctx);
                r.embd.insert(r.embd.end(), e, e + nt * n_embd);
                r.n_tokens += nt;
                ++r.n_chunks;
            }
        }
        r.ms.push_back(ms);
        if (rep == 0) r.gpu_ops = (g_gpu_ops ? g_gpu_ops() : 0) - g0;
    }
    r.ok = r.n_chunks > 0;
    if (!r.ok) fprintf(stderr, "[%s] the prompt produced no image chunk\n", label);
    if (r.ok && o.caption > 0) r.ok = greedy_caption(o, model, mctx, chunks, r);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(mctx);
    return r;
}

void print_run(const Encoded& r) {
    std::vector<double> s = r.ms;
    std::sort(s.begin(), s.end());
    printf("%-10s backend=%s fa=%s splits=%d unsupported_ops=%d chunks=%zu tokens=%zu gpu_ops=%lld "
           "encode_ms min=%.1f median=%.1f",
           r.label.c_str(), r.backend.c_str(), r.fa.c_str(), r.splits, r.unsupported, r.n_chunks, r.n_tokens,
           (long long)r.gpu_ops, s.empty() ? 0.0 : s.front(), median(s));
    printf(" all=[");
    for (size_t i = 0; i < r.ms.size(); ++i) printf("%s%.1f", i ? " " : "", r.ms[i]);
    printf("]\n");
}

struct Diff {
    double cos_min = 1.0, cos_mean = 0.0, rel_l2 = 0.0, max_abs = 0.0;
    size_t cos_min_tok = 0;
};

Diff compare(const Encoded& ref, const Encoded& x, size_t n_embd) {
    Diff d;
    double num = 0.0, den = 0.0;
    for (size_t t = 0; t < ref.n_tokens; ++t) {
        const float* a = ref.embd.data() + t * n_embd;
        const float* b = x.embd.data() + t * n_embd;
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (size_t k = 0; k < n_embd; ++k) {
            ab += (double)a[k] * b[k];
            aa += (double)a[k] * a[k];
            bb += (double)b[k] * b[k];
            const double e = (double)a[k] - b[k];
            num += e * e;
            den += (double)a[k] * a[k];
            d.max_abs = std::max(d.max_abs, std::fabs(e));
        }
        const double c = (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : (aa == bb ? 1.0 : 0.0);
        d.cos_mean += c;
        if (c < d.cos_min) {
            d.cos_min = c;
            d.cos_min_tok = t;
        }
    }
    d.cos_mean /= (double)std::max<size_t>(1, ref.n_tokens);
    d.rel_l2 = den > 0 ? std::sqrt(num / den) : 0.0;
    return d;
}

// the files the ggml, llama, mtmd and AMEVA libraries were mapped from (a stale copy earlier on the search path
// loads without an error and changes the struct layouts)
void print_mapped_libraries() {
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return;
    std::vector<std::string> seen;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        const char* p = strchr(line, '/');
        if (!p) continue;
        std::string path(p);
        while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
        const std::string base = path.substr(path.rfind('/') + 1);
        static const char* names[] = {"libggml", "libllama", "libmtmd", "libac_runtime"};
        bool want = false;
        for (const char* n : names) want = want || base.rfind(n, 0) == 0;
        if (want && std::find(seen.begin(), seen.end(), path) == seen.end()) {
            seen.push_back(path);
            printf("mapped %s\n", path.c_str());
        }
    }
    fclose(f);
}

bool finite_all(const std::vector<float>& v) {
    for (float f : v)
        if (!std::isfinite(f)) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const bool has = i + 1 < argc;
        if (!strcmp(a, "--model") && has) o.model = argv[++i];
        else if (!strcmp(a, "--mmproj") && has) o.mmproj = argv[++i];
        else if (!strcmp(a, "--ref-mmproj") && has) o.ref_mmproj = argv[++i];
        else if (!strcmp(a, "--image") && has) o.image = argv[++i];
        else if (!strcmp(a, "--plugin") && has) o.plugin = argv[++i];
        else if (!strcmp(a, "--prompt") && has) o.prompt = argv[++i];
        else if (!strcmp(a, "--threads") && has) o.threads = atoi(argv[++i]);
        else if (!strcmp(a, "--reps") && has) o.reps = atoi(argv[++i]);
        else if (!strcmp(a, "--caption") && has) o.caption = atoi(argv[++i]);
        else if (!strcmp(a, "--fa") && has) {
            const char* v = argv[++i];
            o.fa = !strcmp(v, "on") ? 1 : !strcmp(v, "off") ? 0 : !strcmp(v, "auto") ? -1 : -2;
        } else if (!strcmp(a, "--gate-cos") && has) o.gate_cos = atof(argv[++i]);
        else if (!strcmp(a, "--gate-ratio") && has) o.gate_ratio = atof(argv[++i]);
        else if (!strcmp(a, "--allow-split")) o.allow_split = true;
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n", a);
            return 2;
        }
    }
    if (!o.model || !o.mmproj || !o.image || !o.plugin || o.threads < 1 || o.reps < 1 || o.caption < 0 || o.fa == -2) {
        fprintf(stderr,
                "usage: %s --model TEXT.gguf --mmproj MMPROJ.gguf --image IMG --plugin libggml-ameva.so\n"
                "          [--ref-mmproj R.gguf] [--prompt P] [--threads 4] [--reps 3] [--caption N]\n"
                "          [--fa auto|on|off] [--gate-ratio 1.5] [--gate-cos 0.999] [--allow-split]\n",
                argv[0]);
        return 2;
    }

    llama_log_set(llama_log_filter, nullptr);
    mtmd_helper_log_set(clip_log, nullptr);
    llama_backend_init();

    ggml_backend_reg_t reg = ggml_backend_load(o.plugin);
    if (!reg) {
        fprintf(stderr, "ggml_backend_load(%s) failed\n", o.plugin);
        return 2;
    }
    g_ameva_name = ggml_backend_reg_name(reg);
    if (!ggml_backend_dev_by_name("AMEVA0")) {
        fprintf(stderr, "device AMEVA0 is not registered after loading %s\n", o.plugin);
        return 2;
    }
    void* handle = dlopen(o.plugin, RTLD_NOW | RTLD_NOLOAD);
    g_gpu_ops = handle ? (counter_fn)dlsym(handle, "ggml_backend_ameva_get_gpu_op_count") : nullptr;
    if (!g_gpu_ops) {
        fprintf(stderr, "AMEVA counter ggml_backend_ameva_get_gpu_op_count not found in %s: GPU work cannot be proven\n",
                o.plugin);
        return 2;
    }

    llama_model_params lp = llama_model_default_params();
    lp.n_gpu_layers = 0;  // the text model stays on the CPU in every run; vocab_only would report n_embd = 0
    llama_model* model = llama_model_load_from_file(o.model, lp);
    if (!model) {
        fprintf(stderr, "cannot load text model %s\n", o.model);
        return 2;
    }
    const size_t n_embd = (size_t)llama_model_n_embd(model);
    print_mapped_libraries();

    Encoded ref = run(o, model, o.ref_mmproj ? o.ref_mmproj : o.mmproj, false, "cpu-ref");
    if (!ref.ok) return 2;
    print_run(ref);
    Encoded cpu;
    if (o.ref_mmproj) {
        cpu = run(o, model, o.mmproj, false, "cpu-model");
        if (!cpu.ok) return 2;
        print_run(cpu);
    }
    Encoded gpu = run(o, model, o.mmproj, true, "ameva");
    if (!gpu.ok && gpu.n_chunks == 0) {
        printf("AMEVA_VISION_PARITY=FAIL\n");
        return 1;
    }
    print_run(gpu);

    bool pass = gpu.ok;
    if (gpu.n_tokens != ref.n_tokens) {
        printf("FAIL token count: ameva %zu, reference %zu\n", gpu.n_tokens, ref.n_tokens);
        pass = false;
    } else {
        if (!finite_all(gpu.embd)) {
            printf("FAIL ameva embeddings contain NaN or Inf\n");
            pass = false;
        }
        const Diff d = compare(ref, gpu, n_embd);
        if (o.ref_mmproj) {
            // floor: the CPU backend's own distance on the tested mmproj (its activation quantization)
            const Diff c = compare(ref, cpu, n_embd);
            printf("cpu-model vs cpu-ref: cos_min=%.7f (token %zu) cos_mean=%.7f rel_l2=%.3e max_abs=%.3e\n",
                   c.cos_min, c.cos_min_tok, c.cos_mean, c.rel_l2, c.max_abs);
            const double ratio = c.rel_l2 > 0 ? d.rel_l2 / c.rel_l2 : (d.rel_l2 > 0 ? INFINITY : 0.0);
            printf("ameva vs cpu-ref:     cos_min=%.7f (token %zu) cos_mean=%.7f rel_l2=%.3e max_abs=%.3e "
                   "(rel_l2 / cpu-model floor = %.2f, gate <= %g)\n",
                   d.cos_min, d.cos_min_tok, d.cos_mean, d.rel_l2, d.max_abs, ratio, o.gate_ratio);
            if (!(ratio <= o.gate_ratio)) pass = false;
        } else {
            printf("ameva vs cpu-ref:     cos_min=%.7f (token %zu) cos_mean=%.7f rel_l2=%.3e max_abs=%.3e "
                   "(gate cos_min >= %g)\n",
                   d.cos_min, d.cos_min_tok, d.cos_mean, d.rel_l2, d.max_abs, o.gate_cos);
            if (!(d.cos_min >= o.gate_cos)) pass = false;
        }
    }
    if (gpu.gpu_ops <= 0) {
        printf("FAIL no AMEVA GPU operation completed during the encode\n");
        pass = false;
    }
    if (gpu.unsupported > 0 && !o.allow_split) {
        printf("FAIL %d operators of the encoder graph are not supported by AMEVA0 (computed on the CPU)\n",
               gpu.unsupported);
        pass = false;
    }
    if (o.caption > 0) {
        const Encoded& c0 = ref;
        size_t div = 0;
        while (div < c0.caption.size() && div < gpu.caption.size() && c0.caption[div] == gpu.caption[div]) ++div;
        const bool same = c0.caption == gpu.caption;
        printf("caption cpu-ref (%zu tokens): %s\n", c0.caption.size(), c0.caption_text.c_str());
        if (o.ref_mmproj) printf("caption cpu-model (%zu tokens): %s\n", cpu.caption.size(), cpu.caption_text.c_str());
        printf("caption ameva   (%zu tokens): %s\n", gpu.caption.size(), gpu.caption_text.c_str());
        if (same) {
            printf("caption IDENTICAL\n");
        } else {
            // the step where the two runs chose differently; at an end of text the margin is not recorded
            const float mr = div < c0.margin.size() ? c0.margin[div] : NAN;
            const float mg = div < gpu.margin.size() ? gpu.margin[div] : NAN;
            printf("caption DIFFERS at token %zu (top-1 lead there: cpu-ref %.3f, ameva %.3f logits)\n", div, mr, mg);
        }
    }
    printf("AMEVA_VISION_PARITY=%s\n", pass ? "PASS" : "FAIL");
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
