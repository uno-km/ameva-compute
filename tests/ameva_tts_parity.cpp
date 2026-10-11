// SPDX-License-Identifier: Apache-2.0
// AMEVA TTS parity: Kokoro-82M (adapters/tts/kokoro) synthesized on the CPU backend and on AMEVA0 from the same
// phonemes, voice and noise seed; waveforms compared, time and real-time factor measured, WAV files written.
//
// Runs (each a full synthesis; the model file is loaded once per backend):
//   cpu-ref    CPU, --ref-model (F32 weights): the reference. Without --ref-model the CPU run of --model is it.
//   cpu        CPU, --model: the CPU's own distance from the reference is the floor (ggml-cpu rounds activations
//              to f16 when the weights are f16).
//   ameva      AMEVA0, --model: the product path.
//   ameva-ref  AMEVA0, --ref-model: the same weights as the reference, so the distance is the backend's alone.
// The comparison runs use the reference's frame counts (durations), so samples line up; whether each backend's
// own rounded durations agree with the reference is reported per run (a token whose duration sum lies within the
// numerical noise of x.5 can round either way).
// Metrics against the reference: relative L2 and SNR of the waveform, log-spectral distance (dB; 1024-point
// frames, hop 256, Hann), relative L2 of the F0 curve. The waveform is sensitive to phase: the harmonic source
// integrates F0 over the utterance, so a small F0 difference shifts every later sample; the log-spectral distance
// and the F0 curve are the gated measures.
// Gate (exit status 0): AMEVA0 ran both graphs with GPU work (plugin counter), every node on AMEVA0, and for each of
// LSD and F0 rel L2: ameva <= --gate-ratio (1.5) x max(floor, --gate-min) where floor is the cpu run's distance.
// With --ref-model, ameva-ref must also stay within the same bound (it should be far below it).
#include "../adapters/tts/kokoro/kokoro.h"

#include "ggml-backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
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
    const char* ref_model = nullptr;
    const char* plugin = nullptr;
    const char* out_dir = nullptr;
    std::string tokens;
    int voice = 0;
    uint64_t seed = 1234;
    int threads = 4;
    int reps = 3;
    int cpu_reps = 1;
    double gate_ratio = 1.5;
    double gate_min_lsd = 0.05;  // dB
    double gate_min_f0 = 1e-4;
    bool gpu_only = false;
};

void usage() {
    fprintf(stderr,
            "usage: ameva_tts_parity --model M.gguf --plugin libggml-ameva.so --tokens 1,2,3 [--ref-model F32.gguf]\n"
            "  [--voice N] [--seed K] [--threads T] [--reps R] [--cpu-reps R] [--out-dir DIR] [--gate-ratio X]\n");
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "--model") o.model = next();
        else if (a == "--ref-model") o.ref_model = next();
        else if (a == "--plugin") o.plugin = next();
        else if (a == "--out-dir") o.out_dir = next();
        else if (a == "--tokens") { v = next(); if (v) o.tokens = v; }
        else if (a == "--tokens-file") {
            v = next();
            FILE* f = v ? fopen(v, "rb") : nullptr;
            if (!f) return false;
            char buf[65536];
            const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            buf[n] = 0;
            o.tokens = buf;
        }
        else if (a == "--voice") { v = next(); if (v) o.voice = atoi(v); }
        else if (a == "--seed") { v = next(); if (v) o.seed = strtoull(v, nullptr, 10); }
        else if (a == "--threads") { v = next(); if (v) o.threads = atoi(v); }
        else if (a == "--reps") { v = next(); if (v) o.reps = atoi(v); }
        else if (a == "--cpu-reps") { v = next(); if (v) o.cpu_reps = atoi(v); }
        else if (a == "--gate-ratio") { v = next(); if (v) o.gate_ratio = atof(v); }
        else if (a == "--gpu-only") o.gpu_only = true;
        else return false;
    }
    return o.model && o.plugin && !o.tokens.empty();
}

std::vector<int32_t> parse_tokens(const std::string& s) {
    std::vector<int32_t> t;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long v = strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        t.push_back((int32_t)v);
        p = e;
    }
    return t;
}

void write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    const uint32_t data = (uint32_t)(x.size() * 2);
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16(1); u32((uint32_t)sr); u32((uint32_t)sr * 2); u16(2); u16(16);
    fwrite("data", 1, 4, f); u32(data);
    for (float v : x) {
        const float c = std::max(-1.0f, std::min(1.0f, v));
        const int16_t s = (int16_t)std::lrint(c * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

void write_f32(const std::string& path, const std::vector<float>& x) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(x.data(), sizeof(float), x.size(), f);
    fclose(f);
}

void fft(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2 * M_PI / (double)len;
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// mean over frames of the RMS over bins of 10 log10(Pa / Pb); powers floored at 1e-10
double lsd_db(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size()), N = 1024, hop = 256;
    if (n < N) return NAN;
    double total = 0;
    size_t frames = 0;
    std::vector<std::complex<double>> fa(N), fb(N);
    for (size_t s = 0; s + N <= n; s += hop) {
        for (size_t i = 0; i < N; ++i) {
            const double w = 0.5 - 0.5 * std::cos(2 * M_PI * (double)i / (double)N);
            fa[i] = a[s + i] * w;
            fb[i] = b[s + i] * w;
        }
        fft(fa);
        fft(fb);
        double acc = 0;
        for (size_t k = 0; k <= N / 2; ++k) {
            const double pa = std::max(std::norm(fa[k]), 1e-10), pb = std::max(std::norm(fb[k]), 1e-10);
            const double d = 10 * std::log10(pa / pb);
            acc += d * d;
        }
        total += std::sqrt(acc / (double)(N / 2 + 1));
        ++frames;
    }
    return total / (double)frames;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        den += (double)b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : NAN;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) m = std::max(m, std::fabs((double)a[i] - b[i]));
    return m;
}

double rms(const std::vector<float>& a) {
    double s = 0;
    for (float v : a) s += (double)v * v;
    return a.empty() ? 0 : std::sqrt(s / (double)a.size());
}

struct Run {
    std::string tag;
    kokoro_result r;
    std::vector<double> ms;  // total per repetition
    int64_t gpu_ops = 0;
    int dur_mismatch = -1;   // tokens whose own rounded duration differs from the reference's
    bool ok = false;
};

double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    const std::vector<int32_t> ph = parse_tokens(o.tokens);
    ggml_backend_reg_t areg = ggml_backend_load(o.plugin);
    if (!areg) {
        fprintf(stderr, "ggml_backend_load(%s) failed\n", o.plugin);
        return 1;
    }
    ggml_backend_dev_t adev = ggml_backend_dev_by_name("AMEVA0");
    if (!adev) {
        fprintf(stderr, "AMEVA0 device not registered by %s\n", o.plugin);
        return 1;
    }
    void* handle = dlopen(o.plugin, RTLD_NOW | RTLD_NOLOAD);
    counter_fn gpu_ops = handle ? (counter_fn)dlsym(handle, "ggml_backend_ameva_get_gpu_op_count") : nullptr;
    if (!gpu_ops) {
        fprintf(stderr, "ggml_backend_ameva_get_gpu_op_count not found in %s: GPU work cannot be proven\n", o.plugin);
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_dev_init(adev, nullptr);
    if (!cpu || !gpu) {
        fprintf(stderr, "backend init failed (cpu %p, AMEVA0 %p)\n", (void*)cpu, (void*)gpu);
        return 1;
    }
    {
        ggml_backend_reg_t creg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
        auto set_threads = (void (*)(ggml_backend_t, int))ggml_backend_reg_get_proc_address(creg, "ggml_backend_set_n_threads");
        if (set_threads) set_threads(cpu, o.threads);
    }
    printf("phonemes %zu voice %d seed %llu threads %d backend %s (%s)\n", ph.size(), o.voice,
           (unsigned long long)o.seed, o.threads, ggml_backend_name(gpu), ggml_backend_dev_description(adev));

    if (o.gpu_only && o.ref_model) {
        fprintf(stderr, "--ref-model has no effect with --gpu-only (no comparison runs); give one of them\n");
        return 2;
    }
    std::string err;
    struct Loaded { kokoro_model *cpu_m = nullptr, *gpu_m = nullptr; } prod, ref;
    // --gpu-only runs the AMEVA model alone: no CPU copy is loaded, so the peak RSS is that of the product path
    if (!o.gpu_only) prod.cpu_m = kokoro_load(o.model, cpu, &err);
    if (prod.cpu_m || o.gpu_only) prod.gpu_m = kokoro_load(o.model, gpu, &err);
    if ((!o.gpu_only && !prod.cpu_m) || !prod.gpu_m) {
        fprintf(stderr, "load %s: %s\n", o.model, err.c_str());
        return 1;
    }
    if (o.ref_model) {
        ref.cpu_m = kokoro_load(o.ref_model, cpu, &err);
        if (ref.cpu_m) ref.gpu_m = kokoro_load(o.ref_model, gpu, &err);
        if (!ref.cpu_m || !ref.gpu_m) {
            fprintf(stderr, "load %s: %s\n", o.ref_model, err.c_str());
            return 1;
        }
    }
    const int sr = kokoro_sample_rate(prod.gpu_m);
    printf("model %s (%s)%s%s\n", o.model, kokoro_weight_type(prod.gpu_m).c_str(), o.ref_model ? ", reference " : "",
           o.ref_model ? o.ref_model : "");

    kokoro_options opt;
    opt.voice = o.voice;
    opt.seed = o.seed;

    auto run = [&](const char* tag, kokoro_model* m, bool is_gpu, int reps, const std::vector<int32_t>& dur) {
        Run R;
        R.tag = tag;
        kokoro_options op = opt;
        op.durations = dur;
        for (int i = 0; i < std::max(1, reps) + (is_gpu ? 1 : 0); ++i) {  // AMEVA: one warm-up (pipelines)
            const int64_t c0 = gpu_ops();
            kokoro_result r;
            if (!kokoro_synthesize(m, ph, op, &r, &err)) {
                fprintf(stderr, "%s: %s\n", tag, err.c_str());
                return R;
            }
            const int64_t c1 = gpu_ops();
            if (is_gpu && i == 0) {
                R.gpu_ops = c1 - c0;
                if (reps > 0) continue;
            }
            R.ms.push_back(r.ms_text + r.ms_synth + r.ms_host);
            R.r = r;
            if (!is_gpu && c1 != c0) fprintf(stderr, "%s: AMEVA counter moved during a CPU run\n", tag);
        }
        R.ok = true;
        return R;
    };

    if (o.gpu_only) {
        // timing / profiling of the product path only (no comparison, no gate except GPU work)
        Run g = run("ameva", prod.gpu_m, true, o.reps, {});
        if (!g.ok) return 1;
        const double secs = (double)g.r.audio.size() / sr;
        const double mn = *std::min_element(g.ms.begin(), g.ms.end());
        printf("ameva gpu-only: audio %.2f s, ms min %.1f med %.1f (text %.0f, synth %.0f, host %.0f), RTF %.3f, "
               "gpu_ops %lld\n", secs, mn, median(g.ms), g.r.ms_text, g.r.ms_synth, g.r.ms_host, mn / 1000.0 / secs,
               (long long)g.gpu_ops);
        if (o.out_dir) write_wav(std::string(o.out_dir) + "/ameva-gpu-only.wav", g.r.audio, sr);
        printf("RESULT %s\n", g.gpu_ops > 0 ? "PASS" : "FAIL");
        return g.gpu_ops > 0 ? 0 : 1;
    }
    std::vector<Run> runs;
    Run rref = run("cpu-ref", o.ref_model ? ref.cpu_m : prod.cpu_m, false, o.cpu_reps, {});
    if (!rref.ok) return 1;
    const std::vector<int32_t> dur = rref.r.durations;
    runs.push_back(rref);
    if (o.ref_model) runs.push_back(run("cpu", prod.cpu_m, false, o.cpu_reps, dur));
    runs.push_back(run("ameva", prod.gpu_m, true, o.reps, dur));
    if (o.ref_model) runs.push_back(run("ameva-ref", ref.gpu_m, true, 1, dur));
    for (auto& R : runs) {
        if (!R.ok) return 1;
        int mis = 0;
        for (size_t i = 0; i < R.r.dur_raw.size(); ++i)
            mis += std::max(1, (int)std::nearbyint(R.r.dur_raw[i])) != dur[i];
        R.dur_mismatch = mis;
    }

    const double secs = (double)rref.r.audio.size() / sr;
    printf("audio %zu samples (%.2f s at %d Hz), frames %zu, nodes text %d synth %d\n", rref.r.audio.size(), secs, sr,
           rref.r.f0.size() / 2, rref.r.nodes_text, rref.r.nodes_synth);
    printf("%-10s %9s %9s %11s %7s %8s %9s %8s %7s %8s %8s %6s %s\n", "run", "ms_min", "ms_med", "text/syn", "RTF",
           "rms", "rel_l2", "max_abs", "SNR_dB", "LSD_dB", "F0_rel", "dur!=", "gpu_ops");
    double floor_lsd = 0, floor_f0 = 0;
    for (const auto& R : runs) {
        if (R.tag == "cpu") {
            floor_lsd = lsd_db(R.r.audio, rref.r.audio);
            floor_f0 = rel_l2(R.r.f0, rref.r.f0);
        }
    }
    const double lim_lsd = o.gate_ratio * std::max(floor_lsd, o.gate_min_lsd);
    const double lim_f0 = o.gate_ratio * std::max(floor_f0, o.gate_min_f0);
    bool pass = true;
    for (const auto& R : runs) {
        const double mn = *std::min_element(R.ms.begin(), R.ms.end());
        const double rl = rel_l2(R.r.audio, rref.r.audio);
        const double snr = rl > 0 ? -20 * std::log10(rl) : INFINITY;
        const double lsd = lsd_db(R.r.audio, rref.r.audio);
        const double f0r = rel_l2(R.r.f0, rref.r.f0);
        char ts[32];
        snprintf(ts, sizeof(ts), "%.0f/%.0f", R.r.ms_text, R.r.ms_synth);
        printf("%-10s %9.1f %9.1f %11s %7.3f %8.4f %9.2e %8.1e %7.2f %8.4f %8.1e %6d %lld\n", R.tag.c_str(), mn,
               median(R.ms), ts, mn / 1000.0 / secs, rms(R.r.audio), rl, max_abs(R.r.audio, rref.r.audio), snr, lsd,
               f0r, R.dur_mismatch, (long long)R.gpu_ops);
        bool nonfinite = false;
        for (float v : R.r.audio) nonfinite |= !std::isfinite(v);
        if (nonfinite) {
            printf("  %s: non-finite samples\n", R.tag.c_str());
            pass = false;
        }
        if (R.tag.rfind("ameva", 0) == 0) {
            if (R.gpu_ops <= 0) {
                printf("  %s: no GPU operations counted\n", R.tag.c_str());
                pass = false;
            }
            if (!(lsd <= lim_lsd) || !(f0r <= lim_f0)) {
                printf("  %s: outside the gate (LSD %.4f > %.4f or F0 %.2e > %.2e)\n", R.tag.c_str(), lsd, lim_lsd, f0r, lim_f0);
                pass = false;
            }
        }
        if (o.out_dir) {
            const std::string base = std::string(o.out_dir) + "/" + R.tag;
            write_wav(base + ".wav", R.r.audio, sr);
            write_f32(base + ".f32", R.r.audio);
            write_f32(base + ".f0.f32", R.r.f0);
        }
    }
    printf("floor (cpu vs cpu-ref): LSD %.4f dB, F0 rel %.2e; gate x%.2f: LSD <= %.4f, F0 <= %.2e\n", floor_lsd, floor_f0,
           o.gate_ratio, lim_lsd, lim_f0);
    printf("RESULT %s\n", pass ? "PASS" : "FAIL");
    kokoro_free(prod.cpu_m);
    kokoro_free(prod.gpu_m);
    kokoro_free(ref.cpu_m);
    kokoro_free(ref.gpu_m);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
