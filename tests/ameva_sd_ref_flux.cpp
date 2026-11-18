// ameva_sd_ref_flux: on-device check of reference-image input with a real multi-file image-edit model
// (FLUX.2 klein 4B: diffusion GGUF + Qwen3-4B text encoder GGUF + FLUX.2 VAE), real stable-diffusion.cpp.
//
//   ameva_sd_ref_flux <dit.gguf> <llm.gguf> <vae.safetensors> <cpu|gpu|gpu_raw> [plugin libggml-ameva-sd.so] [size]
//
// Modes: gpu = the AmfyUI product setting (GPU_THEN_CPU, budget fraction 0.90, mmap: sd.cpp auto-fit places each
// component per phase and streams from disk when it does not fit); gpu_raw = budget-less DEFAULT (everything on the GPU
// at once, can exhaust memory on a 6-8 GB phone); cpu = CPU_ONLY. Memory (MemAvailable, peak RSS) is printed.
//
// Each line starts with "chk" and ends in ok/FAIL. Checks:
//   1  txt2img (no reference) works                                   -> status 0
//   2  one reference image works                                      -> status 0
//   3  the reference changes the output                               -> hash differs from (1)
//   4  same reference + seed is repeatable                            -> hash equals (2)
//   5  a different reference image gives a different output          -> hash differs from (2)
//   6  two references work (maxRefs of the AmfyUI preset is 3)        -> status 0, differs from (2)
//   7  the reference pulls the output toward itself                   -> mean |out-ref| smaller than without ref
//                                                                        (indicator, reported, counted as a check)
//   8  malformed requests are refused with INVALID_ARGUMENT           -> count 4, size mismatch, NULL, dims 0,
//                                                                        args without images, malformed args
//   9  a plain generation after the refusals equals (1)               -> refusals leave no state behind
// REF_DUMP=<prefix> writes every output as <prefix>_<label>.rgb (raw RGB8) and the references as <prefix>_in_ref*.rgb.
#include "ameva_sd_api.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static long kbOf(const char* file, const char* key) {
    FILE* f = std::fopen(file, "r");
    if (!f) return -1;
    char line[256]; long v = -1; const size_t n = std::strlen(key);
    while (std::fgets(line, sizeof line, f)) if (!std::strncmp(line, key, n)) { v = std::atol(line + n); break; }
    std::fclose(f);
    return v;
}
static void memline(const char* tag) {
    std::printf("mem %-14s MemAvailable=%ld MB  VmRSS=%ld MB  VmHWM=%ld MB\n", tag,
                kbOf("/proc/meminfo", "MemAvailable:") / 1024, kbOf("/proc/self/status", "VmRSS:") / 1024,
                kbOf("/proc/self/status", "VmHWM:") / 1024);
    std::fflush(stdout);
}

static uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* what, const char* detail = "") {
    std::printf("chk %-58s %s %s\n", what, ok ? "ok" : "FAIL", detail);
    std::fflush(stdout);
    ok ? ++g_pass : ++g_fail;
}

// Deterministic synthetic reference: two flat colour halves, a diagonal gradient and a white disc.
static std::vector<uint8_t> makeRef(int w, int h, int variant) {
    std::vector<uint8_t> v((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* p = &v[((size_t)y * w + x) * 3];
            const bool left = x < w / 2;
            const int g = (x + y) * 255 / (w + h);
            if (variant == 0) { p[0] = left ? 220 : 30; p[1] = (uint8_t)g; p[2] = left ? 30 : 220; }
            else              { p[0] = left ? 30 : 220; p[1] = 200; p[2] = (uint8_t)(255 - g); }
            const int dx = x - w / 2, dy = y - h / 2;
            if (dx * dx + dy * dy < (w / 5) * (w / 5)) { p[0] = p[1] = p[2] = 245; }
        }
    return v;
}

static double meanAbsDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0;
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += std::abs((int)a[i] - (int)b[i]);
    return s / (double)a.size();
}

// "path:width:height" -> raw RGB8 file contents.
static bool loadRaw(const char* spec, std::vector<uint8_t>& px, int& w, int& h) {
    std::string s(spec);
    const size_t b = s.rfind(':'), a = b == std::string::npos ? b : s.rfind(':', b - 1);
    if (b == std::string::npos || a == std::string::npos) return false;
    w = std::atoi(s.substr(a + 1, b - a - 1).c_str());
    h = std::atoi(s.substr(b + 1).c_str());
    FILE* f = std::fopen(s.substr(0, a).c_str(), "rb");
    if (!f || w <= 0 || h <= 0) { if (f) std::fclose(f); return false; }
    px.assign((size_t)w * h * 3, 0);
    const size_t n = std::fread(px.data(), 1, px.size(), f);
    std::fclose(f);
    return n == px.size();
}

static void dump(const char* label, const std::vector<uint8_t>& v) {
    const char* pre = std::getenv("REF_DUMP");
    if (!pre) return;
    std::string path = std::string(pre) + "_" + label + ".rgb";
    if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(v.data(), 1, v.size(), f); std::fclose(f); }
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s dit.gguf llm.gguf vae cpu|gpu [plugin] [size]\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const bool gpuRaw = std::strcmp(argv[4], "gpu_raw") == 0;
    const bool gpu = std::strcmp(argv[4], "gpu") == 0 || gpuRaw;
    const char* plugin = argc > 5 ? argv[5] : nullptr;
    const int S = argc > 6 ? std::atoi(argv[6]) : 256;
    if (gpu && plugin == nullptr) { std::fprintf(stderr, "gpu needs the plugin path\n"); return 2; }

    AmevaSdLoadOptions o;
    ameva_sd_load_options_init(&o);
    o.diffusion_model_path = argv[1];
    o.llm_path = argv[2];
    o.vae_path = argv[3];
    o.budget_mode = !gpu ? AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY
                         : gpuRaw ? AMEVA_SD_GPU_BUDGET_MODE_DEFAULT : AMEVA_SD_GPU_BUDGET_MODE_GPU_THEN_CPU;
    o.budget_fraction = 0.90f;  // same as AmfyUI GPU_MEMORY_FRACTION
    // BUDGET_FRACTION=<f>: a smaller GPU budget (AmfyUI's debugFraction). With 0.90 and ~6.8 GB free the adapter judges the
    // 5.0 GB of FLUX.2 klein files + a 384 MiB compute reserve as "fits", keeps every weight on the GPU and the compute
    // buffers then fail with AC_ERR_OUT_OF_DEVICE_MEMORY (-202); a smaller fraction selects the per-phase time-share plan.
    if (const char* bf = std::getenv("BUDGET_FRACTION")) o.budget_fraction = (float)std::atof(bf);
    o.backend_plugin_path = plugin;
    o.n_threads = 4;
    AmevaSd* sd = nullptr;
    char err[512] = {0};
    memline("before load");
    auto t0 = std::chrono::steady_clock::now();
    AmevaSdStatus st = ameva_sd_load_ex(&o, &sd, err, sizeof(err));
    std::printf("load status=%d seconds=%.1f %s\n", (int)st,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), st ? err : "");
    memline("after load");
    if (st != AMEVA_SD_OK) return 1;

    std::vector<uint8_t> refA = makeRef(S, S, 0), refB = makeRef(S, S, 1);
    dump("in_refA", refA);
    dump("in_refB", refB);

    const size_t outBytes = (size_t)S * S * 3;
    auto base = [&](AmevaSdGenerateParams& g, const char* prompt) {
        ameva_sd_generate_params_init(&g);
        g.prompt = prompt;
        g.width = S; g.height = S; g.steps = 4; g.cfg_scale = 1.0f; g.seed = 42;
    };
    auto run = [&](AmevaSdGenerateParams& g, const char* label, std::vector<uint8_t>& out) {
        out.assign(outBytes, 0);
        auto t = std::chrono::steady_clock::now();
        AmevaSdStatus s = ameva_sd_generate(sd, &g, out.data(), out.size(), err, sizeof(err));
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        std::printf("gen %-12s status=%d fnv=%016llx seconds=%.1f%s%s\n", label, (int)s,
                    (unsigned long long)fnv1a(out.data(), out.size()), sec, s ? " err=" : "", s ? err : "");
        if (s == AMEVA_SD_OK) dump(label, out);
        memline(label);
        return s;
    };
    auto setRef = [&](AmevaSdGenerateParams& g, int i, const std::vector<uint8_t>& r) {
        g.ref_rgb[i] = r.data(); g.ref_width[i] = S; g.ref_height[i] = S; g.ref_size[i] = r.size();
    };
    auto fnv = [](const std::vector<uint8_t>& v) { return fnv1a(v.data(), v.size()); };

    // Refusals (a multi-file model reaches the per-image checks). REFUSE_ONLY=1 runs just these (no generation).
    auto runRefusals = [&]() {
        AmevaSdGenerateParams g;
        const char* promptEdit = "keep the layout, make it a watercolour painting";
        auto refuse = [&](AmevaSdGenerateParams& p, const char* what, const char* needle) {
            std::vector<uint8_t> tmp(outBytes);
            AmevaSdStatus s = ameva_sd_generate(sd, &p, tmp.data(), tmp.size(), err, sizeof(err));
            bool ok = s == AMEVA_SD_ERR_INVALID_ARGUMENT && std::strstr(err, needle) != nullptr;
            char d[200];
            std::snprintf(d, sizeof d, "(status=%d %.120s)", (int)s, err);
            check(ok, what, d);
        };
        base(g, promptEdit); g.ref_count = 4;
        for (int i = 0; i < 3; ++i) setRef(g, i, refA);
        refuse(g, "8a ref_count 4", "ref_count");
        base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA); g.ref_size[0] = refA.size() - 3;
        refuse(g, "8b buffer size mismatch", "ref_size");
        base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA); g.ref_rgb[0] = nullptr;
        refuse(g, "8c NULL pixel pointer", "NULL");
        base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA); g.ref_width[0] = 0;
        refuse(g, "8d width 0", "ref_width");
        base(g, promptEdit); g.ref_image_args = "preset=flux2";
        refuse(g, "8e ref_image_args without images", "without reference");
        base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA); g.ref_image_args = "preset";
        refuse(g, "8f malformed ref_image_args (no '=')", "ref_image_args");
    };
    if (std::getenv("REFUSE_ONLY") != nullptr) {
        runRefusals();
        ameva_sd_free(sd);
        std::printf("result %s (%d passed, %d failed)\n", g_fail == 0 ? "OK" : "BAD", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    const bool wfOnly = std::getenv("WF_ONLY") != nullptr;  // skip checks 1-9, run only the template scenario
    std::vector<uint8_t> plain, withA, withA2, withB, withAB, again;
    if (!wfOnly) {
    const char* prompt = "a photo of a colourful poster";
    const char* promptEdit = "keep the layout, make it a watercolour painting";

    AmevaSdGenerateParams g;
    base(g, prompt);
    check(run(g, "plain", plain) == AMEVA_SD_OK, "1 txt2img without reference");

    base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA);
    check(run(g, "refA", withA) == AMEVA_SD_OK, "2 one reference image");
    check(fnv(withA) != fnv(plain), "3 reference changes the output");

    base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refA);
    run(g, "refA_again", withA2);
    check(fnv(withA) == fnv(withA2), "4 same reference + seed is repeatable");

    base(g, promptEdit); g.ref_count = 1; setRef(g, 0, refB);
    check(run(g, "refB", withB) == AMEVA_SD_OK && fnv(withB) != fnv(withA), "5 different reference gives a different output");

    base(g, promptEdit); g.ref_count = 2; setRef(g, 0, refA); setRef(g, 1, refB);
    check(run(g, "refAB", withAB) == AMEVA_SD_OK && fnv(withAB) != fnv(withA), "6 two reference images");

    {   // The same prompt without reference, to compare distances to refA fairly.
        std::vector<uint8_t> promptOnly;
        base(g, promptEdit);
        run(g, "edit_noref", promptOnly);
        const double dRef = meanAbsDiff(withA, refA), dNo = meanAbsDiff(promptOnly, refA);
        char d[96];
        std::snprintf(d, sizeof d, "(|out-ref| with ref %.1f, without %.1f)", dRef, dNo);
        check(dRef >= 0 && dNo >= 0 && dRef < dNo, "7 reference pulls the output toward itself", d);
    }

    runRefusals();

    base(g, prompt);
    check(run(g, "plain_after", again) == AMEVA_SD_OK && fnv(again) == fnv(plain), "9 plain generation after refusals unchanged");
    }  // !wfOnly

    // Scenario of the public ComfyUI template "Flux.2 Klein 4B Distilled: Image Edit" (two references, handbag + logo):
    // euler, 4 steps, cfg 1, the template's seed; image 1 = handbag (its size is the output size), image 2 = logo.
    // WF1=<handbag.rgb>:<w>:<h>  WF2=<logo.rgb>:<w>:<h>  (raw RGB8, made from the template's example inputs)
    if (std::getenv("WF1") != nullptr && std::getenv("WF2") != nullptr) {
        std::vector<uint8_t> bag, logo;
        int bw = 0, bh = 0, lw = 0, lh = 0;
        const bool okLoad = loadRaw(std::getenv("WF1"), bag, bw, bh) && loadRaw(std::getenv("WF2"), logo, lw, lh);
        check(okLoad, "10 load template input images (handbag, logo)");
        if (okLoad) {
            const char* wp = "stylize the handbag in image1 with the colours and logo from image 2";
            auto wf = [&](const char* label, bool bagFirst, int nref, std::vector<uint8_t>& out) {
                AmevaSdGenerateParams w;
                ameva_sd_generate_params_init(&w);
                w.prompt = wp; w.width = bw; w.height = bh; w.steps = 4; w.cfg_scale = 1.0f;
                w.sampler = "euler"; w.seed = 432262096973495LL;
                w.ref_count = nref;
                const std::vector<uint8_t>* px[2] = {bagFirst ? &bag : &logo, bagFirst ? &logo : &bag};
                const int pw[2] = {bagFirst ? bw : lw, bagFirst ? lw : bw};
                const int ph[2] = {bagFirst ? bh : lh, bagFirst ? lh : bh};
                for (int i = 0; i < nref; ++i) {
                    w.ref_rgb[i] = px[i]->data(); w.ref_width[i] = pw[i]; w.ref_height[i] = ph[i]; w.ref_size[i] = px[i]->size();
                }
                out.assign((size_t)bw * bh * 3, 0);
                auto t = std::chrono::steady_clock::now();
                AmevaSdStatus s = ameva_sd_generate(sd, &w, out.data(), out.size(), err, sizeof(err));
                std::printf("gen %-12s status=%d fnv=%016llx seconds=%.1f%s%s\n", label, (int)s,
                            (unsigned long long)fnv1a(out.data(), out.size()),
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(), s ? " err=" : "", s ? err : "");
                if (s == AMEVA_SD_OK) dump(label, out);
                memline(label);
                return s;
            };
            std::vector<uint8_t> two, swapped, bagOnly, none;
            check(wf("wf_bag_logo", true, 2, two) == AMEVA_SD_OK, "11 template edit: image1=handbag, image2=logo");
            check(wf("wf_logo_bag", false, 2, swapped) == AMEVA_SD_OK && fnv(swapped) != fnv(two), "12 reference order matters (image1/image2)");
            check(wf("wf_bag_only", true, 1, bagOnly) == AMEVA_SD_OK && fnv(bagOnly) != fnv(two), "13 second reference is consumed");
            check(wf("wf_noref", true, 0, none) == AMEVA_SD_OK && fnv(none) != fnv(bagOnly), "14 reference changes the output (same prompt)");
            const double dRef = meanAbsDiff(two, bag), dNo = meanAbsDiff(none, bag);
            char d[96];
            std::snprintf(d, sizeof d, "(|out-handbag| with refs %.1f, without %.1f)", dRef, dNo);
            check(dRef >= 0 && dNo >= 0 && dRef < dNo, "15 edited image stays closer to the handbag than a no-reference image", d);
        }
    }

    ameva_sd_free(sd);
    std::printf("result %s (%d passed, %d failed)\n", g_fail == 0 ? "OK" : "BAD", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// Build (NDK 26 clang, aarch64; link against the libameva_sd_jni.so that carries the C API):
//   clang++ --target=aarch64-linux-android29 -std=c++17 -O1 -Iinclude tests/ameva_sd_ref_flux.cpp \
//     <lib dir>/libameva_sd_jni.so -static-libstdc++ -llog -ldl -o ameva_sd_ref_flux
