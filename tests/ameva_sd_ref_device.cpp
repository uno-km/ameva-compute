// ameva_sd_ref_device: on-device regression and refusal check of the SD C API against the real library
// (libameva_sd_jni.so, real stable-diffusion.cpp, a real single-file model).
//
// The same source is built twice: against the previous header + library (baseline, without reference images)
// and against the new header + library. Output hashes of the same seed must match; the new build must also
// refuse reference images on a single-file checkpoint with AMEVA_SD_ERR_INVALID_ARGUMENT.
//
//   ameva_sd_ref_device <model.gguf> <cpu|gpu> [plugin libggml-ameva-sd.so]
//
// Prints one line per generation: "gen <label> status=<n> fnv=<hex> bytes=<n> seconds=<s>".
#include "ameva_sd_api.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf cpu|gpu [plugin]\n", argv[0]); return 2; }
    const std::string model = argv[1];
    const bool gpu = std::strcmp(argv[2], "gpu") == 0;
    const char* plugin = argc > 3 ? argv[3] : nullptr;
    if (gpu && plugin == nullptr) { std::fprintf(stderr, "gpu needs the plugin path\n"); return 2; }

    AmevaSdLoadOptions o;
    ameva_sd_load_options_init(&o);
    o.model_path = model.c_str();
    o.budget_mode = gpu ? AMEVA_SD_GPU_BUDGET_MODE_DEFAULT : AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY;
    o.backend_plugin_path = plugin;
    o.n_threads = 4;
    AmevaSd* sd = nullptr;
    char err[512] = {0};
    AmevaSdStatus st = ameva_sd_load_ex(&o, &sd, err, sizeof(err));
    if (st != AMEVA_SD_OK) { std::printf("load FAILED status=%d %s\n", (int)st, err); return 1; }

    const int W = 128, H = 128;
    std::vector<uint8_t> out((size_t)W * H * 3);
    int bad = 0;
    const int64_t seeds[2] = {42, 7};
    for (int s = 0; s < 2; ++s) {
        AmevaSdGenerateParams g;
        ameva_sd_generate_params_init(&g);
        g.prompt = "a red apple on a wooden table";
        g.width = W; g.height = H; g.steps = 1; g.cfg_scale = 1.0f; g.seed = seeds[s];
        auto t0 = std::chrono::steady_clock::now();
        st = ameva_sd_generate(sd, &g, out.data(), out.size(), err, sizeof(err));
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("gen seed=%lld status=%d fnv=%016llx bytes=%zu seconds=%.1f%s%s\n", (long long)seeds[s], (int)st,
                    (unsigned long long)fnv1a(out.data(), out.size()), out.size(), sec, st ? " err=" : "", st ? err : "");
        if (st != AMEVA_SD_OK) ++bad;
        // REF_DUMP=<prefix>: raw RGB8 of every generation (<prefix>_<seed>.rgb) for offline comparison.
        if (st == AMEVA_SD_OK && std::getenv("REF_DUMP") != nullptr) {
            std::string path = std::string(std::getenv("REF_DUMP")) + "_" + std::to_string((long long)seeds[s]) + ".rgb";
            if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(out.data(), 1, out.size(), f); std::fclose(f); }
        }
    }

#ifdef AMEVA_SD_MAX_REF_IMAGES
    // Reference images on a single-file checkpoint must be refused, never dropped.
    std::vector<uint8_t> ref((size_t)8 * 8 * 3, 9);
    {
        AmevaSdGenerateParams g;
        ameva_sd_generate_params_init(&g);
        g.prompt = "x"; g.width = W; g.height = H; g.steps = 1; g.cfg_scale = 1.0f; g.seed = 1;
        g.ref_count = 1; g.ref_rgb[0] = ref.data(); g.ref_width[0] = 8; g.ref_height[0] = 8; g.ref_size[0] = ref.size();
        st = ameva_sd_generate(sd, &g, out.data(), out.size(), err, sizeof(err));
        std::printf("ref single-file status=%d (want %d) err=%s\n", (int)st, (int)AMEVA_SD_ERR_INVALID_ARGUMENT, err);
        if (st != AMEVA_SD_ERR_INVALID_ARGUMENT) ++bad;
        g.ref_count = 4;
        st = ameva_sd_generate(sd, &g, out.data(), out.size(), err, sizeof(err));
        std::printf("ref count 4  status=%d (want %d) err=%s\n", (int)st, (int)AMEVA_SD_ERR_INVALID_ARGUMENT, err);
        if (st != AMEVA_SD_ERR_INVALID_ARGUMENT) ++bad;
    }
    // The refused calls must not have disturbed the next normal generation.
    {
        AmevaSdGenerateParams g;
        ameva_sd_generate_params_init(&g);
        g.prompt = "a red apple on a wooden table";
        g.width = W; g.height = H; g.steps = 1; g.cfg_scale = 1.0f; g.seed = 42;
        st = ameva_sd_generate(sd, &g, out.data(), out.size(), err, sizeof(err));
        std::printf("gen seed=42 after refusals status=%d fnv=%016llx\n", (int)st,
                    (unsigned long long)fnv1a(out.data(), out.size()));
        if (st != AMEVA_SD_OK) ++bad;
    }
#else
    std::printf("(baseline build: header without reference images)\n");
#endif
    ameva_sd_free(sd);
    std::printf("result %s\n", bad == 0 ? "OK" : "BAD");
    return bad == 0 ? 0 : 1;
}
