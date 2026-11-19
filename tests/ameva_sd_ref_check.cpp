// ameva_sd_ref_check: argument validation and forwarding of reference images in ameva_sd_api.cpp.
//
// The adapter is compiled as-is. Only its boundary to stable-diffusion.cpp is replaced: new_sd_ctx() and
// generate_image() are recording test doubles (every other sd.cpp / ggml symbol the adapter references is an
// aborting stub in ameva_sd_ref_check_stubs.c, so an unexpected call fails loudly). No model is loaded and no
// image is generated: this checks what the adapter accepts, refuses and passes on, not image content.
//
// Build (aarch64 Android, NDK clang) and run on a device: see tests/README-style line at the end of this file.
#include "ameva_sd_api.h"
#include "stable-diffusion.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- recording test doubles --------------------------------------------------------------------------------
static sd_ctx_params_t g_load_params;
static int g_generate_calls = 0;
static int g_got_ref_count = -1;
static const sd_image_t* g_got_ref_ptr = nullptr;
static std::string g_got_args;
static std::vector<std::vector<uint8_t>> g_got_pixels;
static std::vector<std::pair<uint32_t, uint32_t>> g_got_dims;
static std::vector<const uint8_t*> g_got_data_ptr;

extern "C" {
void sd_set_log_callback(sd_log_cb_t, void*) {}
void sd_set_progress_callback(sd_progress_cb_t, void*) {}
void sd_cancel_generation(sd_ctx_t*, enum sd_cancel_mode_t) {}
void sd_ctx_params_init(sd_ctx_params_t* p) { std::memset(p, 0, sizeof(*p)); }
void sd_img_gen_params_init(sd_img_gen_params_t* p) { std::memset(p, 0, sizeof(*p)); p->ref_image_args = ""; }
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* p) {
    g_load_params = *p;
    // Strings are only valid during the call in the product; copy what the checks need.
    return reinterpret_cast<sd_ctx_t*>(static_cast<uintptr_t>(0x1000));
}
void free_sd_ctx(sd_ctx_t*) {}
void free_sd_images(sd_image_t*, int) {}
bool generate_image(sd_ctx_t*, const sd_img_gen_params_t* p, sd_image_t** out, int* n) {
    ++g_generate_calls;
    g_got_ref_count = p->ref_images_count;
    g_got_ref_ptr = p->ref_images;
    g_got_args = p->ref_image_args != nullptr ? p->ref_image_args : "(null)";
    g_got_pixels.clear();
    g_got_dims.clear();
    g_got_data_ptr.clear();
    for (int i = 0; i < p->ref_images_count; ++i) {
        const sd_image_t& im = p->ref_images[i];
        g_got_dims.emplace_back(im.width, im.height);
        g_got_data_ptr.push_back(im.data);
        if (im.channel != 3) std::fprintf(stderr, "channel != 3\n");
        g_got_pixels.emplace_back(im.data, im.data + (size_t)im.width * im.height * 3);
    }
    *out = nullptr;
    *n = 0;
    return false; // GENERATE_FAILED: reached the sd.cpp boundary, which is all this test needs
}
}

// ---- tiny harness ------------------------------------------------------------------------------------------
static int g_fail = 0, g_pass = 0;
#define CHECK(cond, what)                                                                  \
    do {                                                                                   \
        if (cond) { ++g_pass; }                                                            \
        else { ++g_fail; std::printf("FAIL  %s  (line %d)\n", what, __LINE__); }           \
    } while (0)

static AmevaSd* load(bool multi_file, bool with_vision = false, bool vision_without_llm = false,
                     AmevaSdStatus* st_out = nullptr, char* err = nullptr, size_t errn = 0) {
    AmevaSdLoadOptions o;
    ameva_sd_load_options_init(&o);
    o.budget_mode = AMEVA_SD_GPU_BUDGET_MODE_CPU_ONLY;
    if (multi_file) o.diffusion_model_path = "/nonexistent/diffusion.gguf"; else o.model_path = "/nonexistent/model.gguf";
    if (!vision_without_llm && with_vision) o.llm_path = "/nonexistent/llm.gguf";
    if (with_vision || vision_without_llm) o.llm_vision_path = "/nonexistent/mmproj.gguf";
    AmevaSd* sd = nullptr;
    char e[512] = {0};
    AmevaSdStatus st = ameva_sd_load_ex(&o, &sd, err ? err : e, err ? errn : sizeof(e));
    if (st_out) *st_out = st;
    return sd;
}

static const int W = 64, H = 64;
static std::vector<uint8_t> g_out(W * H * 3);
static std::vector<uint8_t> g_img[4];

static void base(AmevaSdGenerateParams* g) {
    ameva_sd_generate_params_init(g);
    g->prompt = "x";
    g->width = W;
    g->height = H;
    g->steps = 1;
}
static AmevaSdStatus gen(AmevaSd* sd, const AmevaSdGenerateParams& g, std::string* msg = nullptr) {
    char e[512] = {0};
    AmevaSdStatus st = ameva_sd_generate(sd, &g, g_out.data(), g_out.size(), e, sizeof(e));
    if (msg) *msg = e;
    return st;
}
static void ref(AmevaSdGenerateParams* g, int i, int w, int h, uint8_t fill) {
    g_img[i].assign((size_t)w * h * 3, fill);
    g->ref_rgb[i] = g_img[i].data();
    g->ref_width[i] = w;
    g->ref_height[i] = h;
    g->ref_size[i] = g_img[i].size();
}

int main() {
    std::string m;
    AmevaSdStatus st;

    // -- load: llm_vision_path
    load(true, false, true, &st);
    CHECK(st == AMEVA_SD_ERR_INVALID_ARGUMENT, "load: llm_vision_path without llm_path refused");
    std::memset(&g_load_params, 0, sizeof(g_load_params));
    AmevaSd* vis = load(true, true, false, &st);
    CHECK(st == AMEVA_SD_OK && vis != nullptr, "load: llm + llm_vision_path accepted");
    CHECK(g_load_params.llm_vision_path != nullptr && std::string(g_load_params.llm_vision_path) == "/nonexistent/mmproj.gguf",
          "load: llm_vision_path forwarded to sd_ctx_params_t");
    ameva_sd_free(vis);

    AmevaSd* multi = load(true, false, false, &st);
    CHECK(st == AMEVA_SD_OK && multi != nullptr, "load: multi-file model");
    CHECK(g_load_params.llm_vision_path == nullptr, "load: llm_vision_path stays NULL when not given");
    AmevaSd* single = load(false, false, false, &st);
    CHECK(st == AMEVA_SD_OK && single != nullptr, "load: single-file checkpoint");

    // -- default path (no reference images): sd.cpp receives none
    {
        AmevaSdGenerateParams g; base(&g);
        g_generate_calls = 0;
        st = gen(multi, g);
        CHECK(g_generate_calls == 1 && st == AMEVA_SD_ERR_GENERATE_FAILED, "no refs: reaches generate_image");
        CHECK(g_got_ref_count == 0 && g_got_ref_ptr == nullptr && g_got_args == "", "no refs: ref_images untouched");
        g_generate_calls = 0;
        st = gen(single, g);
        CHECK(g_generate_calls == 1 && g_got_ref_count == 0, "no refs: single-file checkpoint unaffected");
    }

    // -- refusals: generate_image must not be reached
    struct Case { const char* name; AmevaSd* sd; };
    auto refused = [&](const char* name, AmevaSd* sd, const AmevaSdGenerateParams& g, const char* needle) {
        g_generate_calls = 0;
        std::string msg;
        AmevaSdStatus s = gen(sd, g, &msg);
        const bool okc = s == AMEVA_SD_ERR_INVALID_ARGUMENT && g_generate_calls == 0 &&
                         msg.find(needle) != std::string::npos;
        if (!okc) std::printf("  %s: status=%d calls=%d msg='%s'\n", name, (int)s, g_generate_calls, msg.c_str());
        CHECK(okc, name);
    };
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 4;
        for (int i = 0; i < 3; ++i) ref(&g, i, 8, 8, 1);
        refused("ref_count 4 refused", multi, g, "ref_count");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = -1;
        refused("ref_count -1 refused", multi, g, "ref_count");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1;
        refused("NULL ref_rgb refused", multi, g, "NULL");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1); g.ref_size[0] -= 1;
        refused("ref_size mismatch refused", multi, g, "ref_size");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1); g.ref_width[0] = 0;
        refused("width 0 refused", multi, g, "ref_width");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1); g.ref_height[0] = 16385;
        refused("height 16385 refused", multi, g, "ref_width");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 2; ref(&g, 0, 8, 8, 1); ref(&g, 1, 8, 8, 2); g.ref_rgb[1] = nullptr;
        refused("NULL second image refused", multi, g, "NULL");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1);
        refused("single-file checkpoint refused", single, g, "diffusion_model_path");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1); g.ref_image_args = "preset";
        refused("ref_image_args without '=' refused", multi, g, "key=value");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1); g.ref_image_args = "a=1,b=";
        refused("ref_image_args empty value refused", multi, g, "key=value");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_image_args = "preset=qwen";
        refused("ref_image_args without images refused", multi, g, "without reference");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 8, 8, 1);
        g_generate_calls = 0;
        std::string msg;
        int framesOut = 0;
        AmevaSdStatus s = ameva_sd_generate_video(multi, &g, 4, 8, g_out.data(), 4 * W * H * 3, &framesOut, nullptr, 0);
        CHECK(s == AMEVA_SD_ERR_INVALID_ARGUMENT && g_generate_calls == 0, "video with reference images refused");
    }

    // -- accepted: values reach sd.cpp, pixels are owned copies
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 3;
        ref(&g, 0, 8, 6, 11); ref(&g, 1, 5, 5, 22); ref(&g, 2, 16, 4, 33);
        g.ref_image_args = "preset=qwen_layered,pass_to_vlm=true";
        g_generate_calls = 0;
        st = gen(multi, g);
        CHECK(g_generate_calls == 1 && st == AMEVA_SD_ERR_GENERATE_FAILED, "3 refs: reaches generate_image");
        CHECK(g_got_ref_count == 3, "3 refs: count forwarded");
        CHECK(g_got_args == "preset=qwen_layered,pass_to_vlm=true", "3 refs: ref_image_args forwarded");
        bool dims = g_got_dims.size() == 3 && g_got_dims[0] == std::make_pair(8u, 6u) &&
                    g_got_dims[1] == std::make_pair(5u, 5u) && g_got_dims[2] == std::make_pair(16u, 4u);
        CHECK(dims, "3 refs: widths and heights forwarded in order");
        bool pix = g_got_pixels.size() == 3 && g_got_pixels[0] == g_img[0] && g_got_pixels[1] == g_img[1] &&
                   g_got_pixels[2] == g_img[2];
        CHECK(pix, "3 refs: pixel bytes forwarded unchanged");
        bool owned = g_got_data_ptr.size() == 3 && g_got_data_ptr[0] != g_img[0].data() &&
                     g_got_data_ptr[1] != g_img[1].data() && g_got_data_ptr[2] != g_img[2].data();
        CHECK(owned, "3 refs: adapter owns its copies (not the caller's buffers)");
    }
    {
        AmevaSdGenerateParams g; base(&g); g.ref_count = 1; ref(&g, 0, 4, 4, 7);
        g_generate_calls = 0;
        st = gen(multi, g);
        CHECK(g_generate_calls == 1 && g_got_ref_count == 1 && g_got_args == "", "1 ref, default args: empty args forwarded");
    }

    ameva_sd_free(multi);
    ameva_sd_free(single);
    std::printf("ameva_sd_ref_check: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// Build (from the repo root, NDK 26 clang, aarch64) and run on an Android device (Termux):
//   clang++ --target=aarch64-linux-android29 -std=c++17 -O1 -Iinclude -Ithird_party/stable-diffusion.cpp/include \
//     -Ithird_party/stable-diffusion.cpp/ggml/include -Iadapters/llama \
//     tests/ameva_sd_ref_check.cpp tests/ameva_sd_ref_check_stubs.c adapters/llama/ameva_sd_api.cpp \
//     -static-libstdc++ -llog -ldl -o ameva_sd_ref_check
