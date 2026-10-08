// SPDX-License-Identifier: Apache-2.0
// AMEVA VAE graph check: op census and output comparison of the decoder graphs sd.cpp builds for a 2D VAE.
//
// The graphs are rebuilt here node for node from the source of stable-diffusion.cpp at commit 50d6405
// (src/model/vae/auto_encoder_kl.hpp: ResnetBlock, AttnBlock with 1x1 convolutions, Decoder; src/model/vae/tae.hpp:
// TAEBlock, TinyDecoder; src/core/ggml_extend.hpp: ggml_ext_conv_2d, ggml_ext_group_norm, ggml_ext_attention_ext
// without flash attention; src/model/common/block.hpp: UpSampleBlock) with random weights of the stored types
// (convolution kernels f16, every other parameter f32). They are not read from a running sd.cpp: a change of that
// source needs the same change here. The compute memory comes from ggml_gallocr as in sd.cpp (GGMLRunner), so
// results are stored in place of their sources wherever ggml allows it.
//
// Variants:
//   sd1          AutoencoderKL decoder of SD 1.x / 2.x (z = 4, post_quant_conv), ggml_conv_2d_direct
//                (sd.cpp --vae-conv-direct)
//   flux         the same decoder with z = 16 and no post_quant_conv (Flux, Z-Image, SD3 latent format)
//   taesd        TinyDecoder (taesd, z = 4), ggml_conv_2d_direct
//   sd1-im2col   the sd1 decoder with ggml_conv_2d (sd.cpp's default: IM2COL + MUL_MAT)
// Census (every variant): per distinct node signature (op, result type, source types, contiguity, broadcast) the
// number of nodes, one example with its shapes, and whether ggml_backend_supports_op of the AMEVA backend accepts
// it (the reason of a decline is printed by the backend on stderr).
// Comparison (sd1 and taesd; flux as well with AMEVA_VAE_CHECK_ALL=1): the graph is computed by the AMEVA backend
// (f16 kernels, as sd.cpp stores them) and by ggml's CPU backend from the same parameter values and input, and the
// outputs are compared. The CPU reference holds the kernels as f32 (the same f16-representable values): with an f16
// kernel ggml's CPU convolution rounds every activation to f16 before its product
// (ggml_compute_forward_conv_2d_impl, GGML_CPU_FP32_TO_FP16), which the AMEVA kernel does not do. The difference
// from that f16 CPU path is printed as well (rel_rms_cpu_f16, not judged). Required: no node of the three direct
// variants is declined, graph_compute succeeds, no NaN, and the RMS difference from the f32-kernel CPU output is at
// most 1e-3 of its RMS. sd1-im2col is reported only (its IM2COL nodes have no AMEVA kernel).
//
// AMEVA_VAE_CHECK_LATENT=WxH (default 16x16) sets the latent size; the image is 8 times that.
// AMEVA_CPU_BACKEND_LIB=<path of libggml-cpu.so> loads the CPU backend when the registry does not hold one.
// Times are wall times of one graph_compute each, printed with the conditions; they are not a benchmark.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ameva.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

enum ParamKind { P_CONV_W, P_CONV_B, P_NORM_W, P_NORM_B, P_INPUT };

struct Net {
    struct ggml_context* w = nullptr; // parameters and the input
    struct ggml_context* g = nullptr; // graph nodes
    bool direct = true;
    enum ggml_type kernel_type = GGML_TYPE_F16; // sd.cpp: f16; the CPU reference: f32 with the same values
    std::vector<std::pair<struct ggml_tensor*, ParamKind>> params; // creation order: the same for every backend
    struct ggml_tensor* param(ParamKind kind, enum ggml_type type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
        struct ggml_tensor* t = ggml_new_tensor_4d(w, type, n0, n1, n2, n3);
        params.emplace_back(t, kind);
        return t;
    }
};

// Conv2d::forward -> ggml_ext_conv_2d
static struct ggml_tensor* conv(Net& n, struct ggml_tensor* x, int64_t ic, int64_t oc, int k, bool bias = true) {
    struct ggml_tensor* w = n.param(P_CONV_W, n.kernel_type, k, k, ic, oc);
    const int p = k / 2;
    x = n.direct ? ggml_conv_2d_direct(n.g, w, x, 1, 1, p, p, 1, 1) : ggml_conv_2d(n.g, w, x, 1, 1, p, p, 1, 1);
    if (bias) {
        struct ggml_tensor* b = n.param(P_CONV_B, GGML_TYPE_F32, oc);
        x = ggml_add_inplace(n.g, x, ggml_reshape_4d(n.g, b, 1, 1, oc, 1));
    }
    return x;
}

// GroupNorm32::forward -> ggml_ext_group_norm
static struct ggml_tensor* group_norm(Net& n, struct ggml_tensor* x, int64_t c) {
    struct ggml_tensor* w = ggml_reshape_4d(n.g, n.param(P_NORM_W, GGML_TYPE_F32, c), 1, 1, c, 1);
    struct ggml_tensor* b = ggml_reshape_4d(n.g, n.param(P_NORM_B, GGML_TYPE_F32, c), 1, 1, c, 1);
    x = ggml_group_norm(n.g, x, 32, 1e-6f);
    x = ggml_mul_inplace(n.g, x, w);
    return ggml_add_inplace(n.g, x, b);
}

static struct ggml_tensor* resnet(Net& n, struct ggml_tensor* x, int64_t in, int64_t out) {
    struct ggml_tensor* h = group_norm(n, x, in);
    h = ggml_silu_inplace(n.g, h);
    h = conv(n, h, in, out, 3);
    h = group_norm(n, h, out);
    h = ggml_silu_inplace(n.g, h);
    h = conv(n, h, out, out, 3);
    if (out != in) x = conv(n, x, in, out, 1); // nin_shortcut
    return ggml_add(n.g, h, x);
}

static struct ggml_tensor* ext_cont(Net& n, struct ggml_tensor* x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(n.g, x);
}

// AttnBlock::forward (use_linear == false) with ggml_ext_attention_ext(q, k, v, n_head = 1, mask = NULL,
// skip_reshape = false, flash_attn = false)
static struct ggml_tensor* attn(Net& n, struct ggml_tensor* x, int64_t c) {
    struct ggml_tensor* h_ = group_norm(n, x, c);
    const int64_t N = h_->ne[3], h = h_->ne[1], w = h_->ne[0];
    struct ggml_tensor* qkv[3];
    for (int i = 0; i < 3; ++i) {
        struct ggml_tensor* t = conv(n, h_, c, c, 1);
        t = ggml_cont(n.g, ggml_permute(n.g, t, 1, 2, 0, 3));
        qkv[i] = ggml_reshape_3d(n.g, t, c, h * w, N);
    }
    struct ggml_tensor *q = qkv[0], *k = qkv[1], *v = qkv[2];
    const int64_t L_q = q->ne[1], L_k = k->ne[1], d_head = q->ne[0], n_head = 1, n_kv_head = k->ne[0] / d_head;
    q = ggml_reshape_4d(n.g, q, d_head, n_head, L_q, N);
    q = ext_cont(n, ggml_permute(n.g, q, 0, 2, 1, 3));
    q = ggml_reshape_3d(n.g, q, d_head, L_q, n_head * N);
    k = ggml_reshape_4d(n.g, k, d_head, n_kv_head, L_k, N);
    k = ext_cont(n, ggml_permute(n.g, k, 0, 2, 1, 3));
    k = ggml_reshape_3d(n.g, k, d_head, L_k, n_kv_head * N);
    v = ggml_reshape_4d(n.g, v, d_head, n_kv_head, L_k, N);
    v = ext_cont(n, ggml_permute(n.g, v, 1, 2, 0, 3));
    v = ggml_reshape_3d(n.g, v, L_k, d_head, n_kv_head * N);
    struct ggml_tensor* kq = ggml_mul_mat(n.g, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_scale_inplace(n.g, kq, 1.0f / sqrtf((float)d_head));
    kq = ggml_soft_max_inplace(n.g, kq);
    struct ggml_tensor* kqv = ggml_mul_mat(n.g, v, kq);
    kqv = ggml_reshape_4d(n.g, kqv, d_head, L_q, n_head, N);
    kqv = ggml_permute(n.g, kqv, 0, 2, 1, 3);
    kqv = ext_cont(n, kqv);
    kqv = ggml_reshape_3d(n.g, kqv, d_head * n_head, L_q, N);
    h_ = ggml_cont(n.g, ggml_permute(n.g, kqv, 1, 0, 2, 3));
    h_ = ggml_reshape_4d(n.g, h_, w, h, c, N);
    h_ = conv(n, h_, c, c, 1); // proj_out
    return ggml_add(n.g, h_, x);
}

// AutoEncoderKLModel::decode -> Decoder::forward: ch = 128, ch_mult = {1, 2, 4, 4}, num_res_blocks = 2
static struct ggml_tensor* decoder_kl(Net& n, struct ggml_tensor* z, int64_t zc, bool quant) {
    const int ch = 128, ch_mult[4] = { 1, 2, 4, 4 };
    if (quant) z = conv(n, z, zc, zc, 1); // post_quant_conv
    int64_t block_in = ch * ch_mult[3];
    struct ggml_tensor* h = conv(n, z, zc, block_in, 3); // conv_in
    h = resnet(n, h, block_in, block_in);
    h = attn(n, h, block_in);
    h = resnet(n, h, block_in, block_in);
    for (int i = 3; i >= 0; --i) {
        const int64_t block_out = ch * ch_mult[i];
        for (int j = 0; j < 3; ++j) {
            h = resnet(n, h, block_in, block_out);
            block_in = block_out;
        }
        if (i != 0) { // UpSampleBlock
            h = ggml_upscale(n.g, h, 2, GGML_SCALE_MODE_NEAREST);
            h = conv(n, h, block_in, block_in, 3);
        }
    }
    h = group_norm(n, h, block_in);
    h = ggml_silu_inplace(n.g, h);
    return conv(n, h, block_in, 3, 3); // conv_out
}

static struct ggml_tensor* tae_block(Net& n, struct ggml_tensor* x) {
    struct ggml_tensor* h = conv(n, x, 64, 64, 3);
    h = ggml_relu_inplace(n.g, h);
    h = conv(n, h, 64, 64, 3);
    h = ggml_relu_inplace(n.g, h);
    h = conv(n, h, 64, 64, 3);
    h = ggml_add(n.g, h, x);
    return ggml_relu_inplace(n.g, h);
}

// TinyDecoder::forward (num_blocks = 3, channels = 64)
static struct ggml_tensor* decoder_tae(Net& n, struct ggml_tensor* z, int64_t zc) {
    struct ggml_tensor* h = ggml_scale(n.g, z, 1.0f / 3.0f);
    h = ggml_tanh_inplace(n.g, h);
    h = ggml_scale(n.g, h, 3.0f);
    h = conv(n, h, zc, 64, 3);
    h = ggml_relu_inplace(n.g, h);
    for (int level = 0; level < 3; ++level) {
        for (int i = 0; i < 3; ++i) h = tae_block(n, h);
        h = ggml_upscale(n.g, h, 2, GGML_SCALE_MODE_NEAREST);
        h = conv(n, h, 64, 64, 3, false);
    }
    h = tae_block(n, h);
    return conv(n, h, 64, 3, 3);
}

struct Variant {
    const char* name;
    bool tae, direct, quant;
    int zc;
    bool compare;
};

static uint32_t g_seed;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

struct Built {
    Net net;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t ga = nullptr;
    struct ggml_cgraph* gf = nullptr;
    struct ggml_tensor* out = nullptr;
    size_t compute_bytes = 0, param_bytes = 0;
    ~Built() {
        if (ga) ggml_gallocr_free(ga);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (net.g) ggml_free(net.g);
        if (net.w) ggml_free(net.w);
    }
};

// Builds the graph of v for a latent of lw x lh on `be`, with the parameters and the input of seed 0x1234567. The
// kernels are stored as kernel_type; their values are f16 numbers in either case.
static const char* build(Built& b, const Variant& v, ggml_backend_t be, int lw, int lh,
                         enum ggml_type kernel_type = GGML_TYPE_F16) {
    const size_t max_nodes = 4096;
    struct ggml_init_params wp = { 1024 * ggml_tensor_overhead(), nullptr, true };
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
    b.net.w = ggml_init(wp);
    b.net.g = ggml_init(gp);
    b.net.direct = v.direct;
    b.net.kernel_type = kernel_type;
    struct ggml_tensor* z = b.net.param(P_INPUT, GGML_TYPE_F32, lw, lh, v.zc, 1);
    b.out = v.tae ? decoder_tae(b.net, z, v.zc) : decoder_kl(b.net, z, v.zc, v.quant);
    b.gf = ggml_new_graph_custom(b.net.g, max_nodes, false);
    ggml_build_forward_expand(b.gf, b.out);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    b.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(b.net.w, buft);
    if (!b.wbuf) return "parameter allocation";
    b.param_bytes = ggml_backend_buffer_get_size(b.wbuf);
    g_seed = 0x1234567u;
    std::vector<float> f;
    std::vector<ggml_fp16_t> h;
    for (auto& p : b.net.params) {
        struct ggml_tensor* t = p.first;
        const size_t n = (size_t)ggml_nelements(t);
        f.resize(n);
        // kernels: variance 1 / fan-in, so that activations keep their scale through the layers
        const float fan_in = (float)(t->ne[0] * t->ne[1] * t->ne[2]);
        for (float& x : f) {
            const float r = rnd();
            x = p.second == P_CONV_W ? r * sqrtf(3.0f / fan_in)
                : p.second == P_NORM_W ? 1.0f + 0.2f * r
                : p.second == P_INPUT ? 2.0f * r : 0.1f * r;
        }
        if (p.second == P_CONV_W) {
            h.resize(n);
            ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)n);
            if (t->type == GGML_TYPE_F16) {
                ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
            } else {
                ggml_fp16_to_fp32_row(h.data(), f.data(), (int64_t)n);
                ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
            }
        } else {
            ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
        }
    }
    b.ga = ggml_gallocr_new(buft);
    if (!b.ga || !ggml_gallocr_alloc_graph(b.ga, b.gf)) return "compute allocation";
    b.compute_bytes = ggml_gallocr_get_buffer_size(b.ga, 0);
    return nullptr;
}

static std::string shape(const struct ggml_tensor* t) {
    char s[96];
    snprintf(s, sizeof(s), "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
             (long long)t->ne[3]);
    return s;
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
        if (i > 0 && !ggml_are_same_shape(x, t->src[0])) s += "(bcast)";
    }
    return s;
}

static bool is_metadata(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

// Prints the census of gf; returns the number of nodes the AMEVA backend declines.
static int census(const Variant& v, ggml_backend_t be, struct ggml_cgraph* gf) {
    struct Entry {
        int count = 0, declined = 0;
        std::string example;
    };
    std::map<std::string, Entry> sig;
    std::vector<std::string> order;
    int compute = 0, declined = 0;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* t = ggml_graph_node(gf, i);
        if (is_metadata(t->op)) continue;
        ++compute;
        const std::string s = signature(t);
        Entry& e = sig[s];
        if (e.count++ == 0) {
            order.push_back(s);
            for (int k = 0; k < GGML_MAX_SRC && t->src[k]; ++k) e.example += (k ? " x " : "") + shape(t->src[k]);
            e.example += " -> " + shape(t);
        }
        if (!ggml_backend_supports_op(be, t)) {
            ++e.declined;
            ++declined;
        }
    }
    for (const std::string& s : order) {
        const Entry& e = sig[s];
        printf("VAE_CENSUS variant=%s nodes=%d op=\"%s\" example=\"%s\" ameva=%s\n", v.name, e.count, s.c_str(),
               e.example.c_str(), e.declined == 0 ? "yes" : e.declined == e.count ? "NO" : "PARTLY");
    }
    printf("VAE_CENSUS variant=%s graph_nodes=%d compute_nodes=%d signatures=%zu declined_nodes=%d\n", v.name, n_nodes,
           compute, order.size(), declined);
    return declined;
}

static double ms_between(const struct timespec& t0, const struct timespec& t1) {
    return (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

static ggml_backend_t cpu_backend(void) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        if (const char* lib = getenv("AMEVA_CPU_BACKEND_LIB")) ggml_backend_load(lib);
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!cpu) return nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) set_threads(cpu, 4);
    return cpu;
}

int main() {
    int lw = 16, lh = 16;
    if (const char* e = getenv("AMEVA_VAE_CHECK_LATENT")) {
        if (sscanf(e, "%dx%d", &lw, &lh) != 2 || lw < 1 || lh < 1 || lw > 128 || lh > 128) {
            fprintf(stderr, "AMEVA_VAE_CHECK_LATENT must be WxH with 1..128 each (got \"%s\")\n", e);
            return 2;
        }
    }
    const char* all_env = getenv("AMEVA_VAE_CHECK_ALL");
    const bool all = all_env && !strcmp(all_env, "1");
    ggml_backend_t be = ggml_backend_ameva_init(0);
    if (!be) {
        fprintf(stderr, "ggml_backend_ameva_init failed\n");
        return 2;
    }
    ggml_backend_t cpu = cpu_backend();
    if (!cpu) {
        fprintf(stderr, "no CPU backend for the reference (set AMEVA_CPU_BACKEND_LIB)\n");
        return 2;
    }
    const Variant variants[] = {
        { "sd1", false, true, true, 4, true },
        { "flux", false, true, false, 16, all },
        { "taesd", true, true, false, 4, true },
        { "sd1-im2col", false, false, true, 4, false },
    };
    int passed = 0, total = 0;
    for (const Variant& v : variants) {
        Built a;
        const char* fail = build(a, v, be, lw, lh);
        int declined = -1;
        if (!fail) declined = census(v, be, a.gf);
        if (!v.direct) { // reported only
            printf("VAE_CHECK variant=%s latent=%dx%d declined_nodes=%d (census only: sd.cpp's default convolution form)%s%s\n",
                   v.name, lw, lh, declined, fail ? " build failed: " : "", fail ? fail : "");
            continue;
        }
        ++total;
        if (!fail && declined != 0) fail = "the AMEVA backend declines nodes of the graph";
        double ameva_ms = 0, cpu_ms = 0, rel_rms = 0, max_abs = 0, ref_max = 0, psnr = 0, rel_rms_f16 = 0, cpu_f16_ms = 0;
        size_t nan = 0, n_out = 0;
        int64_t d_conv = 0, d_norm = 0, d_fused = 0;
        if (!fail && v.compare) {
            const int64_t c0 = ggml_backend_ameva_get_conv_count(), n0 = ggml_backend_ameva_get_group_norm_count(),
                          f0 = ggml_backend_ameva_get_fused_pair_count();
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            const bool ok = ggml_backend_graph_compute(be, a.gf) == GGML_STATUS_SUCCESS;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ameva_ms = ms_between(t0, t1);
            d_conv = ggml_backend_ameva_get_conv_count() - c0;
            d_norm = ggml_backend_ameva_get_group_norm_count() - n0;
            d_fused = ggml_backend_ameva_get_fused_pair_count() - f0;
            if (!ok) fail = "graph_compute failed on the AMEVA backend";
        }
        if (!fail && v.compare) {
            n_out = (size_t)ggml_nelements(a.out);
            std::vector<float> got(n_out), ref(n_out);
            ggml_backend_tensor_get(a.out, got.data(), 0, n_out * sizeof(float));
            Built c;
            fail = build(c, v, cpu, lw, lh, GGML_TYPE_F32);
            if (!fail) {
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                const bool ok = ggml_backend_graph_compute(cpu, c.gf) == GGML_STATUS_SUCCESS;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                cpu_ms = ms_between(t0, t1);
                if (!ok) fail = "graph_compute failed on the CPU backend";
            }
            if (!fail) {
                ggml_backend_tensor_get(c.out, ref.data(), 0, n_out * sizeof(float));
                double se = 0, sr = 0;
                for (size_t i = 0; i < n_out; ++i) {
                    const double d = (double)got[i] - (double)ref[i];
                    nan += got[i] != got[i] || ref[i] != ref[i];
                    se += d * d;
                    sr += (double)ref[i] * (double)ref[i];
                    max_abs = std::fmax(max_abs, std::fabs(d));
                    ref_max = std::fmax(ref_max, std::fabs((double)ref[i]));
                }
                rel_rms = sr > 0 ? sqrt(se / sr) : 0;
                // peak = the largest magnitude of the reference
                psnr = se > 0 ? 20.0 * log10(ref_max / sqrt(se / (double)n_out)) : INFINITY;
                if (nan) fail = "NaN";
                else if (sr == 0) fail = "the reference output is zero";
                else if (rel_rms > 1e-3) fail = "RMS difference above 1e-3 of the reference RMS";
            }
            // the CPU backend with the f16 kernels (its convolution rounds the activations to f16): reported only
            if (!fail) {
                Built h16;
                const char* e16 = build(h16, v, cpu, lw, lh, GGML_TYPE_F16);
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                if (!e16 && ggml_backend_graph_compute(cpu, h16.gf) != GGML_STATUS_SUCCESS) e16 = "graph_compute";
                clock_gettime(CLOCK_MONOTONIC, &t1);
                cpu_f16_ms = ms_between(t0, t1);
                if (e16) {
                    fail = "the f16-kernel CPU reference failed";
                } else {
                    ggml_backend_tensor_get(h16.out, ref.data(), 0, n_out * sizeof(float));
                    double se = 0, sr = 0;
                    for (size_t i = 0; i < n_out; ++i) {
                        const double d = (double)got[i] - (double)ref[i];
                        se += d * d;
                        sr += (double)ref[i] * (double)ref[i];
                    }
                    rel_rms_f16 = sr > 0 ? sqrt(se / sr) : 0;
                }
            }
        }
        if (!fail) ++passed;
        if (v.compare) {
            printf("VAE_CHECK variant=%s latent=%dx%d image=%dx%d declined_nodes=%d conv_nodes=%lld group_norm_nodes=%lld "
                   "fused_nodes=%lld param_bytes=%zu compute_bytes=%zu out_elements=%zu rel_rms=%.3e max_abs_diff=%.3e "
                   "ref_max_abs=%.3e psnr_db=%.2f rel_rms_cpu_f16=%.3e ameva_ms=%.1f cpu_f32_ms=%.1f cpu_f16_ms=%.1f "
                   "(one run each, 4 CPU threads) %s%s\n",
                   v.name, lw, lh, lw * 8, lh * 8, declined, (long long)d_conv, (long long)d_norm, (long long)d_fused,
                   a.param_bytes, a.compute_bytes, n_out, rel_rms, max_abs, ref_max, psnr, rel_rms_f16, ameva_ms, cpu_ms,
                   cpu_f16_ms,
                   fail ? "FAIL: " : "PASS", fail ? fail : "");
        } else {
            printf("VAE_CHECK variant=%s latent=%dx%d declined_nodes=%d (census only) %s%s\n", v.name, lw, lh, declined,
                   fail ? "FAIL: " : "PASS", fail ? fail : "");
        }
        fflush(stdout);
    }
    printf("VAE_CHECK %d/%d\n", passed, total);
    ggml_backend_free(cpu);
    ggml_backend_free(be);
    return passed == total ? 0 : 1;
}
