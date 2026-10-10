// SPDX-License-Identifier: Apache-2.0
// Kokoro-82M v0.19 on one ggml backend; see kokoro.h for the split between graphs and host.
//
// Layouts: "CT" tensors are [C, T] (ne0 = channels; LSTM, linear, LayerNorm over channels), "TC" tensors are
// [T, 1, C, 1] (ne0 = time; CONV_2D direct with KH = 1 as conv1d, instance norm = ggml_norm over ne0).
// Operations the ONNX export has and ggml has not are composed of ggml ops:
//   - LSTM (bidirectional, gate order i, o, f, c): one MUL_MAT for the input projection of all steps and both
//     directions (the backward direction on the time-reversed input), then per step one batched MUL_MAT (both
//     directions) and the gate arithmetic; each step's hidden state is copied into a state tensor.
//   - ConvTranspose1d with kernel = m * stride: MUL_MAT giving every tap of every output channel, then an overlap-add of
//     m shifted blocks; the depthwise ConvTranspose (k 3, stride 2, padding 1, output padding 1) as even / odd outputs
//     x[i] w1 and x[i] w2 + x[i+1] w0, interleaved by CONCAT.
//   - atan(im / re) with the export's quadrant correction: exact selections (products with 0 / 1 from STEP) and a
//     polynomial on |u| <= tan(pi/8), maximum error 1.9e-8 rad in float32.
//   - reflection padding: GET_ROWS with a reflected index list, or CONCAT of the mirrored column.
#include "kokoro.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kStyle = 128;  // each half of a voice row
constexpr int kHarm = 9;
constexpr int kUp = 300;     // samples per F0 frame
constexpr int kHop = 5;
constexpr int kNfft = 20;
constexpr int kVoiceRows = 511;
constexpr float kPi = 3.14159265358979323846f;

struct ConvP {
    int stride, pad, dil;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

}  // namespace

struct kokoro_model {
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> w;
    std::map<std::string, ConvP> conv;
    std::vector<float> voices;  // [n_voices][511][256]
    int n_voices = 0;
    int sample_rate = 24000;
    std::string wtype;
};

namespace {

// values [2 * j0, 2 * j1) of the noise sequence (pairs j0 .. j1 - 1), clipped at n
void noise_pairs(uint64_t seed, size_t j0, size_t j1, size_t n, float* out) {
    const uint64_t gamma = 0x9E3779B97F4A7C15ull;
    auto mix = [&](uint64_t k) {
        uint64_t z = seed + (k + 1) * gamma;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    for (size_t j = j0; j < j1 && 2 * j < n; ++j) {
        const double u1 = ((double)(mix(2 * j) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        const double u2 = ((double)(mix(2 * j + 1) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double t = 2.0 * M_PI * u2;
        out[2 * j] = (float)(r * std::cos(t));
        if (2 * j + 1 < n) out[2 * j + 1] = (float)(r * std::sin(t));
    }
}

// fn(i0, i1) over [0, n) in up to 4 host threads (every value is a function of its index alone)
template <class F>
void parallel_ranges(size_t n, F fn) {
    const size_t nt = n < 65536 ? 1 : 4;
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(fn, n * t / nt, n * (t + 1) / nt);
    fn(0, n / nt);
    for (auto& x : th) x.join();
}

}  // namespace

void kokoro_noise(uint64_t seed, size_t n, float* out) {
    parallel_ranges((n + 1) / 2, [&](size_t j0, size_t j1) { noise_pairs(seed, j0, j1, n, out); });
}

kokoro_model* kokoro_load(const char* path, ggml_backend* backend, std::string* err) {
    ggml_context* meta = nullptr;
    gguf_init_params gp = {true, &meta};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) {
        if (err) *err = std::string("gguf_init_from_file failed: ") + path;
        return nullptr;
    }
    kokoro_model* m = new kokoro_model();
    m->backend = backend;
    m->ctx_w = meta;
    FILE* f = nullptr;
    try {
        auto str = [&](const char* key) -> std::string {
            const int64_t k = gguf_find_key(g, key);
            if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_STRING) fail(std::string("GGUF key missing: ") + key);
            return gguf_get_val_str(g, k);
        };
        if (str("general.architecture") != "kokoro") fail("GGUF is not a kokoro model");
        m->wtype = str("kokoro.weight_type");
        const int64_t ksr = gguf_find_key(g, "kokoro.sample_rate");
        if (ksr < 0) fail("GGUF key missing: kokoro.sample_rate");
        m->sample_rate = gguf_get_val_i32(g, ksr);
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
            const std::string key = gguf_get_key(g, i);
            if (key.rfind("kokoro.conv.", 0) != 0) continue;
            ConvP c{};
            if (sscanf(gguf_get_val_str(g, i), "%d,%d,%d", &c.stride, &c.pad, &c.dil) != 3) fail("bad " + key);
            m->conv[key.substr(strlen("kokoro.conv."))] = c;
        }
        m->buf_w = ggml_backend_alloc_ctx_tensors(meta, backend);
        if (!m->buf_w) fail("cannot allocate the weights on the backend");
        f = fopen(path, "rb");
        if (!f) fail(std::string("cannot open ") + path);
        const size_t data_off = gguf_get_data_offset(g);
        std::vector<uint8_t> buf;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char* name = gguf_get_tensor_name(g, i);
            ggml_tensor* t = ggml_get_tensor(meta, name);
            const size_t nb = ggml_nbytes(t);
            buf.resize(nb);
            if (fseeko(f, (off_t)(data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nb, f) != nb)
                fail(std::string("short read of tensor ") + name);
            ggml_backend_tensor_set(t, buf.data(), 0, nb);
            m->w[name] = t;
            if (!strcmp(name, "voices")) {
                if (t->type != GGML_TYPE_F32 || t->ne[0] != 2 * kStyle || t->ne[1] != kVoiceRows) fail("voices shape");
                m->n_voices = (int)t->ne[2];
                m->voices.assign((const float*)buf.data(), (const float*)buf.data() + ggml_nelements(t));
            }
        }
        fclose(f);
        f = nullptr;
        if (!m->n_voices) fail("no voices tensor");
    } catch (const std::exception& e) {
        if (f) fclose(f);
        if (err) *err = e.what();
        gguf_free(g);
        kokoro_free(m);
        return nullptr;
    }
    gguf_free(g);
    return m;
}

void kokoro_free(kokoro_model* m) {
    if (!m) return;
    if (m->buf_w) ggml_backend_buffer_free(m->buf_w);
    if (m->ctx_w) ggml_free(m->ctx_w);
    delete m;
}

int kokoro_sample_rate(const kokoro_model* m) { return m->sample_rate; }
int kokoro_n_voices(const kokoro_model* m) { return m->n_voices; }
std::string kokoro_weight_type(const kokoro_model* m) { return m->wtype; }

namespace {

// One graph under construction: its context, the inputs to upload after allocation, a state buffer for tensors
// that outlive the graph or are written step by step (LSTM hidden states).
struct Graph {
    kokoro_model* m;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_context* sctx = nullptr;  // state tensors (allocated before the graph, in their own backend buffer)
    ggml_gallocr_t ga = nullptr;   // owns the graph's tensors; outputs are read before the Graph is destroyed
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> uploads;

    Graph(kokoro_model* model, size_t max_nodes, size_t state_tensors) : m(model) {
        ggml_init_params p = {ggml_tensor_overhead() * max_nodes * 2 + ggml_graph_overhead_custom(max_nodes, false),
                              nullptr, true};
        ctx = ggml_init(p);
        gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_init_params ps = {ggml_tensor_overhead() * (state_tensors + 1), nullptr, true};
        sctx = ggml_init(ps);
        if (!ctx || !gf || !sctx) fail("ggml_init failed");
    }
    ~Graph() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        if (sctx) ggml_free(sctx);
    }

    ggml_tensor* W(const std::string& n) const {
        auto it = m->w.find(n);
        if (it == m->w.end()) fail("weight missing in GGUF: " + n);
        return it->second;
    }
    bool has(const std::string& n) const { return m->w.count(n) != 0; }

    ggml_tensor* in_f32(const std::vector<float>& v, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        if ((int64_t)v.size() != ggml_nelements(t)) fail("input size");
        ggml_set_input(t);
        uploads.emplace_back(t, std::vector<uint8_t>((const uint8_t*)v.data(), (const uint8_t*)(v.data() + v.size())));
        return t;
    }
    ggml_tensor* in_i32(const std::vector<int32_t>& v) {
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)v.size());
        ggml_set_input(t);
        uploads.emplace_back(t, std::vector<uint8_t>((const uint8_t*)v.data(), (const uint8_t*)(v.data() + v.size())));
        return t;
    }
    ggml_tensor* state(int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
        return ggml_new_tensor_3d(sctx, GGML_TYPE_F32, ne0, ne1, ne2);
    }
    void expand(ggml_tensor* t) { ggml_build_forward_expand(gf, t); }

    // ---------------------------------------------------------------------------------------------------------
    // building blocks
    // ---------------------------------------------------------------------------------------------------------
    ggml_tensor* lin(ggml_tensor* x, const std::string& n) {
        return ggml_add(ctx, ggml_mul_mat(ctx, W(n + ".w"), x), W(n + ".b"));
    }
    // A linear layer over a sequence as a 1x1 CONV_2D on a TC tensor [T, 1, Cin, 1] -> [T, 1, Cout, 1]: the weight
    // [Cout, Cin] (ggml [Cin, Cout]) is the kernel [1, 1, Cin, Cout] in the same memory. On AMEVA0 this takes the tiled
    // convolution kernel instead of the GEMV-oriented MUL_MAT path that f16 weights get (ameva_qmm).
    ggml_tensor* lin_tc(ggml_tensor* x, const std::string& n, bool bias = true) {
        ggml_tensor* w = W(n + ".w");
        if (w->ne[0] != x->ne[2]) fail(n + ": linear width");
        ggml_tensor* y = ggml_conv_2d_direct(ctx, ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], ggml_nelements(w) / w->ne[0]),
                                             x, 1, 1, 0, 0, 1, 1);
        if (bias) y = ggml_add_inplace(ctx, y, chan(W(n + ".b")));
        return y;
    }
    ggml_tensor* layer_norm(ggml_tensor* x, const std::string& wn, const std::string& bn, float eps) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), W(wn)), W(bn));
    }
    ggml_tensor* tc(ggml_tensor* ct) {  // [C, T] -> [T, 1, C, 1]
        ggml_tensor* t = ggml_cont(ctx, ggml_transpose(ctx, ct));
        return ggml_reshape_4d(ctx, t, t->ne[0], 1, t->ne[1], 1);
    }
    ggml_tensor* ct(ggml_tensor* tc4) {  // [T, 1, C, 1] -> [C, T]
        return ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, tc4, tc4->ne[0], tc4->ne[2])));
    }
    ggml_tensor* chan(ggml_tensor* v) {  // [C] vector -> [1, 1, C, 1] for broadcasting over TC tensors
        return ggml_reshape_4d(ctx, v, 1, 1, ggml_nelements(v), 1);
    }
    ggml_tensor* conv1d(ggml_tensor* x, const std::string& n, bool bias = true) {
        auto it = m->conv.find(n);
        if (it == m->conv.end()) fail("conv parameters missing in GGUF: " + n);
        ggml_tensor* w = W(n + ".w");
        ggml_tensor* y = ggml_conv_2d_direct(ctx, w, x, it->second.stride, 1, it->second.pad, 0, it->second.dil, 1);
        if (bias) y = ggml_add_inplace(ctx, y, chan(W(n + ".b")));  // in place: fused into the convolution
        return y;
    }
    ggml_tensor* adain(ggml_tensor* x, const std::string& n, ggml_tensor* s) {
        // (1 + gamma) * instance_norm(x) + beta, [gamma, beta] = fc(s)
        ggml_tensor* h = lin(s, n);
        const int64_t c = x->ne[2];
        if (h->ne[0] != 2 * c) fail(n + ": AdaIN width");
        ggml_tensor* gm = chan(ggml_view_1d(ctx, h, c, 0));
        ggml_tensor* bt = chan(ggml_view_1d(ctx, h, c, c * sizeof(float)));
        // instance norm over time = GROUP_NORM with one group per channel; the affine part in place, so that the
        // AMEVA graph engine computes the three nodes as one normalization (its GROUP_NORM chain fusion)
        ggml_tensor* nx = ggml_group_norm(ctx, x, (int)c, 1e-5f);
        nx = ggml_mul_inplace(ctx, nx, ggml_scale_bias(ctx, gm, 1.0f, 1.0f));
        return ggml_add_inplace(ctx, nx, bt);
    }
    ggml_tensor* leaky(ggml_tensor* x, float a) { return ggml_leaky_relu(ctx, x, a, false); }

    ggml_tensor* pool(ggml_tensor* x, const std::string& n) {
        // depthwise ConvTranspose1d k 3, stride 2, padding 1, output padding 1: y[2i] = x[i] w1 + b,
        // y[2i+1] = x[i] w2 + x[i+1] w0 + b (x[L] = 0)
        ggml_tensor* w = W(n + ".w");  // [C, 3]
        const int64_t L = x->ne[0], C = x->ne[2];
        if (w->ne[0] != C || w->ne[1] != 3) fail(n + ": pool shape");
        auto tap = [&](int k) { return chan(ggml_view_1d(ctx, w, C, k * w->nb[1])); };
        ggml_tensor* b = chan(W(n + ".b"));
        ggml_tensor* even = ggml_add(ctx, ggml_mul(ctx, x, tap(1)), b);
        ggml_tensor* nxt = ggml_view_4d(ctx, x, L - 1, 1, C, 1, x->nb[1], x->nb[2], x->nb[3], x->nb[0]);
        nxt = ggml_pad_ext(ctx, nxt, 0, 1, 0, 0, 0, 0, 0, 0);
        ggml_tensor* odd = ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, x, tap(2)), ggml_mul(ctx, nxt, tap(0))), b);
        ggml_tensor* cat = ggml_concat(ctx, ggml_reshape_4d(ctx, even, 1, L, C, 1), ggml_reshape_4d(ctx, odd, 1, L, C, 1), 0);
        return ggml_reshape_4d(ctx, cat, 2 * L, 1, C, 1);
    }

    ggml_tensor* adain_resblk1d(ggml_tensor* x, const std::string& n, ggml_tensor* s, bool upsample) {
        ggml_tensor* r = leaky(adain(x, n + ".n1", s), 0.2f);
        if (upsample) r = pool(r, n + ".pool");
        r = conv1d(r, n + ".c1");
        r = leaky(adain(r, n + ".n2", s), 0.2f);
        r = conv1d(r, n + ".c2");
        ggml_tensor* sc = x;
        if (upsample) sc = ggml_interpolate(ctx, sc, sc->ne[0] * 2, sc->ne[1], sc->ne[2], sc->ne[3], GGML_SCALE_MODE_NEAREST);
        if (has(n + ".sc.w")) sc = conv1d(sc, n + ".sc", false);
        return ggml_scale(ctx, ggml_add(ctx, r, sc), (float)(1.0 / std::sqrt(2.0)));
    }

    ggml_tensor* snake(ggml_tensor* x, const std::string& a, const std::string& ia) {
        // x + (1 / alpha) * sin(alpha * x)^2, in the export's order; the nodes after the first MUL work in place on
        // it, so that the AMEVA graph engine computes the five as one dispatch (its SNAKE chain fusion)
        ggml_tensor* t = ggml_mul(ctx, x, chan(W(a)));
        t = ggml_sqr_inplace(ctx, ggml_sin_inplace(ctx, t));
        t = ggml_mul_inplace(ctx, t, chan(W(ia)));
        return ggml_add_inplace(ctx, t, x);
    }

    ggml_tensor* adain_resblock1(ggml_tensor* x, const std::string& n, ggml_tensor* s) {
        for (int j = 0; j < 3; ++j) {
            const std::string J = std::to_string(j);
            ggml_tensor* t = snake(adain(x, n + ".n1_" + J, s), n + ".a1_" + J, n + ".ia1_" + J);
            t = conv1d(t, n + ".c1_" + J);
            t = snake(adain(t, n + ".n2_" + J, s), n + ".a2_" + J, n + ".ia2_" + J);
            t = conv1d(t, n + ".c2_" + J);
            x = ggml_add(ctx, t, x);
        }
        return x;
    }

    ggml_tensor* conv_transpose(ggml_tensor* x, const std::string& n, int stride, int pad) {
        // ConvTranspose1d [Cin, Cout, K], K = mb * stride: P[l, k + K*cout] = sum_cin W x (a 1x1 convolution with
        // K*Cout output channels), then an overlap-add of the mb blocks of `stride` taps
        ggml_tensor* w = W(n + ".w");  // ggml [Cin, K, Cout]
        const int64_t L = x->ne[0], cin = x->ne[2], K = w->ne[1], cout = w->ne[2];
        if (w->ne[0] != cin || K % stride) fail(n + ": transposed conv shape");
        const int64_t mb = K / stride;
        ggml_tensor* P = ggml_conv_2d_direct(ctx, ggml_reshape_4d(ctx, w, 1, 1, cin, K * cout), x, 1, 1, 0, 0, 1, 1);
        P = ggml_reshape_4d(ctx, P, L, stride, mb, cout);  // [l, j, block, cout]
        ggml_tensor* y = nullptr;
        for (int64_t b = 0; b < mb; ++b) {
            ggml_tensor* q = ggml_view_4d(ctx, P, L, stride, 1, cout, P->nb[1], P->nb[2], P->nb[3], b * P->nb[2]);
            q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 0, 2, 3));  // [stride, L, 1, Cout]: sample j + stride * l
            q = ggml_reshape_2d(ctx, q, stride * L, cout);
            q = ggml_pad_ext(ctx, q, (int)(b * stride), (int)((mb - 1 - b) * stride), 0, 0, 0, 0, 0, 0);
            y = y ? ggml_add(ctx, y, q) : q;
        }
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, W(n + ".b"), 1, cout));
        const int64_t lo = (L - 1) * stride + K - 2 * pad;
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, lo, cout, y->nb[1], pad * y->nb[0]));
        return ggml_reshape_4d(ctx, y, lo, 1, cout, 1);
    }

    // bidirectional LSTM on x [In, T] -> [2H, T] (forward hidden states, then backward)
    ggml_tensor* lstm(ggml_tensor* x, const std::string& n, ggml_tensor* rev) { return lstms({{x, n}}, rev)[0]; }

    // several independent bidirectional LSTMs of the same length T and hidden size H, stepped together: their
    // directions are the batch of one recurrence (one MUL_MAT and one set of gate ops per step for all of them)
    std::vector<ggml_tensor*> lstms(const std::vector<std::pair<ggml_tensor*, std::string>>& in, ggml_tensor* rev) {
        const int64_t H = W(in[0].second + ".wh")->ne[0], T = in[0].first->ne[1];
        const int64_t D = 2 * (int64_t)in.size();  // directions in the batch
        ggml_tensor *xp = nullptr, *wh = nullptr;
        for (const auto& p : in) {
            const std::string& n = p.second;
            ggml_tensor* x = p.first;
            ggml_tensor* wi = W(n + ".wi");  // [In, 4H, 2]
            ggml_tensor* w = W(n + ".wh");   // [H, 4H, 2]
            if (wi->ne[0] != x->ne[0] || wi->ne[1] != 4 * H || w->ne[0] != H || w->ne[1] != 4 * H || x->ne[1] != T)
                fail(n + ": LSTM shape");
            // input projection of every step for both directions as one 1x1 convolution; the backward direction reads
            // the input reversed in time, and its projection reversed in time is the same: W x[T-1-t] = (W x)[T-1-t]
            ggml_tensor* pr = ct(ggml_conv_2d_direct(ctx, ggml_reshape_4d(ctx, wi, 1, 1, x->ne[0], 8 * H), tc(x),
                                                     1, 1, 0, 0, 1, 1));                // [8H, T]
            ggml_tensor* fw_p = ggml_view_2d(ctx, pr, 4 * H, T, pr->nb[1], 0);
            ggml_tensor* bw_p = ggml_get_rows(ctx, ggml_view_2d(ctx, pr, 4 * H, T, pr->nb[1], 4 * H * sizeof(float)), rev);
            ggml_tensor* p1 = ggml_add(ctx, ggml_concat(ctx, fw_p, bw_p, 2), ggml_reshape_3d(ctx, W(n + ".b"), 4 * H, 1, 2));
            xp = xp ? ggml_concat(ctx, xp, p1, 2) : p1;  // [4H, T, D]
            wh = wh ? ggml_concat(ctx, wh, w, 2) : w;    // [H, 4H, D]
        }
        ggml_tensor* out = state(H, T, D);
        ggml_tensor *h = nullptr, *c = nullptr;
        for (int64_t t = 0; t < T; ++t) {
            ggml_tensor* g = ggml_view_3d(ctx, xp, 4 * H, 1, D, xp->nb[1], xp->nb[2], t * xp->nb[1]);
            if (h) g = ggml_add(ctx, ggml_mul_mat(ctx, wh, h), g);
            ggml_tensor* ifo = ggml_sigmoid(ctx, ggml_view_3d(ctx, g, 3 * H, 1, D, g->nb[1], g->nb[2], 0));
            ggml_tensor* cc = ggml_tanh(ctx, ggml_view_3d(ctx, g, H, 1, D, g->nb[1], g->nb[2], 3 * H * sizeof(float)));
            auto gate = [&](int k) {
                return ggml_view_3d(ctx, ifo, H, 1, D, ifo->nb[1], ifo->nb[2], k * H * sizeof(float));
            };
            ggml_tensor* ic = ggml_mul(ctx, gate(0), cc);
            c = c ? ggml_add(ctx, ggml_mul(ctx, gate(2), c), ic) : ic;
            h = ggml_mul(ctx, gate(1), ggml_tanh(ctx, c));
            expand(ggml_cpy(ctx, h, ggml_view_3d(ctx, out, H, 1, D, out->nb[1], out->nb[2], t * out->nb[1])));
        }
        std::vector<ggml_tensor*> res;
        for (int64_t k = 0; k < (int64_t)in.size(); ++k) {
            ggml_tensor* fw = ggml_view_2d(ctx, out, H, T, out->nb[1], 2 * k * out->nb[2]);
            ggml_tensor* bw = ggml_get_rows(ctx, ggml_view_2d(ctx, out, H, T, out->nb[1], (2 * k + 1) * out->nb[2]), rev);
            res.push_back(ggml_concat(ctx, fw, bw, 0));
        }
        return res;
    }

    // atan(im / re) with the export's correction: re < 0 -> (im > 0 ? +pi : -pi) added, otherwise unchanged
    ggml_tensor* phase(ggml_tensor* re, ggml_tensor* im) {
        static const float a[6] = {-3.3333331792e-01f, 1.9999856202e-01f, -1.4280886056e-01f,
                                   1.1032733541e-01f, -8.4170324926e-02f, 4.6331859414e-02f};
        auto one_minus = [&](ggml_tensor* s) { return ggml_scale_bias(ctx, s, -1.0f, 1.0f); };
        auto sel = [&](ggml_tensor* s, ggml_tensor* a1, ggml_tensor* a0) {  // s ? a1 : a0, s in {0, 1}
            return ggml_add(ctx, ggml_mul(ctx, a1, s), ggml_mul(ctx, a0, one_minus(s)));
        };
        ggml_tensor* ax = ggml_abs(ctx, re);
        ggml_tensor* ay = ggml_abs(ctx, im);
        ggml_tensor* big = ggml_step(ctx, ggml_sub(ctx, ay, ax));  // |im| > |re|
        ggml_tensor* t = ggml_div(ctx, sel(big, ax, ay), sel(big, ay, ax));  // min / max, in [0, 1]
        ggml_tensor* hi = ggml_step(ctx, ggml_scale_bias(ctx, t, 1.0f, -0.41421356237309503f));
        ggml_tensor* u = sel(hi, ggml_div(ctx, ggml_scale_bias(ctx, t, 1.0f, -1.0f), ggml_scale_bias(ctx, t, 1.0f, 1.0f)), t);
        ggml_tensor* z = ggml_sqr(ctx, u);
        ggml_tensor* p = ggml_scale_bias(ctx, z, a[5], a[4]);
        for (int k = 3; k >= 0; --k) p = ggml_scale_bias(ctx, ggml_mul(ctx, p, z), 1.0f, a[k]);
        ggml_tensor* r = ggml_add(ctx, u, ggml_mul(ctx, ggml_mul(ctx, u, z), p));
        r = ggml_add(ctx, r, ggml_scale(ctx, hi, kPi / 4.0f));                       // atan(min / max)
        r = sel(big, ggml_scale_bias(ctx, r, -1.0f, kPi / 2.0f), r);               // atan(|im| / |re|)
        ggml_tensor* neg = ggml_step(ctx, ggml_neg(ctx, re));                      // re < 0
        ggml_tensor* pos = ggml_step(ctx, im);                                      // im > 0
        ggml_tensor* qn = sel(pos, ggml_scale_bias(ctx, r, -1.0f, kPi), ggml_scale_bias(ctx, r, 1.0f, -kPi));
        ggml_tensor* qp = ggml_mul(ctx, ggml_sgn(ctx, im), r);
        return sel(neg, qn, qp);
    }
};

void check_and_run(Graph& G, const char* what) {
    ggml_backend_t be = G.m->backend;
    G.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!G.ga) fail("ggml_gallocr_new failed");
    if (!ggml_gallocr_alloc_graph(G.ga, G.gf)) fail(std::string(what) + ": graph allocation failed");
    std::string bad;
    int n_bad = 0;
    for (int i = 0; i < ggml_graph_n_nodes(G.gf); ++i) {
        ggml_tensor* t = ggml_graph_node(G.gf, i);
        if (!ggml_backend_supports_op(be, t)) {
            if (n_bad++ < 8) bad += std::string(" ") + ggml_op_desc(t) + "(" + t->name + ")";
        }
    }
    if (n_bad) {
        fail(std::string(what) + ": " + std::to_string(n_bad) + " node(s) not supported by " + ggml_backend_name(be) +
             ":" + bad);
    }
    for (auto& u : G.uploads) ggml_backend_tensor_set(u.first, u.second.data(), 0, u.second.size());
    const ggml_status st = ggml_backend_graph_compute(be, G.gf);
    if (st != GGML_STATUS_SUCCESS) fail(std::string(what) + ": graph compute failed with status " + std::to_string((int)st));
}

std::vector<int32_t> rev_index(int64_t n) {
    std::vector<int32_t> r(n);
    for (int64_t i = 0; i < n; ++i) r[i] = (int32_t)(n - 1 - i);
    return r;
}

}  // namespace

bool kokoro_synthesize(kokoro_model* m, const std::vector<int32_t>& phonemes, const kokoro_options& opt,
                       kokoro_result* out, std::string* err) {
    ggml_backend_buffer_t sbuf1 = nullptr, sbuf2 = nullptr;
    try {
        *out = kokoro_result();
        if (phonemes.empty()) fail("no phonemes");
        if (opt.voice < 0 || opt.voice >= m->n_voices) fail("voice index out of range");
        if (!(opt.speed > 0.0f)) fail("speed must be positive");
        const int64_t N = (int64_t)phonemes.size() + 2;
        if (N > 512) fail("more than 510 phonemes (ALBERT positions)");
        double th = now_ms();
        std::vector<int32_t> tok(N), pos(N);
        tok[0] = tok[N - 1] = 0;
        for (int64_t i = 0; i < N; ++i) pos[i] = (int32_t)i;
        for (size_t i = 0; i < phonemes.size(); ++i) {
            if (phonemes[i] < 0 || phonemes[i] >= 178) fail("phoneme id out of range");
            tok[i + 1] = phonemes[i];
        }
        const float* row = m->voices.data() + ((size_t)opt.voice * kVoiceRows +
                                              (size_t)std::min<int64_t>((int64_t)phonemes.size(), kVoiceRows - 1)) * 2 * kStyle;
        const std::vector<float> s_dec(row, row + kStyle), s_pred(row + kStyle, row + 2 * kStyle);
        out->ms_host += now_ms() - th;

        // ------------------------------------------------------------------ graph 1: text side
        double t0 = now_ms();
        Graph G1(m, (size_t)(16 * 5 * N + 16384), 8);
        ggml_context* ctx = G1.ctx;
        ggml_tensor* tok_t = G1.in_i32(tok);
        ggml_tensor* pos_t = G1.in_i32(pos);
        ggml_tensor* rev_t = G1.in_i32(rev_index(N));
        ggml_tensor* sp = G1.in_f32(s_pred, kStyle);
        ggml_tensor* ones = G1.in_f32(std::vector<float>(50, 1.0f), 50);

        // PL-BERT
        ggml_tensor* x = ggml_get_rows(ctx, G1.W("bert.word_emb"), tok_t);
        x = ggml_add(ctx, x, ggml_view_1d(ctx, G1.W("bert.type_emb"), G1.W("bert.type_emb")->ne[0], 0));
        x = ggml_add(ctx, x, ggml_get_rows(ctx, G1.W("bert.pos_emb"), pos_t));
        x = G1.layer_norm(x, "bert.emb_ln.w", "bert.emb_ln.b", 1e-12f);
        x = G1.ct(G1.lin_tc(G1.tc(x), "bert.map_in"));  // [768, N]
        const int64_t D = x->ne[0], NH = 12, HD = D / NH;
        for (int l = 0; l < 12; ++l) {
            ggml_tensor* xt = G1.tc(x);  // [N, 1, D, 1]
            auto heads = [&](ggml_tensor* t) {  // TC [N, 1, D, 1] (channel h * HD + d) -> [HD, N, NH]
                return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, t, N, HD, NH), 1, 0, 2, 3));
            };
            ggml_tensor* q = heads(G1.lin_tc(xt, "bert.q"));
            ggml_tensor* k = heads(G1.lin_tc(xt, "bert.k"));
            ggml_tensor* v = heads(G1.lin_tc(xt, "bert.v"));
            // softmax(q k^T / sqrt(HD)) v per head (no mask: one sequence, no padding); the result [HD, NH, N] is the
            // context layer [D, N] with channel h * HD + d
            ggml_tensor* a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            a = G1.tc(ggml_reshape_2d(ctx, a, D, N));
            x = G1.layer_norm(ggml_add(ctx, x, G1.ct(G1.lin_tc(a, "bert.o"))), "bert.attn_ln.w", "bert.attn_ln.b", 1e-12f);
            // NewGELU: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), composed (ggml-cpu's GELU uses an f16 table)
            ggml_tensor* f = G1.lin_tc(G1.tc(x), "bert.ffn");
            ggml_tensor* u = ggml_add(ctx, f, ggml_scale(ctx, ggml_mul(ctx, ggml_sqr(ctx, f), f), 0.044715f));
            u = ggml_scale_bias(ctx, ggml_tanh(ctx, ggml_scale(ctx, u, 0.7978845608028654f)), 1.0f, 1.0f);
            f = ggml_mul(ctx, ggml_scale(ctx, f, 0.5f), u);
            f = G1.ct(G1.lin_tc(f, "bert.ffn_out"));
            x = G1.layer_norm(ggml_add(ctx, f, x), "bert.ffn_ln.w", "bert.ffn_ln.b", 1e-12f);
        }
        ggml_tensor* d = G1.ct(G1.lin_tc(G1.tc(x), "bert_enc"));  // [512, N]

        // text encoder (conv stack); its LSTM runs together with the first duration-encoder LSTM (independent, same T)
        ggml_tensor* te = G1.tc(ggml_get_rows(ctx, G1.W("te.emb"), tok_t));         // [N, 1, 512, 1]
        for (int i = 0; i < 3; ++i) {
            const std::string I = std::to_string(i);
            te = G1.conv1d(te, "te.cnn" + I);
            ggml_tensor* c = G1.layer_norm(G1.ct(te), "te.cnn" + I + ".g", "te.cnn" + I + ".beta", 1e-5f);
            c = G1.leaky(c, 0.2f);
            te = (i < 2) ? G1.tc(c) : c;
        }

        // duration encoder: [x, s] -> (biLSTM -> AdaLayerNorm -> [x, s]) x 3
        ggml_tensor* sb = ggml_repeat(ctx, sp, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kStyle, N));
        d = ggml_concat(ctx, d, sb, 0);
        ggml_tensor* t_en = nullptr;
        for (int i = 0; i < 3; ++i) {
            const std::string I = std::to_string(i);
            if (i == 0) {
                std::vector<ggml_tensor*> r = G1.lstms({{d, "pred.te.lstm0"}, {te, "te.lstm"}}, rev_t);
                d = r[0];
                t_en = r[1];                                                       // [512, N]
            } else {
                d = G1.lstm(d, "pred.te.lstm" + I, rev_t);
            }
            ggml_tensor* h = G1.lin(sp, "pred.te.aln" + I);  // [1024, 1]
            const int64_t C = d->ne[0];
            ggml_tensor* gm = ggml_view_1d(ctx, h, C, 0);
            ggml_tensor* bt = ggml_view_1d(ctx, h, C, C * sizeof(float));
            d = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, d, 1e-5f), ggml_scale_bias(ctx, gm, 1.0f, 1.0f)), bt);
            d = ggml_concat(ctx, d, sb, 0);
        }
        ggml_tensor* ten_keep = G1.state(t_en->ne[0], N);
        G1.expand(ggml_cpy(ctx, t_en, ten_keep));
        ggml_tensor* d_keep = G1.state(d->ne[0], N);
        G1.expand(ggml_cpy(ctx, d, d_keep));
        ggml_tensor* dl = G1.lstm(d, "pred.lstm", rev_t);
        ggml_tensor* dur = ggml_sigmoid(ctx, G1.lin(dl, "pred.dur"));             // [50, N]
        dur = ggml_mul_mat(ctx, dur, ones);                                        // [N, 1] sum over 50
        if (opt.speed != 1.0f) dur = ggml_scale(ctx, dur, 1.0f / opt.speed);
        ggml_set_output(dur);
        G1.expand(dur);

        sbuf1 = ggml_backend_alloc_ctx_tensors(G1.sctx, m->backend);
        if (!sbuf1) fail("cannot allocate the text-side state buffer");
        check_and_run(G1, "text graph");
        out->nodes_text = ggml_graph_n_nodes(G1.gf);
        out->dur_raw.resize(N);
        ggml_backend_tensor_get(dur, out->dur_raw.data(), 0, N * sizeof(float));
        out->ms_text = now_ms() - t0;

        // ------------------------------------------------------------------ host: frames
        th = now_ms();
        if (!opt.durations.empty()) {
            if ((int64_t)opt.durations.size() != N) fail("durations override: one value per token (pads included)");
            out->durations = opt.durations;
        } else {
            out->durations.resize(N);
            for (int64_t i = 0; i < N; ++i) out->durations[i] = std::max(1, (int)std::nearbyint(out->dur_raw[i]));
        }
        std::vector<int32_t> idx;
        for (int64_t i = 0; i < N; ++i) {
            if (out->durations[i] < 1) fail("durations must be at least 1");
            idx.insert(idx.end(), out->durations[i], (int32_t)i);
        }
        const int64_t TF = (int64_t)idx.size();        // frames
        const int64_t F2 = 2 * TF;                     // F0 frames
        const int64_t L = F2 * kUp;                    // samples
        const int64_t NF = L / kHop + 1;               // STFT frames
        const int64_t LW = (NF - 1) * kHop + kNfft;    // inverse transform before trimming
        std::vector<float> flat((size_t)L * kHarm), noise((size_t)L * kHarm);
        kokoro_noise(opt.seed, flat.size(), flat.data());
        parallel_ranges((size_t)L, [&](size_t l0, size_t l1) {  // [L, 9] TC
            for (size_t l = l0; l < l1; ++l)
                for (int h = 0; h < kHarm; ++h) noise[(size_t)h * L + l] = flat[l * kHarm + h];
        });
        std::vector<int32_t> refl(L + kNfft);
        for (int64_t i = 0; i < L + kNfft; ++i) {
            int64_t j = i - kNfft / 2;
            if (j < 0) j = -j;
            if (j >= L) j = 2 * (L - 1) - j;
            refl[i] = (int32_t)j;
        }
        std::vector<float> wsum(LW, 0.0f);
        {
            double w2[kNfft];
            for (int n = 0; n < kNfft; ++n) {
                const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / kNfft);
                w2[n] = w * w;
            }
            for (int64_t i = 0; i < LW; ++i) {
                double s = 0;
                for (int64_t fr = std::max<int64_t>(0, (i - kNfft + kHop) / kHop); fr * kHop <= i; ++fr) {
                    const int64_t k = i - fr * kHop;
                    if (k < kNfft) s += w2[k];
                }
                const float v = (float)s;
                wsum[i] = v > 1.17549435e-38f ? v : 1.0f;  // the export divides only where window_sum > FLT_MIN
            }
        }
        std::vector<float> harm(kHarm);
        for (int h = 0; h < kHarm; ++h) harm[h] = (float)(h + 1);
        out->ms_host += now_ms() - th;

        // ------------------------------------------------------------------ graph 2: synthesis
        t0 = now_ms();
        Graph G2(m, (size_t)(16 * TF + 32768), 4);
        ctx = G2.ctx;
        ggml_tensor* idx_t = G2.in_i32(idx);
        ggml_tensor* rev2 = G2.in_i32(rev_index(TF));
        ggml_tensor* sp2 = G2.in_f32(s_pred, kStyle);
        ggml_tensor* sd2 = G2.in_f32(s_dec, kStyle);
        ggml_tensor* harm_t = G2.in_f32(harm, 1, kHarm);
        ggml_tensor* noise_t = G2.in_f32(noise, L, kHarm);
        ggml_tensor* refl_t = G2.in_i32(refl);
        ggml_tensor* wsum_t = G2.in_f32(wsum, LW);

        ggml_tensor* en = ggml_get_rows(ctx, d_keep, idx_t);                       // [640, TF]
        ggml_tensor* sh = G2.tc(G2.lstm(en, "pred.shared", rev2));                // [TF, 1, 512, 1]
        ggml_tensor* f0 = sh;
        ggml_tensor* nn = sh;
        for (int i = 0; i < 3; ++i) {
            f0 = G2.adain_resblk1d(f0, "pred.F0" + std::to_string(i), sp2, i == 1);
            nn = G2.adain_resblk1d(nn, "pred.N" + std::to_string(i), sp2, i == 1);
        }
        f0 = G2.conv1d(f0, "pred.F0_proj");                                        // [F2, 1, 1, 1]
        nn = G2.conv1d(nn, "pred.N_proj");
        // the F0 curve leaves the graph through a state tensor: f0 is an in-place view of the projection, and the
        // allocator does not keep a view's parent alive for ggml_set_output
        ggml_tensor* f0_keep = G2.state(F2, 1);
        G2.expand(ggml_cpy(ctx, f0, f0_keep));
        ggml_tensor* asr = G2.tc(ggml_get_rows(ctx, ten_keep, idx_t));            // [TF, 1, 512, 1]
        ggml_tensor* f0d = G2.conv1d(f0, "dec.F0_conv");                           // [TF, 1, 1, 1]
        ggml_tensor* nd = G2.conv1d(nn, "dec.N_conv");
        ggml_tensor* xd = ggml_concat(ctx, ggml_concat(ctx, asr, f0d, 2), nd, 2);
        xd = G2.adain_resblk1d(xd, "dec.enc", sd2, false);
        ggml_tensor* asr_res = G2.conv1d(asr, "dec.asr_res");
        for (int i = 0; i < 4; ++i) {
            xd = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, xd, asr_res, 2), f0d, 2), nd, 2);
            xd = G2.adain_resblk1d(xd, "dec.dec" + std::to_string(i), sd2, i == 3);
        }                                                                          // [F2, 1, 512, 1]

        // harmonic source (SineGen at the F0 frame rate, then linear up-sampling of the phase)
        ggml_tensor* f02 = ggml_reshape_2d(ctx, f0, F2, 1);
        ggml_tensor* fn = ggml_mul(ctx, ggml_repeat(ctx, f02, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F2, kHarm)), harm_t);
        ggml_tensor* rad = ggml_scale(ctx, fn, 1.0f / 24000.0f);
        rad = ggml_sub(ctx, rad, ggml_floor(ctx, rad));                            // x mod 1
        ggml_tensor* ph = ggml_cumsum(ctx, rad);
        ph = ggml_scale(ctx, ggml_scale(ctx, ggml_scale(ctx, ph, 2.0f), kPi), (float)kUp);
        ph = ggml_interpolate(ctx, ph, L, kHarm, 1, 1, GGML_SCALE_MODE_BILINEAR);   // [L, 9]
        ggml_tensor* sine = ggml_scale(ctx, ggml_sin(ctx, ph), 0.1f);
        ggml_tensor* uvf = ggml_step(ctx, ggml_scale_bias(ctx, f02, 1.0f, -10.0f));  // f0 > 10
        // nearest up-sampling by 300 as REPEAT of [1, F2] to [300, F2]: sample j + 300 f is frame f
        ggml_tensor* uv = ggml_repeat(ctx, ggml_reshape_2d(ctx, uvf, 1, F2), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kUp, F2));
        uv = ggml_reshape_2d(ctx, uv, L, 1);
        ggml_tensor* namp = ggml_add(ctx, ggml_scale(ctx, uv, 0.003f),
                                     ggml_scale(ctx, ggml_scale_bias(ctx, uv, -1.0f, 1.0f), 0.1f / 3.0f));
        ggml_tensor* sw = ggml_add(ctx, ggml_mul(ctx, sine, uv), ggml_mul(ctx, noise_t, namp));
        ggml_tensor* src = G2.lin_tc(ggml_reshape_4d(ctx, sw, L, 1, kHarm, 1), "gen.src");  // [L, 1, 1, 1]
        src = ggml_reshape_2d(ctx, ggml_tanh(ctx, src), 1, L);                                 // rows of one sample
        ggml_tensor* padded = ggml_reshape_4d(ctx, ggml_get_rows(ctx, src, refl_t), L + kNfft, 1, 1, 1);
        ggml_tensor* spec = ggml_conv_2d_direct(ctx, G2.W("gen.stft"), padded, kHop, 1, 0, 0, 1, 1);  // [NF, 1, 22, 1]
        const int nb_ = kNfft / 2 + 1;
        ggml_tensor* re = ggml_view_4d(ctx, spec, NF, 1, nb_, 1, spec->nb[1], spec->nb[2], spec->nb[3], 0);
        ggml_tensor* im = ggml_view_4d(ctx, spec, NF, 1, nb_, 1, spec->nb[1], spec->nb[2], spec->nb[3], nb_ * spec->nb[2]);
        ggml_tensor* mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));
        ggml_tensor* har = ggml_concat(ctx, mag, G2.phase(re, im), 2);            // [NF, 1, 22, 1]

        // generator
        ggml_tensor* xg = xd;
        static const int kStride[2] = {10, 6};
        for (int i = 0; i < 2; ++i) {
            const std::string I = std::to_string(i);
            xg = G2.leaky(xg, 0.1f);
            ggml_tensor* xs = G2.adain_resblock1(G2.conv1d(har, "gen.nc" + I), "gen.nr" + I, sd2);
            const ggml_tensor* uw = G2.W("gen.ups" + I + ".w");
            xg = G2.conv_transpose(xg, "gen.ups" + I, kStride[i], (int)(uw->ne[1] - kStride[i]) / 2);
            if (i == 1) {  // reflection pad (1, 0)
                ggml_tensor* first = ggml_view_4d(ctx, xg, 1, 1, xg->ne[2], 1, xg->nb[1], xg->nb[2], xg->nb[3], xg->nb[0]);
                xg = ggml_concat(ctx, ggml_cont(ctx, first), xg, 0);
            }
            xg = ggml_add(ctx, xg, xs);
            ggml_tensor* acc = nullptr;
            for (int j = 0; j < 3; ++j) {
                ggml_tensor* r = G2.adain_resblock1(xg, "gen.rb" + std::to_string(3 * i + j), sd2);
                acc = acc ? ggml_add(ctx, acc, r) : r;
            }
            xg = ggml_scale(ctx, acc, (float)(1.0 / 3.0));
        }
        xg = G2.conv1d(G2.leaky(xg, 0.01f), "gen.post");                         // [NF, 1, 22, 1]
        ggml_tensor* mg = ggml_exp(ctx, ggml_view_4d(ctx, xg, NF, 1, nb_, 1, xg->nb[1], xg->nb[2], xg->nb[3], 0));
        ggml_tensor* pp = ggml_sin(ctx, ggml_view_4d(ctx, xg, NF, 1, nb_, 1, xg->nb[1], xg->nb[2], xg->nb[3], nb_ * xg->nb[2]));
        ggml_tensor* cat = ggml_concat(ctx, ggml_mul(ctx, mg, ggml_cos(ctx, pp)), ggml_mul(ctx, mg, ggml_sin(ctx, pp)), 2);
        ggml_tensor* P = ggml_mul_mat(ctx, G2.W("gen.istft"), G2.ct(cat));        // [20, NF]
        ggml_tensor* y = nullptr;
        for (int b = 0; b < kNfft / kHop; ++b) {
            ggml_tensor* q = ggml_cont(ctx, ggml_view_2d(ctx, P, kHop, NF, P->nb[1], b * kHop * sizeof(float)));
            q = ggml_pad_ext(ctx, ggml_reshape_1d(ctx, q, kHop * NF), b * kHop, (kNfft / kHop - 1 - b) * kHop, 0, 0, 0, 0, 0, 0);
            y = y ? ggml_add(ctx, y, q) : q;
        }
        y = ggml_scale(ctx, ggml_div(ctx, y, wsum_t), (float)kNfft / kHop);
        ggml_tensor* audio = ggml_cont(ctx, ggml_view_1d(ctx, y, L, (kNfft / 2) * sizeof(float)));
        ggml_set_output(audio);
        G2.expand(audio);

        sbuf2 = ggml_backend_alloc_ctx_tensors(G2.sctx, m->backend);
        if (!sbuf2) fail("cannot allocate the synthesis state buffer");
        check_and_run(G2, "synthesis graph");
        out->nodes_synth = ggml_graph_n_nodes(G2.gf);
        out->audio.resize(L);
        ggml_backend_tensor_get(audio, out->audio.data(), 0, L * sizeof(float));
        out->f0.resize(F2);
        ggml_backend_tensor_get(f0_keep, out->f0.data(), 0, F2 * sizeof(float));
        out->ms_synth = now_ms() - t0;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        if (sbuf1) ggml_backend_buffer_free(sbuf1);
        if (sbuf2) ggml_backend_buffer_free(sbuf2);
        return false;
    }
    ggml_backend_buffer_free(sbuf1);
    ggml_backend_buffer_free(sbuf2);
    return true;
}
