// SPDX-License-Identifier: Apache-2.0
// AMEVA convolution kernel timing: the tile kernel staging its activations from the source tensor
// (shaders/ameva_conv.comp) against the same tile kernel reading a materialised column matrix
// (shaders/ameva_qgemm.comp), through ac_runtime directly (Vulkan), one convolution per run.
//
//   ameva_conv_bench W,H,T,C,OC,KW,KH,KT[,PW,PH,PT0,PT1[,SW,SH,ST[,DW,DH,DT]]] [ITER] [TYPES]
//
//   source f32 [W, H, T, C] (2D: T = KT = 1), OC output channels, kernel KW x KH x KT.
//   PW, PH: padding on both sides (default (K - 1) / 2); PT0, PT1: padding before and after along t (default
//   (KT - 1) / 2 each; a causal convolution is PT0 = KT - 1, PT1 = 0); S*: stride (default 1); D*: dilation (default 1).
//   ITER: timed runs per path (default 4). TYPES: weight types, default "q8_0,f16" (f16 is the exact mode).
//
// Per weight type, four paths over the same weight buffer. K is made of units of 32 channels at one tap (channels
// padded to a multiple of 32 with zero weights); AMEVA_CONV_BENCH_KORDER orders the units: "group" (default) walks
// all taps of a channel group before the next group, "tap" all channel groups of a tap before the next tap.
//   matrix         ameva_qgemm on the column matrix (built on the CPU, not timed), dispatches sliced as ggml-ameva does
//   conv_s         ameva_conv, scalar staging
//   conv_v         ameva_conv, vec4 staging (stride along w = 1 only)
//   conv_v_direct  conv_v with every invocation writing its own block of the output (OUT_DIRECT in the shader)
// Every path's destination is filled with NaN bytes, computed once and compared with an f32 convolution on the CPU
// that uses the same dequantised weights and the same summation order (max difference over the rms of the
// reference, and a count of NaN). Then the paths are timed in turn, one round that is not counted and ITER rounds:
// wall time from batch begin to the completion event, and the GPU time of the batch when the queue has timestamps.
// AMEVA_CONV_BENCH_REF=muladd builds the reference with a rounded product and a rounded sum per term in place of a
// fused multiply-add (default fma): a driver that does not fuse the kernels' fma() matches that one bit for bit.
// The column matrix is skipped (and reported as skipped) above AMEVA_CONV_BENCH_MATRIX_MAX_MB (default 512).
// AMEVA_CONV_BENCH_ABLATE=1 (kernel tuning) adds builds of both kernels with a part of the staging removed (ABLATE in
// the shaders); their outputs are wrong by construction, are not compared and are labelled ABLATION.
// Exit status 0 only when every path that ran matches the reference.
//
// The kernels come from headers generated at build time (scripts/termux/lab/conv_bench.sh: glslangValidator,
// spirv-val, scripts/embed_spirv.py), each with the SHA-256 the runtime pins at kernel creation.
#include "ac_runtime.h"

#include "qgemm_q8_0_spv.h"
#include "qgemm_f16_spv.h"
#include "conv_q8_0_s_spv.h"
#include "conv_q8_0_v_spv.h"
#include "conv_f16_s_spv.h"
#include "conv_f16_v_spv.h"
#include "conv_q8_0_va1_spv.h"
#include "conv_q8_0_va2_spv.h"
#include "conv_q8_0_va3_spv.h"
#include "conv_f16_va1_spv.h"
#include "conv_f16_va2_spv.h"
#include "conv_f16_va3_spv.h"
#include "conv_q8_0_va4_spv.h"
#include "conv_q8_0_vd_spv.h"
#include "conv_f16_va4_spv.h"
#include "conv_f16_vd_spv.h"
#include "qgemm_q8_0_a1_spv.h"
#include "qgemm_q8_0_a2_spv.h"
#include "qgemm_f16_a1_spv.h"
#include "qgemm_f16_a2_spv.h"

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "CONV_BENCH ERROR: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}
#define CHECK(call)                                                                              \
    do {                                                                                         \
        const AcResult r_ = (call);                                                              \
        if (r_ != AC_SUCCESS) die("%s failed (code %d), line %d", #call, (int)r_, __LINE__);     \
    } while (0)

static uint32_t g_seed = 0x2545F491u;
static float rnd() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return (float)(g_seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

// IEEE binary32 <-> binary16, round-to-nearest-even (the arithmetic of ameva_f2h / ameva_h2f in ameva_common.glsl)
static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return (uint16_t)(sign | (ax > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);
    if (ax < 0x38800000u) {
        if (ax < 0x33000001u) return (uint16_t)sign;
        const uint32_t e = ax >> 23, m = (ax & 0x7FFFFFu) | 0x800000u, shift = 126u - e;
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u))) h += 1u;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (ax >> 13) - (112u << 10);
    const uint32_t rem = ax & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
    return (uint16_t)(sign | h);
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f;
        return s ? -v : v;
    }
    x = e == 31 ? (s | 0x7F800000u | (m << 13)) : (s | ((e + 112u) << 23) | (m << 13));
    float f;
    memcpy(&f, &x, 4);
    return f;
}

struct Shape {
    int W, H, T, C, OC, KW, KH, KT;
    int PW, PH, PT0, PT1, SW, SH, ST, DW, DH, DT;
    int OW, OH, OT, CP, CU, taps;
    uint32_t K, U, N;
    uint64_t P; // output positions
    // order of the units along K: all channel groups of a tap, then the next tap (false), or all taps of a channel
    // group, then the next group (true). Weight k = 32 * unit + channel % 32.
    bool group_major;
    uint32_t unit_of(int tap, int cu) const { return group_major ? (uint32_t)(cu * taps + tap) : (uint32_t)(tap * CU + cu); }
    int tap_of(uint32_t u) const { return group_major ? (int)(u % (uint32_t)taps) : (int)(u / (uint32_t)CU); }
    int cu_of(uint32_t u) const { return group_major ? (int)(u / (uint32_t)taps) : (int)(u % (uint32_t)CU); }
};

static AcContext* g_ctx = nullptr;
static AcDeviceLimits g_limits = {};
static uint64_t g_span = 0;
static std::vector<AcKernel*> g_kernels;
static std::vector<AcBuffer*> g_buffers;

static const AcRuntimeBindingDesc kBindings[7] = {
    { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 3, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 4, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 5, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 6, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }};

static AcKernel* make_kernel(const char* name, const uint32_t* spirv, size_t bytes, const char* sha, uint32_t bindings,
                             uint32_t push_bytes) {
    const AcKernelRuntimeMetadata meta = { name, "main", bindings, kBindings, push_bytes, { 128, 1, 1 }, 0, sha, 0, nullptr };
    const AcKernelDesc kd = { AC_ABI_VERSION, sizeof(AcKernelDesc), spirv, bytes, &meta, 0, nullptr };
    AcKernel* k = nullptr;
    const AcResult r = ac_runtime_kernel_create(g_ctx, &kd, &k);
    if (r != AC_SUCCESS || !k) die("kernel %s: creation failed (code %d)", name, (int)r);
    g_kernels.push_back(k);
    return k;
}

struct Buf {
    AcBuffer* b = nullptr;
    uint64_t bytes = 0; // allocated and bound size (a multiple of 64)
};
static Buf make_buf(uint64_t bytes, const char* what) {
    Buf r;
    r.bytes = (std::max<uint64_t>(bytes, 1) + 63) & ~63ull;
    const AcBufferDesc bd = { AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)r.bytes, AC_BUFFER_USAGE_STORAGE, 0 };
    const AcResult res = ac_runtime_buffer_create(g_ctx, &bd, &r.b);
    if (res != AC_SUCCESS || !r.b) die("buffer %s (%llu bytes): creation failed (code %d)", what, (unsigned long long)r.bytes, (int)res);
    g_buffers.push_back(r.b);
    CHECK(ac_runtime_buffer_fill(r.b, 0, (size_t)r.bytes, 0));
    return r;
}
static AcBufferBinding whole(const Buf& b, const char* what) {
    if (b.bytes > g_span) die("%s (%llu bytes) is larger than one binding (%llu): not handled by this bench", what,
                              (unsigned long long)b.bytes, (unsigned long long)g_span);
    return { b.b, 0, b.bytes };
}

// Submission sizing of ggml-ameva (ggml-ameva.cpp kFlushMac, kDispatchMac, kFlushDispatches)
static const double kFlushMac = 4e9, kDispatchMac = 4e9;
static const uint32_t kFlushDispatches = 128;

struct Recorder {
    AcCommandBatch* batch = nullptr;
    double mac_since_flush = 0;
    uint32_t since_flush = 0, dispatches = 0;
    void dispatch(AcKernel* k, const AcBufferBinding* bind, uint32_t nbind, const void* pc, size_t pc_bytes, AcDim3 grid,
                  double mac) {
        for (int d = 0; d < 3; ++d) {
            const uint32_t g = d == 0 ? grid.x : (d == 1 ? grid.y : grid.z);
            if (g == 0 || g > g_limits.max_compute_workgroup_count[d])
                die("grid dimension %d = %u outside 1..%u", d, g, g_limits.max_compute_workgroup_count[d]);
        }
        CHECK(ac_runtime_batch_dispatch_bindings(batch, k, bind, nbind, pc, pc_bytes, grid));
        ++dispatches;
        mac_since_flush += mac;
        if (++since_flush >= kFlushDispatches || mac_since_flush >= kFlushMac) {
            CHECK(ac_runtime_batch_flush(batch));
            mac_since_flush = 0;
            since_flush = 0;
        }
    }
};

struct RunTime {
    double wall_ms = 0, gpu_ms = 0;
    uint32_t dispatches = 0;
};
template <class F> static RunTime run_once(F&& record) {
    const AcBatchDesc bd = { AC_ABI_VERSION, sizeof(AcBatchDesc), 4096,
                             AC_BATCH_FLAG_EXPLICIT_BARRIERS | (g_limits.timestamps_supported ? AC_BATCH_FLAG_TIMESTAMPS : 0u) };
    Recorder rec;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(ac_runtime_batch_begin_ex(g_ctx, &bd, &rec.batch));
    record(rec);
    AcEvent* ev = nullptr;
    CHECK(ac_runtime_batch_submit(rec.batch, &ev));
    CHECK(ac_runtime_event_wait(ev, 120000000000ull));
    RunTime rt;
    rt.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint64_t wall_ns = 0, gpu_ns = 0;
    if (ac_runtime_event_get_timing(ev, &wall_ns, &gpu_ns) == AC_SUCCESS) rt.gpu_ms = (double)gpu_ns * 1e-6;
    rt.dispatches = rec.dispatches;
    CHECK(ac_runtime_event_destroy(ev));
    ac_runtime_batch_destroy(rec.batch);
    return rt;
}

static int usable_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 1;
    return std::max(1, CPU_COUNT(&set));
}

// f32 convolution, planar output ref[n * P + position] without bias. Each output is the sum over the units and their
// channels in the order of the GPU kernels' K. fused: each term enters with a fused multiply-add, which is what the
// kernels' fma() asks for; otherwise with a rounded product and a rounded sum, which is what a driver computes when
// it does not fuse fma() (GLSL allows both).
static void ref_conv(const Shape& s, const float* x, const float* wd, float* ref, int threads, bool fused) {
    auto work = [&](int na, int nb) {
        for (int n = na; n < nb; ++n) {
            float* out = ref + (uint64_t)n * s.P;
            memset(out, 0, s.P * sizeof(float));
            for (uint32_t u = 0; u < s.U; ++u) {
                const int tap = s.tap_of(u), dx = tap % s.KW, dy = (tap / s.KW) % s.KH, dt = tap / (s.KW * s.KH);
                const int off = dx * s.DW - s.PW;
                const int lo = off < 0 ? (-off + s.SW - 1) / s.SW : 0;
                const int hi = s.W - 1 - off < 0 ? 0 : std::min(s.OW, (s.W - 1 - off) / s.SW + 1);
                for (int c = s.cu_of(u) * 32; c < std::min(s.C, s.cu_of(u) * 32 + 32); ++c) {
                    const float w = wd[(uint64_t)n * s.K + (uint64_t)u * 32 + c % 32];
                    for (int ot = 0; ot < s.OT; ++ot) {
                        const int it = ot * s.ST + dt * s.DT - s.PT0;
                        if (it < 0 || it >= s.T) continue;
                        for (int oh = 0; oh < s.OH; ++oh) {
                            const int ih = oh * s.SH + dy * s.DH - s.PH;
                            if (ih < 0 || ih >= s.H) continue;
                            const float* in = x + (uint64_t)s.W * (ih + (uint64_t)s.H * (it + (uint64_t)s.T * c)) + off;
                            float* o = out + (uint64_t)s.OW * (oh + (uint64_t)s.OH * ot);
                            if (!fused) {
#pragma clang fp contract(off)
                                for (int ow = lo; ow < hi; ++ow) o[ow] = o[ow] + w * in[ow * s.SW];
                            } else if (s.SW == 1) {
                                for (int ow = lo; ow < hi; ++ow) o[ow] = __builtin_fmaf(w, in[ow], o[ow]);
                            } else {
                                for (int ow = lo; ow < hi; ++ow) o[ow] = __builtin_fmaf(w, in[ow * s.SW], o[ow]);
                            }
                        }
                    }
                }
            }
        }
    };
    threads = std::max(1, std::min(threads, (int)s.N));
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(work, (int)((int64_t)s.N * t / threads), (int)((int64_t)s.N * (t + 1) / threads));
    for (std::thread& t : pool) t.join();
}

struct Diff {
    double max_over_rms = 0, rms = 0;
    uint64_t nan = 0;
    bool ok = false;
};
// got: position-major [P][N] (matrix path, no bias) or planar [N][P] with the bias added (conv paths)
static Diff compare(const Shape& s, const float* got, bool planar, const float* ref, const float* bias) {
    Diff d;
    double ss = 0, maxd = 0;
    for (uint32_t n = 0; n < s.N; ++n) {
        for (uint64_t p = 0; p < s.P; ++p) {
            const float r = planar ? ref[n * s.P + p] + bias[n] : ref[n * s.P + p];
            const float g = planar ? got[n * s.P + p] : got[p * s.N + n];
            if (g != g) {
                ++d.nan;
                continue;
            }
            ss += (double)r * r;
            maxd = std::max(maxd, std::fabs((double)g - (double)r));
        }
    }
    const double rms = std::sqrt(ss / ((double)s.N * (double)s.P));
    d.rms = rms;
    d.max_over_rms = rms > 0 ? maxd / rms : maxd;
    d.ok = d.nan == 0 && rms > 0 && d.max_over_rms <= 1e-4; // f32 sums of K terms; fma contraction may differ between GPU and CPU
    return d;
}

struct Path {
    std::string type, name;
    AcKernel* kernel = nullptr;
    bool matrix = false, ablation = false;
    const Buf *q = nullptr, *s = nullptr;
    uint32_t qper = 0, s_row = 0; // Q vector elements per unit, S bytes per weight row
    Diff diff;
    std::vector<double> wall, gpu;
    uint32_t dispatches = 0;
};

struct KernelDef {
    const char* path;   // name in the output
    const char* kernel; // name given to the runtime
    const uint32_t* spv;
    size_t bytes;
    const char* sha;
    bool f16, matrix, vec4, ablation;
};
#define KERNEL(path, name, var, f16, matrix, vec4, ablation) \
    { path, "bench_" name, k##var##Spv, sizeof(k##var##Spv), k##var##SpvSha256, f16, matrix, vec4, ablation }
static const KernelDef kKernels[] = {
    KERNEL("matrix", "qgemm_q8_0", QgemmQ80, false, true, false, false),
    KERNEL("matrix_noload", "qgemm_q8_0_a1", QgemmQ80A1, false, true, false, true),
    KERNEL("matrix_noload_v", "qgemm_q8_0_a2", QgemmQ80A2, false, true, false, true),
    KERNEL("conv_s", "conv_q8_0_s", ConvQ80S, false, false, false, false),
    KERNEL("conv_v", "conv_q8_0_v", ConvQ80V, false, false, true, false),
    KERNEL("conv_v_direct", "conv_q8_0_vd", ConvQ80Vd, false, false, true, false),
    KERNEL("conv_v_nomask", "conv_q8_0_va1", ConvQ80Va1, false, false, true, true),
    KERNEL("conv_v_oneload", "conv_q8_0_va2", ConvQ80Va2, false, false, true, true),
    KERNEL("conv_v_noload", "conv_q8_0_va3", ConvQ80Va3, false, false, true, true),
    KERNEL("conv_v_noload_pmout", "conv_q8_0_va4", ConvQ80Va4, false, false, true, true),
    KERNEL("matrix", "qgemm_f16", QgemmF16, true, true, false, false),
    KERNEL("matrix_noload", "qgemm_f16_a1", QgemmF16A1, true, true, false, true),
    KERNEL("matrix_noload_v", "qgemm_f16_a2", QgemmF16A2, true, true, false, true),
    KERNEL("conv_s", "conv_f16_s", ConvF16S, true, false, false, false),
    KERNEL("conv_v", "conv_f16_v", ConvF16V, true, false, true, false),
    KERNEL("conv_v_direct", "conv_f16_vd", ConvF16Vd, true, false, true, false),
    KERNEL("conv_v_nomask", "conv_f16_va1", ConvF16Va1, true, false, true, true),
    KERNEL("conv_v_oneload", "conv_f16_va2", ConvF16Va2, true, false, true, true),
    KERNEL("conv_v_noload", "conv_f16_va3", ConvF16Va3, true, false, true, true),
    KERNEL("conv_v_noload_pmout", "conv_f16_va4", ConvF16Va4, true, false, true, true),
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s W,H,T,C,OC,KW,KH,KT[,PW,PH,PT0,PT1[,SW,SH,ST[,DW,DH,DT]]] [ITER] [TYPES]\n", argv[0]);
        return 2;
    }
    Shape s = {};
    int v[18];
    int nv = 0;
    for (const char* p = argv[1]; *p && nv < 18;) {
        char* end = nullptr;
        v[nv++] = (int)strtol(p, &end, 10);
        if (end == p || (*end && *end != ',')) die("bad shape \"%s\"", argv[1]);
        p = *end ? end + 1 : end;
    }
    if (nv != 8 && nv != 12 && nv != 15 && nv != 18) die("shape needs 8, 12, 15 or 18 numbers (got %d)", nv);
    s.W = v[0], s.H = v[1], s.T = v[2], s.C = v[3], s.OC = v[4], s.KW = v[5], s.KH = v[6], s.KT = v[7];
    s.PW = nv >= 12 ? v[8] : (s.KW - 1) / 2, s.PH = nv >= 12 ? v[9] : (s.KH - 1) / 2;
    s.PT0 = nv >= 12 ? v[10] : (s.KT - 1) / 2, s.PT1 = nv >= 12 ? v[11] : (s.KT - 1) / 2;
    s.SW = nv >= 15 ? v[12] : 1, s.SH = nv >= 15 ? v[13] : 1, s.ST = nv >= 15 ? v[14] : 1;
    s.DW = nv >= 18 ? v[15] : 1, s.DH = nv >= 18 ? v[16] : 1, s.DT = nv >= 18 ? v[17] : 1;
    const int iters = argc > 2 ? atoi(argv[2]) : 4;
    const std::string types = argc > 3 ? argv[3] : "q8_0,f16";
    if (s.W < 1 || s.H < 1 || s.T < 1 || s.C < 1 || s.OC < 1 || s.KW < 1 || s.KH < 1 || s.KT < 1 || s.KW > 255 ||
        s.KH > 255 || s.KT > 255 || s.PW < 0 || s.PH < 0 || s.PT0 < 0 || s.PT1 < 0 || s.PW > 255 || s.PH > 255 ||
        s.PT0 > 255 || s.SW < 1 || s.SH < 1 || s.ST < 1 || s.SW > 255 || s.SH > 255 || s.ST > 255 || s.DW < 1 ||
        s.DH < 1 || s.DT < 1 || s.DW > 255 || s.DH > 255 || s.DT > 255 || iters < 1)
        die("bad arguments");
    s.OW = (s.W + 2 * s.PW - s.DW * (s.KW - 1) - 1) / s.SW + 1;
    s.OH = (s.H + 2 * s.PH - s.DH * (s.KH - 1) - 1) / s.SH + 1;
    s.OT = (s.T + s.PT0 + s.PT1 - s.DT * (s.KT - 1) - 1) / s.ST + 1;
    if (s.W + 2 * s.PW < s.DW * (s.KW - 1) + 1 || s.H + 2 * s.PH < s.DH * (s.KH - 1) + 1 ||
        s.T + s.PT0 + s.PT1 < s.DT * (s.KT - 1) + 1)
        die("kernel larger than the padded source");
    s.taps = s.KW * s.KH * s.KT;
    s.CP = (s.C + 31) & ~31;
    s.CU = s.CP / 32;
    const char* ko = getenv("AMEVA_CONV_BENCH_KORDER");
    if (ko && *ko && strcmp(ko, "tap") != 0 && strcmp(ko, "group") != 0) die("AMEVA_CONV_BENCH_KORDER must be tap or group");
    s.group_major = !(ko && strcmp(ko, "tap") == 0);
    if (s.CU > 256 || (s.KW - 1) * s.DW > 255 || (s.KH - 1) * s.DH > 255 || (s.KT - 1) * s.DT > 255)
        die("unit table: more than 8192 channels or a tap offset above 255");
    s.K = (uint32_t)s.taps * (uint32_t)s.CP;
    s.U = s.K / 32;
    s.N = (uint32_t)s.OC;
    s.P = (uint64_t)s.OW * s.OH * s.OT;
    const uint64_t src_floats = (uint64_t)s.W * s.H * s.T * s.C;
    if (s.P * s.N > 0xFFFFFFFFull || src_floats > 0x7FFFFFFFull) die("tensor too large for 32-bit element offsets");

    const AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    CHECK(ac_runtime_context_create(&cdesc, &g_ctx));
    g_limits.abi_version = AC_ABI_VERSION;
    g_limits.struct_size = sizeof(AcDeviceLimits);
    CHECK(ac_runtime_context_get_limits(g_ctx, &g_limits));
    g_span = g_limits.max_storage_buffer_range & ~63ull;
    char dev[256] = "?";
    CHECK(ac_runtime_context_get_device_name(g_ctx, dev, sizeof(dev)));
    const double gflop = 2.0 * (double)s.P * s.N * (double)s.taps * s.C * 1e-9;
    printf("CONV_BENCH device=\"%s\" span=%llu shape=%dx%dx%dx%d->%d kernel=%dx%dx%d pad=%d,%d,%d+%d stride=%d,%d,%d "
           "dil=%d,%d,%d out=%dx%dx%d M=%llu N=%u K=%u korder=%s gflop=%.2f iters=%d\n",
           dev, (unsigned long long)g_span, s.W, s.H, s.T, s.C, s.OC, s.KW, s.KH, s.KT, s.PW, s.PH, s.PT0, s.PT1, s.SW,
           s.SH, s.ST, s.DW, s.DH, s.DT, s.OW, s.OH, s.OT, (unsigned long long)s.P, s.N, s.K,
           s.group_major ? "group" : "tap", gflop, iters);
    {
        const char* g = getenv("AMEVA_CONV_BENCH_GRID");
        printf("CONV_BENCH conv_grid=%s\n", g && strcmp(g, "3d") == 0 ? "3d (w tiles in y, rows in z)" : "flat (w tiles x rows in y)");
    }
    fflush(stdout);

    // source, bias, weights (f32; the padded channels of K are zero), unit table (shaders/ameva_conv.comp)
    std::vector<float> x(src_floats), bias(s.N), wf((size_t)s.N * s.K, 0.0f);
    std::vector<uint32_t> units(s.U);
    for (float& f : x) f = rnd();
    for (float& f : bias) f = rnd();
    for (uint32_t n = 0; n < s.N; ++n)
        for (uint32_t u = 0; u < s.U; ++u)
            for (int i = 0; i < 32 && s.cu_of(u) * 32 + i < s.C; ++i) wf[(size_t)n * s.K + (size_t)u * 32 + i] = rnd();
    for (uint32_t u = 0; u < s.U; ++u) {
        const int tap = s.tap_of(u), dx = tap % s.KW, dy = (tap / s.KW) % s.KH, dt = tap / (s.KW * s.KH);
        units[u] = (uint32_t)(dx * s.DW) | (uint32_t)(dy * s.DH) << 8 | (uint32_t)(dt * s.DT) << 16 | (uint32_t)s.cu_of(u) << 24;
    }

    const Buf b_src = make_buf(src_floats * 4, "source");
    const Buf b_bias = make_buf((uint64_t)s.N * 4, "bias");
    const Buf b_units = make_buf((uint64_t)s.U * 4, "unit table");
    const Buf b_dst_c = make_buf(s.P * s.N * 4, "conv dst");
    CHECK(ac_runtime_buffer_write(b_src.b, 0, x.data(), src_floats * 4));
    CHECK(ac_runtime_buffer_write(b_bias.b, 0, bias.data(), (size_t)s.N * 4));
    CHECK(ac_runtime_buffer_write(b_units.b, 0, units.data(), (size_t)s.U * 4));

    // column matrix A[m][k], m = ow + OW * (oh + OH * ot), k = 32 * unit(tap, c / 32) + c % 32
    const uint64_t mat_bytes = s.P * s.K * 4;
    const char* mm = getenv("AMEVA_CONV_BENCH_MATRIX_MAX_MB");
    const uint64_t mat_max = (mm && *mm ? strtoull(mm, nullptr, 10) : 512ull) << 20;
    const bool with_matrix = mat_bytes <= mat_max;
    Buf b_mat, b_dst_m;
    if (with_matrix) {
        b_mat = make_buf(mat_bytes, "column matrix");
        b_dst_m = make_buf(s.P * s.N * 4, "matrix dst");
        const uint64_t chunk = std::max<uint64_t>(1, (8ull << 20) / s.K); // rows per write (32 MiB)
        std::vector<float> rows(chunk * s.K);
        for (uint64_t m0 = 0; m0 < s.P; m0 += chunk) {
            const uint64_t mc = std::min(chunk, s.P - m0);
            std::fill(rows.begin(), rows.begin() + mc * s.K, 0.0f);
            for (uint64_t m = m0; m < m0 + mc; ++m) {
                const int ow = (int)(m % s.OW), oh = (int)((m / s.OW) % s.OH), ot = (int)(m / ((uint64_t)s.OW * s.OH));
                float* row = rows.data() + (m - m0) * s.K;
                for (int dt = 0; dt < s.KT; ++dt)
                    for (int dy = 0; dy < s.KH; ++dy)
                        for (int dx = 0; dx < s.KW; ++dx) {
                            const int iw = ow * s.SW + dx * s.DW - s.PW, ih = oh * s.SH + dy * s.DH - s.PH,
                                      it = ot * s.ST + dt * s.DT - s.PT0;
                            if (iw < 0 || iw >= s.W || ih < 0 || ih >= s.H || it < 0 || it >= s.T) continue;
                            const int tap = dx + s.KW * (dy + s.KH * dt);
                            const float* in = x.data() + iw + (uint64_t)s.W * (ih + (uint64_t)s.H * it);
                            const uint64_t cstride = (uint64_t)s.W * s.H * s.T;
                            for (int c = 0; c < s.C; ++c) row[(size_t)s.unit_of(tap, c / 32) * 32 + c % 32] = in[c * cstride];
                        }
            }
            CHECK(ac_runtime_buffer_write(b_mat.b, m0 * s.K * 4, rows.data(), mc * s.K * 4));
        }
    } else {
        printf("CONV_BENCH path=matrix SKIPPED: column matrix %llu MiB above the limit of %llu MiB "
               "(AMEVA_CONV_BENCH_MATRIX_MAX_MB)\n",
               (unsigned long long)(mat_bytes >> 20), (unsigned long long)(mat_max >> 20));
    }

    const uint32_t stride = (uint32_t)s.SW | (uint32_t)s.SH << 8 | (uint32_t)s.ST << 16;
    const uint32_t pad = (uint32_t)s.PW | (uint32_t)s.PH << 8 | (uint32_t)s.PT0 << 16;

    // weight rows per dispatch: whole 64-row tiles under kDispatchMac (ggml-ameva enc_mul_mat, ameva_qgemm)
    const uint64_t mac_rows = (uint64_t)(kDispatchMac / ((double)s.P * (double)s.K));
    const uint32_t rows_per_slice = (uint32_t)std::min<uint64_t>(s.N, std::max<uint64_t>(64, mac_rows & ~63ull));

    auto record_matrix = [&](Recorder& rec, const Path& p) {
        const uint64_t rows_fit = g_span / ((uint64_t)s.K * 4);
        const uint64_t mc_max = std::min<uint64_t>(rows_fit, 65535ull * 32) & ~31ull;
        if (mc_max == 0) die("one binding holds fewer than 32 rows of the column matrix");
        const AcBufferBinding bq = whole(*p.q, "weights Q"), bs = whole(*p.s, "weights S"), bd = whole(b_dst_m, "matrix dst");
        for (uint32_t n0 = 0; n0 < s.N; n0 += rows_per_slice) {
            const uint32_t nc = std::min(rows_per_slice, s.N - n0);
            for (uint64_t m0 = 0; m0 < s.P; m0 += mc_max) {
                const uint32_t mc = (uint32_t)std::min<uint64_t>(mc_max, s.P - m0);
                const uint64_t off = m0 * s.K * 4;
                if (off % g_limits.min_storage_buffer_offset_alignment) die("matrix row offset not aligned for a binding");
                const AcBufferBinding bind[5] = { bq, bs, bs, { b_mat.b, off, (uint64_t)mc * s.K * 4 }, bd };
                const uint32_t pc[10] = { mc, s.K, nc, n0 * s.U * p.qper, 0u, n0 * p.s_row, 0u,
                                          (uint32_t)(m0 * s.N + n0), s.K, s.N };
                rec.dispatch(p.kernel, bind, 5, pc, sizeof(pc), { (nc + 63) / 64, (mc + 31) / 32, 1 }, (double)mc * nc * s.K);
            }
        }
    };
    const char* gm = getenv("AMEVA_CONV_BENCH_GRID");
    if (gm && *gm && strcmp(gm, "flat") != 0 && strcmp(gm, "3d") != 0) die("AMEVA_CONV_BENCH_GRID must be flat or 3d");
    const bool grid_3d = gm && strcmp(gm, "3d") == 0;
    auto record_conv = [&](Recorder& rec, const Path& p) {
        const AcBufferBinding bq = whole(*p.q, "weights Q"), bs = whole(*p.s, "weights S");
        const AcBufferBinding bind[7] = { bq, bs, bs, whole(b_src, "source"), whole(b_dst_c, "conv dst"), whole(b_bias, "bias"),
                                          whole(b_units, "unit table") };
        // tiles along w times output rows: one grid dimension (rows split over dispatches when it does not hold
        // them), or w tiles in y and rows in z (AMEVA_CONV_BENCH_GRID=3d)
        const uint32_t tiles_w = ((uint32_t)s.OW + 31) / 32, rows = (uint32_t)(s.OH * s.OT);
        const uint32_t rows_max = grid_3d ? g_limits.max_compute_workgroup_count[2] : g_limits.max_compute_workgroup_count[1] / tiles_w;
        if (rows_max == 0) die("one output row has more tiles (%u) than a grid dimension holds", tiles_w);
        for (uint32_t n0 = 0; n0 < s.N; n0 += rows_per_slice) {
            const uint32_t nc = std::min(rows_per_slice, s.N - n0);
            for (uint32_t r0 = 0; r0 < rows; r0 += rows_max) {
                const uint32_t rc = std::min(rows_max, rows - r0);
                const uint32_t pc[19] = { nc, s.K, n0 * s.U * p.qper, 0u, n0 * p.s_row, 0u, (uint32_t)(n0 * s.P), n0,
                                          (uint32_t)s.W, (uint32_t)s.H, (uint32_t)s.T, (uint32_t)s.C,
                                          (uint32_t)s.OW, (uint32_t)s.OH, (uint32_t)s.P, r0, stride, pad, 0u };
                const AcDim3 grid = grid_3d ? AcDim3{ (nc + 63) / 64, tiles_w, rc } : AcDim3{ (nc + 63) / 64, tiles_w * rc, 1 };
                rec.dispatch(p.kernel, bind, 7, pc, sizeof(pc), grid, (double)rc * s.OW * nc * s.K);
            }
        }
    };

    const int threads = usable_cpus();
    const char* rf = getenv("AMEVA_CONV_BENCH_REF");
    if (rf && *rf && strcmp(rf, "fma") != 0 && strcmp(rf, "muladd") != 0) die("AMEVA_CONV_BENCH_REF must be fma or muladd");
    const bool ref_fused = !(rf && strcmp(rf, "muladd") == 0);
    const char* ab = getenv("AMEVA_CONV_BENCH_ABLATE");
    const bool ablate = ab && *ab && strcmp(ab, "0") != 0;
    std::vector<Path> paths;
    std::vector<Buf> wbufs;
    wbufs.reserve(8);
    std::vector<float> wd((size_t)s.N * s.K), ref((size_t)s.N * s.P), got((size_t)s.N * s.P);
    int failed = 0;
    for (size_t pos = 0; pos < types.size();) {
        const size_t comma = std::min(types.find(',', pos), types.size());
        const std::string type = types.substr(pos, comma - pos);
        pos = comma + 1;
        const bool f16 = type == "f16";
        if (!f16 && type != "q8_0") die("unknown weight type \"%s\" (q8_0, f16)", type.c_str());

        // repacked layout (ggml-ameva-repack.h): Q row-major, 32 weights per unit; q8_0: S = one f16 scale per unit
        const uint32_t qb = f16 ? 64 : 32;
        std::vector<uint8_t> q((size_t)s.N * s.U * qb), sc(f16 ? 0 : (size_t)s.N * s.U * 2);
        for (size_t blk = 0; blk < (size_t)s.N * s.U; ++blk) {
            const float* w = wf.data() + blk * 32;
            float* o = wd.data() + blk * 32;
            if (f16) {
                for (int i = 0; i < 32; ++i) {
                    const uint16_t h = f2h(w[i]);
                    memcpy(q.data() + blk * 64 + i * 2, &h, 2);
                    o[i] = h2f(h);
                }
            } else { // ggml quantize_row_q8_0_ref / dequantize_row_q8_0
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[i]));
                const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
                const uint16_t dh = f2h(d);
                memcpy(sc.data() + blk * 2, &dh, 2);
                for (int i = 0; i < 32; ++i) {
                    const int8_t qi = (int8_t)roundf(w[i] * id);
                    q[blk * 32 + i] = (uint8_t)qi;
                    o[i] = (float)qi * h2f(dh);
                }
            }
        }
        wbufs.push_back(make_buf(q.size(), "weights Q"));
        const Buf* bq = &wbufs.back();
        CHECK(ac_runtime_buffer_write(bq->b, 0, q.data(), q.size()));
        wbufs.push_back(make_buf(std::max<size_t>(sc.size(), 4), "weights S"));
        const Buf* bs = &wbufs.back();
        if (!sc.empty()) CHECK(ac_runtime_buffer_write(bs->b, 0, sc.data(), sc.size()));

        const auto t0 = std::chrono::steady_clock::now();
        ref_conv(s, x.data(), wd.data(), ref.data(), threads, ref_fused);
        printf("CONV_BENCH type=%s reference: f32 CPU convolution (%s), %d threads, %.0f ms\n", type.c_str(),
               ref_fused ? "fused multiply-add" : "separate multiply and add", threads,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        fflush(stdout);

        const size_t first = paths.size();
        for (const KernelDef& kd : kKernels) {
            if (kd.f16 != f16 || (kd.matrix && !with_matrix) || (kd.ablation && !ablate)) continue;
            if (kd.vec4 && s.SW != 1) {
                printf("CONV_BENCH type=%s path=%s SKIPPED: stride along w is %d (vec4 staging needs 1)\n", type.c_str(),
                       kd.path, s.SW);
                continue;
            }
            Path p;
            p.type = type, p.name = kd.path, p.matrix = kd.matrix, p.ablation = kd.ablation, p.q = bq, p.s = bs;
            p.kernel = make_kernel(kd.kernel, kd.spv, kd.bytes, kd.sha, kd.matrix ? 5 : 7, kd.matrix ? 40 : 76);
            p.qper = qb / 16, p.s_row = f16 ? 0 : s.U * 2;
            paths.push_back(p);
        }

        // first run of every path of this type: destination NaN, compute, compare with the reference of this type
        for (size_t i = first; i < paths.size(); ++i) {
            Path& p = paths[i];
            const Buf& dstb = p.matrix ? b_dst_m : b_dst_c;
            CHECK(ac_runtime_buffer_fill(dstb.b, 0, (size_t)dstb.bytes, 0xFF));
            const RunTime rt = run_once([&](Recorder& rec) { p.matrix ? record_matrix(rec, p) : record_conv(rec, p); });
            p.dispatches = rt.dispatches;
            if (p.ablation) {
                printf("CONV_BENCH type=%s path=%s ABLATION: a kernel with a part of its work removed, output not compared\n",
                       p.type.c_str(), p.name.c_str());
                continue;
            }
            CHECK(ac_runtime_buffer_read(dstb.b, 0, got.data(), got.size() * 4));
            p.diff = compare(s, got.data(), !p.matrix, ref.data(), bias.data());
            if (!p.diff.ok) ++failed;
            printf("CONV_BENCH type=%s path=%s check: max_diff_over_rms=%.2e ref_rms=%.3f nan=%llu dispatches=%u first_run_ms=%.1f %s\n",
                   p.type.c_str(), p.name.c_str(), p.diff.max_over_rms, p.diff.rms, (unsigned long long)p.diff.nan, p.dispatches,
                   rt.wall_ms, p.diff.ok ? "MATCH" : "MISMATCH");
            fflush(stdout);
        }
    }

    // timing: every path once per round, in turn. One round before them is not timed: the comparison above leaves
    // the GPU idle, and the first run after it would carry the clock ramp of the governor.
    for (int it = -1; it < iters; ++it) {
        for (Path& p : paths) {
            const RunTime rt = run_once([&](Recorder& rec) { p.matrix ? record_matrix(rec, p) : record_conv(rec, p); });
            if (it < 0) continue;
            p.wall.push_back(rt.wall_ms);
            p.gpu.push_back(rt.gpu_ms);
        }
    }
    for (const Path& p : paths) {
        double sum = 0, gsum = 0, mn = 1e300;
        for (size_t i = 0; i < p.wall.size(); ++i) sum += p.wall[i], gsum += p.gpu[i], mn = std::min(mn, p.wall[i]);
        const double ms = sum / p.wall.size();
        double base = 0; // mean of the matrix path of the same type
        for (const Path& m : paths) {
            if (m.matrix && !m.ablation && m.type == p.type) {
                for (double w : m.wall) base += w;
                base /= m.wall.size();
            }
        }
        char rel[48] = "";
        if (base > 0 && !p.matrix) snprintf(rel, sizeof(rel), " matrix_over_this=%.2f", base / ms);
        printf("CONV_BENCH type=%s path=%s ms=%.1f min_ms=%.1f gpu_ms=%.1f gflops=%.1f%s %s\n", p.type.c_str(),
               p.name.c_str(), ms, mn, gsum / p.wall.size(), gflop / (ms * 1e-3), rel,
               p.ablation ? "ABLATION" : (p.diff.ok ? "MATCH" : "MISMATCH"));
    }
    printf("CONV_BENCH RESULT paths=%zu mismatched=%d matrix=%s\n", paths.size(), failed, with_matrix ? "run" : "skipped");
    for (AcKernel* k : g_kernels) CHECK(ac_runtime_kernel_destroy(k));
    for (AcBuffer* b : g_buffers) CHECK(ac_runtime_buffer_destroy(b));
    CHECK(ac_runtime_context_destroy(g_ctx));
    return failed ? 1 : 0;
}
