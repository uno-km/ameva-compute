#include "ggml-ameva.h"
#include "ggml-ameva-impl.h"
#include "ggml-ameva-repack.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"
#include "ggml_ameva_kernels_spv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>

// =============================================================================
// AMEVA graph engine
// =============================================================================
// graph_compute records every node of the graph it receives into ONE ac_runtime command batch and submits it once:
// one fence wait per graph instead of one per node. Dependencies are resolved on the host from the byte ranges the
// nodes read and write (RAW, WAR and WAW on the same buffer get a barrier, independent nodes share a barrier
// interval). Every tensor is bound as a buffer range starting at its offset rounded down to the device's storage
// offset alignment, so no binding exceeds maxStorageBufferRange (256 MiB on Mali-G68); weights larger than one
// binding are multiplied in row slices.
//
// Kernels (shaders/, built by scripts/build_ameva_kernels.py, SPIR-V pinned by SHA-256):
//   MUL_MAT                    ameva_qmv: src0 stored in the AMEVA repacked layout (quantized weights of weight
//                              buffers, ggml-ameva-repack.h), 12 types; for one activation row the work split
//                              (ameva_qmv layouts or mali_qmv) is measured per device and weight shape (token-
//                              generation GEMV selection); ameva_qgemm (64 x 32 tiles) for those weights when many
//                              activation rows are multiplied (prompt processing)
//                              ameva_qmm: src0 read in its raw ggml format, 15 weight types (everything else)
//   ADD SUB MUL DIV            ameva_binary (broadcasting src1)
//   SCALE, UNARY (TANH SILU GELU GELU_ERF GELU_QUICK RELU SIGMOID NEG SQR SQRT EXP ABS)   ameva_unary
//   GLU (REGLU GEGLU SWIGLU GEGLU_ERF GEGLU_QUICK)      ameva_glu
//   RMS_NORM, ROPE (normal, neox, frequency factors), SET_ROWS, GET_ROWS, CPY/CONT/DUP (f32/f16)
//   FLASH_ATTN_EXT             ameva_flash_attn: tiles of query rows of one KV head (GQA heads share K/V loads),
//                              f16 or f32 K/V, f16 mask, softcap, ALiBi; fully masked key blocks skipped
// Index-driven kernels report out-of-range indices through a status buffer; the host fails the graph on them.
// Tensors must live in AMEVA buffers: the scheduler copies inputs produced elsewhere (supports_buft accepts only
// AMEVA buffers), nothing is staged behind its back.

void ameva_increment_gpu_op_count(void);
void ameva_increment_q5_0_cpu_dequant_count(void);
void ameva_increment_f32_gpu_sgemm_dispatch_count(void);
void ameva_increment_context_create_count(void);
void ameva_increment_context_destroy_count(void);
void ameva_increment_kernel_create_count(void);
void ameva_increment_kernel_destroy_count(void);
void ameva_increment_buffer_create_count(void);
void ameva_increment_buffer_destroy_count(void);
void ameva_increment_event_create_count(void);
void ameva_increment_event_destroy_count(void);
void ameva_increment_q5_0_staging_alloc_count(void);
void ameva_increment_q5_0_staging_free_count(void);

// =============================================================================
// Profile (AMEVA_PROFILE=1 prints the totals at process exit)
// =============================================================================
static std::atomic<uint64_t> g_prof_graphs{0}, g_prof_nodes{0}, g_prof_dispatches{0}, g_prof_barriers{0},
    g_prof_mul_mat{0}, g_prof_empty{0}, g_prof_cache_hits{0}, g_prof_cache_misses{0}, g_prof_flushes{0};
static std::atomic<uint64_t> g_prof_encode_ns{0}, g_prof_wait_ns{0}, g_prof_gpu_ns{0}, g_prof_gpu_samples{0};
// parts of encode: contract pass, batch begin, intermediate submissions, final submission (the rest is recording)
static std::atomic<uint64_t> g_prof_contract_ns{0}, g_prof_begin_ns{0}, g_prof_flush_ns{0}, g_prof_submit_ns{0};
// parts of recording: time inside ac_runtime_batch_barrier and ac_runtime_batch_dispatch_bindings (the rest is the
// engine: contract checks of the encoders, bindings, hazard tracking)
static std::atomic<uint64_t> g_prof_rt_barrier_ns{0}, g_prof_rt_dispatch_ns{0};
// CPU time of ameva_encode per ggml op (AMEVA_PROFILE >= 1)
static std::atomic<uint64_t> g_prof_enc_ns[GGML_OP_COUNT], g_prof_enc_nodes[GGML_OP_COUNT];
// pipeline creation (first use of every kernel, inside encoding); excluded from the encode figures above
static std::atomic<uint64_t> g_prof_kcreate_ns{0}, g_prof_kcreate_n{0};
static std::atomic<uint64_t> g_prof_fused{0}; // node pairs fused (ameva_find_fusion)
static std::atomic<uint64_t> g_prof_mmvq{0};  // one-row MUL_MATs run by ameva_mmvq

int64_t ggml_backend_ameva_get_fused_pair_count(void) {
    return (int64_t)g_prof_fused.load();
}

static inline uint64_t ameva_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// AMEVA_PROFILE: 1 = totals at exit, 2 = also GPU time per op kind (a timestamp after every dispatch)
static int ameva_profile_level(void) {
    static const int level = [] {
        const char* e = getenv("AMEVA_PROFILE");
        if (!e) return 0;
        if (strcmp(e, "1") == 0) return 1;
        if (strcmp(e, "2") == 0) return 2;
        return 0;
    }();
    return level;
}

// per-op-kind GPU time (AMEVA_PROFILE=2)
struct AmevaOpKind {
    char name[48];
    uint64_t ns;
    uint64_t dispatches;
};
static std::mutex s_kind_mutex;
static AmevaOpKind s_kinds[96];
static int s_n_kinds = 0;

static uint16_t ameva_kind_id(const char* name) {
    std::lock_guard<std::mutex> lock(s_kind_mutex);
    for (int i = 0; i < s_n_kinds; ++i) {
        if (strcmp(s_kinds[i].name, name) == 0) return (uint16_t)i;
    }
    if (s_n_kinds == (int)(sizeof(s_kinds) / sizeof(s_kinds[0]))) return (uint16_t)(s_n_kinds - 1); // last = "other"
    AmevaOpKind& k = s_kinds[s_n_kinds];
    snprintf(k.name, sizeof(k.name), "%s", s_n_kinds == (int)(sizeof(s_kinds) / sizeof(s_kinds[0])) - 1 ? "other" : name);
    k.ns = 0;
    k.dispatches = 0;
    return (uint16_t)s_n_kinds++;
}

// =============================================================================
// Declined-op audit
// =============================================================================
// An op supports_op declines runs on another backend (the scheduler splits the graph there), and a weight whose
// MUL_MAT is declined stays in CPU memory (llama.cpp probes supports_op with a zero-size AMEVA buffer while placing
// weights). Correct, but not to be invisible: every new (op, type, reason) is printed once, and AMEVA_PROFILE=1 adds
// the query counts at exit.
struct AmevaDecline {
    const char* reason; // static string
    char op[24];
    char type[16];
    char example[64];
    uint64_t queries;
};
static std::mutex s_decline_mutex;
static AmevaDecline s_declines[48];
static int s_n_declines = 0;
static uint64_t s_declines_untracked = 0;

void ameva_audit_decline(const struct ggml_tensor* op, const char* why) {
    if (!op || !why) return;
    const char* on = ggml_op_desc(op);
    const char* tn = ggml_type_name(op->src[0] ? op->src[0]->type : op->type);
    std::lock_guard<std::mutex> lock(s_decline_mutex);
    for (int i = 0; i < s_n_declines; ++i) {
        AmevaDecline& d = s_declines[i];
        if (d.reason == why && strcmp(d.op, on ? on : "?") == 0 && strcmp(d.type, tn ? tn : "?") == 0) {
            d.queries++;
            return;
        }
    }
    if (s_n_declines == (int)(sizeof(s_declines) / sizeof(s_declines[0]))) {
        ++s_declines_untracked;
        return;
    }
    AmevaDecline& d = s_declines[s_n_declines++];
    d.reason = why;
    snprintf(d.op, sizeof(d.op), "%s", on ? on : "?");
    snprintf(d.type, sizeof(d.type), "%s", tn ? tn : "?");
    snprintf(d.example, sizeof(d.example), "%s", op->name);
    d.queries = 1;
    fprintf(stderr, "ggml-ameva: declined %s (%s, %s): %s -- it runs on another backend\n", d.op, d.example, d.type,
            why);
}

static bool ameva_fuse_enabled(void);
static void ameva_qmv_layout_report(void);

static void ameva_profile_report(void) {
    if (ameva_profile_level() == 0) return;
    const uint64_t graphs = g_prof_graphs.load(), gs = g_prof_gpu_samples.load();
    if (ameva_profile_level() >= 2) {
        std::lock_guard<std::mutex> lock(s_kind_mutex);
        std::vector<int> order(s_n_kinds);
        uint64_t total = 0;
        for (int i = 0; i < s_n_kinds; ++i) {
            order[i] = i;
            total += s_kinds[i].ns;
        }
        std::sort(order.begin(), order.end(), [](int a, int b) { return s_kinds[a].ns > s_kinds[b].ns; });
        for (int i : order) {
            const AmevaOpKind& k = s_kinds[i];
            fprintf(stderr, "AMEVA_PROFILE kind=\"%s\" dispatches=%llu gpu_ms=%.1f share=%.1f%% per_dispatch_us=%.1f\n",
                    k.name, (unsigned long long)k.dispatches, k.ns / 1e6, total ? 100.0 * k.ns / total : 0.0,
                    k.dispatches ? k.ns / 1e3 / k.dispatches : 0.0);
        }
    }
    fprintf(stderr, "AMEVA_PROFILE graphs=%llu nodes=%llu dispatches=%llu barriers=%llu flushes=%llu mul_mat=%llu "
                    "empty_nodes=%llu descriptor_cache hits=%llu misses=%llu\n",
            (unsigned long long)graphs, (unsigned long long)g_prof_nodes.load(),
            (unsigned long long)g_prof_dispatches.load(), (unsigned long long)g_prof_barriers.load(),
            (unsigned long long)g_prof_flushes.load(), (unsigned long long)g_prof_mul_mat.load(),
            (unsigned long long)g_prof_empty.load(), (unsigned long long)g_prof_cache_hits.load(),
            (unsigned long long)g_prof_cache_misses.load());
    fprintf(stderr, "AMEVA_PROFILE per_graph_ms encode=%.3f wait=%.3f gpu=%.3f (gpu samples %llu)\n",
            graphs ? g_prof_encode_ns.load() / 1e6 / graphs : 0.0, graphs ? g_prof_wait_ns.load() / 1e6 / graphs : 0.0,
            gs ? g_prof_gpu_ns.load() / 1e6 / gs : 0.0, (unsigned long long)gs);
    if (graphs) {
        const double c = g_prof_contract_ns.load() / 1e6 / graphs, b = g_prof_begin_ns.load() / 1e6 / graphs,
                     f = g_prof_flush_ns.load() / 1e6 / graphs, sm = g_prof_submit_ns.load() / 1e6 / graphs,
                     e = g_prof_encode_ns.load() / 1e6 / graphs;
        fprintf(stderr, "AMEVA_PROFILE encode_ms contract=%.3f begin=%.3f record=%.3f flush=%.3f submit=%.3f\n", c, b,
                e - c - b - f - sm, f, sm);
        const double rb = g_prof_rt_barrier_ns.load() / 1e6 / graphs, rd = g_prof_rt_dispatch_ns.load() / 1e6 / graphs;
        const double kc = g_prof_kcreate_ns.load() / 1e6 / graphs;
        fprintf(stderr, "AMEVA_PROFILE record_ms runtime_barrier=%.3f runtime_dispatch=%.3f kernel_create=%.3f "
                        "engine=%.3f (kernel creation: %llu kernels, %.1f ms in total)\n",
                rb, rd, kc, e - c - b - f - sm - rb - rd - kc, (unsigned long long)g_prof_kcreate_n.load(),
                g_prof_kcreate_ns.load() / 1e6);
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            const uint64_t n = g_prof_enc_nodes[op].load();
            if (!n) continue;
            fprintf(stderr, "AMEVA_PROFILE encode_op op=%s nodes_per_graph=%.1f ms_per_graph=%.3f us_per_node=%.2f\n",
                    ggml_op_name((enum ggml_op)op), (double)n / graphs, g_prof_enc_ns[op].load() / 1e6 / graphs,
                    g_prof_enc_ns[op].load() / 1e3 / n);
        }
    }
    fprintf(stderr, "AMEVA_PROFILE mmq_mul_mat=%lld mmvq_mul_mat=%llu fused_pairs=%llu (AMEVA_FUSE=%s)\n",
            (long long)ggml_backend_ameva_get_mmq_mul_mat_count(), (unsigned long long)g_prof_mmvq.load(),
            (unsigned long long)g_prof_fused.load(), ameva_fuse_enabled() ? "1" : "0");
    fprintf(stderr, "AMEVA_PROFILE repack=%s repacked_tensors=%lld repacked_mib=%.1f qmv_mul_mat=%lld\n",
            ameva_repack_enabled() ? "on" : "off", (long long)ggml_backend_ameva_get_repacked_tensor_count(),
            ggml_backend_ameva_get_repacked_bytes() / 1048576.0, (long long)ggml_backend_ameva_get_qmv_mul_mat_count());
    ameva_qmv_layout_report();
    std::lock_guard<std::mutex> lock(s_decline_mutex);
    for (int i = 0; i < s_n_declines; ++i) {
        fprintf(stderr, "AMEVA_PROFILE declined op=%s type=%s reason=\"%s\" queries=%llu example=%s\n",
                s_declines[i].op, s_declines[i].type, s_declines[i].reason,
                (unsigned long long)s_declines[i].queries, s_declines[i].example);
    }
    if (s_declines_untracked) {
        fprintf(stderr, "AMEVA_PROFILE declined untracked=%llu (table full)\n",
                (unsigned long long)s_declines_untracked);
    }
}

// =============================================================================
// Context, device limits, kernels, status buffer
// =============================================================================
static AcContext* s_ctx = nullptr;
static AcDeviceLimits s_limits = {};
static AcKernel* s_qmm[AMEVA_QMM_VARIANT_COUNT] = {};
static AcKernel* s_qmv[AMEVA_QMV_VARIANT_COUNT] = {};
static AcKernel* s_mali_qmv[AMEVA_MALI_QMV_VARIANT_COUNT] = {};
static char s_device_name[128] = "?";
// Device and driver identity for measurements kept across runs: vendor ID, device ID and pipelineCacheUUID from the
// header of the Vulkan pipeline cache data (VkPipelineCacheHeaderVersionOne, which every driver writes); empty when
// unavailable.
static std::string s_device_key;
static AcKernel* s_qgemm[AMEVA_QGEMM_VARIANT_COUNT] = {};
static AcKernel* s_mmq[AMEVA_MMQ_VARIANT_COUNT] = {};
static AcKernel* s_mmvq[AMEVA_MMVQ_VARIANT_COUNT] = {};
static AcKernel* s_mmvqe[AMEVA_MMVQE_VARIANT_COUNT] = {};
static AcKernel* s_fa[AMEVA_FA_VARIANT_COUNT] = {};
static AcKernel* s_ops[AMEVA_K_COUNT] = {};
static AcBuffer* s_status = nullptr; // [0] out-of-range index count, [1] lowest offending node
// Engine-owned device memory for intermediate results that are not tensors (flash-attention key splits). Sized before
// a graph is recorded (ameva_scratch_reserve) and reused by every node of the graph; the hazard tracker orders reuse.
static AcBuffer* s_scratch = nullptr;
static uint64_t s_scratch_bytes = 0;
static uint32_t* s_status_map = nullptr;
static std::mutex s_mutex;          // context, kernel and status-buffer creation, teardown
static std::mutex s_compute_mutex;  // graph_compute (the status buffer is shared)

// Pipeline cache file: compiled pipelines of earlier runs, so that the first graph does not compile every kernel
// again (Adreno 830: about 110 ms per kernel). AMEVA_PIPELINE_CACHE unset: $HOME/.cache/ameva/ggml-ameva-pipelines.bin;
// =0: no cache file; any other value: that path. Data of another device or driver version is rejected by the runtime
// (header check) and replaced at exit. Saved at teardown when this run added pipelines (the data grew).
static std::string s_pcache_path;
static size_t s_pcache_loaded_bytes = 0;    // cache bytes on disk (loaded or last written)
static uint64_t s_pcache_kcreate_seen = 0;  // kernel creations already accounted for by a save check

static std::string ameva_pipeline_cache_path(void) {
    const char* e = getenv("AMEVA_PIPELINE_CACHE");
    if (e && strcmp(e, "0") == 0) return std::string();
    if (e && *e) return std::string(e);
    const char* home = getenv("HOME");
    if (!home || !*home) {
        fprintf(stderr, "ggml-ameva: pipeline cache off: HOME is not set (set AMEVA_PIPELINE_CACHE=<file>)\n");
        return std::string();
    }
    return std::string(home) + "/.cache/ameva/ggml-ameva-pipelines.bin";
}

// called with s_mutex held, right after the context was created
static void ameva_pipeline_cache_load(AcContext* ctx) {
    s_pcache_path = ameva_pipeline_cache_path();
    if (s_pcache_path.empty()) return;
    FILE* f = fopen(s_pcache_path.c_str(), "rb");
    if (!f) return; // first run: nothing saved yet
    std::vector<uint8_t> data;
    uint8_t chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0 && data.size() < (64u << 20)) data.insert(data.end(), chunk, chunk + n);
    const bool read_error = ferror(f) != 0;
    fclose(f);
    if (read_error || data.empty()) {
        fprintf(stderr, "ggml-ameva: pipeline cache %s could not be read: not used\n", s_pcache_path.c_str());
        return;
    }
    const AcResult r = ac_runtime_pipeline_cache_merge(ctx, data.data(), data.size());
    if (r == AC_SUCCESS) {
        s_pcache_loaded_bytes = data.size();
    } else if (r == AC_ERR_INVALID_ARGUMENT) {
        fprintf(stderr, "ggml-ameva: pipeline cache %s is for another device or driver: not used (rewritten at exit)\n",
                s_pcache_path.c_str());
    } else {
        fprintf(stderr, "ggml-ameva: pipeline cache %s could not be merged (code %d): not used\n", s_pcache_path.c_str(),
                (int)r);
    }
}

// Writes a file of this engine's cache directory: the directory and its parent are created (~/.cache, ~/.cache/ameva;
// an existing directory is fine), the data goes to a per-process temporary file that is renamed into place, so
// concurrent processes each install a complete file. errno describes a failure.
static bool ameva_write_file(const std::string& path, const void* data, size_t size) {
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        const std::string dir = path.substr(0, slash);
        const size_t up = dir.find_last_of('/');
        if (up != std::string::npos && up > 0) mkdir(dir.substr(0, up).c_str(), 0700);
        mkdir(dir.c_str(), 0700);
    }
    const std::string tmp = path + ".tmp." + std::to_string((long long)getpid());
    FILE* f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(data, 1, size, f) == size;
    if (f) ok = (fclose(f) == 0) && ok;
    if (ok) ok = rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) {
        const int err = errno;
        remove(tmp.c_str());
        errno = err;
    }
    return ok;
}

// "<vendor ID>:<device ID>:<pipelineCacheUUID>" in hex from the header of the pipeline cache data (Vulkan spec,
// VkPipelineCacheHeaderVersionOne: header length, header version 1, vendor ID, device ID, 16-byte UUID); the UUID
// changes with the driver build. Empty when the data is unavailable or has another header version.
static std::string ameva_device_key(AcContext* ctx) {
    size_t size = 0;
    if (ac_runtime_pipeline_cache_get_data(ctx, nullptr, &size) != AC_SUCCESS || size < 32) return std::string();
    std::vector<uint8_t> data(size);
    if (ac_runtime_pipeline_cache_get_data(ctx, data.data(), &size) != AC_SUCCESS || size < 32) return std::string();
    uint32_t h[4];
    memcpy(h, data.data(), sizeof(h));
    if (h[0] < 32 || h[1] != 1u) return std::string();
    char key[96];
    int n = snprintf(key, sizeof(key), "%08x:%08x:", h[2], h[3]);
    for (int i = 0; i < 16; ++i) n += snprintf(key + n, sizeof(key) - (size_t)n, "%02x", data[16 + i]);
    return std::string(key);
}

// called with s_mutex held, before the kernels and the context are destroyed
static void ameva_pipeline_cache_save(void) {
    if (!s_ctx || s_pcache_path.empty()) return;
    size_t size = 0;
    if (ac_runtime_pipeline_cache_get_data(s_ctx, nullptr, &size) != AC_SUCCESS || size == 0) return;
    if (size == s_pcache_loaded_bytes) return; // nothing new this run
    std::vector<uint8_t> data(size);
    AcResult r = ac_runtime_pipeline_cache_get_data(s_ctx, data.data(), &size);
    if (r != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: pipeline cache data unavailable (code %d): not saved\n", (int)r);
        return;
    }
    if (!ameva_write_file(s_pcache_path, data.data(), size)) {
        fprintf(stderr, "ggml-ameva: pipeline cache %s could not be written (%s)\n", s_pcache_path.c_str(),
                strerror(errno));
        return;
    }
    s_pcache_loaded_bytes = size;
}

// After a graph completed: when kernels were created since the last check, save the cache if it grew. Library
// destructors are not a reliable save point (an Android process may end without running them: on Galaxy S25 no cache
// file was ever written from teardown), so the cache is saved as soon as new pipelines exist; the teardown save stays.
static void ameva_pipeline_cache_checkpoint(void) {
    const uint64_t n = g_prof_kcreate_n.load();
    if (n == s_pcache_kcreate_seen) return;
    std::lock_guard<std::mutex> lock(s_mutex);
    s_pcache_kcreate_seen = n;
    ameva_pipeline_cache_save();
}

AcContext* ameva_get_ac_context() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_ctx) {
        AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
        AcResult res = ac_runtime_context_create(&cdesc, &s_ctx);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create Vulkan AcContext (code %d)\n", res);
            s_ctx = nullptr;
            return nullptr;
        }
        ameva_increment_context_create_count();
        s_limits.abi_version = AC_ABI_VERSION;
        s_limits.struct_size = sizeof(s_limits);
        res = ac_runtime_context_get_limits(s_ctx, &s_limits);
        if (res != AC_SUCCESS || s_limits.max_storage_buffer_range < 1024 ||
            s_limits.min_storage_buffer_offset_alignment == 0 ||
            (s_limits.min_storage_buffer_offset_alignment & (s_limits.min_storage_buffer_offset_alignment - 1))) {
            fprintf(stderr, "ggml-ameva: device limits unusable (code %d, range %llu, alignment %llu)\n", res,
                    (unsigned long long)s_limits.max_storage_buffer_range,
                    (unsigned long long)s_limits.min_storage_buffer_offset_alignment);
            ac_runtime_context_destroy(s_ctx);
            ameva_increment_context_destroy_count();
            s_ctx = nullptr;
            return nullptr;
        }
        ameva_pipeline_cache_load(s_ctx);
        if (ac_runtime_context_get_device_name(s_ctx, s_device_name, sizeof(s_device_name)) != AC_SUCCESS) {
            snprintf(s_device_name, sizeof(s_device_name), "?");
        }
        s_device_key = ameva_device_key(s_ctx);
    }
    return s_ctx;
}

// Largest byte span one binding may cover, for buffers of any size (the range starts at an aligned offset, so a
// tensor of this size fits from any offset). AMEVA_BINDING_LIMIT=<bytes> (tests only, >= 4096) lowers it so that the
// paths for tensors larger than one binding (weight row slices, row chunks) run with small tensors.
size_t ameva_max_binding_span(void) {
    if (!ameva_get_ac_context()) return 0;
    static const size_t limit = [] {
        const char* e = getenv("AMEVA_BINDING_LIMIT");
        if (!e || !*e) return (size_t)0;
        char* end = nullptr;
        const unsigned long long v = strtoull(e, &end, 10);
        if (!end || *end || v < 4096) GGML_ABORT("ggml-ameva: AMEVA_BINDING_LIMIT must be a byte count >= 4096 (got \"%s\")", e);
        return (size_t)v;
    }();
    size_t span = (size_t)(s_limits.max_storage_buffer_range - s_limits.min_storage_buffer_offset_alignment) & ~(size_t)3;
    if (limit && limit < span) span = limit & ~(size_t)3;
    return span;
}

uint64_t ameva_max_buffer_size(void) {
    if (!ameva_get_ac_context()) return 0;
    uint64_t m = 512ull << 20;
    if (s_limits.max_memory_allocation_size) m = std::min<uint64_t>(m, s_limits.max_memory_allocation_size);
    return m;
}

static const AcRuntimeBindingDesc kBindings[8] = {
    { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 3, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 4, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 5, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 6, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }, { 7, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }};

// One compute pipeline with the workgroup size `local` of its module
static AcKernel* ameva_make_kernel_ex(const char* name, const uint32_t* spirv, size_t bytes, const char* sha,
                                      uint32_t bindings, uint32_t push_bytes, AcDim3 local) {
    AcKernelRuntimeMetadata meta = {
        name, "main", bindings, kBindings, push_bytes, local, 0, sha, 0, nullptr
    };
    AcKernelDesc kdesc = { AC_ABI_VERSION, sizeof(AcKernelDesc), spirv, bytes, &meta, 0, nullptr };
    AcKernel* k = nullptr;
    const uint64_t t0 = ameva_now_ns();
    AcResult res = ac_runtime_kernel_create(s_ctx, &kdesc, &k);
    g_prof_kcreate_ns.fetch_add(ameva_now_ns() - t0);
    g_prof_kcreate_n.fetch_add(1);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: failed to create kernel %s (code %d)\n", name, res);
        return nullptr;
    }
    ameva_increment_kernel_create_count();
    return k;
}

static AcKernel* ameva_make_kernel(const char* name, const uint32_t* spirv, size_t bytes, const char* sha,
                                   uint32_t bindings, uint32_t push_bytes, uint32_t local_x) {
    return ameva_make_kernel_ex(name, spirv, bytes, sha, bindings, push_bytes, { local_x, 1, 1 });
}

static AcKernel* ameva_qmm_kernel(int idx) {
    static char names[AMEVA_QMM_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_qmm[idx] && s_ctx) {
        const AmevaQmmVariant& v = kAmevaQmmVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_qmm_%s_n%u", v.type_name, v.ncols);
        s_qmm[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_QMM_BINDINGS,
                                       AMEVA_QMM_PUSH_CONSTANT_BYTES, AMEVA_QMM_LOCAL_SIZE_X);
    }
    return s_qmm[idx];
}

// ameva_qmv variant idx (weight type, activation rows per workgroup, weight rows per lane group)
static AcKernel* ameva_qmv_kernel(int idx) {
    static char names[AMEVA_QMV_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_qmv[idx] && s_ctx) {
        const AmevaQmvVariant& v = kAmevaQmvVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_qmv_%s_c%u_r%u", v.type_name, v.cols, v.rows);
        s_qmv[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_QMV_BINDINGS,
                                       AMEVA_QMV_PUSH_CONSTANT_BYTES, AMEVA_QMV_LOCAL_SIZE_X);
    }
    return s_qmv[idx];
}

static AcKernel* ameva_mali_qmv_kernel(int idx) {
    static char names[AMEVA_MALI_QMV_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_mali_qmv[idx] && s_ctx) {
        const AmevaMaliQmvVariant& v = kAmevaMaliQmvVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "mali_qmv_%s", v.type_name);
        s_mali_qmv[idx] = ameva_make_kernel_ex(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256,
                                               AMEVA_MALI_QMV_BINDINGS, AMEVA_MALI_QMV_PUSH_CONSTANT_BYTES,
                                               { AMEVA_MALI_QMV_LOCAL_SIZE_X, AMEVA_MALI_QMV_LOCAL_SIZE_Y, 1 });
    }
    return s_mali_qmv[idx];
}

static AcKernel* ameva_qgemm_kernel(int idx) {
    static char names[AMEVA_QGEMM_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_qgemm[idx] && s_ctx) {
        const AmevaQgemmVariant& v = kAmevaQgemmVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_qgemm_%s", v.type_name);
        s_qgemm[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_QGEMM_BINDINGS,
                                         AMEVA_QGEMM_PUSH_CONSTANT_BYTES, AMEVA_QGEMM_LOCAL_SIZE_X);
    }
    return s_qgemm[idx];
}

static AcKernel* ameva_mmq_kernel(int idx) {
    static char names[AMEVA_MMQ_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_mmq[idx] && s_ctx) {
        const AmevaMmqVariant& v = kAmevaMmqVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_mmq_%s", v.type_name);
        s_mmq[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_MMQ_BINDINGS,
                                       AMEVA_MMQ_PUSH_CONSTANT_BYTES, AMEVA_MMQ_LOCAL_SIZE_X);
    }
    return s_mmq[idx];
}

static int ameva_mmq_find(enum ggml_type type) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_MMQ_VARIANT_COUNT; ++i) {
        if (strcmp(kAmevaMmqVariants[i].type_name, tn) == 0) return i;
    }
    return -1;
}

// The integer-dot family of token generation in this process: ameva_mmvq with VK_KHR_shader_integer_dot_product
// where the device offers it, otherwise the same kernel with the dot product emulated by 32-bit multiplications
// (shaders/ameva_mmvq.comp -DEMU_DOT; Mali-G78 r38p0 and Mali-G68 r32p1 have no such extension). The emulation takes
// its activations from ameva_quant_a8 -DEMU (72 scratch bytes per 32-element unit instead of 40) and has no variant
// for q8_0 and the iq4 types, whose products stay on the f32 layouts.
//   AMEVA_MMVQ_EMU=1  the emulation also on a device that has the extension (kernel comparison); 0 or unset: as above
// Measured 2026-10-04, token generation, f32 layouts only -> with the emulation (t/s, three alternating rounds):
//   Mali-G68 r32p1 (Galaxy A53)  Qwen2.5-0.5B 14.4 -> 14.6, Llama-3.2-1B 6.9-7.0 -> 7.8
//   Mali-G78 r38p0 (Galaxy S21)  Qwen 21.8-22.8 -> 22.9-23.1, Llama 8.0-10.2 -> 9.6-11.0
//   Mali-G68 r38p1 (Galaxy A35, which has the extension): f32 13.3-14.8, emulation 16.0-16.4, extension 21.5-21.8
//   (Qwen); 6.8-7.6, 9.0-9.2, 12.2-12.5 (Llama).
// Per shape the emulation is chosen for q4_K (0.68 to 0.85 of the f32 layout's GPU time), is within 10 % for q5_0 and
// slower for q6_K: the decoding of the fifth and sixth bit costs more operations than the dot product saves. It does
// not bring token generation on these devices above the CPU (Galaxy A53 23, Galaxy S21 46 t/s for Qwen).
struct AmevaI8Family {
    bool emulated;
    uint32_t unit_bytes;  // scratch bytes of the quantized activations per 32-element unit
    uint32_t ad_units;    // AD offset in uvec2 elements per unit (the AQ bytes of a unit / 8)
    AmevaOpKernelId quant;
    const AmevaMmvqVariant* variants;
    int count;
};

static const AmevaI8Family& ameva_i8_family(void) {
    static const AmevaI8Family fam = [] {
        const char* e = getenv("AMEVA_MMVQ_EMU");
        if (e && *e && strcmp(e, "0") != 0 && strcmp(e, "1") != 0) {
            GGML_ABORT("ggml-ameva: AMEVA_MMVQ_EMU must be 0 or 1 (got \"%s\")", e);
        }
        const bool emulated = !s_limits.integer_dot_product || (e && strcmp(e, "1") == 0);
        if (emulated) {
            return AmevaI8Family{ true, 72u, 8u, AMEVA_K_QUANT_A8U, kAmevaMmvqeVariants, AMEVA_MMVQE_VARIANT_COUNT };
        }
        return AmevaI8Family{ false, 40u, 4u, AMEVA_K_QUANT_A8, kAmevaMmvqVariants, AMEVA_MMVQ_VARIANT_COUNT };
    }();
    return fam;
}

// Variant of the process's integer-dot family for a weight type and weight rows per lane group, or -1
static int ameva_mmvq_find(enum ggml_type type, uint32_t rows) {
    const AmevaI8Family& fam = ameva_i8_family();
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < fam.count; ++i) {
        if (fam.variants[i].rows == rows && strcmp(fam.variants[i].type_name, tn) == 0) return i;
    }
    return -1;
}

static AcKernel* ameva_mmvq_kernel(int idx) {
    static char names[AMEVA_MMVQ_VARIANT_COUNT + AMEVA_MMVQE_VARIANT_COUNT][44];
    const AmevaI8Family& fam = ameva_i8_family();
    AcKernel** slot = fam.emulated ? &s_mmvqe[idx] : &s_mmvq[idx];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!*slot && s_ctx) {
        const AmevaMmvqVariant& v = fam.variants[idx];
        char* name = names[(fam.emulated ? AMEVA_MMVQ_VARIANT_COUNT : 0) + idx];
        snprintf(name, sizeof(names[0]), "ameva_mmvq%s_%s_r%u", fam.emulated ? "e" : "", v.type_name, v.rows);
        *slot = ameva_make_kernel(name, v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_MMVQ_BINDINGS,
                                  AMEVA_MMVQ_PUSH_CONSTANT_BYTES, AMEVA_MMVQ_LOCAL_SIZE_X);
    }
    return *slot;
}

static AcKernel* ameva_fa_kernel(int idx) {
    static char names[AMEVA_FA_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_fa[idx] && s_ctx) {
        const AmevaFaVariant& v = kAmevaFaVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_flash_attn_d%u", v.dmax);
        s_fa[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_FA_BINDINGS,
                                      AMEVA_FA_PUSH_CONSTANT_BYTES, AMEVA_FA_LOCAL_SIZE_X);
    }
    return s_fa[idx];
}

// smallest flash-attention variant that takes head sizes D and DV
static int ameva_fa_find(uint32_t d, uint32_t dv) {
    for (int i = 0; i < AMEVA_FA_VARIANT_COUNT; ++i) {
        if (d <= kAmevaFaVariants[i].dmax && dv <= kAmevaFaVariants[i].dmax) return i;
    }
    return -1;
}

static AcKernel* ameva_op_kernel(AmevaOpKernelId id) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_ops[id] && s_ctx) {
        const AmevaOpKernel& k = kAmevaOpKernels[id];
        s_ops[id] = ameva_make_kernel(k.name, k.spirv, k.spirv_bytes, k.spirv_sha256, k.bindings,
                                      k.push_constant_bytes, k.local_size_x);
    }
    return s_ops[id];
}

static bool ameva_status_buffer(void) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_status) return true;
    if (!s_ctx) return false;
    AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), 256, AC_BUFFER_USAGE_STORAGE, 0 };
    if (ac_runtime_buffer_create(s_ctx, &bdesc, &s_status) != AC_SUCCESS) {
        s_status = nullptr;
        fprintf(stderr, "ggml-ameva: status buffer allocation failed\n");
        return false;
    }
    ameva_increment_buffer_create_count();
    void* map = nullptr;
    if (ac_runtime_buffer_get_mapped_ptr(s_status, &map) != AC_SUCCESS || !map) {
        ac_runtime_buffer_destroy(s_status);
        ameva_increment_buffer_destroy_count();
        s_status = nullptr;
        fprintf(stderr, "ggml-ameva: status buffer is not host-mapped\n");
        return false;
    }
    s_status_map = (uint32_t*)map;
    return true;
}

// Makes the scratch buffer at least `bytes` large. Called before recording, when no submission of this engine is
// pending (graph_compute waits for its batch), so an outgrown buffer can be released at once.
static bool ameva_scratch_reserve(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (bytes <= s_scratch_bytes) return true;
    if (!s_ctx) return false;
    if (s_scratch) {
        const AcResult r = ac_runtime_buffer_destroy(s_scratch);
        if (r != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: scratch buffer destroy failed (code %d)\n", (int)r);
            return false;
        }
        ameva_increment_buffer_destroy_count();
        s_scratch = nullptr;
        s_scratch_bytes = 0;
    }
    const uint64_t size = std::max<uint64_t>(bytes, 1u << 20); // at least 1 MiB: token generation reuses it
    AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)size, AC_BUFFER_USAGE_STORAGE, 0 };
    if (ac_runtime_buffer_create(s_ctx, &bdesc, &s_scratch) != AC_SUCCESS) {
        s_scratch = nullptr;
        fprintf(stderr, "ggml-ameva: scratch buffer of %llu bytes could not be allocated\n", (unsigned long long)size);
        return false;
    }
    ameva_increment_buffer_create_count();
    s_scratch_bytes = size;
    return true;
}

void ggml_backend_ameva_teardown(void) {
    std::lock_guard<std::mutex> clock(s_compute_mutex);
    std::lock_guard<std::mutex> lock(s_mutex);
    ameva_pipeline_cache_save();
    if (s_scratch) {
        AcResult r = ac_runtime_buffer_destroy(s_scratch);
        if (r != AC_SUCCESS) fprintf(stderr, "ggml-ameva: scratch buffer destroy failed (code %d)\n", r);
        ameva_increment_buffer_destroy_count();
        s_scratch = nullptr;
        s_scratch_bytes = 0;
    }
    if (s_status) {
        AcResult r = ac_runtime_buffer_destroy(s_status);
        if (r != AC_SUCCESS) fprintf(stderr, "ggml-ameva: status buffer destroy failed (code %d)\n", r);
        ameva_increment_buffer_destroy_count();
        s_status = nullptr;
        s_status_map = nullptr;
    }
    for (AcKernel*& k : s_qmm) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_qmv) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_mali_qmv) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_qgemm) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_mmq) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_mmvq) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_mmvqe) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_fa) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    for (AcKernel*& k : s_ops) {
        if (k) {
            ac_runtime_kernel_destroy(k);
            k = nullptr;
            ameva_increment_kernel_destroy_count();
        }
    }
    if (s_ctx) {
        AcResult r = ac_runtime_context_destroy(s_ctx);
        if (r != AC_SUCCESS) {
            // buffers still owned by ggml (model weights, KV cache) keep the context alive; it stays allocated
            fprintf(stderr, "ggml-ameva: context not destroyed at teardown (code %d)\n", r);
        } else {
            s_ctx = nullptr;
            ameva_increment_context_destroy_count();
        }
    }
}

__attribute__((destructor)) static void ameva_plugin_fini(void) {
    ggml_backend_ameva_teardown();
}

// =============================================================================
// Op contract (shared with supports_op in ggml-ameva-device.cpp)
// =============================================================================
static int ameva_qmm_find(enum ggml_type type, uint32_t ncols) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_QMM_VARIANT_COUNT; ++i) {
        if (kAmevaQmmVariants[i].ncols == ncols && strcmp(kAmevaQmmVariants[i].type_name, tn) == 0) return i;
    }
    return -1;
}

static bool is_f32(const struct ggml_tensor* t) { return t && t->type == GGML_TYPE_F32; }
static bool is_f16(const struct ggml_tensor* t) { return t && t->type == GGML_TYPE_F16; }
// quantized KV-cache types the engine writes (SET_ROWS, CPY) and reads (FLASH_ATTN_EXT, CPY), in ggml layout
static bool is_kvq(const struct ggml_tensor* t) {
    return t && (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0);
}
// element type code of K / V in ameva_flash_attn (kv_types) and of the source in ameva_cpy (src_type):
// 0 f32, 1 f16, 2 q8_0, 3 q4_0, -1 none of these
static int kv_type_code(const struct ggml_tensor* t) {
    if (!t) return -1;
    switch (t->type) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_Q8_0: return 2;
        case GGML_TYPE_Q4_0: return 3;
        default: return -1;
    }
}
static bool rows_contiguous(const struct ggml_tensor* t) { return t && t->nb[0] == ggml_type_size(t->type); }
static bool fits_u32(int64_t v) { return v >= 0 && v <= (int64_t)UINT32_MAX; }

// strides and element counts are passed to the kernels as 32-bit values
static bool fits_kernel_indexing(const struct ggml_tensor* t) {
    if (!t) return true;
    const size_t es = ggml_type_size(t->type);
    if (ggml_nelements(t) > (int64_t)UINT32_MAX) return false;
    for (int i = 0; i < 4; ++i) {
        if (!fits_u32(t->ne[i]) || t->nb[i] % es != 0 || !fits_u32((int64_t)(t->nb[i] / es))) return false;
    }
    return true;
}

static bool fits_binding(const struct ggml_tensor* t) {
    const size_t span = ameva_max_binding_span();
    return t && span && ggml_nbytes(t) <= span;
}

// Rows (dimensions 1..3 flattened) at one stride, so that any range of rows is a 2-D view: such a tensor can be
// processed in row chunks that each fit one binding, provided a single row does.
static bool rows_linear(const struct ggml_tensor* t) {
    return t && t->nb[2] == t->ne[1] * t->nb[1] && t->nb[3] == t->ne[2] * t->nb[2];
}

// fits one binding whole, or in row chunks
static bool fits_binding_rows(const struct ggml_tensor* t) {
    const size_t span = ameva_max_binding_span();
    if (fits_binding(t)) return true;
    return t && span && rows_linear(t) && (size_t)t->nb[1] <= span && ggml_row_size(t->type, t->ne[0]) <= span;
}

static int unary_code(const struct ggml_tensor* op) {
    const char* n = ggml_unary_op_name(ggml_get_unary_op(op));
    if (!n) return -1;
    static const char* const names[] = {"TANH", "SILU", "GELU", "GELU_ERF", "GELU_QUICK", "RELU", "SIGMOID",
                                        "NEG", "SQR", "SQRT", "EXP", "ABS"};
    static const int codes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i) {
        if (strcmp(n, names[i]) == 0) return codes[i];
    }
    return -1;
}

static int glu_code(const struct ggml_tensor* op) {
    const char* n = ggml_glu_op_name(ggml_get_glu_op(op));
    if (!n) return -1;
    if (!strcmp(n, "REGLU")) return 0;
    if (!strcmp(n, "GEGLU")) return 1;
    if (!strcmp(n, "SWIGLU")) return 2;
    if (!strcmp(n, "GEGLU_ERF")) return 4;
    if (!strcmp(n, "GEGLU_QUICK")) return 5;
    return -1;
}

// ameva_qmv variant of a weight type: `cols` activation rows per workgroup (1 or 8), `rows` weight rows per lane group
// (cols 1: 4, 2 or 1; cols 8: 1)
static int ameva_qmv_find(enum ggml_type type, uint32_t cols, uint32_t rows) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_QMV_VARIANT_COUNT; ++i) {
        const AmevaQmvVariant& v = kAmevaQmvVariants[i];
        if (v.cols == cols && v.rows == rows && strcmp(v.type_name, tn) == 0) return i;
    }
    return -1;
}

static int ameva_mali_qmv_find(enum ggml_type type) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_MALI_QMV_VARIANT_COUNT; ++i) {
        if (strcmp(kAmevaMaliQmvVariants[i].type_name, tn) == 0) return i;
    }
    return -1;
}

static int ameva_qgemm_find(enum ggml_type type) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_QGEMM_VARIANT_COUNT; ++i) {
        if (strcmp(kAmevaQgemmVariants[i].type_name, tn) == 0) return i;
    }
    return -1;
}

// Weights of this type may be stored repacked: every kernel that reads repacked weights exists for it.
bool ameva_qmv_has_type(enum ggml_type type) {
    return ameva_qmv_find(type, 1, 4) >= 0 && ameva_qmv_find(type, 8, 1) >= 0 && ameva_qgemm_find(type) >= 0;
}

// MUL_MAT: weights in the repacked layout run on ameva_qmv, all others on ameva_qmm. Both kernels take the same
// shapes, so a weight that is repacked after supports_op accepted its MUL_MAT (the layout is decided by the first
// write) is still accepted by graph_compute.
static const char* mul_mat_reason(const struct ggml_tensor* op) {
    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];
    if (!src0 || !src1) return "missing source";
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return "src1/dst not F32";
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (const struct ggml_tensor* base = ameva_repacked_base(src0)) {
        if (!ameva_qmv_has_type(src0->type)) return "repacked weight type has no ameva_qmv kernel";
        if (src0->type != base->type || K != base->ne[0]) return "view of a repacked weight with another type or row length";
        const uint64_t rb = ggml_row_size(base->type, base->ne[0]);
        const uint64_t voff = (uint64_t)((const char*)src0->data - (const char*)base->data);
        if (voff % rb != 0 || voff / rb + (uint64_t)ggml_nrows(src0) > (uint64_t)ggml_nrows(base))
            return "view of a repacked weight that is not a range of its rows";
        if (K % ggml_blck_size(src0->type) != 0) return "K not a multiple of the block size";
    } else {
        const int vi = ameva_qmm_find(src0->type, 1);
        if (vi < 0) return "src0 type has no ameva_qmm kernel";
        const AmevaQmmVariant& v = kAmevaQmmVariants[vi];
        if (K % 32 != 0 || K % (int64_t)v.block_elems != 0) return "K not a multiple of the block size";
        if (ggml_row_size(src0->type, K) != (size_t)(K / v.block_elems) * v.block_bytes) return "block layout mismatch";
    }
    if (!ggml_is_contiguous(src0)) return "weights not contiguous";
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return "batched (ne2/ne3 > 1)";
    if (src1->ne[0] != K || op->ne[0] != N || op->ne[1] != M) return "shape mismatch";
    if (!rows_contiguous(src1) || !rows_contiguous(op)) return "activation or output rows not contiguous";
    if (src1->nb[1] % 16 != 0) return "activation row stride not a multiple of 16 bytes";
    const size_t span = ameva_max_binding_span();
    if (!span || ggml_row_size(src0->type, K) > span) return "one weight row exceeds a buffer binding";
    // activations and output are bound per chunk of rows (a 512-row probe of a 152k-vocabulary output projection is
    // 311 MB, above Mali's 256 MiB binding range); one row of each must fit
    if ((size_t)src1->nb[1] > span || (size_t)op->nb[1] > span) return "one activation or output row exceeds a buffer binding";
    if ((M + 7) / 8 > 65535) return "grid exceeds 65535 workgroups";
    if (!fits_kernel_indexing(src1) || !fits_kernel_indexing(op) || (uint64_t)M * (uint64_t)K > UINT32_MAX / 2 ||
        (uint64_t)M * (uint64_t)N > UINT32_MAX / 2) return "exceeds 32-bit addressing";
    return nullptr;
}

// metadata-only ops: no work on any backend
static bool ameva_op_is_empty(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

// Returns NULL when the AMEVA graph engine can compute op, otherwise the reason it cannot.
const char* ameva_op_unsupported_reason(const struct ggml_tensor* op) {
    if (!op) return "null op";
    if (!ameva_op_is_empty(op->op)) {
        // Repacked weights are read by ameva_qmv as MUL_MAT src0 and by nothing else: no other kernel knows the
        // layout and no op may write it. Another backend gets them through the buffer interface, which converts.
        if (ameva_repacked_base(op)) return "writes a repacked weight";
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (op->src[i] && !(op->op == GGML_OP_MUL_MAT && i == 0) && ameva_repacked_base(op->src[i]))
                return "reads a repacked weight outside MUL_MAT";
        }
        // With AMEVA_REPACK_ALL=1 a quantized tensor becomes repacked at its first write, which usually comes after
        // this question (test-backend-ops asks, then allocates and fills): decline up front every op that would write
        // such a tensor or read it other than as MUL_MAT weights (the quantized KV-cache ops), instead of failing at
        // graph_compute.
        if (ameva_repack_all_enabled()) {
            if (ggml_is_quantized(op->type) && ameva_qmv_has_type(op->type))
                return "AMEVA_REPACK_ALL stores this quantized result repacked";
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                const struct ggml_tensor* s = op->src[i];
                if (s && !(op->op == GGML_OP_MUL_MAT && i == 0) && ggml_is_quantized(s->type) &&
                    ameva_qmv_has_type(s->type))
                    return "AMEVA_REPACK_ALL stores this quantized input repacked";
            }
        }
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return nullptr;
        case GGML_OP_MUL_MAT:
            return mul_mat_reason(op);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            const struct ggml_tensor *a = op->src[0], *b = op->src[1];
            if (!is_f32(a) || !is_f32(b) || !is_f32(op)) return "binary op on non-f32 tensors";
            if (!ggml_are_same_shape(a, op) || !ggml_can_repeat(b, a)) return "unsupported broadcast";
            if (!rows_contiguous(a) || !rows_contiguous(b) || !rows_contiguous(op)) return "rows not contiguous";
            if (!fits_binding(a) || !fits_binding(b) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(b) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_SCALE:
        case GGML_OP_UNARY: {
            const struct ggml_tensor* a = op->src[0];
            if (op->op == GGML_OP_UNARY && unary_code(op) < 0) return "unary op has no ameva kernel";
            if (!is_f32(a) || !is_f32(op) || !ggml_are_same_shape(a, op)) return "unary op on non-f32 or reshaping";
            if (!rows_contiguous(a) || !rows_contiguous(op)) return "rows not contiguous";
            // larger than one binding: processed in row chunks (e.g. the logits of a 256k vocabulary and a 256-token
            // batch on a 128 MiB binding range)
            if ((!fits_binding(a) || !fits_binding(op)) &&
                (!fits_binding_rows(a) || !fits_binding_rows(op) || !rows_linear(a) || !rows_linear(op)))
                return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_GLU: {
            const struct ggml_tensor *a = op->src[0], *b = op->src[1];
            if (glu_code(op) < 0) return "glu op has no ameva kernel";
            if (!is_f32(a) || !is_f32(op) || (b && !is_f32(b))) return "glu on non-f32 tensors";
            if (b ? !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op) : a->ne[0] != 2 * op->ne[0])
                return "glu shape mismatch";
            if (!rows_contiguous(a) || (b && !rows_contiguous(b)) || !rows_contiguous(op)) return "rows not contiguous";
            if (!fits_binding(a) || (b && !fits_binding(b)) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(b) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_RMS_NORM: {
            const struct ggml_tensor* a = op->src[0];
            if (!is_f32(a) || !is_f32(op) || !ggml_are_same_shape(a, op)) return "rms_norm on non-f32";
            if (!rows_contiguous(a) || !rows_contiguous(op)) return "rows not contiguous";
            if (!fits_binding(a) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_ROPE: {
            const struct ggml_tensor *a = op->src[0], *pos = op->src[1], *ff = op->src[2];
            const int32_t* p = (const int32_t*)op->op_params;
            const int n_dims = p[1], mode = p[2];
            if (mode != 0 && mode != 2) return "rope mode other than normal/neox";
            // f32, or f16 to f16: the KV-cache shift of llama.cpp rotates the stored keys of the f16 cache in place
            // (ggml_rope_ext_inplace on a view of cache_k). Declining it does not move the node to the CPU: the
            // tensor is pre-allocated in this backend's buffer and the scheduler aborts.
            if (!((is_f32(a) && is_f32(op)) || (is_f16(a) && is_f16(op))) || !ggml_are_same_shape(a, op))
                return "rope on other than f32 or f16";
            if (!pos || pos->type != GGML_TYPE_I32 || pos->ne[0] != a->ne[2] || !ggml_is_contiguous(pos))
                return "rope positions not one i32 per row";
            // n_dims == 0 rotates nothing (every pair is copied). llama.cpp asks exactly that when it decides where
            // to place rope_freqs: declining it put the frequency factors, and with them every ROPE, on the CPU.
            if (n_dims < 0 || n_dims % 2 || n_dims > a->ne[0] || a->ne[0] % 2) return "rope dimensions";
            if (ff && (!is_f32(ff) || !ggml_is_contiguous(ff) || ff->ne[0] < n_dims / 2)) return "rope freq factors";
            if (!rows_contiguous(a) || !rows_contiguous(op)) return "rows not contiguous";
            if (!fits_binding(a) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_SET_ROWS: {
            const struct ggml_tensor *a = op->src[0], *idx = op->src[1];
            if (!is_f32(a) || !(is_f16(op) || is_f32(op) || is_kvq(op))) return "set_rows type pair";
            if (is_kvq(op) && op->ne[0] % 32) return "set_rows quantized rows not whole blocks";
            if (!idx || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) || !rows_contiguous(idx))
                return "set_rows index type";
            if (op->ne[0] != a->ne[0] || !rows_contiguous(a) || !rows_contiguous(op)) return "set_rows rows";
            if (idx->ne[3] != 1 || idx->nb[1] % 4 || idx->nb[2] % 4) return "set_rows index layout";
            if (!fits_binding(a) || !fits_binding(op) || !fits_binding(idx)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor *a = op->src[0], *idx = op->src[1];
            if (!(is_f32(a) || is_f16(a)) || !is_f32(op)) return "get_rows type pair";
            if (!idx || idx->type != GGML_TYPE_I32) return "get_rows index type";
            if (!rows_contiguous(a) || !rows_contiguous(op) || !rows_contiguous(idx)) return "rows not contiguous";
            if (!fits_binding(a) || !fits_binding(op) || !fits_binding(idx)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op) || !fits_kernel_indexing(idx)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP: {
            const struct ggml_tensor* a = op->src[0];
            if (is_kvq(op)) {
                // f32 -> q8_0 / q4_0 of the same shape (the KV-cache shift writes the rotated K back)
                if (!is_f32(a) || !ggml_are_same_shape(a, op)) return "copy to a quantized type from other than f32 of the same shape";
                if (!rows_contiguous(a) || !rows_contiguous(op) || op->ne[0] % 32) return "copy to a quantized type: rows not whole blocks";
            } else if (is_kvq(a)) {
                // q8_0 / q4_0 -> f32 / f16 (the KV-cache shift casts K to f32)
                if (!(is_f32(op) || is_f16(op))) return "copy from a quantized type to other than f32/f16";
                if (!rows_contiguous(a) || a->ne[0] % 32) return "copy from a quantized type: rows not whole blocks";
            } else if (!(is_f32(a) || is_f16(a)) || !(is_f32(op) || is_f16(op))) {
                return "copy between types other than f32/f16/q8_0/q4_0";
            }
            if (ggml_nelements(a) != ggml_nelements(op)) return "copy element counts differ";
            if (!fits_binding(a) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor *q = op->src[0], *k = op->src[1], *v = op->src[2], *m = op->src[3];
            if (op->src[4]) return "attention sinks";
            if (!is_f32(q) || !is_f32(op) || kv_type_code(k) < 0 || kv_type_code(v) < 0) return "flash_attn types";
            if ((is_kvq(k) && q->ne[0] % 32) || (is_kvq(v) && v->ne[0] % 32))
                return "flash_attn quantized K/V head size not a multiple of 32";
            if (m && !is_f16(m)) return "flash_attn mask not f16";
            if (q->ne[0] != k->ne[0] || ameva_fa_find((uint32_t)q->ne[0], (uint32_t)v->ne[0]) < 0)
                return "head size above 256";
            if (q->ne[0] % 4 || v->ne[0] % 4) return "head size not a multiple of 4";
            if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || (m && m->ne[3] != 1)) return "multiple sequences";
            if (k->ne[2] == 0 || q->ne[2] % k->ne[2] || k->ne[2] != v->ne[2] || k->ne[1] != v->ne[1])
                return "flash_attn head grouping";
            if (m && (m->ne[0] < k->ne[1] || m->ne[1] < q->ne[1])) return "mask shape";
            if (!rows_contiguous(q) || !rows_contiguous(k) || !rows_contiguous(v) || (m && !rows_contiguous(m)) ||
                !ggml_is_contiguous(op)) return "rows not contiguous";
            if (op->ne[0] != v->ne[0] || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1]) return "flash_attn output shape";
            if (!fits_binding(q) || !fits_binding(k) || !fits_binding(v) || (m && !fits_binding(m)) || !fits_binding(op))
                return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(q) || !fits_kernel_indexing(k) || !fits_kernel_indexing(v) ||
                !fits_kernel_indexing(m) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            if ((uint64_t)q->ne[1] * (uint64_t)q->ne[2] > 65535ull * 65535ull) return "grid";
            return nullptr;
        }
        default:
            return "op has no ameva kernel";
    }
}

// =============================================================================
// Bindings and hazards
// =============================================================================
static bool ameva_is_ameva_buffer(const struct ggml_tensor* t) {
    return t && t->buffer && strcmp(ggml_backend_buffer_name(t->buffer), GGML_AMEVA_NAME) == 0;
}

struct AmevaBind {
    AcBufferBinding b;  // buffer range given to the descriptor
    uint32_t elem;      // tensor start inside the range, in elements of the requested size
    const AcBuffer* key;
    uint64_t lo, hi;    // byte span of the tensor inside its buffer (hazard tracking)
};

// Binds bytes [byte_off, byte_off + span) of tensor t's buffer, counting the start in units of esize.
static const char* ameva_bind_span(const struct ggml_tensor* t, uint64_t extra_off, uint64_t span, uint32_t esize,
                                   AmevaBind* out) {
    if (!ameva_is_ameva_buffer(t)) return "tensor is not in an AMEVA buffer";
    auto* bctx = (ggml_backend_ameva_buffer_context*)t->buffer->context;
    if (!bctx || !bctx->ac_buf) return "AMEVA buffer without context";
    const uint64_t off = (uint64_t)((const char*)t->data - (const char*)bctx->mapped_ptr) + extra_off;
    const uint64_t align = s_limits.min_storage_buffer_offset_alignment;
    const uint64_t desc = off & ~(align - 1);
    const uint64_t resid = off - desc;
    if (resid % esize) return "tensor offset not aligned to its element size";
    uint64_t range = std::min<uint64_t>((uint64_t)bctx->size - desc, s_limits.max_storage_buffer_range) & ~3ull;
    if (resid + span > range) return "tensor exceeds one buffer binding";
    out->b.buffer = bctx->ac_buf;
    out->b.offset = desc;
    out->b.range = range;
    out->elem = (uint32_t)(resid / esize);
    out->key = bctx->ac_buf;
    out->lo = off;
    out->hi = off + span;
    return nullptr;
}

static const char* ameva_bind(const struct ggml_tensor* t, uint32_t esize, AmevaBind* out) {
    return ameva_bind_span(t, 0, ggml_nbytes(t), esize, out);
}

struct AmevaRange {
    const AcBuffer* buf;
    uint64_t lo, hi;
};

// Submission sizing. Mobile drivers reset the GPU when one submission runs for seconds (S25: a 256-token Llama-1B
// prompt as one 4 s submission never completed), so the batch is flushed to the queue once the recorded work reaches
// kFlushMac multiply-accumulates or kFlushDispatches dispatches; one dispatch never exceeds kDispatchMac (large
// MUL_MATs and attention are split). Flushing early also lets the GPU start while the rest is recorded.
static const double kFlushMac = 4e9;
static const double kDispatchMac = 4e9;
static const uint32_t kFlushDispatches = 128;

struct AmevaRecorder {
    AcCommandBatch* batch = nullptr;
    std::vector<AmevaRange> written, read;
    uint32_t dispatches = 0;
    uint32_t barriers = 0;
    uint32_t flushes = 0;
    double mac_since_flush = 0;
    uint32_t dispatches_since_flush = 0;
    uint64_t flush_ns = 0;        // time spent in intermediate submissions
    uint64_t rt_barrier_ns = 0;   // time inside ac_runtime_batch_barrier
    uint64_t rt_dispatch_ns = 0;  // time inside ac_runtime_batch_dispatch_bindings
    int node = 0;
    int64_t mul_mat_nodes = 0;    // counted once the graph completed
    int64_t qmv_nodes = 0;
    int64_t mmq_nodes = 0;
    int64_t mmvq_nodes = 0;
    int64_t fused_pairs = 0;      // node pairs computed by one dispatch sequence (ameva_find_fusion)
    // The activation row whose int8 form (ameva_quant_a8) is in the scratch buffer, for the ameva_mmvq products that
    // follow with the same src1 (Q, K, V share theirs, gate and up too). Cleared by a write to the scratch buffer or
    // to the row itself (ameva_record).
    const struct ggml_tensor* a8_src = nullptr;
    AmevaRange a8_range = { nullptr, 0, 0 };
    bool profile_kinds = false;   // AMEVA_PROFILE=2: remember the op kind of every dispatch
    uint16_t kind = 0;
    std::vector<uint16_t> kinds;
};

static bool range_hits(const std::vector<AmevaRange>& v, const AmevaBind& x) {
    for (const AmevaRange& r : v) {
        if (r.buf == x.key && r.lo < x.hi && x.lo < r.hi) return true;
    }
    return false;
}

// Barrier when this dispatch reads what an earlier one of the interval wrote (RAW), or writes what an earlier one
// read or wrote (WAR, WAW); then the dispatch joins the interval. mac: estimated multiply-accumulates of the dispatch
// (submission sizing only).
static const char* ameva_record(AmevaRecorder& rec, AcKernel* kernel, const AmevaBind* reads, int nreads,
                                const AmevaBind* writes, int nwrites, const AcBufferBinding* bindings, uint32_t nbind,
                                const void* pc, size_t pc_size, AcDim3 grid, double mac) {
    bool hazard = false;
    for (int i = 0; i < nreads && !hazard; ++i) hazard = range_hits(rec.written, reads[i]);
    for (int i = 0; i < nwrites && !hazard; ++i) hazard = range_hits(rec.written, writes[i]) || range_hits(rec.read, writes[i]);
    if (hazard) {
        const uint64_t tb = ameva_now_ns();
        const AcResult br = ac_runtime_batch_barrier(rec.batch);
        rec.rt_barrier_ns += ameva_now_ns() - tb;
        if (br != AC_SUCCESS) return "barrier recording failed";
        rec.written.clear();
        rec.read.clear();
        rec.barriers++;
    }
    const uint64_t td = ameva_now_ns();
    AcResult r = ac_runtime_batch_dispatch_bindings(rec.batch, kernel, bindings, nbind, pc, pc_size, grid);
    rec.rt_dispatch_ns += ameva_now_ns() - td;
    if (r != AC_SUCCESS) {
        static char msg[96];
        snprintf(msg, sizeof(msg), "dispatch recording failed (code %d)", (int)r);
        return msg;
    }
    for (int i = 0; i < nreads; ++i) rec.read.push_back({reads[i].key, reads[i].lo, reads[i].hi});
    for (int i = 0; i < nwrites; ++i) rec.written.push_back({writes[i].key, writes[i].lo, writes[i].hi});
    for (int i = 0; i < nwrites && rec.a8_src; ++i) {
        if (writes[i].key == s_scratch ||
            (writes[i].key == rec.a8_range.buf && writes[i].lo < rec.a8_range.hi && rec.a8_range.lo < writes[i].hi)) {
            rec.a8_src = nullptr;
        }
    }
    rec.dispatches++;
    if (rec.profile_kinds) rec.kinds.push_back(rec.kind);
    rec.mac_since_flush += mac;
    if (++rec.dispatches_since_flush >= kFlushDispatches || rec.mac_since_flush >= kFlushMac) {
        // the hazard state stays: a barrier recorded later also covers the flushed commands
        const uint64_t tf = ameva_now_ns();
        AcResult fr = ac_runtime_batch_flush(rec.batch);
        rec.flush_ns += ameva_now_ns() - tf;
        if (fr != AC_SUCCESS) {
            static char msg[96];
            snprintf(msg, sizeof(msg), "intermediate submission failed (code %d)", (int)fr);
            return msg;
        }
        rec.flushes++;
        rec.mac_since_flush = 0;
        rec.dispatches_since_flush = 0;
    }
    return nullptr;
}

// 1-D work of n invocations in workgroups of `local` spread over (x, y) <= 65535
static AcDim3 ameva_grid_1d(uint64_t n, uint32_t local) {
    uint64_t groups = (n + local - 1) / local;
    if (groups == 0) groups = 1;
    const uint32_t gx = (uint32_t)std::min<uint64_t>(groups, 65535);
    const uint32_t gy = (uint32_t)((groups + gx - 1) / gx);
    return { gx, gy, 1 };
}

static AcDim3 ameva_grid_groups(uint64_t groups) {
    if (groups == 0) groups = 1;
    const uint32_t gx = (uint32_t)std::min<uint64_t>(groups, 65535);
    const uint32_t gy = (uint32_t)((groups + gx - 1) / gx);
    return { gx, gy, 1 };
}

static inline uint32_t st(const struct ggml_tensor* t, int d) { return (uint32_t)(t->nb[d] / ggml_type_size(t->type)); }
static inline uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static inline float op_param_f32(const struct ggml_tensor* t, int i) {
    float f;
    memcpy(&f, (const int32_t*)t->op_params + i, sizeof(float));
    return f;
}

// =============================================================================
// Encoders: each records the dispatches of one node
// =============================================================================
// Activation rows from which a MUL_MAT on repacked weights runs as ameva_qgemm (64 x 32 tiles) instead of
// ameva_qmv(COLS = 8). Mali-G68, Qwen2.5-0.5B FFN shapes (scripts/termux/lab/qgemm_crossover.sh): q5_0 is faster
// with ameva_qgemm from M = 8, q4_K 896x4864 from M = 12. AMEVA_QGEMM_MIN_M overrides it (kernel tuning).
static uint32_t ameva_qgemm_min_rows(void) {
    static const uint32_t v = [] {
        const char* e = getenv("AMEVA_QGEMM_MIN_M");
        if (!e || !*e) return 8u;
        char* end = nullptr;
        const unsigned long x = strtoul(e, &end, 10);
        if (!end || *end || x < 2 || x > 65535) GGML_ABORT("ggml-ameva: AMEVA_QGEMM_MIN_M must be 2..65535 (got \"%s\")", e);
        return (uint32_t)x;
    }();
    return v;
}

// Epilogue of a token-generation MUL_MAT (ameva_qmv, one activation row) that also computes the node consuming the
// product (graph-engine fusion): mode 1 ADD (out = v + e), 2 SWIGLU with the product as gate (out = silu(v) * e),
// 3 SWIGLU with the product as up projection (out = silu(e) * v). e and out are f32 vectors of N elements.
struct AmevaEpilogue {
    uint32_t mode;
    const struct ggml_tensor* e;
    const struct ggml_tensor* out;
};

// MMQ (int8 activations x repacked weights with packed integer dot products) for the MUL_MATs that would run
// ameva_qgemm. AMEVA_MMQ unset: on when the device enabled VK_KHR_shader_integer_dot_product, otherwise off with one
// line on stderr; AMEVA_MMQ=1: required (abort when the device lacks it); AMEVA_MMQ=0: off (f32 ameva_qgemm).
// Accuracy: the activations are rounded to int8 per 32 elements (the CPU backend rounds them to q8_0 / q8_K as well).
static bool ameva_mmq_enabled(void) {
    static int mode = -1;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* e = getenv("AMEVA_MMQ");
        if (e && *e && strcmp(e, "0") != 0 && strcmp(e, "1") != 0) {
            GGML_ABORT("ggml-ameva: AMEVA_MMQ must be 0 or 1 (got \"%s\")", e);
        }
        const bool supported = s_limits.integer_dot_product != 0;
        if (e && strcmp(e, "0") == 0) {
            mode = 0;
            fprintf(stderr, "ggml-ameva: MMQ off (AMEVA_MMQ=0): prompt MUL_MAT runs the f32 tiled kernel\n");
        } else if (supported) {
            mode = 1;
            fprintf(stderr, "ggml-ameva: MMQ on (VK_KHR_shader_integer_dot_product, 4x8 packed accelerated: %s)\n",
                    s_limits.integer_dot_product_accelerated ? "yes" : "no");
        } else if (e && strcmp(e, "1") == 0) {
            GGML_ABORT("ggml-ameva: AMEVA_MMQ=1 but the device does not offer VK_KHR_shader_integer_dot_product");
        } else {
            mode = 0;
            fprintf(stderr, "ggml-ameva: MMQ unavailable (no VK_KHR_shader_integer_dot_product): prompt MUL_MAT runs "
                            "the f32 tiled kernel\n");
        }
    });
    return mode == 1;
}

// Scratch bytes per activation unit of 32: 32 int8 + d and two half sums (shaders/ameva_quant_a8.comp)
static const uint64_t kMmqUnitBytes = 40;

// Quality check of block-wise int8 activations (docs/design/opencl-prompt-path.md, stage A): AMEVA_ACT_FQ rounds the
// activations of quantized-weight MUL_MATs to int8 per block and back (shaders/ameva_act_fq.comp) before the f32
// kernels read them, so what a kernel with such activations computes can be measured on any device:
//   256  weights of the 256-element super-block types (q2_K .. q6_K, iq4_xs ...): one scale per 256 activations,
//        ggml's Q8_K, which the CPU backend multiplies these types with; other weight types unchanged
//   32   every quantized weight: one scale per 32 activations (Q8_0)
//   cpu  the CPU backend's choice: 256 for the super-block types, 32 for the others
// The int8 kernels quantize the activations themselves, so the mode needs AMEVA_MMQ=0 and AMEVA_MMVQ=0.
static uint32_t ameva_act_fq_mode(void) { // 0 off, 256, 32, 1 = cpu
    static int mode = -1;
    if (mode < 0) {
        const char* e = getenv("AMEVA_ACT_FQ");
        if (!e || !*e || !strcmp(e, "0")) mode = 0;
        else if (!strcmp(e, "256")) mode = 256;
        else if (!strcmp(e, "32")) mode = 32;
        else if (!strcmp(e, "cpu")) mode = 1;
        else GGML_ABORT("ggml-ameva: AMEVA_ACT_FQ must be 0, 256, 32 or cpu (got \"%s\")", e);
        if (mode) {
            const char* q = getenv("AMEVA_MMQ");
            const char* v = getenv("AMEVA_MMVQ");
            if (!q || strcmp(q, "0") || !v || strcmp(v, "0"))
                GGML_ABORT("ggml-ameva: AMEVA_ACT_FQ=%s needs AMEVA_MMQ=0 and AMEVA_MMVQ=0 (the int8 kernels quantize "
                           "the activations themselves)", e);
            fprintf(stderr, "ggml-ameva: AMEVA_ACT_FQ=%s: activations of quantized-weight MUL_MATs rounded to int8 per "
                            "block before the f32 kernels\n", e);
        }
    }
    return (uint32_t)mode;
}

// Block size of the activation rounding of MUL_MAT `node` (0: none)
static uint32_t ameva_act_fq_blk(const struct ggml_tensor* node) {
    const uint32_t mode = ameva_act_fq_mode();
    if (!mode || node->op != GGML_OP_MUL_MAT) return 0;
    const struct ggml_tensor* src0 = node->src[0];
    const struct ggml_tensor* src1 = node->src[1];
    if (!ggml_is_quantized(src0->type) || src1->type != GGML_TYPE_F32) return 0;
    const bool super = ggml_blck_size(src0->type) == 256;
    const uint32_t blk = mode == 1 ? (super ? 256u : 32u) : (mode == 256 ? (super ? 256u : 0u) : 32u);
    return blk && (uint32_t)src0->ne[0] % blk == 0 ? blk : 0u;
}

// Scratch bytes of the rounded activations of `node` (one f32 row of K per activation row)
static uint64_t ameva_act_fq_bytes(const struct ggml_tensor* node) {
    return ameva_act_fq_blk(node) ? (uint64_t)node->src[1]->ne[1] * (uint64_t)node->src[0]->ne[0] * 4 : 0;
}

static uint32_t ameva_qgemm_min_rows(void);

// Whether MUL_MAT `node` runs ameva_mmq, and then the activation rows quantized per chunk (one scratch fill) and the
// scratch bytes of a chunk.
static bool ameva_mmq_plan(const struct ggml_tensor* node, uint32_t* chunk_rows, uint64_t* scratch_bytes) {
    const struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1];
    if (node->op != GGML_OP_MUL_MAT || !ameva_repacked_base(src0)) return false;
    if ((uint64_t)src1->ne[1] < ameva_qgemm_min_rows() || ameva_mmq_find(src0->type) < 0 || !ameva_mmq_enabled())
        return false;
    const uint64_t U = (uint64_t)src0->ne[0] / 32, M = (uint64_t)src1->ne[1];
    const uint64_t span = ameva_max_binding_span();
    const uint64_t s1 = st(src1, 1), sd = st(node, 1);
    const uint64_t rows_fit = std::min<uint64_t>(span / (s1 * 4), span / (sd * 4));
    const uint64_t r = std::max<uint64_t>(1, std::min<uint64_t>({ M, rows_fit, span / (U * kMmqUnitBytes), 65535ull * 32 }));
    *chunk_rows = (uint32_t)r;
    *scratch_bytes = r * U * kMmqUnitBytes;
    return true;
}

// MUL_MAT with ameva_mmq: per chunk of activation rows, ameva_quant_a8 fills the scratch buffer (int8 values at 0,
// scales and half sums after them), then ameva_mmq runs over the weight slices.
static const char* enc_mul_mat_mmq(AmevaRecorder& rec, const struct ggml_tensor* node, uint32_t chunk_rows) {
    const struct ggml_tensor* src0 = node->src[0];
    const struct ggml_tensor* src1 = node->src[1];
    const struct ggml_tensor* base = ameva_repacked_base(src0);
    const uint32_t K = (uint32_t)src0->ne[0], N = (uint32_t)src0->ne[1], M = (uint32_t)src1->ne[1];
    const uint32_t U = K / 32;
    const int vi = ameva_mmq_find(src0->type);
    AcKernel* kernel = vi >= 0 ? ameva_mmq_kernel(vi) : nullptr;
    AcKernel* quant = ameva_op_kernel(AMEVA_K_QUANT_A8);
    if (!kernel || !quant) return "no ameva_mmq / ameva_quant_a8 kernel";
    AmevaRepackLayout L = {};
    AmevaRepackRegions R = {};
    if (!ameva_repack_layout(base->type, &L) || !ameva_repack_regions(base->type, base->ne[0], ggml_nrows(base), &R))
        return "repacked weight without a layout";
    const size_t row_bytes = ggml_row_size(src0->type, K);
    const uint64_t first_row = (uint64_t)((const char*)src0->data - (const char*)base->data) / row_bytes;
    if (!s_scratch || (uint64_t)chunk_rows * U * kMmqUnitBytes > s_scratch_bytes) return "MMQ scratch not reserved";
    const uint64_t span = ameva_max_binding_span();
    // weight rows per dispatch: one binding per region, 65535 workgroups of 64 rows, kDispatchMac over a chunk
    uint64_t slice_cap = std::min<uint64_t>(span / R.q_row, 65535ull * 64);
    const uint64_t mac_rows = (uint64_t)(kDispatchMac / ((double)chunk_rows * (double)K));
    slice_cap = std::min<uint64_t>(slice_cap, std::max<uint64_t>(64, mac_rows & ~63ull));
    const uint32_t rows_per_slice = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)N, slice_cap));
    const uint32_t s1 = st(src1, 1), sd = st(node, 1);
    const char* e = nullptr;
    for (uint32_t m0 = 0; m0 < M; m0 += chunk_rows) {
        const uint32_t mc = std::min(chunk_rows, M - m0);
        AmevaBind b1;
        e = ameva_bind_span(src1, (uint64_t)m0 * s1 * 4, ((uint64_t)(mc - 1) * s1 + K) * 4, 4, &b1);
        if (e) return e;
        if (b1.elem % 4 != 0) return "activation offset not a multiple of 16 bytes";
        const uint64_t aq_bytes = (uint64_t)mc * U * 32;
        AmevaBind bs = {};
        bs.b.buffer = s_scratch;
        bs.b.offset = 0;
        bs.b.range = std::min<uint64_t>(s_scratch_bytes, s_limits.max_storage_buffer_range) & ~3ull;
        bs.key = s_scratch;
        bs.lo = 0;
        bs.hi = (uint64_t)mc * U * kMmqUnitBytes;
        const uint32_t pcq[6] = { mc, K, b1.elem, s1, 0u, (uint32_t)(aq_bytes / 8) };
        const AcBufferBinding bindq[3] = { b1.b, bs.b, bs.b };
        e = ameva_record(rec, quant, &b1, 1, &bs, 1, bindq, 3, pcq, sizeof(pcq), ameva_grid_1d((uint64_t)mc * U, 64),
                         (double)mc * K);
        if (e) return e;
        for (uint32_t n0 = 0; n0 < N; n0 += rows_per_slice) {
            const uint32_t nc = std::min(rows_per_slice, N - n0);
            const uint64_t row = first_row + n0;
            AmevaBind w[3];
            e = ameva_bind_span(base, R.q + row * R.q_row, (uint64_t)nc * R.q_row, L.qb == 8 ? 8 : 16, &w[0]);
            if (!e) e = ameva_bind_span(base, R.s + row * R.s_row, (uint64_t)nc * R.s_row, 2, &w[2]);
            if (!e) {
                if (R.h_row) e = ameva_bind_span(base, R.h + row * R.h_row, (uint64_t)nc * R.h_row, L.hb == 8 ? 8 : 4, &w[1]);
                else w[1] = w[2];
            }
            AmevaBind bd;
            if (!e) e = ameva_bind_span(node, ((uint64_t)m0 * sd + n0) * 4, ((uint64_t)(mc - 1) * sd + nc) * 4, 4, &bd);
            if (e) return e;
            const uint32_t pc[10] = { mc, K, nc, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2,
                                      0u, (uint32_t)(aq_bytes / 8), bd.elem, sd };
            const AcBufferBinding bind[6] = { w[0].b, w[1].b, w[2].b, bs.b, bs.b, bd.b };
            const AmevaBind reads[4] = { w[0], w[1], w[2], bs };
            e = ameva_record(rec, kernel, reads, 4, &bd, 1, bind, 6, pc, sizeof(pc),
                             { (nc + 63) / 64, (mc + 31) / 32, 1 }, (double)mc * nc * K);
            if (e) return e;
        }
    }
    rec.mul_mat_nodes++;
    rec.qmv_nodes++;
    rec.mmq_nodes++;
    return nullptr;
}

// MMVQ (shaders/ameva_mmvq.comp): token-generation MUL_MAT on a repacked weight with int8 activations and packed
// integer dot products, as ameva_mmq does for prompt-sized products. It is one more family of token-generation GEMV
// layouts (weight rows per lane group x lanes per weight row), measured with the ameva_qmv and mali_qmv layouts per
// device and weight shape (layout selection below). Its results are not those of the f32 layouts bit for bit: the
// activations are rounded to int8 per 32 elements (|error| <= d_a / 2 per element; the CPU backend rounds them to
// q8_0 / q8_K as well).
//   AMEVA_MMVQ unset or "auto"  measured; a layout of this family is chosen for a shape when it is more than
//                 kMmvqTuneMargin faster than the best f32 layout. Without VK_KHR_shader_integer_dot_product the
//                 family is the emulation with 32-bit multiplications (ameva_i8_family above)
//   AMEVA_MMVQ=0  never
//   AMEVA_MMVQ=1  every one-row MUL_MAT on a repacked weight whose type the family has, those not measured:
//                 AMEVA_MMVQ_ROWS=4|2|1 (default 4) x AMEVA_MMVQ_TPR=1|2|4|8|16|32|64 (default 32)
// Token generation, 64 tokens, t/s (2026-10-04; three alternating rounds, five on the Galaxy S21): the build before
// (13 f32 layouts) -> 22 f32 layouts (lanes per row 4, 2, 1 added) -> with the ameva_mmvq layouts:
//   Mali-G68 r38p1 (Galaxy A35)          Qwen2.5-0.5B 12.3-12.7 -> 14.5-14.6 -> 19.0, Llama-3.2-1B 6.9-7.4 -> 7.1-7.5 -> 11.9
//   Adreno 650, Turnip (Galaxy S20)      Qwen 25.7-26.0 -> 27.0-27.1 -> 32.6-32.9, Llama 13.5 -> 13.5 -> 18.8 (587 MHz)
//   Adreno 830 (Galaxy S25, 525 MHz)     Llama 30.9 -> 30.8-31.1 -> 34.3-35.0
//   Mali-G68 r32p1 (Galaxy A53), Mali-G78 r38p0 (Galaxy S21): no VK_KHR_shader_integer_dot_product; the added f32
//   layouts gave 13.7 -> 14.4 and 6.3 -> 7.0 on the A53 and no difference on the S21 (22.1-22.8, 12.0-13.2).
// On the A35 the fastest ameva_mmvq layout was 1 row x 4 lanes for most shapes and took 0.43 to 0.77 of the fastest
// f32 layout's GPU time (q4_K 896x4864: 244 against 562 us); on the S25 it was chosen for 4 of 14 shapes.
// CPU-relative accuracy with these layouts (tests/ameva_llm_parity.cpp, mean KL divergence, gate 0.01): Galaxy S25
// 2.56e-3 / 2.71e-3 / 6.06e-3, Galaxy S20 3.12e-3 / 2.86e-3 / 6.36e-3 (Qwen2.5-0.5B / Llama-3.2-1B / Gemma-2-2B).
enum { AMEVA_MMVQ_OFF = 0, AMEVA_MMVQ_FORCED = 1, AMEVA_MMVQ_AUTO = 2 };
struct AmevaMmvqPolicy {
    int mode;
    uint32_t rows, tpr; // AMEVA_MMVQ_FORCED: the layout
};

static const AmevaMmvqPolicy& ameva_mmvq_policy(void) {
    static const AmevaMmvqPolicy policy = [] {
        AmevaMmvqPolicy p = { AMEVA_MMVQ_AUTO, 4, 32 };
        const char* e = getenv("AMEVA_MMVQ");
        if (e && *e && strcmp(e, "auto") != 0) {
            if (!strcmp(e, "0")) p.mode = AMEVA_MMVQ_OFF;
            else if (!strcmp(e, "1")) p.mode = AMEVA_MMVQ_FORCED;
            else GGML_ABORT("ggml-ameva: AMEVA_MMVQ must be auto, 0 or 1 (got \"%s\")", e);
        }
        const char* r = getenv("AMEVA_MMVQ_ROWS");
        const char* l = getenv("AMEVA_MMVQ_TPR");
        if (r && !*r) r = nullptr;
        if (l && !*l) l = nullptr;
        if ((r || l) && p.mode != AMEVA_MMVQ_FORCED) {
            GGML_ABORT("ggml-ameva: AMEVA_MMVQ_ROWS and AMEVA_MMVQ_TPR set the layout of AMEVA_MMVQ=1");
        }
        if (r) {
            const long v = strtol(r, nullptr, 10);
            if (v != 1 && v != 2 && v != 4) GGML_ABORT("ggml-ameva: AMEVA_MMVQ_ROWS must be 1, 2 or 4 (got \"%s\")", r);
            p.rows = (uint32_t)v;
        }
        if (l) {
            const long v = strtol(l, nullptr, 10);
            if (v != 1 && v != 2 && v != 4 && v != 8 && v != 16 && v != 32 && v != 64) {
                GGML_ABORT("ggml-ameva: AMEVA_MMVQ_TPR must be 1, 2, 4, 8, 16, 32 or 64 (got \"%s\")", l);
            }
            p.tpr = (uint32_t)v;
        }
        const AmevaI8Family& fam = ameva_i8_family();
        if (p.mode == AMEVA_MMVQ_FORCED) {
            fprintf(stderr, "ggml-ameva: MMVQ on for every token-generation MUL_MAT it has a kernel for (AMEVA_MMVQ=1): "
                            "int8 activations, rows=%u tpr=%u, %s\n", p.rows, p.tpr,
                    fam.emulated ? "integer dot product emulated with 32-bit multiplications"
                                 : s_limits.integer_dot_product_accelerated ? "4x8 packed accelerated: yes"
                                                                            : "4x8 packed accelerated: no");
        } else if (p.mode == AMEVA_MMVQ_AUTO && fam.emulated) {
            fprintf(stderr, "ggml-ameva: MMVQ layouts use the integer dot product emulated with 32-bit multiplications "
                            "(%s)\n", s_limits.integer_dot_product ? "AMEVA_MMVQ_EMU=1"
                                                                    : "no VK_KHR_shader_integer_dot_product");
        }
        return p;
    }();
    return policy;
}

// One-row MUL_MAT with ameva_mmvq in the layout rows x tpr: ameva_quant_a8 writes the activation row to the scratch
// buffer (int8 values at 0, scales and half sums after them) unless the row quantized last is this one, then
// ameva_mmvq runs over the weight slices. `ep`: the fused consumer (as for ameva_qmv).
static const char* enc_mul_mat_mmvq(AmevaRecorder& rec, const struct ggml_tensor* node, const AmevaEpilogue* ep,
                                    uint32_t rows, uint32_t tpr) {
    const struct ggml_tensor* src0 = node->src[0];
    const struct ggml_tensor* src1 = node->src[1];
    const struct ggml_tensor* base = ameva_repacked_base(src0);
    const uint32_t K = (uint32_t)src0->ne[0], N = (uint32_t)src0->ne[1];
    const uint32_t U = K / 32;
    const AmevaI8Family& fam = ameva_i8_family();
    const int vi = ameva_mmvq_find(src0->type, rows);
    AcKernel* kernel = vi >= 0 ? ameva_mmvq_kernel(vi) : nullptr;
    AcKernel* quant = ameva_op_kernel(fam.quant);
    if (!kernel || !quant) return "no ameva_mmvq / ameva_quant_a8 kernel";
    AmevaRepackLayout L = {};
    AmevaRepackRegions R = {};
    if (!ameva_repack_layout(base->type, &L) || !ameva_repack_regions(base->type, base->ne[0], ggml_nrows(base), &R))
        return "repacked weight without a layout";
    const size_t row_bytes = ggml_row_size(src0->type, K);
    const uint64_t first_row = (uint64_t)((const char*)src0->data - (const char*)base->data) / row_bytes;
    if (!s_scratch || (uint64_t)U * fam.unit_bytes > s_scratch_bytes) return "MMVQ scratch not reserved";
    AmevaBind be = {}, beo = {};
    if (ep) {
        const char* ee = ameva_bind(ep->e, 4, &be);
        if (!ee) ee = ameva_bind(ep->out, 4, &beo);
        if (ee) return ee;
    }
    const uint32_t ad_off = U * fam.ad_units; // AD after the AQ words of every unit
    AmevaBind bs = {};
    bs.b.buffer = s_scratch;
    bs.b.offset = 0;
    bs.b.range = std::min<uint64_t>(s_scratch_bytes, s_limits.max_storage_buffer_range) & ~3ull;
    bs.key = s_scratch;
    bs.lo = 0;
    bs.hi = (uint64_t)U * fam.unit_bytes;
    const char* e = nullptr;
    if (rec.a8_src != src1) {
        AmevaBind b1;
        e = ameva_bind_span(src1, 0, (uint64_t)K * 4, 4, &b1);
        if (e) return e;
        if (b1.elem % 4 != 0) return "activation offset not a multiple of 16 bytes";
        const uint32_t pcq[6] = { 1u, K, b1.elem, K, 0u, ad_off };
        const AcBufferBinding bindq[3] = { b1.b, bs.b, bs.b };
        e = ameva_record(rec, quant, &b1, 1, &bs, 1, bindq, 3, pcq, sizeof(pcq), ameva_grid_1d(U, 64), (double)K);
        if (e) return e;
        rec.a8_src = src1; // after ameva_record: the write to the scratch buffer cleared the previous row
        rec.a8_range = { b1.key, b1.lo, b1.hi };
    }
    const uint32_t rows_per_wg = (64u / tpr) * rows;
    const uint64_t span = ameva_max_binding_span();
    const uint32_t rows_per_slice = (uint32_t)std::max<uint64_t>(
        1, std::min<uint64_t>({ (uint64_t)N, span / R.q_row, 65535ull * rows_per_wg }));
    for (uint32_t n0 = 0; n0 < N; n0 += rows_per_slice) {
        const uint32_t nc = std::min(rows_per_slice, N - n0);
        const uint64_t row = first_row + n0;
        AmevaBind w[3];
        e = ameva_bind_span(base, R.q + row * R.q_row, (uint64_t)nc * R.q_row, L.qb == 8 ? 8 : 16, &w[0]);
        if (!e) e = ameva_bind_span(base, R.s + row * R.s_row, (uint64_t)nc * R.s_row, 2, &w[2]);
        if (!e) {
            if (R.h_row) e = ameva_bind_span(base, R.h + row * R.h_row, (uint64_t)nc * R.h_row, L.hb == 8 ? 8 : 4, &w[1]);
            else w[1] = w[2]; // no high bits: binding 1 is not read
        }
        AmevaBind bd;
        if (!e) e = ameva_bind_span(node, (uint64_t)n0 * 4, (uint64_t)nc * 4, 4, &bd);
        if (e) return e;
        const uint32_t pc[12] = { K, nc, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2, 0u,
                                  ad_off, bd.elem, tpr, ep ? ep->mode : 0u,
                                  ep ? be.elem + n0 : 0u, ep ? beo.elem + n0 : 0u };
        const AcBufferBinding bind[8] = { w[0].b, w[1].b, w[2].b, bs.b, bs.b, bd.b, ep ? be.b : bd.b,
                                          ep ? beo.b : bd.b };
        const AmevaBind reads[5] = { w[0], w[1], w[2], bs, be };
        const AmevaBind writes[2] = { bd, beo };
        e = ameva_record(rec, kernel, reads, ep ? 5 : 4, writes, ep ? 2 : 1, bind, 8, pc, sizeof(pc),
                         { (nc + rows_per_wg - 1) / rows_per_wg, 1, 1 }, (double)nc * K);
        if (e) return e;
    }
    rec.mul_mat_nodes++;
    rec.qmv_nodes++;
    rec.mmq_nodes++; // int8 activations: ameva_qmm_check compares with the int8-rounded reference
    rec.mmvq_nodes++;
    return nullptr;
}

// Lanes per weight row of ameva_qmv: 16 for every row length. Measured with Qwen2.5-0.5B Q4_K_M token generation
// (2026-10-03): Adreno 650 per-kernel times with 8 / 16 / 64 lanes: q4_K 896x4864 336 / 356 / 611 us, q8_0
// 151936x896 7.3 / 8.2 / 11.4 ms; Mali-G68 decode 14.0 / 14.5 t/s with 8 / 16 lanes, 14.1 with 64 lanes for
// K = 4864. More lanes per row means fewer rows per workgroup and more activation reads per weight byte.
// AMEVA_QMV_TPR=1|2|4|8|16|32|64 fixes it (kernel tuning). This is the prompt variant (8 activation rows per workgroup);
// the one-row variant's layout is chosen per shape (token-generation GEMV selection below).
static uint32_t ameva_env_qmv_tpr(void) {
    static const uint32_t forced = [] {
        const char* e = getenv("AMEVA_QMV_TPR");
        if (!e || !*e) return 0u;
        if (!strcmp(e, "1")) return 1u;
        if (!strcmp(e, "2")) return 2u;
        if (!strcmp(e, "4")) return 4u;
        if (!strcmp(e, "8")) return 8u;
        if (!strcmp(e, "16")) return 16u;
        if (!strcmp(e, "32")) return 32u;
        if (!strcmp(e, "64")) return 64u;
        GGML_ABORT("ggml-ameva: AMEVA_QMV_TPR must be 1, 2, 4, 8, 16, 32 or 64 (got \"%s\")", e);
    }();
    return forced;
}

static uint32_t ameva_qmv_tpr(uint32_t units) {
    (void)units;
    const uint32_t forced = ameva_env_qmv_tpr();
    return forced ? forced : 16u;
}

// =============================================================================
// Token-generation GEMV: layout selection
// =============================================================================
// MUL_MAT of one activation row with a repacked weight runs one of these layouts:
//   ameva_qmv  64 invocations per workgroup; `tpr` lanes per weight row (push constant: 1 .. 64) and `rows` weight
//              rows per lane group (one module per value: 4, 2, 1), so (64 / tpr) * rows weight rows per workgroup.
//              With tpr 1 a lane computes whole rows and nothing is reduced.
//   mali_qmv   32 lanes per weight row, 4 weight rows per workgroup of 128 invocations
//   ameva_mmvq the ameva_qmv work split with int8 activations and integer dot products (AMEVA_MMVQ above); measured
//              where the device offers them, and chosen only when more than kMmvqTuneMargin faster than the best of
//              the two f32 families, because its results differ from theirs by the rounding of the activations
// The fastest layout depends on the device, its driver and the shape, and what the driver reports does not decide
// it: Mali-G68 (r32p1, r38p1) and Mali-G78 (r38p0) report the same subgroup size (16, fixed) and the same limits, yet
// with Qwen2.5-0.5B Q4_K_M (2026-10-04) the fastest layout differed by device and shape. Large shapes: 4 rows x 8
// lanes on the G68 r32p1 (default 6-12 % slower), the default or 4 x 32 on the G68 r38p1, the default, 4 x 8 or
// mali_qmv on the G78. 128-row shapes: 1 row per lane group or mali_qmv on all three (default 1.5-2.8x slower).
// mali_qmv against the default on the large shapes: 1.0-1.9x slower on the G68, 0.9-1.3x on the G78, 1.6-1.9x slower
// on Adreno 830, 4.6-8.3x slower on Adreno 650 (Turnip). So the layout is measured: before the first
// graph that multiplies a weight shape (type, K, N) not seen before, every candidate runs on that weight with a GPU
// timestamp after each dispatch (ameva_qmv_tune_shape), and the fastest is kept unless the default is within
// kQmvTuneMargin of it. Results are kept per device and driver (s_device_key) and per kernel build (SPIR-V SHA-256
// of both families) in the tune file, so a device measures each shape once.
// The measurement ranks layouts by GPU time per dispatch. Token speed also depends on the clock the GPU governor
// picks, which the engine does not control: on the Galaxy S21 (Mali-G78, "Interactive" governor, 2026-10-04) 27 runs
// of one build had a mean GPU clock of 465 to 820 MHz and 10.7 to 25.6 t/s (correlation 0.83). The governor stayed at
// a low clock more often with the ameva_qmv layouts (22.4-22.7 t/s per GHz of mean clock, median 11.6-14.7 t/s) than
// with mali_qmv (20.3 t/s per GHz, median 15.8 t/s): scripts/termux/lab/gpu_clock_ab.sh.
//   AMEVA_QMV_KERNEL=auto|qmv|mali   auto (default): measured; qmv or mali: that family, nothing measured
//   AMEVA_QMV_ROWS=1|2|4, AMEVA_QMV_TPR=1|2|4|8|16|32|64   fix the ameva_qmv layout (nothing measured;
//                                    AMEVA_QMV_TPR also sets the prompt variant's lanes per row)
//   AMEVA_MALI_QMV=1|on|0|off        earlier name of AMEVA_QMV_KERNEL=mali|qmv
//   AMEVA_QMV_TUNE=0                 nothing measured: the default layout (ameva_qmv, 4 rows, 16 lanes)
//   AMEVA_QMV_TUNE_FILE=<path>|0     where measurements are kept (default $HOME/.cache/ameva/ggml-ameva-qmv-tune.txt);
//                                    0: measured in every process and not kept
struct AmevaQmvConfig {
    uint8_t mali; // 1: mali_qmv; 0: ameva_qmv or ameva_mmvq with rows and tpr
    uint8_t rows; // weight rows per lane group (1, 2, 4)
    uint8_t tpr;  // lanes per weight row (1 .. 64)
    uint8_t i8;   // 1: ameva_mmvq (int8 activations, integer dot products)
};
static const AmevaQmvConfig kQmvDefault = { 0, 4, 16, 0 };
static const AmevaQmvConfig kQmvMali = { 1, 4, 32, 0 };
static const double kQmvTuneMargin = 0.05;  // another layout replaces the default only when more than 5 % faster
// an ameva_mmvq layout replaces the f32 layout chosen for a shape only when more than 10 % faster
static const double kMmvqTuneMargin = 0.10;
// Sanity bound of the measurement for an ameva_mmvq layout: |y - y_default| <= this * max|y_default| (the rounding
// of a uniform activation row to int8 gives about 0.005). The exact bound per case is checked by ameva_qmm_check.
static const double kMmvqTuneTolerance = 0.05;
static const uint32_t kQmvTuneRows = 8192;  // weight rows timed (a larger weight: its first rows)
static const uint32_t kQmvTuneRounds = 5;   // timed rounds after one warm-up round; the median of each layout counts
// a layout's output may differ from the default's by summation order only: |y - y_default| <= this * max|y_default|
static const double kQmvTuneTolerance = 1e-4;

static bool qmv_config_eq(const AmevaQmvConfig& a, const AmevaQmvConfig& b) {
    return a.mali == b.mali && a.i8 == b.i8 && (a.mali || (a.rows == b.rows && a.tpr == b.tpr));
}

static uint32_t qmv_rows_per_wg(const AmevaQmvConfig& c) {
    return c.mali ? 4u : (64u / c.tpr) * c.rows;
}

static const char* qmv_config_str(const AmevaQmvConfig& c, char* buf, size_t n) {
    if (c.mali) snprintf(buf, n, "mali_qmv");
    else snprintf(buf, n, "%s rows=%u tpr=%u", !c.i8 ? "ameva_qmv" : ameva_i8_family().emulated ? "ameva_mmvq(emu)" : "ameva_mmvq",
                  (unsigned)c.rows, (unsigned)c.tpr);
    return buf;
}

struct AmevaQmvPolicy {
    bool tune;             // measured per shape
    bool forced;           // fixed by AMEVA_QMV_KERNEL / AMEVA_MALI_QMV / AMEVA_QMV_ROWS / AMEVA_QMV_TPR
    AmevaQmvConfig config; // the layout when nothing is measured
};

static const AmevaQmvPolicy& ameva_qmv_policy(void) {
    static const AmevaQmvPolicy policy = [] {
        AmevaQmvPolicy p = { true, false, kQmvDefault };
        const char* kernel = getenv("AMEVA_QMV_KERNEL");
        const char* legacy = getenv("AMEVA_MALI_QMV");
        const char* rows = getenv("AMEVA_QMV_ROWS");
        const char* tune = getenv("AMEVA_QMV_TUNE");
        if (kernel && !*kernel) kernel = nullptr;
        if (legacy && *legacy) {
            if (kernel) GGML_ABORT("ggml-ameva: set AMEVA_QMV_KERNEL or AMEVA_MALI_QMV, not both");
            if (!strcmp(legacy, "1") || !strcmp(legacy, "on")) kernel = "mali";
            else if (!strcmp(legacy, "0") || !strcmp(legacy, "off")) kernel = "qmv";
            else GGML_ABORT("ggml-ameva: AMEVA_MALI_QMV must be 1, on, 0 or off (got \"%s\")", legacy);
        }
        if (!kernel) kernel = "auto";
        const bool is_auto = !strcmp(kernel, "auto"), is_mali = !strcmp(kernel, "mali");
        if (!is_auto && !is_mali && strcmp(kernel, "qmv") != 0) {
            GGML_ABORT("ggml-ameva: AMEVA_QMV_KERNEL must be auto, qmv or mali (got \"%s\")", kernel);
        }
        uint32_t nrows = 0;
        if (rows && *rows) {
            if (!strcmp(rows, "1")) nrows = 1;
            else if (!strcmp(rows, "2")) nrows = 2;
            else if (!strcmp(rows, "4")) nrows = 4;
            else GGML_ABORT("ggml-ameva: AMEVA_QMV_ROWS must be 1, 2 or 4 (got \"%s\")", rows);
        }
        const uint32_t tpr = ameva_env_qmv_tpr();
        if (is_mali && (nrows || tpr)) {
            GGML_ABORT("ggml-ameva: AMEVA_QMV_ROWS and AMEVA_QMV_TPR set the ameva_qmv layout, but mali_qmv is "
                       "selected (AMEVA_QMV_KERNEL or AMEVA_MALI_QMV)");
        }
        if (tune && *tune && strcmp(tune, "0") != 0 && strcmp(tune, "1") != 0) {
            GGML_ABORT("ggml-ameva: AMEVA_QMV_TUNE must be 0 or 1 (got \"%s\")", tune);
        }
        p.forced = !is_auto || nrows || tpr;
        if (p.forced && tune && !strcmp(tune, "1")) {
            GGML_ABORT("ggml-ameva: AMEVA_QMV_TUNE=1 measures the token-generation GEMV layout, which "
                       "AMEVA_QMV_KERNEL / AMEVA_MALI_QMV / AMEVA_QMV_ROWS / AMEVA_QMV_TPR fix");
        }
        if (is_mali) p.config = kQmvMali;
        else if (p.forced) p.config = { 0, (uint8_t)(nrows ? nrows : 4u), (uint8_t)(tpr ? tpr : 16u), 0 };
        p.tune = !p.forced && !(tune && !strcmp(tune, "0"));
        char c[48];
        if (p.forced) {
            fprintf(stderr, "ggml-ameva: token-generation GEMV layout fixed by the environment: %s\n",
                    qmv_config_str(p.config, c, sizeof(c)));
        } else if (!p.tune) {
            fprintf(stderr, "ggml-ameva: token-generation GEMV layout not measured (AMEVA_QMV_TUNE=0): %s\n",
                    qmv_config_str(p.config, c, sizeof(c)));
        }
        return p;
    }();
    return policy;
}

// one measured weight shape
enum { AMEVA_I8_UNMEASURED = 0, AMEVA_I8_MEASURED = 1, AMEVA_I8_CHOSEN = 2 };
struct AmevaQmvShape {
    int type;
    uint32_t K, N;
    AmevaQmvConfig config; // the f32 layout chosen (ameva_qmv or mali_qmv)
    double us, default_us; // median GPU time per dispatch of `config` and of the default layout
    bool from_file;
    uint8_t i8_state;      // AMEVA_I8_*: whether the ameva_mmvq layouts were measured, and whether `i8` replaces `config`
    AmevaQmvConfig i8;     // the fastest ameva_mmvq layout (i8_state != AMEVA_I8_UNMEASURED)
    double i8_us;          // its median GPU time per dispatch, the quantization of the activation row included
};
static std::mutex s_tune_mutex; // the measured shapes and the tune file
static std::vector<AmevaQmvShape> s_qmv_shapes;
static bool s_tune_file_read = false;
static std::string s_tune_path;

// caller holds s_tune_mutex
static AmevaQmvShape* ameva_qmv_shape_find(int type, uint32_t K, uint32_t N) {
    for (AmevaQmvShape& s : s_qmv_shapes) {
        if (s.type == type && s.K == K && s.N == N) return &s;
    }
    return nullptr;
}

// Whether the ameva_mmvq layouts take part in the measurements of this process
static bool ameva_mmvq_measured(void) {
    return ameva_mmvq_policy().mode == AMEVA_MMVQ_AUTO && ameva_qmv_policy().tune;
}

// Is `node` a MUL_MAT that enc_mul_mat runs with the one-row GEMV on a repacked weight?
static bool ameva_qmv_one_row(const struct ggml_tensor* node) {
    if (!node || node->op != GGML_OP_MUL_MAT || ggml_is_empty(node) || node->src[1]->ne[1] != 1) return false;
    uint32_t rows = 0;
    uint64_t bytes = 0;
    if (ameva_mmq_plan(node, &rows, &bytes)) return false;
    return ameva_repacked_base(node->src[0]) != nullptr && ameva_qgemm_min_rows() > 1;
}

// Layout of a one-row MUL_MAT; false when the shape should have been measured (ameva_qmv_tune_graph) and was not
static bool ameva_qmv_choice(const struct ggml_tensor* node, AmevaQmvConfig* out) {
    const AmevaMmvqPolicy& mp = ameva_mmvq_policy();
    if (mp.mode == AMEVA_MMVQ_FORCED && ameva_mmvq_find(node->src[0]->type, mp.rows) >= 0) {
        *out = { 0, (uint8_t)mp.rows, (uint8_t)mp.tpr, 1 };
        return true;
    }
    const AmevaQmvPolicy& p = ameva_qmv_policy();
    if (!p.tune) {
        *out = p.config;
        return true;
    }
    const struct ggml_tensor* src0 = node->src[0];
    std::lock_guard<std::mutex> lock(s_tune_mutex);
    const AmevaQmvShape* s = ameva_qmv_shape_find((int)src0->type, (uint32_t)src0->ne[0], (uint32_t)src0->ne[1]);
    if (!s) return false;
    const bool with_i8 = mp.mode == AMEVA_MMVQ_AUTO && ameva_mmvq_find(src0->type, 4) >= 0;
    if (with_i8 && s->i8_state == AMEVA_I8_UNMEASURED) return false;
    *out = with_i8 && s->i8_state == AMEVA_I8_CHOSEN ? s->i8 : s->config;
    return true;
}

// Scratch bytes of one-row MUL_MAT `node` when it runs ameva_mmvq (its quantized activation row), otherwise 0
static uint64_t ameva_mmvq_scratch_bytes(const struct ggml_tensor* node) {
    AmevaQmvConfig c;
    if (!ameva_qmv_one_row(node) || !ameva_qmv_choice(node, &c) || !c.i8) return 0;
    return (uint64_t)(node->src[0]->ne[0] / 32) * ameva_i8_family().unit_bytes;
}

// Kernel build of a weight type: SPIR-V SHA-256 prefixes of its three one-row ameva_qmv modules (rows 4, 2, 1; 7 hex
// digits each), of its mali_qmv module (16 hex digits) and of its three ameva_mmvq modules (7 hex digits each); "-"
// where there is no kernel
static void ameva_qmv_kernel_ids(int type, char qmv[24], char mali[24], char mmvq[24]) {
    static const uint32_t kRows[3] = { 4, 2, 1 };
    int n = 0, m = 0;
    for (uint32_t r : kRows) {
        const int vi = ameva_qmv_find((enum ggml_type)type, 1, r);
        n += snprintf(qmv + n, 24 - (size_t)n, "%.7s", vi >= 0 ? kAmevaQmvVariants[vi].spirv_sha256 : "-");
        const int qi = ameva_mmvq_find((enum ggml_type)type, r);
        m += snprintf(mmvq + m, 24 - (size_t)m, "%.7s", qi >= 0 ? ameva_i8_family().variants[qi].spirv_sha256 : "-");
    }
    const int mvi = ameva_mali_qmv_find((enum ggml_type)type);
    snprintf(mali, 24, "%.16s", mvi >= 0 ? kAmevaMaliQmvVariants[mvi].spirv_sha256 : "-");
}

static int ameva_type_from_name(const char* name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((enum ggml_type)t);
        if (n && !strcmp(n, name)) return t;
    }
    return -1;
}

static std::string ameva_qmv_tune_path(void) {
    const char* e = getenv("AMEVA_QMV_TUNE_FILE");
    if (e && strcmp(e, "0") == 0) return std::string();
    if (e && *e) return std::string(e);
    const char* home = getenv("HOME");
    if (!home || !*home) {
        fprintf(stderr, "ggml-ameva: GEMV measurements not kept: HOME is not set (set AMEVA_QMV_TUNE_FILE=<file>)\n");
        return std::string();
    }
    return std::string(home) + "/.cache/ameva/ggml-ameva-qmv-tune.txt";
}

// Tune file: line 1 "ggml-ameva-qmv-tune 2", line 2 "device <s_device_key> <device name>", then one line per shape:
// <type> <K> <N> <ameva_qmv SPIR-V id> <mali_qmv SPIR-V id> <ameva_mmvq SPIR-V id> <mali> <rows> <tpr> <us>
// <default us> <i8 state> <i8 rows> <i8 tpr> <i8 us>  (i8 state: AMEVA_I8_*).
// Read once (caller holds s_tune_mutex). A file of another device, driver or format, or one that does not parse, is
// not used (said once; rewritten after the next measurement); lines of other kernel builds are skipped.
static void ameva_qmv_tune_file_read(void) {
    if (s_tune_file_read) return;
    s_tune_file_read = true;
    s_tune_path = ameva_qmv_tune_path();
    if (s_tune_path.empty()) return;
    if (s_device_key.empty()) {
        fprintf(stderr, "ggml-ameva: GEMV measurements not kept: device and driver identity unavailable (pipeline "
                        "cache header)\n");
        s_tune_path.clear();
        return;
    }
    FILE* f = fopen(s_tune_path.c_str(), "r");
    if (!f) return; // nothing measured on this device yet
    std::vector<AmevaQmvShape> got;
    char line[512];
    int ln = 0, stale = 0;
    const char* bad = nullptr;
    while (!bad && fgets(line, sizeof(line), f)) {
        ++ln;
        if (ln == 1) {
            if (strcmp(line, "ggml-ameva-qmv-tune 2\n") != 0) bad = "another format";
            continue;
        }
        if (ln == 2) {
            char key[128];
            if (sscanf(line, "device %127s", key) != 1) bad = "no device line";
            else if (s_device_key != key) bad = "another device or driver";
            continue;
        }
        char tn[24], qs[24], ms[24], vs[24];
        unsigned K = 0, N = 0, mali = 0, rows = 0, tpr = 0, i8s = 0, i8r = 0, i8t = 0;
        double us = 0, dus = 0, i8us = 0;
        if (sscanf(line, "%23s %u %u %23s %23s %23s %u %u %u %lf %lf %u %u %u %lf", tn, &K, &N, qs, ms, vs, &mali, &rows,
                   &tpr, &us, &dus, &i8s, &i8r, &i8t, &i8us) != 15) {
            bad = "malformed line";
            break;
        }
        const int type = ameva_type_from_name(tn);
        const auto rows_ok = [](unsigned r) { return r == 1 || r == 2 || r == 4; };
        const auto tpr_ok = [](unsigned l) { return l == 1 || l == 2 || l == 4 || l == 8 || l == 16 || l == 32 || l == 64; };
        if (type < 0 || K == 0 || N == 0 || K % 32 != 0 || mali > 1 || (!mali && (!rows_ok(rows) || !tpr_ok(tpr))) ||
            i8s > AMEVA_I8_CHOSEN || (i8s != AMEVA_I8_UNMEASURED && (!rows_ok(i8r) || !tpr_ok(i8t)))) {
            bad = "invalid entry";
            break;
        }
        char cq[24], cm[24], cv[24];
        ameva_qmv_kernel_ids(type, cq, cm, cv);
        if (strcmp(cq, qs) != 0 || strcmp(cm, ms) != 0 || strcmp(cv, vs) != 0) {
            ++stale; // measured with other kernels
            continue;
        }
        AmevaQmvShape s = { type, K, N, mali ? kQmvMali : AmevaQmvConfig{ 0, (uint8_t)rows, (uint8_t)tpr, 0 }, us, dus,
                            true, (uint8_t)i8s, AmevaQmvConfig{ 0, (uint8_t)i8r, (uint8_t)i8t, 1 }, i8us };
        got.push_back(s);
    }
    const bool read_error = ferror(f) != 0;
    fclose(f);
    if (read_error) bad = "read error";
    if (bad) {
        fprintf(stderr, "ggml-ameva: GEMV tune file %s not used (line %d: %s); shapes are measured again and the file "
                        "rewritten\n", s_tune_path.c_str(), ln, bad);
        return;
    }
    for (const AmevaQmvShape& s : got) s_qmv_shapes.push_back(s);
    fprintf(stderr, "ggml-ameva: GEMV layouts of %zu weight shapes from %s%s\n", got.size(), s_tune_path.c_str(),
            stale ? " (entries of other kernel builds skipped)" : "");
}

// caller holds s_tune_mutex
static void ameva_qmv_tune_file_write(void) {
    if (s_tune_path.empty()) return;
    std::string out = "ggml-ameva-qmv-tune 2\ndevice " + s_device_key + " " + s_device_name + "\n";
    char line[320];
    for (const AmevaQmvShape& s : s_qmv_shapes) {
        char qs[24], ms[24], vs[24];
        ameva_qmv_kernel_ids(s.type, qs, ms, vs);
        snprintf(line, sizeof(line), "%s %u %u %s %s %s %u %u %u %.3f %.3f %u %u %u %.3f\n",
                 ggml_type_name((enum ggml_type)s.type), s.K, s.N, qs, ms, vs, (unsigned)s.config.mali,
                 (unsigned)s.config.rows, (unsigned)s.config.tpr, s.us, s.default_us, (unsigned)s.i8_state,
                 (unsigned)s.i8.rows, (unsigned)s.i8.tpr, s.i8_us);
        out += line;
    }
    if (!ameva_write_file(s_tune_path, out.data(), out.size())) {
        fprintf(stderr, "ggml-ameva: GEMV tune file %s could not be written (%s)\n", s_tune_path.c_str(),
                strerror(errno));
    }
}

// Measures every layout on the weight of one-row MUL_MAT `node` and returns the chosen one in *out. A layout whose
// kernel cannot be created or whose result differs from the default layout's beyond summation order is reported and
// left out; failing to measure at all (no GPU timestamps, allocation or submission failure) is an error.
static const char* ameva_qmv_tune_shape(const struct ggml_tensor* node, AmevaQmvShape* out) {
    const struct ggml_tensor* src0 = node->src[0];
    const struct ggml_tensor* base = ameva_repacked_base(src0);
    const uint32_t K = (uint32_t)src0->ne[0], N = (uint32_t)src0->ne[1];
    const int mvi = ameva_mali_qmv_find(src0->type);
    AmevaRepackLayout L = {};
    AmevaRepackRegions R = {};
    if (!base) return "no ameva_qmv kernel for this weight";
    if (!ameva_repack_layout(base->type, &L) || !ameva_repack_regions(base->type, base->ne[0], ggml_nrows(base), &R))
        return "repacked weight without a layout";
    const uint64_t first_row = (uint64_t)((const char*)src0->data - (const char*)base->data) / ggml_row_size(src0->type, K);

    struct Cand {
        AmevaQmvConfig c;
        AcKernel* k;
        double median_us;
        bool ok;
    };
    std::vector<Cand> cands;
    char cs[48];
    // ameva_mmvq layouts and, before them in every round, the quantization of the activation row
    const AmevaI8Family& fam = ameva_i8_family();
    const bool with_i8 = ameva_mmvq_measured() && ameva_mmvq_find(src0->type, 4) >= 0;
    AcKernel* quant = with_i8 ? ameva_op_kernel(fam.quant) : nullptr;
    if (with_i8 && !quant) return "no ameva_quant_a8 kernel for the ameva_mmvq layouts";
    auto add = [&](const AmevaQmvConfig& c) {
        AcKernel* k = nullptr;
        if (c.mali) {
            k = ameva_mali_qmv_kernel(mvi);
        } else if (c.i8) {
            const int vi = ameva_mmvq_find(src0->type, c.rows);
            k = vi >= 0 ? ameva_mmvq_kernel(vi) : nullptr;
        } else {
            const int vi = ameva_qmv_find(src0->type, 1, c.rows);
            k = vi >= 0 ? ameva_qmv_kernel(vi) : nullptr;
        }
        if (k) cands.push_back({ c, k, 0.0, true });
        else if (!cands.empty())
            fprintf(stderr, "ggml-ameva: WARNING: GEMV layout %s left out: kernel creation failed\n",
                    qmv_config_str(c, cs, sizeof(cs)));
        return k != nullptr;
    };
    if (!add(kQmvDefault)) return "default ameva_qmv kernel unavailable";
    static const uint8_t kRows[3] = { 4, 2, 1 };
    static const uint8_t kTpr[7] = { 16, 8, 32, 64, 4, 2, 1 };
    for (uint8_t r : kRows) {
        for (uint8_t t : kTpr) {
            const AmevaQmvConfig c = { 0, r, t, 0 };
            if (!qmv_config_eq(c, kQmvDefault)) add(c);
        }
    }
    if (mvi >= 0) add(kQmvMali);
    if (with_i8) {
        for (uint8_t r : kRows) {
            for (uint8_t t : kTpr) add({ 0, r, t, 1 });
        }
    }
    const uint32_t nc = (uint32_t)cands.size();

    // the first rows of the weight, within one binding per region and 65535 workgroups of one row
    uint64_t rows64 = std::min<uint64_t>(N, kQmvTuneRows);
    rows64 = std::min<uint64_t>(rows64, ameva_max_binding_span() / R.q_row);
    rows64 = std::min<uint64_t>(rows64, 65535);
    const uint32_t nt = (uint32_t)std::max<uint64_t>(1, rows64);
    AmevaBind w[3];
    const char* e = ameva_bind_span(base, R.q + first_row * R.q_row, (uint64_t)nt * R.q_row, L.qb == 8 ? 8 : 16, &w[0]);
    if (!e) e = ameva_bind_span(base, R.s + first_row * R.s_row, (uint64_t)nt * R.s_row, 2, &w[2]);
    if (!e) {
        if (R.h_row) e = ameva_bind_span(base, R.h + first_row * R.h_row, (uint64_t)nt * R.h_row, L.hb == 8 ? 8 : 4, &w[1]);
        else w[1] = w[2];
    }
    if (e) return e;

    // one host-mapped buffer: the activation row, its int8 form (ameva_mmvq layouts), then one output range per
    // layout (aligned binding offsets)
    struct Owned {
        AcBuffer* buf = nullptr;
        ~Owned() {
            if (buf) {
                ac_runtime_buffer_destroy(buf);
                ameva_increment_buffer_destroy_count();
            }
        }
    } owned;
    const uint64_t align = s_limits.min_storage_buffer_offset_alignment;
    const uint64_t a_bytes = ((uint64_t)K * 4 + align - 1) & ~(align - 1);
    const uint64_t d_bytes = ((uint64_t)nt * 4 + align - 1) & ~(align - 1);
    const uint32_t U = K / 32;
    const uint64_t q_bytes = with_i8 ? (((uint64_t)U * fam.unit_bytes + align - 1) & ~(align - 1)) : 0;
    const uint64_t o_base = a_bytes + q_bytes;
    const uint64_t size = o_base + d_bytes * nc;
    const AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)size, AC_BUFFER_USAGE_STORAGE, 0 };
    if (ac_runtime_buffer_create(s_ctx, &bdesc, &owned.buf) != AC_SUCCESS) {
        owned.buf = nullptr;
        return "buffer for the GEMV measurement unavailable";
    }
    ameva_increment_buffer_create_count();
    void* map = nullptr;
    if (ac_runtime_buffer_get_mapped_ptr(owned.buf, &map) != AC_SUCCESS || !map) return "GEMV measurement buffer is not host-mapped";
    float* act = (float*)map;
    uint32_t seed = 0x2545f491u;
    for (uint32_t i = 0; i < K; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        act[i] = (float)(seed >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
    // outputs start as NaN: a layout that leaves rows unwritten fails the comparison
    for (uint32_t c = 0; c < nc; ++c) {
        float* d = (float*)((char*)map + o_base + c * d_bytes);
        for (uint32_t i = 0; i < nt; ++i) d[i] = NAN;
    }
    if (ac_runtime_buffer_flush(owned.buf, 0, size) != AC_SUCCESS) return "GEMV measurement buffer flush failed";

    // Rounds of every layout in turn (a warm-up round first), one submission per round (bounded GPU time per
    // submission) and one barrier between dispatches so that each runs alone. A dispatch's time runs from the end of
    // the dispatch before it, so the first dispatch of a submission also counts the gap since the previous submission:
    // each round therefore starts with an extra dispatch of the default layout whose time is not used. With ameva_mmvq
    // layouts the second dispatch of a round quantizes the activation row; its time is added to theirs.
    const uint32_t lead = with_i8 ? 2u : 1u;
    const uint32_t per_round = nc + lead, nd = per_round * (1 + kQmvTuneRounds);
    const AcBatchDesc batch_desc = { AC_ABI_VERSION, sizeof(AcBatchDesc), nd,
                                     AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_DISPATCH_TIMESTAMPS };
    AcCommandBatch* batch = nullptr;
    AcResult r = ac_runtime_batch_begin_ex(s_ctx, &batch_desc, &batch);
    if (r == AC_ERR_FEATURE_NOT_SUPPORTED) {
        return "the GPU queue has no timestamps, so the GEMV layout cannot be measured (AMEVA_QMV_TUNE=0 runs the "
               "default layout)";
    }
    if (r != AC_SUCCESS) return "batch for the GEMV measurement could not be begun";
    for (uint32_t round = 0; round <= kQmvTuneRounds && r == AC_SUCCESS; ++round) {
        for (uint32_t j = 0; j < per_round && r == AC_SUCCESS; ++j) {
            const uint32_t c = j < lead ? 0 : j - lead; // j == 0: the lead dispatch of the round
            if (round || j) r = ac_runtime_batch_barrier(batch);
            if (r != AC_SUCCESS) break;
            const AcBufferBinding ba = { owned.buf, 0, a_bytes }, bq = { owned.buf, a_bytes, q_bytes };
            if (with_i8 && j == 1) {
                const uint32_t pcq[6] = { 1u, K, 0u, K, 0u, U * fam.ad_units };
                const AcBufferBinding bindq[3] = { ba, bq, bq };
                r = ac_runtime_batch_dispatch_bindings(batch, quant, bindq, 3, pcq, sizeof(pcq), ameva_grid_1d(U, 64));
                continue;
            }
            const AmevaQmvConfig& cfg = cands[c].c;
            const AcBufferBinding bd = { owned.buf, o_base + c * d_bytes, d_bytes };
            const uint32_t rpw = qmv_rows_per_wg(cfg);
            if (cfg.i8) {
                const AcBufferBinding bind[8] = { w[0].b, w[1].b, w[2].b, bq, bq, bd, bd, bd };
                const uint32_t pc[12] = { K, nt, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2, 0u,
                                          U * fam.ad_units, 0u,
                                          (uint32_t)cfg.tpr, 0u, 0u, 0u };
                r = ac_runtime_batch_dispatch_bindings(batch, cands[c].k, bind, 8, pc, sizeof(pc),
                                                       { (nt + rpw - 1) / rpw, 1, 1 });
                continue;
            }
            const AcBufferBinding bind[7] = { w[0].b, w[1].b, w[2].b, ba, bd, ba, bd };
            const uint32_t pc[14] = { 1u, K, nt, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2, 0u, 0u, K, nt,
                                      cfg.mali ? 32u : (uint32_t)cfg.tpr, 0u, 0u, 0u };
            r = ac_runtime_batch_dispatch_bindings(batch, cands[c].k, bind, 7, pc, sizeof(pc), { (nt + rpw - 1) / rpw, 1, 1 });
        }
        if (r == AC_SUCCESS) r = ac_runtime_batch_flush(batch); // bounded submissions
    }
    if (r != AC_SUCCESS) {
        ac_runtime_batch_destroy(batch);
        return "GEMV measurement could not be recorded";
    }
    AcEvent* ev = nullptr;
    r = ac_runtime_batch_submit(batch, &ev);
    if (r != AC_SUCCESS || !ev) {
        ac_runtime_batch_destroy(batch);
        return "GEMV measurement submission failed";
    }
    ameva_increment_event_create_count();
    const AcResult wr = ac_runtime_event_wait(ev, 30000000000ull);
    std::vector<uint64_t> times(nd);
    uint32_t got = 0;
    const AcResult tr = wr == AC_SUCCESS ? ac_runtime_batch_get_dispatch_times(batch, times.data(), nd, &got) : wr;
    const AcResult er = ac_runtime_event_destroy(ev);
    ameva_increment_event_destroy_count();
    ac_runtime_batch_destroy(batch);
    if (wr != AC_SUCCESS || er != AC_SUCCESS) return "GEMV measurement did not complete";
    if (tr != AC_SUCCESS || got != nd) return "per-dispatch GPU times of the GEMV measurement unavailable";
    if (ac_runtime_buffer_invalidate(owned.buf, 0, size) != AC_SUCCESS) return "GEMV measurement buffer invalidate failed";

    const float* ref = (const float*)((const char*)map + o_base);
    double ref_max = 0.0;
    for (uint32_t i = 0; i < nt; ++i) {
        if (!std::isfinite(ref[i])) return "the default GEMV layout produced a non-finite value";
        ref_max = std::max(ref_max, (double)std::fabs(ref[i]));
    }
    std::vector<double> us(kQmvTuneRounds);
    double quant_us = 0.0;
    if (with_i8) {
        for (uint32_t round = 1; round <= kQmvTuneRounds; ++round) us[round - 1] = times[round * per_round + 1] / 1e3;
        std::sort(us.begin(), us.end());
        quant_us = us[kQmvTuneRounds / 2];
    }
    for (uint32_t c = 0; c < nc; ++c) {
        for (uint32_t round = 1; round <= kQmvTuneRounds; ++round) us[round - 1] = times[round * per_round + lead + c] / 1e3;
        std::sort(us.begin(), us.end());
        cands[c].median_us = us[kQmvTuneRounds / 2] + (cands[c].c.i8 ? quant_us : 0.0);
        if (c == 0) continue;
        const float* y = (const float*)((const char*)map + o_base + c * d_bytes);
        double diff = 0.0;
        bool finite = true;
        for (uint32_t i = 0; i < nt && finite; ++i) {
            finite = std::isfinite(y[i]);
            diff = std::max(diff, (double)std::fabs(y[i] - ref[i]));
        }
        const double tol = cands[c].c.i8 ? kMmvqTuneTolerance : kQmvTuneTolerance;
        if (!finite || diff > tol * ref_max + 1e-6) {
            cands[c].ok = false;
            fprintf(stderr, "ggml-ameva: WARNING: GEMV layout %s left out for %s %ux%u: its result differs from the "
                            "default layout's (%s, max difference %.3g, max |y| %.3g, allowed %.3g)\n",
                    qmv_config_str(cands[c].c, cs, sizeof(cs)), ggml_type_name(src0->type), N, K,
                    finite ? "finite" : "non-finite values", diff, ref_max, tol * ref_max);
        }
    }
    uint32_t best = 0;      // fastest f32 layout
    int best8 = -1;         // fastest ameva_mmvq layout
    for (uint32_t c = 1; c < nc; ++c) {
        if (!cands[c].ok) continue;
        if (cands[c].c.i8) {
            if (best8 < 0 || cands[c].median_us < cands[best8].median_us) best8 = (int)c;
        } else if (cands[c].median_us < cands[best].median_us) {
            best = c;
        }
    }
    const uint32_t chosen = best != 0 && cands[best].median_us < (1.0 - kQmvTuneMargin) * cands[0].median_us ? best : 0;
    const bool take8 = best8 >= 0 && cands[best8].median_us < (1.0 - kMmvqTuneMargin) * cands[chosen].median_us;
    *out = { (int)src0->type, K, N, cands[chosen].c, cands[chosen].median_us, cands[0].median_us, false,
             (uint8_t)(!with_i8 ? AMEVA_I8_UNMEASURED : take8 ? AMEVA_I8_CHOSEN : AMEVA_I8_MEASURED),
             best8 >= 0 ? cands[best8].c : AmevaQmvConfig{ 0, 4, 32, 1 }, best8 >= 0 ? cands[best8].median_us : 0.0 };
    double mali_us = 0.0;
    for (const Cand& c : cands) {
        if (c.c.mali) mali_us = c.median_us;
    }
    char c8[48] = "", i8note[96] = "";
    if (with_i8 && best8 >= 0) {
        snprintf(i8note, sizeof(i8note), ", %s %.1f us with %.1f us of quantization",
                 qmv_config_str(cands[best8].c, c8, sizeof(c8)), cands[best8].median_us, quant_us);
    }
    fprintf(stderr, "ggml-ameva: GEMV layout for %s %ux%u: %s (f32 %s %.1f us, default %.1f us, mali_qmv %.1f us%s; "
                    "%u layouts, %u of %u rows timed)\n",
            ggml_type_name(src0->type), N, K, take8 ? "ameva_mmvq" : "f32", qmv_config_str(cands[chosen].c, cs, sizeof(cs)),
            cands[chosen].median_us, cands[0].median_us, mali_us, i8note, nc, nt, N);
    if (ameva_profile_level() >= 1) {
        for (const Cand& c : cands) {
            fprintf(stderr, "AMEVA_PROFILE qmv_tune type=%s N=%u K=%u layout=\"%s\" median_us=%.2f%s\n",
                    ggml_type_name(src0->type), N, K, qmv_config_str(c.c, cs, sizeof(cs)), c.median_us,
                    c.ok ? "" : " (left out: result differs)");
        }
    }
    return nullptr;
}

// Before a graph is recorded: measures the layout of every one-row MUL_MAT weight shape not measured yet (once per
// shape on a device; see above). Caller holds s_compute_mutex.
static const char* ameva_qmv_tune_graph(struct ggml_cgraph* cgraph, int n_nodes) {
    if (!ameva_qmv_policy().tune) return nullptr;
    const AmevaMmvqPolicy& mp = ameva_mmvq_policy();
    std::lock_guard<std::mutex> lock(s_tune_mutex);
    ameva_qmv_tune_file_read();
    int measured = 0;
    for (int i = 0; i < n_nodes; i++) {
        const struct ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (!ameva_qmv_one_row(node)) continue;
        const struct ggml_tensor* src0 = node->src[0];
        // AMEVA_MMVQ=1 fixes the layout of the types its family has
        if (mp.mode == AMEVA_MMVQ_FORCED && ameva_mmvq_find(src0->type, mp.rows) >= 0) continue;
        const bool with_i8 = ameva_mmvq_measured() && ameva_mmvq_find(src0->type, 4) >= 0;
        // a shape measured without the ameva_mmvq layouts (AMEVA_MMVQ=0 at the time) is measured again with them
        AmevaQmvShape* have = ameva_qmv_shape_find((int)src0->type, (uint32_t)src0->ne[0], (uint32_t)src0->ne[1]);
        if (have && !(with_i8 && have->i8_state == AMEVA_I8_UNMEASURED)) continue;
        AmevaQmvShape s = {};
        if (const char* e = ameva_qmv_tune_shape(node, &s)) return e;
        if (have) *have = s;
        else s_qmv_shapes.push_back(s);
        ++measured;
    }
    if (measured) ameva_qmv_tune_file_write();
    return nullptr;
}

// AMEVA_PROFILE: the layout of every weight shape measured or read from the tune file in this process
static void ameva_qmv_layout_report(void) {
    std::lock_guard<std::mutex> lock(s_tune_mutex);
    char cs[48];
    const bool with_i8 = ameva_mmvq_policy().mode == AMEVA_MMVQ_AUTO;
    for (const AmevaQmvShape& s : s_qmv_shapes) {
        const bool take8 = with_i8 && s.i8_state == AMEVA_I8_CHOSEN;
        char c8[48] = "-";
        if (s.i8_state != AMEVA_I8_UNMEASURED) qmv_config_str(s.i8, c8, sizeof(c8));
        fprintf(stderr, "AMEVA_PROFILE qmv_layout type=%s N=%u K=%u layout=\"%s\" us=%.1f default_us=%.1f f32=\"%s\" "
                        "f32_us=%.1f i8=\"%s\" i8_us=%.1f source=%s\n",
                ggml_type_name((enum ggml_type)s.type), s.N, s.K, take8 ? c8 : qmv_config_str(s.config, cs, sizeof(cs)),
                take8 ? s.i8_us : s.us, s.default_us, qmv_config_str(s.config, cs, sizeof(cs)), s.us, c8, s.i8_us,
                s.from_file ? "tune_file" : "measured");
    }
}

// MUL_MAT. Weights are multiplied in slices of rows that fit one binding (per region for repacked weights) and at most
// 65535 workgroups; activation rows in chunks (see below). Repacked weights bind their Q, H and S regions.
static const char* enc_mul_mat(AmevaRecorder& rec, const struct ggml_tensor* node,
                               const AmevaEpilogue* ep = nullptr) {
    {
        uint32_t chunk_rows = 0;
        uint64_t scratch_bytes = 0;
        if (ameva_mmq_plan(node, &chunk_rows, &scratch_bytes)) {
            if (ep) return "epilogue on a prompt-sized MUL_MAT";
            return enc_mul_mat_mmq(rec, node, chunk_rows);
        }
    }
    const struct ggml_tensor* src0 = node->src[0];
    const struct ggml_tensor* src1 = node->src[1];
    const struct ggml_tensor* base = ameva_repacked_base(src0);
    const uint32_t K = (uint32_t)src0->ne[0], N = (uint32_t)src0->ne[1], M = (uint32_t)src1->ne[1];
    const uint32_t ncols = M > 1 ? 8u : 1u;
    const size_t row_bytes = ggml_row_size(src0->type, K);
    AcKernel* kernel = nullptr;
    AmevaRepackLayout L = {};
    AmevaRepackRegions R = {};
    uint64_t first_row = 0;     // row of `base` where src0 starts (src0 may be a row range of a repacked weight)
    uint32_t rows_per_wg = 4;   // ameva_qmm: 4 rows x 16 lanes
    uint32_t cols_per_wg = ncols;
    uint32_t tpr = 0;
    uint64_t slice_row_bytes = row_bytes;
    const bool gemm = base && M >= ameva_qgemm_min_rows();
    if (ep && (!base || gemm || M != 1)) return "epilogue outside the one-row ameva_qmv path";
    AmevaBind be = {}, beo = {};
    if (ep) {
        const char* ee = ameva_bind(ep->e, 4, &be);
        if (!ee) ee = ameva_bind(ep->out, 4, &beo);
        if (ee) return ee;
    }
    if (base) {
        if (gemm) {
            const int gi = ameva_qgemm_find(src0->type);
            kernel = gi >= 0 ? ameva_qgemm_kernel(gi) : nullptr;
            if (!kernel) return "no ameva_qgemm kernel";
            rows_per_wg = 64; // shaders/ameva_qgemm.comp tile: 64 weight rows x 32 activation rows
            cols_per_wg = 32;
        } else if (ncols == 1) {
            // token generation: the layout chosen for this shape (measured, or fixed by the environment)
            AmevaQmvConfig cfg;
            if (!ameva_qmv_choice(node, &cfg)) return "token-generation GEMV layout of this weight shape not measured";
            if (cfg.i8) return enc_mul_mat_mmvq(rec, node, ep, cfg.rows, cfg.tpr);
            if (cfg.mali) {
                const int mvi = ameva_mali_qmv_find(src0->type);
                kernel = mvi >= 0 ? ameva_mali_qmv_kernel(mvi) : nullptr;
                if (!kernel) return "no mali_qmv kernel";
            } else {
                const int vi = ameva_qmv_find(src0->type, 1, cfg.rows);
                kernel = vi >= 0 ? ameva_qmv_kernel(vi) : nullptr;
                if (!kernel) return "no ameva_qmv kernel";
            }
            tpr = cfg.mali ? 32u : cfg.tpr;
            rows_per_wg = qmv_rows_per_wg(cfg);
        } else {
            const int vi = ameva_qmv_find(src0->type, ncols, 1);
            kernel = vi >= 0 ? ameva_qmv_kernel(vi) : nullptr;
            if (!kernel) return "no ameva_qmv kernel";
            tpr = ameva_qmv_tpr(K / 32);
            rows_per_wg = (64u / tpr) * kAmevaQmvVariants[vi].rows;
        }
        if (!ameva_repack_layout(base->type, &L) || !ameva_repack_regions(base->type, base->ne[0], ggml_nrows(base), &R))
            return "repacked weight without a layout";
        first_row = (uint64_t)((const char*)src0->data - (const char*)base->data) / row_bytes;
        slice_row_bytes = R.q_row; // the largest of the three regions per row
    } else {
        const int vi = ameva_qmm_find(src0->type, ncols);
        kernel = vi >= 0 ? ameva_qmm_kernel(vi) : nullptr;
        if (!kernel) return "no ameva_qmm kernel";
    }
    // AMEVA_ACT_FQ: the activations rounded to int8 per block into the scratch buffer; the kernels below read them
    // from there (contiguous rows of K)
    const uint32_t fq_blk = ameva_act_fq_blk(node);
    const uint32_t s1_src = st(src1, 1);
    AmevaBind bfq = {};
    if (fq_blk) {
        if (!s_scratch || (uint64_t)M * K * 4 > s_scratch_bytes) return "AMEVA_ACT_FQ scratch not reserved";
        AcKernel* fq = ameva_op_kernel(AMEVA_K_ACT_FQ);
        if (!fq) return "no ameva_act_fq kernel";
        AmevaBind ba;
        const char* ef = ameva_bind_span(src1, 0, ((uint64_t)(M - 1) * s1_src + K) * 4, 4, &ba);
        if (ef) return ef;
        if (ba.elem % 4 != 0) return "activation offset not a multiple of 16 bytes";
        bfq.b.buffer = s_scratch;
        bfq.b.offset = 0;
        bfq.b.range = std::min<uint64_t>(s_scratch_bytes, s_limits.max_storage_buffer_range) & ~3ull;
        bfq.key = s_scratch;
        bfq.lo = 0;
        bfq.hi = (uint64_t)M * K * 4;
        if (bfq.hi > bfq.b.range) return "AMEVA_ACT_FQ activations larger than one binding";
        const uint32_t pcf[6] = { M, K, ba.elem, s1_src, 0u, fq_blk };
        const AcBufferBinding bindf[2] = { ba.b, bfq.b };
        ef = ameva_record(rec, fq, &ba, 1, &bfq, 1, bindf, 2, pcf, sizeof(pcf),
                          ameva_grid_1d((uint64_t)M * (K / fq_blk), 64), (double)M * K);
        if (ef) return ef;
    }
    const char* e = nullptr;
    const uint64_t span = ameva_max_binding_span();
    uint64_t slice_cap = std::min<uint64_t>(span / slice_row_bytes, 65535ull * rows_per_wg);
    if (gemm) {
        // a dispatch covers every activation row, so its multiply-accumulates are bounded by its weight rows (whole
        // 64-row tiles)
        const uint64_t mac_rows = (uint64_t)(kDispatchMac / ((double)M * (double)K));
        slice_cap = std::min<uint64_t>(slice_cap, std::max<uint64_t>(64, mac_rows & ~63ull));
    }
    const uint32_t rows_per_slice = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)N, slice_cap));
    const uint32_t s1 = fq_blk ? K : s1_src, sd = st(node, 1);
    // activation/output rows per dispatch: at most kDispatchMac multiply-accumulates and one binding for the rows of
    // src1 and of dst it touches; whole 8-row tiles when the 8-column kernel runs
    const uint64_t rows_fit = std::max<uint64_t>(1, std::min<uint64_t>(span / ((uint64_t)s1 * 4), span / ((uint64_t)sd * 4)));
    for (uint32_t n0 = 0; n0 < N; n0 += rows_per_slice) {
        const uint32_t nc = std::min(rows_per_slice, N - n0);
        AmevaBind w[3]; // ggml layout: w[0] = the rows; repacked: Q, H, S of the rows
        if (base) {
            const uint64_t row = first_row + n0;
            // vector elements of the Q and H bindings (shaders/ameva_qmv.comp QVEC, HVEC); S in bytes, read at even
            // addresses
            e = ameva_bind_span(base, R.q + row * R.q_row, (uint64_t)nc * R.q_row, L.qb == 8 ? 8 : 16, &w[0]);
            if (!e) e = ameva_bind_span(base, R.s + row * R.s_row, (uint64_t)nc * R.s_row, 2, &w[2]);
            if (!e) {
                if (R.h_row) e = ameva_bind_span(base, R.h + row * R.h_row, (uint64_t)nc * R.h_row, L.hb == 8 ? 8 : 4, &w[1]);
                else w[1] = w[2]; // no high bits: binding 1 is not read
            }
        } else {
            // weights are addressed in bytes (every ggml block size is even, the kernel reads at even addresses)
            e = ameva_bind_span(src0, (uint64_t)n0 * row_bytes, (uint64_t)nc * row_bytes, 2, &w[0]);
        }
        if (e) return e;
        uint64_t mc_max = std::min<uint64_t>(rows_fit, (uint64_t)(kDispatchMac / ((double)nc * (double)K)));
        mc_max = ncols == 1 ? 1 : (mc_max >= cols_per_wg ? mc_max - mc_max % cols_per_wg : mc_max);
        if (gemm) mc_max = std::min<uint64_t>(rows_fit, 65535ull * 32); // bounded by binding and grid only
        mc_max = std::max<uint64_t>(1, mc_max);
        for (uint32_t m0 = 0; m0 < M; m0 += (uint32_t)std::min<uint64_t>(mc_max, M)) {
            const uint32_t mc = (uint32_t)std::min<uint64_t>(mc_max, M - m0);
            // src1 rows [m0, m0 + mc) and columns [n0, n0 + nc) of the same dst rows, each bound as its own range
            AmevaBind b1, bd;
            if (fq_blk) { // rows [m0, m0 + mc) of the rounded activations in the scratch buffer
                b1 = bfq;
                b1.elem = m0 * K;
                b1.lo = (uint64_t)m0 * K * 4;
                b1.hi = (uint64_t)(m0 + mc) * K * 4;
                e = nullptr;
            } else {
                e = ameva_bind_span(src1, (uint64_t)m0 * s1 * 4, ((uint64_t)(mc - 1) * s1 + K) * 4, 4, &b1);
            }
            if (!e) e = ameva_bind_span(node, ((uint64_t)m0 * sd + n0) * 4, ((uint64_t)(mc - 1) * sd + nc) * 4, 4, &bd);
            if (e) return e;
            if (b1.elem % 4 != 0) return "activation offset not a multiple of 16 bytes";
            const AcDim3 grid = { (nc + rows_per_wg - 1) / rows_per_wg, (mc + cols_per_wg - 1) / cols_per_wg, 1 };
            const double mac = (double)mc * nc * K;
            if (gemm) {
                const uint32_t pc[10] = { mc, K, nc, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2,
                                          b1.elem, bd.elem, s1, sd };
                const AcBufferBinding bind[5] = { w[0].b, w[1].b, w[2].b, b1.b, bd.b };
                const AmevaBind reads[4] = { w[0], w[1], w[2], b1 };
                e = ameva_record(rec, kernel, reads, 4, &bd, 1, bind, 5, pc, sizeof(pc), grid, mac);
            } else if (base) {
                const uint32_t pc[14] = { mc, K, nc, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2,
                                          b1.elem, bd.elem, s1, sd, tpr, ep ? ep->mode : 0u,
                                          ep ? be.elem + n0 : 0u, ep ? beo.elem + n0 : 0u };
                const AcBufferBinding bind[7] = { w[0].b, w[1].b, w[2].b, b1.b, bd.b, ep ? be.b : b1.b,
                                                  ep ? beo.b : bd.b };
                const AmevaBind reads[5] = { w[0], w[1], w[2], b1, be };
                const AmevaBind writes[2] = { bd, beo };
                e = ameva_record(rec, kernel, reads, ep ? 5 : 4, writes, ep ? 2 : 1, bind, 7, pc, sizeof(pc), grid,
                                 mac);
            } else {
                const uint32_t pc[8] = { mc, K, nc, w[0].elem * 2, b1.elem, bd.elem, s1, sd };
                const AcBufferBinding bind[3] = { w[0].b, b1.b, bd.b };
                const AmevaBind reads[2] = { w[0], b1 };
                e = ameva_record(rec, kernel, reads, 2, &bd, 1, bind, 3, pc, sizeof(pc), grid, mac);
            }
            if (e) return e;
        }
    }
    rec.mul_mat_nodes++;
    if (base) rec.qmv_nodes++;
    return nullptr;
}

static const char* enc_binary(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *b = node->src[1];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_BINARY);
    if (!kernel) return "no ameva_binary kernel";
    AmevaBind ba, bb, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(b, 4, &bb);
    if (!e) e = ameva_bind(node, 4, &bd);
    if (e) return e;
    const uint32_t op = node->op == GGML_OP_ADD ? 0u : node->op == GGML_OP_SUB ? 1u : node->op == GGML_OP_MUL ? 2u : 3u;
    const uint32_t pc[21] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2], (uint32_t)node->ne[3],
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bb.elem, (uint32_t)b->ne[0], (uint32_t)b->ne[1], (uint32_t)b->ne[2],
                              (uint32_t)b->ne[3], st(b, 1), st(b, 2), st(b, 3),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3), op };
    const AcBufferBinding bind[3] = { ba.b, bb.b, bd.b };
    const AmevaBind reads[2] = { ba, bb };
    return ameva_record(rec, kernel, reads, 2, &bd, 1, bind, 3, pc, sizeof(pc), ameva_grid_1d(ggml_nelements(node), 256),
                        (double)ggml_nelements(node));
}

static const char* enc_unary(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor* a = node->src[0];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_UNARY);
    if (!kernel) return "no ameva_unary kernel";
    uint32_t op = 0, p0 = 0, p1 = 0;
    if (node->op == GGML_OP_SCALE) {
        p0 = fbits(op_param_f32(node, 0));
        p1 = fbits(op_param_f32(node, 1));
    } else {
        op = (uint32_t)unary_code(node);
    }
    AmevaBind ba, bd;
    const char* e = nullptr;
    if (fits_binding(a) && fits_binding(node)) {
        e = ameva_bind(a, 4, &ba);
        if (!e) e = ameva_bind(node, 4, &bd);
        if (e) return e;
        const uint32_t pc[15] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2],
                                  (uint32_t)node->ne[3], ba.elem, st(a, 1), st(a, 2), st(a, 3),
                                  bd.elem, st(node, 1), st(node, 2), st(node, 3), op, p0, p1 };
        const AcBufferBinding bind[2] = { ba.b, bd.b };
        return ameva_record(rec, kernel, &ba, 1, &bd, 1, bind, 2, pc, sizeof(pc),
                            ameva_grid_1d(ggml_nelements(node), 256), (double)ggml_nelements(node));
    }
    // row chunks (both tensors rows_linear, checked by the contract): rows [r0, r0 + rc) as a 2-D view
    const uint64_t span = ameva_max_binding_span();
    const uint64_t n0 = (uint64_t)node->ne[0], rows = (uint64_t)ggml_nrows(node);
    const uint64_t sa = st(a, 1), sd = st(node, 1);
    const uint64_t rc_max = std::max<uint64_t>(1, std::min<uint64_t>(span / (sa * 4), span / (sd * 4)));
    for (uint64_t r0 = 0; r0 < rows; r0 += rc_max) {
        const uint64_t rc = std::min<uint64_t>(rc_max, rows - r0);
        e = ameva_bind_span(a, r0 * sa * 4, ((rc - 1) * sa + n0) * 4, 4, &ba);
        if (!e) e = ameva_bind_span(node, r0 * sd * 4, ((rc - 1) * sd + n0) * 4, 4, &bd);
        if (e) return e;
        const uint32_t pc[15] = { (uint32_t)n0, (uint32_t)rc, 1u, 1u, ba.elem, (uint32_t)sa, 0u, 0u,
                                  bd.elem, (uint32_t)sd, 0u, 0u, op, p0, p1 };
        const AcBufferBinding bind[2] = { ba.b, bd.b };
        e = ameva_record(rec, kernel, &ba, 1, &bd, 1, bind, 2, pc, sizeof(pc), ameva_grid_1d(n0 * rc, 256),
                         (double)(n0 * rc));
        if (e) return e;
    }
    return nullptr;
}

static const char* enc_glu(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *g = node->src[1];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_GLU);
    if (!kernel) return "no ameva_glu kernel";
    AmevaBind ba, bg, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(g ? g : a, 4, &bg);
    if (!e) e = ameva_bind(node, 4, &bd);
    if (e) return e;
    uint32_t a_off = ba.elem, g_off = bg.elem;
    const struct ggml_tensor* gs = g ? g : a;
    if (!g) {
        const uint32_t nc = (uint32_t)node->ne[0];
        const bool swapped = ((const int32_t*)node->op_params)[1] != 0;
        a_off = ba.elem + (swapped ? nc : 0);
        g_off = ba.elem + (swapped ? 0 : nc);
    }
    const uint32_t pc[17] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2], (uint32_t)node->ne[3],
                              a_off, st(a, 1), st(a, 2), st(a, 3),
                              g_off, st(gs, 1), st(gs, 2), st(gs, 3),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3), (uint32_t)glu_code(node) };
    const AcBufferBinding bind[3] = { ba.b, bg.b, bd.b };
    const AmevaBind reads[2] = { ba, bg };
    return ameva_record(rec, kernel, reads, 2, &bd, 1, bind, 3, pc, sizeof(pc), ameva_grid_1d(ggml_nelements(node), 256),
                        (double)ggml_nelements(node));
}

// RMS_NORM; with mul (the following MUL node, fused) also mul = node * w for a norm weight w of one row
static const char* enc_rms_norm(AmevaRecorder& rec, const struct ggml_tensor* node,
                                const struct ggml_tensor* mul = nullptr, const struct ggml_tensor* w = nullptr) {
    const struct ggml_tensor* a = node->src[0];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_RMS_NORM);
    if (!kernel) return "no ameva_rms_norm kernel";
    AmevaBind ba, bd, bw = {}, bo = {};
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(node, 4, &bd);
    if (!e && mul) e = ameva_bind(w, 4, &bw);
    if (!e && mul) e = ameva_bind(mul, 4, &bo);
    if (e) return e;
    const uint64_t nrows = (uint64_t)node->ne[1] * node->ne[2] * node->ne[3];
    const uint32_t pc[19] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2], (uint32_t)nrows,
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3), fbits(op_param_f32(node, 0)),
                              mul ? 1u : 0u, mul ? bw.elem : 0u, mul ? bo.elem : 0u,
                              mul ? st(mul, 1) : 0u, mul ? st(mul, 2) : 0u, mul ? st(mul, 3) : 0u };
    const AcBufferBinding bind[4] = { ba.b, bd.b, mul ? bw.b : ba.b, mul ? bo.b : bd.b };
    const AmevaBind reads[2] = { ba, bw };
    const AmevaBind writes[2] = { bd, bo };
    return ameva_record(rec, kernel, reads, mul ? 2 : 1, writes, mul ? 2 : 1, bind, 4, pc, sizeof(pc),
                        ameva_grid_groups(nrows), (mul ? 3.0 : 2.0) * (double)ggml_nelements(node));
}

static const char* enc_rope(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *pos = node->src[1], *ff = node->src[2];
    const bool f16 = a->type == GGML_TYPE_F16; // shaders/ameva_rope.comp -DF16: offsets and strides in f16 elements
    AcKernel* kernel = ameva_op_kernel(f16 ? AMEVA_K_ROPE_F16 : AMEVA_K_ROPE);
    if (!kernel) return "no ameva_rope kernel";
    AmevaBind ba, bp, bf, bd;
    const char* e = ameva_bind(a, f16 ? 2 : 4, &ba);
    if (!e) e = ameva_bind(pos, 4, &bp);
    if (!e) e = ameva_bind(ff ? ff : pos, 4, &bf); // any valid range when there are no frequency factors
    if (!e) e = ameva_bind(node, f16 ? 2 : 4, &bd);
    if (e) return e;
    const int32_t* p = (const int32_t*)node->op_params;
    const int n_dims = p[1], mode = p[2], n_ctx_orig = p[4];
    const float freq_base = op_param_f32(node, 5), freq_scale = op_param_f32(node, 6);
    const float ext_factor = op_param_f32(node, 7), attn_factor = op_param_f32(node, 8);
    const float beta_fast = op_param_f32(node, 9), beta_slow = op_param_f32(node, 10);
    float corr[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
    // the same expression as ggml-cpu; unused when n_dims == 0 (no pair is rotated)
    const float theta_scale = n_dims > 0 ? powf(freq_base, -2.0f / n_dims) : 1.0f;
    const uint32_t pc[23] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2], (uint32_t)node->ne[3],
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3),
                              bp.elem, ff ? bf.elem : 0u, (uint32_t)n_dims, mode == 2 ? 1u : 0u,
                              ff ? 1u : 0u, fbits(theta_scale), fbits(freq_scale), fbits(ext_factor),
                              fbits(attn_factor), fbits(corr[0]), fbits(corr[1]) };
    const AcBufferBinding bind[4] = { ba.b, bp.b, bf.b, bd.b };
    const AmevaBind reads[3] = { ba, bp, bf };
    return ameva_record(rec, kernel, reads, 3, &bd, 1, bind, 4, pc, sizeof(pc),
                        ameva_grid_1d((uint64_t)ggml_nelements(node) / 2, 256), (double)ggml_nelements(node) * 8.0);
}

static AmevaBind ameva_status_bind(void) {
    AmevaBind s = {};
    s.b.buffer = s_status;
    s.b.offset = 0;
    s.b.range = 256;
    s.key = s_status;
    s.lo = 0;
    s.hi = 8;
    return s;
}

// f32 rows -> q8_0 / q4_0 blocks with ameva_set_rows_q: SET_ROWS (idx given) or a same-shape CPY (idx == nullptr)
static const char* enc_store_quantized(AmevaRecorder& rec, const struct ggml_tensor* node,
                                       const struct ggml_tensor* a, const struct ggml_tensor* idx) {
    AcKernel* kernel = ameva_op_kernel(node->type == GGML_TYPE_Q8_0 ? AMEVA_K_SET_ROWS_Q8_0 : AMEVA_K_SET_ROWS_Q4_0);
    if (!kernel) return "no ameva_set_rows_q kernel";
    AmevaBind ba, bi, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e && idx) e = ameva_bind(idx, 4, &bi);
    if (!e) e = ameva_bind(node, 2, &bd); // dst addressed in bytes: elem * 2
    if (e) return e;
    if (!idx) bi = ba; // binding 1 is not read in identity mode
    const AmevaBind bs = ameva_status_bind();
    const uint32_t mode = !idx ? 0u : (idx->type == GGML_TYPE_I64 ? 2u : 1u);
    const uint32_t nb = (uint32_t)(a->ne[0] / 32);
    const uint32_t pc[21] = { nb, (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3],
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              idx ? bi.elem : 0u, idx ? (uint32_t)(idx->nb[0] / 4) : 0u,
                              idx ? (uint32_t)(idx->nb[1] / 4) : 0u, idx ? (uint32_t)(idx->nb[2] / 4) : 0u,
                              idx ? (uint32_t)idx->ne[1] : 1u, idx ? (uint32_t)idx->ne[2] : 1u, mode,
                              (uint32_t)node->ne[1],
                              bd.elem * 2u, (uint32_t)node->nb[1], (uint32_t)node->nb[2], (uint32_t)node->nb[3],
                              (uint32_t)rec.node };
    const AcBufferBinding bind[4] = { ba.b, bi.b, bs.b, bd.b };
    const AmevaBind reads[2] = { ba, bi };
    const uint64_t blocks = (uint64_t)nb * a->ne[1] * a->ne[2] * a->ne[3];
    return ameva_record(rec, kernel, reads, idx ? 2 : 1, &bd, 1, bind, 4, pc, sizeof(pc), ameva_grid_1d(blocks, 64),
                        (double)ggml_nelements(a));
}

static const char* enc_set_rows(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *idx = node->src[1];
    if (is_kvq(node)) return enc_store_quantized(rec, node, a, idx);
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_SET_ROWS);
    if (!kernel) return "no ameva_set_rows kernel";
    const bool f16 = node->type == GGML_TYPE_F16;
    AmevaBind ba, bi, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(idx, 4, &bi);
    if (!e) e = ameva_bind(node, f16 ? 2 : 4, &bd);
    if (e) return e;
    const AmevaBind bs = ameva_status_bind();
    const uint32_t idx64 = idx->type == GGML_TYPE_I64 ? 1u : 0u;
    const uint32_t pc[22] = { (uint32_t)a->ne[0], (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3],
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bi.elem, (uint32_t)(idx->nb[0] / 4), (uint32_t)(idx->nb[1] / 4), (uint32_t)(idx->nb[2] / 4),
                              (uint32_t)idx->ne[1], (uint32_t)idx->ne[2], idx64, (uint32_t)node->ne[1],
                              bd.elem, st(node, 1), st(node, 2), st(node, 3), f16 ? 1u : 0u, (uint32_t)rec.node };
    const AcBufferBinding bind[4] = { ba.b, bi.b, bs.b, bd.b };
    const AmevaBind reads[2] = { ba, bi };
    // the status buffer is only touched by atomics (order-independent); it is not a hazard between dispatches
    return ameva_record(rec, kernel, reads, 2, &bd, 1, bind, 4, pc, sizeof(pc), ameva_grid_1d(ggml_nelements(a), 256),
                        (double)ggml_nelements(a));
}

static const char* enc_get_rows(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *idx = node->src[1];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_GET_ROWS);
    if (!kernel) return "no ameva_get_rows kernel";
    const bool f16 = a->type == GGML_TYPE_F16;
    AmevaBind ba, bi, bd;
    const char* e = ameva_bind(a, f16 ? 2 : 4, &ba);
    if (!e) e = ameva_bind(idx, 4, &bi);
    if (!e) e = ameva_bind(node, 4, &bd);
    if (e) return e;
    const AmevaBind bs = ameva_status_bind();
    const uint32_t pc[19] = { (uint32_t)node->ne[0], (uint32_t)idx->ne[0], (uint32_t)idx->ne[1], (uint32_t)idx->ne[2],
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bi.elem, st(idx, 0), st(idx, 1), st(idx, 2),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3),
                              f16 ? 1u : 0u, (uint32_t)a->ne[1], (uint32_t)rec.node };
    const AcBufferBinding bind[4] = { ba.b, bi.b, bs.b, bd.b };
    const AmevaBind reads[2] = { ba, bi };
    return ameva_record(rec, kernel, reads, 2, &bd, 1, bind, 4, pc, sizeof(pc), ameva_grid_1d(ggml_nelements(node), 256),
                        (double)ggml_nelements(node));
}

static const char* enc_cpy(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor* a = node->src[0];
    if (is_kvq(node)) return enc_store_quantized(rec, node, a, nullptr);
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_CPY);
    if (!kernel) return "no ameva_cpy kernel";
    const int stype = kv_type_code(a);
    const bool sq = stype >= 2, df16 = node->type == GGML_TYPE_F16;
    AmevaBind ba, bd;
    const char* e = ameva_bind(a, (stype == 0) ? 4 : 2, &ba); // f16 in halves; quantized in bytes (elem * 2)
    if (!e) e = ameva_bind(node, df16 ? 2 : 4, &bd);
    if (e) return e;
    // a quantized source is addressed in bytes: offset and strides in bytes, a_s0 = block size
    const uint32_t pc[19] = { (uint32_t)a->ne[0], (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3],
                              sq ? ba.elem * 2u : ba.elem, sq ? (uint32_t)a->nb[0] : st(a, 0),
                              sq ? (uint32_t)a->nb[1] : st(a, 1), sq ? (uint32_t)a->nb[2] : st(a, 2),
                              sq ? (uint32_t)a->nb[3] : st(a, 3), (uint32_t)node->ne[0], (uint32_t)node->ne[1],
                              (uint32_t)node->ne[2], bd.elem, st(node, 0), st(node, 1), st(node, 2),
                              st(node, 3), (uint32_t)stype, df16 ? 1u : 0u };
    const AcBufferBinding bind[2] = { ba.b, bd.b };
    return ameva_record(rec, kernel, &ba, 1, &bd, 1, bind, 2, pc, sizeof(pc), ameva_grid_1d(ggml_nelements(a), 256),
                        (double)ggml_nelements(a));
}

// Keys per split of flash attention (0: no split). Token generation puts every row of a KV head into one tile, so
// without splits each KV head's cache is walked by one workgroup; with splits the keys are spread over workgroups and
// ameva_fa_combine merges them. Split when the tiles alone give fewer than 16 workgroups and the cache has at least two
// splits. AMEVA_FA_SPLIT_KEYS=<keys> (a multiple of 64, or 0 = never split) overrides the 256 keys per split.
static uint32_t ameva_fa_split_keys(const struct ggml_tensor* node, uint32_t* n_split) {
    static const uint32_t forced = [] {
        const char* e = getenv("AMEVA_FA_SPLIT_KEYS");
        if (!e || !*e) return 256u;
        char* end = nullptr;
        const unsigned long v = strtoul(e, &end, 10);
        if (!end || *end || v % 64 || v > (1ul << 24)) {
            GGML_ABORT("ggml-ameva: AMEVA_FA_SPLIT_KEYS must be 0 or a multiple of 64 (got \"%s\")", e);
        }
        return (uint32_t)v;
    }();
    *n_split = 1;
    const struct ggml_tensor *q = node->src[0], *k = node->src[1], *v = node->src[2];
    const int fi = ameva_fa_find((uint32_t)q->ne[0], (uint32_t)v->ne[0]);
    if (forced == 0 || fi < 0) return 0;
    const uint64_t n_head = (uint64_t)q->ne[2], n_head_kv = (uint64_t)k->ne[2], n_kv = (uint64_t)k->ne[1];
    const uint64_t tiles = ((uint64_t)q->ne[1] * (n_head / n_head_kv) + kAmevaFaVariants[fi].bq - 1) /
                           kAmevaFaVariants[fi].bq;
    const uint64_t splits = (n_kv + forced - 1) / forced;
    if (tiles * n_head_kv >= 16 || splits < 2 || splits > 65535) return 0;
    // the partials of every split must fit one binding
    const uint64_t bytes = splits * (uint64_t)q->ne[1] * n_head * ((uint64_t)v->ne[0] + 2) * 4;
    if (bytes > ameva_max_binding_span()) return 0;
    *n_split = (uint32_t)splits;
    return forced;
}

static uint64_t ameva_fa_scratch_bytes(const struct ggml_tensor* node) {
    uint32_t n_split = 1;
    if (!ameva_fa_split_keys(node, &n_split)) return 0;
    return (uint64_t)n_split * (uint64_t)node->src[0]->ne[1] * (uint64_t)node->src[0]->ne[2] *
           ((uint64_t)node->src[2]->ne[0] + 2) * 4;
}

static const char* enc_flash_attn(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *q = node->src[0], *k = node->src[1], *v = node->src[2], *m = node->src[3];
    const int fi = ameva_fa_find((uint32_t)q->ne[0], (uint32_t)v->ne[0]);
    AcKernel* kernel = fi >= 0 ? ameva_fa_kernel(fi) : nullptr;
    if (!kernel) return "no ameva_flash_attn kernel";
    const AmevaFaVariant& fv = kAmevaFaVariants[fi];
    // K / V: f32 in words, f16 in halves, q8_0 / q4_0 in bytes (bound in halves: elem * 2)
    const int kt = kv_type_code(k), vt = kv_type_code(v);
    AmevaBind bq, bk, bv, bm, bo;
    const char* e = ameva_bind(q, 4, &bq);
    if (!e) e = ameva_bind(k, kt == 0 ? 4 : 2, &bk);
    if (!e) e = ameva_bind(v, vt == 0 ? 4 : 2, &bv);
    if (!e) e = ameva_bind(m ? m : q, m ? 2 : 4, &bm);
    if (!e) e = ameva_bind(node, 4, &bo);
    if (e) return e;
    const float max_bias = op_param_f32(node, 1);
    const uint32_t n_head = (uint32_t)q->ne[2], n_head_kv = (uint32_t)k->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2((double)n_head));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    uint32_t n_split = 1;
    const uint32_t split_keys = ameva_fa_split_keys(node, &n_split);
    uint32_t pc[31] = { (uint32_t)q->ne[0], (uint32_t)v->ne[0], (uint32_t)k->ne[1], (uint32_t)q->ne[1],
                        n_head, n_head_kv, (uint32_t)kt | ((uint32_t)vt << 4u), m ? 1u : 0u,
                        bq.elem, st(q, 1), st(q, 2), kt >= 2 ? bk.elem * 2u : bk.elem,
                        kt >= 2 ? (uint32_t)k->nb[1] : st(k, 1), kt >= 2 ? (uint32_t)k->nb[2] : st(k, 2),
                        vt >= 2 ? bv.elem * 2u : bv.elem, vt >= 2 ? (uint32_t)v->nb[1] : st(v, 1),
                        vt >= 2 ? (uint32_t)v->nb[2] : st(v, 2), m ? bm.elem : 0u, m ? st(m, 1) : 0u, m ? st(m, 2) : 0u,
                        m ? (uint32_t)m->ne[2] : 1u, bo.elem, fbits(op_param_f32(node, 0)), fbits(op_param_f32(node, 2)),
                        fbits(max_bias), fbits(m0), fbits(m1), n_head_log2, 0u, split_keys, 0u };
    const AmevaBind reads[4] = { bq, bk, bv, bm };
    if (split_keys) {
        // pass 1: partials of n_split key ranges into the scratch buffer (reserved by graph_compute); pass 2: merge
        const uint64_t rows_out = (uint64_t)q->ne[1] * n_head;
        const uint64_t bytes = (uint64_t)n_split * rows_out * ((uint64_t)v->ne[0] + 2) * 4;
        if (!s_scratch || bytes > s_scratch_bytes) return "flash-attention scratch not reserved";
        AmevaBind bp = {};
        bp.b.buffer = s_scratch;
        bp.b.offset = 0;
        bp.b.range = std::min<uint64_t>(s_scratch_bytes, s_limits.max_storage_buffer_range) & ~3ull;
        bp.key = s_scratch;
        bp.lo = 0;
        bp.hi = bytes;
        const AcBufferBinding bind[6] = { bq.b, bk.b, bv.b, bm.b, bo.b, bp.b };
        const uint64_t tiles = (rows_out / n_head * (n_head / n_head_kv) + fv.bq - 1) / fv.bq;
        e = ameva_record(rec, kernel, reads, 4, &bp, 1, bind, 6, pc, sizeof(pc),
                         { (uint32_t)tiles, n_head_kv, n_split },
                         (double)n_head_kv * (double)k->ne[1] * (double)fv.bq * (double)(q->ne[0] + v->ne[0]));
        if (e) return e;
        AcKernel* comb = ameva_op_kernel(AMEVA_K_FA_COMBINE);
        if (!comb) return "no ameva_fa_combine kernel";
        const uint32_t pcc[5] = { (uint32_t)v->ne[0], (uint32_t)rows_out, n_split, 0u, bo.elem };
        const AcBufferBinding bindc[2] = { bp.b, bo.b };
        return ameva_record(rec, comb, &bp, 1, &bo, 1, bindc, 2, pcc, sizeof(pcc), ameva_grid_groups(rows_out),
                            (double)rows_out * n_split * (double)v->ne[0]);
    }
    const AcBufferBinding bind[6] = { bq.b, bk.b, bv.b, bm.b, bo.b, bo.b }; // binding 5 is not written
    // a tile holds bq rows (query position x head of one KV group) of one KV head: grid (tiles, n_head_kv). Long prompts
    // are split over several dispatches so that none exceeds kDispatchMac; every dispatch is recorded as writing the
    // whole output (the rows of a tile are interleaved over heads)
    const uint64_t rows = (uint64_t)q->ne[1] * (n_head / n_head_kv);
    const uint64_t tiles = (rows + fv.bq - 1) / fv.bq;
    const double mac_per_tile = (double)fv.bq * (double)n_head_kv * (double)k->ne[1] * (double)(q->ne[0] + v->ne[0]);
    const uint64_t per_dispatch =
        std::min<uint64_t>(65535, std::max<uint64_t>(1, (uint64_t)(kDispatchMac / std::max(1.0, mac_per_tile))));
    for (uint64_t t0 = 0; t0 < tiles; t0 += per_dispatch) {
        const uint64_t nt = std::min<uint64_t>(per_dispatch, tiles - t0);
        pc[28] = (uint32_t)t0;
        e = ameva_record(rec, kernel, reads, 4, &bo, 1, bind, 6, pc, sizeof(pc), { (uint32_t)nt, n_head_kv, 1 },
                         (double)nt * mac_per_tile);
        if (e) return e;
    }
    return nullptr;
}

// Graph-engine fusion of adjacent node pairs (AMEVA_FUSE=0 disables it). The fused dispatch computes node i+1 as well
// and still writes node i's output ("write-through"), so the result stays correct whoever else reads node i (the
// backend sees one split of the graph, so other uses of an intermediate cannot be ruled out). Pairs:
//   RMS_NORM -> MUL by a one-row weight            ameva_rms_norm with mul
//   MUL_MAT (ameva_qmv, one row) -> ADD of an N-vector  (bias, residual)   ameva_qmv epilogue
//   MUL_MAT (ameva_qmv, one row) -> SWIGLU (split) with an N-vector           ameva_qmv epilogue
// Each saves one dispatch and its barrier; the values are those of the unfused nodes (same f32 operations).
static bool ameva_fuse_enabled(void) {
    static const bool on = [] {
        const char* e = getenv("AMEVA_FUSE");
        if (!e || !*e) return true;
        if (strcmp(e, "0") == 0) return false;
        if (strcmp(e, "1") == 0) return true;
        GGML_ABORT("ggml-ameva: AMEVA_FUSE must be 0 or 1 (got \"%s\")", e);
    }();
    return on;
}

enum AmevaFuseKind { AMEVA_FUSE_NONE = 0, AMEVA_FUSE_RMS_MUL, AMEVA_FUSE_MM_EPILOGUE };

struct AmevaFusion {
    AmevaFuseKind kind;
    const struct ggml_tensor* w;  // RMS_MUL: norm weight
    AmevaEpilogue ep;             // MM_EPILOGUE
};

// f32 vector of exactly n elements in one contiguous row
static bool is_f32_vector(const struct ggml_tensor* t, int64_t n) {
    return is_f32(t) && rows_contiguous(t) && t->ne[0] == n && ggml_nelements(t) == n;
}

// byte ranges of two tensors in the same buffer intersect
static bool tensors_overlap(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    if (!a || !b || !a->data || !b->data || a->buffer != b->buffer) return false;
    const char *a0 = (const char*)a->data, *b0 = (const char*)b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}
// the same elements in the same layout (an in-place result): element k is read and written by the same invocation
static bool tensors_alias(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    return a->buffer == b->buffer && a->data == b->data && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);
}
// the output of the fused second node may only overlap a tensor the dispatch reads or writes element by element at
// the same index (in-place); any other overlap (e.g. memory of an input that died at the first node, reused by the
// allocator) would let one workgroup overwrite data another one still reads
static bool fused_output_safe(const struct ggml_tensor* out, const struct ggml_tensor* t, bool allow_alias) {
    return !tensors_overlap(out, t) || (allow_alias && tensors_alias(out, t));
}

static AmevaFusion ameva_find_fusion(const struct ggml_tensor* n0, const struct ggml_tensor* n1) {
    AmevaFusion f = { AMEVA_FUSE_NONE, nullptr, { 0, nullptr, nullptr } };
    if (!n1 || ggml_is_empty(n1)) return f;
    if (n0->op == GGML_OP_RMS_NORM && n1->op == GGML_OP_MUL && n1->src[0] == n0 && n1->src[1] != n0) {
        const struct ggml_tensor* w = n1->src[1];
        if (is_f32_vector(w, n0->ne[0]) && is_f32(n1) && ggml_are_same_shape(n1, n0) && rows_contiguous(n1) &&
            fused_output_safe(n1, n0->src[0], true) && fused_output_safe(n1, n0, true) &&
            fused_output_safe(n1, w, false)) {
            f.kind = AMEVA_FUSE_RMS_MUL;
            f.w = w;
        }
        return f;
    }
    if (n0->op != GGML_OP_MUL_MAT || !ameva_repacked_base(n0->src[0]) || n0->src[1]->ne[1] != 1) return f;
    const int64_t N = n0->ne[0];
    if (!is_f32_vector(n1, N)) return f;
    const struct ggml_tensor* other = n1->src[0] == n0 ? n1->src[1] : (n1->src[1] == n0 ? n1->src[0] : nullptr);
    if (!other || other == n0 || !is_f32_vector(other, N)) return f;
    // the activation row is read by every workgroup: the epilogue output must not overlap it at all
    if (!fused_output_safe(n1, n0->src[1], false) || !fused_output_safe(n1, n0, true) ||
        !fused_output_safe(n1, other, true)) return f;
    if (n1->op == GGML_OP_ADD) {
        f.ep = { 1u, other, n1 };
    } else if (n1->op == GGML_OP_GLU && n1->src[1] && glu_code(n1) == 2) {  // SWIGLU, split form: silu(src0) * src1
        f.ep = { n1->src[0] == n0 ? 2u : 3u, other, n1 };
    } else {
        return f;
    }
    f.kind = AMEVA_FUSE_MM_EPILOGUE;
    return f;
}

static const char* ameva_encode(AmevaRecorder& rec, const struct ggml_tensor* node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: return enc_mul_mat(rec, node);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: return enc_binary(rec, node);
        case GGML_OP_SCALE:
        case GGML_OP_UNARY: return enc_unary(rec, node);
        case GGML_OP_GLU: return enc_glu(rec, node);
        case GGML_OP_RMS_NORM: return enc_rms_norm(rec, node);
        case GGML_OP_ROPE: return enc_rope(rec, node);
        case GGML_OP_SET_ROWS: return enc_set_rows(rec, node);
        case GGML_OP_GET_ROWS: return enc_get_rows(rec, node);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP: return enc_cpy(rec, node);
        case GGML_OP_FLASH_ATTN_EXT: return enc_flash_attn(rec, node);
        default: return "op has no ameva encoder";
    }
}

// =============================================================================
// Backend
// =============================================================================
static ggml_guid_t ggml_backend_ameva_guid() {
    static ggml_guid guid = {
        0x41, 0x4d, 0x45, 0x56, 0x41, 0x2d, 0x43, 0x4f,
        0x4d, 0x50, 0x55, 0x54, 0x45, 0x30, 0x31, 0x00
    };
    return &guid;
}

bool ggml_backend_is_ameva(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_ameva_guid());
}

struct ggml_backend_ameva_context {
    int device_id;
};

static const char* ggml_backend_ameva_name(ggml_backend_t backend) {
    (void)backend;
    return GGML_AMEVA_NAME;
}

static void ggml_backend_ameva_free(ggml_backend_t backend) {
    if (!backend) return;
    delete (ggml_backend_ameva_context*)backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_ameva_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    (void)backend;
    if (!cgraph) return GGML_STATUS_FAILED;
    AcContext* ctx = ameva_get_ac_context();
    if (!ctx || !ameva_status_buffer()) return GGML_STATUS_FAILED;

    std::lock_guard<std::mutex> compute_lock(s_compute_mutex);
    const AmevaFailureInjectionPoint fi = ggml_backend_ameva_get_failure_injection();
    const uint64_t t0 = ameva_now_ns();
    const int n_nodes = ggml_graph_n_nodes(cgraph); // public accessors: struct ggml_cgraph is internal to ggml

    // contract first: nothing is recorded for a graph this backend cannot run completely
    int work = 0;
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (!node || ameva_op_is_empty(node->op)) continue;
        if (const char* why = ameva_op_unsupported_reason(node)) {
            fprintf(stderr, "ggml-ameva: node %d (%s, op %s) cannot run here: %s (FAIL-FAST)\n", i, node->name,
                    ggml_op_desc(node), why);
            return GGML_STATUS_FAILED;
        }
        if (!ggml_is_empty(node)) ++work;
    }
    g_prof_graphs.fetch_add(1);
    if (work == 0) return GGML_STATUS_SUCCESS;
    const uint64_t t_contract = ameva_now_ns();
    if (fi == AMEVA_FAIL_TENSOR_RANGE_OOB || fi == AMEVA_FAIL_MAP_SRC0 || fi == AMEVA_FAIL_MAP_SRC1 ||
        fi == AMEVA_FAIL_MAP_DST || fi == AMEVA_FAIL_OFFSET_OVERFLOW || fi == AMEVA_FAIL_FLUSH_SRC0 ||
        fi == AMEVA_FAIL_FLUSH_SRC1) {
        fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: point %d triggered while binding\n", (int)fi);
        return GGML_STATUS_FAILED;
    }

    // token-generation GEMV layouts of weight shapes this device has not measured yet (once per shape)
    if (const char* e = ameva_qmv_tune_graph(cgraph, n_nodes)) {
        fprintf(stderr, "ggml-ameva: GEMV layout measurement failed: %s (FAIL-FAST)\n", e);
        return GGML_STATUS_FAILED;
    }

    // scratch for flash-attention key splits and MMQ activations: the largest need of any node, before anything is
    // recorded
    uint64_t scratch_need = 0;
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (!node || ggml_is_empty(node)) continue;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            scratch_need = std::max<uint64_t>(scratch_need, ameva_fa_scratch_bytes(node));
        } else if (node->op == GGML_OP_MUL_MAT) {
            uint32_t rows = 0;
            uint64_t bytes = 0;
            if (ameva_mmq_plan(node, &rows, &bytes)) scratch_need = std::max<uint64_t>(scratch_need, bytes);
            else scratch_need = std::max<uint64_t>(scratch_need, ameva_mmvq_scratch_bytes(node));
            scratch_need = std::max<uint64_t>(scratch_need, ameva_act_fq_bytes(node));
        }
    }
    if (scratch_need && !ameva_scratch_reserve(scratch_need)) {
        fprintf(stderr, "ggml-ameva: scratch of %llu bytes unavailable (FAIL-FAST)\n", (unsigned long long)scratch_need);
        return GGML_STATUS_FAILED;
    }

    s_status_map[0] = 0;
    s_status_map[1] = 0xFFFFFFFFu;
    if (ac_runtime_buffer_flush(s_status, 0, 256) != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: status buffer flush failed (FAIL-FAST)\n");
        return GGML_STATUS_FAILED;
    }

    AmevaRecorder rec;
    // Nodes may split into many dispatches (weight row slices, activation row chunks, attention tiles, row chunks of
    // large unary ops); recording checks the capacity per dispatch and a graph that exceeds it fails with
    // AC_ERR_DEVICE_BUSY instead of being cut silently. The capacity sizes nothing but the per-dispatch timestamp pool,
    // so it is the runtime maximum unless AMEVA_PROFILE=2 asks for those timestamps.
    rec.profile_kinds = ameva_profile_level() >= 2;
    const uint32_t capacity = rec.profile_kinds
                                  ? (uint32_t)std::min<uint64_t>(AC_BATCH_MAX_DISPATCHES, (uint64_t)work * 64 + 1024)
                                  : (uint32_t)AC_BATCH_MAX_DISPATCHES;
    const AcBatchDesc bdesc = { AC_ABI_VERSION, sizeof(AcBatchDesc), capacity,
                                AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_TIMESTAMPS |
                                    (rec.profile_kinds ? AC_BATCH_FLAG_DISPATCH_TIMESTAMPS : 0u) };
    AcResult br = ac_runtime_batch_begin_ex(ctx, &bdesc, &rec.batch);
    if (br != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: batch begin failed (code %d, FAIL-FAST)\n", (int)br);
        return GGML_STATUS_FAILED;
    }
    const uint64_t t_begin = ameva_now_ns();
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (!node || ameva_op_is_empty(node->op)) continue;
        // A node without elements has nothing to compute (llama.cpp builds e.g. the output projection with zero
        // rows when a ubatch requests no logits). No dispatch is recorded; the node is counted.
        if (ggml_is_empty(node)) {
            g_prof_empty.fetch_add(1);
            continue;
        }
        rec.node = i;
        if (rec.profile_kinds) {
            char kind[48];
            if (node->op == GGML_OP_MUL_MAT) {
                const bool rp = ameva_repacked_base(node->src[0]) != nullptr;
                uint32_t mmq_rows = 0;
                uint64_t mmq_bytes = 0;
                const char* path = !rp ? "qmm"
                                       : ameva_mmq_plan(node, &mmq_rows, &mmq_bytes)
                                             ? "mmq"
                                             : (node->src[1]->ne[1] >= (int64_t)ameva_qgemm_min_rows() ? "qgemm" : "qmv");
                char layout[16] = "";
                AmevaQmvConfig qc;
                if (ameva_qmv_one_row(node) && ameva_qmv_choice(node, &qc)) {
                    if (qc.mali) snprintf(layout, sizeof(layout), " mali");
                    else snprintf(layout, sizeof(layout), " %sr%ut%u", !qc.i8 ? "" : ameva_i8_family().emulated ? "e8" : "i8",
                                  (unsigned)qc.rows, (unsigned)qc.tpr);
                }
                snprintf(kind, sizeof(kind), "MUL_MAT %s %s %s %lldx%lld%s", ggml_type_name(node->src[0]->type),
                         node->src[1]->ne[1] == 1 ? "M=1" : "M>1", path, (long long)node->src[0]->ne[1],
                         (long long)node->src[0]->ne[0], layout);
            } else {
                snprintf(kind, sizeof(kind), "%s", ggml_op_desc(node));
            }
            rec.kind = ameva_kind_id(kind);
        }
        const uint64_t te = ameva_now_ns(), kc0 = g_prof_kcreate_ns.load(std::memory_order_relaxed);
        const AmevaFusion fz = ameva_fuse_enabled() && i + 1 < n_nodes
                                   ? ameva_find_fusion(node, ggml_graph_node(cgraph, i + 1))
                                   : AmevaFusion{ AMEVA_FUSE_NONE, nullptr, { 0, nullptr, nullptr } };
        const char* enc_err = fz.kind == AMEVA_FUSE_RMS_MUL
                                  ? enc_rms_norm(rec, node, ggml_graph_node(cgraph, i + 1), fz.w)
                                  : fz.kind == AMEVA_FUSE_MM_EPILOGUE ? enc_mul_mat(rec, node, &fz.ep)
                                                                      : ameva_encode(rec, node);
        if (node->op >= 0 && node->op < GGML_OP_COUNT) {
            const uint64_t kc = g_prof_kcreate_ns.load(std::memory_order_relaxed) - kc0;
            g_prof_enc_ns[node->op].fetch_add(ameva_now_ns() - te - kc, std::memory_order_relaxed);
            g_prof_enc_nodes[node->op].fetch_add(1, std::memory_order_relaxed);
        }
        if (const char* e = enc_err) {
            fprintf(stderr, "ggml-ameva: node %d (%s, op %s): %s (FAIL-FAST)\n", i, node->name, ggml_op_desc(node), e);
            ac_runtime_batch_destroy(rec.batch);
            return GGML_STATUS_FAILED;
        }
        if (fz.kind != AMEVA_FUSE_NONE) {
            ++i; // node i + 1 was computed by the dispatches of node i
            rec.fused_pairs++;
        }
    }
    AcBatchStats stats = {};
    stats.abi_version = AC_ABI_VERSION;
    stats.struct_size = sizeof(stats);
    ac_runtime_batch_get_stats(rec.batch, &stats);

    if (fi == AMEVA_FAIL_DISPATCH) {
        fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: DISPATCH_FAIL triggered\n");
        ac_runtime_batch_destroy(rec.batch);
        return GGML_STATUS_FAILED;
    }
    // AMEVA_TRACE=1: one line per graph before submission and one after completion, so a run that stops making
    // progress shows which graph it stopped in
    static const bool trace = getenv("AMEVA_TRACE") && strcmp(getenv("AMEVA_TRACE"), "1") == 0;
    const uint64_t graph_no = g_prof_graphs.load();
    if (trace) {
        fprintf(stderr, "AMEVA_TRACE graph=%llu nodes=%d work=%d dispatches=%u barriers=%u flushes=%u encode_ms=%.3f\n",
                (unsigned long long)graph_no, n_nodes, work, rec.dispatches, rec.barriers, rec.flushes,
                (ameva_now_ns() - t0) / 1e6);
    }
    AcEvent* ev = nullptr;
    const uint64_t t_submit = ameva_now_ns();
    AcResult sr = ac_runtime_batch_submit(rec.batch, &ev);
    const uint64_t t1 = ameva_now_ns();
    if (sr != AC_SUCCESS || !ev) {
        fprintf(stderr, "ggml-ameva: batch submit failed (code %d, FAIL-FAST)\n", (int)sr);
        ac_runtime_batch_destroy(rec.batch);
        return GGML_STATUS_FAILED;
    }
    ameva_increment_event_create_count();
    const uint64_t timeout_ns = fi == AMEVA_FAIL_WAIT_TIMEOUT ? 1ull : 30000000000ull;
    AcResult wr = ac_runtime_event_wait(ev, timeout_ns);
    const uint64_t t2 = ameva_now_ns();
    uint64_t gpu_ns = 0;
    if (wr == AC_SUCCESS) {
        uint64_t wall = 0;
        if (ac_runtime_event_get_timing(ev, &wall, &gpu_ns) == AC_SUCCESS && gpu_ns > 0) {
            g_prof_gpu_ns.fetch_add(gpu_ns);
            g_prof_gpu_samples.fetch_add(1);
        }
    }
    if (trace) {
        fprintf(stderr, "AMEVA_TRACE graph=%llu wait=%d wait_ms=%.3f gpu_ms=%.3f\n", (unsigned long long)graph_no,
                (int)wr, (t2 - t1) / 1e6, gpu_ns / 1e6);
    }
    if (wr == AC_SUCCESS && rec.profile_kinds && !rec.kinds.empty()) {
        std::vector<uint64_t> times(rec.kinds.size());
        uint32_t n = 0;
        const AcResult tr = ac_runtime_batch_get_dispatch_times(rec.batch, times.data(), (uint32_t)times.size(), &n);
        if (tr != AC_SUCCESS || n != rec.kinds.size()) {
            fprintf(stderr, "ggml-ameva: per-dispatch GPU times unavailable (code %d, %u of %zu)\n", (int)tr, n,
                    rec.kinds.size());
        } else {
            std::lock_guard<std::mutex> lock(s_kind_mutex);
            for (uint32_t i = 0; i < n; ++i) {
                s_kinds[rec.kinds[i]].ns += times[i];
                s_kinds[rec.kinds[i]].dispatches++;
            }
        }
    }
    // on a wait failure the event destroy waits again (bounded) and keeps everything if the GPU is still busy
    AcResult er = ac_runtime_event_destroy(ev);
    ameva_increment_event_destroy_count();
    ac_runtime_batch_destroy(rec.batch);
    if (wr != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: graph of %d dispatches: wait failed with code %d (FAIL-FAST)\n", rec.dispatches,
                (int)wr);
        return GGML_STATUS_FAILED;
    }
    if (er != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: event destroy failed with code %d (FAIL-FAST)\n", (int)er);
        return GGML_STATUS_FAILED;
    }
    if (fi == AMEVA_FAIL_INVALIDATE_DST || ac_runtime_buffer_invalidate(s_status, 0, 256) != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: status buffer invalidate failed%s (FAIL-FAST)\n",
                fi == AMEVA_FAIL_INVALIDATE_DST ? " [FAILURE_INJECTION]" : "");
        return GGML_STATUS_FAILED;
    }
    if (s_status_map[0] != 0) {
        const uint32_t n = s_status_map[1];
        struct ggml_tensor* bad = n < (uint32_t)n_nodes ? ggml_graph_node(cgraph, (int)n) : nullptr;
        fprintf(stderr, "ggml-ameva: %u row index(es) out of range, first in node %u (%s, op %s) (FAIL-FAST)\n",
                s_status_map[0], n, bad ? bad->name : "?", bad ? ggml_op_desc(bad) : "?");
        return GGML_STATUS_FAILED;
    }

    g_prof_nodes.fetch_add((uint64_t)work);
    g_prof_dispatches.fetch_add(rec.dispatches);
    g_prof_barriers.fetch_add(rec.barriers);
    g_prof_flushes.fetch_add(rec.flushes);
    g_prof_cache_hits.fetch_add(stats.descriptor_cache_hits);
    g_prof_cache_misses.fetch_add(stats.descriptor_cache_misses);
    g_prof_encode_ns.fetch_add(t1 - t0);
    g_prof_contract_ns.fetch_add(t_contract - t0);
    g_prof_begin_ns.fetch_add(t_begin - t_contract);
    g_prof_flush_ns.fetch_add(rec.flush_ns);
    g_prof_rt_barrier_ns.fetch_add(rec.rt_barrier_ns);
    g_prof_rt_dispatch_ns.fetch_add(rec.rt_dispatch_ns);
    g_prof_submit_ns.fetch_add(t1 - t_submit);
    g_prof_wait_ns.fetch_add(t2 - t1);
    for (int i = 0; i < work; ++i) ameva_increment_gpu_op_count(); // counted only after the GPU finished
    g_prof_mul_mat.fetch_add((uint64_t)rec.mul_mat_nodes);
    ameva_count_mul_mat(rec.mul_mat_nodes);
    ameva_count_qmv_mul_mat(rec.qmv_nodes);
    ameva_count_mmq_mul_mat(rec.mmq_nodes);
    g_prof_fused.fetch_add((uint64_t)rec.fused_pairs);
    g_prof_mmvq.fetch_add((uint64_t)rec.mmvq_nodes);
    ameva_pipeline_cache_checkpoint();
    return GGML_STATUS_SUCCESS;
}

// Designated initializers: see ggml-ameva-device.cpp. synchronize stays NULL (ggml treats it as optional)
// because graph_compute returns only after the batch it submitted has completed; there is never pending work.
static const struct ggml_backend_i ggml_backend_ameva_i = {
    .get_name      = ggml_backend_ameva_name,
    .free          = ggml_backend_ameva_free,
    .synchronize   = nullptr,
    .graph_compute = ggml_backend_ameva_graph_compute,
};

const char* ameva_abi_check(void);                // ggml-ameva-device.cpp
extern struct ggml_backend_device g_ameva_device; // ggml-ameva-device.cpp

// The backend carries the device it was created for. It used to look the device up by name in the global
// registry, which yields NULL whenever the backend is created before (or without) registration.
ggml_backend_t ameva_backend_create(ggml_backend_dev_t dev) {
    if (const char* why = ameva_abi_check()) {
        fprintf(stderr, "ggml-ameva: cannot create the AMEVA backend: %s\n", why);
        return nullptr;
    }
    if (dev != &g_ameva_device) {
        fprintf(stderr, "ggml-ameva: cannot create the AMEVA backend for foreign device %p\n", (void*)dev);
        return nullptr;
    }
    if (!ameva_get_ac_context()) return nullptr;
    // the plugin destructor does not run when the loader never dlcloses the backend (llama.cpp b8471 on
    // Android), so the profile report is tied to process exit instead
    static std::once_flag s_report_once;
    std::call_once(s_report_once, [] { std::atexit(ameva_profile_report); });
    auto* backend = new ggml_backend{
        .guid    = ggml_backend_ameva_guid(),
        .iface   = ggml_backend_ameva_i,
        .device  = dev,
        .context = new ggml_backend_ameva_context{ 0 },
    };
    return backend;
}

ggml_backend_t ggml_backend_ameva_init(size_t dev_num) {
    if (dev_num != 0) {
        fprintf(stderr, "ggml-ameva: device %zu does not exist (AMEVA exposes one device, AMEVA0)\n", dev_num);
        return nullptr;
    }
    return ameva_backend_create(&g_ameva_device);
}

// Dynamic Loader Entrypoints
extern "C" {
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
    // NULL makes the loader print "failed to initialize backend from <path>"; a later --device AMEVA0 then
    // fails as an unknown device instead of crashing inside the scheduler.
    if (const char* why = ameva_abi_check()) {
        fprintf(stderr, "ggml-ameva: refusing to load: %s\n", why);
        return nullptr;
    }
    return ggml_backend_ameva_reg();
}

GGML_BACKEND_API int ggml_backend_score(void) {
    return 100;
}
}
