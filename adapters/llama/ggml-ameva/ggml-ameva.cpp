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
#include <vector>

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
//                              buffers, ggml-ameva-repack.h), 12 types; ameva_qgemm (64 x 32 tiles) for those
//                              weights when many activation rows are multiplied (prompt processing)
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
    }
    fprintf(stderr, "AMEVA_PROFILE repack=%s repacked_tensors=%lld repacked_mib=%.1f qmv_mul_mat=%lld\n",
            ameva_repack_enabled() ? "on" : "off", (long long)ggml_backend_ameva_get_repacked_tensor_count(),
            ggml_backend_ameva_get_repacked_bytes() / 1048576.0, (long long)ggml_backend_ameva_get_qmv_mul_mat_count());
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
static AcKernel* s_qgemm[AMEVA_QGEMM_VARIANT_COUNT] = {};
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

static AcKernel* ameva_make_kernel(const char* name, const uint32_t* spirv, size_t bytes, const char* sha,
                                   uint32_t bindings, uint32_t push_bytes, uint32_t local_x) {
    AcKernelRuntimeMetadata meta = {
        name, "main", bindings, kBindings, push_bytes, { local_x, 1, 1 }, 0, sha, 0, nullptr
    };
    AcKernelDesc kdesc = { AC_ABI_VERSION, sizeof(AcKernelDesc), spirv, bytes, &meta, 0, nullptr };
    AcKernel* k = nullptr;
    AcResult res = ac_runtime_kernel_create(s_ctx, &kdesc, &k);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: failed to create kernel %s (code %d)\n", name, res);
        return nullptr;
    }
    ameva_increment_kernel_create_count();
    return k;
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

static AcKernel* ameva_qmv_kernel(int idx) {
    static char names[AMEVA_QMV_VARIANT_COUNT][40];
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_qmv[idx] && s_ctx) {
        const AmevaQmvVariant& v = kAmevaQmvVariants[idx];
        snprintf(names[idx], sizeof(names[idx]), "ameva_qmv_%s_c%u", v.type_name, v.cols);
        s_qmv[idx] = ameva_make_kernel(names[idx], v.spirv, v.spirv_bytes, v.spirv_sha256, AMEVA_QMV_BINDINGS,
                                       AMEVA_QMV_PUSH_CONSTANT_BYTES, AMEVA_QMV_LOCAL_SIZE_X);
    }
    return s_qmv[idx];
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
    for (AcKernel*& k : s_qgemm) {
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

static int ameva_qmv_find(enum ggml_type type, uint32_t cols) {
    const char* tn = ggml_type_name(type);
    if (!tn) return -1;
    for (int i = 0; i < AMEVA_QMV_VARIANT_COUNT; ++i) {
        if (kAmevaQmvVariants[i].cols == cols && strcmp(kAmevaQmvVariants[i].type_name, tn) == 0) return i;
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
    return ameva_qmv_find(type, 1) >= 0 && ameva_qmv_find(type, 8) >= 0 && ameva_qgemm_find(type) >= 0;
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
            if (!is_f32(a) || !is_f32(op) || !ggml_are_same_shape(a, op)) return "rope on non-f32";
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
            if (!is_f32(a) || !(is_f16(op) || is_f32(op))) return "set_rows type pair";
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
            if (!(is_f32(a) || is_f16(a)) || !(is_f32(op) || is_f16(op))) return "copy between types other than f32/f16";
            if (ggml_nelements(a) != ggml_nelements(op)) return "copy element counts differ";
            if (!fits_binding(a) || !fits_binding(op)) return "tensor exceeds a buffer binding";
            if (!fits_kernel_indexing(a) || !fits_kernel_indexing(op)) return "exceeds 32-bit indexing";
            return nullptr;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor *q = op->src[0], *k = op->src[1], *v = op->src[2], *m = op->src[3];
            if (op->src[4]) return "attention sinks";
            if (!is_f32(q) || !is_f32(op) || !k || !v || k->type != v->type || !(is_f16(k) || is_f32(k)))
                return "flash_attn types";
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
    int node = 0;
    int64_t mul_mat_nodes = 0;    // counted once the graph completed
    int64_t qmv_nodes = 0;
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
        if (ac_runtime_batch_barrier(rec.batch) != AC_SUCCESS) return "barrier recording failed";
        rec.written.clear();
        rec.read.clear();
        rec.barriers++;
    }
    AcResult r = ac_runtime_batch_dispatch_bindings(rec.batch, kernel, bindings, nbind, pc, pc_size, grid);
    if (r != AC_SUCCESS) {
        static char msg[96];
        snprintf(msg, sizeof(msg), "dispatch recording failed (code %d)", (int)r);
        return msg;
    }
    for (int i = 0; i < nreads; ++i) rec.read.push_back({reads[i].key, reads[i].lo, reads[i].hi});
    for (int i = 0; i < nwrites; ++i) rec.written.push_back({writes[i].key, writes[i].lo, writes[i].hi});
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

// Lanes per weight row of ameva_qmv: 16 for every row length. Measured with Qwen2.5-0.5B Q4_K_M token generation
// (2026-10-03): Adreno 650 per-kernel times with 8 / 16 / 64 lanes: q4_K 896x4864 336 / 356 / 611 us, q8_0
// 151936x896 7.3 / 8.2 / 11.4 ms; Mali-G68 decode 14.0 / 14.5 t/s with 8 / 16 lanes, 14.1 with 64 lanes for
// K = 4864. More lanes per row means fewer rows per workgroup and more activation reads per weight byte.
// AMEVA_QMV_TPR=4|8|16|32|64 fixes it (kernel tuning).
static uint32_t ameva_qmv_tpr(uint32_t units) {
    static const uint32_t forced = [] {
        const char* e = getenv("AMEVA_QMV_TPR");
        if (!e || !*e) return 0u;
        if (!strcmp(e, "4")) return 4u;
        if (!strcmp(e, "8")) return 8u;
        if (!strcmp(e, "16")) return 16u;
        if (!strcmp(e, "32")) return 32u;
        if (!strcmp(e, "64")) return 64u;
        GGML_ABORT("ggml-ameva: AMEVA_QMV_TPR must be 4, 8, 16, 32 or 64 (got \"%s\")", e);
    }();
    (void)units;
    return forced ? forced : 16u;
}

// MUL_MAT. Weights are multiplied in slices of rows that fit one binding (per region for repacked weights) and at most
// 65535 workgroups; activation rows in chunks (see below). Repacked weights bind their Q, H and S regions.
static const char* enc_mul_mat(AmevaRecorder& rec, const struct ggml_tensor* node) {
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
    if (base) {
        if (gemm) {
            const int gi = ameva_qgemm_find(src0->type);
            kernel = gi >= 0 ? ameva_qgemm_kernel(gi) : nullptr;
            if (!kernel) return "no ameva_qgemm kernel";
            rows_per_wg = 64; // shaders/ameva_qgemm.comp tile: 64 weight rows x 32 activation rows
            cols_per_wg = 32;
        } else {
            const int vi = ameva_qmv_find(src0->type, ncols);
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
    const uint32_t s1 = st(src1, 1), sd = st(node, 1);
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
            e = ameva_bind_span(src1, (uint64_t)m0 * s1 * 4, ((uint64_t)(mc - 1) * s1 + K) * 4, 4, &b1);
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
                const uint32_t pc[11] = { mc, K, nc, w[0].elem, R.h_row ? w[1].elem : 0u, w[2].elem * 2,
                                          b1.elem, bd.elem, s1, sd, tpr };
                const AcBufferBinding bind[5] = { w[0].b, w[1].b, w[2].b, b1.b, bd.b };
                const AmevaBind reads[4] = { w[0], w[1], w[2], b1 };
                e = ameva_record(rec, kernel, reads, 4, &bd, 1, bind, 5, pc, sizeof(pc), grid, mac);
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

static const char* enc_rms_norm(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor* a = node->src[0];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_RMS_NORM);
    if (!kernel) return "no ameva_rms_norm kernel";
    AmevaBind ba, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(node, 4, &bd);
    if (e) return e;
    const uint64_t nrows = (uint64_t)node->ne[1] * node->ne[2] * node->ne[3];
    const uint32_t pc[13] = { (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2], (uint32_t)nrows,
                              ba.elem, st(a, 1), st(a, 2), st(a, 3),
                              bd.elem, st(node, 1), st(node, 2), st(node, 3), fbits(op_param_f32(node, 0)) };
    const AcBufferBinding bind[2] = { ba.b, bd.b };
    return ameva_record(rec, kernel, &ba, 1, &bd, 1, bind, 2, pc, sizeof(pc), ameva_grid_groups(nrows),
                        2.0 * (double)ggml_nelements(node));
}

static const char* enc_rope(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *pos = node->src[1], *ff = node->src[2];
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_ROPE);
    if (!kernel) return "no ameva_rope kernel";
    AmevaBind ba, bp, bf, bd;
    const char* e = ameva_bind(a, 4, &ba);
    if (!e) e = ameva_bind(pos, 4, &bp);
    if (!e) e = ameva_bind(ff ? ff : pos, 4, &bf); // any valid range when there are no frequency factors
    if (!e) e = ameva_bind(node, 4, &bd);
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

static const char* enc_set_rows(AmevaRecorder& rec, const struct ggml_tensor* node) {
    const struct ggml_tensor *a = node->src[0], *idx = node->src[1];
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
    AcKernel* kernel = ameva_op_kernel(AMEVA_K_CPY);
    if (!kernel) return "no ameva_cpy kernel";
    const bool sf16 = a->type == GGML_TYPE_F16, df16 = node->type == GGML_TYPE_F16;
    AmevaBind ba, bd;
    const char* e = ameva_bind(a, sf16 ? 2 : 4, &ba);
    if (!e) e = ameva_bind(node, df16 ? 2 : 4, &bd);
    if (e) return e;
    const uint32_t pc[19] = { (uint32_t)a->ne[0], (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3],
                              ba.elem, st(a, 0), st(a, 1), st(a, 2),
                              st(a, 3), (uint32_t)node->ne[0], (uint32_t)node->ne[1], (uint32_t)node->ne[2],
                              bd.elem, st(node, 0), st(node, 1), st(node, 2),
                              st(node, 3), sf16 ? 1u : 0u, df16 ? 1u : 0u };
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
    const bool kv16 = k->type == GGML_TYPE_F16;
    AmevaBind bq, bk, bv, bm, bo;
    const char* e = ameva_bind(q, 4, &bq);
    if (!e) e = ameva_bind(k, kv16 ? 2 : 4, &bk);
    if (!e) e = ameva_bind(v, kv16 ? 2 : 4, &bv);
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
                        n_head, n_head_kv, kv16 ? 1u : 0u, m ? 1u : 0u,
                        bq.elem, st(q, 1), st(q, 2), bk.elem,
                        st(k, 1), st(k, 2), bv.elem, st(v, 1),
                        st(v, 2), m ? bm.elem : 0u, m ? st(m, 1) : 0u, m ? st(m, 2) : 0u,
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

    // scratch for flash-attention key splits: the largest need of any node, before anything is recorded
    uint64_t scratch_need = 0;
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (node && node->op == GGML_OP_FLASH_ATTN_EXT && !ggml_is_empty(node))
            scratch_need = std::max<uint64_t>(scratch_need, ameva_fa_scratch_bytes(node));
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
    // MUL_MAT and attention may split into several dispatches; recording checks the capacity per dispatch and a graph
    // that exceeds it fails with AC_ERR_DEVICE_BUSY instead of being cut silently
    const uint32_t capacity = (uint32_t)std::min<uint64_t>(AC_BATCH_MAX_DISPATCHES, (uint64_t)work * 8 + 256);
    rec.profile_kinds = ameva_profile_level() >= 2;
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
                const char* path = !rp ? "qmm" : (node->src[1]->ne[1] >= (int64_t)ameva_qgemm_min_rows() ? "qgemm" : "qmv");
                snprintf(kind, sizeof(kind), "MUL_MAT %s %s %s %lldx%lld", ggml_type_name(node->src[0]->type),
                         node->src[1]->ne[1] == 1 ? "M=1" : "M>1", path, (long long)node->src[0]->ne[1],
                         (long long)node->src[0]->ne[0]);
            } else {
                snprintf(kind, sizeof(kind), "%s", ggml_op_desc(node));
            }
            rec.kind = ameva_kind_id(kind);
        }
        if (const char* e = ameva_encode(rec, node)) {
            fprintf(stderr, "ggml-ameva: node %d (%s, op %s): %s (FAIL-FAST)\n", i, node->name, ggml_op_desc(node), e);
            ac_runtime_batch_destroy(rec.batch);
            return GGML_STATUS_FAILED;
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
    g_prof_submit_ns.fetch_add(t1 - t_submit);
    g_prof_wait_ns.fetch_add(t2 - t1);
    for (int i = 0; i < work; ++i) ameva_increment_gpu_op_count(); // counted only after the GPU finished
    g_prof_mul_mat.fetch_add((uint64_t)rec.mul_mat_nodes);
    ameva_count_mul_mat(rec.mul_mat_nodes);
    ameva_count_qmv_mul_mat(rec.qmv_nodes);
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
