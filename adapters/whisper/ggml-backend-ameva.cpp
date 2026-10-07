/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ggml-backend-ameva.cpp
 * @brief Formal AMEVA Compute Hardware Backend Implementation for GGML / whisper.cpp
 * Fully hardened with Backend Context Ownership, Concurrency Safety, and SPIR-V Integrity.
 */

#include "ggml-backend-ameva.h"
#include "../../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h"

#include "ac_types.h"
#include "ac_error.h" // error-state API; ac_runtime.h would clash with ac_types.h (AcContext etc.)
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <mutex>
#include <new>

static inline double ameva_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

#define UNUSED(x) (void)(x)

// Global Telemetry Counters
static AmevaBackendCounters g_counters = {};
static std::mutex s_counters_mutex;
static bool s_bounds_violation = false;

// SHA-256 of src/shaders/stt/matmul_{f16,q5_0}_f32.spv (glslangValidator 16.5.0, --target-env vulkan1.0), rewritten
// and re-pinned 2026-10-02. The previous pins (f16 4200b2b5..., q5_0 cd343cfb...) belonged to binaries whose
// source was in no branch and whose files were on none of the reachable devices.
static const char * const kF16SpirvSha256 = "63ab1fecca95ef16bc1387881421d900a134c5769861bb61c548bb6f85539e50";
static const char * const kQ50SpirvSha256 = "343fd4b667135404fbe552965b59c33c1c4a37f6488f304264bccea407485829";

static const size_t INITIAL_BUF_CAP = 32 * 1024 * 1024; // 32 MB

struct MatMulPushConstants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

// ============================================================================
// Per-Backend Hardware Context (P0-4: Strict Resource Ownership & Concurrency)
// ============================================================================

// P0-1: Exact WeightCacheKey structure with full SHA-256 payload identity
struct WeightCacheKey {
    uint8_t sha256[32];
    size_t byte_size;
    enum ggml_type type;
    int64_t ne[4];

    bool operator==(const WeightCacheKey & other) const {
        if (byte_size != other.byte_size || type != other.type) return false;
        if (ne[0] != other.ne[0] || ne[1] != other.ne[1] || ne[2] != other.ne[2] || ne[3] != other.ne[3]) return false;
        return memcmp(sha256, other.sha256, 32) == 0;
    }
};

struct WeightCacheKeyHasher {
    size_t operator()(const WeightCacheKey & k) const {
        uint64_t h;
        memcpy(&h, k.sha256, sizeof(uint64_t));
        return (size_t)(h ^ (uint64_t)k.byte_size ^ ((uint64_t)k.type << 32) ^ (uint64_t)k.ne[0]);
    }
};

struct CachedWeightBuffer {
    AcVkBuffer * buf;
    size_t       cap;
    size_t       size;
    WeightCacheKey key;
    std::vector<uint8_t> payload;
};

// Unified FIPS 180-2 SHA-256 header (P1-2: single source of truth)
#include "ac_vk_sha256.h"

// P0-2: Production Cache Diagnostics State
static AmevaCacheDiagnostics s_cache_diagnostics = {};
static std::mutex s_diag_mutex;

void ameva_backend_get_cache_diagnostics(AmevaCacheDiagnostics * out_diag) {
    if (out_diag) {
        std::lock_guard<std::mutex> lk(s_diag_mutex);
        *out_diag = s_cache_diagnostics;
    }
}

void ameva_backend_reset_cache_diagnostics(void) {
    std::lock_guard<std::mutex> lk(s_diag_mutex);
    memset(&s_cache_diagnostics, 0, sizeof(s_cache_diagnostics));
}

struct ameva_backend_context {
    const char * name;
    AcVkContext * vk_ctx;
    AcVkKernel  * kernel_f16;
    AcVkKernel  * kernel_q5_0;
    AcVkBuffer  * buf_w;
    AcVkBuffer  * buf_a;
    AcVkBuffer  * buf_o;
    size_t        buf_w_cap;
    size_t        buf_a_cap;
    size_t        buf_o_cap;
    std::mutex    dispatch_mutex;
    uint32_t      graph_run_id;
    uint64_t      dispatch_index;
    bool          initialized;

    std::unordered_map<WeightCacheKey, CachedWeightBuffer, WeightCacheKeyHasher> weight_cache;
};

static struct ameva_backend_context * s_cached_test_ctx = nullptr;
static std::mutex s_test_ctx_mutex;

// Every live backend context, so ameva_backend_teardown_hardware() can reach all AMEVA GPU state in the process.
static std::unordered_set<struct ameva_backend_context *> s_live_ctxs;
static std::mutex s_live_ctx_mutex;

void ameva_backend_clear_test_cache(void) {
    std::lock_guard<std::mutex> lock(s_test_ctx_mutex);
    if (s_cached_test_ctx) {
        for (auto & pair : s_cached_test_ctx->weight_cache) {
            if (pair.second.buf) {
                ac_vk_buffer_destroy(pair.second.buf);
            }
        }
        s_cached_test_ctx->weight_cache.clear();
    }
    std::lock_guard<std::mutex> lk(s_diag_mutex);
    s_cache_diagnostics.entry_count = 0;
}

static char s_graph_kind[32] = "UNKNOWN";

// Inference scope (ameva_backend_begin/end_inference). Run 1 and every later run are counted separately.
static uint32_t s_current_inference_id = 1;
static uint32_t s_inference_run_count = 0;
// whisper.cpp tags each decoder evaluation with set_graph_kind("DECODER"): decoder passes per run [run 1, run 2+]
static uint64_t s_decoder_pass_count[2] = {0, 0};

void ameva_backend_set_graph_kind(const char * kind) {
    if (kind) {
        strncpy(s_graph_kind, kind, sizeof(s_graph_kind) - 1);
        s_graph_kind[sizeof(s_graph_kind) - 1] = '\0';
        if (strcmp(kind, "DECODER") == 0) {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            s_decoder_pass_count[s_current_inference_id <= 1 ? 0 : 1]++;
        }
    }
}

// Helper to safely load and validate SPIR-V binary (P0-5)
static bool load_spirv_binary(const char * path, std::vector<uint32_t> & out_code) {
    if (!path) return false;
    std::ifstream spv_file(path, std::ios::binary | std::ios::ate);
    if (!spv_file.is_open()) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Cannot open SPIR-V binary %s\n", path);
        return false;
    }
    size_t spv_size = (size_t)spv_file.tellg();
    if (spv_size < 4 || (spv_size % 4 != 0)) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Invalid SPIR-V size %zu for %s (not 4-byte aligned)\n", spv_size, path);
        return false;
    }
    spv_file.seekg(0, std::ios::beg);
    out_code.resize(spv_size / sizeof(uint32_t));
    spv_file.read(reinterpret_cast<char*>(out_code.data()), spv_size);
    size_t read_bytes = spv_file.gcount();
    if (read_bytes != spv_size) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Partial SPIR-V read: expected %zu, read %zu for %s\n", spv_size, read_bytes, path);
        return false;
    }
    spv_file.close();

    // Check SPIR-V Magic Header
    if (out_code[0] != 0x07230203u) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Invalid SPIR-V magic header 0x%08X in %s (expected 0x07230203)\n", out_code[0], path);
        return false;
    }
    return true;
}

// P0-6: Allocate-Before-Destroy Buffer Expansion
static bool ensure_buffer_capacity(struct ameva_backend_context * ctx, AcVkBuffer ** buf, size_t * current_cap, size_t needed_size, VkBufferUsageFlags usage) {
    if (needed_size > SIZE_MAX / 2) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Buffer needed size exceeds SIZE_MAX / 2: %zu\n", needed_size);
        return false;
    }
    if (*buf && *current_cap >= needed_size) {
        return true;
    }
    size_t new_cap = (needed_size < INITIAL_BUF_CAP) ? INITIAL_BUF_CAP : (needed_size * 2);
    AcVkBuffer * new_buf = nullptr;
    AcResult res = ac_vk_buffer_create(ctx->vk_ctx, new_cap, usage, &new_buf);
    if (res != AC_SUCCESS || !new_buf) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_create failed for cap %zu with code %d (retaining old buffer)\n", new_cap, res);
        return false;
    }
    if (*buf) {
        ac_vk_buffer_destroy(*buf);
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
    *buf = new_buf;
    *current_cap = new_cap;
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_create_count++;
    }
    return true;
}

// Initialize hardware pipeline for a specific backend context
static bool init_hardware_pipeline(struct ameva_backend_context * ctx) {
    if (!ctx) return false;
    if (ctx->vk_ctx != nullptr) return true;

    AcResult res = AC_SUCCESS;
    const char * init_fail_stage = getenv("AMEVA_INIT_FAIL_STAGE");

    // Stage 1: Context Creation
    if (init_fail_stage && strcmp(init_fail_stage, "context") == 0) {
        fprintf(stderr, "[AMEVA_INIT_FAIL_INJECTION] Simulated failure at stage: context\n");
        return false;
    }

    res = ac_vk_context_create(&ctx->vk_ctx);
    if (res != AC_SUCCESS || !ctx->vk_ctx) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_context_create failed with code %d\n", res);
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.context_create_count++;
    }

    if (ctx->vk_ctx->device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Vulkan context bound to CPU fallback! Rejected.\n");
        goto fail_kernel;
    }

    // Stage 2: Kernel Creation
    if (init_fail_stage && strcmp(init_fail_stage, "kernel") == 0) {
        fprintf(stderr, "[AMEVA_INIT_FAIL_INJECTION] Simulated failure at stage: kernel\n");
        goto fail_kernel;
    }

    // 2.1 F16 Kernel Creation with SHA-256 Authentication (P0-5)
    {
        const char * env_path = getenv("AMEVA_SPIRV_F16_PATH");
        if (!env_path) env_path = getenv("AMEVA_SPIRV_PATH");
        const char * spv_path = env_path ? env_path : "/data/data/com.termux/files/home/matmul_f16_f32.spv";

        std::vector<uint32_t> spv_code;
        if (!load_spirv_binary(spv_path, spv_code)) {
            goto fail_kernel;
        }

        const char * allow_unverified = getenv("AMEVA_ALLOW_UNVERIFIED_SPIRV");
        bool enforce_sha256 = (allow_unverified == nullptr || strcmp(allow_unverified, "1") != 0);

        AcKernelMetadata meta = {};
        meta.kernel_name = "matmul_f16_f32";
        meta.entry_point = "main";
        meta.binding_count = 3;
        meta.bindings = nullptr;
        meta.push_constant_size = sizeof(MatMulPushConstants);
        meta.local_size = {16, 16, 1};
        meta.required_features = AC_FEATURE_NONE;
        meta.expected_spirv_sha256 = enforce_sha256 ? kF16SpirvSha256 : nullptr;

        res = ac_vk_kernel_create_with_metadata(ctx->vk_ctx, spv_code.data(), spv_code.size() * sizeof(uint32_t), &meta, &ctx->kernel_f16);
        if (res != AC_SUCCESS || !ctx->kernel_f16) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_kernel_create_with_metadata(F16) failed with code %d\n", res);
            goto fail_kernel;
        }
        {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.kernel_create_count++;
        }
    }

    // 2.2 Q5_0 Kernel Creation with SHA-256 Authentication (P0-5)
    {
        const char * env_path = getenv("AMEVA_SPIRV_Q5_0_PATH");
        const char * spv_path = env_path ? env_path : "/data/data/com.termux/files/home/matmul_q5_0_f32.spv";

        std::vector<uint32_t> spv_code;
        if (!load_spirv_binary(spv_path, spv_code)) {
            goto fail_kernel;
        }

        const char * allow_unverified = getenv("AMEVA_ALLOW_UNVERIFIED_SPIRV");
        bool enforce_sha256 = (allow_unverified == nullptr || strcmp(allow_unverified, "1") != 0);

        AcKernelMetadata meta = {};
        meta.kernel_name = "matmul_q5_0_f32";
        meta.entry_point = "main";
        meta.binding_count = 3;
        meta.bindings = nullptr;
        meta.push_constant_size = sizeof(MatMulPushConstants);
        meta.local_size = {16, 16, 1};
        meta.required_features = AC_FEATURE_NONE;
        meta.expected_spirv_sha256 = enforce_sha256 ? kQ50SpirvSha256 : nullptr;

        res = ac_vk_kernel_create_with_metadata(ctx->vk_ctx, spv_code.data(), spv_code.size() * sizeof(uint32_t), &meta, &ctx->kernel_q5_0);
        if (res != AC_SUCCESS || !ctx->kernel_q5_0) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_kernel_create_with_metadata(Q5_0) failed with code %d\n", res);
            goto fail_kernel;
        }
        {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.kernel_create_count++;
        }
    }

    // Stage 3: Weight Buffer Allocation
    if (init_fail_stage && strcmp(init_fail_stage, "buf_w") == 0) {
        fprintf(stderr, "[AMEVA_INIT_FAIL_INJECTION] Simulated failure at stage: buf_w\n");
        goto fail_buf_w;
    }
    res = ac_vk_buffer_create(ctx->vk_ctx, INITIAL_BUF_CAP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ctx->buf_w);
    if (res != AC_SUCCESS || !ctx->buf_w) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_create(ctx->buf_w) failed with code %d\n", res);
        goto fail_buf_w;
    }
    ctx->buf_w_cap = INITIAL_BUF_CAP;
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_create_count++;
    }

    // Stage 4: Activation Buffer Allocation
    if (init_fail_stage && strcmp(init_fail_stage, "buf_a") == 0) {
        fprintf(stderr, "[AMEVA_INIT_FAIL_INJECTION] Simulated failure at stage: buf_a\n");
        goto fail_buf_a;
    }
    res = ac_vk_buffer_create(ctx->vk_ctx, INITIAL_BUF_CAP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ctx->buf_a);
    if (res != AC_SUCCESS || !ctx->buf_a) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_create(ctx->buf_a) failed with code %d\n", res);
        goto fail_buf_a;
    }
    ctx->buf_a_cap = INITIAL_BUF_CAP;
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_create_count++;
    }

    // Stage 5: Output Buffer Allocation
    if (init_fail_stage && strcmp(init_fail_stage, "buf_o") == 0) {
        fprintf(stderr, "[AMEVA_INIT_FAIL_INJECTION] Simulated failure at stage: buf_o\n");
        goto fail_buf_o;
    }
    res = ac_vk_buffer_create(ctx->vk_ctx, INITIAL_BUF_CAP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ctx->buf_o);
    if (res != AC_SUCCESS || !ctx->buf_o) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_create(ctx->buf_o) failed with code %d\n", res);
        goto fail_buf_o;
    }
    ctx->buf_o_cap = INITIAL_BUF_CAP;
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_create_count++;
    }

    return true;

fail_buf_o:
    if (ctx->buf_a) {
        ac_vk_buffer_destroy(ctx->buf_a);
        ctx->buf_a = nullptr;
        ctx->buf_a_cap = 0;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
fail_buf_a:
    if (ctx->buf_w) {
        ac_vk_buffer_destroy(ctx->buf_w);
        ctx->buf_w = nullptr;
        ctx->buf_w_cap = 0;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
fail_buf_w:
    if (ctx->kernel_q5_0) {
        ac_vk_kernel_destroy(ctx->kernel_q5_0);
        ctx->kernel_q5_0 = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.kernel_destroy_count++;
    }
    if (ctx->kernel_f16) {
        ac_vk_kernel_destroy(ctx->kernel_f16);
        ctx->kernel_f16 = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.kernel_destroy_count++;
    }
fail_kernel:
    if (ctx->vk_ctx) {
        ac_vk_context_destroy(ctx->vk_ctx);
        ctx->vk_ctx = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.context_destroy_count++;
    }
    return false;
}

// Teardown specific backend context resources (P0-4)
static void teardown_backend_context(struct ameva_backend_context * ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(ctx->dispatch_mutex);
    for (auto & kv : ctx->weight_cache) {
        if (kv.second.buf) {
            ac_vk_buffer_destroy(kv.second.buf);
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.buffer_destroy_count++;
            g_counters.weight_cache_entry_destroy_count++;
        }
    }
    ctx->weight_cache.clear();

    if (ctx->buf_w) {
        ac_vk_buffer_destroy(ctx->buf_w);
        ctx->buf_w = nullptr;
        ctx->buf_w_cap = 0;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
    if (ctx->buf_a) {
        ac_vk_buffer_destroy(ctx->buf_a);
        ctx->buf_a = nullptr;
        ctx->buf_a_cap = 0;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
    if (ctx->buf_o) {
        ac_vk_buffer_destroy(ctx->buf_o);
        ctx->buf_o = nullptr;
        ctx->buf_o_cap = 0;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_destroy_count++;
    }
    if (ctx->kernel_q5_0) {
        ac_vk_kernel_destroy(ctx->kernel_q5_0);
        ctx->kernel_q5_0 = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.kernel_destroy_count++;
    }
    if (ctx->kernel_f16) {
        ac_vk_kernel_destroy(ctx->kernel_f16);
        ctx->kernel_f16 = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.kernel_destroy_count++;
    }
    if (ctx->vk_ctx) {
        ac_vk_context_destroy(ctx->vk_ctx);
        ctx->vk_ctx = nullptr;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.context_destroy_count++;
    }
    ctx->initialized = false;
}

// Releases all AMEVA GPU state in the process: pipeline, buffers and weight cache of every live backend context,
// and the shared context of ameva_backend_run_cached_gpu_matmul. Live backends stay valid; their next
// graph_compute re-initializes the hardware pipeline (teardown_backend_context clears ctx->initialized).
// Backends release their own state in ameva_backend_free, so ordinary shutdown does not need this call.
void ameva_backend_teardown_hardware(void) {
    {
        std::lock_guard<std::mutex> lock(s_live_ctx_mutex);
        for (struct ameva_backend_context * ctx : s_live_ctxs) {
            teardown_backend_context(ctx);
        }
    }
    std::lock_guard<std::mutex> lock(s_test_ctx_mutex);
    if (s_cached_test_ctx) {
        teardown_backend_context(s_cached_test_ctx);
        delete s_cached_test_ctx;
        s_cached_test_ctx = nullptr;
    }
    std::lock_guard<std::mutex> lk(s_diag_mutex);
    s_cache_diagnostics.entry_count = 0;
}

static bool execute_vulkan_matmul(struct ameva_backend_context * ctx, struct ggml_tensor * t, uint32_t graph_run_id, uint64_t dispatch_index, int node_index) {
    if (!ctx || !t || !ctx->vk_ctx || !ctx->buf_w || !ctx->buf_a || !ctx->buf_o) {
        return false;
    }

    // P0-2: Fail Closed Fault Injection Oracle
    const char * fail_stage = getenv("AMEVA_RUNTIME_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "dispatch") == 0) {
        fprintf(stderr, "[AMEVA_FAIL_INJECTION] AMEVA_FAIL_CLOSED_REASON=INJECTED_DISPATCH_FAILURE\n");
        return false;
    }

    AcVkKernel * selected_kernel = nullptr;
    if (t->src[0]->type == GGML_TYPE_F16) {
        selected_kernel = ctx->kernel_f16;
    } else if (t->src[0]->type == GGML_TYPE_Q5_0) {
        selected_kernel = ctx->kernel_q5_0;
    } else {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Unsupported src0 tensor type %d for dispatch\n", (int)t->src[0]->type);
        return false;
    }
    if (!selected_kernel) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Selected Vulkan kernel is NULL for type %d\n", (int)t->src[0]->type);
        return false;
    }

    const char * current_kind = s_graph_kind;
    if (strcmp(current_kind, "UNKNOWN") == 0) {
        const char * env_kind = getenv("AMEVA_GRAPH_KIND");
        if (env_kind) {
            current_kind = env_kind;
        } else {
            const char * n = (t->name[0] ? t->name : (t->src[0]->name[0] ? t->src[0]->name : ""));
            if (strstr(n, "cross") || strstr(n, "cross_attn")) {
                current_kind = "CROSS";
            } else if (strstr(n, "encoder") || strstr(n, "enc")) {
                current_kind = "ENCODER";
            } else {
                current_kind = "UNKNOWN";
            }
        }
    }

    if (strcmp(current_kind, "UNKNOWN") == 0) {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.unknown_graph_kind_count++;
    }

    uint32_t K = (uint32_t)t->src[0]->ne[0];
    uint32_t N = (uint32_t)t->src[0]->ne[1];
    uint32_t M = (uint32_t)t->src[1]->ne[1];

    size_t weight_bytes = ggml_nbytes(t->src[0]);
    size_t act_bytes    = ggml_nbytes(t->src[1]);
    size_t out_bytes    = ggml_nbytes(t);

    if (!ensure_buffer_capacity(ctx, &ctx->buf_a, &ctx->buf_a_cap, act_bytes,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !ensure_buffer_capacity(ctx, &ctx->buf_o, &ctx->buf_o_cap, out_bytes,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Buffer capacity expansion failed\n");
        return false;
    }

    // 1. GPU Weight Cache Lookup & Staging using full SHA-256 + Shape + Dtype Identity
    if (!t->src[0]) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Missing weight tensor t->src[0]\n");
        return false;
    }
    std::vector<uint8_t> weight_host(weight_bytes);
    if (t->src[0]->data) {
        memcpy(weight_host.data(), t->src[0]->data, weight_bytes);
    } else if (t->src[0]->buffer) {
        ggml_backend_tensor_get(t->src[0], weight_host.data(), 0, weight_bytes);
    } else {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Weight tensor has neither data nor buffer (fail-closed)\n");
        return false;
    }

    WeightCacheKey weight_key = {};
    weight_key.byte_size = weight_bytes;
    weight_key.type = t->src[0]->type;
    weight_key.ne[0] = t->src[0]->ne[0];
    weight_key.ne[1] = t->src[0]->ne[1];
    weight_key.ne[2] = t->src[0]->ne[2];
    weight_key.ne[3] = t->src[0]->ne[3];
    ac_vk_sha256_hash(weight_host.data(), weight_bytes, weight_key.sha256);

    AcVkBuffer * weight_gpu_buf = nullptr;

    {
        std::lock_guard<std::mutex> lk(s_diag_mutex);
        s_cache_diagnostics.lookup_count++;
    }

    auto it_w = ctx->weight_cache.find(weight_key);
    if (it_w != ctx->weight_cache.end()) {
        {
            std::lock_guard<std::mutex> lk(s_diag_mutex);
            s_cache_diagnostics.hash_match_count++;
            s_cache_diagnostics.payload_recheck_count++;
        }
        // P0-1: Cache hit payload identity re-check to guarantee zero collision
        if (it_w->second.size != weight_bytes ||
            it_w->second.payload.size() != weight_bytes ||
            memcmp(weight_host.data(), it_w->second.payload.data(), weight_bytes) != 0) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: Cache collision or payload corruption detected!\n");
            {
                std::lock_guard<std::mutex> lk(s_diag_mutex);
                s_cache_diagnostics.payload_mismatch_count++;
                s_cache_diagnostics.alias_count++;
            }
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.cache_key_alias_count++;
            return false; // Fail-closed!
        }
        weight_gpu_buf = it_w->second.buf;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.weight_cache_hit_count++;
    } else {
        {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.weight_cache_miss_count++;
            if (s_current_inference_id <= 1) {
                g_counters.weight_upload_count_first_run++;
                g_counters.inference_run_1_weight_upload_count++;
            } else {
                g_counters.weight_upload_count_second_run++;
                g_counters.inference_run_2_weight_upload_count++;
            }
        }
        AcVkBuffer * new_weight_buf = nullptr;
        // rounded up to 4 bytes: Q5_0 rows are 22-byte blocks and matmul_q5_0_f32.comp reads whole 32-bit words
        AcResult res_w = ac_vk_buffer_create(ctx->vk_ctx, (weight_bytes + 3) & ~(size_t)3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &new_weight_buf);
        if (res_w != AC_SUCCESS || !new_weight_buf) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_create for cached weight failed: %d\n", res_w);
            return false;
        }
        {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.buffer_create_count++;
            g_counters.weight_cache_entry_create_count++;
        }

        double tw0 = ameva_get_time_ms();
        void * w_ptr = nullptr;
        res_w = ac_vk_buffer_map(new_weight_buf, &w_ptr);
        if (res_w != AC_SUCCESS || !w_ptr) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_map for cached weight failed: %d\n", res_w);
            ac_vk_buffer_destroy(new_weight_buf);
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.buffer_destroy_count++;
            g_counters.weight_cache_entry_destroy_count++;
            return false;
        }
        memcpy(w_ptr, weight_host.data(), weight_bytes);
        res_w = ac_vk_buffer_flush(new_weight_buf, 0, weight_bytes);
        if (res_w != AC_SUCCESS) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_flush for cached weight failed: %d\n", res_w);
            ac_vk_buffer_destroy(new_weight_buf);
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.buffer_destroy_count++;
            g_counters.weight_cache_entry_destroy_count++;
            return false;
        }
        double tw1 = ameva_get_time_ms();
        {
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.weight_upload_time_ms += (tw1 - tw0);
            g_counters.weight_upload_bytes += weight_bytes;
        }
        CachedWeightBuffer entry;
        entry.buf = new_weight_buf;
        entry.cap = weight_bytes;
        entry.size = weight_bytes;
        entry.key = weight_key;
        entry.payload = std::move(weight_host);
        ctx->weight_cache[weight_key] = std::move(entry);
        {
            std::lock_guard<std::mutex> lk(s_diag_mutex);
            s_cache_diagnostics.entry_count = ctx->weight_cache.size();
        }
        weight_gpu_buf = new_weight_buf;
    }

    // 2. Upload activations with timing telemetry
    double ta0 = ameva_get_time_ms();
    void * a_ptr = nullptr;
    AcResult res = ac_vk_buffer_map(ctx->buf_a, &a_ptr);
    if (res != AC_SUCCESS || !a_ptr) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_map(buf_a) failed with code %d\n", res);
        return false;
    }
    if (t->src[1]->buffer) {
        ggml_backend_tensor_get(t->src[1], a_ptr, 0, act_bytes);
    } else if (t->src[1]->data) {
        memcpy(a_ptr, t->src[1]->data, act_bytes);
    } else {
        return false;
    }
    res = ac_vk_buffer_flush(ctx->buf_a, 0, act_bytes);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_flush(buf_a) failed with code %d\n", res);
        return false;
    }
    double ta1 = ameva_get_time_ms();
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.activation_upload_time_ms += (ta1 - ta0);
        g_counters.activation_upload_bytes += act_bytes;
        g_counters.internal_activation_upload_count++;
    }

    // 3. Dispatch Vulkan compute pipeline with timing telemetry
    MatMulPushConstants pc = { M, N, K };
    AcVkBuffer * bufs[3] = { weight_gpu_buf, ctx->buf_a, ctx->buf_o };
    uint32_t group_x = (N + 15) / 16;
    uint32_t group_y = (M + 15) / 16;
    uint32_t group_z = 1;

    double td0 = ameva_get_time_ms();
    AcVkEvent * event = nullptr;
    res = ac_vk_dispatch(ctx->vk_ctx, selected_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, group_z, &event);
    double td1 = ameva_get_time_ms();
    if (res != AC_SUCCESS || !event) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_dispatch failed with code %d\n", res);
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.dispatch_api_time_ms += (td1 - td0);
        g_counters.event_create_count++;
        // ac_vk_dispatch allocates, records and submits one command buffer with one completion fence per call
        g_counters.graph_command_buffer_count++;
        g_counters.graph_completion_event_count++;
    }

    // 4. Wait on GPU event with timing telemetry
    if (fail_stage && strcmp(fail_stage, "wait") == 0) {
        fprintf(stderr, "[AMEVA_FAIL_INJECTION] AMEVA_FAIL_CLOSED_REASON=INJECTED_WAIT_FAILURE\n");
        ac_vk_event_destroy(event);
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.event_destroy_count++;
        g_counters.event_create_destroy_count++;
        return false;
    }
    double te0 = ameva_get_time_ms();
    res = ac_vk_event_wait(event, 5000000000ULL);
    double te1 = ameva_get_time_ms();
    ac_vk_event_destroy(event);
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.event_destroy_count++;
        g_counters.event_create_destroy_count++;
        g_counters.event_wait_time_ms += (te1 - te0);
        g_counters.node_level_event_wait_count++; // the host waits for every node's dispatch before the next one
    }
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_event_wait failed with code %d\n", res);
        return false;
    }

    // 5. Invalidate device memory and read back output with timing telemetry
    if (fail_stage && strcmp(fail_stage, "readback") == 0) {
        fprintf(stderr, "[AMEVA_FAIL_INJECTION] AMEVA_FAIL_CLOSED_REASON=INJECTED_READBACK_FAILURE\n");
        return false;
    }
    double to0 = ameva_get_time_ms();
    res = ac_vk_buffer_invalidate(ctx->buf_o, 0, out_bytes);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_invalidate(buf_o) failed with code %d\n", res);
        return false;
    }

    if (t->data && t->data != ctx->buf_o->mapped_ptr) {
        res = ac_vk_buffer_readback(ctx->buf_o, t->data, 0, out_bytes);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_readback failed with code %d\n", res);
            return false;
        }
    } else {
        std::vector<uint8_t> s_readback_temp(out_bytes);
        res = ac_vk_buffer_readback(ctx->buf_o, s_readback_temp.data(), 0, out_bytes);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_buffer_readback failed with code %d\n", res);
            return false;
        }
        if (t->buffer) {
            ggml_backend_tensor_set(t, s_readback_temp.data(), 0, out_bytes);
        }
    }
    double to1 = ameva_get_time_ms();
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.output_readback_time_ms += (to1 - to0);
        g_counters.output_readback_bytes += out_bytes;
        g_counters.internal_output_readback_count++;
    }

    const char * tensor_name = t->name[0] ? t->name : (t->src[0]->name[0] ? t->src[0]->name : "unnamed");
    printf("[AMEVA_DISPATCH] GRAPH_KIND=%s GRAPH_RUN_ID=%u DISPATCH_INDEX=%llu BACKEND_SUBGRAPH_NODE_INDEX=%d NODE_NAME=%s OP=%d SRC0_TYPE=%d SRC1_TYPE=%d DST_TYPE=%d SRC0_NE=[%lld,%lld,%lld,%lld] SRC1_NE=[%lld,%lld,%lld,%lld] DST_NE=[%lld,%lld,%lld,%lld] DISPATCH_RESULT=0 WAIT_RESULT=0 INVALIDATE_RESULT=0\n",
           current_kind, graph_run_id, (unsigned long long)dispatch_index, node_index,
           tensor_name,
           (int)t->op, (int)t->src[0]->type, (int)t->src[1]->type, (int)t->type,
           (long long)t->src[0]->ne[0], (long long)t->src[0]->ne[1], (long long)t->src[0]->ne[2], (long long)t->src[0]->ne[3],
           (long long)t->src[1]->ne[0], (long long)t->src[1]->ne[1], (long long)t->src[1]->ne[2], (long long)t->src[1]->ne[3],
           (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    fflush(stdout);

    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        if (strcmp(current_kind, "ENCODER") == 0) {
            g_counters.encoder_dispatch_count++;
        } else if (strcmp(current_kind, "CROSS") == 0) {
            g_counters.cross_dispatch_count++;
        }
    }

    return true;
}

// ============================================================================
// 1. AMEVA Buffer Implementation with Strict Bounds Defense
// ============================================================================

struct ameva_buffer_context {
    void * ptr;
    size_t size;
};

static bool validate_buffer_bounds(struct ameva_buffer_context * ctx, struct ggml_tensor * tensor, size_t offset, size_t size) {
    if (!ctx || !ctx->ptr || !tensor) {
        s_bounds_violation = true;
        ac_runtime_set_error(AC_ERR_NULL_POINTER, AMEVA_BACKEND_OP_BUFFER_MAP, offset, size, 0);
        return false;
    }
    if (offset + size < offset) { // Integer overflow
        s_bounds_violation = true;
        ac_runtime_set_error(AC_ERR_INVALID_ARGUMENT, AMEVA_BACKEND_OP_COPY_HTOD, offset, size, 0);
        return false;
    }
    size_t nbytes = ggml_nbytes(tensor);
    if (offset + size > nbytes) { // Tensor overflow
        s_bounds_violation = true;
        ac_runtime_set_error(AC_ERR_INVALID_BUFFER_SIZE, AMEVA_BACKEND_OP_COPY_HTOD, offset, size, nbytes);
        return false;
    }
    if (tensor->data) {
        uintptr_t t_start = (uintptr_t)tensor->data + offset;
        uintptr_t t_end   = t_start + size;
        uintptr_t b_start = (uintptr_t)ctx->ptr;
        uintptr_t b_end   = b_start + ctx->size;
        if (t_start < b_start || t_end > b_end) { // Buffer bounds violation
            s_bounds_violation = true;
            ac_runtime_set_error(AC_ERR_INVALID_BUFFER_SIZE, AMEVA_BACKEND_OP_COPY_HTOD, offset, size, ctx->size);
            return false;
        }
    }
    return true;
}

static void ameva_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer && buffer->context) {
        struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
        if (ctx->ptr) {
            free(ctx->ptr);
            ctx->ptr = NULL;
        }
        free(ctx);
        buffer->context = NULL;
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_free_count++;
    }
}

static void * ameva_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return NULL;
    return ((struct ameva_buffer_context *)buffer->context)->ptr;
}

static void ameva_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (!buffer || !buffer->context) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, tensor, offset, size)) return;
    memset((char *)tensor->data + offset, value, size);
}

static void ameva_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !data) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, tensor, offset, size)) return;
    memcpy((char *)tensor->data + offset, data, size);
}

static void ameva_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !data) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, const_cast<struct ggml_tensor*>(tensor), offset, size)) return;
    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ameva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (ctx->ptr && ctx->size > 0) {
        memset(ctx->ptr, value, ctx->size);
    }
}

static const struct ggml_backend_buffer_i ameva_backend_buffer_i = {
    /* .free_buffer     = */ ameva_buffer_free_buffer,
    /* .get_base        = */ ameva_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ameva_buffer_memset_tensor,
    /* .set_tensor      = */ ameva_buffer_set_tensor,
    /* .get_tensor      = */ ameva_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ameva_buffer_clear,
    /* .reset           = */ NULL,
};

// ============================================================================
// 2. AMEVA Buffer Type Implementation
// ============================================================================

static const char * ameva_buft_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "AMEVA";
}

static ggml_backend_buffer_t ameva_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)malloc(sizeof(struct ameva_buffer_context));
    if (!ctx) return NULL;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, size != 0 ? size : 64) != 0 || !ptr) {
        free(ctx);
        return NULL;
    }

    ctx->ptr = ptr;
    ctx->size = size;
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.buffer_alloc_count++;
    }

    return ggml_backend_buffer_init(buft, ameva_backend_buffer_i, ctx, size);
}

static size_t ameva_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ameva_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return (size_t)256 * 1024 * 1024; // 256 MB conservative mobile device upper bound
}

static size_t ameva_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ameva_buft_is_host(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return true; // ARM UMA shared memory architecture
}

static const struct ggml_backend_buffer_type_i ameva_backend_buffer_type_i = {
    /* .get_name       = */ ameva_buft_get_name,
    /* .alloc_buffer   = */ ameva_buft_alloc_buffer,
    /* .get_alignment  = */ ameva_buft_get_alignment,
    /* .get_max_size   = */ ameva_buft_get_max_size,
    /* .get_alloc_size = */ ameva_buft_get_alloc_size,
    /* .is_host        = */ ameva_buft_is_host,
};

ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(void) {
    static struct ggml_backend_buffer_type s_buft = {
        /* .iface   = */ ameva_backend_buffer_type_i,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &s_buft;
}

// ============================================================================
// 3. AMEVA Backend Implementation (P0-4 Concurrency & Context Isolation)
// ============================================================================

static const char * ameva_backend_get_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "AMEVA";
}

static void ameva_backend_free(ggml_backend_t backend) {
    if (backend) {
        if (backend->context) {
            struct ameva_backend_context * ctx = (struct ameva_backend_context *)backend->context;
            {
                std::lock_guard<std::mutex> lock(s_live_ctx_mutex);
                s_live_ctxs.erase(ctx);
            }
            teardown_backend_context(ctx);
            delete ctx;
            backend->context = NULL;
        }
        free(backend);
    }
}

static void ameva_backend_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    std::lock_guard<std::mutex> lk(s_counters_mutex);
    g_counters.sync_count++;
}

static inline bool is_view_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static enum ggml_status ameva_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (!backend || !backend->context) return GGML_STATUS_FAILED;
    struct ameva_backend_context * ctx = (struct ameva_backend_context *)backend->context;
    std::lock_guard<std::mutex> lock(ctx->dispatch_mutex);

    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.graph_compute_call_count++;
    }
    if (!cgraph) return GGML_STATUS_FAILED;
    AmevaBackendErrorState err_state = ac_runtime_get_error_state();
    if (err_state.first_error != 0 || s_bounds_violation) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Prior buffer bounds/runtime error detected in generation %llu (err=%d, op=%d). Aborting.\n",
                (unsigned long long)err_state.generation, err_state.first_error, (int)err_state.operation);
        return GGML_STATUS_FAILED;
    }

    ctx->graph_run_id++;

    const int n_nodes = ggml_graph_n_nodes(cgraph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (!node) continue;
        if (is_view_op(node->op)) continue;

        if (ggml_backend_dev_supports_op(ggml_backend_ameva_device(), node)) {
            ctx->dispatch_index++;
            {
                std::lock_guard<std::mutex> lk(s_counters_mutex);
                g_counters.ameva_dispatch_attempt_count++;
            }

            // 1. Initialize hardware pipeline if not already active
            if (!ctx->initialized) {
                if (!init_hardware_pipeline(ctx)) {
                    std::lock_guard<std::mutex> lk(s_counters_mutex);
                    g_counters.ameva_dispatch_failure_count++;
                    fprintf(stderr, "[AMEVA_BACKEND] FATAL: Hardware pipeline initialization failed\n");
                    return GGML_STATUS_FAILED;
                }
                ctx->initialized = true;
            }

            // 2. Perform real Vulkan compute dispatch
            if (!execute_vulkan_matmul(ctx, node, ctx->graph_run_id, ctx->dispatch_index, i)) {
                std::lock_guard<std::mutex> lk(s_counters_mutex);
                g_counters.ameva_dispatch_failure_count++;
                fprintf(stderr, "[AMEVA_BACKEND] FATAL: Vulkan MatMul dispatch failed\n");
                return GGML_STATUS_FAILED;
            }

            // 3. Strict invariant: ONLY increment on successful GPU execution
            {
                std::lock_guard<std::mutex> lk(s_counters_mutex);
                g_counters.ameva_dispatch_success_count++;
                g_counters.ameva_gpu_op_count++;
                if (s_current_inference_id <= 1) {
                    g_counters.inference_run_1_dispatch_count++;
                } else {
                    g_counters.inference_run_2_dispatch_count++;
                }
                if (strcmp(s_graph_kind, "ENCODER") == 0) {
                    if (s_current_inference_id <= 1) {
                        g_counters.run_1_encoder_dispatch_count++;
                    } else {
                        g_counters.run_2_encoder_dispatch_count++;
                    }
                } else if (strcmp(s_graph_kind, "CROSS") == 0) {
                    if (s_current_inference_id <= 1) {
                        g_counters.run_1_cross_dispatch_count++;
                    } else {
                        g_counters.run_2_cross_dispatch_count++;
                    }
                } else if (strcmp(s_graph_kind, "DECODER") == 0) {
                    if (s_current_inference_id <= 1) {
                        g_counters.run_1_decoder_dispatch_count++;
                    } else {
                        g_counters.run_2_decoder_dispatch_count++;
                    }
                }
            }
        } else {
            // Fail-Fast if an unsupported node was scheduled to AMEVA backend
            std::lock_guard<std::mutex> lk(s_counters_mutex);
            g_counters.unsupported_node_encounters++;
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: Unsupported node in AMEVA subgraph: op=%d, name=%s\n",
                    (int)node->op, (node->name[0] ? node->name : "unnamed"));
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ameva_backend_i = {
    /* .get_name                = */ ameva_backend_get_name,
    /* .free                    = */ ameva_backend_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ameva_backend_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ameva_backend_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_t ggml_backend_ameva_init(const char * params) {
    UNUSED(params);
    struct ameva_backend_context * ctx = new (std::nothrow) ameva_backend_context();
    if (!ctx) return NULL;
    ctx->name = "AMEVA";
    ctx->vk_ctx = nullptr;
    ctx->kernel_f16 = nullptr;
    ctx->kernel_q5_0 = nullptr;
    ctx->buf_w = nullptr;
    ctx->buf_a = nullptr;
    ctx->buf_o = nullptr;
    ctx->buf_w_cap = 0;
    ctx->buf_a_cap = 0;
    ctx->buf_o_cap = 0;
    ctx->graph_run_id = 0;
    ctx->dispatch_index = 0;
    ctx->initialized = false;

    ggml_backend_t backend = (ggml_backend_t)calloc(1, sizeof(struct ggml_backend));
    if (!backend) {
        delete ctx;
        return NULL;
    }

    backend->iface = ameva_backend_i;
    backend->device = ggml_backend_ameva_device();
    backend->context = ctx;
    {
        std::lock_guard<std::mutex> lock(s_live_ctx_mutex);
        s_live_ctxs.insert(ctx);
    }

    return backend;
}

// ============================================================================
// 4. AMEVA Device Implementation with Strict Dtype/Shape Validation
// ============================================================================

static const char * ameva_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "AMEVA0";
}

static const char * ameva_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "AMEVA Vulkan STT Hardware Accelerator";
}

static void ameva_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    if (free)  *free  = 0;
    if (total) *total = 0;
}

static enum ggml_backend_dev_type ameva_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ameva_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    UNUSED(dev);
    if (!props) return;
    props->name = "AMEVA0";
    props->description = "AMEVA Vulkan STT Hardware Accelerator";
    props->memory_free = 0;
    props->memory_total = 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = NULL;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ameva_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    return ggml_backend_ameva_init(params);
}

static ggml_backend_buffer_type_t ameva_device_get_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_ameva_buffer_type();
}

static bool ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.supports_op_query_count++;
    }
    if (!op) return false;

    // 1. Operation must be GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) return false;

    const char * trace_env = getenv("AMEVA_TRACE_GRAPH");
    bool do_trace = (trace_env != nullptr && strcmp(trace_env, "1") == 0);

    const char * fail_reason = nullptr;

    // 2. Strict Dtype Validation: SRC0 in {F16, Q5_0}, SRC1=F32, DST=F32
    if (!op->src[0] || !op->src[1]) fail_reason = "NULL_OPERANDS";
    else if (op->src[0]->type != GGML_TYPE_F16 && op->src[0]->type != GGML_TYPE_Q5_0) fail_reason = "SRC0_UNSUPPORTED_TYPE";
    else if (op->src[1]->type != GGML_TYPE_F32) fail_reason = "SRC1_NOT_F32";
    else if (op->type != GGML_TYPE_F32) fail_reason = "DST_NOT_F32";

    // 3. Batch dimensions: Must be 1 (2D matrix multiplication only)
    else if (op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1) fail_reason = "SRC0_BATCH_DIM_GT_1";
    else if (op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1) fail_reason = "SRC1_BATCH_DIM_GT_1";
    else if (op->ne[2] != 1 || op->ne[3] != 1) fail_reason = "DST_BATCH_DIM_GT_1";

    // 4. Dimension compatibility
    int64_t K0 = op->src[0] ? op->src[0]->ne[0] : 0;
    int64_t N  = op->src[0] ? op->src[0]->ne[1] : 0;
    int64_t K1 = op->src[1] ? op->src[1]->ne[0] : 0;
    int64_t M  = op->src[1] ? op->src[1]->ne[1] : 0;

    if (!fail_reason) {
        if (K0 <= 0 || N <= 0 || K1 <= 0 || M <= 0) fail_reason = "INVALID_DIM_LE_ZERO";
        else if (K0 != K1) fail_reason = "K_DIM_MISMATCH";
        else if (op->ne[0] != N || op->ne[1] != M) fail_reason = "DST_DIM_MISMATCH";
        else if ((uint64_t)K0 > 65536 || (uint64_t)N > 65536 || (uint64_t)M > 65536) fail_reason = "DIM_OVERFLOW";
        else {
            // Check Q5_0 specific contract
            if (op->src[0]->type == GGML_TYPE_Q5_0) {
                const char * q5_contract = getenv("AMEVA_Q5_0_CONTRACT");
                if (!q5_contract || strcmp(q5_contract, "none") == 0 || strcmp(q5_contract, "0") == 0) {
                    fail_reason = "Q5_0_CONTRACT_DISABLED_DEFAULT_NONE";
                } else if (strcmp(q5_contract, "restricted") == 0) {
                    // Restricted contract: only accept proven 1280x1280 (query weight)
                    if (K0 != 1280 || N != 1280) {
                        fail_reason = "Q5_0_CONTRACT_RESTRICTED_TO_1280x1280";
                    }
                } else if (strcmp(q5_contract, "all") == 0) {
                    // All contract: only accept shapes in approved shape registry
                    bool in_registry = (K0 == 1280 && N == 1280) ||
                                       (K0 == 1280 && N == 5120) ||
                                       (K0 == 5120 && N == 1280);
                    if (!in_registry) {
                        fail_reason = "Q5_0_SHAPE_NOT_IN_APPROVED_REGISTRY";
                    }
                } else {
                    fail_reason = "Q5_0_UNKNOWN_CONTRACT_MODE";
                }
                if (!fail_reason && (K0 % 32 != 0)) {
                    fail_reason = "SRC0_Q5_0_K_NOT_MULTIPLE_OF_32";
                }
            }

            if (!fail_reason) {
                uint32_t group_x = ((uint32_t)N + 15) / 16;
                uint32_t group_y = ((uint32_t)M + 15) / 16;
                if (group_x == 0 || group_x > 65535 || group_y == 0 || group_y > 65535) fail_reason = "DISPATCH_GROUP_LIMIT";
                else {
                    size_t weight_bytes = ggml_nbytes(op->src[0]);
                    uint64_t act_elems    = (uint64_t)K0 * (uint64_t)M;
                    uint64_t out_elems    = (uint64_t)N  * (uint64_t)M;
                    size_t act_bytes    = act_elems * sizeof(float);
                    size_t out_bytes    = out_elems * sizeof(float);
                    const size_t MAX_BUFFER_BUDGET = 64 * 1024 * 1024;
                    if (weight_bytes > MAX_BUFFER_BUDGET || act_bytes > MAX_BUFFER_BUDGET || out_bytes > MAX_BUFFER_BUDGET) {
                        fail_reason = "BUFFER_BUDGET_EXCEEDED";
                    } else if (op->src[0]->type == GGML_TYPE_F16 && (!ggml_is_contiguous(op->src[0]) || op->src[0]->nb[0] != sizeof(ggml_fp16_t) || op->src[0]->nb[1] != (size_t)K0 * sizeof(ggml_fp16_t))) {
                        fail_reason = "SRC0_F16_UNSUPPORTED_STRIDE_OR_NON_CONTIGUOUS";
                    } else if (op->src[0]->type == GGML_TYPE_Q5_0 && (!ggml_is_contiguous(op->src[0]) || op->src[0]->nb[0] != 22 || op->src[0]->nb[1] != (size_t)(K0 / 32) * 22)) {
                        fail_reason = "SRC0_Q5_0_UNSUPPORTED_STRIDE_OR_NON_CONTIGUOUS";
                    } else if (!ggml_is_contiguous(op->src[1]) || op->src[1]->nb[0] != sizeof(float) || op->src[1]->nb[1] != (size_t)K0 * sizeof(float)) {
                        fail_reason = "SRC1_UNSUPPORTED_STRIDE_OR_NON_CONTIGUOUS";
                    } else if (!ggml_is_contiguous(op) || op->nb[0] != sizeof(float) || op->nb[1] != (size_t)N * sizeof(float)) {
                        fail_reason = "DST_UNSUPPORTED_STRIDE_OR_NON_CONTIGUOUS";
                    }
                }
            }
        }
    }

    if (do_trace) {
        const char * n = (op->name[0] ? op->name : (op->src[0] && op->src[0]->name[0] ? op->src[0]->name : "unnamed"));
        int s0_type = op->src[0] ? (int)op->src[0]->type : -1;
        int s1_type = op->src[1] ? (int)op->src[1]->type : -1;
        printf("[AMEVA_TRACE_OP] GRAPH_KIND=%s NODE_NAME=%s SRC0_TYPE=%d SRC1_TYPE=%d DST_TYPE=%d SRC0_NE=[%lld,%lld,%lld,%lld] SRC1_NE=[%lld,%lld,%lld,%lld] DST_NE=[%lld,%lld,%lld,%lld] SUPPORTED=%s REASON=%s\n",
               s_graph_kind, n, s0_type, s1_type, (int)op->type,
               op->src[0] ? (long long)op->src[0]->ne[0] : 0LL, op->src[0] ? (long long)op->src[0]->ne[1] : 0LL, op->src[0] ? (long long)op->src[0]->ne[2] : 0LL, op->src[0] ? (long long)op->src[0]->ne[3] : 0LL,
               op->src[1] ? (long long)op->src[1]->ne[0] : 0LL, op->src[1] ? (long long)op->src[1]->ne[1] : 0LL, op->src[1] ? (long long)op->src[1]->ne[2] : 0LL, op->src[1] ? (long long)op->src[1]->ne[3] : 0LL,
               (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3],
               fail_reason ? "false" : "true", fail_reason ? fail_reason : "AMEVA_SUPPORTED");
        fflush(stdout);
    }

    if (fail_reason != nullptr) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        g_counters.supported_pattern_match_count++;
    }
    return true;
}

static bool ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    UNUSED(dev);
    return buft == ggml_backend_ameva_buffer_type();
}

static bool ameva_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ameva_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ameva_backend_device_i = {
    /* .get_name              = */ ameva_device_get_name,
    /* .get_description       = */ ameva_device_get_description,
    /* .get_memory            = */ ameva_device_get_memory,
    /* .get_type              = */ ameva_device_get_type,
    /* .get_props             = */ ameva_device_get_props,
    /* .init_backend          = */ ameva_device_init_backend,
    /* .get_buffer_type       = */ ameva_device_get_buffer_type,
    /* .get_host_buffer_type  = */ NULL,
    /* .buffer_from_host_ptr  = */ NULL,
    /* .supports_op           = */ ameva_device_supports_op,
    /* .supports_buft         = */ ameva_device_supports_buft,
    /* .offload_op            = */ ameva_device_offload_op,
    /* .event_new             = */ NULL,
    /* .event_free            = */ NULL,
    /* .event_synchronize     = */ NULL,
};

ggml_backend_dev_t ggml_backend_ameva_device(void) {
    static struct ggml_backend_device s_dev = {
        /* .iface   = */ ameva_backend_device_i,
        /* .reg     = */ NULL,
        /* .context = */ NULL,
    };
    return &s_dev;
}

// ============================================================================
// 5. AMEVA Backend Registration Implementation
// ============================================================================

static const char * ameva_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "AMEVA";
}

static size_t ameva_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ameva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    UNUSED(reg);
    if (index == 0) {
        return ggml_backend_ameva_device();
    }
    return NULL;
}

static const struct ggml_backend_reg_i ameva_backend_reg_i = {
    /* .get_name         = */ ameva_reg_get_name,
    /* .get_device_count = */ ameva_reg_get_device_count,
    /* .get_device       = */ ameva_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_ameva_reg(void) {
    static struct ggml_backend_reg s_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ameva_backend_reg_i,
        /* .context     = */ NULL,
    };
    return &s_reg;
}

void ameva_backend_get_counters(AmevaBackendCounters * out_counters) {
    if (out_counters) {
        std::lock_guard<std::mutex> lk(s_counters_mutex);
        *out_counters = g_counters;
    }
}

void ameva_backend_reset_counters(void) {
    std::lock_guard<std::mutex> lk(s_counters_mutex);
    g_counters.supports_op_query_count = 0;
    g_counters.supported_pattern_match_count = 0;
    g_counters.graph_compute_call_count = 0;
    g_counters.ameva_dispatch_attempt_count = 0;
    g_counters.ameva_dispatch_success_count = 0;
    g_counters.ameva_gpu_op_count = 0;
    g_counters.ameva_dispatch_failure_count = 0;
    g_counters.encoder_dispatch_count = 0;
    g_counters.cross_dispatch_count = 0;
    g_counters.unknown_graph_kind_count = 0;
    g_counters.weight_cache_hit_count = 0;
    g_counters.weight_cache_miss_count = 0;
    g_counters.weight_upload_count_first_run = 0;
    g_counters.weight_upload_count_second_run = 0;
    g_counters.internal_activation_upload_count = 0;
    g_counters.internal_output_readback_count = 0;
    g_counters.event_create_destroy_count = 0;
    g_counters.weight_upload_time_ms = 0.0;
    g_counters.activation_upload_time_ms = 0.0;
    g_counters.dispatch_api_time_ms = 0.0;
    g_counters.event_wait_time_ms = 0.0;
    g_counters.output_readback_time_ms = 0.0;
    g_counters.weight_upload_bytes = 0;
    g_counters.activation_upload_bytes = 0;
    g_counters.output_readback_bytes = 0;
    g_counters.cache_key_alias_count = 0;
    g_counters.weight_cache_entry_create_count = 0;
    g_counters.weight_cache_entry_destroy_count = 0;
    g_counters.inference_run_count = 0;
    g_counters.inference_run_1_dispatch_count = 0;
    g_counters.inference_run_2_dispatch_count = 0;
    g_counters.inference_run_1_weight_upload_count = 0;
    g_counters.inference_run_2_weight_upload_count = 0;
    g_counters.run_1_encoder_dispatch_count = 0;
    g_counters.run_1_cross_dispatch_count = 0;
    g_counters.run_1_decoder_dispatch_count = 0;
    g_counters.run_2_encoder_dispatch_count = 0;
    g_counters.run_2_cross_dispatch_count = 0;
    g_counters.run_2_decoder_dispatch_count = 0;
    g_counters.run_2_context_forward_pass_count = 0;
    g_counters.run_2_context_matmuls_per_pass = 0;
    g_counters.run_2_context_dispatch_count = 0;
    g_counters.graph_command_buffer_count = 0;
    g_counters.graph_completion_event_count = 0;
    g_counters.node_level_event_wait_count = 0;
    strcpy(s_graph_kind, "UNKNOWN");
    s_bounds_violation = false;
    s_current_inference_id = 1;
    s_inference_run_count = 0;
    s_decoder_pass_count[0] = s_decoder_pass_count[1] = 0;
}

void ameva_backend_begin_inference(ggml_backend_t backend, uint32_t inference_id) {
    (void)backend;
    std::lock_guard<std::mutex> lk(s_counters_mutex);
    s_current_inference_id = (inference_id == 0) ? 1 : inference_id;
    if (s_current_inference_id > s_inference_run_count) {
        s_inference_run_count = s_current_inference_id;
    }
    g_counters.inference_run_count = s_inference_run_count;
}

// Closes the scope opened by begin_inference and derives the run-2 "context" telemetry from measured counts:
// decoder passes run 2+ executed beyond run 1, times the dispatches one decoder pass issued. When the passes
// are not uniform the decomposition is undefined; it is then left at 0 and the reason is printed.
void ameva_backend_end_inference(ggml_backend_t backend, uint32_t inference_id) {
    (void)backend;
    std::lock_guard<std::mutex> lk(s_counters_mutex);
    const uint32_t id = (inference_id == 0) ? 1 : inference_id;
    if (id != s_current_inference_id) {
        fprintf(stderr, "[AMEVA_BACKEND] ERROR: end_inference(%u) does not close the open inference %u; "
                        "per-run telemetry after this point is misattributed\n", id, s_current_inference_id);
    }
    if (s_current_inference_id <= 1) return;
    const uint64_t p1 = s_decoder_pass_count[0], p2 = s_decoder_pass_count[1];
    const uint64_t d1 = g_counters.run_1_decoder_dispatch_count, d2 = g_counters.run_2_decoder_dispatch_count;
    const bool uniform = p1 > 0 && p2 > 0 && d1 % p1 == 0 && d2 % p2 == 0 && d1 / p1 == d2 / p2;
    if (!uniform) {
        fprintf(stderr, "[AMEVA_BACKEND] run-2 context decomposition undefined: decoder passes are not uniform "
                        "(run 1: %llu dispatches / %llu passes, run 2+: %llu / %llu)\n",
                (unsigned long long)d1, (unsigned long long)p1, (unsigned long long)d2, (unsigned long long)p2);
    }
    g_counters.run_2_context_forward_pass_count = (uniform && p2 > p1) ? p2 - p1 : 0;
    g_counters.run_2_context_matmuls_per_pass = uniform ? d2 / p2 : 0;
    g_counters.run_2_context_dispatch_count =
        g_counters.run_2_context_forward_pass_count * g_counters.run_2_context_matmuls_per_pass;
}

void ameva_backend_register(void) {
    static std::once_flag s_flag;
    std::call_once(s_flag, []() {
        ggml_backend_reg_t reg = ggml_backend_ameva_reg();
        ggml_backend_dev_t dev = ggml_backend_ameva_device();
        dev->reg = reg;
        ggml_backend_register(reg);
        ggml_backend_device_register(dev);
    });
}

bool ameva_backend_run_cached_gpu_matmul(
    const void * weight_data,
    size_t weight_bytes,
    const float * act_data,
    float * out_data,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    enum ggml_type type
) {
    if (!weight_data || !act_data || !out_data || M == 0 || N == 0 || K == 0) return false;

    // moved to file static
    std::lock_guard<std::mutex> lock(s_test_ctx_mutex);

    if (!s_cached_test_ctx) {
        s_cached_test_ctx = new (std::nothrow) ameva_backend_context();
        if (!s_cached_test_ctx) return false;
        s_cached_test_ctx->name = "AMEVA_TEST";
    }

    struct ameva_backend_context * ctx = s_cached_test_ctx;
    if (!ctx->initialized) {
        if (!init_hardware_pipeline(ctx)) {
            teardown_backend_context(ctx);
            delete ctx;
            s_cached_test_ctx = nullptr;
            return false;
        }
        ctx->initialized = true;
    }

    AcVkKernel * selected_kernel = (type == GGML_TYPE_Q5_0) ? ctx->kernel_q5_0 : ctx->kernel_f16;
    if (!selected_kernel) return false;

    size_t act_bytes = (size_t)M * (size_t)K * sizeof(float);
    size_t out_bytes = (size_t)M * (size_t)N * sizeof(float);

    if (!ensure_buffer_capacity(ctx, &ctx->buf_a, &ctx->buf_a_cap, act_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !ensure_buffer_capacity(ctx, &ctx->buf_o, &ctx->buf_o_cap, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    // 1. Weight Cache Lookup & Staging using full SHA-256 + Shape + Dtype Identity
    WeightCacheKey weight_key = {};
    weight_key.byte_size = weight_bytes;
    weight_key.type = type;
    weight_key.ne[0] = K;
    weight_key.ne[1] = N;
    weight_key.ne[2] = 1;
    weight_key.ne[3] = 1;
    ac_vk_sha256_hash(weight_data, weight_bytes, weight_key.sha256);

    AcVkBuffer * weight_gpu_buf = nullptr;

    {
        std::lock_guard<std::mutex> lk(s_diag_mutex);
        s_cache_diagnostics.lookup_count++;
    }

    auto it_w = ctx->weight_cache.find(weight_key);
    if (it_w != ctx->weight_cache.end()) {
        {
            std::lock_guard<std::mutex> lk(s_diag_mutex);
            s_cache_diagnostics.hash_match_count++;
            s_cache_diagnostics.payload_recheck_count++;
        }
        if (it_w->second.size != weight_bytes ||
            it_w->second.payload.size() != weight_bytes ||
            memcmp(weight_data, it_w->second.payload.data(), weight_bytes) != 0) {
            fprintf(stderr, "[AMEVA_BACKEND] FATAL: Cache collision or payload corruption detected in cached matmul\n");
            {
                std::lock_guard<std::mutex> lk(s_diag_mutex);
                s_cache_diagnostics.payload_mismatch_count++;
                s_cache_diagnostics.alias_count++;
            }
            std::lock_guard<std::mutex> clk(s_counters_mutex);
            g_counters.cache_key_alias_count++;
            return false;
        }
        weight_gpu_buf = it_w->second.buf;
        std::lock_guard<std::mutex> clk(s_counters_mutex);
        g_counters.weight_cache_hit_count++;
    } else {
        AcVkBuffer * new_weight_buf = nullptr;
        // rounded up to 4 bytes: Q5_0 rows are 22-byte blocks and matmul_q5_0_f32.comp reads whole 32-bit words
        AcResult res_w = ac_vk_buffer_create(ctx->vk_ctx, (weight_bytes + 3) & ~(size_t)3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &new_weight_buf);
        if (res_w != AC_SUCCESS || !new_weight_buf) {
            return false;
        }
        {
            std::lock_guard<std::mutex> clk(s_counters_mutex);
            g_counters.buffer_create_count++;
            g_counters.weight_cache_entry_create_count++;
            g_counters.weight_cache_miss_count++;
        }
        void * w_ptr = nullptr;
        res_w = ac_vk_buffer_map(new_weight_buf, &w_ptr);
        if (res_w != AC_SUCCESS || !w_ptr) {
            ac_vk_buffer_destroy(new_weight_buf);
            return false;
        }
        memcpy(w_ptr, weight_data, weight_bytes);
        res_w = ac_vk_buffer_flush(new_weight_buf, 0, weight_bytes);
        if (res_w != AC_SUCCESS) {
            ac_vk_buffer_destroy(new_weight_buf);
            return false;
        }

        CachedWeightBuffer entry;
        entry.buf = new_weight_buf;
        entry.cap = weight_bytes;
        entry.size = weight_bytes;
        entry.key = weight_key;
        entry.payload.assign((const uint8_t*)weight_data, (const uint8_t*)weight_data + weight_bytes);
        ctx->weight_cache[weight_key] = std::move(entry);
        {
            std::lock_guard<std::mutex> lk(s_diag_mutex);
            s_cache_diagnostics.entry_count = ctx->weight_cache.size();
        }
        weight_gpu_buf = new_weight_buf;
    }

    // 2. Upload activations
    void * a_ptr = nullptr;
    AcResult res = ac_vk_buffer_map(ctx->buf_a, &a_ptr);
    if (res != AC_SUCCESS || !a_ptr) return false;
    memcpy(a_ptr, act_data, act_bytes);
    res = ac_vk_buffer_flush(ctx->buf_a, 0, act_bytes);
    if (res != AC_SUCCESS) return false;

    // 3. Dispatch Vulkan compute pipeline
    MatMulPushConstants pc = { M, N, K };
    AcVkBuffer * bufs[3] = { weight_gpu_buf, ctx->buf_a, ctx->buf_o };
    uint32_t group_x = (N + 15) / 16;
    uint32_t group_y = (M + 15) / 16;
    uint32_t group_z = 1;

    AcVkEvent * event = nullptr;
    res = ac_vk_dispatch(ctx->vk_ctx, selected_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, group_z, &event);
    if (res != AC_SUCCESS || !event) return false;

    res = ac_vk_event_wait(event, 5000000000ULL);
    ac_vk_event_destroy(event);
    if (res != AC_SUCCESS) return false;

    // 4. Invalidate and readback
    res = ac_vk_buffer_invalidate(ctx->buf_o, 0, out_bytes);
    if (res != AC_SUCCESS) return false;

    res = ac_vk_buffer_readback(ctx->buf_o, out_data, 0, out_bytes);
    if (res != AC_SUCCESS) return false;

    return true;
}

// P0-4: Hardened Backend Error State API implementations
AmevaBackendErrorState ameva_backend_get_last_buffer_error(void) {
    return ac_runtime_get_error_state();
}

void ameva_backend_clear_buffer_error(void) {
    ac_runtime_reset_error_state();
    s_bounds_violation = false;
}

int32_t ameva_backend_consume_last_buffer_error(void) {
    s_bounds_violation = false;
    return ac_runtime_consume_error();
}


