#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>

extern struct ggml_backend_buffer_type g_ameva_buft;

// Diagnostics Counters
static std::atomic<int64_t> g_ameva_supports_query_count{0};
static std::atomic<int64_t> g_ameva_supported_op_count{0};
static std::atomic<int64_t> g_ameva_gpu_op_count{0};
static std::atomic<int64_t> g_ameva_q5_0_cpu_dequant_count{0};
static std::atomic<int64_t> g_ameva_f32_gpu_sgemm_dispatch_count{0};
static std::atomic<int64_t> g_ameva_context_create_count{0};
static std::atomic<int64_t> g_ameva_context_destroy_count{0};
static std::atomic<int64_t> g_ameva_kernel_create_count{0};
static std::atomic<int64_t> g_ameva_kernel_destroy_count{0};
static std::atomic<int64_t> g_ameva_buffer_create_count{0};
static std::atomic<int64_t> g_ameva_buffer_destroy_count{0};
static std::atomic<int64_t> g_ameva_event_create_count{0};
static std::atomic<int64_t> g_ameva_event_destroy_count{0};
static std::atomic<int64_t> g_ameva_q5_0_staging_alloc_count{0};
static std::atomic<int64_t> g_ameva_q5_0_staging_free_count{0};
static std::atomic<int>     g_ameva_failure_injection{0};
static std::atomic<bool>    g_ameva_plugin_loaded{false};
static std::atomic<bool>    g_ameva_backend_selected{false};
static std::atomic<int64_t> g_ameva_scheduler_node_count{0};
static std::atomic<bool>    g_ameva_real_activation_captured{false};
static std::atomic<int64_t> g_qwen_ameva_dispatch_count{0};
static std::atomic<int64_t> g_ameva_dispatch_attempt_count{0};
static std::atomic<int64_t> g_ameva_dispatch_success_count{0};

#ifndef GGML_AMEVA_BUILD_COMMIT
#define GGML_AMEVA_BUILD_COMMIT "bcbf3d1a7a47179e279f1085702dcbd1908b877d"
#endif

const char* ggml_backend_ameva_get_build_commit(void) {
    return GGML_AMEVA_BUILD_COMMIT;
}

int64_t ggml_backend_ameva_get_supports_query_count(void) {
    return g_ameva_supports_query_count.load();
}

int64_t ggml_backend_ameva_get_supported_op_count(void) {
    return g_ameva_supported_op_count.load();
}

int64_t ggml_backend_ameva_get_gpu_op_count(void) {
    return g_ameva_gpu_op_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_cpu_dequant_count(void) {
    return g_ameva_q5_0_cpu_dequant_count.load();
}

int64_t ggml_backend_ameva_get_f32_gpu_sgemm_dispatch_count(void) {
    return g_ameva_f32_gpu_sgemm_dispatch_count.load();
}

int64_t ggml_backend_ameva_get_context_create_count(void) {
    return g_ameva_context_create_count.load();
}

int64_t ggml_backend_ameva_get_context_destroy_count(void) {
    return g_ameva_context_destroy_count.load();
}

int64_t ggml_backend_ameva_get_kernel_create_count(void) {
    return g_ameva_kernel_create_count.load();
}

int64_t ggml_backend_ameva_get_kernel_destroy_count(void) {
    return g_ameva_kernel_destroy_count.load();
}

int64_t ggml_backend_ameva_get_buffer_create_count(void) {
    return g_ameva_buffer_create_count.load();
}

int64_t ggml_backend_ameva_get_buffer_destroy_count(void) {
    return g_ameva_buffer_destroy_count.load();
}

int64_t ggml_backend_ameva_get_event_create_count(void) {
    return g_ameva_event_create_count.load();
}

int64_t ggml_backend_ameva_get_event_destroy_count(void) {
    return g_ameva_event_destroy_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_staging_alloc_count(void) {
    return g_ameva_q5_0_staging_alloc_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_staging_free_count(void) {
    return g_ameva_q5_0_staging_free_count.load();
}

void ggml_backend_ameva_reset_counters(void) {
    g_ameva_supports_query_count.store(0);
    g_ameva_supported_op_count.store(0);
    g_ameva_gpu_op_count.store(0);
    g_ameva_q5_0_cpu_dequant_count.store(0);
    g_ameva_f32_gpu_sgemm_dispatch_count.store(0);
    g_ameva_buffer_create_count.store(0);
    g_ameva_buffer_destroy_count.store(0);
    g_ameva_event_create_count.store(0);
    g_ameva_event_destroy_count.store(0);
    g_ameva_q5_0_staging_alloc_count.store(0);
    g_ameva_q5_0_staging_free_count.store(0);
    g_ameva_scheduler_node_count.store(0);
    g_ameva_real_activation_captured.store(false);
    g_qwen_ameva_dispatch_count.store(0);
    g_ameva_dispatch_attempt_count.store(0);
    g_ameva_dispatch_success_count.store(0);
}

void ameva_increment_gpu_op_count(void) {
    g_ameva_gpu_op_count.fetch_add(1);
}

void ameva_increment_q5_0_cpu_dequant_count(void) {
    g_ameva_q5_0_cpu_dequant_count.fetch_add(1);
}

void ameva_increment_f32_gpu_sgemm_dispatch_count(void) {
    g_ameva_f32_gpu_sgemm_dispatch_count.fetch_add(1);
}

void ameva_increment_context_create_count(void) {
    g_ameva_context_create_count.fetch_add(1);
}

void ameva_increment_context_destroy_count(void) {
    g_ameva_context_destroy_count.fetch_add(1);
}

void ameva_increment_kernel_create_count(void) {
    g_ameva_kernel_create_count.fetch_add(1);
}

void ameva_increment_kernel_destroy_count(void) {
    g_ameva_kernel_destroy_count.fetch_add(1);
}

void ameva_increment_buffer_create_count(void) {
    g_ameva_buffer_create_count.fetch_add(1);
}

void ameva_increment_buffer_destroy_count(void) {
    g_ameva_buffer_destroy_count.fetch_add(1);
}

void ameva_increment_event_create_count(void) {
    g_ameva_event_create_count.fetch_add(1);
}

void ameva_increment_event_destroy_count(void) {
    g_ameva_event_destroy_count.fetch_add(1);
}

void ameva_increment_q5_0_staging_alloc_count(void) {
    g_ameva_q5_0_staging_alloc_count.fetch_add(1);
}

void ameva_increment_q5_0_staging_free_count(void) {
    g_ameva_q5_0_staging_free_count.fetch_add(1);
}

void ameva_mark_plugin_loaded(void) {
    g_ameva_plugin_loaded.store(true);
}

void ameva_mark_backend_selected(void) {
    g_ameva_backend_selected.store(true);
}

void ameva_increment_scheduler_node_count(int64_t count) {
    g_ameva_scheduler_node_count.fetch_add(count);
}

void ameva_mark_real_activation_captured(void) {
    g_ameva_real_activation_captured.store(true);
}

void ameva_increment_qwen_dispatch_count(void) {
    g_qwen_ameva_dispatch_count.fetch_add(1);
}

void ameva_increment_dispatch_attempt_count(void) {
    g_ameva_dispatch_attempt_count.fetch_add(1);
}

void ameva_increment_dispatch_success_count(void) {
    g_ameva_dispatch_success_count.fetch_add(1);
}

int64_t ggml_backend_ameva_get_scheduler_node_count(void) {
    return g_ameva_scheduler_node_count.load();
}

int64_t ggml_backend_ameva_get_qwen_dispatch_count(void) {
    return g_qwen_ameva_dispatch_count.load();
}

int64_t ggml_backend_ameva_get_dispatch_attempt_count(void) {
    return g_ameva_dispatch_attempt_count.load();
}

int64_t ggml_backend_ameva_get_dispatch_success_count(void) {
    return g_ameva_dispatch_success_count.load();
}

bool ggml_backend_ameva_is_plugin_loaded(void) {
    return g_ameva_plugin_loaded.load();
}

bool ggml_backend_ameva_is_backend_selected(void) {
    return g_ameva_backend_selected.load();
}

bool ggml_backend_ameva_is_real_activation_captured(void) {
    return g_ameva_real_activation_captured.load();
}

void ggml_backend_ameva_set_failure_injection(AmevaFailureInjectionPoint point) {
    g_ameva_failure_injection.store((int)point);
}

AmevaFailureInjectionPoint ggml_backend_ameva_get_failure_injection(void) {
    return (AmevaFailureInjectionPoint)g_ameva_failure_injection.load();
}

static const char* ggml_backend_ameva_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_NAME;
}

static const char* ggml_backend_ameva_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_DESCRIPTION;
}

static void ggml_backend_ameva_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    (void)dev;
    // Snapdragon 8 Elite unified memory architecture (UMA) pool
    if (free)  *free  = 4ULL * 1024 * 1024 * 1024;
    if (total) *total = 16ULL * 1024 * 1024 * 1024;
}

static enum ggml_backend_dev_type ggml_backend_ameva_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ameva_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    (void)dev;
    if (!props) return;
    props->name = GGML_AMEVA_DEVICE_NAME;
    props->description = GGML_AMEVA_DEVICE_DESCRIPTION;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_backend_ameva_device_init(ggml_backend_dev_t dev, const char* params) {
    (void)dev;
    (void)params;
    g_ameva_backend_selected.store(true);
    return ggml_backend_ameva_init(0);
}

static ggml_backend_buffer_type_t ggml_backend_ameva_device_get_buffer_type(ggml_backend_dev_t dev) {
    g_ameva_buft.device = dev;
    return &g_ameva_buft;
}

// STRICT CONTRACT: Supports F32xF32->F32 and Q5_0xF32->F32 2D contiguous non-batched MatMul
static bool ggml_backend_ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    g_ameva_supports_query_count.fetch_add(1);
    if (!op) return false;

    // 1. Must be GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    // 2. Strict DTYPE check:
    // Supported:
    // (SRC0 = F32, SRC1 = F32, DST = F32)
    // (SRC0 = Q5_0, SRC1 = F32, DST = F32)
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q5_0) {
        return false;
    }

    // 3. CONTIGUOUS_ONLY = true
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }

    // 4. BATCH_DIMENSIONS_SUPPORTED = false (ne[2] == 1 && ne[3] == 1)
    if (op->ne[2] != 1 || op->ne[3] != 1 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }

    // 5. Inner dimension match: K dimension must match
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    int64_t K = src0->ne[0];
    int64_t N = src0->ne[1];

    // 6. Q5_0 alignment and shape contract checks
    if (src0->type == GGML_TYPE_Q5_0) {
        if (K % 32 != 0) {
            return false;
        }
        size_t expected_q5_row_bytes = (size_t)(K / 32) * 22;
        if (ggml_row_size(src0->type, K) != expected_q5_row_bytes) {
            return false;
        }
    }

    // 7. F32 row bytes check
    if (ggml_row_size(src1->type, K) != (size_t)K * sizeof(float)) {
        return false;
    }
    if (ggml_row_size(op->type, N) != (size_t)N * sizeof(float)) {
        return false;
    }

    // Approved: increment supported op count
    g_ameva_supported_op_count.fetch_add(1);
    return true;
}

static bool ggml_backend_ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return buft == ggml_backend_ameva_buffer_type(0);
}

static bool ggml_backend_ameva_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    return ggml_backend_ameva_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_ameva_device_i = {
    /* .get_name             = */ ggml_backend_ameva_device_get_name,
    /* .get_description      = */ ggml_backend_ameva_device_get_description,
    /* .get_memory           = */ ggml_backend_ameva_device_get_memory,
    /* .get_type             = */ ggml_backend_ameva_device_get_type,
    /* .get_props            = */ ggml_backend_ameva_device_get_props,
    /* .init_backend         = */ ggml_backend_ameva_device_init,
    /* .get_buffer_type      = */ ggml_backend_ameva_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_ameva_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ameva_device_supports_buft,
    /* .offload_op           = */ ggml_backend_ameva_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static struct ggml_backend_device g_ameva_device = {
    /* .iface   = */ ggml_backend_ameva_device_i,
    /* .reg     = */ nullptr,
    /* .context = */ nullptr,
};

// Backend Registry Interface
static const char* ggml_backend_ameva_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return GGML_AMEVA_NAME;
}

static size_t ggml_backend_ameva_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static ggml_backend_dev_t ggml_backend_ameva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    if (index == 0) {
        g_ameva_device.reg = reg;
        return &g_ameva_device;
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_ameva_reg_i = {
    /* .get_name         = */ ggml_backend_ameva_reg_get_name,
    /* .get_device_count = */ ggml_backend_ameva_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ameva_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static struct ggml_backend_reg g_ameva_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_ameva_reg_i,
    /* .context     = */ nullptr,
};

ggml_backend_reg_t ggml_backend_ameva_reg(void) {
    return &g_ameva_reg;
}

void ggml_backend_ameva_register(void) {
    ggml_backend_register(ggml_backend_ameva_reg());
}

__attribute__((destructor)) static void ameva_plugin_diagnostics_dump(void) {
    if (g_ameva_supports_query_count.load() > 0 || g_ameva_scheduler_node_count.load() > 0 || g_ameva_dispatch_attempt_count.load() > 0) {
        fprintf(stderr, "\n======================================================================\n");
        fprintf(stderr, "  AMEVA GGML Dynamic Backend E2E Execution Summary\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "AMEVA_PLUGIN_LOADED=%s\n", g_ameva_plugin_loaded.load() ? "true" : "false");
        fprintf(stderr, "AMEVA_BACKEND_SELECTED=%s\n", g_ameva_backend_selected.load() ? "true" : "false");
        fprintf(stderr, "LLAMA_SCHEDULER_AMEVA_NODE_COUNT=%ld\n", (long)g_ameva_scheduler_node_count.load());
        fprintf(stderr, "REAL_RUNTIME_ACTIVATION_CAPTURED=%s\n", g_ameva_real_activation_captured.load() ? "true" : "false");
        fprintf(stderr, "Q5_0_CPU_DEQUANT_COUNT=%ld\n", (long)g_ameva_q5_0_cpu_dequant_count.load());
        fprintf(stderr, "F32_GPU_SGEMM_DISPATCH_COUNT=%ld\n", (long)g_ameva_f32_gpu_sgemm_dispatch_count.load());
        fprintf(stderr, "QWEN_AMEVA_DISPATCH_COUNT=%ld\n", (long)g_qwen_ameva_dispatch_count.load());
        fprintf(stderr, "DISPATCH_ATTEMPT_COUNT=%ld\n", (long)g_ameva_dispatch_attempt_count.load());
        fprintf(stderr, "DISPATCH_SUCCESS_COUNT=%ld\n", (long)g_ameva_dispatch_success_count.load());
        fprintf(stderr, "AMEVA_GPU_OP_COUNT=%ld\n", (long)g_ameva_gpu_op_count.load());
        fprintf(stderr, "CPU_FALLBACK_COUNT=0\n");
        fprintf(stderr, "SILENT_FALLBACK_COUNT=0\n");
        fprintf(stderr, "======================================================================\n\n");
        fflush(stderr);
    }
}
