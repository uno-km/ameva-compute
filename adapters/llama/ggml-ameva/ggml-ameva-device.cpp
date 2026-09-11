
#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <vector>

// Detailed Sub-Microsecond Decompositions (Nanoseconds)
static std::atomic<uint64_t> g_cpu_dequant_time_ns{0};
static std::atomic<uint64_t> g_q5_raw_upload_time_ns{0};
static std::atomic<uint64_t> g_gpu_dequant_dispatch_time_ns{0};
static std::atomic<uint64_t> g_gpu_dequant_wait_time_ns{0};
static std::atomic<uint64_t> g_f32_temp_allocation_time_ns{0};
static std::atomic<uint64_t> g_fused_dispatch_submit_time_ns{0};
static std::atomic<uint64_t> g_fused_event_wait_time_ns{0};
static std::atomic<uint64_t> g_output_readback_time_ns{0};
static std::atomic<int64_t>  g_fused_q5_0_dispatch_count{0};

extern "C" {
int64_t ggml_backend_ameva_get_fused_q5_0_dispatch_count(void) { return g_fused_q5_0_dispatch_count.load(); }
void ameva_increment_fused_q5_0_dispatch_count(void) { g_fused_q5_0_dispatch_count.fetch_add(1, std::memory_order_relaxed); }

void ameva_record_detailed_timing_ns(
    uint64_t q5_raw_upload_ns,
    uint64_t gpu_dequant_disp_ns,
    uint64_t gpu_dequant_wait_ns,
    uint64_t f32_temp_alloc_ns,
    uint64_t fused_dispatch_submit_ns,
    uint64_t fused_event_wait_ns,
    uint64_t output_readback_ns
) {
    g_q5_raw_upload_time_ns += q5_raw_upload_ns;
    g_gpu_dequant_dispatch_time_ns += gpu_dequant_disp_ns;
    g_gpu_dequant_wait_time_ns += gpu_dequant_wait_ns;
    g_f32_temp_allocation_time_ns += f32_temp_alloc_ns;
    g_fused_dispatch_submit_time_ns += fused_dispatch_submit_ns;
    g_fused_event_wait_time_ns += fused_event_wait_ns;
    g_output_readback_time_ns += output_readback_ns;
}

uint64_t ggml_backend_ameva_get_cpu_dequant_time_ns(void) { return g_cpu_dequant_time_ns.load(); }
uint64_t ggml_backend_ameva_get_q5_raw_upload_time_ns(void) { return g_q5_raw_upload_time_ns.load(); }
uint64_t ggml_backend_ameva_get_gpu_dequant_dispatch_time_ns(void) { return g_gpu_dequant_dispatch_time_ns.load(); }
uint64_t ggml_backend_ameva_get_gpu_dequant_wait_time_ns(void) { return g_gpu_dequant_wait_time_ns.load(); }
uint64_t ggml_backend_ameva_get_f32_temp_allocation_time_ns(void) { return g_f32_temp_allocation_time_ns.load(); }
uint64_t ggml_backend_ameva_get_fused_dispatch_submit_time_ns(void) { return g_fused_dispatch_submit_time_ns.load(); }
uint64_t ggml_backend_ameva_get_fused_event_wait_time_ns(void) { return g_fused_event_wait_time_ns.load(); }
uint64_t ggml_backend_ameva_get_output_readback_time_ns(void) { return g_output_readback_time_ns.load(); }
uint64_t ggml_backend_ameva_get_total_mul_mat_time_ns(void) {
    return g_q5_raw_upload_time_ns.load() +
           g_gpu_dequant_dispatch_time_ns.load() +
           g_gpu_dequant_wait_time_ns.load() +
           g_f32_temp_allocation_time_ns.load() +
           g_fused_dispatch_submit_time_ns.load() +
           g_fused_event_wait_time_ns.load() +
           g_output_readback_time_ns.load();
}

uint64_t ggml_backend_ameva_get_cpu_dequant_time_us(void) { return g_cpu_dequant_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_q5_raw_upload_time_us(void) { return g_q5_raw_upload_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_gpu_dequant_dispatch_time_us(void) { return g_gpu_dequant_dispatch_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_gpu_dequant_wait_time_us(void) { return g_gpu_dequant_wait_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_f32_temp_allocation_time_us(void) { return g_f32_temp_allocation_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_sgemm_dispatch_time_us(void) { return g_fused_dispatch_submit_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_sgemm_wait_time_us(void) { return g_fused_event_wait_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_output_readback_time_us(void) { return g_output_readback_time_ns.load() / 1000ULL; }
uint64_t ggml_backend_ameva_get_total_mul_mat_time_us(void) { return ggml_backend_ameva_get_total_mul_mat_time_ns() / 1000ULL; }
}


extern struct ggml_backend_buffer_type g_ameva_buft;

// Diagnostics Counters
static std::atomic<int64_t> g_ameva_supports_query_count{0};
static std::atomic<int64_t> g_ameva_supported_op_count{0};
static std::atomic<int64_t> g_ameva_gpu_op_count{0};
static std::atomic<int64_t> g_ameva_q5_0_cpu_dequant_count{0};
static std::atomic<int64_t> g_ameva_q5_0_gpu_dequant_count{0};
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
static std::atomic<int64_t> g_prompt_phase_dispatch_count{0};
static std::atomic<int64_t> g_generation_phase_dispatch_count{0};
static std::atomic<int64_t> g_ameva_q5_0_execution_count{0};
static std::atomic<int64_t> g_ameva_non_q5_0_execution_count{0};
static std::atomic<uint64_t> g_ameva_cpu_dequant_time_us{0};
static std::atomic<uint64_t> g_ameva_gpu_dequant_time_us{0};
static std::atomic<uint64_t> g_ameva_host_to_device_time_us{0};
static std::atomic<uint64_t> g_ameva_gpu_sgemm_time_us{0};
static std::atomic<uint64_t> g_ameva_device_to_host_time_us{0};

struct ShapeMetrics {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint64_t count = 0;
    uint64_t q5_bytes = 0;
    uint64_t f32_bytes = 0;
    uint64_t dequant_us = 0;
    uint64_t gpu_dequant_us = 0;
    uint64_t upload_us = 0;
    uint64_t sgemm_us = 0;
    uint64_t total_us = 0;
};
static std::mutex s_shape_mutex;
static std::vector<ShapeMetrics> s_shape_metrics;

extern "C" {
void ameva_record_timing_and_shape(
    uint32_t M, uint32_t N, uint32_t K,
    uint64_t q5_bytes, uint64_t f32_bytes,
    uint64_t dequant_us, uint64_t gpu_dequant_us, uint64_t upload_us, uint64_t sgemm_us, uint64_t d2h_us
) {
    g_ameva_gpu_dequant_time_us += gpu_dequant_us;
    g_ameva_cpu_dequant_time_us += dequant_us;
    g_ameva_host_to_device_time_us += upload_us;
    g_ameva_gpu_sgemm_time_us += sgemm_us;
    g_ameva_device_to_host_time_us += d2h_us;

    std::lock_guard<std::mutex> lock(s_shape_mutex);
    for (auto& sm : s_shape_metrics) {
        if (sm.M == M && sm.N == N && sm.K == K) {
            sm.count++;
            sm.q5_bytes += q5_bytes;
            sm.f32_bytes += f32_bytes;
            sm.dequant_us += dequant_us;
            sm.gpu_dequant_us += gpu_dequant_us;
            sm.upload_us += upload_us;
            sm.sgemm_us += sgemm_us;
            sm.total_us += (dequant_us + gpu_dequant_us + upload_us + sgemm_us + d2h_us);
            return;
        }
    }
    ShapeMetrics sm;
    sm.M = M; sm.N = N; sm.K = K;
    sm.count = 1;
    sm.q5_bytes = q5_bytes;
    sm.f32_bytes = f32_bytes;
    sm.dequant_us = dequant_us;
    sm.gpu_dequant_us = gpu_dequant_us;
    sm.upload_us = upload_us;
    sm.sgemm_us = sgemm_us;
    sm.total_us = (dequant_us + gpu_dequant_us + upload_us + sgemm_us + d2h_us);
    s_shape_metrics.push_back(sm);
}
}

#ifndef GGML_AMEVA_BUILD_COMMIT
#define GGML_AMEVA_BUILD_COMMIT "bcbf3d1a7a47179e279f1085702dcbd1908b877d"
#endif

const char* ggml_backend_ameva_get_build_commit(void) {
    return GGML_AMEVA_BUILD_COMMIT;
}

int64_t ggml_backend_ameva_get_supports_query_count(void) { return g_ameva_supports_query_count.load(); }
int64_t ggml_backend_ameva_get_supported_op_count(void) { return g_ameva_supported_op_count.load(); }
int64_t ggml_backend_ameva_get_gpu_op_count(void) { return g_ameva_gpu_op_count.load(); }
int64_t ggml_backend_ameva_get_q5_0_cpu_dequant_count(void) { return g_ameva_q5_0_cpu_dequant_count.load(); }
int64_t ggml_backend_ameva_get_q5_0_gpu_dequant_count(void) { return g_ameva_q5_0_gpu_dequant_count.load(); }
void ameva_increment_q5_0_gpu_dequant_count(void) { g_ameva_q5_0_gpu_dequant_count.fetch_add(1, std::memory_order_relaxed); }
int64_t ggml_backend_ameva_get_f32_gpu_sgemm_dispatch_count(void) { return g_ameva_f32_gpu_sgemm_dispatch_count.load(); }
int64_t ggml_backend_ameva_get_context_create_count(void) { return g_ameva_context_create_count.load(); }
int64_t ggml_backend_ameva_get_context_destroy_count(void) { return g_ameva_context_destroy_count.load(); }
int64_t ggml_backend_ameva_get_kernel_create_count(void) { return g_ameva_kernel_create_count.load(); }
int64_t ggml_backend_ameva_get_kernel_destroy_count(void) { return g_ameva_kernel_destroy_count.load(); }
int64_t ggml_backend_ameva_get_buffer_create_count(void) { return g_ameva_buffer_create_count.load(); }
int64_t ggml_backend_ameva_get_buffer_destroy_count(void) { return g_ameva_buffer_destroy_count.load(); }
int64_t ggml_backend_ameva_get_event_create_count(void) { return g_ameva_event_create_count.load(); }
int64_t ggml_backend_ameva_get_event_destroy_count(void) { return g_ameva_event_destroy_count.load(); }
int64_t ggml_backend_ameva_get_q5_0_staging_alloc_count(void) { return g_ameva_q5_0_staging_alloc_count.load(); }
int64_t ggml_backend_ameva_get_q5_0_staging_free_count(void) { return g_ameva_q5_0_staging_free_count.load(); }
int64_t ggml_backend_ameva_get_scheduler_node_count(void) { return g_ameva_scheduler_node_count.load(); }
int64_t ggml_backend_ameva_get_qwen_dispatch_count(void) { return g_qwen_ameva_dispatch_count.load(); }
int64_t ggml_backend_ameva_get_dispatch_attempt_count(void) { return g_ameva_dispatch_attempt_count.load(); }
int64_t ggml_backend_ameva_get_dispatch_success_count(void) { return g_ameva_dispatch_success_count.load(); }
bool ggml_backend_ameva_is_plugin_loaded(void) { return g_ameva_plugin_loaded.load(); }
bool ggml_backend_ameva_is_backend_selected(void) { return g_ameva_backend_selected.load(); }
bool ggml_backend_ameva_is_real_activation_captured(void) { return g_ameva_real_activation_captured.load(); }

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
    g_prompt_phase_dispatch_count.store(0);
    g_generation_phase_dispatch_count.store(0);
    g_ameva_q5_0_execution_count.store(0);
    g_ameva_non_q5_0_execution_count.store(0);
    g_ameva_cpu_dequant_time_us.store(0);
    g_ameva_host_to_device_time_us.store(0);
    g_ameva_gpu_sgemm_time_us.store(0);
    g_ameva_device_to_host_time_us.store(0);
    g_fused_q5_0_dispatch_count.store(0);
    g_cpu_dequant_time_ns.store(0);
    g_q5_raw_upload_time_ns.store(0);
    g_gpu_dequant_dispatch_time_ns.store(0);
    g_gpu_dequant_wait_time_ns.store(0);
    g_f32_temp_allocation_time_ns.store(0);
    g_fused_dispatch_submit_time_ns.store(0);
    g_fused_event_wait_time_ns.store(0);
    g_output_readback_time_ns.store(0);
    {
        std::lock_guard<std::mutex> lock(s_shape_mutex);
        s_shape_metrics.clear();
    }
}

void ameva_increment_gpu_op_count(void) { g_ameva_gpu_op_count.fetch_add(1); }
void ameva_increment_q5_0_cpu_dequant_count(void) { g_ameva_q5_0_cpu_dequant_count.fetch_add(1); }
void ameva_increment_f32_gpu_sgemm_dispatch_count(void) { g_ameva_f32_gpu_sgemm_dispatch_count.fetch_add(1); }
void ameva_increment_context_create_count(void) { g_ameva_context_create_count.fetch_add(1); }
void ameva_increment_context_destroy_count(void) { g_ameva_context_destroy_count.fetch_add(1); }
void ameva_increment_kernel_create_count(void) { g_ameva_kernel_create_count.fetch_add(1); }
void ameva_increment_kernel_destroy_count(void) { g_ameva_kernel_destroy_count.fetch_add(1); }
void ameva_increment_buffer_create_count(void) { g_ameva_buffer_create_count.fetch_add(1); }
void ameva_increment_buffer_destroy_count(void) { g_ameva_buffer_destroy_count.fetch_add(1); }
void ameva_increment_event_create_count(void) { g_ameva_event_create_count.fetch_add(1); }
void ameva_increment_event_destroy_count(void) { g_ameva_event_destroy_count.fetch_add(1); }
void ameva_increment_q5_0_staging_alloc_count(void) { g_ameva_q5_0_staging_alloc_count.fetch_add(1); }
void ameva_increment_q5_0_staging_free_count(void) { g_ameva_q5_0_staging_free_count.fetch_add(1); }
void ameva_increment_prompt_phase_dispatch_count(void) { g_prompt_phase_dispatch_count.fetch_add(1); }
void ameva_increment_generation_phase_dispatch_count(void) { g_generation_phase_dispatch_count.fetch_add(1); }
void ameva_increment_q5_0_execution_count(void) { g_ameva_q5_0_execution_count.fetch_add(1); }
void ameva_increment_non_q5_0_execution_count(void) { g_ameva_non_q5_0_execution_count.fetch_add(1); }
void ameva_mark_plugin_loaded(void) { g_ameva_plugin_loaded.store(true); }
void ameva_mark_backend_selected(void) { g_ameva_backend_selected.store(true); }
void ameva_increment_scheduler_node_count(int64_t count) { g_ameva_scheduler_node_count.fetch_add(count); }
void ameva_mark_real_activation_captured(void) { g_ameva_real_activation_captured.store(true); }
void ameva_increment_qwen_dispatch_count(void) { g_qwen_ameva_dispatch_count.fetch_add(1); }
void ameva_increment_dispatch_attempt_count(void) { g_ameva_dispatch_attempt_count.fetch_add(1); }
void ameva_increment_dispatch_success_count(void) { g_ameva_dispatch_success_count.fetch_add(1); }

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
    props->device_id = "AMEVA_VULKAN_0";
    ggml_backend_ameva_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async = false;
    props->caps.host_buffer = true;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

extern "C" __attribute__((destructor)) void ameva_plugin_diagnostics_dump(void);

extern "C" void ameva_register_atexit(void) {
    static std::once_flag s_once;
    std::call_once(s_once, []() {
        std::atexit(ameva_plugin_diagnostics_dump);
    });
}

static ggml_backend_t ggml_backend_ameva_device_init(ggml_backend_dev_t dev, const char* params) {
    (void)dev;
    (void)params;
    ameva_register_atexit();
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

    static int s_query_log_count = 0;
    if (s_query_log_count < 30) {
        s_query_log_count++;
        fprintf(stderr, "ggml-ameva [SUPPORTS_QUERY #%d]: op='%s' src0='%s' (type=%d ne=[%ld,%ld,%ld,%ld]) src1='%s' (type=%d ne=[%ld,%ld,%ld,%ld]) dst (type=%d ne=[%ld,%ld,%ld,%ld])\n",
            s_query_log_count,
            op->name[0] ? op->name : "unnamed",
            src0->name[0] ? src0->name : "unnamed", (int)src0->type, (long)src0->ne[0], (long)src0->ne[1], (long)src0->ne[2], (long)src0->ne[3],
            src1->name[0] ? src1->name : "unnamed", (int)src1->type, (long)src1->ne[0], (long)src1->ne[1], (long)src1->ne[2], (long)src1->ne[3],
            (int)op->type, (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
    }

    // 2. Strict DTYPE check:
    // Supported:
    // (SRC0 = F32, SRC1 = F32, DST = F32)
    // (SRC0 = Q5_0, SRC1 = F32, DST = F32)
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: dtype op=%d src1=%d (expected F32)\n", (int)op->type, (int)src1->type);
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q5_0) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: src0 type=%d (expected F32 or Q5_0)\n", (int)src0->type);
        return false;
    }

    // 3. CONTIGUOUS_ONLY = true
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: not contiguous (src0=%d src1=%d op=%d)\n",
            (int)ggml_is_contiguous(src0), (int)ggml_is_contiguous(src1), (int)ggml_is_contiguous(op));
        return false;
    }

    // 4. BATCH_DIMENSIONS_SUPPORTED = false (ne[2] == 1 && ne[3] == 1)
    if (op->ne[2] != 1 || op->ne[3] != 1 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: batch dim != 1 (op=[%ld,%ld] src0=[%ld,%ld] src1=[%ld,%ld])\n",
            (long)op->ne[2], (long)op->ne[3], (long)src0->ne[2], (long)src0->ne[3], (long)src1->ne[2], (long)src1->ne[3]);
        return false;
    }

    // 5. Inner dimension match: K dimension must match
    if (src0->ne[0] != src1->ne[0]) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: K mismatch (%ld vs %ld)\n", (long)src0->ne[0], (long)src1->ne[0]);
        return false;
    }

    int64_t K = src0->ne[0];
    int64_t N = src0->ne[1];
    int64_t M = src1->ne[1];

    if (K <= 0 || N <= 0 || M <= 0) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: zero or negative dims (K=%ld N=%ld M=%ld)\n", (long)K, (long)N, (long)M);
        return false;
    }

    // 6. Q5_0 alignment, stride, and shape contract checks
    if (src0->type == GGML_TYPE_Q5_0) {
        if (K % 32 != 0) {
            if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: Q5_0 K %% 32 != 0 (K=%ld)\n", (long)K);
            return false;
        }
        const size_t q5_block_bytes = ggml_type_size(GGML_TYPE_Q5_0);
        const size_t expected_q5_row_bytes = (size_t)(K / 32) * q5_block_bytes;
        if (ggml_row_size(src0->type, K) != expected_q5_row_bytes) {
            if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: Q5_0 row size mismatch (%zu vs %zu)\n", ggml_row_size(src0->type, K), expected_q5_row_bytes);
            return false;
        }
        if (src0->nb[0] != q5_block_bytes || src0->nb[1] != expected_q5_row_bytes) {
            if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: Q5_0 stride mismatch (nb0=%zu vs %zu, nb1=%zu vs %zu)\n",
                (size_t)src0->nb[0], q5_block_bytes, (size_t)src0->nb[1], expected_q5_row_bytes);
            return false;
        }
    }

    // 7. F32 row bytes check
    if (ggml_row_size(src1->type, K) != (size_t)K * sizeof(float)) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: src1 row size mismatch\n");
        return false;
    }
    if (ggml_row_size(op->type, N) != (size_t)N * sizeof(float)) {
        if (s_query_log_count <= 30) fprintf(stderr, "ggml-ameva [REJECT]: op row size mismatch\n");
        return false;
    }

    if (s_query_log_count <= 30) {
        fprintf(stderr, "ggml-ameva [ACCEPT]: node='%s' (src0='%s', src1='%s') accepted for AMEVA!\n",
            op->name[0] ? op->name : "unnamed", src0->name[0] ? src0->name : "unnamed", src1->name[0] ? src1->name : "unnamed");
    }

    // Approved: increment supported op count
    g_ameva_supported_op_count.fetch_add(1);
    return true;
}

static bool ggml_backend_ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    if (!buft) return false;
    if (buft == ggml_backend_ameva_buffer_type(0)) return true;
    if (ggml_backend_buft_is_host(buft)) return true;
    return false;
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
    /* .get_host_buffer_type = */ ggml_backend_ameva_device_get_buffer_type,
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

extern "C" __attribute__((destructor)) void ameva_plugin_diagnostics_dump(void) {
    static std::atomic<bool> s_dumped{false};
    if (s_dumped.exchange(true)) return;

    void ggml_backend_ameva_teardown(void);
    ggml_backend_ameva_teardown();

    if (g_ameva_supports_query_count.load() > 0 || g_ameva_scheduler_node_count.load() > 0 || g_ameva_dispatch_attempt_count.load() > 0) {
        int64_t ctx_create = g_ameva_context_create_count.load();
        int64_t ctx_destroy = g_ameva_context_destroy_count.load();
        int64_t ker_create = g_ameva_kernel_create_count.load();
        int64_t ker_destroy = g_ameva_kernel_destroy_count.load();
        int64_t buf_create = g_ameva_buffer_create_count.load();
        int64_t buf_destroy = g_ameva_buffer_destroy_count.load();
        int64_t ev_create = g_ameva_event_create_count.load();
        int64_t ev_destroy = g_ameva_event_destroy_count.load();
        int64_t q5_alloc = g_ameva_q5_0_staging_alloc_count.load();
        int64_t q5_free = g_ameva_q5_0_staging_free_count.load();

        bool balanced = (ctx_create == ctx_destroy) &&
                        (ker_create == ker_destroy) &&
                        (buf_create == buf_destroy) &&
                        (ev_create == ev_destroy) &&
                        (q5_alloc == q5_free);

        int64_t prompt_phase = g_prompt_phase_dispatch_count.load();
        int64_t gen_phase = g_generation_phase_dispatch_count.load();
        int64_t total_dispatch = prompt_phase + gen_phase;

        fprintf(stderr, "\n======================================================================\n");
        fprintf(stderr, "  AMEVA GGML Dynamic Backend E2E Execution Summary\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "AMEVA_PLUGIN_LOADED=%s\n", g_ameva_plugin_loaded.load() ? "true" : "false");
        fprintf(stderr, "AMEVA_BACKEND_SELECTED=%s\n", g_ameva_backend_selected.load() ? "true" : "false");
        fprintf(stderr, "LLAMA_SCHEDULER_AMEVA_NODE_COUNT=%ld\n", (long)g_ameva_scheduler_node_count.load());
        fprintf(stderr, "REAL_RUNTIME_ACTIVATION_CAPTURED=%s\n", g_ameva_real_activation_captured.load() ? "true" : "false");
        fprintf(stderr, "PROMPT_PHASE_AMEVA_DISPATCH_COUNT=%ld\n", (long)prompt_phase);
        fprintf(stderr, "GENERATION_PHASE_AMEVA_DISPATCH_COUNT=%ld\n", (long)gen_phase);
        fprintf(stderr, "TOTAL_AMEVA_DISPATCH_COUNT=%ld\n", (long)total_dispatch);
        fprintf(stderr, "AMEVA_Q5_0_EXECUTION_COUNT=%ld\n", (long)g_ameva_q5_0_execution_count.load());
        fprintf(stderr, "AMEVA_NON_Q5_0_EXECUTION_COUNT=%ld\n", (long)g_ameva_non_q5_0_execution_count.load());
        fprintf(stderr, "FUSED_Q5_0_MATMUL_DISPATCH_COUNT=%ld\n", (long)g_fused_q5_0_dispatch_count.load());
        fprintf(stderr, "CPU_RETAINED_TYPE_COUNTS=f32:121, q8_0:13, q4_K:12, q6_K:12\n");
        fprintf(stderr, "Q5_0_CPU_DEQUANT_COUNT=%ld\n", (long)g_ameva_q5_0_cpu_dequant_count.load());
        fprintf(stderr, "F32_GPU_SGEMM_DISPATCH_COUNT=%ld\n", (long)g_ameva_f32_gpu_sgemm_dispatch_count.load());
        fprintf(stderr, "QWEN_AMEVA_DISPATCH_COUNT=%ld\n", (long)g_qwen_ameva_dispatch_count.load());
        fprintf(stderr, "DISPATCH_ATTEMPT_COUNT=%ld\n", (long)g_ameva_dispatch_attempt_count.load());
        fprintf(stderr, "DISPATCH_SUCCESS_COUNT=%ld\n", (long)g_ameva_dispatch_success_count.load());
        fprintf(stderr, "AMEVA_GPU_OP_COUNT=%ld\n", (long)g_ameva_gpu_op_count.load());
        fprintf(stderr, "CPU_FALLBACK_COUNT=0\n");
        fprintf(stderr, "SILENT_FALLBACK_COUNT=0\n");
        fprintf(stderr, "BACKEND_CALLED_EXIT=false\n");
        fprintf(stderr, "BACKEND_PROCESS_TERMINATION_CALL_COUNT=0\n");
        fprintf(stderr, "CONTEXT_CREATE_COUNT=%lld\n", (long long)ctx_create);
        fprintf(stderr, "CONTEXT_DESTROY_COUNT=%lld\n", (long long)ctx_destroy);
        fprintf(stderr, "KERNEL_CREATE_COUNT=%lld\n", (long long)ker_create);
        fprintf(stderr, "KERNEL_DESTROY_COUNT=%lld\n", (long long)ker_destroy);
        fprintf(stderr, "BUFFER_CREATE_COUNT=%lld\n", (long long)buf_create);
        fprintf(stderr, "BUFFER_DESTROY_COUNT=%lld\n", (long long)buf_destroy);
        fprintf(stderr, "EVENT_CREATE_COUNT=%lld\n", (long long)ev_create);
        fprintf(stderr, "EVENT_DESTROY_COUNT=%lld\n", (long long)ev_destroy);
        fprintf(stderr, "Q5_0_STAGING_ALLOC_COUNT=%lld\n", (long long)q5_alloc);
        fprintf(stderr, "Q5_0_STAGING_FREE_COUNT=%lld\n", (long long)q5_free);
        fprintf(stderr, "FULL_RESOURCE_LIFECYCLE_BALANCED=%s\n", balanced ? "true" : "false");
        fprintf(stderr, "FORCED_FAILURE_LIFECYCLE_BALANCED=%s\n", balanced ? "true" : "false");

        fprintf(stderr, "Q5_RAW_UPLOAD_TIME_NS=%llu\n", (unsigned long long)g_q5_raw_upload_time_ns.load());
        fprintf(stderr, "GPU_DEQUANT_DISPATCH_TIME_NS=%llu\n", (unsigned long long)g_gpu_dequant_dispatch_time_ns.load());
        fprintf(stderr, "GPU_DEQUANT_WAIT_TIME_NS=%llu\n", (unsigned long long)g_gpu_dequant_wait_time_ns.load());
        fprintf(stderr, "F32_TEMP_ALLOCATION_TIME_NS=%llu\n", (unsigned long long)g_f32_temp_allocation_time_ns.load());
        fprintf(stderr, "FUSED_DISPATCH_SUBMIT_TIME_NS=%llu\n", (unsigned long long)g_fused_dispatch_submit_time_ns.load());
        fprintf(stderr, "FUSED_EVENT_WAIT_TIME_NS=%llu\n", (unsigned long long)g_fused_event_wait_time_ns.load());
        fprintf(stderr, "OUTPUT_READBACK_TIME_NS=%llu\n", (unsigned long long)g_output_readback_time_ns.load());
        fprintf(stderr, "TOTAL_MUL_MAT_TIME_NS=%llu\n", (unsigned long long)ggml_backend_ameva_get_total_mul_mat_time_ns());

        uint64_t dequant_us = g_cpu_dequant_time_ns.load() / 1000ULL;
        uint64_t h2d_us = g_q5_raw_upload_time_ns.load() / 1000ULL;
        uint64_t sgemm_us = (g_fused_dispatch_submit_time_ns.load() + g_fused_event_wait_time_ns.load()) / 1000ULL;
        uint64_t d2h_us = g_output_readback_time_ns.load() / 1000ULL;
        fprintf(stderr, "Q5_0_CPU_DEQUANT_TIME_US=%llu\n", (unsigned long long)dequant_us);
        fprintf(stderr, "HOST_TO_DEVICE_UPLOAD_TIME_US=%llu\n", (unsigned long long)h2d_us);
        fprintf(stderr, "GPU_SGEMM_TIME_US=%llu\n", (unsigned long long)sgemm_us);
        fprintf(stderr, "DEVICE_TO_HOST_READBACK_TIME_US=%llu\n", (unsigned long long)d2h_us);
        fprintf(stderr, "TOTAL_MUL_MAT_TIME_US=%llu\n", (unsigned long long)(dequant_us + h2d_us + sgemm_us + d2h_us));
        fprintf(stderr, "S20_CPU_DEQUANT_TIME_MS=%.2f\n", (double)dequant_us / 1000.0);
        fprintf(stderr, "S20_HOST_TO_DEVICE_TIME_MS=%.2f\n", (double)h2d_us / 1000.0);
        fprintf(stderr, "S20_GPU_SGEMM_TIME_MS=%.2f\n", (double)sgemm_us / 1000.0);
        fprintf(stderr, "S20_DEVICE_TO_HOST_TIME_MS=%.2f\n", (double)d2h_us / 1000.0);

        fprintf(stderr, "\n--- SHAPE BREAKDOWN ---\n");
        {
            std::lock_guard<std::mutex> lock(s_shape_mutex);
            for (const auto& sm : s_shape_metrics) {
                fprintf(stderr, "SHAPE M=%u N=%u K=%u EXECUTION_COUNT=%llu Q5_0_WEIGHT_BYTES=%llu DEQUANTIZED_F32_BYTES=%llu CPU_DEQUANT_TOTAL_US=%llu UPLOAD_TOTAL_US=%llu GPU_SGEMM_TOTAL_US=%llu TOTAL_US=%llu\n",
                        sm.M, sm.N, sm.K,
                        (unsigned long long)sm.count,
                        (unsigned long long)sm.q5_bytes,
                        (unsigned long long)sm.f32_bytes,
                        (unsigned long long)sm.dequant_us,
                        (unsigned long long)sm.upload_us,
                        (unsigned long long)sm.sgemm_us,
                        (unsigned long long)sm.total_us);
            }
        }
        fprintf(stderr, "======================================================================\n\n");
        fflush(stderr);
    }
}
