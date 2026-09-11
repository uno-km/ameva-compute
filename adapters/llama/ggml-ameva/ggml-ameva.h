#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_AMEVA_NAME "AMEVA"
#define GGML_AMEVA_DEVICE_NAME "AMEVA0"
#define GGML_AMEVA_DEVICE_DESCRIPTION "AMEVA Vulkan Compute SGEMM Accelerator"

// Backend & Device Lifecycle
GGML_API ggml_backend_t ggml_backend_ameva_init(size_t dev_num);
GGML_API bool ggml_backend_is_ameva(ggml_backend_t backend);
GGML_API ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(size_t dev_num);
GGML_API ggml_backend_reg_t ggml_backend_ameva_reg(void);
GGML_API void ggml_backend_ameva_register(void);
GGML_API void ggml_backend_ameva_teardown(void);
GGML_API void ameva_buffer_teardown(void);

// Dynamic Loader Entrypoints (dlopen / dlsym)
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);
GGML_BACKEND_API int ggml_backend_score(void);

// Diagnostic Counters (Ground Truth Verification)
GGML_API int64_t ggml_backend_ameva_get_supports_query_count(void);
GGML_API int64_t ggml_backend_ameva_get_supported_op_count(void);
GGML_API int64_t ggml_backend_ameva_get_gpu_op_count(void);
GGML_API int64_t ggml_backend_ameva_get_q5_0_cpu_dequant_count(void);
GGML_API int64_t ggml_backend_ameva_get_q5_0_gpu_dequant_count(void);
GGML_API int64_t ggml_backend_ameva_get_f32_gpu_sgemm_dispatch_count(void);
GGML_API int64_t ggml_backend_ameva_get_context_create_count(void);
GGML_API int64_t ggml_backend_ameva_get_context_destroy_count(void);
GGML_API int64_t ggml_backend_ameva_get_kernel_create_count(void);
GGML_API int64_t ggml_backend_ameva_get_kernel_destroy_count(void);
GGML_API int64_t ggml_backend_ameva_get_buffer_create_count(void);
GGML_API int64_t ggml_backend_ameva_get_buffer_destroy_count(void);
GGML_API int64_t ggml_backend_ameva_get_event_create_count(void);
GGML_API int64_t ggml_backend_ameva_get_event_destroy_count(void);
GGML_API int64_t ggml_backend_ameva_get_q5_0_staging_alloc_count(void);
GGML_API int64_t ggml_backend_ameva_get_q5_0_staging_free_count(void);
GGML_API int64_t ggml_backend_ameva_get_scheduler_node_count(void);
GGML_API int64_t ggml_backend_ameva_get_qwen_dispatch_count(void);
GGML_API int64_t ggml_backend_ameva_get_dispatch_attempt_count(void);
GGML_API int64_t ggml_backend_ameva_get_dispatch_success_count(void);
GGML_API bool ggml_backend_ameva_is_plugin_loaded(void);
GGML_API bool ggml_backend_ameva_is_backend_selected(void);
GGML_API bool ggml_backend_ameva_is_real_activation_captured(void);
GGML_API const char* ggml_backend_ameva_get_build_commit(void);
GGML_API void ggml_backend_ameva_reset_counters(void);
GGML_API void ameva_plugin_diagnostics_dump(void);

GGML_API int64_t ggml_backend_ameva_get_fused_q5_0_dispatch_count(void);
void ameva_increment_fused_q5_0_dispatch_count(void);

// Detailed Timing Decomposition (Nanoseconds with std::chrono::steady_clock)
GGML_API uint64_t ggml_backend_ameva_get_cpu_dequant_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_q5_raw_upload_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_gpu_dequant_dispatch_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_gpu_dequant_wait_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_f32_temp_allocation_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_fused_dispatch_submit_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_fused_event_wait_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_output_readback_time_ns(void);
GGML_API uint64_t ggml_backend_ameva_get_total_mul_mat_time_ns(void);

// Microsecond compatibility getters
GGML_API uint64_t ggml_backend_ameva_get_cpu_dequant_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_q5_raw_upload_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_gpu_dequant_dispatch_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_gpu_dequant_wait_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_f32_temp_allocation_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_sgemm_dispatch_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_sgemm_wait_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_output_readback_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_total_mul_mat_time_us(void);

void ameva_record_detailed_timing_ns(
    uint64_t q5_raw_upload_ns,
    uint64_t gpu_dequant_disp_ns,
    uint64_t gpu_dequant_wait_ns,
    uint64_t f32_temp_alloc_ns,
    uint64_t fused_dispatch_submit_ns,
    uint64_t fused_event_wait_ns,
    uint64_t output_readback_ns
);

// Failure Injection Hooks (P0 Hardening Verification)
typedef enum {
    AMEVA_FAIL_NONE = 0,
    AMEVA_FAIL_MAP_SRC0 = 1,
    AMEVA_FAIL_MAP_SRC1 = 2,
    AMEVA_FAIL_MAP_DST = 3,
    AMEVA_FAIL_FLUSH_SRC0 = 4,
    AMEVA_FAIL_FLUSH_SRC1 = 5,
    AMEVA_FAIL_DISPATCH = 6,
    AMEVA_FAIL_WAIT_TIMEOUT = 7,
    AMEVA_FAIL_INVALIDATE_DST = 8,
    AMEVA_FAIL_OFFSET_OVERFLOW = 9,
    AMEVA_FAIL_TENSOR_RANGE_OOB = 10,
} AmevaFailureInjectionPoint;

GGML_API void ggml_backend_ameva_set_failure_injection(AmevaFailureInjectionPoint point);
GGML_API AmevaFailureInjectionPoint ggml_backend_ameva_get_failure_injection(void);

#ifdef __cplusplus
}
#endif
