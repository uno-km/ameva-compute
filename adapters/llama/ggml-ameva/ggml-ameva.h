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

// Timing & Shape Diagnostics
GGML_API uint64_t ggml_backend_ameva_get_cpu_dequant_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_host_to_device_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_gpu_sgemm_time_us(void);
GGML_API uint64_t ggml_backend_ameva_get_device_to_host_time_us(void);
void ameva_record_timing_and_shape(
    uint32_t M, uint32_t N, uint32_t K,
    uint64_t q5_bytes, uint64_t f32_bytes,
    uint64_t dequant_us, uint64_t upload_us, uint64_t sgemm_us, uint64_t d2h_us
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
