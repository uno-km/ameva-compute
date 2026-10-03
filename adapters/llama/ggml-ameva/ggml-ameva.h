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

// Dynamic Loader Entrypoints (dlopen / dlsym)
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);
GGML_BACKEND_API int ggml_backend_score(void);

// Diagnostic Counters (Ground Truth Verification)
// Since the AMEVA graph engine every graph runs as one GPU batch and every weight type is read in its raw format:
//   gpu_op_count             graph nodes completed on the GPU (any op)
//   f32_gpu_sgemm_dispatch   MUL_MAT nodes completed on the GPU, whatever the weight type (name kept for ABI)
//   q5_0_cpu_dequant_count   stays 0 -- nothing is dequantized on the CPU (name kept for ABI)
//   q5_0_staging_alloc/free  stay 0 -- tensors outside AMEVA buffers are never staged by the backend; the
//                            scheduler copies them (names kept for ABI)
//   repacked_tensor_count    tensors stored in the AMEVA repacked weight layout, and repacked_bytes their size
//                            (stored state: not cleared by reset_counters)
//   qmv_mul_mat_count        MUL_MAT nodes completed on the GPU with repacked weights (ameva_qmv kernels)
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
GGML_API int64_t ggml_backend_ameva_get_repacked_tensor_count(void);
GGML_API int64_t ggml_backend_ameva_get_repacked_bytes(void);
GGML_API int64_t ggml_backend_ameva_get_qmv_mul_mat_count(void);
GGML_API const char* ggml_backend_ameva_get_build_commit(void);
GGML_API void ggml_backend_ameva_reset_counters(void);

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
