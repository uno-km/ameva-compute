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

// Runtime Diagnostic Counters (Ground Truth Verification)
GGML_API int64_t ggml_backend_ameva_get_supported_op_count(void);
GGML_API int64_t ggml_backend_ameva_get_gpu_op_count(void);
GGML_API void ggml_backend_ameva_reset_counters(void);

#ifdef __cplusplus
}
#endif
