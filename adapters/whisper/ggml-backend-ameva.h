/**
 * @file ggml-backend-ameva.h
 * @brief Formal AMEVA Compute Hardware Backend Adapter for GGML / whisper.cpp
 *
 * Implements the standard ggml_backend, ggml_backend_device, and
 * ggml_backend_buffer_type interfaces according to ggml-backend-impl.h.
 */

#ifndef GGML_BACKEND_AMEVA_H
#define GGML_BACKEND_AMEVA_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t gpu_op_count;
    uint64_t cpu_fallback_count;
    uint64_t sync_count;
    uint64_t dispatch_failure_count;
    uint64_t buffer_alloc_count;
    uint64_t buffer_free_count;
} AmevaBackendCounters;

/**
 * @brief Retrieve or initialize the AMEVA backend registry entry
 */
GGML_API ggml_backend_reg_t ggml_backend_ameva_reg(void);

/**
 * @brief Retrieve the AMEVA backend device handle
 */
GGML_API ggml_backend_dev_t ggml_backend_ameva_device(void);

/**
 * @brief Retrieve the AMEVA unified buffer type
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(void);

/**
 * @brief Initialize an AMEVA backend compute instance
 */
GGML_API ggml_backend_t ggml_backend_ameva_init(const char * params);

/**
 * @brief Retrieve snapshot of backend execution counters
 */
GGML_API void ameva_backend_get_counters(AmevaBackendCounters * out_counters);

/**
 * @brief Automatically register AMEVA backend into GGML global registry
 */
GGML_API void ameva_backend_register(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_BACKEND_AMEVA_H */
