/**
 * @file ggml-backend-ameva.h
 * @brief Formal AMEVA Compute Hardware Backend Adapter for GGML / whisper.cpp
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
    uint64_t supports_op_query_count;
    uint64_t supported_pattern_match_count;
    uint64_t graph_compute_call_count;
    uint64_t ameva_dispatch_attempt_count;
    uint64_t ameva_dispatch_success_count;
    uint64_t ameva_gpu_op_count; // Invariant: ameva_gpu_op_count == ameva_dispatch_success_count
    uint64_t ameva_dispatch_failure_count;
    uint64_t buffer_alloc_count;
    uint64_t buffer_free_count;
    uint64_t sync_count;

    // Symmetric Lifecycle counters
    uint64_t context_create_count;
    uint64_t context_destroy_count;
    uint64_t kernel_create_count;
    uint64_t kernel_destroy_count;
    uint64_t buffer_create_count;
    uint64_t buffer_destroy_count;
    uint64_t event_create_count;
    uint64_t event_destroy_count;
} AmevaBackendCounters;

GGML_API ggml_backend_reg_t ggml_backend_ameva_reg(void);
GGML_API ggml_backend_dev_t ggml_backend_ameva_device(void);
GGML_API ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(void);
GGML_API ggml_backend_t ggml_backend_ameva_init(const char * params);
GGML_API void ameva_backend_get_counters(AmevaBackendCounters * out_counters);
GGML_API void ameva_backend_reset_counters(void);
GGML_API void ameva_backend_teardown_hardware(void);
GGML_API void ameva_backend_register(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_BACKEND_AMEVA_H */
