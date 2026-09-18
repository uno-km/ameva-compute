/**
 * @file ggml-backend-ameva.h
 * @brief Formal AMEVA Compute Hardware Backend Adapter for GGML / whisper.cpp
 */

#ifndef GGML_BACKEND_AMEVA_H
#define GGML_BACKEND_AMEVA_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ac_types.h"

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
    uint64_t unsupported_node_encounters;
    uint64_t cpu_retry_after_accept;
    uint64_t buffer_alloc_count;
    uint64_t buffer_free_count;
    uint64_t sync_count;

    // Granular graph kind dispatch counters
    uint64_t encoder_dispatch_count;
    uint64_t cross_dispatch_count;
    uint64_t unknown_graph_kind_count;

    // Granular graph kind dispatch counters across inference runs (P0-1)
    uint64_t run_1_encoder_dispatch_count;
    uint64_t run_1_cross_dispatch_count;
    uint64_t run_1_decoder_dispatch_count;
    uint64_t run_2_encoder_dispatch_count;
    uint64_t run_2_cross_dispatch_count;
    uint64_t run_2_decoder_dispatch_count;
    uint64_t run_2_context_forward_pass_count;
    uint64_t run_2_context_matmuls_per_pass;
    uint64_t run_2_context_dispatch_count;

    // Graph batching architectural counters
    uint64_t graph_command_buffer_count;
    uint64_t graph_completion_event_count;
    uint64_t node_level_event_wait_count;

    // Symmetric Lifecycle counters
    uint64_t context_create_count;
    uint64_t context_destroy_count;
    uint64_t kernel_create_count;
    uint64_t kernel_destroy_count;
    uint64_t buffer_create_count;
    uint64_t buffer_destroy_count;
    uint64_t event_create_count;
    uint64_t event_destroy_count;

    // Performance & Weight GPU Cache Telemetry
    uint64_t weight_cache_hit_count;
    uint64_t weight_cache_miss_count;
    uint64_t weight_upload_count_first_run;
    uint64_t weight_upload_count_second_run;
    uint64_t internal_activation_upload_count;
    uint64_t internal_output_readback_count;
    uint64_t event_create_destroy_count;

    // Cache Identity, Collision Defense & Symmetric Lifecycle Telemetry (P0-1, P0-4)
    uint64_t cache_key_alias_count;
    uint64_t weight_cache_entry_create_count;
    uint64_t weight_cache_entry_destroy_count;

    // Explicit Multi-Run Telemetry (P0-5)
    uint64_t inference_run_count;
    uint64_t inference_run_1_dispatch_count;
    uint64_t inference_run_2_dispatch_count;
    uint64_t inference_run_1_weight_upload_count;
    uint64_t inference_run_2_weight_upload_count;

    double weight_upload_time_ms;
    double activation_upload_time_ms;
    double dispatch_api_time_ms;
    double event_wait_time_ms;
    double output_readback_time_ms;

    uint64_t weight_upload_bytes;
    uint64_t activation_upload_bytes;
    uint64_t output_readback_bytes;
} AmevaBackendCounters;

GGML_API ggml_backend_reg_t ggml_backend_ameva_reg(void);
GGML_API ggml_backend_dev_t ggml_backend_ameva_device(void);
GGML_API ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(void);
GGML_API ggml_backend_t ggml_backend_ameva_init(const char * params);
GGML_API void ameva_backend_get_counters(AmevaBackendCounters * out_counters);
GGML_API void ameva_backend_reset_counters(void);
GGML_API void ameva_backend_teardown_hardware(void);
GGML_API void ameva_backend_register(void);
GGML_API void ameva_backend_set_graph_kind(const char * kind);

// P0-2: Production Cache Diagnostics API
typedef struct {
    uint64_t lookup_count;
    uint64_t hash_match_count;
    uint64_t payload_recheck_count;
    uint64_t payload_mismatch_count;
    uint64_t alias_count;
    uint64_t entry_count;
} AmevaCacheDiagnostics;

GGML_API void ameva_backend_get_cache_diagnostics(AmevaCacheDiagnostics * out_diag);
GGML_API void ameva_backend_reset_cache_diagnostics(void);
GGML_API void ameva_backend_clear_test_cache(void);
GGML_API bool ameva_backend_run_cached_gpu_matmul(
    const void * weight_data,
    size_t weight_bytes,
    const float * act_data,
    float * out_data,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    enum ggml_type type
);

// P0-5: Explicit Multi-Run Boundary Management API
GGML_API void ameva_backend_begin_inference(ggml_backend_t backend, uint32_t inference_id);
GGML_API void ameva_backend_end_inference(ggml_backend_t backend, uint32_t inference_id);

// P0-4: Hardened Backend Error State APIs
GGML_API AmevaBackendErrorState ameva_backend_get_last_buffer_error(void);
GGML_API void ameva_backend_clear_buffer_error(void);
GGML_API int32_t ameva_backend_consume_last_buffer_error(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_BACKEND_AMEVA_H */
