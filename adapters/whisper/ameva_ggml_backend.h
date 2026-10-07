/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ameva_ggml_backend.h
 * @brief AMEVA Compute Hardware Backend Adapter for GGML / whisper.cpp
 *
 * Implements ggml_backend interface to offload Whisper tensor operations
 * (specifically 2D SGEMM encoder projections) to AMEVA Vulkan/OpenCL runtime.
 */

#ifndef AMEVA_GGML_BACKEND_H
#define AMEVA_GGML_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of ggml types to prevent direct header entanglement
struct ggml_tensor;
struct ggml_cgraph;

typedef struct ameva_backend_counters {
    uint64_t gpu_op_count;
    uint64_t cpu_fallback_op_count;
    uint64_t h2d_copy_count;
    uint64_t d2h_copy_count;
    uint64_t sync_count;
    uint64_t dispatch_failure_count;
} AmevaBackendCounters;

typedef struct ameva_ggml_backend_ctx AmevaGgmlBackendCtx;

/**
 * @brief Create an AMEVA GGML backend instance
 */
AmevaGgmlBackendCtx* ameva_ggml_backend_init(void);

/**
 * @brief Destroy an AMEVA GGML backend instance
 */
void ameva_ggml_backend_free(AmevaGgmlBackendCtx* ctx);

/**
 * @brief Query if a specific GGML tensor operator is supported by AMEVA GPU backend
 */
bool ameva_ggml_supports_op(AmevaGgmlBackendCtx* ctx, const struct ggml_tensor* op);

/**
 * @brief Execute a compute graph on AMEVA GPU with explicit CPU fallback logging
 */
int ameva_ggml_graph_compute(AmevaGgmlBackendCtx* ctx, struct ggml_cgraph* cgraph);

/**
 * @brief Synchronize pending AMEVA GPU dispatches
 */
void ameva_ggml_synchronize(AmevaGgmlBackendCtx* ctx);

/**
 * @brief Retrieve snapshot of backend execution counters
 */
void ameva_ggml_get_counters(AmevaGgmlBackendCtx* ctx, AmevaBackendCounters* out_counters);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_GGML_BACKEND_H */
