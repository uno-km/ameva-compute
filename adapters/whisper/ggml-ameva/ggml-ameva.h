/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ggml-ameva.h
 * @brief AMEVA Compute Hardware Vulkan Backend Adapter for whisper.cpp
 *
 * Provides single-operator GPU offload interface for Whisper encoder MatMul
 * operations targeting ARM Mali / Qualcomm Adreno mobile GPUs.
 */

#ifndef GGML_AMEVA_H
#define GGML_AMEVA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

typedef struct ameva_stt_stats {
    uint64_t intercept_attempt_count;
    uint64_t intercept_success_count;
    uint64_t intercept_failure_count;
    uint64_t cpu_fallback_count;
    uint64_t silent_fallback_count;
    uint64_t graph_mutation_leak_count;
    uint64_t context_create_count;
    uint64_t kernel_create_count;
} AmevaSttStats;

/**
 * @brief Initialize the AMEVA Vulkan STT runtime
 * @param spv_path Absolute path to matmul_f16_f32.spv binary
 * @return true on success, false on failure (no fallback)
 */
bool ameva_stt_vulkan_init(const char* spv_path);

/**
 * @brief Free AMEVA Vulkan resources
 */
void ameva_stt_vulkan_free(void);

/**
 * @brief Check if the tensor corresponds to the target encoder MatMul operator (node_7)
 */
bool ameva_stt_vulkan_is_target_node(const struct ggml_tensor* t);

/**
 * @brief Unified evaluation hook with RAII-guaranteed op restoration and failure injection
 * @param t Current tensor in scheduler
 * @param ask true when scheduler asks if node is needed, false when executing
 * @return true if node handled / needed, false otherwise
 */
bool ameva_stt_vulkan_eval_hook(struct ggml_tensor* t, bool ask);

/**
 * @brief Execute the MatMul operator on AMEVA Vulkan GPU
 * @param t Target tensor to compute
 * @return true on success, aborts/fails on failure (fail-fast)
 */
bool ameva_stt_vulkan_compute_node(struct ggml_tensor* t);

/**
 * @brief Retrieve snapshot of AMEVA execution statistics
 */
void ameva_stt_vulkan_get_stats(AmevaSttStats* out_stats);

/**
 * @brief Get total number of successful AMEVA GPU dispatches
 */
uint64_t ameva_stt_vulkan_get_gpu_op_count(void);

/**
 * @brief Get execution latency of last GPU dispatch in microseconds
 */
int64_t ameva_stt_vulkan_get_last_dispatch_us(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_AMEVA_H */
