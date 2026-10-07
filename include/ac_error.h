/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_error.h
 * @brief Formal Generation-Based Error Tracking Definitions and ABI
 * Owned by AMEVA Compute Runtime Core.
 */

#ifndef AC_ERROR_H
#define AC_ERROR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ac_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Formal AMEVA Hardware Backend Operation Kinds for Error Tracking
 */
typedef enum AmevaBackendOpKind {
    AMEVA_BACKEND_OP_NONE = 0,
    AMEVA_BACKEND_OP_BUFFER_ALLOC,
    AMEVA_BACKEND_OP_BUFFER_FREE,
    AMEVA_BACKEND_OP_BUFFER_MAP,
    AMEVA_BACKEND_OP_BUFFER_UNMAP,
    AMEVA_BACKEND_OP_COPY_HTOD,
    AMEVA_BACKEND_OP_COPY_DTOH,
    AMEVA_BACKEND_OP_DISPATCH,
    AMEVA_BACKEND_OP_SYNC
} AmevaBackendOpKind;

/**
 * @brief Sticky Error State Structure for AMEVA Runtime and Adapters
 * Implements generation-based first-error-wins semantics.
 */
typedef struct AmevaBackendErrorState {
    uint64_t generation;
    int32_t  first_error;
    AmevaBackendOpKind operation;
    size_t   offset;
    size_t   requested_bytes;
    size_t   capacity_bytes;
} AmevaBackendErrorState;

#define AC_GLOBAL_CONTEXT_ID 0ULL

/* Default / Global Context Error APIs (Legacy & Single-Context ABI Compatibility) */
void ac_runtime_set_error(
    int32_t error_code,
    AmevaBackendOpKind op,
    size_t offset,
    size_t requested_bytes,
    size_t capacity_bytes
);

AmevaBackendErrorState ac_runtime_get_error_state(void);
int32_t ac_runtime_consume_error(void);
uint64_t ac_runtime_advance_generation(void);
uint64_t ac_runtime_get_generation(void);
void ac_runtime_reset_error_state(void);

/* Multi-Context Error Tracking APIs (P0 Concurrency & Context Isolation) */
void ac_runtime_set_error_ctx(
    uint64_t context_id,
    int32_t error_code,
    AmevaBackendOpKind op,
    size_t offset,
    size_t requested_bytes,
    size_t capacity_bytes
);

AmevaBackendErrorState ac_runtime_get_error_state_ctx(uint64_t context_id);
int32_t ac_runtime_consume_error_ctx(uint64_t context_id);
uint64_t ac_runtime_advance_generation_ctx(uint64_t context_id);
uint64_t ac_runtime_get_generation_ctx(uint64_t context_id);
void ac_runtime_reset_error_state_ctx(uint64_t context_id);

#ifdef __cplusplus
}
#endif

#endif /* AC_ERROR_H */
