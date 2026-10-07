/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file amuda_cache.h
 * @brief AMUDA On-Disk JIT Binary Program Cache Interface
 *
 * Implements SHA-256 program signature hashing, atomic disk serialization,
 * and zero-overhead binary reconstitution (clCreateProgramWithBinary).
 */

#ifndef AMUDA_CACHE_H
#define AMUDA_CACHE_H

#include "amuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute SHA-256 hex string (64 characters + null terminator) for input buffer.
 */
void ac_sha256_hex(const void* data, size_t length, char out_hex[65]);

/**
 * @brief Retrieve or compile a kernel with persistent on-disk binary caching.
 *
 * Checks ~/.cache/amuda/<hash>.bin:
 * - If found: loads directly using ac_kernel_load_binary (< 0.5 ms).
 * - If missing: compiles using ac_kernel_load_source, extracts driver binary,
 *   and atomically caches it to disk for future executions.
 */
AcResult ac_kernel_load_cached(
    AcContext* ctx,
    const char* source_code,
    const char* func_name,
    AcKernel** out_kernel
);

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_CACHE_H */
