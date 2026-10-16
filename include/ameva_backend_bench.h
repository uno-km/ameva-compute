/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ameva_backend_bench.h
 * @brief Warm-up micro-benchmark used by AmevaLlm's "auto" backend selection: times a fixed
 * GEMV (M=1 MUL_MAT, the shape decode-time inference actually runs) on cpu/vulkan/opencl and
 * reports elapsed ms, so the caller can pick whichever backend is actually fastest on this
 * device instead of a fixed cpu-then-gpu rule.
 */

#ifndef AMEVA_BACKEND_BENCH_H
#define AMEVA_BACKEND_BENCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AMEVA_BENCH_OK = 0,
    AMEVA_BENCH_ERR_BACKEND_UNAVAILABLE,
    AMEVA_BENCH_ERR_FORK_FAILED,
    AMEVA_BENCH_ERR_INVALID_ARGUMENT
} AmevaBenchStatus;

/**
 * @param backend_name "cpu", "vulkan", or "opencl".
 * @param ggml_ameva_plugin_path Path to libggml-ameva.so; required (and only used) for
 *        "vulkan"/"opencl", ignored for "cpu".
 * @param out_ms Average elapsed ms per matmul over a fixed rep count (1 untimed warmup rep first,
 *        so the GPU's one-time pipeline-compile cost -- ~110ms on Adreno 830 per the pipeline
 *        cache's own docs -- doesn't get counted as the "speed" of that backend).
 *
 * "vulkan" and "opencl" run in a forked child process: ggml-ameva.cpp memoizes its AMEVA_BACKEND
 * choice in a function-local static on first use and never re-reads it, so there is no way to
 * benchmark both inside one process -- each GPU candidate gets a fresh child with its own
 * AMEVA_BACKEND set before that child ever touches the plugin. The child only calls plain C
 * (setenv/dlopen/ggml), never anything JVM/ART-aware, before _exit().
 *
 * @return AMEVA_BENCH_OK with *out_ms set, or AMEVA_BENCH_ERR_BACKEND_UNAVAILABLE if that backend
 *         could not even run this benchmark (not "slow" -- genuinely unusable: no device, plugin
 *         load failed, or the compute call itself failed). Never silently reports a backend as
 *         viable when it is not.
 */
AmevaBenchStatus ameva_backend_bench(
    const char* backend_name,
    const char* ggml_ameva_plugin_path,
    double* out_ms,
    char* err_buf,
    size_t err_buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* AMEVA_BACKEND_BENCH_H */
