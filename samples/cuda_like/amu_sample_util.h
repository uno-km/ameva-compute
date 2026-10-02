/* SPDX-License-Identifier: Apache-2.0 */
/* Helpers shared by the AMUDA CUDA-like samples. Every failure exits non-zero with the reason. */
#ifndef AMU_SAMPLE_UTIL_H
#define AMU_SAMPLE_UTIL_H

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "amuda_rt.h"

/* set by amu_sample_begin() as the first statement of main(), so even an init failure names its sample */
static const char* g_sample_name = "?";

static inline const char* amu_backend_env(void) {
    const char* b = getenv("AMUDA_BACKEND");
    return b && *b ? b : "vulkan";
}

#define AMU_CHECK(call)                                                                                \
    do {                                                                                               \
        amuStatus amu_s_ = (call);                                                                     \
        if (amu_s_ != AMU_SUCCESS) {                                                                   \
            fprintf(stderr, "[%s:%d] %s -> %s\n  detail: %s\n", __FILE__, __LINE__, #call,               \
                    amuGetErrorName(amu_s_), amuGetLastErrorDetail());                                 \
            printf("AMU_RESULT sample=%s backend=%s status=ERROR error=%s\n", g_sample_name,               \
                   amu_backend_env(), amuGetErrorName(amu_s_));                                        \
            exit(2);                                                                                   \
        }                                                                                              \
    } while (0)

/* the verdict line is held back until teardown succeeded (amu_sample_end): a crash or error while releasing
 * GPU resources must never leave a PASS line behind */
static char g_result_line[768];

static inline void amu_sample_begin(const char* sample) {
    g_sample_name = sample;
    setvbuf(stdout, NULL, _IOLBF, 0); /* a crash must not discard already-printed lines in a pipe buffer */
}

_Noreturn static inline void amu_sample_die(const char* error, const char* what) {
    fprintf(stderr, "%s: %s\n", g_sample_name, what);
    printf("AMU_RESULT sample=%s backend=%s status=ERROR error=%s\n", g_sample_name, amu_backend_env(), error);
    exit(2);
}

/* plain host memory: inputs are generated here and the CPU reference reads only host memory.
 * Mapped GPU memory can be uncached on Mali/Adreno, so a CPU loop over it measures the mapping, not the CPU. */
static inline void* amu_host_alloc(size_t bytes) {
    void* p = malloc(bytes ? bytes : 1);
    if (!p) amu_sample_die("HOST_OUT_OF_MEMORY", "host malloc failed");
    return p;
}

static inline double amu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* xorshift32: deterministic inputs on every device */
static inline uint32_t amu_rand_u32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}
static inline float amu_rand_unit(uint32_t* s) { return (float)(amu_rand_u32(s) >> 8) * (1.0f / 16777216.0f); }

static int amu_cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

typedef struct {
    double first_wall_ms; /* includes pipeline/program creation */
    double wall_med_ms;   /* launch+sync wall time, median of steady-state runs */
    double gpu_med_ms;    /* GPU timestamp median, <0 if unavailable */
} amu_timing;

/* Runs launch_fn `iters` times (each followed by amuDeviceSynchronize) after one warm-up launch. */
#define AMU_TIME_LAUNCHES(ctx, iters, timing_out, LAUNCH_EXPR)                                        \
    do {                                                                                               \
        double amu_wall_[64], amu_gpu_[64];                                                            \
        int amu_it_ = (iters) > 64 ? 64 : (iters);                                                     \
        double amu_t0_ = amu_now_ms();                                                                 \
        AMU_CHECK(LAUNCH_EXPR);                                                                        \
        AMU_CHECK(amuDeviceSynchronize(ctx));                                                          \
        (timing_out).first_wall_ms = amu_now_ms() - amu_t0_;                                           \
        int amu_gpu_ok_ = 1;                                                                           \
        for (int amu_i_ = 0; amu_i_ < amu_it_; ++amu_i_) {                                             \
            double amu_t1_ = amu_now_ms();                                                             \
            AMU_CHECK(LAUNCH_EXPR);                                                                    \
            AMU_CHECK(amuDeviceSynchronize(ctx));                                                      \
            amu_wall_[amu_i_] = amu_now_ms() - amu_t1_;                                                \
            amu_gpu_[amu_i_] = amuLastGpuTimeMs(ctx);                                                  \
            if (amu_gpu_[amu_i_] < 0) amu_gpu_ok_ = 0;                                                 \
        }                                                                                              \
        qsort(amu_wall_, (size_t)amu_it_, sizeof(double), amu_cmp_double);                             \
        qsort(amu_gpu_, (size_t)amu_it_, sizeof(double), amu_cmp_double);                              \
        (timing_out).wall_med_ms = amu_wall_[amu_it_ / 2];                                             \
        (timing_out).gpu_med_ms = amu_gpu_ok_ ? amu_gpu_[amu_it_ / 2] : -1.0;                          \
    } while (0)

static inline void amu_print_header(amuContext* ctx, const char* sample) {
    printf("[%s] backend=%s device=\"%s\"\n", sample, amuBackendName(amuGetBackend(ctx)), amuGetDeviceName(ctx));
}

static inline void amu_record_result(amuContext* ctx, const char* sample, int pass, double err, const char* err_kind,
                                     const amu_timing* t, double cpu_ref_ms, const char* extra) {
    char gpu[32];
    if (t->gpu_med_ms >= 0)
        snprintf(gpu, sizeof(gpu), "%.3f", t->gpu_med_ms);
    else
        snprintf(gpu, sizeof(gpu), "n/a");
    snprintf(g_result_line, sizeof(g_result_line),
             "AMU_RESULT sample=%s backend=%s status=%s %s=%.3g first_ms=%.3f wall_ms=%.3f gpu_ms=%s "
             "cpu_ref_ms=%.3f%s%s",
             sample, amuBackendName(amuGetBackend(ctx)), pass ? "PASS" : "FAIL", err_kind, err, t->first_wall_ms,
             t->wall_med_ms, gpu, cpu_ref_ms, extra ? " " : "", extra ? extra : "");
}

/* call after every GPU resource was released with AMU_CHECK: prints the held verdict, returns the exit code */
static inline int amu_sample_end(int pass) {
    if (!g_result_line[0]) amu_sample_die("INTERNAL", "amu_sample_end called before amu_record_result");
    printf("%s\n", g_result_line);
    return pass ? 0 : 1;
}

#endif
