/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ameva_backend_bench.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace {

void setMsg(char* buf, size_t size, const char* fmt, const char* detail) {
    if (buf == nullptr || size == 0) return;
    std::snprintf(buf, size, fmt, detail != nullptr ? detail : "(no detail)");
}

double nowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// M=1 MUL_MAT: the shape llama.cpp's decode step actually runs (one token at a time), not an
// arbitrary "representative" size -- see CLAUDE.md's own Q5_0 GEMV baseline (AMEVA GEMV 12.3 t/s,
// CPU 50.3 t/s at this exact kind of shape).
constexpr int64_t kK = 1024, kM = 1024, kN = 1;
constexpr int kWarmupReps = 1;
constexpr int kTimedReps = 10;

// Returns the average ms per graph_compute() call over kTimedReps, or -1.0 on any failure
// (no silent "0ms" for a backend that couldn't actually run the op).
double runMatmulBench(ggml_backend_t backend) {
    // ggml_new_graph() reserves GGML_DEFAULT_GRAPH_SIZE node/leaf slots regardless of how many
    // nodes this graph actually has (3 tensors) -- confirmed on-device: 16 KiB aborts with
    // "needed 83712, available 16384". 1 MiB is comfortable headroom, not a tuned minimum.
    ggml_init_params iparams = { /*.mem_size=*/ 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context* ctx = ggml_init(iparams);
    if (ctx == nullptr) return -1.0;

    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, kM); // weight-like [K, M]
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, kN); // activation-like [K, 1]
    ggml_tensor* c = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return -1.0;
    }

    std::vector<float> aData((size_t)(kK * kM), 0.01f);
    std::vector<float> bData((size_t)(kK * kN), 0.02f);
    ggml_backend_tensor_set(a, aData.data(), 0, aData.size() * sizeof(float));
    ggml_backend_tensor_set(b, bData.data(), 0, bData.size() * sizeof(float));

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    bool ok = true;
    for (int i = 0; i < kWarmupReps && ok; ++i) {
        ok = (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }

    double elapsedMs = -1.0;
    if (ok) {
        double t0 = nowMs();
        for (int i = 0; i < kTimedReps && ok; ++i) {
            ok = (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        }
        if (ok) elapsedMs = (nowMs() - t0) / (double)kTimedReps;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return elapsedMs;
}

double benchCpu() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) return -1.0;
    double ms = runMatmulBench(backend);
    ggml_backend_free(backend);
    return ms;
}

// Runs entirely in the forked child: only plain C calls (setenv/dlopen via ggml_backend_load/
// ggml compute), then _exit() -- never returns, never touches anything JVM/ART-aware.
[[noreturn]] void benchGpuChild(const char* backendName, const char* pluginPath, int writeFd) {
    double result = -1.0;
    setenv("AMEVA_BACKEND", backendName, 1);

    ggml_backend_reg_t reg = ggml_backend_load(pluginPath);
    if (reg != nullptr && ggml_backend_reg_dev_count(reg) > 0) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            result = runMatmulBench(backend);
            ggml_backend_free(backend);
        }
    }

    ssize_t written = write(writeFd, &result, sizeof(result));
    close(writeFd);
    _exit((written == (ssize_t)sizeof(result) && result >= 0.0) ? 0 : 1);
}

} // namespace

AmevaBenchStatus ameva_backend_bench(
    const char* backend_name,
    const char* ggml_ameva_plugin_path,
    double* out_ms,
    char* err_buf,
    size_t err_buf_size
) {
    if (out_ms != nullptr) *out_ms = -1.0;
    if (backend_name == nullptr || out_ms == nullptr) {
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: invalid argument (%s)", "NULL pointer");
        return AMEVA_BENCH_ERR_INVALID_ARGUMENT;
    }

    if (std::strcmp(backend_name, "cpu") == 0) {
        double ms = benchCpu();
        if (ms < 0.0) {
            setMsg(err_buf, err_buf_size, "ameva_backend_bench: cpu backend %s", "failed to run benchmark matmul");
            return AMEVA_BENCH_ERR_BACKEND_UNAVAILABLE;
        }
        *out_ms = ms;
        return AMEVA_BENCH_OK;
    }

    const bool isVulkan = std::strcmp(backend_name, "vulkan") == 0;
    const bool isOpenCl = std::strcmp(backend_name, "opencl") == 0;
    if (!isVulkan && !isOpenCl) {
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: unknown backend \"%s\"", backend_name);
        return AMEVA_BENCH_ERR_INVALID_ARGUMENT;
    }
    if (ggml_ameva_plugin_path == nullptr || ggml_ameva_plugin_path[0] == '\0') {
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: %s requires a plugin path", backend_name);
        return AMEVA_BENCH_ERR_INVALID_ARGUMENT;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: pipe() failed (%s)", strerror(errno));
        return AMEVA_BENCH_ERR_FORK_FAILED;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: fork() failed (%s)", strerror(errno));
        return AMEVA_BENCH_ERR_FORK_FAILED;
    }
    if (pid == 0) {
        close(pipefd[0]);
        benchGpuChild(backend_name, ggml_ameva_plugin_path, pipefd[1]); // never returns
    }

    close(pipefd[1]);
    double childResult = -1.0;
    ssize_t n = read(pipefd[0], &childResult, sizeof(childResult));
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);

    if (n != (ssize_t)sizeof(childResult) || childResult < 0.0) {
        setMsg(err_buf, err_buf_size, "ameva_backend_bench: %s backend unavailable", backend_name);
        return AMEVA_BENCH_ERR_BACKEND_UNAVAILABLE;
    }
    *out_ms = childResult;
    return AMEVA_BENCH_OK;
}
