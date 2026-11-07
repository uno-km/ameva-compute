/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// split_gemv_bench [CPU_THREADS] [TOKENS]: can the CPU and the GPU together generate a token faster than either alone?
// (ameva-foundation N-20261107-decode-roofline, RQ-38). Token generation reads every weight once per token and both
// processors pull DRAM bandwidth that the other leaves unused (S21: CPU 24-26, GPU 17-19, both at once 28.6 GB/s), but
// every matrix split between them costs one hand-over (S21 OpenCL round trip 118-152 us).
// One "token" = the 183 GEMVs of Gemma 2 2B (26 layers x q 2048x2304, k 1024x2304, v 1024x2304, o 2304x2048,
// gate 9216x2304, up 9216x2304, down 2304x9216, then the head 256000x2304), 4-bit weights (16 bytes per 32 weights,
// one float scale per row: 0.5 + 4/K bytes per weight, about the bytes of q4_K), int8 activations.
//   GPU: rows [0, a) of every matrix in 16-row tiles (the AMEVA layout), work groups of 16 rows x 4 k-parts.
//   CPU: rows [a, N) row-major, NEON sdot, CPU_THREADS pinned worker threads (big cores) that spin between matrices.
// Modes per CPU share s (fraction of the rows): s = 0 "gpu_batched" (all 183 kernels enqueued, one wait per token, what
// the AMEVA backend does), then s = 0, 0.15, ... 1 with one hand-over per matrix (enqueue the GPU part, the CPU computes
// its part, spin on the GPU event). A sampled check compares both parts with a scalar reference (max relative error).
// Build: clang -O2 -march=armv8.2-a+dotprod -I<CL headers> -o split_gemv_bench split_gemv_bench.c -lOpenCL -lpthread
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, e_, __LINE__); exit(1); } } while (0)

static const char* kSrc =
    "__kernel __attribute__((reqd_work_group_size(16, 4, 1)))\n"
    "void gemv(__global const uint4* W, __global const char* x, __global const float* sc, __global float* y,\n"
    "          int U, int rows) {\n"
    "    __local float red[64];\n"
    "    const int lr = get_local_id(0), kp = get_local_id(1);\n"
    "    const int r = get_group_id(0) * 16 + lr;\n"
    "    int acc = 0;\n"
    "    if (r < rows) {\n"
    "        const int base = (r >> 4) * U * 16 + (r & 15);\n"
    "        const int T = min(16, rows - (r & ~15));\n"
    "        const int b2 = (r >> 4) * U * 16;\n"
    "        for (int u = kp; u < U; u += 4) {\n"
    "            const uint4 w = W[b2 + u * T + (r & 15)];\n"
    "            const char16 x0 = vload16(0, x + u * 32), x1 = vload16(0, x + u * 32 + 16);\n"
    "            const uint ws[4] = { w.x, w.y, w.z, w.w };\n"
    "            for (int c = 0; c < 4; ++c) {\n"
    "                const uint v = ws[c];\n"
    "                for (int b = 0; b < 4; ++b) {\n"
    "                    const int byte = (v >> (8 * b)) & 0xFF;\n"
    "                    const int j = 4 * c + b;\n"
    "                    acc += ((byte & 15) - 8) * ((const char*)&x0)[j] + ((byte >> 4) - 8) * ((const char*)&x1)[j];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    red[kp * 16 + lr] = (float)acc;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    if (kp == 0 && r < rows) y[r] = sc[r] * (red[lr] + red[16 + lr] + red[32 + lr] + red[48 + lr]);\n"
    "}\n";

typedef struct { int N, K; } Shape;

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

// one matrix: GPU part (tiled) in a cl buffer, CPU part (row-major) in host memory
typedef struct {
    int N, K, U, a;          // a = GPU rows
    cl_mem w, sc, y;
    uint8_t* cw;             // CPU rows, (N - a) x U x 16 bytes
    float* csc;
    float* cy;
} Mat;

// ---- CPU workers ----
typedef struct {
    atomic_int gen, done;
    const Mat* m;
    const int8_t* x;
    int nthreads;
} Pool;
static Pool g_pool;

static void cpu_rows(const Mat* m, const int8_t* x, int r0, int r1) {
    const uint8x16_t lo = vdupq_n_u8(15);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* w = m->cw + (size_t)r * m->U * 16;
        int32x4_t acc = vdupq_n_s32(0);
        for (int u = 0; u < m->U; ++u) {
            const uint8x16_t q = vld1q_u8(w + u * 16);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, lo)), eight);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), eight);
            acc = vdotq_s32(acc, wl, vld1q_s8(x + u * 32));
            acc = vdotq_s32(acc, wh, vld1q_s8(x + u * 32 + 16));
        }
        m->cy[r] = m->csc[r] * (float)vaddvq_s32(acc);
    }
}

typedef struct { int id, core; } WorkerArg;
static void* worker(void* p) {
    const WorkerArg* a = (const WorkerArg*)p;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(a->core, &s);
    if (sched_setaffinity(0, sizeof(s), &s) != 0) printf("SPLIT_NOTE worker %d unpinned (core %d)\n", a->id, a->core);
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load_explicit(&g_pool.gen, memory_order_acquire)) == seen) {}
        if (g < 0) return NULL;
        seen = g;
        const Mat* m = g_pool.m;
        const int n = m->N - m->a, T = g_pool.nthreads;
        const int r0 = n * a->id / T, r1 = n * (a->id + 1) / T;
        cpu_rows(m, g_pool.x, r0, r1);
        atomic_fetch_add_explicit(&g_pool.done, 1, memory_order_release);
    }
}

static void cpu_run(const Mat* m, const int8_t* x) {
    if (m->N - m->a <= 0) return;
    g_pool.m = m;
    g_pool.x = x;
    atomic_store(&g_pool.done, 0);
    atomic_fetch_add_explicit(&g_pool.gen, 1, memory_order_release);
    while (atomic_load_explicit(&g_pool.done, memory_order_acquire) < g_pool.nthreads) {}
}

int main(int argc, char** argv) {
    const int nthreads = argc > 1 ? atoi(argv[1]) : 3;
    const int tokens = argc > 2 ? atoi(argv[2]) : 5;
    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[128] = "";
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSrc, NULL, &err); CHECK(err);
    if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
        char log[8192] = "";
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "FAIL build\n%s\n", log);
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "gemv", &err); CHECK(err);

    // the matrices of one token
    Shape layer[7] = { {2048, 2304}, {1024, 2304}, {1024, 2304}, {2304, 2048}, {9216, 2304}, {9216, 2304}, {2304, 9216} };
    const int nm = 26 * 7 + 1;
    Shape* sh = (Shape*)malloc(sizeof(Shape) * nm);
    for (int l = 0; l < 26; ++l) for (int i = 0; i < 7; ++i) sh[l * 7 + i] = layer[i];
    sh[nm - 1] = (Shape){ 256000, 2304 };
    double bytes = 0;
    for (int i = 0; i < nm; ++i) bytes += (double)sh[i].N * sh[i].K / 2 + sh[i].N * 4.0;
    // activations per K
    int8_t* xs[3]; const int ks[3] = { 2304, 2048, 9216 };
    cl_mem xb[3];
    for (int i = 0; i < 3; ++i) {
        xs[i] = (int8_t*)malloc(ks[i]);
        for (int j = 0; j < ks[i]; ++j) xs[i][j] = (int8_t)((j * 37 + i * 11) % 255 - 127);
        xb[i] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ks[i], xs[i], &err); CHECK(err);
    }
    Mat* m = (Mat*)calloc(nm, sizeof(Mat));
    printf("SPLIT_DEVICE gpu=\"%s\" cpu_threads=%d tokens=%d matrices=%d bytes_per_token=%.0f MB\n", name, nthreads, tokens,
           nm, bytes / 1e6);
    const double shares[] = { -1.0, 0.0, 0.15, 0.25, 0.35, 0.45, 1.0 };
    // workers on the big cores (highest cpuinfo_max_freq first), the main thread drives the GPU
    {
        int cores[16], nc = 0; long fr[16];
        cpu_set_t al; sched_getaffinity(0, sizeof(al), &al);
        for (int c = 0; c < 16; ++c) if (CPU_ISSET(c, &al)) {
            char p[96]; snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
            FILE* f = fopen(p, "r"); long v = 0; if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
            cores[nc] = c; fr[nc++] = v;
        }
        for (int i = 0; i < nc; ++i) for (int j = i + 1; j < nc; ++j) if (fr[j] > fr[i]) {
            long t = fr[i]; fr[i] = fr[j]; fr[j] = t; int tc = cores[i]; cores[i] = cores[j]; cores[j] = tc; }
        g_pool.nthreads = nthreads;
        atomic_store(&g_pool.gen, 0);
        static WorkerArg wa[16]; pthread_t th;
        printf("SPLIT_CORES workers=");
        for (int t = 0; t < nthreads; ++t) {
            wa[t] = (WorkerArg){ t, cores[t % nc] };
            printf("%d%s", wa[t].core, t + 1 < nthreads ? "," : "\n");
            pthread_create(&th, NULL, worker, &wa[t]);
        }
    }
    for (int si = 0; si < (int)(sizeof(shares) / sizeof(shares[0])); ++si) {
        const double s = shares[si] < 0 ? 0.0 : shares[si];
        const int batched = shares[si] < 0;
        // build the matrices for this share (GPU rows a multiple of 16)
        for (int i = 0; i < nm; ++i) {
            Mat* mm = &m[i];
            if (mm->w) { clReleaseMemObject(mm->w); clReleaseMemObject(mm->sc); clReleaseMemObject(mm->y); }
            free(mm->cw); free(mm->csc); free(mm->cy);
            memset(mm, 0, sizeof(*mm));
        }
        // every matrix in its own memory, as in a model (no layer reads another layer's bytes from a cache)
        for (int i = 0; i < nm; ++i) {
            Mat* mm = &m[i];
            mm->N = sh[i].N; mm->K = sh[i].K; mm->U = sh[i].K / 32;
            mm->a = (int)((1.0 - s) * mm->N) / 16 * 16;
            const int ga = mm->a, ca = mm->N - mm->a, U = mm->U;
            if (ga > 0) {
                uint8_t* t = (uint8_t*)malloc((size_t)ga * U * 16);
                for (size_t b = 0; b < (size_t)ga * U * 16; ++b) t[b] = (uint8_t)(b * 2654435761u >> 24);
                float* sc = (float*)malloc(sizeof(float) * ga);
                for (int r = 0; r < ga; ++r) sc[r] = 1e-3f * (1 + r % 7);
                mm->w = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)ga * U * 16, t, &err); CHECK(err);
                mm->sc = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ga, sc, &err); CHECK(err);
                mm->y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * ga, NULL, &err); CHECK(err);
                free(t); free(sc);
            }
            if (ca > 0) {
                mm->cw = (uint8_t*)malloc((size_t)ca * U * 16);
                for (size_t b = 0; b < (size_t)ca * U * 16; ++b) mm->cw[b] = (uint8_t)((b + 7) * 2246822519u >> 24);
                mm->csc = (float*)malloc(sizeof(float) * ca);
                for (int r = 0; r < ca; ++r) mm->csc[r] = 1e-3f * (1 + r % 5);
                mm->cy = (float*)malloc(sizeof(float) * ca);
            }
        }
        float* gy = (float*)malloc(sizeof(float) * 256000);
        double best = 1e30, med[16]; int nt = 0;
        for (int t = -1; t < tokens; ++t) {
            const double t0 = now_us();
            for (int i = 0; i < nm; ++i) {
                Mat* mm = &m[i];
                const int xi = mm->K == 2304 ? 0 : mm->K == 2048 ? 1 : 2;
                cl_event ev = NULL;
                if (mm->a > 0) {
                    const int U = mm->U, rows = mm->a;
                    CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &mm->w)); CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &xb[xi]));
                    CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &mm->sc)); CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &mm->y));
                    CHECK(clSetKernelArg(k, 4, sizeof(int), &U)); CHECK(clSetKernelArg(k, 5, sizeof(int), &rows));
                    const size_t g[2] = { (size_t)(rows + 15) / 16 * 16, 4 }, l[2] = { 16, 4 };
                    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, batched ? NULL : &ev));
                    if (!batched) CHECK(clFlush(q));
                }
                if (!batched) {
                    cpu_run(mm, xs[xi]);
                    if (ev) {
                        cl_int st = CL_QUEUED;
                        while (st != CL_COMPLETE) {
                            CHECK(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL));
                            if (st < 0) { fprintf(stderr, "FAIL kernel status %d\n", st); return 1; }
                        }
                        clReleaseEvent(ev);
                    }
                }
            }
            if (batched) CHECK(clFinish(q));
            const double us = now_us() - t0;
            if (t >= 0) { med[nt++] = us; if (us < best) best = us; }
        }
        for (int i = 0; i < nt; ++i) for (int j = i + 1; j < nt; ++j) if (med[j] < med[i]) { double x = med[i]; med[i] = med[j]; med[j] = x; }
        // sampled check on the head (largest matrix): GPU rows and CPU rows against a scalar reference
        double maxrel = 0;
        {
            Mat* mm = &m[nm - 1];
            const int8_t* x = xs[0];
            if (mm->a > 0) CHECK(clEnqueueReadBuffer(q, mm->y, CL_TRUE, 0, sizeof(float) * mm->a, gy, 0, NULL, NULL));
            for (int smp = 0; smp < 64; ++smp) {
                const int r = (int)((long)smp * 4001 % mm->N);
                const int gpu = r < mm->a;
                // reference from the same bytes the processor read
                double ref = 0;
                if (gpu) {
                    // regenerate the GPU's tile bytes: buffer byte index of unit (r, u) = ((r>>4)*U*16 + u*T + (r&15)) * 16
                    const int U = mm->U, T = (mm->a - (r & ~15)) < 16 ? mm->a - (r & ~15) : 16;
                    for (int u = 0; u < U; ++u) {
                        const size_t base = ((size_t)(r >> 4) * U * 16 + (size_t)u * T + (r & 15)) * 16;
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t byte = (uint8_t)((base + j) * 2654435761u >> 24);
                            ref += ((byte & 15) - 8) * x[u * 32 + j] + ((byte >> 4) - 8) * x[u * 32 + 16 + j];
                        }
                    }
                    ref *= 1e-3 * (1 + r % 7);
                } else {
                    const int cr = r - mm->a, U = mm->U;
                    for (int u = 0; u < U; ++u)
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t byte = (uint8_t)(((size_t)cr * U * 16 + u * 16 + j + 7) * 2246822519u >> 24);
                            ref += ((byte & 15) - 8) * x[u * 32 + j] + ((byte >> 4) - 8) * x[u * 32 + 16 + j];
                        }
                    ref *= 1e-3 * (1 + cr % 5);
                }
                const double got = gpu ? gy[r] : mm->cy[r - mm->a];
                const double rel = fabs(got - ref) / (fabs(ref) > 1e-6 ? fabs(ref) : 1.0);
                if (rel > maxrel) maxrel = rel;
            }
        }
        printf("SPLIT mode=%s cpu_share=%.2f median_ms=%.2f best_ms=%.2f tokens_per_s=%.2f GB/s=%.1f check_max_rel=%.2e %s\n",
               batched ? "gpu_batched" : "handover", s, med[nt / 2] / 1e3, best / 1e3, 1e6 / med[nt / 2],
               bytes / med[nt / 2] / 1e3, maxrel, maxrel < 1e-4 ? "PASS" : "FAIL");
        fflush(stdout);
        free(gy);
    }
    atomic_store(&g_pool.gen, -1);
    return 0;
}
