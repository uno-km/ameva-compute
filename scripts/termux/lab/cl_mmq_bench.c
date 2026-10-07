/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// cl_mmq_bench M N K [REPS]: OpenCL int8 matrix multiplication on Arm Mali (lab experiment).
//   dst[m][n] = sum_k (wq[n][k] * ws[n][k/32]) * (aq[m][k] * as[m][k/32])   (Q8_0-style blocks of 32, int8 values)
// Two kernels with the same work split (one work-item: 4 weight rows x 4 activation rows, 32 values per step):
//   i8dot  packed int8 dot products, arm_dot_acc (cl_arm_integer_dot_product_accumulate_int8), one int sum per block
//   f32    the same values converted to float and multiplied with fma
// Both are checked against a double-precision CPU reference; the median kernel time of REPS runs (event profiling)
// gives multiply-accumulates per second. The point is the int8 dot product that the r32p1 driver offers in OpenCL but
// not in Vulkan; the kernels are not tuned, so the figures bound what a tuned kernel reaches from below.
// Build: clang -O2 -o cl_mmq_bench cl_mmq_bench.c -lOpenCL
// Run:   LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/vendor/lib64/egl ./cl_mmq_bench 64 4864 896 20
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* kSrc =
    "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable\n"
    "__kernel void i8dot(__global const char* wq, __global const float* ws, __global const char* aq,\n"
    "                    __global const float* as, __global float* dst, int M, int N, int K) {\n"
    "    const int n0 = get_global_id(0) * 4, m0 = get_global_id(1) * 4, B = K / 32;\n"
    "    if (n0 >= N || m0 >= M) return;\n"
    "    float acc[4][4] = {{0}};\n"
    "    for (int b = 0; b < B; ++b) {\n"
    "        char16 w[4][2], a[4][2];\n"
    "        for (int i = 0; i < 4; ++i) {\n"
    "            w[i][0] = vload16(0, wq + (size_t)(n0 + i) * K + b * 32);\n"
    "            w[i][1] = vload16(0, wq + (size_t)(n0 + i) * K + b * 32 + 16);\n"
    "            a[i][0] = vload16(0, aq + (size_t)(m0 + i) * K + b * 32);\n"
    "            a[i][1] = vload16(0, aq + (size_t)(m0 + i) * K + b * 32 + 16);\n"
    "        }\n"
    "        for (int i = 0; i < 4; ++i) {\n"
    "            const float wsc = ws[(n0 + i) * B + b];\n"
    "            for (int j = 0; j < 4; ++j) {\n"
    "                int s = 0;\n"
    "                for (int h = 0; h < 2; ++h) {\n"
    "                    s = arm_dot_acc(w[i][h].s0123, a[j][h].s0123, s);\n"
    "                    s = arm_dot_acc(w[i][h].s4567, a[j][h].s4567, s);\n"
    "                    s = arm_dot_acc(w[i][h].s89ab, a[j][h].s89ab, s);\n"
    "                    s = arm_dot_acc(w[i][h].scdef, a[j][h].scdef, s);\n"
    "                }\n"
    "                acc[i][j] += (float)s * wsc * as[(m0 + j) * B + b];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    for (int j = 0; j < 4; ++j)\n"
    "        for (int i = 0; i < 4; ++i) dst[(size_t)(m0 + j) * N + n0 + i] = acc[i][j];\n"
    "}\n"
    "__kernel void f32(__global const char* wq, __global const float* ws, __global const char* aq,\n"
    "                  __global const float* as, __global float* dst, int M, int N, int K) {\n"
    "    const int n0 = get_global_id(0) * 4, m0 = get_global_id(1) * 4, B = K / 32;\n"
    "    if (n0 >= N || m0 >= M) return;\n"
    "    float acc[4][4] = {{0}};\n"
    "    for (int b = 0; b < B; ++b) {\n"
    "        float wsc[4], asc[4];\n"
    "        for (int i = 0; i < 4; ++i) { wsc[i] = ws[(n0 + i) * B + b]; asc[i] = as[(m0 + i) * B + b]; }\n"
    "        for (int q = 0; q < 8; ++q) {\n"
    "            float4 w[4], a[4];\n"
    "            for (int i = 0; i < 4; ++i) {\n"
    "                w[i] = convert_float4(vload4(0, wq + (size_t)(n0 + i) * K + b * 32 + q * 4)) * wsc[i];\n"
    "                a[i] = convert_float4(vload4(0, aq + (size_t)(m0 + i) * K + b * 32 + q * 4)) * asc[i];\n"
    "            }\n"
    "            for (int i = 0; i < 4; ++i)\n"
    "                for (int j = 0; j < 4; ++j) acc[i][j] += dot(w[i], a[j]);\n"
    "        }\n"
    "    }\n"
    "    for (int j = 0; j < 4; ++j)\n"
    "        for (int i = 0; i < 4; ++i) dst[(size_t)(m0 + j) * N + n0 + i] = acc[i][j];\n"
    "}\n"
    // Arithmetic peak: 8 independent chains per work-item, each step's operand is the next chain's previous value,
    // so nothing folds; one arm_dot_acc and one float4 fma are 4 multiply-accumulates each.
    "#define STEP8(OP) t0 = OP(s1, s0); t1 = OP(s2, s1); t2 = OP(s3, s2); t3 = OP(s4, s3); \\\n"
    "    t4 = OP(s5, s4); t5 = OP(s6, s5); t6 = OP(s7, s6); t7 = OP(s0, s7); \\\n"
    "    s0 = t0; s1 = t1; s2 = t2; s3 = t3; s4 = t4; s5 = t5; s6 = t6; s7 = t7;\n"
    "#define DOT(x, acc) arm_dot_acc(as_char4(x), b, acc)\n"
    "__kernel void peak_i8(__global int* out, int iters) {\n"
    "    const char4 b = (char4)(1, -1, 2, (char)get_local_id(0));\n"
    "    int s0 = get_global_id(0), s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 + 4, s5 = s0 + 5, s6 = s0 + 6,\n"
    "        s7 = s0 + 7, t0, t1, t2, t3, t4, t5, t6, t7;\n"
    "    for (int i = 0; i < iters; ++i) { STEP8(DOT) }\n"
    "    out[get_global_id(0)] = s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^ s6 ^ s7;\n"
    "}\n"
    "#define FMA4(x, acc) fma(x, c, acc)\n"
    "__kernel void peak_f32(__global float* out, int iters) {\n"
    "    const float4 c = (float4)(0.999f, 0.998f, 0.997f, 0.996f + get_local_id(0) * 1e-6f);\n"
    "    float4 s0 = (float4)(get_global_id(0)) * 1e-6f, s1 = s0 + 1e-3f, s2 = s0 + 2e-3f, s3 = s0 + 3e-3f,\n"
    "           s4 = s0 + 4e-3f, s5 = s0 + 5e-3f, s6 = s0 + 6e-3f, s7 = s0 + 7e-3f, t0, t1, t2, t3, t4, t5, t6, t7;\n"
    "    for (int i = 0; i < iters; ++i) { STEP8(FMA4) }\n"
    "    const float4 r = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;\n"
    "    out[get_global_id(0)] = r.x + r.y + r.z + r.w;\n"
    "}\n";


#define CHECK(x)                                                                         \
    do {                                                                                 \
        cl_int e_ = (x);                                                                 \
        if (e_ != CL_SUCCESS) {                                                          \
            fprintf(stderr, "%s:%d: %s failed with %d\n", __FILE__, __LINE__, #x, e_);   \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static unsigned g_seed = 0x9e3779b9u;
static unsigned rnd(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

// median kernel time (ms) of R runs after two warm-up runs
static double time_kernel(cl_command_queue q, cl_kernel k, cl_uint dims, const size_t* gsz, const size_t* lsz, int R) {
    double* t = malloc(sizeof(double) * R);
    for (int r = -2; r < R; ++r) {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(q, k, dims, NULL, gsz, lsz, 0, NULL, &ev));
        CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL));
        if (r >= 0) t[r] = (double)(t1 - t0) * 1e-6;
        clReleaseEvent(ev);
    }
    qsort(t, R, sizeof(double), cmp_double);
    const double med = t[R / 2];
    free(t);
    return med;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s M N K [REPS]   (M = 0: arithmetic peak of int8 dot and fp32 fma only)\n", argv[0]);
        return 2;
    }
    const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), R = argc > 4 ? atoi(argv[4]) : 20;
    if (M % 4 || N % 4 || K % 32 || R < 1) {
        fprintf(stderr, "M and N must be multiples of 4, K of 32\n");
        return 2;
    }
    const int B = K / 32;
    const int MM = M ? M : 4, NN = M ? N : 4; // peak mode: small placeholder matrices
    signed char* wq = malloc((size_t)NN * K);
    signed char* aq = malloc((size_t)MM * K);
    float* ws = malloc(sizeof(float) * NN * B);
    float* as = malloc(sizeof(float) * MM * B);
    float* out = malloc(sizeof(float) * MM * NN);
    for (size_t i = 0; i < (size_t)N * K; ++i) wq[i] = (signed char)((int)(rnd() % 255) - 127);
    for (size_t i = 0; i < (size_t)M * K; ++i) aq[i] = (signed char)((int)(rnd() % 255) - 127);
    for (int i = 0; i < N * B; ++i) ws[i] = (float)(rnd() % 1000 + 1) * 1e-5f;
    for (int i = 0; i < M * B; ++i) as[i] = (float)(rnd() % 1000 + 1) * 1e-4f;

    cl_platform_id plat;
    cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[256], ver[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err);
    const cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CHECK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSrc, NULL, &err);
    CHECK(err);
    if (clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0", NULL, NULL) != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    if (M == 0) {
        // arithmetic peak: 64 Ki work-items of 64, ITERS steps of 8 chains (4 multiply-accumulates each)
        const int iters = 2048;
        const size_t g = 64 * 1024, l = 64;
        cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * g, NULL, &err);
        CHECK(err);
        printf("device=%s | %s arithmetic peak reps=%d\n", name, ver, R);
        const char* pk[2] = { "peak_i8", "peak_f32" };
        for (int kk = 0; kk < 2; ++kk) {
            cl_kernel k = clCreateKernel(prog, pk[kk], &err);
            CHECK(err);
            CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &bo));
            CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
            const double med = time_kernel(q, k, 1, &g, &l, R);
            printf("kernel=%s median_ms=%.4f gmac_per_s=%.2f\n", pk[kk], med, (double)g * iters * 8 * 4 / (med * 1e-3) / 1e9);
            clReleaseKernel(k);
        }
        clReleaseMemObject(bo);
        clReleaseProgram(prog);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return 0;
    }
    cl_mem bwq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)N * K, wq, &err);
    CHECK(err);
    cl_mem baq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)M * K, aq, &err);
    CHECK(err);
    cl_mem bws = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * B, ws, &err);
    CHECK(err);
    cl_mem bas = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * B, as, &err);
    CHECK(err);
    cl_mem bd = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);
    CHECK(err);

    printf("device=%s | %s M=%d N=%d K=%d reps=%d\n", name, ver, M, N, K, R);
    const char* names[2] = { "i8dot", "f32" };
    for (int kk = 0; kk < 2; ++kk) {
        cl_kernel k = clCreateKernel(prog, names[kk], &err);
        CHECK(err);
        CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &bwq));
        CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &bws));
        CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &baq));
        CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &bas));
        CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &bd));
        CHECK(clSetKernelArg(k, 5, sizeof(int), &M));
        CHECK(clSetKernelArg(k, 6, sizeof(int), &N));
        CHECK(clSetKernelArg(k, 7, sizeof(int), &K));
        const size_t gsz[2] = { (size_t)N / 4, (size_t)M / 4 };
        const double med = time_kernel(q, k, 2, gsz, NULL, R);
        CHECK(clEnqueueReadBuffer(q, bd, CL_TRUE, 0, sizeof(float) * M * N, out, 0, NULL, NULL));
        // reference on a sample of outputs (rows m spread over M, every 7th n)
        double max_rel = 0.0;
        for (int m = 0; m < M; m += (M > 8 ? M / 8 : 1)) {
            for (int n = 0; n < N; n += 7) {
                double ref = 0.0, mag = 0.0;
                for (int b = 0; b < B; ++b) {
                    long long s = 0;
                    for (int i = 0; i < 32; ++i) s += (long long)wq[(size_t)n * K + b * 32 + i] * aq[(size_t)m * K + b * 32 + i];
                    const double sc = (double)ws[n * B + b] * as[m * B + b];
                    ref += (double)s * sc;
                    mag += fabs((double)s * sc);
                }
                const double rel = fabs(out[(size_t)m * N + n] - ref) / (mag > 0 ? mag : 1.0);
                if (rel > max_rel) max_rel = rel;
            }
        }
        printf("kernel=%s median_ms=%.4f gmac_per_s=%.2f max_rel_err=%.2e %s\n", names[kk], med,
               (double)M * N * K / (med * 1e-3) / 1e9, max_rel, max_rel < 1e-4 ? "PASS" : "FAIL");
        clReleaseKernel(k);
    }
    clReleaseMemObject(bwq);
    clReleaseMemObject(baq);
    clReleaseMemObject(bws);
    clReleaseMemObject(bas);
    clReleaseMemObject(bd);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
