/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// cl_peak_bench [MB] [REPS]: the measured ceilings of an OpenCL GPU, the denominators of utilization figures
// (achieved / ceiling) that compare engines across models and devices (ameva-foundation N-20261107-prior-art,
// research program "roofline" item):
//   read_bw   GB/s   every work-item sums float4 loads, consecutive work-items read consecutive float4 (MB buffer)
//   copy_bw   GB/s   float4 load and store (bytes counted both ways)
//   fp32_fma  GFLOP/s  8 independent float4 fma chains per work-item, 2 flops per fma lane
//   fp16_fma  GFLOP/s  the same with half8 (cl_khr_fp16)
//   i8_dot    GOP/s    8 independent arm_dot_acc(char4, char4, int) chains, 8 ops per dot
//                      (cl_arm_integer_dot_product_accumulate_int8)
// Each figure is the median kernel time (event profiling) of REPS runs after one warm-up run. The compute kernels
// write their sums so that the compiler keeps the arithmetic; the bandwidth kernels check one sum on the CPU.
// Build: clang -O2 -o cl_peak_bench cl_peak_bench.c -lOpenCL
// Run:   LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/vendor/lib64/egl ./cl_peak_bench 256 9
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, e_, __LINE__); exit(1); } } while (0)

static const char* kSrc =
    "__kernel void read_bw(__global const float4* src, __global float* out, int n4, int iters) {\n"
    "    const int g = get_global_id(0), gs = get_global_size(0);\n"
    "    float4 s = (float4)(0.0f);\n"
    "    for (int i = g; i < n4; i += gs) s += src[i];\n"
    "    out[g] = s.x + s.y + s.z + s.w;\n"
    "}\n"
    "__kernel void copy_bw(__global const float4* src, __global float4* dst, int n4, int iters) {\n"
    "    const int g = get_global_id(0), gs = get_global_size(0);\n"
    "    for (int i = g; i < n4; i += gs) dst[i] = src[i];\n"
    "}\n"
    "__kernel void fp32_fma(__global float* out, float a, float b, int iters) {\n"
    "    const int g = get_global_id(0);\n"
    "    float4 x0 = (float4)(g), x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f, x4 = x0 + 4.0f, x5 = x0 + 5.0f,\n"
    "           x6 = x0 + 6.0f, x7 = x0 + 7.0f;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);\n"
    "        x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);\n"
    "    }\n"
    "    float4 s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;\n"
    "    out[g] = s.x + s.y + s.z + s.w;\n"
    "}\n"
    "#ifdef HAS_FP16\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void fp16_fma(__global float* out, float af, float bf, int iters) {\n"
    "    const int g = get_global_id(0);\n"
    "    const half a = (half)af, b = (half)bf;\n"
    "    half8 x0 = (half8)((half)(g & 7)), x1 = x0 + (half)1, x2 = x0 + (half)2, x3 = x0 + (half)3, x4 = x0 + (half)4,\n"
    "          x5 = x0 + (half)5, x6 = x0 + (half)6, x7 = x0 + (half)7;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);\n"
    "        x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);\n"
    "    }\n"
    "    half8 s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;\n"
    "    out[g] = (float)(s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7);\n"
    "}\n"
    "#endif\n"
    "#ifdef HAS_I8DOT\n"
    "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable\n"
    "__kernel void i8_dot(__global float* out, float af, float bf, int iters) {\n"
    "    const int g = get_global_id(0);\n"
    "    const char4 a = (char4)((char)af, 1, -1, 2), b = (char4)((char)bf, -2, 1, 1);\n"
    "    int x0 = g, x1 = g + 1, x2 = g + 2, x3 = g + 3, x4 = g + 4, x5 = g + 5, x6 = g + 6, x7 = g + 7;\n"
    "    char4 c0 = a, c1 = b;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        x0 = arm_dot_acc(c0, c1, x0); x1 = arm_dot_acc(c1, c0, x1); x2 = arm_dot_acc(c0, c0, x2);\n"
    "        x3 = arm_dot_acc(c1, c1, x3); x4 = arm_dot_acc(c0, c1, x4); x5 = arm_dot_acc(c1, c0, x5);\n"
    "        x6 = arm_dot_acc(c0, c0, x6); x7 = arm_dot_acc(c1, c1, x7);\n"
    "        c0 = c0.yzwx;\n"
    "    }\n"
    "    out[g] = (float)(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);\n"
    "}\n"
    "#endif\n";

static int cmp_u64(const void* a, const void* b) {
    const cl_ulong x = *(const cl_ulong*)a, y = *(const cl_ulong*)b;
    return x < y ? -1 : x > y;
}

static double median_ms(cl_command_queue q, cl_kernel k, size_t global, size_t local, int reps) {
    cl_ulong t[64];
    if (reps > 64) reps = 64;
    for (int r = -1; r < reps; ++r) {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, local ? &local : NULL, 0, NULL, &ev));
        CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0 = 0, t1 = 0;
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL));
        clReleaseEvent(ev);
        if (r >= 0) t[r] = t1 - t0;
    }
    qsort(t, (size_t)reps, sizeof(t[0]), cmp_u64);
    return t[reps / 2] / 1e6;
}

int main(int argc, char** argv) {
    const int mb = argc > 1 ? atoi(argv[1]) : 256;
    const int reps = argc > 2 ? atoi(argv[2]) : 9;
    if (mb < 16 || reps < 1) { fprintf(stderr, "usage: cl_peak_bench [MB >= 16] [REPS >= 1]\n"); return 2; }
    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[256] = "", ver[256] = "", ext[8192] = "";
    cl_uint cus = 0, mhz = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, NULL);
    const int fp16 = strstr(ext, "cl_khr_fp16") != NULL;
    const int i8 = strstr(ext, "cl_arm_integer_dot_product_accumulate_int8") != NULL;
    printf("PEAK_DEVICE name=\"%s\" version=\"%s\" compute_units=%u max_clock_mhz=%u fp16=%d i8dot=%d buffer_mb=%d reps=%d\n",
           name, ver, cus, mhz, fp16, i8, mb, reps);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    const cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSrc, NULL, &err); CHECK(err);
    char opts[64];
    snprintf(opts, sizeof(opts), "%s%s", fp16 ? "-DHAS_FP16 " : "", i8 ? "-DHAS_I8DOT" : "");
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[16384] = "";
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "FAIL build:\n%s\n", log);
        return 1;
    }
    const size_t bytes = (size_t)mb << 20, n4 = bytes / 16;
    float* host = (float*)malloc(bytes);
    for (size_t i = 0; i < bytes / 4; ++i) host[i] = (float)(i & 1023) * 0.001f;
    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host, &err); CHECK(err);
    cl_mem dst = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err); CHECK(err);
    const size_t gbw = (size_t)cus * 1024, gcomp = (size_t)cus * 4096;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (gcomp > gbw ? gcomp : gbw) * sizeof(float), NULL, &err); CHECK(err);
    const int n4i = (int)n4, one = 1, iters = 4096;
    const float a = 0.999f, b = 0.001f;

    cl_kernel k = clCreateKernel(prog, "read_bw", &err); CHECK(err);
    CHECK(clSetKernelArg(k, 0, sizeof(src), &src)); CHECK(clSetKernelArg(k, 1, sizeof(out), &out));
    CHECK(clSetKernelArg(k, 2, sizeof(int), &n4i)); CHECK(clSetKernelArg(k, 3, sizeof(int), &one));
    double ms = median_ms(q, k, gbw, 64, reps);
    {   /* one partial sum checked on the CPU: work-item 0 reads float4 0, gbw, 2 gbw, ... */
        float got = 0.0f;
        CHECK(clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof(float), &got, 0, NULL, NULL));
        double want = 0.0;
        for (size_t i = 0; i < n4; i += gbw) for (int j = 0; j < 4; ++j) want += host[i * 4 + j];
        const double rel = want != 0.0 ? (got - want) / want : got;
        printf("PEAK read_bw=%.2f GB/s (%.3f ms, %zu work-items, check rel_err=%.2e%s)\n", bytes / ms / 1e6, ms, gbw, rel,
               (rel > 1e-3 || rel < -1e-3) ? " FAILED" : "");
    }
    clReleaseKernel(k);

    k = clCreateKernel(prog, "copy_bw", &err); CHECK(err);
    CHECK(clSetKernelArg(k, 0, sizeof(src), &src)); CHECK(clSetKernelArg(k, 1, sizeof(dst), &dst));
    CHECK(clSetKernelArg(k, 2, sizeof(int), &n4i)); CHECK(clSetKernelArg(k, 3, sizeof(int), &one));
    ms = median_ms(q, k, gbw, 64, reps);
    printf("PEAK copy_bw=%.2f GB/s (%.3f ms, read + write)\n", 2.0 * bytes / ms / 1e6, ms);
    clReleaseKernel(k);

    const char* names[3] = { "fp32_fma", "fp16_fma", "i8_dot" };
    const int have[3] = { 1, fp16, i8 };
    const double ops_per_iter[3] = { 8 * 4 * 2, 8 * 8 * 2, 8 * 8 };
    for (int t = 0; t < 3; ++t) {
        if (!have[t]) { printf("PEAK %s=NA (extension missing)\n", names[t]); continue; }
        k = clCreateKernel(prog, names[t], &err); CHECK(err);
        CHECK(clSetKernelArg(k, 0, sizeof(out), &out)); CHECK(clSetKernelArg(k, 1, sizeof(float), &a));
        CHECK(clSetKernelArg(k, 2, sizeof(float), &b)); CHECK(clSetKernelArg(k, 3, sizeof(int), &iters));
        ms = median_ms(q, k, gcomp, 64, reps);
        printf("PEAK %s=%.1f %s (%.3f ms, %zu work-items x %d iterations)\n", names[t],
               ops_per_iter[t] * iters * (double)gcomp / ms / 1e6, t == 2 ? "GOP/s" : "GFLOP/s", ms, gcomp, iters);
        clReleaseKernel(k);
    }
    clReleaseMemObject(src); clReleaseMemObject(dst); clReleaseMemObject(out);
    clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(host);
    return 0;
}
