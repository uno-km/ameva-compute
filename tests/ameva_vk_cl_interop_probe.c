// SPDX-License-Identifier: Apache-2.0
// ameva_vk_cl_interop_probe: can an OpenCL kernel on a Mali GPU work on the memory of an ac_runtime (Vulkan) buffer
// without a copy? Stage B of docs/design/opencl-prompt-path.md runs the prompt-sized quantized MUL_MATs of a
// ggml-ameva graph as OpenCL block-integer kernels on the graph's own buffers; that needs this to hold.
//
// 1. ac_runtime (Vulkan) creates a storage buffer and maps it persistently (ac_runtime_buffer_get_mapped_ptr).
// 2. The host writes a float pattern into the first half through that mapping.
// 3. OpenCL imports the same memory with clImportMemoryARM (CL_IMPORT_TYPE_HOST_ARM): no allocation, no copy.
// 4. An OpenCL kernel reads the first half and writes 2x + 1 into the second half.
// 5. The host invalidates the Vulkan mapping and checks every value; times are printed for the import and for one
//    kernel round trip (enqueue to clFinish).
// Prints VK_CL_INTEROP=PASS or FAIL with the reason. Exit status 0 only on PASS.
// Result (2026-10-06, S21 and A53): FAIL at step 3, CL_OUT_OF_HOST_MEMORY (-6); the AHB path is in
// ameva_vk_cl_ahb_host_probe.c and ameva_vk_cl_gpu_handoff_probe.c.
// Build: clang -O2 -I include -o probe tests/ameva_vk_cl_interop_probe.c -L<dir> -lac_runtime -lOpenCL -lm
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ac_runtime.h"

#ifndef CL_IMPORT_TYPE_ARM
#define CL_IMPORT_TYPE_ARM 0x40B2
#endif
#ifndef CL_IMPORT_TYPE_HOST_ARM
#define CL_IMPORT_TYPE_HOST_ARM 0x40B3
#endif
typedef intptr_t probe_import_prop;
typedef cl_mem (*clImportMemoryARM_fn)(cl_context, cl_mem_flags, const probe_import_prop*, void*, size_t, cl_int*);

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static int fail(const char* why, long code) {
    printf("VK_CL_INTEROP=FAIL (%s, code %ld)\n", why, code);
    return 1;
}

static const char* kSrc =
    "__kernel void twice_plus_one(__global float* a, int nhalf) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < nhalf) a[nhalf + i] = a[i] * 2.0f + 1.0f;\n"
    "}\n";

int main(int argc, char** argv) {
    const size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 16;
    const size_t bytes = mb << 20, n = bytes / sizeof(float), half = n / 2;

    // 1. Vulkan buffer through ac_runtime
    AcContextDesc cd = { AC_ABI_VERSION, (uint32_t)sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
    AcContext* ctx = NULL;
    AcResult r = ac_runtime_context_create(&cd, &ctx);
    if (r != AC_SUCCESS) return fail("ac_runtime_context_create", (long)r);
    char dev[256] = { 0 };
    ac_runtime_context_get_device_name(ctx, dev, sizeof(dev));
    AcBufferDesc bd = { AC_ABI_VERSION, (uint32_t)sizeof(AcBufferDesc), bytes,
                        AC_BUFFER_USAGE_STORAGE | AC_BUFFER_USAGE_TRANSFER_SRC | AC_BUFFER_USAGE_TRANSFER_DST, 0 };
    AcBuffer* buf = NULL;
    r = ac_runtime_buffer_create(ctx, &bd, &buf);
    if (r != AC_SUCCESS) return fail("ac_runtime_buffer_create", (long)r);
    void* p = NULL;
    r = ac_runtime_buffer_get_mapped_ptr(buf, &p);
    if (r != AC_SUCCESS || !p) return fail("ac_runtime_buffer_get_mapped_ptr", (long)r);
    printf("vulkan device: %s, buffer %zu MB mapped at %p (page offset %lu)\n", dev, mb, p,
           (unsigned long)((uintptr_t)p & 4095u));

    // 2. pattern through the Vulkan mapping
    float* f = (float*)p;
    for (size_t i = 0; i < half; ++i) f[i] = (float)(i % 1000003) * 0.5f;
    for (size_t i = half; i < n; ++i) f[i] = -1.0f;
    ac_runtime_buffer_flush(buf, 0, bytes);

    // 3. OpenCL import of the same memory
    cl_platform_id plat;
    cl_device_id cdev;
    cl_int e = clGetPlatformIDs(1, &plat, NULL);
    if (e != CL_SUCCESS) return fail("clGetPlatformIDs", e);
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cdev, NULL);
    if (e != CL_SUCCESS) return fail("clGetDeviceIDs", e);
    cl_context cctx = clCreateContext(NULL, 1, &cdev, NULL, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateContext", e);
    cl_command_queue q = clCreateCommandQueueWithProperties(cctx, cdev, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateCommandQueueWithProperties", e);
    clImportMemoryARM_fn import = (clImportMemoryARM_fn)clGetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM");
    if (!import) return fail("clImportMemoryARM not exported", 0);
    const probe_import_prop props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_HOST_ARM, 0 };
    double t0 = now_ms();
    cl_mem m = import(cctx, CL_MEM_READ_WRITE, props, p, bytes, &e);
    const double t_import = now_ms() - t0;
    if (e != CL_SUCCESS || !m) return fail("clImportMemoryARM(CL_IMPORT_TYPE_HOST_ARM) of the Vulkan mapping", e);

    // 4. kernel on the imported memory
    cl_program prog = clCreateProgramWithSource(cctx, 1, &kSrc, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateProgramWithSource", e);
    e = clBuildProgram(prog, 1, &cdev, "", NULL, NULL);
    if (e != CL_SUCCESS) return fail("clBuildProgram", e);
    cl_kernel k = clCreateKernel(prog, "twice_plus_one", &e);
    if (e != CL_SUCCESS) return fail("clCreateKernel", e);
    const cl_int h = (cl_int)half;
    clSetKernelArg(k, 0, sizeof(cl_mem), &m);
    clSetKernelArg(k, 1, sizeof(cl_int), &h);
    const size_t g = (half + 63) / 64 * 64, l = 64;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        t0 = now_ms();
        e = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
        if (e != CL_SUCCESS) return fail("clEnqueueNDRangeKernel", e);
        e = clFinish(q);
        if (e != CL_SUCCESS) return fail("clFinish", e);
        const double t = now_ms() - t0;
        if (t < best) best = t;
    }

    // 5. check through the Vulkan mapping
    ac_runtime_buffer_invalidate(buf, 0, bytes);
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < half; ++i) {
        const float want = f[i] * 2.0f + 1.0f;
        if (f[half + i] != want) {
            if (!bad) first_bad = i;
            ++bad;
        }
    }
    const double got_v = bad ? (double)f[half + first_bad] : 0.0, want_v = bad ? (double)(f[first_bad] * 2.0f + 1.0f) : 0.0;
    printf("import %.3f ms, kernel round trip (best of 5, %zu MB read + written) %.3f ms\n", t_import, mb / 2, best);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseMemObject(m);
    clReleaseCommandQueue(q);
    clReleaseContext(cctx);
    ac_runtime_buffer_destroy(buf);
    ac_runtime_context_destroy(ctx);
    if (bad) {
        printf("VK_CL_INTEROP=FAIL (%zu of %zu values wrong, first at %zu: %g, want %g)\n", bad, half, first_bad, got_v,
               want_v);
        return 1;
    }
    printf("VK_CL_INTEROP=PASS (%zu values written by OpenCL read back through the Vulkan mapping)\n", half);
    return 0;
}
