/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// cl_hw_bench [MHz] [REPS]: per-unit throughput of an OpenCL GPU, in units per clock per compute unit, the hardware
// side of a kernel cost model (ameva-foundation N-20261107-hw-bottleneck):
//   gld     global loads of 4..64 bytes per work-item from a working set of 4 KiB .. 64 MiB (L1, L2, DRAM)
//   lld     local-memory loads of 4..64 bytes per work-item, each lane its own address or every lane the same one
//   bar     a pair of work-group barriers around one local load and store, work groups of 64, 128 and 256
//   mmi     the inner loop of shaders/ameva_mmq.comp (64 x 32 tile, 128 work-items, 4 x 4 outputs each, 16 packed
//           int8 dot products per 32-bit word of a 32-element unit) on data that stays in local memory:
//           EPI=0 integer sums only, EPI=1 with the float scale step of the kernel; BAR=1 adds its two barriers
//   gls     16-byte loads of a matrix walked row by row by each work-item (the weight stream of ameva_mmqb), the
//           rows stored row-major (one cache line per lane and load) or in 16-row tiles (contiguous per warp)
// MHz (default: the cap, gpu_cap_mhz) converts seconds to clocks; the clock the governor actually ran is sampled after
// every test (gpu_now_mhz, -1 where the node cannot be read) and printed, so that a throttled test is visible.
// Each figure is the median kernel time (event profiling) of REPS runs after one warm-up run.
// Build: clang -O2 -I<OpenCL headers> -o cl_hw_bench cl_hw_bench.c -lOpenCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, e_, __LINE__); exit(1); } } while (0)

static const char* kSrc =
    "#ifdef HAS_I8DOT\n"
    "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable\n"
    "#endif\n"
    "__kernel void gld(__global const T* src, __global T* out, int iters, int mask) {\n"
    "    const int g = get_global_id(0), gs = get_global_size(0);\n"
    "    T a0 = (T)(0), a1 = (T)(0), a2 = (T)(0), a3 = (T)(0);\n"
    "    int idx = g;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        a0 ^= src[idx & mask]; a1 ^= src[(idx + gs) & mask]; a2 ^= src[(idx + 2 * gs) & mask];\n"
    "        a3 ^= src[(idx + 3 * gs) & mask];\n"
    "        idx += 4 * gs;\n"
    "    }\n"
    "    out[g] = a0 ^ a1 ^ a2 ^ a3;\n"
    "}\n"
    // gls: 16-byte loads of a matrix of `rows` rows of `upr` 16-byte units, each work-item walking one row (row g %
    // rows) unit by unit, as a weight row is read by ameva_mmqb. tile=0: row-major, so the lanes of a warp touch one
    // cache line each per load; tile=1: the same bytes with the units of 16 consecutive rows next to each other, so
    // 16 lanes read 256 contiguous bytes per load. upr must be a power of two and rows a multiple of 16.
    "__kernel void gls(__global const uint4* src, __global uint4* out, int iters, int rows, int upr, int tile) {\n"
    "    const int g = get_global_id(0), r = g % rows, m = upr - 1;\n"
    "    uint4 a0 = (uint4)(0), a1 = (uint4)(0), a2 = (uint4)(0), a3 = (uint4)(0);\n"
    "    for (int i = 0; i < iters; i += 4) {\n"
    "        const int u0 = i & m, u1 = (i + 1) & m, u2 = (i + 2) & m, u3 = (i + 3) & m;\n"
    "        if (tile) {\n"
    "            const int b = (r >> 4) * upr * 16 + (r & 15);\n"
    "            a0 ^= src[b + 16 * u0]; a1 ^= src[b + 16 * u1]; a2 ^= src[b + 16 * u2]; a3 ^= src[b + 16 * u3];\n"
    "        } else {\n"
    "            const int b = r * upr;\n"
    "            a0 ^= src[b + u0]; a1 ^= src[b + u1]; a2 ^= src[b + u2]; a3 ^= src[b + u3];\n"
    "        }\n"
    "    }\n"
    "    out[g] = a0 ^ a1 ^ a2 ^ a3;\n"
    "}\n"
    "__kernel void lld(__global T* out, int iters, int lmask, int bcast) {\n"
    "    __local T s[LN];\n"
    "    const int l = get_local_id(0), ls = get_local_size(0);\n"
    "    for (int i = l; i < LN; i += ls) s[i] = (T)(i);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    T a0 = (T)(0), a1 = (T)(0), a2 = (T)(0), a3 = (T)(0);\n"
    "    int idx = bcast ? 0 : l;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        a0 ^= s[idx & lmask]; a1 ^= s[(idx + ls) & lmask]; a2 ^= s[(idx + 2 * ls) & lmask];\n"
    "        a3 ^= s[(idx + 3 * ls) & lmask];\n"
    "        idx += 4 * ls + 1;\n"
    "    }\n"
    "    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3;\n"
    "}\n"
    "__kernel void bar(__global uint* out, int iters) {\n"
    "    __local uint s[256];\n"
    "    const int l = get_local_id(0), m = get_local_size(0) - 1;\n"
    "    uint a = (uint)l;\n"
    "    s[l] = a;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        a += s[(l + 1) & m];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        s[l] = a;\n"
    "    }\n"
    "    out[get_global_id(0)] = a;\n"
    "}\n"
    "#ifdef HAS_I8DOT\n"
    "#define D4(i, W, a) i += (int4)(arm_dot(as_char4(W.x), as_char4(a)), arm_dot(as_char4(W.y), as_char4(a)), \\\n"
    "                                arm_dot(as_char4(W.z), as_char4(a)), arm_dot(as_char4(W.w), as_char4(a)))\n"
    "__kernel void mmi(__global float4* out, int iters) {\n"
    "    __local uint4 sW[128];\n"
    "    __local uint4 sA[64];\n"
    "    __local float4 sS[160];\n"
    "    const int t = get_local_id(0), tn = t & 15, tm = t >> 4;\n"
    "    sW[t] = (uint4)(t * 3 + 1, t * 5 + 2, t * 7 + 3, t * 11 + 4);\n"
    "    if (t < 64) sA[t] = (uint4)(t * 13 + 1, t * 17 + 2, t * 19 + 3, t * 23 + 4);\n"
    "    sS[t] = (float4)(t * 0.001f);\n"
    "    if (t < 32) sS[128 + t] = (float4)(t * 0.002f);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    float4 acc0 = (float4)(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;\n"
    "    int4 j0 = (int4)(0), j1 = j0, j2 = j0, j3 = j0;\n"
    "    for (int u = 0; u < iters; ++u) {\n"
    "#if BAR\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "        const float4 da = sS[128 + tm];\n"
    "#pragma unroll\n"
    "        for (int h = 0; h < 2; ++h) {\n"
    "            int4 i0 = (int4)(0), i1 = i0, i2 = i0, i3 = i0;\n"
    "#pragma unroll\n"
    "            for (int w = 4 * h; w < 4 * h + 4; ++w) {\n"
    "                const uint4 W = sW[(w * 16 + tn + u) & 127];\n"
    "                const uint4 A = sA[(w * 8 + tm) & 63];\n"
    "                D4(i0, W, A.x); D4(i1, W, A.y); D4(i2, W, A.z); D4(i3, W, A.w);\n"
    "            }\n"
    "#if EPI\n"
    "            const float4 sc = sS[h * 16 + tn], mc = sS[32 + h * 16 + tn], sa = sS[64 + h * 8 + tm];\n"
    "            acc0 += da.x * (sc * convert_float4(i0)) + mc * sa.x;\n"
    "            acc1 += da.y * (sc * convert_float4(i1)) + mc * sa.y;\n"
    "            acc2 += da.z * (sc * convert_float4(i2)) + mc * sa.z;\n"
    "            acc3 += da.w * (sc * convert_float4(i3)) + mc * sa.w;\n"
    "#else\n"
    "            j0 += i0; j1 += i1; j2 += i2; j3 += i3;\n"
    "#endif\n"
    "        }\n"
    "#if BAR\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "    }\n"
    "    const int g = get_global_id(0);\n"
    "    out[4 * g + 0] = acc0 + convert_float4(j0); out[4 * g + 1] = acc1 + convert_float4(j1);\n"
    "    out[4 * g + 2] = acc2 + convert_float4(j2); out[4 * g + 3] = acc3 + convert_float4(j3);\n"
    "}\n"
    "#endif\n"
    // shf: sub_group_shuffle of 32-bit values, four independent chains per work-item (ALU=1: the same chains with an
    // xor-add in place of the shuffle, the arithmetic baseline)
    "#ifdef HAS_SHUFFLE\n"
    "#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable\n"
    "__kernel void shf(__global uint* out, int iters) {\n"
    "    const uint l = get_sub_group_local_id(), n = get_sub_group_size();\n"
    "    uint a = get_global_id(0), b = a * 3u, c = a * 5u, d = a * 7u;\n"
    "    for (int i = 0; i < iters; ++i) {\n"
    "#if ALU\n"
    "        a = (a ^ l) + (uint)i; b = (b ^ (l + 3u)) + (uint)i; c = (c ^ (l + 5u)) + (uint)i; d = (d ^ (l + 7u)) + (uint)i;\n"
    "#else\n"
    "        a = sub_group_shuffle(a, (l + 1u) % n) + (uint)i; b = sub_group_shuffle(b, (l + 3u) % n) + (uint)i;\n"
    "        c = sub_group_shuffle(c, (l + 5u) % n) + (uint)i; d = sub_group_shuffle(d, (l + 7u) % n) + (uint)i;\n"
    "#endif\n"
    "    }\n"
    "    out[get_global_id(0)] = a ^ b ^ c ^ d;\n"
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
        CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, &ev));
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

static long read_long(const char* path) {
    FILE* f = fopen(path, "r");
    long v = -1;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
    return v;
}

// GPU clock in MHz: Exynos Mali (/sys/kernel/gpu: gpu_clock in MHz, gpu_max_clock in kHz), Qualcomm Adreno
// (/sys/class/kgsl/kgsl-3d0: gpuclk and max_gpuclk in Hz). -1 where the node is absent or not readable (Adreno gpuclk
// is not readable by an app on the S20), printed as such.
// Samsung's Snapdragon kernels also have /sys/kernel/gpu, with gpu_max_clock in MHz (S20: 587), so kgsl is asked first.
static int is_kgsl(void) { return read_long("/sys/class/kgsl/kgsl-3d0/max_gpuclk") > 0; }
static long gpu_now_mhz(void) {
    if (is_kgsl()) {
        const long v = read_long("/sys/class/kgsl/kgsl-3d0/gpuclk");
        return v > 0 ? v / 1000000 : -1;
    }
    return read_long("/sys/kernel/gpu/gpu_clock");
}
static long gpu_cap_mhz(void) {
    if (is_kgsl()) return read_long("/sys/class/kgsl/kgsl-3d0/max_gpuclk") / 1000000;
    const long v = read_long("/sys/kernel/gpu/gpu_max_clock");
    return v > 0 ? v / 1000 : -1;
}

static cl_context ctx;
static cl_device_id dev;

static cl_program build(const char* opts) {
    cl_int err;
    cl_program p = clCreateProgramWithSource(ctx, 1, &kSrc, NULL, &err); CHECK(err);
    if (clBuildProgram(p, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[16384] = "";
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "FAIL build (%s):\n%s\n", opts, log);
        exit(1);
    }
    return p;
}

static void kinfo(cl_kernel k, const char* label) {
    cl_ulong priv = 0, lmem = 0;
    size_t wg = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, NULL);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
    printf("HW_KINFO %s private=%llu local=%llu max_wg=%zu\n", label, (unsigned long long)priv,
           (unsigned long long)lmem, wg);
}

int main(int argc, char** argv) {
    long mhz = argc > 1 ? atol(argv[1]) : 0;
    const int reps = argc > 2 ? atoi(argv[2]) : 7;
    if (mhz <= 0) mhz = gpu_cap_mhz();
    if (mhz <= 0 || reps < 1) { fprintf(stderr, "usage: cl_hw_bench [MHz] [REPS >= 1]\n"); return 2; }
    cl_platform_id plat; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[256] = "", ext[8192] = "";
    cl_uint cus = 0;
    cl_ulong lmem = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL);
    const int i8 = strstr(ext, "cl_arm_integer_dot_product_accumulate_int8") != NULL;
    const int shuf = strstr(ext, "cl_khr_subgroup_shuffle ") != NULL || strstr(ext, "cl_khr_subgroup_shuffle") != NULL;
    printf("HW_DEVICE name=\"%s\" compute_units=%u clock_mhz=%ld local_mem=%llu i8dot=%d subgroups=%d shuffle=%d "
           "reps=%d\n", name, cus, mhz, (unsigned long long)lmem, i8, strstr(ext, "cl_khr_subgroups") != NULL, shuf,
           reps);
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    const cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);
    const double clk_per_ms = (double)mhz * 1e3 * cus; // core-clocks per ms over the whole GPU

    const size_t big = (size_t)64 << 20;
    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, big, NULL, &err); CHECK(err);
    {
        const cl_uint pat = 0x01020304u;
        CHECK(clEnqueueFillBuffer(q, src, &pat, sizeof(pat), 0, big, 0, NULL, NULL));
        CHECK(clFinish(q));
    }
    const size_t gitems = (size_t)cus * 1024;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gitems * 64, NULL, &err); CHECK(err);

    // HW_STREAM_SECONDS=S: only a sustained 16-byte read stream over the whole 64 MiB buffer for S seconds (gld,
    // launched back to back), the GPU side of the decode roofline; run next to cpu_bw_bench (STREAM=1) it gives what
    // the two pull together. Prints HW_STREAM GB/s and the launches.
    // HW_SYNC_REPS=N: the round trip of one tiny kernel (64 work-items) from enqueue to the host seeing it finished,
    // N times: clFinish (the driver's wait) and a spin on clGetEventInfo (no sleeping). This is the cost of every
    // hand-over between the CPU and the GPU inside a token (a CPU+GPU split of one MUL_MAT needs one per matrix or
    // per layer). Prints HW_SYNC median and 90th percentile in microseconds.
    const char* sync_env = getenv("HW_SYNC_REPS");
    if (sync_env && atoi(sync_env) > 0) {
        const int n = atoi(sync_env) > 2000 ? 2000 : atoi(sync_env);
        cl_program p = build("-DT=uint -DLN=256");
        cl_kernel k = clCreateKernel(p, "bar", &err); CHECK(err);
        const int iters = 1;
        CHECK(clSetKernelArg(k, 0, sizeof(out), &out)); CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
        const size_t g = 64, l = 64;
        static double us[2][2000];
        for (int mode = 0; mode < 2; ++mode) {
            for (int i = -20; i < n; ++i) {
                struct timespec a, b;
                clock_gettime(CLOCK_MONOTONIC, &a);
                cl_event ev;
                CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev));
                if (mode == 0) {
                    CHECK(clFinish(q));
                } else {
                    CHECK(clFlush(q));
                    cl_int st = CL_QUEUED;
                    while (st != CL_COMPLETE) {
                        CHECK(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL));
                        if (st < 0) { fprintf(stderr, "FAIL event status %d\n", st); exit(1); }
                    }
                }
                clock_gettime(CLOCK_MONOTONIC, &b);
                clReleaseEvent(ev);
                if (i >= 0) us[mode][i] = (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) * 1e-3;
            }
            for (int x = 0; x < n; ++x)  // insertion sort, n <= 2000
                for (int y = x; y > 0 && us[mode][y - 1] > us[mode][y]; --y) {
                    const double t = us[mode][y]; us[mode][y] = us[mode][y - 1]; us[mode][y - 1] = t;
                }
            printf("HW_SYNC mode=%s n=%d median_us=%.1f p90_us=%.1f min_us=%.1f\n", mode ? "spin_event" : "clFinish", n,
                   us[mode][n / 2], us[mode][n * 9 / 10], us[mode][0]);
        }
        return 0;
    }

    const char* stream_env = getenv("HW_STREAM_SECONDS");
    if (stream_env && atof(stream_env) > 0) {
        const double secs = atof(stream_env);
        cl_program p = build("-DT=uint4 -DLN=256");
        cl_kernel k = clCreateKernel(p, "gld", &err); CHECK(err);
        const int iters = 64, mask = (int)(big / 16) - 1;
        CHECK(clSetKernelArg(k, 0, sizeof(src), &src)); CHECK(clSetKernelArg(k, 1, sizeof(out), &out));
        CHECK(clSetKernelArg(k, 2, sizeof(int), &iters)); CHECK(clSetKernelArg(k, 3, sizeof(int), &mask));
        const size_t local = 64;
        const double per_launch = (double)gitems * iters * 4 * 16;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &gitems, &local, 0, NULL, NULL)); // warm-up
        CHECK(clFinish(q));
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const double t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
        double t = t0;
        long launches = 0;
        while (t - t0 < secs) {
            for (int i = 0; i < 8; ++i) CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &gitems, &local, 0, NULL, NULL));
            CHECK(clFinish(q));
            launches += 8;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            t = ts.tv_sec + ts.tv_nsec * 1e-9;
        }
        printf("HW_STREAM GB/s=%.2f seconds=%.2f launches=%ld clk_now=%ld\n", per_launch * launches / (t - t0) / 1e9,
               t - t0, launches, gpu_now_mhz());
        return 0;
    }

    static const struct { const char* t; int bytes; } widths[] = {
        { "uint", 4 }, { "uint2", 8 }, { "uint4", 16 }, { "uint8", 32 }, { "uint16", 64 } };
    char opts[256];

    // gld: L1-resident (4 KiB) at every width, then a working-set sweep at 16 bytes
    for (int w = 0; w < 5; ++w) {
        const int lnum = 4096 / widths[w].bytes;
        snprintf(opts, sizeof(opts), "-DT=%s -DLN=%d%s", widths[w].t, lnum, i8 ? " -DHAS_I8DOT" : "");
        cl_program p = build(opts);
        cl_kernel k = clCreateKernel(p, "gld", &err); CHECK(err);
        const long sets[] = { 4096, 16384, 65536, 262144, 1 << 20, 4 << 20, 16 << 20, 64 << 20 };
        const int nsets = widths[w].bytes == 16 ? 8 : 1;
        for (int s = 0; s < nsets; ++s) {
            const int mask = (int)(sets[s] / widths[w].bytes) - 1;
            const int iters = 64;
            CHECK(clSetKernelArg(k, 0, sizeof(src), &src)); CHECK(clSetKernelArg(k, 1, sizeof(out), &out));
            CHECK(clSetKernelArg(k, 2, sizeof(int), &iters)); CHECK(clSetKernelArg(k, 3, sizeof(int), &mask));
            const double ms = median_ms(q, k, gitems, 64, reps);
            const double bytes = (double)gitems * iters * 4 * widths[w].bytes;
            printf("HW gld width=%d set=%ld B/clk/core=%.2f GB/s=%.1f ms=%.3f clk_now=%ld\n", widths[w].bytes,
                   sets[s], bytes / (ms * clk_per_ms), bytes / ms / 1e6, ms, gpu_now_mhz());
        }
        // lld at the same width: own address per lane, then one address for every lane
        cl_kernel kl = clCreateKernel(p, "lld", &err); CHECK(err);
        for (int bc = 0; bc < 2; ++bc) {
            const int iters = 256, lmask = lnum - 1;
            CHECK(clSetKernelArg(kl, 0, sizeof(out), &out)); CHECK(clSetKernelArg(kl, 1, sizeof(int), &iters));
            CHECK(clSetKernelArg(kl, 2, sizeof(int), &lmask)); CHECK(clSetKernelArg(kl, 3, sizeof(int), &bc));
            const double ms = median_ms(q, kl, gitems, 128, reps);
            const double loads = (double)gitems * iters * 4;
            printf("HW lld width=%d bcast=%d B/clk/core=%.2f lane_loads/clk/core=%.2f ms=%.3f clk_now=%ld\n",
                   widths[w].bytes, bc, loads * widths[w].bytes / (ms * clk_per_ms), loads / (ms * clk_per_ms), ms,
                   gpu_now_mhz());
        }
        clReleaseKernel(kl); clReleaseKernel(k); clReleaseProgram(p);
    }

    // gls: the same bytes read row-major (one cache line per lane and load) and in 16-row tiles (contiguous), 64
    // units of 16 bytes per row (1 KiB, a q4_K row of K = 2048), working sets 4 KiB .. 16 MiB
    {
        snprintf(opts, sizeof(opts), "-DT=uint -DLN=1024");
        cl_program p = build(opts);
        cl_kernel k = clCreateKernel(p, "gls", &err); CHECK(err);
        const int upr = 64, iters = 256;
        const int rowsv[] = { 16, 64, 256, 1024, 4096, 16384 };
        for (int s = 0; s < 6; ++s) {
            for (int tile = 0; tile < 2; ++tile) {
                CHECK(clSetKernelArg(k, 0, sizeof(src), &src)); CHECK(clSetKernelArg(k, 1, sizeof(out), &out));
                CHECK(clSetKernelArg(k, 2, sizeof(int), &iters)); CHECK(clSetKernelArg(k, 3, sizeof(int), &rowsv[s]));
                CHECK(clSetKernelArg(k, 4, sizeof(int), &upr)); CHECK(clSetKernelArg(k, 5, sizeof(int), &tile));
                const double ms = median_ms(q, k, gitems, 64, reps);
                const double bytes = (double)gitems * iters * 16;
                printf("HW gls layout=%s set=%ld B/clk/core=%.2f GB/s=%.1f ms=%.3f clk_now=%ld\n",
                       tile ? "tile16" : "rowmajor", (long)rowsv[s] * upr * 16, bytes / (ms * clk_per_ms),
                       bytes / ms / 1e6, ms, gpu_now_mhz());
            }
        }
        clReleaseKernel(k); clReleaseProgram(p);
    }

    // bar
    {
        snprintf(opts, sizeof(opts), "-DT=uint -DLN=1024");
        cl_program p = build(opts);
        cl_kernel k = clCreateKernel(p, "bar", &err); CHECK(err);
        const size_t wgs[] = { 64, 128, 256 };
        for (int i = 0; i < 3; ++i) {
            const int iters = 1024;
            CHECK(clSetKernelArg(k, 0, sizeof(out), &out)); CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
            const double ms = median_ms(q, k, gitems, wgs[i], reps);
            const double wg_iters = (double)(gitems / wgs[i]) * iters;
            printf("HW bar wg=%zu core_clk_per_wg_iteration=%.1f (2 barriers + 1 local load/store) ms=%.3f clk_now=%ld\n",
                   wgs[i], ms * clk_per_ms / wg_iters, ms, gpu_now_mhz());
        }
        clReleaseKernel(k); clReleaseProgram(p);
    }

    // shf: shuffles per clock and core (lane operations), and the ALU baseline of the same chains
    if (!shuf) printf("HW_SKIP shf reason=device lacks cl_khr_subgroup_shuffle\n");
    for (int alu = 0; shuf && alu < 2; ++alu) {
        // OpenCL C 3.0: under the default 1.2 the Mali compiler does not declare the sub-group built-ins
        snprintf(opts, sizeof(opts), "-cl-std=CL3.0 -DT=uint -DLN=1024 -DHAS_SHUFFLE -DALU=%d", alu);
        cl_program p = build(opts);
        cl_kernel k = clCreateKernel(p, "shf", &err); CHECK(err);
        const int iters = 1024;
        CHECK(clSetKernelArg(k, 0, sizeof(out), &out)); CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
        const double ms = median_ms(q, k, gitems, 64, reps);
        const double ops = (double)gitems * iters * 4;
        printf("HW shf alu=%d lane_ops/clk/core=%.2f ms=%.3f clk_now=%ld\n", alu, ops / (ms * clk_per_ms), ms,
               gpu_now_mhz());
        clReleaseKernel(k); clReleaseProgram(p);
    }

    // mmi
    if (!i8) printf("HW_SKIP mmi reason=device lacks cl_arm_integer_dot_product_accumulate_int8\n");
    if (i8) {
        for (int v = 0; v < 4; ++v) {
            const int epi = v & 1, bar = v >> 1;
            snprintf(opts, sizeof(opts), "-DT=uint -DLN=1024 -DHAS_I8DOT -DEPI=%d -DBAR=%d", epi, bar);
            cl_program p = build(opts);
            cl_kernel k = clCreateKernel(p, "mmi", &err); CHECK(err);
            char label[32];
            snprintf(label, sizeof(label), "mmi_epi%d_bar%d", epi, bar);
            kinfo(k, label);
            const int iters = 256;
            CHECK(clSetKernelArg(k, 0, sizeof(out), &out)); CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
            const size_t items = (size_t)cus * 512; // 4 work groups of 128 per compute unit
            cl_mem o2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, items * 64, NULL, &err); CHECK(err);
            CHECK(clSetKernelArg(k, 0, sizeof(o2), &o2));
            const double ms = median_ms(q, k, items, 128, reps);
            const double dots = (double)items * iters * 128; // packed dot products (8 ops each)
            printf("HW mmi epi=%d bar=%d GOP/s=%.1f dot4/clk/core=%.2f ms=%.3f clk_now=%ld\n", epi, bar,
                   dots * 8 / ms / 1e6, dots / (ms * clk_per_ms), ms, gpu_now_mhz());
            clReleaseMemObject(o2); clReleaseKernel(k); clReleaseProgram(p);
        }
    }
    clReleaseMemObject(src); clReleaseMemObject(out);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}
