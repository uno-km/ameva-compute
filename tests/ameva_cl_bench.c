// SPDX-License-Identifier: Apache-2.0
// ameva_cl_bench KERNELS.cl M N K [REPS]: the OpenCL q4_K prototype kernels for Arm Mali
// (adapters/llama/ggml-ameva/opencl/ameva_cl_q4k.cl) on one MUL_MAT shape, timed and checked.
//
//   M == 1   token generation: gemv_q4k_i8 (int8 activations, arm_dot_acc), gemv_q4k_f32 (float weights and
//            activations) and gemv_q4k_i8_blk (activations quantized per 256-element block, integer accumulation over
//            the block), one work-item per weight row
//   M  > 1   prompt processing: gemm_q4k_i8, tiles of 64 weight rows x 32 activation rows (M a multiple of 32, N of 64),
//            and gemm_q4k_i8_direct, the same product without local memory (AMEVA_CL_DRW x AMEVA_CL_DRA rows per
//            work-item), and gemm_q4k_i8_blk, the direct kernel with the activations quantized per 256-element block
//            (quant_a8b: one d per block, as ggml's Q8_K) and integer accumulation over the block.
//            AMEVA_CL_ONLY=tile|direct|blk runs one of them.
// K is a multiple of 256 (q4_K blocks). The weights are random nibbles and random 6-bit scales in the AMEVA repacked
// q4_K layout, the activations random floats. Every kernel is checked on a sample of outputs against a
// double-precision reference that uses the same weights and, for the int8 kernels, the same int8 activations
// (d = amax / 127, round to even), so that PASS means the kernel's arithmetic and not the quantization; the distance
// of the int8 result from the unquantized product is printed for information. Times are medians of REPS runs from
// OpenCL event profiling (start to end of the command on the device).
// Build: clang -O2 -o ameva_cl_bench ameva_cl_bench.c -lOpenCL -lm
// Run:   LD_LIBRARY_PATH=$PREFIX/lib:/system/lib64:/vendor/lib64:/vendor/lib64/egl ./ameva_cl_bench ameva_cl_q4k.cl 1 8192 2048
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// m * 2^-14 for 1 <= m < 2048 as IEEE binary16 (exact: at most 11 significant bits, exponent >= -14)
static uint16_t half_of_scaled(unsigned m) {
    int e = 0;
    while ((m >> e) > 1) ++e;                       // m = 1.f * 2^e
    const unsigned frac = (m << (10 - e)) & 0x3FFu; // 10 fraction bits
    return (uint16_t)(((unsigned)(e - 14 + 15) << 10) | frac);
}
static double half_value(uint16_t h) { return ldexp((double)((h & 0x3FFu) | 0x400u), (int)((h >> 10) & 31) - 15 - 10); }

static void k4_scale_min(const uint8_t* s, int j, int* sc, int* mn) {
    if (j < 4) {
        *sc = s[j] & 63;
        *mn = s[j + 4] & 63;
    } else {
        *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

// Median kernel time (ms) of R runs. The runs are enqueued back to back after two warm-up runs and waited for once:
// a phone GPU that idles between single short commands lowers its clock, and a kernel timed alone then looks slower
// than the same kernel inside a graph.
static double time_kernel(cl_command_queue q, cl_kernel k, cl_uint dims, const size_t* gsz, const size_t* lsz, int R) {
    double* t = malloc(sizeof(double) * R);
    cl_event* ev = malloc(sizeof(cl_event) * (R + 2));
    for (int r = 0; r < R + 2; ++r) CHECK(clEnqueueNDRangeKernel(q, k, dims, NULL, gsz, lsz, 0, NULL, &ev[r]));
    CHECK(clFlush(q));
    CHECK(clWaitForEvents((cl_uint)(R + 2), ev));
    for (int r = 0; r < R + 2; ++r) {
        cl_ulong t0, t1;
        CHECK(clGetEventProfilingInfo(ev[r], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev[r], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL));
        if (r >= 2) t[r - 2] = (double)(t1 - t0) * 1e-6;
        clReleaseEvent(ev[r]);
    }
    free(ev);
    qsort(t, R, sizeof(double), cmp_double);
    const double med = t[R / 2];
    free(t);
    return med;
}

// int8 activations stored unit-major (unit u of every row side by side) back to row-major, the host's order
static int8_t* to_row_major(int8_t* um, int M, int U) {
    int8_t* rm = malloc((size_t)M * U * 32);
    for (int m = 0; m < M; ++m) {
        for (int u = 0; u < U; ++u) memcpy(rm + ((size_t)m * U + u) * 32, um + ((size_t)u * M + m) * 32, 32);
    }
    free(um);
    return rm;
}

static char* read_file(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    buf[n] = 0;
    *len = (size_t)n;
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s KERNELS.cl M N K [REPS]\n", argv[0]);
        return 2;
    }
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), R = argc > 5 ? atoi(argv[5]) : 20;
    if (M < 1 || N < 1 || K < 256 || K % 256 || R < 1 || (M > 1 && (M % 32 || N % 64))) {
        fprintf(stderr, "K must be a multiple of 256; with M > 1, M a multiple of 32 and N of 64\n");
        return 2;
    }
    const int U = K / 32, B = K / 256;
    uint8_t* wq = malloc((size_t)N * U * 16);
    uint8_t* ws = malloc((size_t)N * B * 16);
    float* a = malloc(sizeof(float) * (size_t)M * K);
    int8_t* aqh = malloc((size_t)M * K);
    float* adh = malloc(sizeof(float) * (size_t)M * U);
    float* out = malloc(sizeof(float) * (size_t)M * N);
    for (size_t i = 0; i < (size_t)N * U * 16; ++i) wq[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)N * B; ++i) {
        uint8_t* s = ws + i * 16;
        const uint16_t d = half_of_scaled(rnd() % 200 + 1), dmin = half_of_scaled(rnd() % 400 + 1);
        memcpy(s, &d, 2);
        memcpy(s + 2, &dmin, 2);
        for (int j = 4; j < 16; ++j) s[j] = (uint8_t)rnd();
    }
    for (size_t i = 0; i < (size_t)M * K; ++i) {
        a[i] = ((float)(rnd() >> 8) * (2.0f / 16777216.0f) - 1.0f) * (0.25f + (float)((i / 32) % 7));
    }
    // the kernel's activation quantization on the host, for the reference
    for (size_t g = 0; g < (size_t)M * U; ++g) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(a[g * 32 + i]));
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        adh[g] = d;
        for (int i = 0; i < 32; ++i) {
            float q = rintf(a[g * 32 + i] * id);
            q = q < -127.0f ? -127.0f : (q > 127.0f ? 127.0f : q);
            aqh[g * 32 + i] = (int8_t)q;
        }
    }

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
    size_t src_len = 0;
    const char* src = read_file(argv[1], &src_len);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    CHECK(err);
    // AMEVA_CL_GEMV_ROWS: weight rows per work-item of the token-generation kernels (default 1); AMEVA_CL_LOCAL: their
    // local size (default 64); AMEVA_CL_UPR: units per barrier round of the tile kernel (default 1); AMEVA_CL_OPTS:
    // further build options (kernel experiments)
    const int gemv_rows = getenv("AMEVA_CL_GEMV_ROWS") ? atoi(getenv("AMEVA_CL_GEMV_ROWS")) : 1;
    const int upr = getenv("AMEVA_CL_UPR") ? atoi(getenv("AMEVA_CL_UPR")) : 1;
    const size_t gemv_local = getenv("AMEVA_CL_LOCAL") ? (size_t)atoi(getenv("AMEVA_CL_LOCAL")) : 64;
    const char* more_opts = getenv("AMEVA_CL_OPTS") ? getenv("AMEVA_CL_OPTS") : "";
    // AMEVA_CL_DRW / AMEVA_CL_DRA: weight and activation rows per work-item of gemm_q4k_i8_direct (defaults 4 and 8);
    // AMEVA_CL_DLOCAL="x,y": its local size (default: the driver's choice); AMEVA_CL_ONLY=tile|direct
    const int drw = getenv("AMEVA_CL_DRW") ? atoi(getenv("AMEVA_CL_DRW")) : 4;
    const int dra = getenv("AMEVA_CL_DRA") ? atoi(getenv("AMEVA_CL_DRA")) : 8;
    const char* only = getenv("AMEVA_CL_ONLY") ? getenv("AMEVA_CL_ONLY") : "";
    // AMEVA_CL_ATRANS=1: the int8 activations unit-major (build option ATRANS)
    const int atrans = getenv("AMEVA_CL_ATRANS") ? atoi(getenv("AMEVA_CL_ATRANS")) != 0 : 0;
    size_t dlocal[2] = { 0, 0 };
    if (getenv("AMEVA_CL_DLOCAL") && sscanf(getenv("AMEVA_CL_DLOCAL"), "%zu,%zu", &dlocal[0], &dlocal[1]) != 2) {
        fprintf(stderr, "AMEVA_CL_DLOCAL must be x,y\n");
        return 2;
    }
    if (drw < 1 || drw > 8 || dra < 1 || dra > 1024 || (M > 1 && (N % drw || M % dra))) {
        fprintf(stderr, "AMEVA_CL_DRW must be 1..8 and a divisor of N, AMEVA_CL_DRA 1..1024 and a divisor of M\n");
        return 2;
    }
    if (gemv_rows < 1 || gemv_rows > 16 || gemv_local < 1 || gemv_local > 512 || upr < 1 || upr > 8 || U % upr) {
        fprintf(stderr, "AMEVA_CL_GEMV_ROWS must be 1..16, AMEVA_CL_LOCAL 1..512, AMEVA_CL_UPR 1..8 and a divisor of K / 32\n");
        return 2;
    }
    char opts[512];
    snprintf(opts, sizeof(opts), "-cl-std=CL3.0 -DGEMV_ROWS=%d -DUPR=%d -DDRW=%d -DDRA=%d %s%s", gemv_rows, upr, drw, dra,
             atrans ? "-DATRANS " : "", more_opts);
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        static char log[16384];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    // AMEVA_CL_PEAK=1: the hardware-limit kernels instead of the MUL_MAT kernels. AMEVA_CL_PEAK_ITEMS work-items
    // (default 65536, local size 64), AMEVA_CL_PEAK_REPS reps per item (default 512). The load kernels run with a
    // window of 64 vectors (1 KiB) at an offset of the item's own (stride 64 vectors: no two items share memory),
    // with the same window for every item (stride 0), and with overlapping windows of 4096 vectors (64 KiB).
    if (getenv("AMEVA_CL_PEAK") && atoi(getenv("AMEVA_CL_PEAK"))) {
        const size_t items = getenv("AMEVA_CL_PEAK_ITEMS") ? (size_t)atol(getenv("AMEVA_CL_PEAK_ITEMS")) : 65536;
        const int reps = getenv("AMEVA_CL_PEAK_REPS") ? atoi(getenv("AMEVA_CL_PEAK_REPS")) : 512;
        const size_t g = (items + 63) / 64 * 64, l = 64;
        const size_t vecs = g * 64 + 8192; // 16-byte vectors: a window per item at stride 64, and room for the large window
        int32_t* host = malloc(vecs * 16);
        for (size_t i = 0; i < vecs * 4; ++i) host[i] = (int32_t)rnd();
        cl_mem bbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vecs * 16, host, &err);
        CHECK(err);
        cl_mem bout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, g * sizeof(int), NULL, &err);
        CHECK(err);
        printf("device=%s | %s peak: items=%zu reps=%d buffer=%.1f MB\n", name, ver, g, reps, (double)vecs * 16 / 1048576.0);
        const struct {
            const char* kernel;
            const char* what;
            int loads, mask, stride;
        } runs[] = {
            { "peak_dot", "8 dot products on registers", 0, 0, 0 },
            { "peak_fma", "8 float4 fma on registers", 0, 0, 0 },
            { "peak_load", "2 loads, 1 KiB window per item", 1, 63, 64 },
            { "peak_load", "2 loads, one 1 KiB window for all", 1, 63, 0 },
            { "peak_load", "2 loads, 64 KiB windows", 1, 4095, 64 },
            { "peak_load_dot", "2 loads + 8 dots, 1 KiB window per item", 1, 63, 64 },
            { "peak_load_dot", "2 loads + 8 dots, one 1 KiB window", 1, 63, 0 },
            { "peak_load_dot", "2 loads + 8 dots, 64 KiB windows", 1, 4095, 64 },
        };
        for (size_t i = 0; i < sizeof(runs) / sizeof(runs[0]); ++i) {
            cl_kernel k = clCreateKernel(prog, runs[i].kernel, &err);
            CHECK(err);
            int ai = 0;
            if (runs[i].loads) CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bbuf));
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bout));
            CHECK(clSetKernelArg(k, ai++, sizeof(int), &reps));
            if (runs[i].loads) {
                CHECK(clSetKernelArg(k, ai++, sizeof(int), &runs[i].mask));
                CHECK(clSetKernelArg(k, ai++, sizeof(int), &runs[i].stride));
            }
            const double med = time_kernel(q, k, 1, &g, &l, R);
            const double rps = (double)g * reps / (med * 1e-3);
            printf("peak kernel=%-13s %-42s median_ms=%8.3f reps_per_s=%7.2f G  mac_equiv=%8.1f GMAC/s\n", runs[i].kernel,
                   runs[i].what, med, rps / 1e9, rps * 32 / 1e9);
            clReleaseKernel(k);
        }
        clReleaseMemObject(bbuf);
        clReleaseMemObject(bout);
        free(host);
        return 0;
    }
    cl_mem bwq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)N * U * 16, wq, &err);
    CHECK(err);
    cl_mem bws = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)N * B * 16, ws, &err);
    CHECK(err);
    cl_mem ba = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (size_t)M * K, a, &err);
    CHECK(err);
    cl_mem baq = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * K, NULL, &err);
    CHECK(err);
    cl_mem bad = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)M * U, NULL, &err);
    CHECK(err);
    cl_mem bas = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)M * U, NULL, &err);
    CHECK(err);
    cl_mem bd = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * (size_t)M * N, NULL, &err);
    CHECK(err);
    // quant_a8b: d and the 8 unit sums (two int4) per 256-element block
    cl_mem badb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)M * B, NULL, &err);
    CHECK(err);
    cl_mem babs = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * B * 32, NULL, &err);
    CHECK(err);

    printf("device=%s | %s M=%d N=%d K=%d reps=%d weights=%.1f MB gemv_rows=%d local=%zu upr=%d opts=\"%s\"\n", name,
           ver, M, N, K, R, (double)N * U * 16 / 1048576.0, gemv_rows, gemv_local, upr, more_opts);

    // activations -> int8 (once per activation row in a graph; timed on its own)
    {
        cl_kernel k = clCreateKernel(prog, "quant_a8", &err);
        CHECK(err);
        CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &ba));
        CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &baq));
        CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &bad));
        CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &bas));
        CHECK(clSetKernelArg(k, 4, sizeof(int), &M));
        CHECK(clSetKernelArg(k, 5, sizeof(int), &K));
        const size_t g = (((size_t)M * U + 63) / 64) * 64, l = 64;
        const double med = time_kernel(q, k, 1, &g, &l, R);
        int8_t* back = malloc((size_t)M * K);
        CHECK(clEnqueueReadBuffer(q, baq, CL_TRUE, 0, (size_t)M * K, back, 0, NULL, NULL));
        if (atrans) back = to_row_major(back, M, U);
        size_t diff = 0;
        for (size_t i = 0; i < (size_t)M * K; ++i) diff += back[i] != aqh[i];
        // the device's values are the ones the kernels read: the reference uses them, and a difference from the host
        // (a tie rounded the other way, a last-bit difference in 1 / d) is reported, not hidden
        memcpy(aqh, back, (size_t)M * K);
        free(back);
        printf("kernel=quant_a8 median_ms=%.4f values_differing_from_host=%zu of %zu\n", med, diff, (size_t)M * K);
        clReleaseKernel(k);
    }

    const int gemm = M > 1;
    const char* names[3] = { gemm ? "gemm_q4k_i8" : "gemv_q4k_i8", gemm ? "gemm_q4k_i8_direct" : "gemv_q4k_f32",
                             gemm ? "gemm_q4k_i8_blk" : "gemv_q4k_i8_blk" };
    for (int kk = 0; kk < 3; ++kk) {
        const int blk = kk == 2;
        const int direct = gemm && kk >= 1; // no local memory, AMEVA_CL_DRW x AMEVA_CL_DRA rows per work-item
        if (gemm && only[0] && strcmp(only, blk ? "blk" : (direct ? "direct" : "tile")) != 0) continue;
        if (blk) {
            // the activations again, one d per 256-element block, and the host's reference values with them
            cl_kernel kq = clCreateKernel(prog, "quant_a8b", &err);
            CHECK(err);
            CHECK(clSetKernelArg(kq, 0, sizeof(cl_mem), &ba));
            CHECK(clSetKernelArg(kq, 1, sizeof(cl_mem), &baq));
            CHECK(clSetKernelArg(kq, 2, sizeof(cl_mem), &badb));
            CHECK(clSetKernelArg(kq, 3, sizeof(cl_mem), &babs));
            CHECK(clSetKernelArg(kq, 4, sizeof(int), &M));
            CHECK(clSetKernelArg(kq, 5, sizeof(int), &K));
            const size_t gq = (((size_t)M * B + 63) / 64) * 64, lq = 64;
            const double medq = time_kernel(q, kq, 1, &gq, &lq, R);
            for (size_t gb = 0; gb < (size_t)M * B; ++gb) {
                float amax = 0.0f;
                for (int i = 0; i < 256; ++i) amax = fmaxf(amax, fabsf(a[gb * 256 + i]));
                const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                for (int u = 0; u < 8; ++u) adh[gb * 8 + u] = d;
                for (int i = 0; i < 256; ++i) {
                    float v = rintf(a[gb * 256 + i] * id);
                    v = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
                    aqh[gb * 256 + i] = (int8_t)v;
                }
            }
            int8_t* back = malloc((size_t)M * K);
            CHECK(clEnqueueReadBuffer(q, baq, CL_TRUE, 0, (size_t)M * K, back, 0, NULL, NULL));
            if (atrans) back = to_row_major(back, M, U);
            size_t diff = 0;
            for (size_t i = 0; i < (size_t)M * K; ++i) diff += back[i] != aqh[i];
            memcpy(aqh, back, (size_t)M * K);
            free(back);
            printf("kernel=quant_a8b median_ms=%.4f values_differing_from_host=%zu of %zu\n", medq, diff, (size_t)M * K);
            clReleaseKernel(kq);
        }
        cl_kernel k = clCreateKernel(prog, names[kk], &err);
        CHECK(err);
        const int f32 = !gemm && kk == 1;
        int ai = 0;
        CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bwq));
        CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bws));
        if (f32) {
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &ba));
        } else if (blk) {
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &baq));
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &badb));
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &babs));
        } else {
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &baq));
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bad));
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bas));
        }
        CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bd));
        if (gemm) CHECK(clSetKernelArg(k, ai++, sizeof(int), &M));
        CHECK(clSetKernelArg(k, ai++, sizeof(int), &N));
        CHECK(clSetKernelArg(k, ai++, sizeof(int), &K));
        double med;
        if (direct) {
            size_t gsz[2] = { (size_t)N / drw, (size_t)M / dra };
            if (dlocal[0] && dlocal[1]) { // padded items return at once (n0 >= N or m0 >= M)
                gsz[0] = (gsz[0] + dlocal[0] - 1) / dlocal[0] * dlocal[0];
                gsz[1] = (gsz[1] + dlocal[1] - 1) / dlocal[1] * dlocal[1];
            }
            med = time_kernel(q, k, 2, gsz, dlocal[0] && dlocal[1] ? dlocal : NULL, R);
            printf("%s: rows per item %d x %d, local %zu,%zu\n", blk ? "blk" : "direct", drw, dra, dlocal[0], dlocal[1]);
        } else if (gemm) {
            // per staged unit: lw lo and hi (uint4) per weight row, lws dl and ml per weight row, la two int4 per
            // activation row, las d and d * sum per activation row
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 64 * 2 * 16, NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 64 * 2 * sizeof(float), NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 32 * 2 * 16, NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 32 * 2 * sizeof(float), NULL));
            const size_t gsz[2] = { (size_t)N / 64 * 128, (size_t)M / 32 }, lsz[2] = { 128, 1 };
            med = time_kernel(q, k, 2, gsz, lsz, R);
        } else {
            const size_t items = blk ? (size_t)N : ((size_t)N + gemv_rows - 1) / gemv_rows; // blk: one row per item
            const size_t l = gemv_local, g = ((items + l - 1) / l) * l;
            med = time_kernel(q, k, 1, &g, &l, R);
        }
        CHECK(clEnqueueReadBuffer(q, bd, CL_TRUE, 0, sizeof(float) * (size_t)M * N, out, 0, NULL, NULL));

        // reference on a sample: rows m spread over M, every 7th weight row
        double max_rel = 0.0, max_rel_unq = 0.0;
        for (int m = 0; m < M; m += (M > 8 ? M / 8 : 1)) {
            for (int n = 0; n < N; n += 7) {
                double ref = 0.0, unq = 0.0, mag = 0.0;
                for (int u = 0; u < U; ++u) {
                    const uint8_t* s = ws + ((size_t)n * B + u / 8) * 16;
                    uint16_t hd, hm;
                    memcpy(&hd, s, 2);
                    memcpy(&hm, s + 2, 2);
                    int sc, mn;
                    k4_scale_min(s + 4, u & 7, &sc, &mn);
                    const double dl = half_value(hd) * sc, ml = half_value(hm) * mn;
                    const uint8_t* qb = wq + ((size_t)n * U + u) * 16;
                    const size_t g = (size_t)m * U + u;
                    long long dot = 0, sum = 0;
                    for (int i = 0; i < 32; ++i) {
                        const int v = i < 16 ? (qb[i] & 15) : (qb[i - 16] >> 4);
                        const double w = dl * v - ml;
                        dot += (long long)v * aqh[g * 32 + i];
                        sum += aqh[g * 32 + i];
                        unq += w * (double)a[g * 32 + i];
                        mag += fabs(w * (double)a[g * 32 + i]);
                    }
                    ref += (double)adh[g] * (dl * (double)dot - ml * (double)sum);
                }
                const double o = out[(size_t)m * N + n];
                const double target = f32 ? unq : ref;
                max_rel = fmax(max_rel, fabs(o - target) / (mag > 0 ? mag : 1.0));
                max_rel_unq = fmax(max_rel_unq, fabs(o - unq) / (mag > 0 ? mag : 1.0));
            }
        }
        const double mac = (double)M * N * K;
        printf("kernel=%s median_ms=%.4f gmac_per_s=%.2f weights_gb_per_s=%.2f max_rel_err=%.2e (vs unquantized %.2e) %s\n",
               names[kk], med, mac / (med * 1e-3) / 1e9, (double)N * U * 16 / (med * 1e-3) / 1e9, max_rel, max_rel_unq,
               max_rel < 1e-4 ? "PASS" : "FAIL");
        clReleaseKernel(k);
    }
    clReleaseMemObject(bwq);
    clReleaseMemObject(bws);
    clReleaseMemObject(ba);
    clReleaseMemObject(baq);
    clReleaseMemObject(bad);
    clReleaseMemObject(bas);
    clReleaseMemObject(bd);
    clReleaseMemObject(badb);
    clReleaseMemObject(babs);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
