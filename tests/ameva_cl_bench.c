// SPDX-License-Identifier: Apache-2.0
// ameva_cl_bench KERNELS.cl M N K [REPS]: the OpenCL q4_K prototype kernels for Arm Mali
// (adapters/llama/ggml-ameva/opencl/ameva_cl_q4k.cl) on one MUL_MAT shape, timed and checked.
//
//   M == 1   token generation: gemv_q4k_i8 (int8 activations, arm_dot_acc) and gemv_q4k_f32 (float weights and
//            activations), one work-item per weight row
//   M  > 1   prompt processing: gemm_q4k_i8, tiles of 64 weight rows x 32 activation rows (M a multiple of 32, N of 64)
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
    if (gemv_rows < 1 || gemv_rows > 16 || gemv_local < 1 || gemv_local > 512 || upr < 1 || upr > 8 || U % upr) {
        fprintf(stderr, "AMEVA_CL_GEMV_ROWS must be 1..16, AMEVA_CL_LOCAL 1..512, AMEVA_CL_UPR 1..8 and a divisor of K / 32\n");
        return 2;
    }
    char opts[512];
    snprintf(opts, sizeof(opts), "-cl-std=CL3.0 -DGEMV_ROWS=%d -DUPR=%d %s", gemv_rows, upr, more_opts);
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        static char log[16384];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
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
    const char* names[2] = { gemm ? "gemm_q4k_i8" : "gemv_q4k_i8", "gemv_q4k_f32" };
    for (int kk = 0; kk < (gemm ? 1 : 2); ++kk) {
        cl_kernel k = clCreateKernel(prog, names[kk], &err);
        CHECK(err);
        const int f32 = kk == 1;
        int ai = 0;
        CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bwq));
        CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &bws));
        if (f32) {
            CHECK(clSetKernelArg(k, ai++, sizeof(cl_mem), &ba));
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
        if (gemm) {
            // per staged unit: lw lo and hi (uint4) per weight row, lws dl and ml per weight row, la two int4 per
            // activation row, las d and d * sum per activation row
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 64 * 2 * 16, NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 64 * 2 * sizeof(float), NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 32 * 2 * 16, NULL));
            CHECK(clSetKernelArg(k, ai++, (size_t)upr * 32 * 2 * sizeof(float), NULL));
            const size_t gsz[2] = { (size_t)N / 64 * 128, (size_t)M / 32 }, lsz[2] = { 128, 1 };
            med = time_kernel(q, k, 2, gsz, lsz, R);
        } else {
            const size_t items = ((size_t)N + gemv_rows - 1) / gemv_rows;
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
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
