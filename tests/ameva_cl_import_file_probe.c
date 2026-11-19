/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
 */
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
// ameva_cl_import_file_probe: can an OpenCL GPU use the pages of an mmap'ed file as a buffer without a copy?
// Research question RQ-L1 of docs/design/resident-runtime-and-load-path.md (load path without the weight copy).
//
// For each memory source (anonymous mmap, file mmap MAP_PRIVATE read-write, file mmap MAP_SHARED read-only) and
// each way of making an OpenCL buffer from it
//   copy     clCreateBuffer(COPY_HOST_PTR)                the copy the product does today
//   usehost  clCreateBuffer(USE_HOST_PTR)                 standard zero-copy request (the driver may still copy)
//   armimp   clImportMemoryARM(CL_IMPORT_TYPE_HOST_ARM)   Mali host import, no allocation by contract
// the probe prints: the creation status and time, whether a kernel reading the buffer returns the right values for
// every word, the best kernel time, and whether the kernel sees a host write made after the buffer was created
// (shared = yes means the GPU reads the host pages themselves, i.e. no private copy exists).
// Pages are touched before the buffer is created, so the creation times measure the import or copy, not page faults;
// the touch time is printed separately.
// A failed creation is printed with its OpenCL status; no case is skipped silently and none falls back to another.
// Prints ALL_CASES_OK or SOME_CASES_FAILED; exit status 0 only for ALL_CASES_OK.
// Build (Termux):  clang -O2 -I include -o cl_import_file_probe tests/ameva_cl_import_file_probe.c -lOpenCL
// Run: LD_LIBRARY_PATH=$PREFIX/lib:/system/lib64:/vendor/lib64:/vendor/lib64/egl ./cl_import_file_probe <file> [MB=64]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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

static const char* kSrc =
    "__kernel void mix_words(__global const uint* src, __global uint* dst, uint n) {\n"
    "    uint i = get_global_id(0);\n"
    "    if (i < n) dst[i] = src[i] * 2654435761u + i;\n"
    "}\n";

enum { SRC_ANON, SRC_ANON_PREAD, SRC_FILE_PRIV, SRC_FILE_RO, N_SRC };
enum { M_COPY, M_USEHOST, M_ARMIMP, N_METH };
static const char* kSrcName[N_SRC] = { "anon", "anon_pread", "file_priv", "file_ro" };
static const char* kMethName[N_METH] = { "copy", "usehost", "armimp" };

typedef struct {
    cl_context ctx;
    cl_command_queue q;
    cl_kernel k;
    cl_mem out;
    clImportMemoryARM_fn import;
    uint32_t* outHost;
    size_t bytes;
} Env;

// One kernel run over `in`; returns 0 and the best time when every output word matches the host's view of `host`.
static int run_kernel(Env* env, cl_mem in, const uint32_t* host, double* best_ms, size_t* bad_out) {
    const cl_uint n = (cl_uint)(env->bytes / 4);
    clSetKernelArg(env->k, 0, sizeof(cl_mem), &in);
    clSetKernelArg(env->k, 1, sizeof(cl_mem), &env->out);
    clSetKernelArg(env->k, 2, sizeof(cl_uint), &n);
    const size_t l = 64, g = ((size_t)n + l - 1) / l * l;
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now_ms();
        cl_int e = clEnqueueNDRangeKernel(env->q, env->k, 1, NULL, &g, &l, 0, NULL, NULL);
        if (e == CL_SUCCESS) e = clFinish(env->q);
        if (e != CL_SUCCESS) return (int)e;
        const double t = now_ms() - t0;
        if (t < best) best = t;
    }
    cl_int e = clEnqueueReadBuffer(env->q, env->out, CL_TRUE, 0, env->bytes, env->outHost, 0, NULL, NULL);
    if (e != CL_SUCCESS) return (int)e;
    size_t bad = 0;
    for (cl_uint i = 0; i < n; ++i)
        if (env->outHost[i] != host[i] * 2654435761u + i) ++bad;
    *best_ms = best;
    *bad_out = bad;
    return 0;
}

// Resident accounting of the mapping that contains addr (Private_Dirty after an import = the driver made private copies).
static void smaps_report(const char* tag, const char* when, const void* addr) {
    FILE* f = fopen("/proc/self/smaps", "r");
    if (!f) return;
    char line[512];
    int in = 0;
    const uintptr_t a = (uintptr_t)addr;
    long rss = 0, pd = 0, pc = 0, sc = 0, an = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long lo, hi;
        if (sscanf(line, "%llx-%llx ", &lo, &hi) == 2) { in = (a >= lo && a < hi); continue; }
        if (!in) continue;
        long v;
        if (sscanf(line, "Rss: %ld kB", &v) == 1) rss = v;
        else if (sscanf(line, "Private_Dirty: %ld kB", &v) == 1) pd = v;
        else if (sscanf(line, "Private_Clean: %ld kB", &v) == 1) pc = v;
        else if (sscanf(line, "Shared_Clean: %ld kB", &v) == 1) sc = v;
        else if (sscanf(line, "Anonymous: %ld kB", &v) == 1) an = v;
    }
    fclose(f);
    printf("  smaps %s %s: Rss=%ld kB Private_Dirty=%ld kB Private_Clean=%ld kB Shared_Clean=%ld kB Anonymous=%ld kB\n",
           tag, when, rss, pd, pc, sc, an);
}

static int run_case(Env* env, int src, int meth, void* host, int writable) {
    uint32_t* w = (uint32_t*)host;
    const cl_mem_flags ro = CL_MEM_READ_ONLY;
    cl_int e = CL_SUCCESS;
    cl_mem m = NULL;
    const double t0 = now_ms();
    if (meth == M_COPY) {
        m = clCreateBuffer(env->ctx, ro | CL_MEM_COPY_HOST_PTR, env->bytes, host, &e);
    } else if (meth == M_USEHOST) {
        m = clCreateBuffer(env->ctx, ro | CL_MEM_USE_HOST_PTR, env->bytes, host, &e);
    } else {
        if (!env->import) {
            printf("CASE src=%s method=%s create=FAIL status=clImportMemoryARM_not_exported\n", kSrcName[src], kMethName[meth]);
            return 1;
        }
        const probe_import_prop props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_HOST_ARM, 0 };
        m = env->import(env->ctx, ro, props, host, env->bytes, &e);
    }
    const double t_create = now_ms() - t0;
    if (e != CL_SUCCESS || !m) {
        printf("CASE src=%s method=%s create=FAIL status=%d create_ms=%.3f\n", kSrcName[src], kMethName[meth], (int)e, t_create);
        return 1;
    }
    smaps_report(kSrcName[src], kMethName[meth], host);
    double best = 0;
    size_t bad = 0;
    int rc = run_kernel(env, m, w, &best, &bad);
    if (rc) {
        printf("CASE src=%s method=%s create=ok create_ms=%.3f kernel=FAIL status=%d\n", kSrcName[src], kMethName[meth], t_create, rc);
        clReleaseMemObject(m);
        return 1;
    }
    // Does the kernel see a host write made after creation? Only for writable memory.
    const char* shared = "n/a(read-only)";
    if (writable) {
        const uint32_t old = w[0];
        w[0] = old ^ 0xA5A5A5A5u;
        double b2;
        size_t bad2 = 0;
        // out[0] is computed from what the GPU reads; compare with the old and the new host word
        rc = run_kernel(env, m, w, &b2, &bad2);
        if (rc) {
            shared = "kernel_error";
        } else {
            // bad2 counts words that differ from the NEW host view; 0 means the GPU read the new word (shared)
            shared = bad2 == 0 ? "yes" : "no";
        }
        w[0] = old;
    }
    printf("CASE src=%s method=%s create=ok create_ms=%.3f kernel_best_ms=%.3f correct=%s bad_words=%zu shared_after_host_write=%s\n",
           kSrcName[src], kMethName[meth], t_create, best, bad == 0 ? "yes" : "NO", bad, shared);
    clReleaseMemObject(m);
    return bad == 0 ? 0 : 1;
}

static void print_ext_hits(cl_device_id dev) {
    size_t n = 0;
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, NULL, &n);
    char* s = (char*)calloc(n + 1, 1);
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, n, s, NULL);
    printf("extensions of interest:");
    char* save = NULL;
    for (char* t = strtok_r(s, " ", &save); t; t = strtok_r(NULL, " ", &save))
        if (strstr(t, "import") || strstr(t, "host_ptr") || strstr(t, "ion") || strstr(t, "dma") || strstr(t, "unified"))
            printf(" %s", t);
    printf("\n");
    free(s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [MB=64]\n", argv[0]);
        return 2;
    }
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t bytes = (argc > 2 ? (size_t)atol(argv[2]) : 64) << 20;
    bytes = bytes / page * page;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < bytes) {
        fprintf(stderr, "file smaller than %zu bytes (size %lld)\n", bytes, (long long)st.st_size);
        return 2;
    }

    cl_platform_id plat;
    cl_device_id dev;
    cl_int e = clGetPlatformIDs(1, &plat, NULL);
    if (e != CL_SUCCESS) { printf("clGetPlatformIDs failed %d\n", (int)e); return 2; }
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (e != CL_SUCCESS) { printf("clGetDeviceIDs failed %d\n", (int)e); return 2; }
    char name[256] = { 0 };
    cl_uint cl_line = 0, unified = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_line), &cl_line, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
    printf("device: %s, cacheline %u, host_unified_memory %u, probe size %zu MB, page %zu\n", name, cl_line, unified, bytes >> 20, page);
    print_ext_hits(dev);

    Env env;
    memset(&env, 0, sizeof(env));
    env.bytes = bytes;
    env.ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    if (e != CL_SUCCESS) { printf("clCreateContext failed %d\n", (int)e); return 2; }
    env.q = clCreateCommandQueueWithProperties(env.ctx, dev, NULL, &e);
    if (e != CL_SUCCESS) { printf("clCreateCommandQueue failed %d\n", (int)e); return 2; }
    env.import = (clImportMemoryARM_fn)clGetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM");
    cl_program prog = clCreateProgramWithSource(env.ctx, 1, &kSrc, NULL, &e);
    if (e == CL_SUCCESS) e = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
    if (e != CL_SUCCESS) {
        printf("program build failed %d\n", (int)e);
        size_t ln = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ln);
        char* log = (char*)calloc(ln + 1, 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ln, log, NULL);
        printf("build log:\n%s\n", log);
        return 2;
    }
    env.k = clCreateKernel(prog, "mix_words", &e);
    env.out = clCreateBuffer(env.ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &e);
    if (e != CL_SUCCESS) { printf("output buffer failed %d\n", (int)e); return 2; }
    env.outHost = (uint32_t*)malloc(bytes);

    int failures = 0;
    for (int src = 0; src < N_SRC; ++src) {
        void* host;
        int writable = 1;
        void* region = NULL;
        if (src == SRC_ANON || src == SRC_ANON_PREAD) {
            // guard pages on both sides keep the mapping a VMA of its own, so the smaps numbers are this buffer's
            region = mmap(NULL, bytes + 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            host = MAP_FAILED;
            if (region != MAP_FAILED) {
                mprotect(region, page, PROT_NONE);
                mprotect((char*)region + page + bytes, page, PROT_NONE);
                host = (char*)region + page;
                if (src == SRC_ANON) {
                    uint32_t* w = (uint32_t*)host;
                    for (size_t i = 0; i < bytes / 4; ++i) w[i] = (uint32_t)(i * 2246822519u + 12345u);
                } else {
                    // the product load path candidate: the file's bytes read into an anonymous buffer
                    const double tr = now_ms();
                    size_t got = 0;
                    while (got < bytes) {
                        ssize_t r = pread(fd, (char*)host + got, bytes - got, (off_t)got);
                        if (r <= 0) { perror("pread"); break; }
                        got += (size_t)r;
                    }
                    printf("SOURCE anon_pread: read %zu MB from the file in %.1f ms\n", got >> 20, now_ms() - tr);
                    if (got != bytes) { munmap(region, bytes + 2 * page); host = MAP_FAILED; }
                }
            }
        } else if (src == SRC_FILE_PRIV) {
            host = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        } else {
            host = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
            writable = 0;
        }
        if (host == MAP_FAILED) {
            perror("mmap");
            ++failures;
            continue;
        }
        // touch every page (page cache residency); sum keeps the loads alive
        const double tt = now_ms();
        volatile uint32_t sink = 0;
        for (size_t o = 0; o < bytes; o += page) sink += ((volatile uint32_t*)host)[o / 4];
        printf("SOURCE %s: mapped at %p (page offset %lu), touch %.1f ms\n", kSrcName[src], host,
               (unsigned long)((uintptr_t)host & (page - 1)), now_ms() - tt);
        (void)sink;
        smaps_report(kSrcName[src], "before", host);
        for (int meth = 0; meth < N_METH; ++meth) failures += run_case(&env, src, meth, host, writable);
        if (region) munmap(region, bytes + 2 * page); else munmap(host, bytes);
    }
    printf("%s (%d failed case(s))\n", failures ? "SOME_CASES_FAILED" : "ALL_CASES_OK", failures);
    return failures ? 1 : 0;
}
