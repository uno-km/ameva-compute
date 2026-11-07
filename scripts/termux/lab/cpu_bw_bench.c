/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// cpu_bw_bench [MiB] [SECONDS] [THREADS,...]: DRAM read bandwidth of the CPU cores of a phone, the CPU side of the
// decode roofline (ameva-foundation N-20261107-decode-roofline): token generation reads every weight once per token,
// so its ceiling is (bandwidth the processor actually pulls) / (bytes per token).
//   - a buffer of MiB (default 512, far above the system cache) is split among the threads; each thread reads its
//     slice again and again with 64-byte NEON loads for SECONDS (default 2) and the bytes read are counted
//   - threads are pinned to the allowed cores in order of decreasing cpuinfo_max_freq (the big cores first), one
//     thread per core; the allowed set (cpuset of this process) and the cores used are printed
//   - THREADS (default 1,2,4,all allowed) is a comma-separated list of thread counts
// STREAM=1: one count only (the first of THREADS), for running next to the GPU's stream (cl_hw_bench, HW_STREAM_SECONDS)
// to measure what both pull together. Prints CPU_BW lines.
// Build: clang -O2 -o cpu_bw_bench cpu_bw_bench.c -lpthread
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static long read_long(const char* path) {
    FILE* f = fopen(path, "r");
    long v = -1;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
    return v;
}

typedef struct {
    const uint8_t* base;
    size_t bytes;
    int core;
    double seconds;
    atomic_int* go;
    double out_bytes, out_secs;
    uint64_t sink;
} Job;

static void* run(void* p) {
    Job* j = (Job*)p;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(j->core, &s);
    // a core in the affinity mask can go offline (the S20 hot-plugs its big cores): the thread then runs unpinned and
    // says so (CPU_BW_UNPINNED); it is not dropped
    if (sched_setaffinity(0, sizeof(s), &s) != 0) printf("CPU_BW_UNPINNED core=%d (offline or refused)\n", j->core);
    while (!atomic_load(j->go)) {}
    uint64x2_t acc = vdupq_n_u64(0);
    double bytes = 0;
    const double t0 = now_s();
    double t = t0;
    while (t - t0 < j->seconds) {
        const uint64_t* q = (const uint64_t*)j->base;
        const size_t n = j->bytes / 64;
        for (size_t i = 0; i < n; ++i, q += 8) {
            const uint64x2x4_t v = vld1q_u64_x4(q);
            acc = veorq_u64(acc, veorq_u64(veorq_u64(v.val[0], v.val[1]), veorq_u64(v.val[2], v.val[3])));
        }
        bytes += (double)n * 64;
        t = now_s();
    }
    j->out_bytes = bytes;
    j->out_secs = t - t0;
    j->sink = vgetq_lane_u64(acc, 0) ^ vgetq_lane_u64(acc, 1);
    return NULL;
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? (size_t)atol(argv[1]) : 512;
    const double secs = argc > 2 ? atof(argv[2]) : 2.0;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { fprintf(stderr, "FAIL sched_getaffinity\n"); return 1; }
    int cores[64], nc = 0;
    long freq[64];
    for (int c = 0; c < 64; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        cores[nc] = c;
        freq[nc++] = read_long(path);
    }
    for (int a = 0; a < nc; ++a)  // big cores first
        for (int b = a + 1; b < nc; ++b)
            if (freq[b] > freq[a]) {
                long tf = freq[a]; freq[a] = freq[b]; freq[b] = tf;
                int tc = cores[a]; cores[a] = cores[b]; cores[b] = tc;
            }
    int counts[16], ncount = 0;
    if (argc > 3) {
        char* s = strdup(argv[3]);
        char* save = NULL;
        for (char* tok = strtok_r(s, ",", &save); tok && ncount < 16; tok = strtok_r(NULL, ",", &save)) counts[ncount++] = atoi(tok);
        free(s);
    } else {
        const int def[] = { 1, 2, 4, nc };
        for (int i = 0; i < 4; ++i) if (def[i] <= nc && (i == 0 || def[i] != counts[ncount - 1])) counts[ncount++] = def[i];
    }
    if (getenv("STREAM")) ncount = 1;
    printf("CPU_BW_SETUP allowed=");
    for (int i = 0; i < nc; ++i) printf("%d(%ldMHz)%s", cores[i], freq[i] / 1000, i + 1 < nc ? "," : "");
    printf(" buffer_mib=%zu seconds=%.1f\n", mib, secs);
    uint8_t* buf = NULL; // posix_memalign: aligned_alloc is missing from the Android 13 libc of the S20
    if (posix_memalign((void**)&buf, 4096, mib << 20) != 0) buf = NULL;
    if (!buf) { fprintf(stderr, "FAIL alloc %zu MiB\n", mib); return 1; }
    for (size_t i = 0; i < (mib << 20); i += 64) buf[i] = (uint8_t)i;  // touch every line
    for (int k = 0; k < ncount; ++k) {
        // STREAM: at most the cores this process may run on now (the S20 took its prime core out of the set)
        const int nt = getenv("STREAM") && counts[k] > nc ? nc : counts[k];
        if (nt < 1 || nt > nc) { printf("CPU_BW_SKIP threads=%d reason=only %d allowed cores\n", nt, nc); continue; }
        Job jobs[64];
        pthread_t th[64];
        atomic_int go = 0;
        const size_t slice = ((mib << 20) / nt) & ~(size_t)63;
        for (int i = 0; i < nt; ++i) {
            jobs[i] = (Job){ buf + (size_t)i * slice, slice, cores[i], secs, &go, 0, 0, 0 };
            pthread_create(&th[i], NULL, run, &jobs[i]);
        }
        atomic_store(&go, 1);
        double total = 0, tmax = 0;
        uint64_t sink = 0;
        for (int i = 0; i < nt; ++i) {
            pthread_join(th[i], NULL);
            total += jobs[i].out_bytes / jobs[i].out_secs;
            if (jobs[i].out_secs > tmax) tmax = jobs[i].out_secs;
            sink ^= jobs[i].sink;
        }
        printf("CPU_BW threads=%d cores=", nt);
        for (int i = 0; i < nt; ++i) printf("%d%s", cores[i], i + 1 < nt ? "," : "");
        printf(" GB/s=%.2f secs=%.2f sink=%llu\n", total / 1e9, tmax, (unsigned long long)(sink & 1));
        fflush(stdout);
    }
    free(buf);
    return 0;
}
