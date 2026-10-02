/* SPDX-License-Identifier: Apache-2.0 */
#include "amu_sample_util.h"
#include "histogram256.acu.h"

/* hist accumulates with atomicAdd: zero it before every timed launch */
static amuStatus hist_reset_and_launch(amuFunction* fn, amuDim3 grid, amuDim3 block, amuBuffer* data, amuBuffer* hist,
                                       int n) {
    amuStatus s = amuMemset(hist, 0, 256 * sizeof(uint32_t));
    if (s != AMU_SUCCESS) return s;
    return amu_launch_histogram256(fn, grid, block, data, hist, n);
}

int main(int argc, char** argv) {
    amu_sample_begin("histogram256");
    int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
    if (n <= 0) amu_sample_die("BAD_ARGUMENT", "n must be > 0");
    const uint32_t block = 256;
    const size_t bytes = (size_t)n * sizeof(uint32_t);

    uint32_t* hd = (uint32_t*)amu_host_alloc(bytes);
    uint32_t seed = 2026u;
    for (int i = 0; i < n; ++i) {
        uint32_t r = amu_rand_u32(&seed);
        hd[i] = (r & 1u) ? (r % 32u) : r; /* skewed: heavy contention on bins 0..31 */
    }

    amuContext* ctx;
    AMU_CHECK(amuInitFromEnv(&ctx));
    amu_print_header(ctx, "histogram256");

    amuBuffer *data, *hist;
    AMU_CHECK(amuMalloc(ctx, bytes, &data));
    AMU_CHECK(amuMalloc(ctx, 256 * sizeof(uint32_t), &hist));
    AMU_CHECK(amuMemcpyHtoD(data, 0, hd, bytes));

    amuFunction* fn;
    AMU_CHECK(amu_load_histogram256(ctx, &fn));
    amuDim3 grid = amuDim(128, 1, 1); /* grid-stride loop covers n */
    amu_timing t;
    AMU_TIME_LAUNCHES(ctx, 10, t, hist_reset_and_launch(fn, grid, amuDim(block, 1, 1), data, hist, n));

    uint32_t hh[256];
    AMU_CHECK(amuMemcpyDtoH(hh, hist, 0, sizeof(hh)));

    uint32_t ref[256];
    memset(ref, 0, sizeof(ref));
    double t0 = amu_now_ms();
    for (int i = 0; i < n; ++i) ref[hd[i] & 255u]++;
    double cpu_ms = amu_now_ms() - t0;

    int wrong = 0;
    uint64_t total = 0;
    for (int b = 0; b < 256; ++b) {
        total += hh[b];
        if (hh[b] != ref[b]) {
            if (wrong < 5) fprintf(stderr, "bin %d: gpu=%u cpu=%u\n", b, hh[b], ref[b]);
            wrong++;
        }
    }
    char extra[96];
    snprintf(extra, sizeof(extra), "n=%d total=%llu", n, (unsigned long long)total);
    int pass = wrong == 0 && total == (uint64_t)n;
    amu_record_result(ctx, "histogram256", pass, (double)wrong, "wrong_bins", &t, cpu_ms, extra);

    AMU_CHECK(amuModuleUnload(fn));
    AMU_CHECK(amuFree(data)); AMU_CHECK(amuFree(hist));
    AMU_CHECK(amuDestroy(ctx));
    free(hd);
    return amu_sample_end(pass);
}
