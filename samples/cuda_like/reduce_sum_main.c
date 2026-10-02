/* SPDX-License-Identifier: Apache-2.0 */
#include "amu_sample_util.h"
#include "reduce_sum.acu.h"

/* out accumulates with atomicAdd: zero it before every timed launch */
static amuStatus reduce_reset_and_launch(amuFunction* fn, amuDim3 grid, amuDim3 block, amuBuffer* in, amuBuffer* out,
                                         int n) {
    amuStatus s = amuMemset(out, 0, sizeof(float)); /* all-zero bits == 0.0f */
    if (s != AMU_SUCCESS) return s;
    return amu_launch_reduce_sum(fn, grid, block, in, out, n);
}

int main(int argc, char** argv) {
    amu_sample_begin("reduce_sum");
    int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
    if (n <= 0) amu_sample_die("BAD_ARGUMENT", "n must be > 0");
    const uint32_t BLOCK = 256; /* must match #define BLOCK in reduce_sum.acu */
    const size_t bytes = (size_t)n * sizeof(float);

    float* hin = (float*)amu_host_alloc(bytes);
    uint32_t seed = 99u;
    for (int i = 0; i < n; ++i) hin[i] = amu_rand_unit(&seed);

    amuContext* ctx;
    AMU_CHECK(amuInitFromEnv(&ctx));
    amu_print_header(ctx, "reduce_sum");

    amuBuffer *in, *out;
    AMU_CHECK(amuMalloc(ctx, bytes, &in));
    AMU_CHECK(amuMalloc(ctx, sizeof(float), &out));
    AMU_CHECK(amuMemcpyHtoD(in, 0, hin, bytes));

    amuFunction* fn;
    AMU_CHECK(amu_load_reduce_sum(ctx, &fn));
    amuDim3 grid = amuDim((uint32_t)((n + (int)(2 * BLOCK) - 1) / (int)(2 * BLOCK)), 1, 1);
    amu_timing t;
    AMU_TIME_LAUNCHES(ctx, 10, t, reduce_reset_and_launch(fn, grid, amuDim(BLOCK, 1, 1), in, out, n));

    float gpu_sum = 0.0f;
    AMU_CHECK(amuMemcpyDtoH(&gpu_sum, out, 0, sizeof(float)));

    double t0 = amu_now_ms();
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += hin[i];
    double cpu_ms = amu_now_ms() - t0;

    double rel = fabs((double)gpu_sum - ref) / fabs(ref);
    char extra[96];
    snprintf(extra, sizeof(extra), "n=%d gpu_sum=%.6f cpu_sum=%.6f", n, gpu_sum, ref);
    int pass = rel <= 1e-4;
    amu_record_result(ctx, "reduce_sum", pass, rel, "rel_err", &t, cpu_ms, extra);

    AMU_CHECK(amuModuleUnload(fn));
    AMU_CHECK(amuFree(in)); AMU_CHECK(amuFree(out));
    AMU_CHECK(amuDestroy(ctx));
    free(hin);
    return amu_sample_end(pass);
}
