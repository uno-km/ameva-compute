/* SPDX-License-Identifier: Apache-2.0 */
#include "amu_sample_util.h"
#include "vector_add.acu.h"

int main(int argc, char** argv) {
    amu_sample_begin("vector_add");
    int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
    if (n <= 0) amu_sample_die("BAD_ARGUMENT", "n must be > 0");
    const size_t bytes = (size_t)n * sizeof(float);

    float* ha = (float*)amu_host_alloc(bytes);
    float* hb = (float*)amu_host_alloc(bytes);
    float* hc = (float*)amu_host_alloc(bytes);
    float* ref = (float*)amu_host_alloc(bytes);
    uint32_t seed = 12345u;
    for (int i = 0; i < n; ++i) {
        ha[i] = amu_rand_unit(&seed);
        hb[i] = amu_rand_unit(&seed);
    }

    amuContext* ctx;
    AMU_CHECK(amuInitFromEnv(&ctx));
    amu_print_header(ctx, "vector_add");

    amuBuffer *a, *b, *c;
    AMU_CHECK(amuMalloc(ctx, bytes, &a));
    AMU_CHECK(amuMalloc(ctx, bytes, &b));
    AMU_CHECK(amuMalloc(ctx, bytes, &c));
    AMU_CHECK(amuMemcpyHtoD(a, 0, ha, bytes));
    AMU_CHECK(amuMemcpyHtoD(b, 0, hb, bytes));
    AMU_CHECK(amuMemset(c, 0, bytes));

    amuFunction* fn;
    AMU_CHECK(amu_load_vector_add(ctx, &fn));
    const uint32_t block = 256;
    amuDim3 grid = amuDim((uint32_t)((n + (int)block - 1) / (int)block), 1, 1);
    amu_timing t;
    AMU_TIME_LAUNCHES(ctx, 10, t, amu_launch_vector_add(fn, grid, amuDim(block, 1, 1), a, b, c, n));

    double t0 = amu_now_ms();
    AMU_CHECK(amuMemcpyDtoH(hc, c, 0, bytes));
    double d2h_ms = amu_now_ms() - t0;

    t0 = amu_now_ms();
    for (int i = 0; i < n; ++i) ref[i] = ha[i] + hb[i];
    double cpu_ms = amu_now_ms() - t0;

    double max_err = 0.0;
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        double e = fabs((double)hc[i] - (double)ref[i]);
        if (e > max_err) max_err = e;
        if (e != 0.0 && bad < 5) {
            fprintf(stderr, "mismatch at %d: gpu=%.9g cpu=%.9g\n", i, hc[i], ref[i]);
            bad++;
        }
    }
    char extra[128];
    double gbps = t.gpu_med_ms > 0 ? (3.0 * (double)bytes) / (t.gpu_med_ms * 1e6) : -1.0;
    snprintf(extra, sizeof(extra), "n=%d gpu_gbps=%.2f d2h_ms=%.3f", n, gbps, d2h_ms);
    int pass = max_err == 0.0;
    amu_record_result(ctx, "vector_add", pass, max_err, "max_abs_err", &t, cpu_ms, extra);

    AMU_CHECK(amuModuleUnload(fn));
    AMU_CHECK(amuFree(a)); AMU_CHECK(amuFree(b)); AMU_CHECK(amuFree(c));
    AMU_CHECK(amuDestroy(ctx));
    free(ha); free(hb); free(hc); free(ref);
    return amu_sample_end(pass);
}
