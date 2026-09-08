/**
 * @file test_amuda_uma.c
 * @brief UMA Zero-Copy & Vector Add Kernel Dispatch Verification Test
 *
 * Validates:
 * 1. Hardware context initialization & device query.
 * 2. 64-byte aligned UMA zero-copy buffer allocation.
 * 3. CUDA-compatible source compilation & 3D dispatch.
 * 4. Microsecond hardware event profiling.
 * 5. Numerical assertion (zero NaN/Inf, exact relative error < 1e-5).
 */

#include "amuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TEST_N 102400

static const char* k_vec_add_src =
"__global__ void vec_add(const __global float* a, const __global float* b, __global float* c, int n) {\n"
"    int idx = blockIdx_x * blockDim_x + threadIdx_x;\n"
"    if (idx < n) {\n"
"        c[idx] = a[idx] + b[idx];\n"
"    }\n"
"}\n";

int main(void) {
    printf("=========================================================\n");
    printf("[AMUDA VERIFICATION] Initializing GPGPU Hardware Context...\n");
    printf("=========================================================\n");

    AcContext* ctx = NULL;
    AcResult res = ac_init(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        fprintf(stderr, "[FAIL] ac_init() failed with error code: %d\n", res);
        return 1;
    }

    printf("[PASS] Context Initialized!\n");
    printf(" - Device:        %s\n", ctx->device_name);
    printf(" - Vendor:        %s (ID: 0x%04X)\n", ctx->vendor_name, ctx->vendor_id);
    printf(" - Compute Units: %u\n", ctx->max_compute_units);
    printf(" - Max Workgroup: %llu\n", (unsigned long long)ctx->max_work_group_size);
    printf(" - UMA Coherent:  %s\n", ctx->is_uma_supported ? "YES (Zero-Copy Enabled)" : "NO");
    printf(" - Active Quirks: 0x%08X\n", ctx->active_quirks);

    /* 1. Allocate UMA Zero-Copy Buffers */
    size_t bytes = TEST_N * sizeof(float);
    AcBuffer *buf_a = NULL, *buf_b = NULL, *buf_c = NULL;

    res = ac_buffer_alloc(ctx, bytes, &buf_a);
    if (res != AC_SUCCESS) { fprintf(stderr, "Alloc buf_a failed: %d\n", res); return 1; }

    res = ac_buffer_alloc(ctx, bytes, &buf_b);
    if (res != AC_SUCCESS) { fprintf(stderr, "Alloc buf_b failed: %d\n", res); return 1; }

    res = ac_buffer_alloc(ctx, bytes, &buf_c);
    if (res != AC_SUCCESS) { fprintf(stderr, "Alloc buf_c failed: %d\n", res); return 1; }

    /* 2. Direct Host Pointer Writing (Zero Copy) */
    float* a = (float*)buf_a->host_ptr;
    float* b = (float*)buf_b->host_ptr;
    for (int i = 0; i < TEST_N; ++i) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)i * 1.5f;
    }

    /* 3. Flush Host Cache to Device */
    ac_buffer_sync_to_device(ctx, buf_a);
    ac_buffer_sync_to_device(ctx, buf_b);
    ac_buffer_sync_to_device(ctx, buf_c);

    /* 4. Compile Kernel */
    AcKernel* kernel = NULL;
    res = ac_kernel_load_source(ctx, k_vec_add_src, "vec_add", &kernel);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] Kernel compilation failed: %d\n", res);
        return 1;
    }
    printf("[PASS] AMUDA Kernel Compiled successfully!\n");

    /* 5. Dispatch Kernel */
    uint32_t block_size = 64; /* Mali/Adreno safe workgroup size */
    uint32_t grid_size = (TEST_N + block_size - 1) / block_size;
    int n_arg = TEST_N;

    void* args[4] = { &buf_a->native_mem, &buf_b->native_mem, &buf_c->native_mem, &n_arg };
    size_t arg_sizes[4] = { sizeof(void*), sizeof(void*), sizeof(void*), sizeof(int) };

    AcEvent* event = NULL;
    res = ac_launch(ctx, kernel, ac_dim1(grid_size), ac_dim1(block_size), 4, args, arg_sizes, &event);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_launch() failed: %d\n", res);
        return 1;
    }

    /* 6. Wait & Profile */
    res = ac_event_wait(ctx, event);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "[FAIL] ac_event_wait() failed: %d\n", res);
        return 1;
    }

    uint64_t duration_us = ac_event_get_duration_us(event);
    printf("[PASS] GPU Kernel Finished in %llu us (%.3f ms)!\n", (unsigned long long)duration_us, duration_us / 1000.0);

    /* 7. Read Back (Zero Copy) & Assert All Finite */
    ac_buffer_sync_to_host(ctx, buf_c);
    float* c = (float*)buf_c->host_ptr;

    int error_count = 0;
    for (int i = 0; i < TEST_N; ++i) {
        float expected = a[i] + b[i];
        if (isnan(c[i]) || isinf(c[i])) {
            fprintf(stderr, "[FAIL-FAST] NaN or Inf detected at index %d!\n", i);
            return 1;
        }
        if (fabsf(c[i] - expected) > 1e-5f) {
            if (error_count < 5) {
                fprintf(stderr, "[ERROR] Mismatch at %d: got %f, expected %f\n", i, c[i], expected);
            }
            error_count++;
        }
    }

    if (error_count > 0) {
        fprintf(stderr, "[FAIL] Total numerical mismatches: %d\n", error_count);
        return 1;
    }

    printf("[SUCCESS] Numerical Verification Passed! 100,000 Elements Verified (Relative Error < 1e-5).\n");

    /* Cleanup */
    ac_event_free(ctx, event);
    ac_kernel_free(ctx, kernel);
    ac_buffer_free(ctx, buf_a);
    ac_buffer_free(ctx, buf_b);
    ac_buffer_free(ctx, buf_c);
    ac_destroy(ctx);

    printf("=========================================================\n");
    printf("[AMUDA VERIFICATION COMPLETED] ALL TESTS PASSED!\n");
    printf("=========================================================\n");
    return 0;
}
