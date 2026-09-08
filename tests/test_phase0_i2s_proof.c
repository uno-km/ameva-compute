/**
 * @file test_phase0_i2s_proof.c
 * @brief Phase 0: BitNet I2_S CPU Format Local Algorithm Verification (로컬 알고리즘 검증)
 * 
 * Dependencies: Standard C99 library only (NO OpenCL, NO Vulkan, NO GPU driver).
 * 
 * Verification Gates:
 *   1. Canonical Decoder (ggml-cpu/quants.c dequantize_row_i2_s baseline)
 *   2. Independent Decoder (Direct bitfield extraction & switch-case)
 *   3. Handcrafted Golden Vector bit-exact verification
 *   4. Boundary conditions: N = 1, 127, 128, 129 with exact value and canary bounds check
 *   5. +32B Tensor Scale Trailer check (float32 scale extraction, trailer bytes [4..31] not assumed zero)
 *   6. Truncated buffer fail-fast rejection & output buffer immutability check
 *   7. Arithmetic overflow fail-fast rejection without n + 127 wrap
 * 
 * Note: Staged for next commit: Upstream BitNet quantizer fixture round-trip verification.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define CANARY_GUARD 0xDEADBEEF
#define SENTINEL_VAL -9999.0f

/* ============================================================================
 * Safe Ceiling Division Helper (Prevents n + 127 signed overflow)
 * ============================================================================ */
static inline int64_t safe_ceil_div128(int64_t n) {
    if (n <= 0) return 0;
    /* n / 128 + ((n % 128) != 0 ? 1 : 0) avoids any n + 127 addition overflow */
    return (n / 128) + ((n % 128 != 0) ? 1 : 0);
}

/* ============================================================================
 * Decoder 1: Canonical Reference (from BitNet ggml-cpu/quants.c)
 * ============================================================================ */
static const float k_map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };

static int dequantize_i2s_canonical(const uint8_t * in_bytes, size_t in_size,
                                    float scale, int64_t n,
                                    float * out_weights, size_t out_capacity) {
    if (!in_bytes || !out_weights) return -1;
    if (n < 0 || (uint64_t)n > (UINT64_MAX / 128ULL)) return -2;
    if (out_capacity < (size_t)n) return -2;
    if (n == 0) return 0;

    int64_t num_blocks = safe_ceil_div128(n);
    if ((uint64_t)num_blocks > (SIZE_MAX / 32ULL)) return -2;
    size_t required_bytes = (size_t)num_blocks * 32ULL;
    if (in_size < required_bytes) return -3; /* Truncated buffer */

    int64_t done = 0;
    while (done < n) {
        int64_t cols0 = 32 < (n - done - 0 * 32) ? 32 : (n - done - 0 * 32);
        int64_t cols1 = 32 < (n - done - 1 * 32) ? 32 : (n - done - 1 * 32);
        int64_t cols2 = 32 < (n - done - 2 * 32) ? 32 : (n - done - 2 * 32);
        int64_t cols3 = 32 < (n - done - 3 * 32) ? 32 : (n - done - 3 * 32);

        for (int gp = 0; gp < 32; gp++) {
            uint8_t byte = in_bytes[(done / 4) + gp];
            uint8_t c0 = (byte >> 6) & 0x03;
            uint8_t c1 = (byte >> 4) & 0x03;
            uint8_t c2 = (byte >> 2) & 0x03;
            uint8_t c3 = (byte >> 0) & 0x03;

            if (gp < cols0) out_weights[done + 0 * 32 + gp] = scale * k_map2bit[c0];
            if (gp < cols1) out_weights[done + 1 * 32 + gp] = scale * k_map2bit[c1];
            if (gp < cols2) out_weights[done + 2 * 32 + gp] = scale * k_map2bit[c2];
            if (gp < cols3) out_weights[done + 3 * 32 + gp] = scale * k_map2bit[c3];
        }
        done += 128;
    }
    return 0;
}

/* ============================================================================
 * Decoder 2: Independent Implementation (Nested Block/Lane Loop + Switch)
 * ============================================================================ */
static int dequantize_i2s_independent(const uint8_t * in_bytes, size_t in_size,
                                      float scale, int64_t n,
                                      float * out_weights, size_t out_capacity) {
    if (!in_bytes || !out_weights) return -1;
    if (n < 0 || (uint64_t)n > (UINT64_MAX / 128ULL)) return -2;
    if (out_capacity < (size_t)n) return -2;
    if (n == 0) return 0;

    int64_t num_blocks = safe_ceil_div128(n);
    if ((uint64_t)num_blocks > (SIZE_MAX / 32ULL)) return -2;
    size_t required_bytes = (size_t)num_blocks * 32ULL;
    if (in_size < required_bytes) return -3; /* Truncated buffer */

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        const uint8_t * blk_ptr = in_bytes + (blk * 32);
        int64_t blk_base = blk * 128;

        for (int lane = 0; lane < 32; lane++) {
            uint8_t b = blk_ptr[lane];

            uint8_t q[4];
            q[0] = (b >> 6) & 0x03;
            q[1] = (b >> 4) & 0x03;
            q[2] = (b >> 2) & 0x03;
            q[3] = (b >> 0) & 0x03;

            for (int sub = 0; sub < 4; sub++) {
                int64_t global_idx = blk_base + (sub * 32) + lane;
                if (global_idx < n) {
                    float ternary = 0.0f;
                    switch (q[sub]) {
                        case 0: ternary = -1.0f; break;
                        case 1: ternary =  0.0f; break;
                        case 2: ternary = +1.0f; break;
                        case 3: ternary =  0.0f; break; /* Reserved */
                    }
                    out_weights[global_idx] = ternary * scale;
                }
            }
        }
    }
    return 0;
}

/* ============================================================================
 * Helper: Safe Allocation with Canary Guard (Asserts Removed)
 * ============================================================================ */
typedef struct {
    float * data;
    size_t count;
    void * raw_mem;
} CanaryBuffer;

static int create_canary_buffer(size_t count, CanaryBuffer * out_cb) {
    if (!out_cb) return -1;
    out_cb->count = count;
    size_t total = sizeof(uint32_t) + count * sizeof(float) + sizeof(uint32_t);
    out_cb->raw_mem = malloc(total);
    if (!out_cb->raw_mem) return -2;

    uint8_t * p = (uint8_t *)out_cb->raw_mem;
    uint32_t * head = (uint32_t *)p;
    *head = CANARY_GUARD;

    out_cb->data = (float *)(p + sizeof(uint32_t));
    for (size_t i = 0; i < count; i++) {
        out_cb->data[i] = SENTINEL_VAL;
    }

    uint32_t * tail = (uint32_t *)(p + sizeof(uint32_t) + count * sizeof(float));
    *tail = CANARY_GUARD;
    return 0;
}

static int verify_canary_buffer(const CanaryBuffer * cb) {
    if (!cb || !cb->raw_mem) return -1;
    uint8_t * p = (uint8_t *)cb->raw_mem;
    uint32_t * head = (uint32_t *)p;
    uint32_t * tail = (uint32_t *)(p + sizeof(uint32_t) + cb->count * sizeof(float));
    if (*head != CANARY_GUARD) return -2; /* Head corrupted (Underflow) */
    if (*tail != CANARY_GUARD) return -3; /* Tail corrupted (Overflow) */
    return 0;
}

static void free_canary_buffer(CanaryBuffer * cb) {
    if (cb && cb->raw_mem) {
        free(cb->raw_mem);
        cb->raw_mem = NULL;
        cb->data = NULL;
        cb->count = 0;
    }
}

/* ============================================================================
 * Helper: Calculate Mathematical Expected Weight for Index i
 * ============================================================================ */
static float get_expected_weight(const uint8_t * in_bytes, int64_t i, float scale) {
    int64_t blk = i / 128;
    int64_t rem = i % 128;
    int64_t sub = rem / 32;
    int64_t lane = rem % 32;

    uint8_t byte_val = in_bytes[blk * 32 + lane];
    int shift = 6 - (int)(sub * 2);
    uint8_t q = (byte_val >> shift) & 0x03;

    return scale * k_map2bit[q];
}

/* ============================================================================
 * Main Verification Suite
 * ============================================================================ */
int main(void) {
    printf("====================================================================\n");
    printf("Phase 0: BitNet I2_S CPU Format Local Algorithm Verification\n");
    printf("====================================================================\n");

    int total_failures = 0;

    /* ------------------------------------------------------------------------
     * Test 1: Handcrafted Golden Vector (32 Bytes -> 128 Weights)
     * ------------------------------------------------------------------------ */
    printf("\n[Test 1] Handcrafted Golden Byte Vector Verification...\n");
    {
        uint8_t golden_bytes[32];
        memset(golden_bytes, 0, sizeof(golden_bytes));
        golden_bytes[0] = 0x9A; /* 10 01 10 10 -> +1,  0, +1, +1 */
        golden_bytes[1] = 0x14; /* 00 01 01 00 -> -1,  0,  0, -1 */
        golden_bytes[2] = 0x00; /* 00 00 00 00 -> -1, -1, -1, -1 */
        golden_bytes[3] = 0x55; /* 01 01 01 01 ->  0,  0,  0,  0 */
        golden_bytes[4] = 0xAA; /* 10 10 10 10 -> +1, +1, +1, +1 */
        golden_bytes[5] = 0xFF; /* 11 11 11 11 ->  0,  0,  0,  0 (Reserved) */
        for (int i = 6; i < 32; i++) {
            golden_bytes[i] = (uint8_t)((i * 37 + 13) & 0xFF);
        }

        float scale = 2.5f;
        CanaryBuffer cb1, cb2;
        if (create_canary_buffer(128, &cb1) != 0 || create_canary_buffer(128, &cb2) != 0) {
            printf("  -> FAIL: Memory allocation failed.\n");
            total_failures++;
            goto test1_cleanup;
        }

        int res1 = dequantize_i2s_canonical(golden_bytes, 32, scale, 128, cb1.data, 128);
        int res2 = dequantize_i2s_independent(golden_bytes, 32, scale, 128, cb2.data, 128);

        if (res1 != 0 || res2 != 0) {
            printf("  -> FAIL: Dequantization returned error (res1=%d, res2=%d)\n", res1, res2);
            total_failures++;
            goto test1_cleanup;
        }

        if (verify_canary_buffer(&cb1) != 0 || verify_canary_buffer(&cb2) != 0) {
            printf("  -> FAIL: Canary guard corruption detected.\n");
            total_failures++;
            goto test1_cleanup;
        }

        /* Verify individual spot-check values */
        int t1_mismatch = 0;
        if (cb1.data[0] != +2.5f || cb1.data[32] != 0.0f || cb1.data[64] != +2.5f || cb1.data[96] != +2.5f) t1_mismatch++;
        if (cb1.data[1] != -2.5f || cb1.data[33] != 0.0f || cb1.data[65] != 0.0f || cb1.data[97] != -2.5f) t1_mismatch++;
        if (cb1.data[2] != -2.5f || cb1.data[34] != -2.5f || cb1.data[66] != -2.5f || cb1.data[98] != -2.5f) t1_mismatch++;
        if (cb1.data[3] != 0.0f  || cb1.data[35] != 0.0f  || cb1.data[67] != 0.0f  || cb1.data[99] != 0.0f) t1_mismatch++;
        if (cb1.data[4] != +2.5f || cb1.data[36] != +2.5f || cb1.data[68] != +2.5f || cb1.data[100] != +2.5f) t1_mismatch++;
        if (cb1.data[5] != 0.0f  || cb1.data[37] != 0.0f  || cb1.data[69] != 0.0f  || cb1.data[101] != 0.0f) t1_mismatch++;

        /* Cross-check all 128 weights */
        for (int i = 0; i < 128; i++) {
            float exp_w = get_expected_weight(golden_bytes, i, scale);
            if (cb1.data[i] != exp_w || cb2.data[i] != exp_w) {
                t1_mismatch++;
                break;
            }
        }

        if (t1_mismatch == 0) {
            printf("  -> PASS: All 128 weights matched exact ternary mathematics.\n");
        } else {
            printf("  -> FAIL: Golden vector mismatch detected (%d errors).\n", t1_mismatch);
            total_failures++;
        }

test1_cleanup:
        free_canary_buffer(&cb1);
        free_canary_buffer(&cb2);
    }

    /* ------------------------------------------------------------------------
     * Test 2: Boundary Tests (N = 1, 127, 128, 129) with Exact Value Check
     * ------------------------------------------------------------------------ */
    printf("\n[Test 2] Boundary Conditions (N = 1, 127, 128, 129) Exact Output Value Check...\n");
    int test_dims[] = { 1, 127, 128, 129 };
    for (int d = 0; d < 4; d++) {
        int64_t n = test_dims[d];
        int64_t blks = safe_ceil_div128(n);
        size_t in_size = (size_t)blks * 32ULL;
        uint8_t * in_bytes = (uint8_t *)malloc(in_size);
        if (!in_bytes) {
            printf("  -> N = %3lld: FAIL (Memory allocation failed)\n", (long long)n);
            total_failures++;
            continue;
        }

        for (size_t i = 0; i < in_size; i++) {
            in_bytes[i] = (uint8_t)((i * 43 + 7) & 0xFF);
        }

        float test_scale = 1.25f;
        CanaryBuffer cb;
        if (create_canary_buffer((size_t)n, &cb) != 0) {
            printf("  -> N = %3lld: FAIL (Canary buffer allocation failed)\n", (long long)n);
            total_failures++;
            free(in_bytes);
            continue;
        }

        int r1 = dequantize_i2s_canonical(in_bytes, in_size, test_scale, n, cb.data, (size_t)n);
        int r2 = verify_canary_buffer(&cb);

        int val_mismatch = 0;
        if (r1 == 0 && r2 == 0) {
            /* Inspect EVERY decoded value against analytical expectation */
            for (int64_t i = 0; i < n; i++) {
                float expected = get_expected_weight(in_bytes, i, test_scale);
                if (cb.data[i] != expected) {
                    val_mismatch++;
                    break;
                }
            }
        }

        if (r1 == 0 && r2 == 0 && val_mismatch == 0) {
            printf("  -> N = %3lld: PASS (Exact values matched, Canary bounds intact)\n", (long long)n);
        } else {
            printf("  -> N = %3lld: FAIL (r1=%d, canary=%d, val_mismatch=%d)\n", (long long)n, r1, r2, val_mismatch);
            total_failures++;
        }

        free_canary_buffer(&cb);
        free(in_bytes);
    }

    /* ------------------------------------------------------------------------
     * Test 3: +32B Tensor Scale Trailer Check (Bytes [4..31] not assumed zero)
     * ------------------------------------------------------------------------ */
    printf("\n[Test 3] +32B Tensor Scale Trailer Check (Arbitrary non-zero trailer bytes supported)...\n");
    {
        size_t payload_bytes = 32;
        size_t trailer_bytes = 32;
        size_t total_tensor_bytes = payload_bytes + trailer_bytes;
        uint8_t * tensor_buf = (uint8_t *)malloc(total_tensor_bytes);
        if (!tensor_buf) {
            printf("  -> FAIL: Buffer allocation failed.\n");
            total_failures++;
            goto test3_cleanup;
        }

        /* Fill weights payload */
        for (size_t i = 0; i < payload_bytes; i++) tensor_buf[i] = 0xAA; /* All bits 10 -> +1 */

        /* Embed scale into trailer offset 0 (bytes 32..35) */
        float expected_scale = 0.00341796875f;
        memcpy(tensor_buf + payload_bytes, &expected_scale, sizeof(float));

        /* Fill trailer bytes [4..31] with arbitrary NON-ZERO reserved metadata to verify no zero assumption */
        for (size_t p = 4; p < 32; p++) {
            tensor_buf[payload_bytes + p] = (uint8_t)(0xAA ^ (p & 0xFF));
        }

        /* Readback via GGML Trailer specification */
        float extracted_scale = *(const float *)(tensor_buf + payload_bytes);

        CanaryBuffer cb;
        if (create_canary_buffer(128, &cb) != 0) {
            printf("  -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            free(tensor_buf);
            goto test3_cleanup;
        }

        int res = dequantize_i2s_canonical(tensor_buf, payload_bytes, extracted_scale, 128, cb.data, 128);

        if (res == 0 && extracted_scale == expected_scale && cb.data[0] == expected_scale) {
            printf("  -> PASS: Scale Trailer extracted (Float32 Scale: %f, arbitrary trailer bytes preserved)\n", extracted_scale);
        } else {
            printf("  -> FAIL: Scale trailer extraction failed (res=%d, extracted=%f, expected=%f)\n",
                   res, extracted_scale, expected_scale);
            total_failures++;
        }

        free_canary_buffer(&cb);
        free(tensor_buf);
    }
test3_cleanup:

    /* ------------------------------------------------------------------------
     * Test 4: Truncated Input Rejection & Output Buffer Immutability Check
     * ------------------------------------------------------------------------ */
    printf("\n[Test 4] Truncated Input Rejection & Output Buffer Immutability Check...\n");
    {
        /* Require 64 bytes for N=129, but only supply 32 bytes */
        uint8_t small_buf[32];
        memset(small_buf, 0, sizeof(small_buf));

        CanaryBuffer cb;
        if (create_canary_buffer(129, &cb) != 0) {
            printf("  -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            goto test4_cleanup;
        }

        /* Record initial buffer state with sentinel */
        for (size_t i = 0; i < 129; i++) cb.data[i] = SENTINEL_VAL;

        int r1 = dequantize_i2s_canonical(small_buf, 32, 1.0f, 129, cb.data, 129);

        /* Verify output immutability: On error, NO element must be modified */
        int buffer_mutated = 0;
        for (size_t i = 0; i < 129; i++) {
            if (cb.data[i] != SENTINEL_VAL) {
                buffer_mutated = 1;
                break;
            }
        }

        int canary_status = verify_canary_buffer(&cb);

        if (r1 == -3 && buffer_mutated == 0 && canary_status == 0) {
            printf("  -> PASS: Truncated buffer rejected (code -3), output buffer remained 100%% immutable.\n");
        } else {
            printf("  -> FAIL: Error handling defect (r1=%d, mutated=%d, canary=%d)\n", r1, buffer_mutated, canary_status);
            total_failures++;
        }

        free_canary_buffer(&cb);
    }
test4_cleanup:

    /* ------------------------------------------------------------------------
     * Test 5: Arithmetic Overflow Guard Check (No n + 127 wrap)
     * ------------------------------------------------------------------------ */
    printf("\n[Test 5] Arithmetic Overflow Guard Check (No n + 127 wrap)...\n");
    {
        int64_t overflow_n = INT64_MAX - 50; /* n + 127 would wrap to negative if not guarded */
        float dummy_out[4];
        uint8_t dummy_in[32];

        int r = dequantize_i2s_canonical(dummy_in, 32, 1.0f, overflow_n, dummy_out, 4);

        if (r == -2) {
            printf("  -> PASS: Overflow dimension safely trapped with code -2 without integer wrap.\n");
        } else {
            printf("  -> FAIL: Overflow dimension was not trapped (r=%d)\n", r);
            total_failures++;
        }
    }

    /* ------------------------------------------------------------------------
     * Test 6: 10,000 Pseudo-Random Blocks Cross-Decoder Stress Check
     * ------------------------------------------------------------------------ */
    printf("\n[Test 6] 10,000 Blocks (1,280,000 Weights) Dual-Decoder Stress Check...\n");
    {
        size_t test_blocks = 10000;
        size_t total_weights = test_blocks * 128;
        size_t byte_size = test_blocks * 32;
        uint8_t * random_bytes = (uint8_t *)malloc(byte_size);
        float * out1 = (float *)malloc(total_weights * sizeof(float));
        float * out2 = (float *)malloc(total_weights * sizeof(float));

        if (!random_bytes || !out1 || !out2) {
            printf("  -> FAIL: Stress test memory allocation failed.\n");
            total_failures++;
            goto test6_cleanup;
        }

        uint32_t lfsr = 0x12345678;
        for (size_t i = 0; i < byte_size; i++) {
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xD0000001u);
            random_bytes[i] = (uint8_t)(lfsr & 0xFF);
        }

        int res1 = dequantize_i2s_canonical(random_bytes, byte_size, 1.875f, (int64_t)total_weights, out1, total_weights);
        int res2 = dequantize_i2s_independent(random_bytes, byte_size, 1.875f, (int64_t)total_weights, out2, total_weights);

        if (res1 != 0 || res2 != 0) {
            printf("  -> FAIL: Stress test execution returned error (res1=%d, res2=%d)\n", res1, res2);
            total_failures++;
            goto test6_cleanup;
        }

        int mismatch = memcmp(out1, out2, total_weights * sizeof(float));
        if (mismatch == 0) {
            printf("  -> PASS: Dual decoders produced 1,280,000 identical float weights (0 mismatches).\n");
        } else {
            printf("  -> FAIL: Dual decoders diverged on stress test.\n");
            total_failures++;
        }

test6_cleanup:
        if (out1) free(out1);
        if (out2) free(out2);
        if (random_bytes) free(random_bytes);
    }

    printf("\n====================================================================\n");
    printf("Summary: %d Test Suites Passed, %d Failures\n", 6 - total_failures, total_failures);
    printf("====================================================================\n");
    return (total_failures == 0) ? 0 : 1;
}
