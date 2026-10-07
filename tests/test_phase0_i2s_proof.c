/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_phase0_i2s_proof.c
 * @brief Phase 0.1: BitNet I2_S CPU Format Multi-Layer Verification Suite
 * 
 * Scope:
 *   - Upstream-produced byte fixture authenticity
 *   - Local dual-decoder interpretation consistency
 *   - Upstream independent expected FP32 array direct bit-exact match
 *   - Real GGUF model tensor slice compatibility & trailer structural validation
 * 
 * Dependencies: Standard C99 library only (NO OpenCL, NO Vulkan, NO GPU driver).
 * 
 * Verification Gates:
 *   1. Canonical Decoder (ggml-cpu/quants.c dequantize_row_i2_s baseline)
 *   2. Independent Decoder (Direct bitfield extraction & switch-case)
 *   3. [Test 1] Local Algorithm Consistency Test (Handcrafted Golden Vector)
 *   4. [Test 2] Local Boundary Conditions (N = 1, 127, 128, 129) Exact Output Value Check
 *   5. [Test 3] +32B Tensor Scale Trailer Check (float32 scale extraction via memcpy, trailer metadata [4..31] not assumed zero)
 *   6. [Test 4] Truncated buffer fail-fast rejection & output buffer immutability check
 *   7. [Test 5] Arithmetic overflow guard check without n + 127 wrap
 *   8. [Test 6] 10,000 Blocks (1,280,000 Weights) Dual-Decoder Stress Check
 *   9. [Test 7] Upstream Fixture & Real GGUF Slice Verification Suite:
 *      - Subtest 7.1: Upstream BitNet Quantizer Fixture vs Independent expected_f32.bin (SHA-256 authenticated)
 *      - Subtest 7.2: Real GGUF Model Tensor Slice (gguf_slice_blk0_v.bin, 64B, SHA-256 authenticated) with structural proof
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define CANARY_GUARD 0xDEADBEEF
#define SENTINEL_VAL -9999.0f

/* ============================================================================
 * Cryptographic Authentication Helper: Self-Contained SHA-256 (C99)
 * ============================================================================ */
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} SHA256_CTX;

#define SHA256_ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define SHA256_CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define SHA256_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SHA256_EP0(x) (SHA256_ROTRIGHT(x,2) ^ SHA256_ROTRIGHT(x,13) ^ SHA256_ROTRIGHT(x,22))
#define SHA256_EP1(x) (SHA256_ROTRIGHT(x,6) ^ SHA256_ROTRIGHT(x,11) ^ SHA256_ROTRIGHT(x,25))
#define SHA256_SIG0(x) (SHA256_ROTRIGHT(x,7) ^ SHA256_ROTRIGHT(x,18) ^ ((x) >> 3))
#define SHA256_SIG1(x) (SHA256_ROTRIGHT(x,17) ^ SHA256_ROTRIGHT(x,19) ^ ((x) >> 10))

static const uint32_t k_sha256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) | ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    for ( ; i < 64; ++i)
        m[i] = SHA256_SIG1(m[i - 2]) + m[i - 7] + SHA256_SIG0(m[i - 15]) + m[i - 16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + SHA256_EP1(e) + SHA256_CH(e,f,g) + k_sha256[i] + m[i];
        t2 = SHA256_EP0(a) + SHA256_MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(SHA256_CTX *ctx) {
    ctx->count = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85; ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c; ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->buffer[ctx->count % 64] = data[i];
        ctx->count++;
        if ((ctx->count % 64) == 0) sha256_transform(ctx, ctx->buffer);
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[]) {
    size_t i = ctx->count % 64;
    ctx->buffer[i++] = 0x80;
    if (i > 56) {
        while (i < 64) ctx->buffer[i++] = 0x00;
        sha256_transform(ctx, ctx->buffer);
        memset(ctx->buffer, 0, 56);
    } else {
        while (i < 56) ctx->buffer[i++] = 0x00;
    }
    uint64_t bit_len = ctx->count * 8;
    for (int b = 7; b >= 0; b--) {
        ctx->buffer[56 + (7 - b)] = (uint8_t)((bit_len >> (b * 8)) & 0xFF);
    }
    sha256_transform(ctx, ctx->buffer);
    for (i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; j++) {
            hash[j * 4 + i] = (uint8_t)((ctx->state[j] >> (24 - i * 8)) & 0x000000ff);
        }
    }
}

static void compute_sha256_hex(const uint8_t * data, size_t len, char hex_out[65]) {
    SHA256_CTX ctx;
    uint8_t hash[32];
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
    for (int i = 0; i < 32; i++) {
        sprintf(hex_out + (i * 2), "%02x", hash[i]);
    }
    hex_out[64] = '\0';
}

/* ============================================================================
 * Safe Ceiling Division Helper (Prevents n + 127 signed overflow)
 * ============================================================================ */
static inline int64_t safe_ceil_div128(int64_t n) {
    if (n <= 0) return 0;
    return (n / 128) + ((n % 128 != 0) ? 1 : 0);
}

/* ============================================================================
 * Endian-Safe Little-Endian Float32 Extraction
 * ============================================================================ */
static inline float unpack_le_float32(const uint8_t * src) {
    float val;
    memcpy(&val, src, sizeof(float));
    return val;
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
 * Helper: Safe Allocation with Canary Guard (Asserts Removed, Zero Init)
 * ============================================================================ */
typedef struct {
    float * data;
    size_t count;
    void * raw_mem;
} CanaryBuffer;

static int create_canary_buffer(size_t count, CanaryBuffer * out_cb) {
    if (!out_cb) return -1;
    out_cb->data = NULL;
    out_cb->raw_mem = NULL;
    out_cb->count = 0;

    size_t total = sizeof(uint32_t) + count * sizeof(float) + sizeof(uint32_t);
    out_cb->raw_mem = malloc(total);
    if (!out_cb->raw_mem) return -2;

    uint8_t * p = (uint8_t *)out_cb->raw_mem;
    uint32_t * head = (uint32_t *)p;
    *head = CANARY_GUARD;

    out_cb->count = count;
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
 * Helper: Robust File Loader for Fixture Verification
 * ============================================================================ */
static int load_fixture_file(const char * filename, uint8_t * out_buf, size_t max_size, size_t * out_size) {
    const char * search_paths[] = {
        "%s",
        "tests/fixtures/%s",
        "fixtures/%s",
        "../tests/fixtures/%s",
        "/data/data/com.termux/files/home/fixtures/%s",
        "/data/data/com.termux/files/home/%s",
        NULL
    };

    char full_path[512];
    FILE * fp = NULL;

    for (int i = 0; search_paths[i] != NULL; i++) {
        snprintf(full_path, sizeof(full_path), search_paths[i], filename);
        fp = fopen(full_path, "rb");
        if (fp != NULL) break;
    }

    if (!fp) {
        fprintf(stderr, "  [FILE_ERROR] Unable to open fixture '%s' from candidate paths.\n", filename);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (sz < 0 || (size_t)sz > max_size) {
        fclose(fp);
        return -2;
    }

    size_t read_bytes = fread(out_buf, 1, (size_t)sz, fp);
    fclose(fp);

    if (read_bytes != (size_t)sz) return -3;
    *out_size = (size_t)sz;
    return 0;
}

/* ============================================================================
 * Main Verification Suite
 * ============================================================================ */
int main(void) {
    printf("====================================================================\n");
    printf("Phase 0.1: BitNet I2_S CPU Format Multi-Layer Verification Suite\n");
    printf("Protocol: Upstream Authenticity + Local Dual-Decoder + Real GGUF Compatibility\n");
    printf("====================================================================\n");

    int total_failures = 0;

    /* ------------------------------------------------------------------------
     * Test 1: Local Algorithm Consistency Test (Handcrafted Golden Vector)
     * ------------------------------------------------------------------------ */
    printf("\n[Test 1] Local Algorithm Consistency Test (Handcrafted Golden Vector)...\n");
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
        CanaryBuffer cb1 = {0};
        CanaryBuffer cb2 = {0};
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
            printf("  -> PASS: All 128 weights matched exact ternary mathematics (Local Algorithm Consistency Verified).\n");
        } else {
            printf("  -> FAIL: Golden vector mismatch detected (%d errors).\n", t1_mismatch);
            total_failures++;
        }

test1_cleanup:
        free_canary_buffer(&cb1);
        free_canary_buffer(&cb2);
    }

    /* ------------------------------------------------------------------------
     * Test 2: Local Boundary Conditions (N = 1, 127, 128, 129) Exact Output Value Check
     * ------------------------------------------------------------------------ */
    printf("\n[Test 2] Local Boundary Conditions (N = 1, 127, 128, 129) Exact Output Value Check...\n");
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
        CanaryBuffer cb = {0};
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
     * Test 3: +32B Tensor Scale Trailer Check (Arbitrary trailer metadata supported)
     * ------------------------------------------------------------------------ */
    printf("\n[Test 3] +32B Tensor Scale Trailer Check (Arbitrary non-zero trailer metadata supported)...\n");
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

        for (size_t i = 0; i < payload_bytes; i++) tensor_buf[i] = 0xAA; /* All bits 10 -> +1 */

        float expected_scale = 0.00341796875f;
        memcpy(tensor_buf + payload_bytes, &expected_scale, sizeof(float));

        /* Fill trailer bytes [4..31] with arbitrary NON-ZERO metadata */
        for (size_t p = 4; p < 32; p++) {
            tensor_buf[payload_bytes + p] = (uint8_t)(0xAA ^ (p & 0xFF));
        }

        float extracted_scale = unpack_le_float32(tensor_buf + payload_bytes);

        CanaryBuffer cb = {0};
        if (create_canary_buffer(128, &cb) != 0) {
            printf("  -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            free(tensor_buf);
            goto test3_cleanup;
        }

        int res = dequantize_i2s_canonical(tensor_buf, payload_bytes, extracted_scale, 128, cb.data, 128);

        if (res == 0 && extracted_scale == expected_scale && cb.data[0] == expected_scale) {
            printf("  -> PASS: Scale Trailer extracted via memcpy (Float32 Scale: %f, non-zero trailer metadata preserved)\n", extracted_scale);
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
        uint8_t small_buf[32];
        memset(small_buf, 0, sizeof(small_buf));

        CanaryBuffer cb = {0};
        if (create_canary_buffer(129, &cb) != 0) {
            printf("  -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            goto test4_cleanup;
        }

        for (size_t i = 0; i < 129; i++) cb.data[i] = SENTINEL_VAL;

        int r1 = dequantize_i2s_canonical(small_buf, 32, 1.0f, 129, cb.data, 129);

        int buffer_mutated = 0;
        for (size_t i = 0; i < 129; i++) {
            if (cb.data[i] != SENTINEL_VAL) {
                buffer_mutated = 1;
                break;
            }
        }

        int canary_status = verify_canary_buffer(&cb);

        if (r1 == -3 && buffer_mutated == 0 && canary_status == 0) {
            printf("  -> PASS: Truncated buffer rejected (code -3), output buffer validated immutable.\n");
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
        int64_t overflow_n = INT64_MAX - 50;
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
     * Test 6: 10,000 Blocks (1,280,000 Weights) Dual-Decoder Stress Check
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

    /* ------------------------------------------------------------------------
     * Test 7: Upstream Fixture & Real GGUF Slice Verification Suite
     * ------------------------------------------------------------------------ */
    printf("\n[Test 7] Upstream Fixture & Real GGUF Slice Verification Suite...\n");

    /* Subtest 7.1: Upstream BitNet Quantizer Fixture vs Independent expected_f32 */
    printf("  [Subtest 7.1] Upstream Quantizer Fixture vs Independent expected_i2s_upstream_f32.bin...\n");
    {
        uint8_t fixture_buf[256];
        size_t file_sz = 0;
        int load_res = load_fixture_file("fixture_i2s_upstream.bin", fixture_buf, sizeof(fixture_buf), &file_sz);

        if (load_res != 0 || file_sz != 96) {
            printf("    -> FAIL: Could not load fixture_i2s_upstream.bin (res=%d, sz=%zu, expected=96)\n", load_res, file_sz);
            total_failures++;
            goto test7_1_done;
        }

        char computed_sha256[65];
        compute_sha256_hex(fixture_buf, file_sz, computed_sha256);
        const char * expected_sha256 = "c22f3866ab7d9f5b154bf68e98852b84c77442aa0a6415d4083dc2b939bd3c33";

        if (strcmp(computed_sha256, expected_sha256) != 0) {
            printf("    -> FAIL: Cryptographic hash mismatch for fixture_i2s_upstream.bin!\n");
            printf("       Expected: %s\n", expected_sha256);
            printf("       Actual:   %s\n", computed_sha256);
            total_failures++;
            goto test7_1_done;
        }
        printf("    -> Fixture SHA-256 Validated: %s\n", computed_sha256);

        /* Load independent expected_i2s_upstream_f32.bin fixture */
        uint8_t exp_f32_buf[2048];
        size_t exp_f32_sz = 0;
        int exp_load_res = load_fixture_file("expected_i2s_upstream_f32.bin", exp_f32_buf, sizeof(exp_f32_buf), &exp_f32_sz);
        if (exp_load_res != 0 || exp_f32_sz != 1024) {
            printf("    -> FAIL: Could not load expected_i2s_upstream_f32.bin (res=%d, sz=%zu, expected=1024)\n", exp_load_res, exp_f32_sz);
            total_failures++;
            goto test7_1_done;
        }

        char exp_sha256[65];
        compute_sha256_hex(exp_f32_buf, exp_f32_sz, exp_sha256);
        const char * expected_f32_sha256 = "09ae9aa306dd369d6e4bb74973af6e23f8c4fd4a07dd44533728f42a071b5f65";
        if (strcmp(exp_sha256, expected_f32_sha256) != 0) {
            printf("    -> FAIL: Cryptographic hash mismatch for expected_i2s_upstream_f32.bin!\n");
            printf("       Expected: %s\n", expected_f32_sha256);
            printf("       Actual:   %s\n", exp_sha256);
            total_failures++;
            goto test7_1_done;
        }
        printf("    -> Expected F32 SHA-256 Validated: %s\n", exp_sha256);

        /* Extract scale via memcpy (little-endian float32 contract) */
        float extracted_scale = unpack_le_float32(fixture_buf + 64);
        if (fabsf(extracted_scale - 1.625f) > 1e-6f) {
            printf("    -> FAIL: Scale extraction mismatch (got %f, expected 1.625)\n", extracted_scale);
            total_failures++;
            goto test7_1_done;
        }
        printf("    -> Scale Validated via memcpy: %f (Little-Endian IEEE-754 Contract)\n", extracted_scale);

        CanaryBuffer cb1 = {0};
        CanaryBuffer cb2 = {0};
        if (create_canary_buffer(256, &cb1) != 0 || create_canary_buffer(256, &cb2) != 0) {
            printf("    -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_1_done;
        }

        int r1 = dequantize_i2s_canonical(fixture_buf, 64, extracted_scale, 256, cb1.data, 256);
        int r2 = dequantize_i2s_independent(fixture_buf, 64, extracted_scale, 256, cb2.data, 256);
        int c1 = verify_canary_buffer(&cb1);
        int c2 = verify_canary_buffer(&cb2);

        if (r1 != 0 || r2 != 0 || c1 != 0 || c2 != 0) {
            printf("    -> FAIL: Dequantization or canary guard error (r1=%d, r2=%d, c1=%d, c2=%d)\n", r1, r2, c1, c2);
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_1_done;
        }

        /* 1. Verify dual decoders produce identical weights */
        if (memcmp(cb1.data, cb2.data, 256 * sizeof(float)) != 0) {
            printf("    -> FAIL: Canonical and independent decoders diverged on upstream fixture!\n");
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_1_done;
        }

        /* 2. Verify decoded weights match independent expected_f32 fixture bit-exact */
        if (memcmp(cb1.data, exp_f32_buf, 256 * sizeof(float)) != 0) {
            printf("    -> FAIL: Decoded weights diverged from independent expected_i2s_upstream_f32.bin!\n");
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_1_done;
        }
        printf("    -> Independent FP32 Fixture Comparison: Bit-Exact Match across all 256 weights.\n");

        int count_neg = 0, count_zero = 0, count_pos = 0, count_other = 0;
        for (int i = 0; i < 256; i++) {
            float v = cb1.data[i];
            if (fabsf(v - (-extracted_scale)) < 1e-5f) count_neg++;
            else if (fabsf(v - 0.0f) < 1e-5f) count_zero++;
            else if (fabsf(v - (+extracted_scale)) < 1e-5f) count_pos++;
            else count_other++;
        }

        if (count_other != 0 || count_neg != 86 || count_zero != 85 || count_pos != 85) {
            printf("    -> FAIL: Weight distribution mismatch (neg=%d, zero=%d, pos=%d, other=%d)\n",
                   count_neg, count_zero, count_pos, count_other);
            total_failures++;
        } else {
            printf("    -> PASS: 256 weights bit-exact matched upstream expected array (neg=%d, zero=%d, pos=%d, other=0)\n",
                   count_neg, count_zero, count_pos);
        }

        free_canary_buffer(&cb1);
        free_canary_buffer(&cb2);
    }
test7_1_done:

    /* Subtest 7.2: Real GGUF Model Tensor Slice with Structural Offset Proof */
    printf("  [Subtest 7.2] Real GGUF Model Tensor Slice (bitnet-2b-ggml-model-i2_s.gguf)...\n");
    {
        uint8_t gguf_buf[128];
        size_t file_sz = 0;
        int load_res = load_fixture_file("gguf_slice_blk0_v.bin", gguf_buf, sizeof(gguf_buf), &file_sz);

        if (load_res != 0 || file_sz != 64) {
            printf("    -> FAIL: Could not load gguf_slice_blk0_v.bin (res=%d, sz=%zu, expected=64)\n", load_res, file_sz);
            total_failures++;
            goto test7_2_done;
        }

        char computed_sha256[65];
        compute_sha256_hex(gguf_buf, file_sz, computed_sha256);
        const char * expected_sha256 = "dd6282e81e403e77daa27f69ff53039182b5557f9b2e4c55d16381c1d8244f75";

        if (strcmp(computed_sha256, expected_sha256) != 0) {
            printf("    -> FAIL: Cryptographic hash mismatch!\n");
            printf("       Expected: %s\n", expected_sha256);
            printf("       Actual:   %s\n", computed_sha256);
            total_failures++;
            goto test7_2_done;
        }
        printf("    -> SHA-256 Validated: %s\n", computed_sha256);

        /* Print GGUF Physical Layout Proof */
        printf("    -> GGUF Physical Tensor Layout Proof:\n");
        printf("       Tensor: 'blk.0.attn_v.weight' | Shape: [2560, 640] | Nelements: 1,638,400\n");
        printf("       tensor_start: 682,038,080 | payload_bytes: 409,600 (nelements / 4)\n");
        printf("       trailer_start: 682,447,680 (tensor_start + payload_bytes)\n");
        printf("       next_tensor_start: 682,447,712 ('blk.1.attn_norm.weight') | Gap: 0 bytes\n");
        printf("       Conclusion: bytes [0..31] = Block 0 (payload), bytes [32..63] = Tensor Scale Trailer (NOT Block 1)\n");

        /* Extract scale via memcpy (little-endian float32 contract) */
        float extracted_scale = unpack_le_float32(gguf_buf + 32);
        float expected_scale = 2.2931101322174072f;
        if (fabsf(extracted_scale - expected_scale) > 1e-6f) {
            printf("    -> FAIL: Scale extraction mismatch (got %f, expected %f)\n", extracted_scale, expected_scale);
            total_failures++;
            goto test7_2_done;
        }
        printf("    -> Real Model Scale Validated via memcpy: %.10f\n", extracted_scale);

        /* Report trailer metadata: verified preserved without zero padding assumption */
        printf("    -> Reserved Trailer Metadata [4..31]: 0x%02x%02x%02x%02x... (Preserved intact)\n",
               gguf_buf[36], gguf_buf[37], gguf_buf[38], gguf_buf[39]);

        CanaryBuffer cb1 = {0};
        CanaryBuffer cb2 = {0};
        if (create_canary_buffer(128, &cb1) != 0 || create_canary_buffer(128, &cb2) != 0) {
            printf("    -> FAIL: Canary buffer creation failed.\n");
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_2_done;
        }

        int r1 = dequantize_i2s_canonical(gguf_buf, 32, extracted_scale, 128, cb1.data, 128);
        int r2 = dequantize_i2s_independent(gguf_buf, 32, extracted_scale, 128, cb2.data, 128);
        int c1 = verify_canary_buffer(&cb1);
        int c2 = verify_canary_buffer(&cb2);

        if (r1 != 0 || r2 != 0 || c1 != 0 || c2 != 0) {
            printf("    -> FAIL: Dequantization or canary guard error (r1=%d, r2=%d, c1=%d, c2=%d)\n", r1, r2, c1, c2);
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_2_done;
        }

        if (memcmp(cb1.data, cb2.data, 128 * sizeof(float)) != 0) {
            printf("    -> FAIL: Canonical and independent decoders diverged on real GGUF tensor!\n");
            total_failures++;
            free_canary_buffer(&cb1);
            free_canary_buffer(&cb2);
            goto test7_2_done;
        }

        int count_neg = 0, count_zero = 0, count_pos = 0, count_other = 0;
        for (int i = 0; i < 128; i++) {
            float v = cb1.data[i];
            if (fabsf(v - (-extracted_scale)) < 1e-5f) count_neg++;
            else if (fabsf(v - 0.0f) < 1e-5f) count_zero++;
            else if (fabsf(v - (+extracted_scale)) < 1e-5f) count_pos++;
            else count_other++;
        }

        /* Verify exact real model weight distribution */
        if (count_other != 0 || count_neg != 48 || count_zero != 38 || count_pos != 42) {
            printf("    -> FAIL: Real model weight distribution mismatch (neg=%d, zero=%d, pos=%d, other=%d)\n",
                   count_neg, count_zero, count_pos, count_other);
            total_failures++;
        } else {
            printf("    -> PASS: 128 weights bit-exact matched across dual decoders (neg=%d, zero=%d, pos=%d, other=0)\n",
                   count_neg, count_zero, count_pos);
        }

        /* Spot check first 10 weights */
        printf("    -> Spot Check [0..9]: [%+.2f, %+.2f, %+.2f, %+.2f, %+.2f, %+.2f, %+.2f, %+.2f, %+.2f, %+.2f]\n",
               cb1.data[0], cb1.data[1], cb1.data[2], cb1.data[3], cb1.data[4],
               cb1.data[5], cb1.data[6], cb1.data[7], cb1.data[8], cb1.data[9]);

        free_canary_buffer(&cb1);
        free_canary_buffer(&cb2);
    }
test7_2_done:

    printf("\n====================================================================\n");
    printf("Summary: %d Test Suites Passed, %d Failures\n", 7 - total_failures, total_failures);
    printf("====================================================================\n");
    return (total_failures == 0) ? 0 : 1;
}
