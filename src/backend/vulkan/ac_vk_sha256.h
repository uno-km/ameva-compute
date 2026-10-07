/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#ifndef AC_VK_SHA256_H
#define AC_VK_SHA256_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} AcSha256Ctx;

#define AC_SHA256_ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define AC_SHA256_CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define AC_SHA256_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define AC_SHA256_EP0(x) (AC_SHA256_ROTRIGHT(x,2) ^ AC_SHA256_ROTRIGHT(x,13) ^ AC_SHA256_ROTRIGHT(x,22))
#define AC_SHA256_EP1(x) (AC_SHA256_ROTRIGHT(x,6) ^ AC_SHA256_ROTRIGHT(x,11) ^ AC_SHA256_ROTRIGHT(x,25))
#define AC_SHA256_SIG0(x) (AC_SHA256_ROTRIGHT(x,7) ^ AC_SHA256_ROTRIGHT(x,18) ^ ((x) >> 3))
#define AC_SHA256_SIG1(x) (AC_SHA256_ROTRIGHT(x,17) ^ AC_SHA256_ROTRIGHT(x,19) ^ ((x) >> 10))

static const uint32_t k_ac_sha256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static inline void ac_sha256_transform(AcSha256Ctx *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) | ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    for ( ; i < 64; ++i)
        m[i] = AC_SHA256_SIG1(m[i - 2]) + m[i - 7] + AC_SHA256_SIG0(m[i - 15]) + m[i - 16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + AC_SHA256_EP1(e) + AC_SHA256_CH(e,f,g) + k_ac_sha256[i] + m[i];
        t2 = AC_SHA256_EP0(a) + AC_SHA256_MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static inline void ac_sha256_init(AcSha256Ctx *ctx) {
    ctx->count = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85; ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c; ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static inline void ac_sha256_update(AcSha256Ctx *ctx, const uint8_t data[], size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->buffer[ctx->count % 64] = data[i];
        ctx->count++;
        if ((ctx->count % 64) == 0) ac_sha256_transform(ctx, ctx->buffer);
    }
}

static inline void ac_sha256_final(AcSha256Ctx *ctx, uint8_t hash[]) {
    size_t i = ctx->count % 64;
    ctx->buffer[i++] = 0x80;
    if (i > 56) {
        while (i < 64) ctx->buffer[i++] = 0x00;
        ac_sha256_transform(ctx, ctx->buffer);
        memset(ctx->buffer, 0, 56);
    } else {
        while (i < 56) ctx->buffer[i++] = 0x00;
    }
    uint64_t bit_len = ctx->count * 8;
    for (int b = 7; b >= 0; b--) {
        ctx->buffer[56 + (7 - b)] = (uint8_t)((bit_len >> (b * 8)) & 0xFF);
    }
    ac_sha256_transform(ctx, ctx->buffer);
    for (i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; j++) {
            hash[j * 4 + i] = (uint8_t)((ctx->state[j] >> (24 - i * 8)) & 0x000000ff);
        }
    }
}

static inline void ac_vk_sha256_hash(const void * data, size_t len, uint8_t out[32]) {
    AcSha256Ctx ctx;
    ac_sha256_init(&ctx);
    ac_sha256_update(&ctx, (const uint8_t *)data, len);
    ac_sha256_final(&ctx, out);
}

static inline void ac_sha256_compute_hex(const uint8_t * data, size_t len, char hex_out[65]) {
    AcSha256Ctx ctx;
    uint8_t hash[32];
    ac_sha256_init(&ctx);
    ac_sha256_update(&ctx, data, len);
    ac_sha256_final(&ctx, hash);
    for (int i = 0; i < 32; i++) {
        sprintf(hex_out + (i * 2), "%02x", hash[i]);
    }
    hex_out[64] = '\0';
}

#ifdef __cplusplus
}
#endif

#endif /* AC_VK_SHA256_H */
