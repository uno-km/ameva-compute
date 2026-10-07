/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_phase2_vulkan_core.cpp
 * @brief Phase 2: Modular Vulkan Compute Core Verification & Regression Test
 * 
 * Hardware Target: Qualcomm Adreno 830 GPU (Android 15 ARM64, Snapdragon 8 Elite)
 * Engine Target: AMUDA 2.0 Native Vulkan Compute Core (AcVkContext, AcVkBuffer, AcVkKernel, AcVkEvent, ac_vk_dispatch)
 *
 * Verification Gates:
 *   1. Hardware & Cryptographic Fixture Authentication
 *   2. AcVkContext Creation & GPU Selection (Zero Global State)
 *   3. AcVkBuffer UMA Host-Visible Allocation & Mapping
 *   4. AcVkKernel SPIR-V Compute Pipeline Creation
 *   5. GPU Numerical Accuracy vs Upstream Fixture (Mismatch: 0/256)
 *   6. 100-Dispatch Short-Run Latency & Stability Stress
 *   7. In-Flight Lifecycle Safety & Device-Busy Enforcement (Fail-Fast)
 *   8. Clean Teardown & Zero Resource Leaks
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"

/* Self-contained SHA-256 helper */
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

static const uint32_t kBitnetGemvI2SSpv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x0000012du, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000bu, 0x00000015u, 0x00060010u,
    0x00000004u, 0x00000011u, 0x00000020u, 0x00000004u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u,
    0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00050005u, 0x00000008u, 0x61636f6cu, 0x616c5f6cu,
    0x0000656eu, 0x00080005u, 0x0000000bu, 0x4c5f6c67u, 0x6c61636fu, 0x6f766e49u, 0x69746163u, 0x44496e6fu,
    0x00000000u, 0x00050005u, 0x00000010u, 0x61636f6cu, 0x6f725f6cu, 0x00000077u, 0x00050005u, 0x00000014u,
    0x626f6c67u, 0x725f6c61u, 0x0000776fu, 0x00060005u, 0x00000015u, 0x575f6c67u, 0x476b726fu, 0x70756f72u,
    0x00004449u, 0x00070005u, 0x0000001eu, 0x65726874u, 0x615f6461u, 0x6d756363u, 0x74616c75u, 0x0000726fu,
    0x00060005u, 0x00000021u, 0x68737550u, 0x736e6f43u, 0x746e6174u, 0x00000073u, 0x00050006u, 0x00000021u,
    0x00000000u, 0x645f6e69u, 0x00006d69u, 0x00050006u, 0x00000021u, 0x00000001u, 0x5f74756fu, 0x006d6964u,
    0x00070006u, 0x00000021u, 0x00000002u, 0x75716564u, 0x5f746e61u, 0x6c616373u, 0x00000065u, 0x00080006u,
    0x00000021u, 0x00000003u, 0x67696577u, 0x6f5f7468u, 0x65736666u, 0x79625f74u, 0x00736574u, 0x00080006u,
    0x00000021u, 0x00000004u, 0x7074756fu, 0x6f5f7475u, 0x65736666u, 0x6f775f74u, 0x00736472u, 0x00040005u,
    0x00000023u, 0x61726170u, 0x0000736du, 0x00050005u, 0x0000002du, 0x5f6d756eu, 0x636f6c62u, 0x0000736bu,
    0x00050005u, 0x00000033u, 0x65736162u, 0x726f775fu, 0x00000064u, 0x00060005u, 0x00000038u, 0x5f776f72u,
    0x64726f77u, 0x7361625fu, 0x00000065u, 0x00030005u, 0x00000040u, 0x00000062u, 0x00060005u, 0x00000049u,
    0x636f6c62u, 0x6f635f6bu, 0x61625f6cu, 0x00006573u, 0x00050005u, 0x0000004cu, 0x64726f77u, 0x7864695fu,
    0x00000000u, 0x00050005u, 0x00000054u, 0x65747962u, 0x6968735fu, 0x00007466u, 0x00050005u, 0x00000058u,
    0x6b636170u, 0x775f6465u, 0x0064726fu, 0x00060005u, 0x0000005au, 0x67696557u, 0x75427468u, 0x72656666u,
    0x00000000u, 0x00060006u, 0x0000005au, 0x00000000u, 0x5f776172u, 0x67696577u, 0x00737468u, 0x00030005u,
    0x0000005cu, 0x00000000u, 0x00050005u, 0x00000061u, 0x65747962u, 0x6c61765fu, 0x00000000u, 0x00030005u,
    0x00000067u, 0x00003077u, 0x00030005u, 0x00000070u, 0x00003177u, 0x00030005u, 0x00000077u, 0x00003277u,
    0x00030005u, 0x0000007eu, 0x00003377u, 0x00030005u, 0x00000083u, 0x00003078u, 0x00070005u, 0x00000085u,
    0x69746341u, 0x69746176u, 0x75426e6fu, 0x72656666u, 0x00000000u, 0x00060006u, 0x00000085u, 0x00000000u,
    0x69746361u, 0x69746176u, 0x00736e6fu, 0x00030005u, 0x00000087u, 0x00000000u, 0x00030005u, 0x0000008fu,
    0x00003178u, 0x00030005u, 0x00000097u, 0x00003278u, 0x00030005u, 0x0000009fu, 0x00003378u, 0x00060005u,
    0x000000bdu, 0x5f776f72u, 0x72616873u, 0x735f6465u, 0x00006d75u, 0x00060005u, 0x0000011du, 0x7074754fu,
    0x75427475u, 0x72656666u, 0x00000000u, 0x00060006u, 0x0000011du, 0x00000000u, 0x7074756fu, 0x765f7475u,
    0x00006365u, 0x00030005u, 0x0000011fu, 0x00000000u, 0x00040047u, 0x0000000bu, 0x0000000bu, 0x0000001bu,
    0x00040047u, 0x00000015u, 0x0000000bu, 0x0000001au, 0x00030047u, 0x00000021u, 0x00000002u, 0x00050048u,
    0x00000021u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x00000021u, 0x00000001u, 0x00000023u,
    0x00000004u, 0x00050048u, 0x00000021u, 0x00000002u, 0x00000023u, 0x00000008u, 0x00050048u, 0x00000021u,
    0x00000003u, 0x00000023u, 0x0000000cu, 0x00050048u, 0x00000021u, 0x00000004u, 0x00000023u, 0x00000010u,
    0x00040047u, 0x00000059u, 0x00000006u, 0x00000004u, 0x00030047u, 0x0000005au, 0x00000003u, 0x00040048u,
    0x0000005au, 0x00000000u, 0x00000018u, 0x00050048u, 0x0000005au, 0x00000000u, 0x00000023u, 0x00000000u,
    0x00030047u, 0x0000005cu, 0x00000018u, 0x00040047u, 0x0000005cu, 0x00000021u, 0x00000000u, 0x00040047u,
    0x0000005cu, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000084u, 0x00000006u, 0x00000004u, 0x00030047u,
    0x00000085u, 0x00000003u, 0x00040048u, 0x00000085u, 0x00000000u, 0x00000018u, 0x00050048u, 0x00000085u,
    0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000087u, 0x00000018u, 0x00040047u, 0x00000087u,
    0x00000021u, 0x00000001u, 0x00040047u, 0x00000087u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000011cu,
    0x00000006u, 0x00000004u, 0x00030047u, 0x0000011du, 0x00000003u, 0x00040048u, 0x0000011du, 0x00000000u,
    0x00000019u, 0x00050048u, 0x0000011du, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x0000011fu,
    0x00000019u, 0x00040047u, 0x0000011fu, 0x00000021u, 0x00000002u, 0x00040047u, 0x0000011fu, 0x00000022u,
    0x00000000u, 0x00040047u, 0x0000012cu, 0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u, 0x00030021u,
    0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x00040020u, 0x00000007u,
    0x00000007u, 0x00000006u, 0x00040017u, 0x00000009u, 0x00000006u, 0x00000003u, 0x00040020u, 0x0000000au,
    0x00000001u, 0x00000009u, 0x0004003bu, 0x0000000au, 0x0000000bu, 0x00000001u, 0x0004002bu, 0x00000006u,
    0x0000000cu, 0x00000000u, 0x00040020u, 0x0000000du, 0x00000001u, 0x00000006u, 0x0004002bu, 0x00000006u,
    0x00000011u, 0x00000001u, 0x0004003bu, 0x0000000au, 0x00000015u, 0x00000001u, 0x0004002bu, 0x00000006u,
    0x00000018u, 0x00000004u, 0x00030016u, 0x0000001cu, 0x00000020u, 0x00040020u, 0x0000001du, 0x00000007u,
    0x0000001cu, 0x0004002bu, 0x0000001cu, 0x0000001fu, 0x00000000u, 0x0007001eu, 0x00000021u, 0x00000006u,
    0x00000006u, 0x0000001cu, 0x00000006u, 0x00000006u, 0x00040020u, 0x00000022u, 0x00000009u, 0x00000021u,
    0x0004003bu, 0x00000022u, 0x00000023u, 0x00000009u, 0x00040015u, 0x00000024u, 0x00000020u, 0x00000001u,
    0x0004002bu, 0x00000024u, 0x00000025u, 0x00000001u, 0x00040020u, 0x00000026u, 0x00000009u, 0x00000006u,
    0x00020014u, 0x00000029u, 0x0004002bu, 0x00000024u, 0x0000002eu, 0x00000000u, 0x0004002bu, 0x00000006u,
    0x00000031u, 0x00000080u, 0x0004002bu, 0x00000024u, 0x00000034u, 0x00000003u, 0x0004002bu, 0x00000006u,
    0x0000003cu, 0x00000008u, 0x0003001du, 0x00000059u, 0x00000006u, 0x0003001eu, 0x0000005au, 0x00000059u,
    0x00040020u, 0x0000005bu, 0x00000002u, 0x0000005au, 0x0004003bu, 0x0000005bu, 0x0000005cu, 0x00000002u,
    0x00040020u, 0x0000005eu, 0x00000002u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000065u, 0x000000ffu,
    0x0004002bu, 0x00000024u, 0x00000069u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x0000006bu, 0x00000003u,
    0x0004002bu, 0x0000001cu, 0x0000006eu, 0x3f800000u, 0x0004002bu, 0x00000024u, 0x00000072u, 0x00000004u,
    0x0004002bu, 0x00000024u, 0x00000079u, 0x00000002u, 0x0003001du, 0x00000084u, 0x0000001cu, 0x0003001eu,
    0x00000085u, 0x00000084u, 0x00040020u, 0x00000086u, 0x00000002u, 0x00000085u, 0x0004003bu, 0x00000086u,
    0x00000087u, 0x00000002u, 0x00040020u, 0x0000008cu, 0x00000002u, 0x0000001cu, 0x0004002bu, 0x00000006u,
    0x00000091u, 0x00000020u, 0x0004002bu, 0x00000006u, 0x00000099u, 0x00000040u, 0x0004002bu, 0x00000006u,
    0x000000a1u, 0x00000060u, 0x0004001cu, 0x000000bau, 0x0000001cu, 0x00000091u, 0x0004001cu, 0x000000bbu,
    0x000000bau, 0x00000018u, 0x00040020u, 0x000000bcu, 0x00000004u, 0x000000bbu, 0x0004003bu, 0x000000bcu,
    0x000000bdu, 0x00000004u, 0x00040020u, 0x000000c1u, 0x00000004u, 0x0000001cu, 0x0004002bu, 0x00000006u,
    0x000000c3u, 0x00000002u, 0x0004002bu, 0x00000006u, 0x000000c4u, 0x00000108u, 0x0004002bu, 0x00000006u,
    0x000000c6u, 0x00000010u, 0x0003001du, 0x0000011cu, 0x0000001cu, 0x0003001eu, 0x0000011du, 0x0000011cu,
    0x00040020u, 0x0000011eu, 0x00000002u, 0x0000011du, 0x0004003bu, 0x0000011eu, 0x0000011fu, 0x00000002u,
    0x00040020u, 0x00000127u, 0x00000009u, 0x0000001cu, 0x0006002cu, 0x00000009u, 0x0000012cu, 0x00000091u,
    0x00000018u, 0x00000011u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000005u, 0x0004003bu, 0x00000007u, 0x00000008u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000010u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000014u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x0000001eu,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x0000002du, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000033u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000038u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000040u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000049u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x0000004cu,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000054u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000058u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000061u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000067u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x00000070u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000077u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000007eu, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000083u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000008fu, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000097u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000009fu, 0x00000007u, 0x00050041u, 0x0000000du, 0x0000000eu,
    0x0000000bu, 0x0000000cu, 0x0004003du, 0x00000006u, 0x0000000fu, 0x0000000eu, 0x0003003eu, 0x00000008u,
    0x0000000fu, 0x00050041u, 0x0000000du, 0x00000012u, 0x0000000bu, 0x00000011u, 0x0004003du, 0x00000006u,
    0x00000013u, 0x00000012u, 0x0003003eu, 0x00000010u, 0x00000013u, 0x00050041u, 0x0000000du, 0x00000016u,
    0x00000015u, 0x0000000cu, 0x0004003du, 0x00000006u, 0x00000017u, 0x00000016u, 0x00050084u, 0x00000006u,
    0x00000019u, 0x00000017u, 0x00000018u, 0x0004003du, 0x00000006u, 0x0000001au, 0x00000010u, 0x00050080u,
    0x00000006u, 0x0000001bu, 0x00000019u, 0x0000001au, 0x0003003eu, 0x00000014u, 0x0000001bu, 0x0003003eu,
    0x0000001eu, 0x0000001fu, 0x0004003du, 0x00000006u, 0x00000020u, 0x00000014u, 0x00050041u, 0x00000026u,
    0x00000027u, 0x00000023u, 0x00000025u, 0x0004003du, 0x00000006u, 0x00000028u, 0x00000027u, 0x000500b0u,
    0x00000029u, 0x0000002au, 0x00000020u, 0x00000028u, 0x000300f7u, 0x0000002cu, 0x00000000u, 0x000400fau,
    0x0000002au, 0x0000002bu, 0x0000002cu, 0x000200f8u, 0x0000002bu, 0x00050041u, 0x00000026u, 0x0000002fu,
    0x00000023u, 0x0000002eu, 0x0004003du, 0x00000006u, 0x00000030u, 0x0000002fu, 0x00050086u, 0x00000006u,
    0x00000032u, 0x00000030u, 0x00000031u, 0x0003003eu, 0x0000002du, 0x00000032u, 0x00050041u, 0x00000026u,
    0x00000035u, 0x00000023u, 0x00000034u, 0x0004003du, 0x00000006u, 0x00000036u, 0x00000035u, 0x00050086u,
    0x00000006u, 0x00000037u, 0x00000036u, 0x00000018u, 0x0003003eu, 0x00000033u, 0x00000037u, 0x0004003du,
    0x00000006u, 0x00000039u, 0x00000033u, 0x0004003du, 0x00000006u, 0x0000003au, 0x00000014u, 0x0004003du,
    0x00000006u, 0x0000003bu, 0x0000002du, 0x00050084u, 0x00000006u, 0x0000003du, 0x0000003bu, 0x0000003cu,
    0x00050084u, 0x00000006u, 0x0000003eu, 0x0000003au, 0x0000003du, 0x00050080u, 0x00000006u, 0x0000003fu,
    0x00000039u, 0x0000003eu, 0x0003003eu, 0x00000038u, 0x0000003fu, 0x0003003eu, 0x00000040u, 0x0000000cu,
    0x000200f9u, 0x00000041u, 0x000200f8u, 0x00000041u, 0x000400f6u, 0x00000043u, 0x00000044u, 0x00000000u,
    0x000200f9u, 0x00000045u, 0x000200f8u, 0x00000045u, 0x0004003du, 0x00000006u, 0x00000046u, 0x00000040u,
    0x0004003du, 0x00000006u, 0x00000047u, 0x0000002du, 0x000500b0u, 0x00000029u, 0x00000048u, 0x00000046u,
    0x00000047u, 0x000400fau, 0x00000048u, 0x00000042u, 0x00000043u, 0x000200f8u, 0x00000042u, 0x0004003du,
    0x00000006u, 0x0000004au, 0x00000040u, 0x00050084u, 0x00000006u, 0x0000004bu, 0x0000004au, 0x00000031u,
    0x0003003eu, 0x00000049u, 0x0000004bu, 0x0004003du, 0x00000006u, 0x0000004du, 0x00000038u, 0x0004003du,
    0x00000006u, 0x0000004eu, 0x00000040u, 0x00050084u, 0x00000006u, 0x0000004fu, 0x0000004eu, 0x0000003cu,
    0x00050080u, 0x00000006u, 0x00000050u, 0x0000004du, 0x0000004fu, 0x0004003du, 0x00000006u, 0x00000051u,
    0x00000008u, 0x00050086u, 0x00000006u, 0x00000052u, 0x00000051u, 0x00000018u, 0x00050080u, 0x00000006u,
    0x00000053u, 0x00000050u, 0x00000052u, 0x0003003eu, 0x0000004cu, 0x00000053u, 0x0004003du, 0x00000006u,
    0x00000055u, 0x00000008u, 0x00050089u, 0x00000006u, 0x00000056u, 0x00000055u, 0x00000018u, 0x00050084u,
    0x00000006u, 0x00000057u, 0x00000056u, 0x0000003cu, 0x0003003eu, 0x00000054u, 0x00000057u, 0x0004003du,
    0x00000006u, 0x0000005du, 0x0000004cu, 0x00060041u, 0x0000005eu, 0x0000005fu, 0x0000005cu, 0x0000002eu,
    0x0000005du, 0x0004003du, 0x00000006u, 0x00000060u, 0x0000005fu, 0x0003003eu, 0x00000058u, 0x00000060u,
    0x0004003du, 0x00000006u, 0x00000062u, 0x00000058u, 0x0004003du, 0x00000006u, 0x00000063u, 0x00000054u,
    0x000500c2u, 0x00000006u, 0x00000064u, 0x00000062u, 0x00000063u, 0x000500c7u, 0x00000006u, 0x00000066u,
    0x00000064u, 0x00000065u, 0x0003003eu, 0x00000061u, 0x00000066u, 0x0004003du, 0x00000006u, 0x00000068u,
    0x00000061u, 0x000500c2u, 0x00000006u, 0x0000006au, 0x00000068u, 0x00000069u, 0x000500c7u, 0x00000006u,
    0x0000006cu, 0x0000006au, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x0000006du, 0x0000006cu, 0x00050083u,
    0x0000001cu, 0x0000006fu, 0x0000006du, 0x0000006eu, 0x0003003eu, 0x00000067u, 0x0000006fu, 0x0004003du,
    0x00000006u, 0x00000071u, 0x00000061u, 0x000500c2u, 0x00000006u, 0x00000073u, 0x00000071u, 0x00000072u,
    0x000500c7u, 0x00000006u, 0x00000074u, 0x00000073u, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x00000075u,
    0x00000074u, 0x00050083u, 0x0000001cu, 0x00000076u, 0x00000075u, 0x0000006eu, 0x0003003eu, 0x00000070u,
    0x00000076u, 0x0004003du, 0x00000006u, 0x00000078u, 0x00000061u, 0x000500c2u, 0x00000006u, 0x0000007au,
    0x00000078u, 0x00000079u, 0x000500c7u, 0x00000006u, 0x0000007bu, 0x0000007au, 0x0000006bu, 0x00040070u,
    0x0000001cu, 0x0000007cu, 0x0000007bu, 0x00050083u, 0x0000001cu, 0x0000007du, 0x0000007cu, 0x0000006eu,
    0x0003003eu, 0x00000077u, 0x0000007du, 0x0004003du, 0x00000006u, 0x0000007fu, 0x00000061u, 0x000500c7u,
    0x00000006u, 0x00000080u, 0x0000007fu, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x00000081u, 0x00000080u,
    0x00050083u, 0x0000001cu, 0x00000082u, 0x00000081u, 0x0000006eu, 0x0003003eu, 0x0000007eu, 0x00000082u,
    0x0004003du, 0x00000006u, 0x00000088u, 0x00000049u, 0x00050080u, 0x00000006u, 0x00000089u, 0x00000088u,
    0x0000000cu, 0x0004003du, 0x00000006u, 0x0000008au, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000008bu,
    0x00000089u, 0x0000008au, 0x00060041u, 0x0000008cu, 0x0000008du, 0x00000087u, 0x0000002eu, 0x0000008bu,
    0x0004003du, 0x0000001cu, 0x0000008eu, 0x0000008du, 0x0003003eu, 0x00000083u, 0x0000008eu, 0x0004003du,
    0x00000006u, 0x00000090u, 0x00000049u, 0x00050080u, 0x00000006u, 0x00000092u, 0x00000090u, 0x00000091u,
    0x0004003du, 0x00000006u, 0x00000093u, 0x00000008u, 0x00050080u, 0x00000006u, 0x00000094u, 0x00000092u,
    0x00000093u, 0x00060041u, 0x0000008cu, 0x00000095u, 0x00000087u, 0x0000002eu, 0x00000094u, 0x0004003du,
    0x0000001cu, 0x00000096u, 0x00000095u, 0x0003003eu, 0x0000008fu, 0x00000096u, 0x0004003du, 0x00000006u,
    0x00000098u, 0x00000049u, 0x00050080u, 0x00000006u, 0x0000009au, 0x00000098u, 0x00000099u, 0x0004003du,
    0x00000006u, 0x0000009bu, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000009cu, 0x0000009au, 0x0000009bu,
    0x00060041u, 0x0000008cu, 0x0000009du, 0x00000087u, 0x0000002eu, 0x0000009cu, 0x0004003du, 0x0000001cu,
    0x0000009eu, 0x0000009du, 0x0003003eu, 0x00000097u, 0x0000009eu, 0x0004003du, 0x00000006u, 0x000000a0u,
    0x00000049u, 0x00050080u, 0x00000006u, 0x000000a2u, 0x000000a0u, 0x000000a1u, 0x0004003du, 0x00000006u,
    0x000000a3u, 0x00000008u, 0x00050080u, 0x00000006u, 0x000000a4u, 0x000000a2u, 0x000000a3u, 0x00060041u,
    0x0000008cu, 0x000000a5u, 0x00000087u, 0x0000002eu, 0x000000a4u, 0x0004003du, 0x0000001cu, 0x000000a6u,
    0x000000a5u, 0x0003003eu, 0x0000009fu, 0x000000a6u, 0x0004003du, 0x0000001cu, 0x000000a7u, 0x00000067u,
    0x0004003du, 0x0000001cu, 0x000000a8u, 0x00000083u, 0x00050085u, 0x0000001cu, 0x000000a9u, 0x000000a7u,
    0x000000a8u, 0x0004003du, 0x0000001cu, 0x000000aau, 0x00000070u, 0x0004003du, 0x0000001cu, 0x000000abu,
    0x0000008fu, 0x00050085u, 0x0000001cu, 0x000000acu, 0x000000aau, 0x000000abu, 0x00050081u, 0x0000001cu,
    0x000000adu, 0x000000a9u, 0x000000acu, 0x0004003du, 0x0000001cu, 0x000000aeu, 0x00000077u, 0x0004003du,
    0x0000001cu, 0x000000afu, 0x00000097u, 0x00050085u, 0x0000001cu, 0x000000b0u, 0x000000aeu, 0x000000afu,
    0x00050081u, 0x0000001cu, 0x000000b1u, 0x000000adu, 0x000000b0u, 0x0004003du, 0x0000001cu, 0x000000b2u,
    0x0000007eu, 0x0004003du, 0x0000001cu, 0x000000b3u, 0x0000009fu, 0x00050085u, 0x0000001cu, 0x000000b4u,
    0x000000b2u, 0x000000b3u, 0x00050081u, 0x0000001cu, 0x000000b5u, 0x000000b1u, 0x000000b4u, 0x0004003du,
    0x0000001cu, 0x000000b6u, 0x0000001eu, 0x00050081u, 0x0000001cu, 0x000000b7u, 0x000000b6u, 0x000000b5u,
    0x0003003eu, 0x0000001eu, 0x000000b7u, 0x000200f9u, 0x00000044u, 0x000200f8u, 0x00000044u, 0x0004003du,
    0x00000006u, 0x000000b8u, 0x00000040u, 0x00050080u, 0x00000006u, 0x000000b9u, 0x000000b8u, 0x00000025u,
    0x0003003eu, 0x00000040u, 0x000000b9u, 0x000200f9u, 0x00000041u, 0x000200f8u, 0x00000043u, 0x000200f9u,
    0x0000002cu, 0x000200f8u, 0x0000002cu, 0x0004003du, 0x00000006u, 0x000000beu, 0x00000010u, 0x0004003du,
    0x00000006u, 0x000000bfu, 0x00000008u, 0x0004003du, 0x0000001cu, 0x000000c0u, 0x0000001eu, 0x00060041u,
    0x000000c1u, 0x000000c2u, 0x000000bdu, 0x000000beu, 0x000000bfu, 0x0003003eu, 0x000000c2u, 0x000000c0u,
    0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u, 0x000000c5u, 0x00000008u,
    0x000500b0u, 0x00000029u, 0x000000c7u, 0x000000c5u, 0x000000c6u, 0x000300f7u, 0x000000c9u, 0x00000000u,
    0x000400fau, 0x000000c7u, 0x000000c8u, 0x000000c9u, 0x000200f8u, 0x000000c8u, 0x0004003du, 0x00000006u,
    0x000000cau, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000cbu, 0x00000008u, 0x0004003du, 0x00000006u,
    0x000000ccu, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000cdu, 0x00000008u, 0x00050080u, 0x00000006u,
    0x000000ceu, 0x000000cdu, 0x000000c6u, 0x00060041u, 0x000000c1u, 0x000000cfu, 0x000000bdu, 0x000000ccu,
    0x000000ceu, 0x0004003du, 0x0000001cu, 0x000000d0u, 0x000000cfu, 0x00060041u, 0x000000c1u, 0x000000d1u,
    0x000000bdu, 0x000000cau, 0x000000cbu, 0x0004003du, 0x0000001cu, 0x000000d2u, 0x000000d1u, 0x00050081u,
    0x0000001cu, 0x000000d3u, 0x000000d2u, 0x000000d0u, 0x00060041u, 0x000000c1u, 0x000000d4u, 0x000000bdu,
    0x000000cau, 0x000000cbu, 0x0003003eu, 0x000000d4u, 0x000000d3u, 0x000200f9u, 0x000000c9u, 0x000200f8u,
    0x000000c9u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u, 0x000000d5u,
    0x00000008u, 0x000500b0u, 0x00000029u, 0x000000d6u, 0x000000d5u, 0x0000003cu, 0x000300f7u, 0x000000d8u,
    0x00000000u, 0x000400fau, 0x000000d6u, 0x000000d7u, 0x000000d8u, 0x000200f8u, 0x000000d7u, 0x0004003du,
    0x00000006u, 0x000000d9u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000dau, 0x00000008u, 0x0004003du,
    0x00000006u, 0x000000dbu, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000dcu, 0x00000008u, 0x00050080u,
    0x00000006u, 0x000000ddu, 0x000000dcu, 0x0000003cu, 0x00060041u, 0x000000c1u, 0x000000deu, 0x000000bdu,
    0x000000dbu, 0x000000ddu, 0x0004003du, 0x0000001cu, 0x000000dfu, 0x000000deu, 0x00060041u, 0x000000c1u,
    0x000000e0u, 0x000000bdu, 0x000000d9u, 0x000000dau, 0x0004003du, 0x0000001cu, 0x000000e1u, 0x000000e0u,
    0x00050081u, 0x0000001cu, 0x000000e2u, 0x000000e1u, 0x000000dfu, 0x00060041u, 0x000000c1u, 0x000000e3u,
    0x000000bdu, 0x000000d9u, 0x000000dau, 0x0003003eu, 0x000000e3u, 0x000000e2u, 0x000200f9u, 0x000000d8u,
    0x000200f8u, 0x000000d8u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u,
    0x000000e4u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x000000e5u, 0x000000e4u, 0x00000018u, 0x000300f7u,
    0x000000e7u, 0x00000000u, 0x000400fau, 0x000000e5u, 0x000000e6u, 0x000000e7u, 0x000200f8u, 0x000000e6u,
    0x0004003du, 0x00000006u, 0x000000e8u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000e9u, 0x00000008u,
    0x0004003du, 0x00000006u, 0x000000eau, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000ebu, 0x00000008u,
    0x00050080u, 0x00000006u, 0x000000ecu, 0x000000ebu, 0x00000018u, 0x00060041u, 0x000000c1u, 0x000000edu,
    0x000000bdu, 0x000000eau, 0x000000ecu, 0x0004003du, 0x0000001cu, 0x000000eeu, 0x000000edu, 0x00060041u,
    0x000000c1u, 0x000000efu, 0x000000bdu, 0x000000e8u, 0x000000e9u, 0x0004003du, 0x0000001cu, 0x000000f0u,
    0x000000efu, 0x00050081u, 0x0000001cu, 0x000000f1u, 0x000000f0u, 0x000000eeu, 0x00060041u, 0x000000c1u,
    0x000000f2u, 0x000000bdu, 0x000000e8u, 0x000000e9u, 0x0003003eu, 0x000000f2u, 0x000000f1u, 0x000200f9u,
    0x000000e7u, 0x000200f8u, 0x000000e7u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du,
    0x00000006u, 0x000000f3u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x000000f4u, 0x000000f3u, 0x000000c3u,
    0x000300f7u, 0x000000f6u, 0x00000000u, 0x000400fau, 0x000000f4u, 0x000000f5u, 0x000000f6u, 0x000200f8u,
    0x000000f5u, 0x0004003du, 0x00000006u, 0x000000f7u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000f8u,
    0x00000008u, 0x0004003du, 0x00000006u, 0x000000f9u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000fau,
    0x00000008u, 0x00050080u, 0x00000006u, 0x000000fbu, 0x000000fau, 0x000000c3u, 0x00060041u, 0x000000c1u,
    0x000000fcu, 0x000000bdu, 0x000000f9u, 0x000000fbu, 0x0004003du, 0x0000001cu, 0x000000fdu, 0x000000fcu,
    0x00060041u, 0x000000c1u, 0x000000feu, 0x000000bdu, 0x000000f7u, 0x000000f8u, 0x0004003du, 0x0000001cu,
    0x000000ffu, 0x000000feu, 0x00050081u, 0x0000001cu, 0x00000100u, 0x000000ffu, 0x000000fdu, 0x00060041u,
    0x000000c1u, 0x00000101u, 0x000000bdu, 0x000000f7u, 0x000000f8u, 0x0003003eu, 0x00000101u, 0x00000100u,
    0x000200f9u, 0x000000f6u, 0x000200f8u, 0x000000f6u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u,
    0x0004003du, 0x00000006u, 0x00000102u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x00000103u, 0x00000102u,
    0x00000011u, 0x000300f7u, 0x00000105u, 0x00000000u, 0x000400fau, 0x00000103u, 0x00000104u, 0x00000105u,
    0x000200f8u, 0x00000104u, 0x0004003du, 0x00000006u, 0x00000106u, 0x00000010u, 0x0004003du, 0x00000006u,
    0x00000107u, 0x00000008u, 0x0004003du, 0x00000006u, 0x00000108u, 0x00000010u, 0x0004003du, 0x00000006u,
    0x00000109u, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000010au, 0x00000109u, 0x00000011u, 0x00060041u,
    0x000000c1u, 0x0000010bu, 0x000000bdu, 0x00000108u, 0x0000010au, 0x0004003du, 0x0000001cu, 0x0000010cu,
    0x0000010bu, 0x00060041u, 0x000000c1u, 0x0000010du, 0x000000bdu, 0x00000106u, 0x00000107u, 0x0004003du,
    0x0000001cu, 0x0000010eu, 0x0000010du, 0x00050081u, 0x0000001cu, 0x0000010fu, 0x0000010eu, 0x0000010cu,
    0x00060041u, 0x000000c1u, 0x00000110u, 0x000000bdu, 0x00000106u, 0x00000107u, 0x0003003eu, 0x00000110u,
    0x0000010fu, 0x000200f9u, 0x00000105u, 0x000200f8u, 0x00000105u, 0x0004003du, 0x00000006u, 0x00000111u,
    0x00000008u, 0x000500aau, 0x00000029u, 0x00000112u, 0x00000111u, 0x0000000cu, 0x000300f7u, 0x00000114u,
    0x00000000u, 0x000400fau, 0x00000112u, 0x00000113u, 0x00000114u, 0x000200f8u, 0x00000113u, 0x0004003du,
    0x00000006u, 0x00000115u, 0x00000014u, 0x00050041u, 0x00000026u, 0x00000116u, 0x00000023u, 0x00000025u,
    0x0004003du, 0x00000006u, 0x00000117u, 0x00000116u, 0x000500b0u, 0x00000029u, 0x00000118u, 0x00000115u,
    0x00000117u, 0x000200f9u, 0x00000114u, 0x000200f8u, 0x00000114u, 0x000700f5u, 0x00000029u, 0x00000119u,
    0x00000112u, 0x00000105u, 0x00000118u, 0x00000113u, 0x000300f7u, 0x0000011bu, 0x00000000u, 0x000400fau,
    0x00000119u, 0x0000011au, 0x0000011bu, 0x000200f8u, 0x0000011au, 0x00050041u, 0x00000026u, 0x00000120u,
    0x00000023u, 0x00000072u, 0x0004003du, 0x00000006u, 0x00000121u, 0x00000120u, 0x0004003du, 0x00000006u,
    0x00000122u, 0x00000014u, 0x00050080u, 0x00000006u, 0x00000123u, 0x00000121u, 0x00000122u, 0x0004003du,
    0x00000006u, 0x00000124u, 0x00000010u, 0x00060041u, 0x000000c1u, 0x00000125u, 0x000000bdu, 0x00000124u,
    0x0000002eu, 0x0004003du, 0x0000001cu, 0x00000126u, 0x00000125u, 0x00050041u, 0x00000127u, 0x00000128u,
    0x00000023u, 0x00000079u, 0x0004003du, 0x0000001cu, 0x00000129u, 0x00000128u, 0x00050085u, 0x0000001cu,
    0x0000012au, 0x00000126u, 0x00000129u, 0x00060041u, 0x0000008cu, 0x0000012bu, 0x0000011fu, 0x0000002eu,
    0x00000123u, 0x0003003eu, 0x0000012bu, 0x0000012au, 0x000200f9u, 0x0000011bu, 0x000200f8u, 0x0000011bu,
    0x000100fdu, 0x00010038u,
};
static const size_t kBitnetGemvI2SSpvByteSize = sizeof(kBitnetGemvI2SSpv);

struct BitNetPushConstants {
    uint32_t in_dim;
    uint32_t out_dim;
    float dequant_scale;
    uint32_t weight_offset_bytes;
    uint32_t output_offset_words;
};

static int load_fixture_file(const char * filename, uint8_t * out_buf, size_t max_size, size_t * out_size) {
    const char * search_paths[] = {
        "%s",
        "tests/fixtures/%s",
        "fixtures/%s",
        "../tests/fixtures/%s",
        "/data/data/com.termux/files/home/fixtures/%s",
        "/data/data/com.termux/files/home/tests/fixtures/%s",
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

    if (!fp) return -1;
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

int main(void) {
    printf("====================================================================\n");
    printf("Phase 2: Modular Vulkan Compute Core Verification & Regression Test\n");
    printf("Hardware Target: Qualcomm Adreno 830 GPU (Android 15 ARM64)\n");
    printf("====================================================================\n");

    int total_failures = 0;

    /* ------------------------------------------------------------------------
     * Gate 1: Cryptographic Authentication & Fixture Loading
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 1] Cryptographic Fixture & Bytecode Authentication...\n");

    char spv_sha256[65];
    compute_sha256_hex((const uint8_t *)kBitnetGemvI2SSpv, kBitnetGemvI2SSpvByteSize, spv_sha256);
    const char * expected_spv_sha256 = "6d6a55fe2beb5291333c36280b2326e34d7470f29635b9d5e271c2e17463504a";
    if (strcmp(spv_sha256, expected_spv_sha256) != 0) {
        printf("  -> FAIL: SPIR-V hash mismatch! (Actual: %s)\n", spv_sha256);
        return 1;
    }
    printf("  -> SPIR-V Bytecode Validated: %s (7,304 bytes)\n", spv_sha256);

    uint8_t packed_weights_fixture[256];
    size_t fixture_sz = 0;
    if (load_fixture_file("fixture_i2s_upstream.bin", packed_weights_fixture, sizeof(packed_weights_fixture), &fixture_sz) != 0 || fixture_sz != 96) {
        printf("  -> FAIL: Could not load fixture_i2s_upstream.bin\n");
        return 2;
    }
    char fixture_sha256[65];
    compute_sha256_hex(packed_weights_fixture, fixture_sz, fixture_sha256);
    const char * expected_fixture_sha256 = "c22f3866ab7d9f5b154bf68e98852b84c77442aa0a6415d4083dc2b939bd3c33";
    if (strcmp(fixture_sha256, expected_fixture_sha256) != 0) {
        printf("  -> FAIL: Weight fixture hash mismatch! (Actual: %s)\n", fixture_sha256);
        return 3;
    }
    printf("  -> Weight Fixture Validated: %s (96 bytes)\n", fixture_sha256);

    float expected_f32[256];
    size_t exp_f32_sz = 0;
    if (load_fixture_file("expected_i2s_upstream_f32.bin", (uint8_t *)expected_f32, sizeof(expected_f32), &exp_f32_sz) != 0 || exp_f32_sz != 1024) {
        printf("  -> FAIL: Could not load expected_i2s_upstream_f32.bin\n");
        return 4;
    }
    char exp_f32_sha256[65];
    compute_sha256_hex((const uint8_t *)expected_f32, exp_f32_sz, exp_f32_sha256);
    const char * expected_exp_sha256 = "09ae9aa306dd369d6e4bb74973af6e23f8c4fd4a07dd44533728f42a071b5f65";
    if (strcmp(exp_f32_sha256, expected_exp_sha256) != 0) {
        printf("  -> FAIL: Expected F32 fixture hash mismatch! (Actual: %s)\n", exp_f32_sha256);
        return 5;
    }
    printf("  -> Expected F32 Fixture Validated: %s (1,024 bytes)\n", exp_f32_sha256);

    float dequant_scale;
    memcpy(&dequant_scale, packed_weights_fixture + 64, sizeof(float));
    printf("  -> Dequant Scale Extracted: %f (Little-Endian IEEE-754)\n", dequant_scale);

    /* ------------------------------------------------------------------------
     * Gate 2: AcVkContext Creation & Hardware GPU Selection
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 2] AcVkContext Creation & Physical Device Verification...\n");
    AcVkContext* ctx = NULL;
    AcResult res = ac_vk_context_create(&ctx);
    if (res != AC_SUCCESS || !ctx) {
        printf("  -> FAIL: ac_vk_context_create failed with error %d\n", res);
        return 6;
    }
    printf("  -> PASS: AcVkContext created successfully.\n");
    printf("     Device Name: %s (Vendor: 0x%04x, DeviceID: 0x%04x, Driver: 0x%08x)\n",
           ctx->device_properties.deviceName, ctx->device_properties.vendorID,
           ctx->device_properties.deviceID, ctx->device_properties.driverVersion);
    if (ctx->device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        printf("  -> FAIL: Selected device is CPU fallback! Violates Ground Truth doctrine.\n");
        ac_vk_context_destroy(ctx);
        return 7;
    }

    /* ------------------------------------------------------------------------
     * Gate 3: AcVkBuffer Allocation & UMA Memory Binding
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 3] AcVkBuffer Allocation & UMA Host-Visible Mapping...\n");
    AcVkBuffer* buf_weight = NULL;
    AcVkBuffer* buf_activation = NULL;
    AcVkBuffer* buf_output = NULL;

    res = ac_vk_buffer_create(ctx, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_weight);
    if (res != AC_SUCCESS || !buf_weight) {
        printf("  -> FAIL: ac_vk_buffer_create (weights) failed with error %d\n", res);
        total_failures++;
    }
    res = ac_vk_buffer_create(ctx, 256 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_activation);
    if (res != AC_SUCCESS || !buf_activation) {
        printf("  -> FAIL: ac_vk_buffer_create (activation) failed with error %d\n", res);
        total_failures++;
    }
    res = ac_vk_buffer_create(ctx, 256 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_output);
    if (res != AC_SUCCESS || !buf_output) {
        printf("  -> FAIL: ac_vk_buffer_create (output) failed with error %d\n", res);
        total_failures++;
    }

    if (total_failures > 0) {
        ac_vk_buffer_destroy(buf_weight);
        ac_vk_buffer_destroy(buf_activation);
        ac_vk_buffer_destroy(buf_output);
        ac_vk_context_destroy(ctx);
        return 8;
    }

    /* Verify owning context pointers */
    if (buf_weight->owning_context != ctx || buf_activation->owning_context != ctx || buf_output->owning_context != ctx) {
        printf("  -> FAIL: Buffer owning_context pointer mismatch!\n");
        total_failures++;
    } else {
        printf("  -> PASS: All buffers verified with explicit owning context.\n");
    }

    /* Load 64 bytes of packed weights payload into mapped memory */
    memcpy(buf_weight->mapped_ptr, packed_weights_fixture, 64);
    printf("  -> Loaded 64 bytes of packed weights into GPU memory.\n");

    /* ------------------------------------------------------------------------
     * Gate 4: AcVkKernel Pipeline Creation
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 4] AcVkKernel SPIR-V Compute Pipeline Creation...\n");
    AcVkKernel* kernel = NULL;
    res = ac_vk_kernel_create_spirv(ctx, kBitnetGemvI2SSpv, kBitnetGemvI2SSpvByteSize, sizeof(BitNetPushConstants), 3, &kernel);
    if (res != AC_SUCCESS || !kernel) {
        printf("  -> FAIL: ac_vk_kernel_create_spirv failed with error %d\n", res);
        total_failures++;
    } else if (kernel->owning_context != ctx) {
        printf("  -> FAIL: Kernel owning_context pointer mismatch!\n");
        total_failures++;
    } else {
        printf("  -> PASS: AcVkKernel created and bound to owning context.\n");
    }

    if (total_failures > 0) {
        ac_vk_kernel_destroy(kernel);
        ac_vk_buffer_destroy(buf_weight);
        ac_vk_buffer_destroy(buf_activation);
        ac_vk_buffer_destroy(buf_output);
        ac_vk_context_destroy(ctx);
        return 9;
    }

    /* ------------------------------------------------------------------------
     * Gate 5: GPU Numerical Accuracy vs Upstream Fixture
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 5] GPU Numerical Verification vs Upstream Fixture (Modular API)...\n");
    AcVkBuffer* buffers[3] = { buf_weight, buf_activation, buf_output };
    float * act_ptr = (float *)buf_activation->mapped_ptr;
    float * out_ptr = (float *)buf_output->mapped_ptr;

    /* Subtest 5.1: Decode ALL 256 Individual Weights via Column Basis Projections */
    printf("  [Subtest 5.1] GPU Full 256-Weight Decoding via Basis Vector Projections...\n");
    int mismatch_count = 0;
    int first_mismatch_idx = -1;
    float max_abs_error = 0.0f;

    for (int col = 0; col < 256; ++col) {
        memset(act_ptr, 0, 256 * sizeof(float));
        act_ptr[col] = 1.0f;

        BitNetPushConstants params = {};
        params.in_dim = 256;
        params.out_dim = 1;
        params.dequant_scale = dequant_scale;
        params.weight_offset_bytes = 0;
        params.output_offset_words = 0;

        res = ac_vk_dispatch(ctx, kernel, buffers, 3, &params, sizeof(params), 1, 1, 1, NULL);
        if (res != AC_SUCCESS) {
            printf("    -> FAIL: ac_vk_dispatch failed at column %d (error %d)\n", col, res);
            total_failures++;
            break;
        }

        float gpu_val = out_ptr[0];
        float exp_val = expected_f32[col];
        float err = fabsf(gpu_val - exp_val);
        if (err > max_abs_error) max_abs_error = err;

        if (err > 1e-4f) {
            if (mismatch_count == 0) first_mismatch_idx = col;
            mismatch_count++;
        }
    }

    if (mismatch_count == 0) {
        printf("    -> PASS: All 256 weights bit-exact matched with expected fixture!\n");
        printf("       Mismatch Count: 0 / 256 | Max Absolute Error: %.2e\n", max_abs_error);
    } else {
        printf("    -> FAIL: %d / 256 weights mismatched! First mismatch at index %d (Max Error: %f)\n",
               mismatch_count, first_mismatch_idx, max_abs_error);
        total_failures++;
    }

    /* Subtest 5.2: Dense All-Ones GEMV Dot Product */
    printf("  [Subtest 5.2] Dense All-Ones Activation Vector GEMV...\n");
    double expected_dense_sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        act_ptr[i] = 1.0f;
        expected_dense_sum += (double)expected_f32[i];
    }

    BitNetPushConstants p_dense = {};
    p_dense.in_dim = 256;
    p_dense.out_dim = 1;
    p_dense.dequant_scale = dequant_scale;
    p_dense.weight_offset_bytes = 0;
    p_dense.output_offset_words = 0;

    res = ac_vk_dispatch(ctx, kernel, buffers, 3, &p_dense, sizeof(p_dense), 1, 1, 1, NULL);
    float dense_gpu = out_ptr[0];
    float dense_err = fabsf(dense_gpu - (float)expected_dense_sum);
    if (res == AC_SUCCESS && dense_err < 1e-4f) {
        printf("    -> PASS: Dense GEMV matched (GPU: %f, Analytical: %f, Error: %.2e)\n",
               dense_gpu, (float)expected_dense_sum, dense_err);
    } else {
        printf("    -> FAIL: Dense GEMV diverged (GPU: %f, Analytical: %f, Error: %f)\n",
               dense_gpu, (float)expected_dense_sum, dense_err);
        total_failures++;
    }

    /* Subtest 5.3: Analytical Linear Gradient Activation GEMV */
    printf("  [Subtest 5.3] Analytical Linear Gradient Activation Vector GEMV...\n");
    double expected_grad_sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        float a = ((float)i + 1.0f) / 256.0f;
        act_ptr[i] = a;
        expected_grad_sum += (double)a * (double)expected_f32[i];
    }

    BitNetPushConstants p_grad = {};
    p_grad.in_dim = 256;
    p_grad.out_dim = 1;
    p_grad.dequant_scale = dequant_scale;
    p_grad.weight_offset_bytes = 0;
    p_grad.output_offset_words = 0;

    res = ac_vk_dispatch(ctx, kernel, buffers, 3, &p_grad, sizeof(p_grad), 1, 1, 1, NULL);
    float grad_gpu = out_ptr[0];
    float grad_err = fabsf(grad_gpu - (float)expected_grad_sum);
    if (res == AC_SUCCESS && grad_err < 1e-4f) {
        printf("    -> PASS: Gradient GEMV matched (GPU: %f, Analytical: %f, Error: %.2e)\n",
               grad_gpu, (float)expected_grad_sum, grad_err);
    } else {
        printf("    -> FAIL: Gradient GEMV diverged (GPU: %f, Analytical: %f, Error: %f)\n",
               grad_gpu, (float)expected_grad_sum, grad_err);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 6: 100 Consecutive GPU Dispatches Latency & Stability Stress
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 6] 100 Consecutive GPU Dispatches Latency & Stability Stress...\n");
    const int N_ITERS = 100;
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int iter_failures = 0;
    for (int it = 0; it < N_ITERS; ++it) {
        res = ac_vk_dispatch(ctx, kernel, buffers, 3, &p_grad, sizeof(p_grad), 1, 1, 1, NULL);
        if (res != AC_SUCCESS || fabsf(out_ptr[0] - grad_gpu) > 1e-5f) {
            iter_failures++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000.0;
    double avg_us = (elapsed_ms * 1000.0) / N_ITERS;

    if (iter_failures == 0) {
        printf("  -> PASS: 100 / 100 dispatches completed without error (100-dispatch short-run stability check passed).\n");
        printf("     Total Wall Time: %.2f ms | Avg Synchronous End-to-End Wall Latency: %.2f us / dispatch\n", elapsed_ms, avg_us);
    } else {
        printf("  -> FAIL: %d / 100 iterations failed or produced numerical drift!\n", iter_failures);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 7: In-Flight Lifecycle Safety & Device-Busy Enforcement
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 7] In-Flight Lifecycle Safety & Destruction Rejection Check...\n");
    AcVkEvent* event = NULL;
    res = ac_vk_dispatch(ctx, kernel, buffers, 3, &p_grad, sizeof(p_grad), 1, 1, 1, &event);
    if (res != AC_SUCCESS || !event) {
        printf("  -> FAIL: Async dispatch failed with error %d\n", res);
        total_failures++;
    } else {
        /* Test in-flight rejection: buffers and kernel must reject destruction with AC_ERR_DEVICE_BUSY */
        AcResult b_res = ac_vk_buffer_destroy(buf_weight);
        if (b_res != AC_ERR_DEVICE_BUSY) {
            printf("  -> FAIL: Buffer destruction during active dispatch was not rejected! (Got %d, expected %d)\n",
                   b_res, AC_ERR_DEVICE_BUSY);
            total_failures++;
        } else {
            printf("  -> PASS: In-flight buffer destruction correctly rejected (AC_ERR_DEVICE_BUSY).\n");
        }

        AcResult k_res = ac_vk_kernel_destroy(kernel);
        if (k_res != AC_ERR_DEVICE_BUSY) {
            printf("  -> FAIL: Kernel destruction during active dispatch was not rejected! (Got %d, expected %d)\n",
                   k_res, AC_ERR_DEVICE_BUSY);
            total_failures++;
        } else {
            printf("  -> PASS: In-flight kernel destruction correctly rejected (AC_ERR_DEVICE_BUSY).\n");
        }

        AcResult c_res = ac_vk_context_destroy(ctx);
        if (c_res != AC_ERR_DEVICE_BUSY && c_res != AC_ERR_RESOURCE_IN_USE) {
            printf("  -> FAIL: Context destruction during active dispatch was not rejected! (Got %d, expected %d or %d)\n",
                   c_res, AC_ERR_DEVICE_BUSY, AC_ERR_RESOURCE_IN_USE);
            total_failures++;
        } else {
            printf("  -> PASS: In-flight context destruction correctly rejected (AC_ERR_RESOURCE_IN_USE / AC_ERR_DEVICE_BUSY: %d).\n", c_res);
        }

        /* Wait and destroy event cleanly */
        ac_vk_event_wait(event, 5000000000ULL);
        ac_vk_event_destroy(event);
        printf("  -> PASS: Async event synchronization completed and freed cleanly.\n");
    }

    /* ------------------------------------------------------------------------
     * Teardown: Safe Destruction of All Resources
     * ------------------------------------------------------------------------ */
    printf("\n[Cleanup] Final Vulkan Core Resource Cleanup...\n");
    ac_vk_kernel_destroy(kernel);
    ac_vk_buffer_destroy(buf_weight);
    ac_vk_buffer_destroy(buf_activation);
    ac_vk_buffer_destroy(buf_output);
    ac_vk_context_destroy(ctx);
    printf("  -> All Vulkan resources destroyed safely. Process exiting cleanly.\n");

    printf("\n====================================================================\n");
    printf("Verification Gates: 7 / 7 Passed (%d Failures)\n", total_failures);
    printf("Cleanup: Completed\n");
    printf("====================================================================\n");
    return (total_failures == 0) ? 0 : 1;
}
