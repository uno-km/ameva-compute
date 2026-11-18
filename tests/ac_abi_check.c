/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
 */
/* ac_abi_check: the C ABI contract of libac_runtime.so.2 (docs/abi/AC_RUNTIME_ABI.md).
 *
 *   1. ac_runtime_abi_version() of the loaded library equals the header's AC_ABI_VERSION.
 *   2. AC_ABI_COMPATIBLE: the accepted range of caller versions.
 *   3. ac_runtime_context_create accepts every caller version from AC_ABI_MIN_MINOR up to the runtime's and refuses
 *      a newer minor, another major, a minor below the floor and low-byte bits, with AC_ERR_INVALID_ARGUMENT and a NULL
 *      context. Acceptance creates a real context on the default backend (a GPU is needed).
 *   4. Prints the size and the field offsets of every public struct as "ABI_LAYOUT <struct> <field|size> <bytes>".
 *      scripts/termux/build_runtime_backend.sh compares these lines with docs/abi/ac_runtime_layout.txt: a layout
 *      change that the document and the version do not record fails the build.
 *
 * Prints AC_ABI_CHECK=PASS (n checks) or AC_ABI_CHECK=FAIL; exit status 0 only on PASS.
 * Build: clang -O2 -I include tests/ac_abi_check.c libac_runtime.so.2 -o ac_abi_check
 */
#include "ac_runtime.h"

#include <stddef.h>
#include <stdio.h>

static int g_checks = 0, g_failed = 0;

static void check(int ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        printf("  FAIL: %s\n", what);
    }
}

static void create_with(uint32_t version, int expect_ok) {
    AcContextDesc d;
    d.abi_version = version;
    d.struct_size = (uint32_t)sizeof(AcContextDesc);
    d.backend_kind = AC_BACKEND_VULKAN;
    d.flags = 0;
    AcContext* ctx = (AcContext*)0x1;
    const AcResult r = ac_runtime_context_create(&d, &ctx);
    char what[160];
    if (expect_ok) {
        snprintf(what, sizeof(what), "context_create with caller version 0x%08X = %d (expected AC_SUCCESS)",
                 (unsigned)version, (int)r);
        check(r == AC_SUCCESS && ctx != NULL, what);
        if (r == AC_SUCCESS && ctx) ac_runtime_context_destroy(ctx);
    } else {
        snprintf(what, sizeof(what), "context_create with caller version 0x%08X = %d (expected AC_ERR_INVALID_ARGUMENT)",
                 (unsigned)version, (int)r);
        check(r == AC_ERR_INVALID_ARGUMENT && ctx == NULL, what);
    }
}

#define SIZE(T) printf("ABI_LAYOUT %s size %u\n", #T, (unsigned)sizeof(T))
#define OFF(T, f) printf("ABI_LAYOUT %s %s %u\n", #T, #f, (unsigned)offsetof(T, f))

static void print_layout(void) {
    SIZE(AcDim3); OFF(AcDim3, x); OFF(AcDim3, y); OFF(AcDim3, z);
    SIZE(AcSpecConstant);
    SIZE(AcSpecConstantSchemaEntry);
    SIZE(AcRuntimeBindingDesc);
    SIZE(AcKernelRuntimeMetadata);
    SIZE(AcContextDesc); OFF(AcContextDesc, abi_version); OFF(AcContextDesc, struct_size);
    OFF(AcContextDesc, backend_kind); OFF(AcContextDesc, flags);
    SIZE(AcBufferDesc); OFF(AcBufferDesc, abi_version); OFF(AcBufferDesc, struct_size);
    SIZE(AcKernelDesc); OFF(AcKernelDesc, abi_version); OFF(AcKernelDesc, struct_size);
    SIZE(AcLaunchDesc); OFF(AcLaunchDesc, abi_version); OFF(AcLaunchDesc, struct_size);
    SIZE(AcBatchDesc); OFF(AcBatchDesc, abi_version); OFF(AcBatchDesc, struct_size);
    SIZE(AcBufferBinding);
    SIZE(AcBatchStats); OFF(AcBatchStats, abi_version); OFF(AcBatchStats, struct_size);
    SIZE(AcDeviceLimits); OFF(AcDeviceLimits, abi_version); OFF(AcDeviceLimits, struct_size);
    SIZE(AcKernelSourceDesc); OFF(AcKernelSourceDesc, abi_version); OFF(AcKernelSourceDesc, struct_size);
    OFF(AcKernelSourceDesc, source); OFF(AcKernelSourceDesc, source_size); OFF(AcKernelSourceDesc, build_options);
    OFF(AcKernelSourceDesc, metadata); OFF(AcKernelSourceDesc, flags);
}

int main(void) {
    const uint32_t v = ac_runtime_abi_version();
    printf("AC_ABI header 0x%08X runtime 0x%08X min_minor %u\n", (unsigned)AC_ABI_VERSION, (unsigned)v,
           (unsigned)AC_ABI_MIN_MINOR);
    check(v == AC_ABI_VERSION, "ac_runtime_abi_version() differs from the header's AC_ABI_VERSION");

    check(AC_ABI_COMPATIBLE(0x00020100u), "2.1 caller not compatible");
    check(AC_ABI_COMPATIBLE(0x00020200u), "2.2 caller not compatible");
    check(AC_ABI_COMPATIBLE(0x00020300u), "2.3 caller not compatible");
    check(!AC_ABI_COMPATIBLE(0x00020400u), "2.4 caller (newer than the runtime) compatible");
    check(!AC_ABI_COMPATIBLE(0x00020000u), "2.0 caller (below the floor) compatible");
    check(!AC_ABI_COMPATIBLE(0x00030000u), "3.0 caller compatible");
    check(!AC_ABI_COMPATIBLE(0x00010100u), "1.1 caller compatible");
    check(!AC_ABI_COMPATIBLE(0x00020301u), "low-byte bits accepted");
    check(!AC_ABI_COMPATIBLE(0x99999999u), "0x99999999 accepted");

    create_with(0x00020100u, 1);
    create_with(AC_ABI_VERSION, 1);
    create_with(0x00020400u, 0);
    create_with(0x00020000u, 0);
    create_with(0x00030000u, 0);
    create_with(0x00020301u, 0);

    print_layout();
    printf("AC_ABI_CHECK=%s (%d checks, %d failed)\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
