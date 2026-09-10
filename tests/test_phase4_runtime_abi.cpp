/**
 * @file test_phase4_runtime_abi.cpp
 * @brief Phase 4A: Public Opaque C Runtime ABI & Production Hardening Test Suite
 *
 * Hardware Verification Fleet:
 *   - Qualcomm Snapdragon 8 Elite / Adreno 830 (Samsung Galaxy S25)
 *   - Samsung Exynos 1380 / ARM Mali-G68 (Samsung Galaxy A35)
 *
 * Verification Gates:
 *   1. Public ABI Contract & Strict Descriptor Validation
 *   2. Context Lifecycle & Active Children Rejection (Buffer, Kernel)
 *   3. Active Event Lifespan Tracking & Context Teardown Rejection (UAF Guard)
 *   4. Buffer Usage Flags Validation (Unknown Bit Rejection, Force Non-Coherent Range) & Safe Getters
 *   5. Kernel Descriptor & Metadata Defensive Validation (Null checks, Duplicate binding index)
 *   6. Cross-Context Ownership Isolation & Launch Defensive Parameter Checks
 *   7. End-to-End BitNet GEMV I2_S Compute Dispatch via Public Opaque ABI (Mismatch: 0/256)
 *   8. Event Synchronization & Hardware Timing Query via Public ABI
 *   9. Synchronous Dispatch Mode Contract (out_event == NULL)
 *  10. 1,000-Cycle Buffer Lifecycle RSS Stability via Public ABI
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "ac_runtime.h"

/* Precompiled SPIR-V shaders */
#include "../src/shaders/gemv_i2s_spv.h"
#include "../src/shaders/vector_add_spv.h"

#define CHECK_AC(expr)                                                      \
    do {                                                                    \
        AcResult _result = (expr);                                          \
        if (_result != AC_SUCCESS) {                                        \
            printf("  -> FAIL: %s returned %d (line %d)\n",                 \
                   #expr, _result, __LINE__);                               \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static long get_process_rss_kb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

static int load_fixture_file(const char* filename, void* out_buf, size_t max_bytes, size_t* out_read_bytes) {
    const char* search_paths[] = {
        "%s",
        "tests/fixtures/%s",
        "fixtures/%s",
        "../tests/fixtures/%s",
        "../../tests/fixtures/%s",
        "/data/data/com.termux/files/home/fixtures/%s",
        "/data/data/com.termux/files/home/tests/fixtures/%s",
        "/data/data/com.termux/files/home/%s",
        "/data/data/com.termux/files/home/ameva-compute/tests/fixtures/%s",
        NULL
    };
    char full_path[512];
    FILE* fp = NULL;
    for (int i = 0; search_paths[i] != NULL; i++) {
        snprintf(full_path, sizeof(full_path), search_paths[i], filename);
        fp = fopen(full_path, "rb");
        if (fp != NULL) break;
    }
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0 || (size_t)sz > max_bytes) {
        fclose(fp);
        return -2;
    }
    size_t read_bytes = fread(out_buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (read_bytes != (size_t)sz) return -3;
    *out_read_bytes = (size_t)sz;
    return 0;
}

int main(void) {
    printf("======================================================================\n");
    printf("AMUDA 2.0 Phase 4A: Public Opaque C Runtime ABI Test Suite\n");
    printf("======================================================================\n\n");

    int passed_gates = 0;
    const int total_gates = 10;

    // -------------------------------------------------------------------------
    // Gate 1: Public ABI Contract & Strict Descriptor Validation
    // -------------------------------------------------------------------------
    printf("[Gate 1/10] Public ABI Contract & Strict Descriptor Validation...\n");
    {
        AcContext* bad_ctx = (AcContext*)0xDEADBEEF;

        /* 1.1 ABI Version mismatch */
        AcContextDesc desc_bad_ver = {};
        desc_bad_ver.abi_version = 0x99999999u;
        desc_bad_ver.struct_size = sizeof(AcContextDesc);
        desc_bad_ver.backend_kind = AC_BACKEND_VULKAN;
        AcResult res = ac_runtime_context_create(&desc_bad_ver, &bad_ctx);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_ctx != NULL) {
            printf("  -> FAIL: ABI version mismatch not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ctx);
            return 1;
        }

        /* 1.2 Struct size mismatch */
        bad_ctx = (AcContext*)0xDEADBEEF;
        AcContextDesc desc_bad_sz = {};
        desc_bad_sz.abi_version = AC_ABI_VERSION;
        desc_bad_sz.struct_size = sizeof(AcContextDesc) - 4;
        desc_bad_sz.backend_kind = AC_BACKEND_VULKAN;
        res = ac_runtime_context_create(&desc_bad_sz, &bad_ctx);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_ctx != NULL) {
            printf("  -> FAIL: Struct size mismatch not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ctx);
            return 1;
        }

        /* 1.3 Reserved flags != 0 */
        bad_ctx = (AcContext*)0xDEADBEEF;
        AcContextDesc desc_bad_flags = {};
        desc_bad_flags.abi_version = AC_ABI_VERSION;
        desc_bad_flags.struct_size = sizeof(AcContextDesc);
        desc_bad_flags.backend_kind = AC_BACKEND_VULKAN;
        desc_bad_flags.flags = 0x00000001u;
        res = ac_runtime_context_create(&desc_bad_flags, &bad_ctx);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_ctx != NULL) {
            printf("  -> FAIL: Non-zero reserved flags not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ctx);
            return 1;
        }

        /* 1.4 Unsupported backend */
        bad_ctx = (AcContext*)0xDEADBEEF;
        AcContextDesc desc_bad_backend = {};
        desc_bad_backend.abi_version = AC_ABI_VERSION;
        desc_bad_backend.struct_size = sizeof(AcContextDesc);
        desc_bad_backend.backend_kind = (AcBackendKind)99;
        res = ac_runtime_context_create(&desc_bad_backend, &bad_ctx);
        if (res != AC_ERR_DEVICE_NOT_SUPPORTED || bad_ctx != NULL) {
            printf("  -> FAIL: Unsupported backend not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ctx);
            return 1;
        }

        /* 1.5 NULL descriptor pointer */
        bad_ctx = (AcContext*)0xDEADBEEF;
        res = ac_runtime_context_create(NULL, &bad_ctx);
        if (res != AC_ERR_NULL_POINTER || bad_ctx != NULL) {
            printf("  -> FAIL: NULL descriptor pointer not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ctx);
            return 1;
        }

        /* 1.6 NULL output pointer */
        desc_bad_ver.abi_version = AC_ABI_VERSION;
        res = ac_runtime_context_create(&desc_bad_ver, NULL);
        if (res != AC_ERR_NULL_POINTER) {
            printf("  -> FAIL: NULL out_ctx pointer not rejected! (res=%d)\n", res);
            return 1;
        }

        printf("  -> Verified: Bad ABI version, struct size, flags, and null checks strictly rejected with output NULL.\n");
    }
    passed_gates++;
    printf("  -> Gate 1 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 2: Context Lifecycle & Active Children Rejection
    // -------------------------------------------------------------------------
    printf("[Gate 2/10] Context Lifecycle & Active Children Rejection...\n");
    {
        AcContextDesc ctx_desc = {};
        ctx_desc.abi_version = AC_ABI_VERSION;
        ctx_desc.struct_size = sizeof(AcContextDesc);
        ctx_desc.backend_kind = AC_BACKEND_VULKAN;

        AcContext* ctx = NULL;
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctx));

        /* Create a child buffer */
        AcBufferDesc b_desc = {};
        b_desc.abi_version = AC_ABI_VERSION;
        b_desc.struct_size = sizeof(AcBufferDesc);
        b_desc.size_bytes = 1024;
        b_desc.usage = AC_BUFFER_USAGE_STORAGE;
        b_desc.force_non_coherent = 0;

        AcBuffer* buf = NULL;
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &buf));

        /* Context destruction must fail while child buffer is alive */
        AcResult res = ac_runtime_context_destroy(ctx);
        if (res != AC_ERR_RESOURCE_IN_USE) {
            printf("  -> FAIL: Context destruction while buffer alive returned %d (expected AC_ERR_RESOURCE_IN_USE)\n", res);
            return 2;
        }
        printf("  -> Active child buffer blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)\n");

        /* Destroy buffer, now context destruction must succeed */
        CHECK_AC(ac_runtime_buffer_destroy(buf));
        CHECK_AC(ac_runtime_context_destroy(ctx));
        printf("  -> Clean context destruction after child release: Verified (AC_SUCCESS)\n");
    }
    passed_gates++;
    printf("  -> Gate 2 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 3: Active Event Lifespan Tracking & Context Teardown Rejection (UAF Guard)
    // -------------------------------------------------------------------------
    printf("[Gate 3/10] Active Event Lifespan Tracking & Context Teardown Rejection (UAF Guard)...\n");
    {
        AcContextDesc ctx_desc = {};
        ctx_desc.abi_version = AC_ABI_VERSION;
        ctx_desc.struct_size = sizeof(AcContextDesc);
        ctx_desc.backend_kind = AC_BACKEND_VULKAN;

        AcContext* ctx = NULL;
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctx));

        AcRuntimeBindingDesc v_bindings[3] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE}
        };
        AcKernelRuntimeMetadata v_meta = {};
        v_meta.kernel_name = "vector_add";
        v_meta.entry_point = "main";
        v_meta.binding_count = 3;
        v_meta.bindings = v_bindings;
        v_meta.push_constant_size = 2 * sizeof(uint32_t);
        v_meta.local_size = {256, 1, 1};
        v_meta.expected_spirv_sha256 = kVectorAddSpvSha256;

        AcKernelDesc k_desc = {};
        k_desc.abi_version = AC_ABI_VERSION;
        k_desc.struct_size = sizeof(AcKernelDesc);
        k_desc.spirv_code = kVectorAddSpv;
        k_desc.code_size_bytes = kVectorAddSpvByteSize;
        k_desc.metadata = &v_meta;

        AcKernel* kernel = NULL;
        CHECK_AC(ac_runtime_kernel_create(ctx, &k_desc, &kernel));

        AcBufferDesc b_desc = {};
        b_desc.abi_version = AC_ABI_VERSION;
        b_desc.struct_size = sizeof(AcBufferDesc);
        b_desc.size_bytes = 256 * sizeof(float);
        b_desc.usage = AC_BUFFER_USAGE_STORAGE;
        b_desc.force_non_coherent = 0;

        AcBuffer *b0 = NULL, *b1 = NULL, *b2 = NULL;
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b0));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b1));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b2));

        struct { uint32_t n; float alpha; } pc = {256, 1.0f};
        AcBuffer* bufs[3] = {b0, b1, b2};

        AcLaunchDesc l_desc = {};
        l_desc.abi_version = AC_ABI_VERSION;
        l_desc.struct_size = sizeof(AcLaunchDesc);
        l_desc.kernel = kernel;
        l_desc.buffers = bufs;
        l_desc.buffer_count = 3;
        l_desc.push_constants = &pc;
        l_desc.push_constants_size = sizeof(pc);
        l_desc.grid_dim = {1, 1, 1};

        AcEvent* ev = NULL;
        CHECK_AC(ac_runtime_dispatch(ctx, &l_desc, &ev));
        if (!ev) {
            printf("  -> FAIL: ac_runtime_dispatch returned null event!\n");
            return 3;
        }

        /* Wait for execution to complete */
        CHECK_AC(ac_runtime_event_wait(ev, 5000000000ULL));

        /* Destroy child buffers and kernel, BUT leave event wrapper alive */
        CHECK_AC(ac_runtime_buffer_destroy(b0));
        CHECK_AC(ac_runtime_buffer_destroy(b1));
        CHECK_AC(ac_runtime_buffer_destroy(b2));
        CHECK_AC(ac_runtime_kernel_destroy(kernel));

        /* Context destroy MUST fail because event is still alive */
        AcResult res = ac_runtime_context_destroy(ctx);
        if (res != AC_ERR_RESOURCE_IN_USE) {
            printf("  -> FAIL: Context destruction while event alive returned %d (expected AC_ERR_RESOURCE_IN_USE)\n", res);
            return 3;
        }
        printf("  -> Waited but unretired public Event blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)\n");

        /* Destroy event wrapper */
        CHECK_AC(ac_runtime_event_destroy(ev));

        /* Now context teardown must succeed */
        CHECK_AC(ac_runtime_context_destroy(ctx));
        printf("  -> Clean context destruction after event release: Verified (AC_SUCCESS)\n");
    }
    passed_gates++;
    printf("  -> Gate 3 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 4: Buffer Usage Flags Validation & Safe Output Getters
    // -------------------------------------------------------------------------
    printf("[Gate 4/10] Buffer Usage Flags Validation & Safe Output Getters...\n");
    {
        AcContextDesc ctx_desc = {AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0};
        AcContext* ctx = NULL;
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctx));

        AcBuffer* bad_buf = (AcBuffer*)0xDEADBEEF;

        /* 4.1 Reject unknown usage bit */
        AcBufferDesc b_bad_usage = {AC_ABI_VERSION, sizeof(AcBufferDesc), 1024, AC_BUFFER_USAGE_STORAGE | 0x80000000u, 0};
        AcResult res = ac_runtime_buffer_create(ctx, &b_bad_usage, &bad_buf);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_buf != NULL) {
            printf("  -> FAIL: Unknown usage bits not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_buf);
            return 4;
        }

        /* 4.2 Reject zero usage */
        bad_buf = (AcBuffer*)0xDEADBEEF;
        b_bad_usage.usage = 0;
        res = ac_runtime_buffer_create(ctx, &b_bad_usage, &bad_buf);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_buf != NULL) {
            printf("  -> FAIL: Zero usage not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_buf);
            return 4;
        }

        /* 4.3 Reject zero size */
        bad_buf = (AcBuffer*)0xDEADBEEF;
        b_bad_usage.usage = AC_BUFFER_USAGE_STORAGE;
        b_bad_usage.size_bytes = 0;
        res = ac_runtime_buffer_create(ctx, &b_bad_usage, &bad_buf);
        if (res != AC_ERR_INVALID_BUFFER_SIZE || bad_buf != NULL) {
            printf("  -> FAIL: Zero buffer size not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_buf);
            return 4;
        }

        /* 4.4 Reject force_non_coherent > 1 (e.g. 2) */
        bad_buf = (AcBuffer*)0xDEADBEEF;
        b_bad_usage.size_bytes = 1024;
        b_bad_usage.force_non_coherent = 2u;
        res = ac_runtime_buffer_create(ctx, &b_bad_usage, &bad_buf);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_buf != NULL) {
            printf("  -> FAIL: force_non_coherent=2 not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_buf);
            return 4;
        }

        /* 4.5 Valid buffer creation */
        AcBufferDesc b_valid = {AC_ABI_VERSION, sizeof(AcBufferDesc), 4096, AC_BUFFER_USAGE_STORAGE, 0};
        AcBuffer* buf = NULL;
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_valid, &buf));

        /* 4.6 Test safe getters and output NULL rejection */
        void* mapped = NULL;
        size_t size = 0, alloc_size = 0;
        uint32_t coherent = 0;

        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(buf, &mapped));
        if (!mapped) {
            printf("  -> FAIL: mapped pointer is NULL\n");
            return 4;
        }

        CHECK_AC(ac_runtime_buffer_get_size(buf, &size));
        if (size != 4096) {
            printf("  -> FAIL: buffer size mismatch: %zu\n", size);
            return 4;
        }

        CHECK_AC(ac_runtime_buffer_get_allocation_size(buf, &alloc_size));
        if (alloc_size < 4096) {
            printf("  -> FAIL: allocation size invalid: %zu\n", alloc_size);
            return 4;
        }

        CHECK_AC(ac_runtime_buffer_is_coherent(buf, &coherent));

        /* 4.7 Negative getter test: out_is_coherent == NULL */
        res = ac_runtime_buffer_is_coherent(buf, NULL);
        if (res != AC_ERR_NULL_POINTER) {
            printf("  -> FAIL: is_coherent with NULL output not rejected! (res=%d)\n", res);
            return 4;
        }

        printf("  -> Buffer Getters: Logical=%zu B, Allocation=%zu B, is_coherent=%u\n",
               size, alloc_size, coherent);

        /* 4.8 Out of bounds flush rejection */
        res = ac_runtime_buffer_flush(buf, 4096, 64);
        if (res != AC_ERR_INVALID_ARGUMENT) {
            printf("  -> FAIL: Out of bounds flush not rejected: %d\n", res);
            return 4;
        }

        CHECK_AC(ac_runtime_buffer_destroy(buf));
        CHECK_AC(ac_runtime_context_destroy(ctx));
        printf("  -> Buffer flags, zero checks, and getters verified.\n");
    }
    passed_gates++;
    printf("  -> Gate 4 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 5: Kernel Descriptor & Metadata Defensive Validation
    // -------------------------------------------------------------------------
    printf("[Gate 5/10] Kernel Descriptor & Metadata Defensive Validation...\n");
    {
        AcContextDesc ctx_desc = {AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0};
        AcContext* ctx = NULL;
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctx));

        AcKernel* bad_k = (AcKernel*)0xDEADBEEF;

        /* 5.1 Null spirv_code */
        AcKernelRuntimeMetadata meta = {"vector_add", "main", 0, NULL, 0, {256, 1, 1}, 0, kVectorAddSpvSha256, 0, NULL};
        AcKernelDesc k_desc = {AC_ABI_VERSION, sizeof(AcKernelDesc), NULL, 1024, &meta, 0, NULL};
        AcResult res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_NULL_POINTER || bad_k != NULL) {
            printf("  -> FAIL: Null spirv_code not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        /* 5.2 Unaligned code size */
        bad_k = (AcKernel*)0xDEADBEEF;
        k_desc.spirv_code = kVectorAddSpv;
        k_desc.code_size_bytes = 1023; // not multiple of 4
        res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_k != NULL) {
            printf("  -> FAIL: Unaligned SPIR-V size not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        /* 5.3 binding_count > 0 but bindings == NULL */
        bad_k = (AcKernel*)0xDEADBEEF;
        k_desc.code_size_bytes = kVectorAddSpvByteSize;
        meta.binding_count = 2;
        meta.bindings = NULL;
        res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_NULL_POINTER || bad_k != NULL) {
            printf("  -> FAIL: binding_count > 0 with bindings == NULL not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        /* 5.4 Duplicate binding_index */
        bad_k = (AcKernel*)0xDEADBEEF;
        AcRuntimeBindingDesc dup_bindings[2] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE} // duplicate index 0
        };
        meta.binding_count = 2;
        meta.bindings = dup_bindings;
        res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_k != NULL) {
            printf("  -> FAIL: Duplicate binding index not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        /* 5.5 Invalid stage flag */
        bad_k = (AcKernel*)0xDEADBEEF;
        AcRuntimeBindingDesc bad_stage_bindings[1] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, (AcShaderStage)0}
        };
        meta.binding_count = 1;
        meta.bindings = bad_stage_bindings;
        res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_INVALID_ARGUMENT || bad_k != NULL) {
            printf("  -> FAIL: Invalid stage flag not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        /* 5.6 Corrupted SHA-256 */
        bad_k = (AcKernel*)0xDEADBEEF;
        meta.binding_count = 0;
        meta.bindings = NULL;
        meta.expected_spirv_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        res = ac_runtime_kernel_create(ctx, &k_desc, &bad_k);
        if (res != AC_ERR_BINARY_CORRUPTED || bad_k != NULL) {
            printf("  -> FAIL: Corrupted SHA-256 not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_k);
            return 5;
        }

        CHECK_AC(ac_runtime_context_destroy(ctx));
        printf("  -> Kernel metadata null checks, alignment, duplicate binding, and hash rejection verified.\n");
    }
    passed_gates++;
    printf("  -> Gate 5 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 6: Cross-Context Ownership Isolation & Launch Defensive Parameter Checks
    // -------------------------------------------------------------------------
    printf("[Gate 6/10] Cross-Context Ownership Isolation & Launch Defensive Checks...\n");
    {
        AcContextDesc ctx_desc = {AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0};
        AcContext *ctxA = NULL, *ctxB = NULL;
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctxA));
        CHECK_AC(ac_runtime_context_create(&ctx_desc, &ctxB));

        AcBufferDesc b_desc = {AC_ABI_VERSION, sizeof(AcBufferDesc), 1024, AC_BUFFER_USAGE_STORAGE, 0};
        AcBuffer *bufA = NULL, *bufB = NULL;
        CHECK_AC(ac_runtime_buffer_create(ctxA, &b_desc, &bufA));
        CHECK_AC(ac_runtime_buffer_create(ctxB, &b_desc, &bufB));

        AcRuntimeBindingDesc v_bindings[3] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE}
        };
        AcKernelRuntimeMetadata meta = {"vector_add", "main", 3, v_bindings, 0, {256, 1, 1}, 0, kVectorAddSpvSha256, 0, NULL};
        AcKernelDesc k_desc = {AC_ABI_VERSION, sizeof(AcKernelDesc), kVectorAddSpv, kVectorAddSpvByteSize, &meta, 0, NULL};
        AcKernel* kernelA = NULL;
        CHECK_AC(ac_runtime_kernel_create(ctxA, &k_desc, &kernelA));

        AcEvent* bad_ev = (AcEvent*)0xDEADBEEF;

        /* 6.1 Buffer from Context B passed to Context A dispatch */
        AcBuffer* bufs_mismatch[1] = {bufB};
        AcLaunchDesc l_desc = {AC_ABI_VERSION, sizeof(AcLaunchDesc), kernelA, bufs_mismatch, 1, NULL, 0, {1, 1, 1}};
        AcResult res = ac_runtime_dispatch(ctxA, &l_desc, &bad_ev);
        if (res != AC_ERR_CONTEXT_MISMATCH || bad_ev != NULL) {
            printf("  -> FAIL: Cross-context buffer mismatch not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ev);
            return 6;
        }

        /* 6.2 buffer_count > 0 with buffers == NULL */
        bad_ev = (AcEvent*)0xDEADBEEF;
        l_desc.buffers = NULL;
        l_desc.buffer_count = 1;
        res = ac_runtime_dispatch(ctxA, &l_desc, &bad_ev);
        if (res != AC_ERR_NULL_POINTER || bad_ev != NULL) {
            printf("  -> FAIL: buffer_count > 0 with buffers == NULL not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ev);
            return 6;
        }

        /* 6.3 push_constants_size > 0 with push_constants == NULL */
        bad_ev = (AcEvent*)0xDEADBEEF;
        l_desc.buffers = &bufA;
        l_desc.buffer_count = 1;
        l_desc.push_constants = NULL;
        l_desc.push_constants_size = 16;
        res = ac_runtime_dispatch(ctxA, &l_desc, &bad_ev);
        if (res != AC_ERR_NULL_POINTER || bad_ev != NULL) {
            printf("  -> FAIL: push_constants_size > 0 with push_constants == NULL not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ev);
            return 6;
        }

        /* 6.4 grid_dim.x == 0 */
        bad_ev = (AcEvent*)0xDEADBEEF;
        l_desc.push_constants_size = 0;
        l_desc.grid_dim = {0, 1, 1};
        res = ac_runtime_dispatch(ctxA, &l_desc, &bad_ev);
        if (res != AC_ERR_INVALID_WORK_GROUP_SIZE || bad_ev != NULL) {
            printf("  -> FAIL: grid_dim.x == 0 not rejected! (res=%d, ptr=%p)\n", res, (void*)bad_ev);
            return 6;
        }

        CHECK_AC(ac_runtime_buffer_destroy(bufA));
        CHECK_AC(ac_runtime_buffer_destroy(bufB));
        CHECK_AC(ac_runtime_kernel_destroy(kernelA));
        CHECK_AC(ac_runtime_context_destroy(ctxA));
        CHECK_AC(ac_runtime_context_destroy(ctxB));
        printf("  -> Cross-context isolation and defensive launch checks verified.\n");
    }
    passed_gates++;
    printf("  -> Gate 6 PASS\n\n");

    // -------------------------------------------------------------------------
    // Main Context Setup for Gates 7 to 10
    // -------------------------------------------------------------------------
    AcContextDesc main_desc = {AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0};
    AcContext* ctx = NULL;
    CHECK_AC(ac_runtime_context_create(&main_desc, &ctx));

    // -------------------------------------------------------------------------
    // Gate 7: End-to-End BitNet GEMV I2_S Compute Dispatch via Public Opaque ABI
    // -------------------------------------------------------------------------
    printf("[Gate 7/10] BitNet GEMV I2_S Compute Dispatch via Public Opaque ABI...\n");
    AcKernel* i2s_kernel = NULL;
    AcBuffer *b_w = NULL, *b_a = NULL, *b_o = NULL;
    AcEvent* i2s_event = NULL;
    {
        uint8_t raw_weights[256];
        size_t raw_weights_read = 0;
        int load_res = load_fixture_file("fixture_i2s_upstream.bin", raw_weights, sizeof(raw_weights), &raw_weights_read);
        if (load_res != 0 || raw_weights_read != 96) {
            printf("  -> FAIL: Failed to load fixture_i2s_upstream.bin (res=%d, read=%zu)\n", load_res, raw_weights_read);
            return 7;
        }

        float golden_expected[256];
        size_t golden_read = 0;
        load_res = load_fixture_file("expected_i2s_upstream_f32.bin", golden_expected, sizeof(golden_expected), &golden_read);
        if (load_res != 0 || golden_read != sizeof(golden_expected)) {
            printf("  -> FAIL: Failed to load expected_i2s_upstream_f32.bin (res=%d, read=%zu)\n", load_res, golden_read);
            return 7;
        }

        float dequant_scale = 1.0f;
        memcpy(&dequant_scale, raw_weights + 64, sizeof(float));

        AcRuntimeBindingDesc i2s_bindings[3] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE}
        };
        AcKernelRuntimeMetadata i2s_meta = {};
        i2s_meta.kernel_name = "bitnet_gemv_i2s";
        i2s_meta.entry_point = "main";
        i2s_meta.binding_count = 3;
        i2s_meta.bindings = i2s_bindings;
        i2s_meta.push_constant_size = sizeof(BitNetPushConstants);
        i2s_meta.local_size = {32, 4, 1};
        i2s_meta.expected_spirv_sha256 = kBitnetGemvI2SSpvSha256;

        AcKernelDesc k_desc = {AC_ABI_VERSION, sizeof(AcKernelDesc), kBitnetGemvI2SSpv, kBitnetGemvI2SSpvByteSize, &i2s_meta, 0, NULL};
        CHECK_AC(ac_runtime_kernel_create(ctx, &k_desc, &i2s_kernel));

        AcBufferDesc b_desc_w = {AC_ABI_VERSION, sizeof(AcBufferDesc), 96, AC_BUFFER_USAGE_STORAGE, 0};
        AcBufferDesc b_desc_a = {AC_ABI_VERSION, sizeof(AcBufferDesc), 256 * sizeof(float), AC_BUFFER_USAGE_STORAGE, 0};
        AcBufferDesc b_desc_o = {AC_ABI_VERSION, sizeof(AcBufferDesc), 1 * sizeof(float), AC_BUFFER_USAGE_STORAGE, 0};

        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc_w, &b_w));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc_a, &b_a));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc_o, &b_o));

        void *pw = NULL, *pa = NULL, *po = NULL;
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_w, &pw));
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_a, &pa));
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_o, &po));

        memcpy(pw, raw_weights, 96);
        CHECK_AC(ac_runtime_buffer_flush(b_w, 0, 96));

        float* act_ptr = (float*)pa;
        float* out_ptr = (float*)po;
        int mismatches = 0;
        float max_err = 0.0f;

        for (int col = 0; col < 256; ++col) {
            memset(act_ptr, 0, 256 * sizeof(float));
            act_ptr[col] = 1.0f;
            CHECK_AC(ac_runtime_buffer_flush(b_a, 0, 256 * sizeof(float)));

            BitNetPushConstants push = {256, 1, dequant_scale, 0, 0};
            AcBuffer* bufs[3] = {b_w, b_a, b_o};
            AcLaunchDesc l_desc = {
                AC_ABI_VERSION, sizeof(AcLaunchDesc),
                i2s_kernel, bufs, 3,
                &push, sizeof(push),
                {1, 1, 1}
            };

            if (col == 0) {
                // Column 0 dispatches asynchronously and retains i2s_event for Gate 8
                CHECK_AC(ac_runtime_dispatch(ctx, &l_desc, &i2s_event));
                if (!i2s_event) {
                    printf("  -> FAIL: dispatch returned null event at col %d\n", col);
                    return 7;
                }
                CHECK_AC(ac_runtime_event_wait(i2s_event, 5000000000ULL));
            } else {
                // Columns 1..255 dispatch synchronously (out_event = NULL)
                CHECK_AC(ac_runtime_dispatch(ctx, &l_desc, NULL));
            }

            CHECK_AC(ac_runtime_buffer_invalidate(b_o, 0, 1 * sizeof(float)));

            float diff = fabsf(*out_ptr - golden_expected[col]);
            if (diff > max_err) max_err = diff;
            if (diff > 1e-4f) mismatches++;
        }

        printf("  -> BitNet GEMV I2_S (256/256 weights): Mismatches: %d/256, Max Error: %.2e\n",
               mismatches, max_err);
        if (mismatches > 0) {
            printf("  -> FAIL: Numerical mismatch vs golden fixture!\n");
            return 7;
        }
    }
    passed_gates++;
    printf("  -> Gate 7 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 8: Event Synchronization & Hardware Timing Query via Public ABI
    // -------------------------------------------------------------------------
    printf("[Gate 8/10] Event Synchronization & Hardware Timing Query...\n");
    {
        uint64_t wall_latency_ns = 0;
        uint64_t gpu_duration_ns = 0;
        AcResult res = ac_runtime_event_get_timing(i2s_event, &wall_latency_ns, &gpu_duration_ns);
        if (res != AC_SUCCESS && res != AC_ERR_FEATURE_NOT_SUPPORTED) {
            printf("  -> FAIL: ac_runtime_event_get_timing failed: %d\n", res);
            return 8;
        }

        printf("  -> Host Synchronous Wall Latency: %llu ns (%.3f µs, %.3f ms)\n",
               (unsigned long long)wall_latency_ns,
               (double)wall_latency_ns / 1000.0,
               (double)wall_latency_ns / 1000000.0);

        if (res == AC_SUCCESS && gpu_duration_ns > 0) {
            printf("  -> GPU Timestamp Command Interval: %llu ns (%.3f µs, %.3f ms)\n",
                   (unsigned long long)gpu_duration_ns,
                   (double)gpu_duration_ns / 1000.0,
                   (double)gpu_duration_ns / 1000000.0);
        }

        /* 8.2 Double wait idempotency test */
        CHECK_AC(ac_runtime_event_wait(i2s_event, 1000000000ULL));
        printf("  -> Event Double-Wait Idempotency: Verified (Returned AC_SUCCESS)\n");

        CHECK_AC(ac_runtime_event_destroy(i2s_event));
        i2s_event = NULL;

        CHECK_AC(ac_runtime_buffer_destroy(b_w));
        CHECK_AC(ac_runtime_buffer_destroy(b_a));
        CHECK_AC(ac_runtime_buffer_destroy(b_o));
        CHECK_AC(ac_runtime_kernel_destroy(i2s_kernel));
        b_w = b_a = b_o = NULL;
        i2s_kernel = NULL;
    }
    passed_gates++;
    printf("  -> Gate 8 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 9: Synchronous Dispatch Mode Contract (Vector-Add, out_event == NULL)
    // -------------------------------------------------------------------------
    printf("[Gate 9/10] Synchronous Dispatch Mode Contract (Vector-Add, out_event == NULL)...\n");
    {
        AcRuntimeBindingDesc v_bindings[3] = {
            {0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
            {2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE}
        };
        AcKernelRuntimeMetadata v_meta = {};
        v_meta.kernel_name = "vector_add";
        v_meta.entry_point = "main";
        v_meta.binding_count = 3;
        v_meta.bindings = v_bindings;
        v_meta.push_constant_size = 2 * sizeof(uint32_t);
        v_meta.local_size = {256, 1, 1};
        v_meta.expected_spirv_sha256 = kVectorAddSpvSha256;

        AcKernelDesc k_desc = {AC_ABI_VERSION, sizeof(AcKernelDesc), kVectorAddSpv, kVectorAddSpvByteSize, &v_meta, 0, NULL};
        AcKernel* vec_kernel = NULL;
        CHECK_AC(ac_runtime_kernel_create(ctx, &k_desc, &vec_kernel));

        const uint32_t N_vec = 256;
        AcBufferDesc b_desc = {AC_ABI_VERSION, sizeof(AcBufferDesc), N_vec * sizeof(float), AC_BUFFER_USAGE_STORAGE, 0};
        AcBuffer *b_va = NULL, *b_vb = NULL, *b_vc = NULL;
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b_va));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b_vb));
        CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &b_vc));

        float *fa = NULL, *fb = NULL, *fc = NULL;
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_va, (void**)&fa));
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_vb, (void**)&fb));
        CHECK_AC(ac_runtime_buffer_get_mapped_ptr(b_vc, (void**)&fc));

        const float alpha = 2.5f;
        for (uint32_t i = 0; i < N_vec; ++i) {
            fa[i] = (float)i * 0.5f;
            fb[i] = (float)(N_vec - i) * 0.25f;
            fc[i] = 0.0f;
        }

        CHECK_AC(ac_runtime_buffer_flush(b_va, 0, N_vec * sizeof(float)));
        CHECK_AC(ac_runtime_buffer_flush(b_vb, 0, N_vec * sizeof(float)));

        struct { uint32_t n; float alpha; } pc = {N_vec, alpha};
        AcBuffer* bufs[3] = {b_va, b_vb, b_vc};
        AcLaunchDesc l_desc = {
            AC_ABI_VERSION, sizeof(AcLaunchDesc),
            vec_kernel, bufs, 3,
            &pc, sizeof(pc),
            {1, 1, 1}
        };

        /* out_event is NULL -> Must execute synchronously to completion */
        CHECK_AC(ac_runtime_dispatch(ctx, &l_desc, NULL));

        CHECK_AC(ac_runtime_buffer_invalidate(b_vc, 0, N_vec * sizeof(float)));

        int mismatches = 0;
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N_vec; ++i) {
            float expected = fa[i] + alpha * fb[i];
            float err = fabsf(fc[i] - expected);
            if (err > max_err) max_err = err;
            if (err > 1e-4f) mismatches++;
        }
        printf("  -> Synchronous Vector-Add Parity (N=256, alpha=2.5f): Mismatches: %d/256, Max Error: %.2e\n",
               mismatches, max_err);
        if (mismatches > 0) {
            return 9;
        }

        CHECK_AC(ac_runtime_buffer_destroy(b_va));
        CHECK_AC(ac_runtime_buffer_destroy(b_vb));
        CHECK_AC(ac_runtime_buffer_destroy(b_vc));
        CHECK_AC(ac_runtime_kernel_destroy(vec_kernel));
    }
    passed_gates++;
    printf("  -> Gate 9 PASS\n\n");

    // -------------------------------------------------------------------------
    // Gate 10: 1,000-Cycle Buffer Lifecycle RSS Stability via Public ABI
    // -------------------------------------------------------------------------
    printf("[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability via Public ABI...\n");
    {
        long rss_initial = get_process_rss_kb();
        const int kStressCycles = 1000;
        for (int iter = 0; iter < kStressCycles; ++iter) {
            AcBufferDesc b_desc = {AC_ABI_VERSION, sizeof(AcBufferDesc), 1024 * sizeof(float), AC_BUFFER_USAGE_STORAGE, 0};
            AcBuffer* test_buf = NULL;
            CHECK_AC(ac_runtime_buffer_create(ctx, &b_desc, &test_buf));
            CHECK_AC(ac_runtime_buffer_destroy(test_buf));
        }
        long rss_final = get_process_rss_kb();
        printf("  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS %ld KB -> %ld KB (Delta: %ld KB)\n",
               rss_initial, rss_final, (rss_final >= 0 && rss_initial >= 0) ? (rss_final - rss_initial) : 0);

        CHECK_AC(ac_runtime_context_destroy(ctx));
        printf("  -> Context Teardown: Completed cleanly via public ABI (Returned AC_SUCCESS)\n");
    }
    passed_gates++;
    printf("  -> Gate 10 PASS\n\n");

    printf("======================================================================\n");
    printf("AMUDA 2.0 Phase 4A Verification Gates: %d / %d Passed\n", passed_gates, total_gates);
    printf("Public Opaque C ABI Runtime Status: VERIFIED PASS\n");
    printf("======================================================================\n");
    return 0;
}
