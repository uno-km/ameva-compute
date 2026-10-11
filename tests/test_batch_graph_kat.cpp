// SPDX-License-Identifier: Apache-2.0
// Known-answer tests for the graph batch API (ac_runtime_batch_begin_ex / _dispatch_bindings / _barrier).
//
//   CHAIN       1000 dependent dispatches in one submission: x[k+1] = P x[k] with P a cyclic shift (0/1 weights, so
//               every result is exact in fp32); each step binds sub-ranges of one buffer at aligned offsets and is
//               separated from the next by an explicit barrier. x[1000] must equal x[0] rotated by 1000 mod 64.
//   CACHE       the same batch recorded again: every descriptor set comes from the context cache.
//   OWNERSHIP   batch destroyed before its completion event, and the reverse order: both complete and release.
//   MEMORY      30000 dispatches flushed every 128: exact result, and process RSS grows by less than 48 MiB.
//   VALIDATION  misaligned offset, out-of-range binding, unknown flag, capacity overflow, barrier after submit.
//   LIFETIME    a context with a live event refuses destruction; after release it is destroyed.
// Exit status 0 only if every check passed; each line reports a measured value.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "ac_runtime.h"
#include "../adapters/llama/ggml-ameva/ggml_ameva_kernels_spv.h"

static int g_failures = 0;

#define EXPECT(cond, ...)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            ++g_failures;                                                                                              \
            printf("  FAIL line %d: ", __LINE__);                                                                      \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
        }                                                                                                              \
    } while (0)

#define REQUIRE_AC(expr)                                                                                               \
    do {                                                                                                               \
        AcResult _r = (expr);                                                                                          \
        if (_r != AC_SUCCESS) {                                                                                        \
            printf("  FATAL: %s returned %d (line %d)\n", #expr, (int)_r, __LINE__);                                   \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)

static const uint32_t kDim = 64;    // vector length = matrix order
static const uint32_t kSteps = 1000;

struct Rig {
    AcContext* ctx = nullptr;
    AcKernel* kernel = nullptr;
    AcBuffer* w = nullptr;
    AcBuffer* x = nullptr;
    uint64_t stride = 0;
};

// QMM push constants: M, K, N, offset_src0, offset_src1, offset_dst, stride_src1, stride_dst
static const uint32_t kPc[8] = {1u, kDim, kDim, 0u, 0u, 0u, kDim, kDim}; // offsets carried by the descriptors

// flush_every > 0: submit what is recorded after every flush_every dispatches (ac_runtime_batch_flush)
static AcResult record_chain(Rig& rig, AcCommandBatch* batch, uint32_t flush_every = 0) {
    for (uint32_t k = 0; k < kSteps; ++k) {
        const AcBufferBinding b[3] = {
            {rig.w, 0, (uint64_t)kDim * kDim * sizeof(float)},
            {rig.x, (uint64_t)k * rig.stride, (uint64_t)kDim * sizeof(float)},
            {rig.x, (uint64_t)(k + 1) * rig.stride, (uint64_t)kDim * sizeof(float)},
        };
        AcResult r = ac_runtime_batch_dispatch_bindings(batch, rig.kernel, b, 3, kPc, sizeof(kPc), {kDim / 4, 1, 1});
        if (r != AC_SUCCESS) return r;
        r = ac_runtime_batch_barrier(batch);
        if (r != AC_SUCCESS) return r;
        if (flush_every && (k + 1) % flush_every == 0) {
            r = ac_runtime_batch_flush(batch);
            if (r != AC_SUCCESS) return r;
        }
    }
    return AC_SUCCESS;
}

// resident set size of this process in KiB (Mali kbase GPU memory is counted in it); -1 where /proc is unavailable
static long rss_kib() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kib = -1;
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "VmRSS:", 6)) kib = atol(line + 6);
    }
    fclose(f);
    return kib;
}

// x[kSteps] must be x[0] rotated by kSteps; every intermediate step written
static uint32_t chain_mismatches(const Rig& rig, const float* xp, const std::vector<float>& x0, uint32_t* written) {
    const float* last = (const float*)((const char*)xp + (size_t)kSteps * rig.stride);
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < kDim; ++i) {
        if (last[i] != x0[(i + kDim - (kSteps % kDim)) % kDim]) ++mismatches;
    }
    *written = 0;
    for (uint32_t k = 1; k <= kSteps; ++k) {
        if (!isnan(((const float*)((const char*)xp + (size_t)k * rig.stride))[0])) ++*written;
    }
    return mismatches;
}

int main() {
    printf("AMUDA graph batch KAT\n");
    Rig rig;
    AcContextDesc cdesc = {AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0};
    REQUIRE_AC(ac_runtime_context_create(&cdesc, &rig.ctx));

    AcDeviceLimits lim = {};
    lim.abi_version = AC_ABI_VERSION;
    lim.struct_size = sizeof(lim);
    REQUIRE_AC(ac_runtime_context_get_limits(rig.ctx, &lim));
    printf("LIMITS max_storage_buffer_range=%llu min_storage_offset_alignment=%llu max_alloc=%llu subgroup=%u "
           "push_constants=%u timestamps=%u\n",
           (unsigned long long)lim.max_storage_buffer_range,
           (unsigned long long)lim.min_storage_buffer_offset_alignment,
           (unsigned long long)lim.max_memory_allocation_size, lim.subgroup_size, lim.max_push_constants_size,
           lim.timestamps_supported);

    // kernel: AMEVA QMM, f32 weights, one activation column
    int vi = -1;
    for (int i = 0; i < AMEVA_QMM_VARIANT_COUNT; ++i) {
        if (!strcmp(kAmevaQmmVariants[i].type_name, "f32") && kAmevaQmmVariants[i].ncols == 1) vi = i;
    }
    if (vi < 0) {
        printf("  FATAL: f32 ncols=1 variant missing from ggml_ameva_kernels_spv.h\n");
        return 2;
    }
    static const AcRuntimeBindingDesc kBindings[3] = {{0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
                                                      {1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE},
                                                      {2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE}};
    const AmevaQmmVariant& v = kAmevaQmmVariants[vi];
    AcKernelRuntimeMetadata meta = {"kat_qmm_f32", "main", 3, kBindings, sizeof(kPc), {64, 1, 1}, 0, v.spirv_sha256, 0, nullptr};
    AcKernelDesc kdesc = {AC_ABI_VERSION, sizeof(AcKernelDesc), v.spirv, v.spirv_bytes, &meta, 0, nullptr};
    REQUIRE_AC(ac_runtime_kernel_create(rig.ctx, &kdesc, &rig.kernel));

    // P: row n has a single 1 at column (n + 63) % 64, so (P x)[n] = x[n - 1]: a rotation by one
    rig.stride = 256;
    while (rig.stride % lim.min_storage_buffer_offset_alignment) rig.stride += 256;
    AcBufferDesc wdesc = {AC_ABI_VERSION, sizeof(AcBufferDesc), kDim * kDim * sizeof(float), AC_BUFFER_USAGE_STORAGE, 0};
    AcBufferDesc xdesc = {AC_ABI_VERSION, sizeof(AcBufferDesc), (size_t)(kSteps + 1) * rig.stride,
                          AC_BUFFER_USAGE_STORAGE, 0};
    REQUIRE_AC(ac_runtime_buffer_create(rig.ctx, &wdesc, &rig.w));
    REQUIRE_AC(ac_runtime_buffer_create(rig.ctx, &xdesc, &rig.x));
    float* wp = nullptr;
    float* xp = nullptr;
    REQUIRE_AC(ac_runtime_buffer_get_mapped_ptr(rig.w, (void**)&wp));
    REQUIRE_AC(ac_runtime_buffer_get_mapped_ptr(rig.x, (void**)&xp));
    memset(wp, 0, kDim * kDim * sizeof(float));
    for (uint32_t n = 0; n < kDim; ++n) wp[n * kDim + (n + kDim - 1) % kDim] = 1.0f;
    memset(xp, 0xFF, (size_t)(kSteps + 1) * rig.stride); // NaN everywhere: unwritten steps cannot pass
    std::vector<float> x0(kDim);
    for (uint32_t i = 0; i < kDim; ++i) x0[i] = (float)(i * 3 + 1) * 0.25f;
    memcpy(xp, x0.data(), kDim * sizeof(float));
    REQUIRE_AC(ac_runtime_buffer_flush(rig.w, 0, kDim * kDim * sizeof(float)));
    REQUIRE_AC(ac_runtime_buffer_flush(rig.x, 0, (size_t)(kSteps + 1) * rig.stride));

    // ---- CHAIN ----
    AcBatchDesc bd = {AC_ABI_VERSION, sizeof(AcBatchDesc), kSteps, AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_TIMESTAMPS};
    AcCommandBatch* batch = nullptr;
    REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &bd, &batch));
    REQUIRE_AC(record_chain(rig, batch));
    AcBatchStats st = {};
    st.abi_version = AC_ABI_VERSION;
    st.struct_size = sizeof(st);
    REQUIRE_AC(ac_runtime_batch_get_stats(batch, &st));
    const uint64_t first_misses = st.descriptor_cache_misses;
    AcEvent* ev = nullptr;
    REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));
    REQUIRE_AC(ac_runtime_event_wait(ev, 10000000000ull));
    uint64_t wall = 0, gpu = 0;
    const AcResult tr = ac_runtime_event_get_timing(ev, &wall, &gpu);
    REQUIRE_AC(ac_runtime_event_destroy(ev));
    ac_runtime_batch_destroy(batch);
    REQUIRE_AC(ac_runtime_buffer_invalidate(rig.x, 0, (size_t)(kSteps + 1) * rig.stride));
    uint32_t steps_written = 0;
    const uint32_t mismatches = chain_mismatches(rig, xp, x0, &steps_written);
    printf("CHAIN dispatches=%u barriers=%u steps_written=%u/%u final_mismatches=%u wall_us=%.1f gpu_us=%.1f%s\n",
           st.dispatches, st.barriers, steps_written, kSteps, mismatches, wall / 1e3, gpu / 1e3,
           tr == AC_SUCCESS ? "" : " (gpu timing unavailable)");
    EXPECT(st.dispatches == kSteps && st.barriers == kSteps, "dispatch/barrier count");
    EXPECT(steps_written == kSteps && mismatches == 0, "chain result");

    // ---- FLUSH: the same chain cut into 10 submissions; order and visibility must not change ----
    memset((char*)xp + rig.stride, 0xFF, (size_t)kSteps * rig.stride);
    REQUIRE_AC(ac_runtime_buffer_flush(rig.x, 0, (size_t)(kSteps + 1) * rig.stride));
    REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &bd, &batch));
    REQUIRE_AC(record_chain(rig, batch, 100));
    const AcResult r_noop = ac_runtime_batch_flush(batch); // nothing recorded since the last flush
    REQUIRE_AC(ac_runtime_batch_get_stats(batch, &st));
    REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));
    REQUIRE_AC(ac_runtime_event_wait(ev, 10000000000ull));
    uint64_t fwall = 0, fgpu = 0;
    ac_runtime_event_get_timing(ev, &fwall, &fgpu);
    REQUIRE_AC(ac_runtime_event_destroy(ev));
    ac_runtime_batch_destroy(batch);
    REQUIRE_AC(ac_runtime_buffer_invalidate(rig.x, 0, (size_t)(kSteps + 1) * rig.stride));
    uint32_t fwritten = 0;
    const uint32_t fmismatch = chain_mismatches(rig, xp, x0, &fwritten);
    printf("FLUSH flushes=%u empty_flush=%d steps_written=%u/%u final_mismatches=%u wall_us=%.1f gpu_us=%.1f\n",
           st.flushes, (int)r_noop, fwritten, kSteps, fmismatch, fwall / 1e3, fgpu / 1e3);
    EXPECT(st.flushes == kSteps / 100 && r_noop == AC_SUCCESS, "flush count (an empty flush is a no-op)");
    EXPECT(fwritten == kSteps && fmismatch == 0, "flushed chain result");

    // ---- CACHE: identical ranges again -> no new descriptor sets ----
    REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &bd, &batch));
    REQUIRE_AC(record_chain(rig, batch));
    REQUIRE_AC(ac_runtime_batch_get_stats(batch, &st));
    printf("CACHE first_recording_misses=%llu second_recording hits=%llu misses=%llu\n",
           (unsigned long long)first_misses, (unsigned long long)st.descriptor_cache_hits,
           (unsigned long long)st.descriptor_cache_misses);
    EXPECT(first_misses == kSteps, "first recording should create one set per distinct range triple");
    EXPECT(st.descriptor_cache_hits == kSteps && st.descriptor_cache_misses == 0, "second recording should hit");
    REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));

    // ---- OWNERSHIP (a): batch released first, the event still completes and releases the submission ----
    ac_runtime_batch_destroy(batch);
    const AcResult wa = ac_runtime_event_wait(ev, 10000000000ull);
    const AcResult da = ac_runtime_event_destroy(ev);
    printf("OWNERSHIP batch_first wait=%d destroy=%d\n", (int)wa, (int)da);
    EXPECT(wa == AC_SUCCESS && da == AC_SUCCESS, "batch-first release order");

    // ---- OWNERSHIP (b): event released first, then the batch ----
    REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &bd, &batch));
    REQUIRE_AC(record_chain(rig, batch));
    REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));
    const AcResult db = ac_runtime_event_destroy(ev); // waits for completion
    ac_runtime_batch_destroy(batch);
    printf("OWNERSHIP event_first destroy=%d\n", (int)db);
    EXPECT(db == AC_SUCCESS, "event-first release order");

    // ---- MEMORY: a long graph cut into submissions holds bounded driver memory. A command buffer keeps its driver
    // memory until it is reset or freed (Mali-G68: about 100 KB per dispatch, 3 GB for 30000 dispatches before the
    // runtime recorded completed flushed command buffers again), so RSS must not grow with the length of the graph.
    // The chain ping-pongs between slots 0 and 1 and ends in slot 0: x0 rotated by kMemSteps.
    {
        const uint32_t kMemSteps = 30000, kMemFlush = 128, kMemMark = 2000;
        AcBatchDesc md = {AC_ABI_VERSION, sizeof(AcBatchDesc), kMemSteps, AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_TIMESTAMPS};
        REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &md, &batch));
        long rss_mark = -1, rss_end = -1;
        for (uint32_t k = 0; k < kMemSteps; ++k) {
            const AcBufferBinding b[3] = {
                {rig.w, 0, (uint64_t)kDim * kDim * sizeof(float)},
                {rig.x, (uint64_t)(k % 2) * rig.stride, (uint64_t)kDim * sizeof(float)},
                {rig.x, (uint64_t)((k + 1) % 2) * rig.stride, (uint64_t)kDim * sizeof(float)},
            };
            REQUIRE_AC(ac_runtime_batch_dispatch_bindings(batch, rig.kernel, b, 3, kPc, sizeof(kPc), {kDim / 4, 1, 1}));
            REQUIRE_AC(ac_runtime_batch_barrier(batch));
            if ((k + 1) % kMemFlush == 0) REQUIRE_AC(ac_runtime_batch_flush(batch));
            if (k + 1 == kMemMark) rss_mark = rss_kib();
        }
        rss_end = rss_kib();
        REQUIRE_AC(ac_runtime_batch_get_stats(batch, &st));
        REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));
        REQUIRE_AC(ac_runtime_event_wait(ev, 60000000000ull));
        REQUIRE_AC(ac_runtime_event_destroy(ev));
        ac_runtime_batch_destroy(batch);
        REQUIRE_AC(ac_runtime_buffer_invalidate(rig.x, 0, 2 * rig.stride));
        uint32_t mmis = 0;
        for (uint32_t i = 0; i < kDim; ++i) {
            if (xp[i] != x0[(i + kDim - (kMemSteps % kDim)) % kDim]) ++mmis;
        }
        const bool rss_known = rss_mark >= 0 && rss_end >= 0;
        const double growth_mib = rss_known ? (double)(rss_end - rss_mark) / 1024.0 : 0.0;
        printf("MEMORY dispatches=%u flushes=%u final_mismatches=%u rss_growth_MiB=%.1f (dispatch %u -> %u)%s\n",
               st.dispatches, st.flushes, mmis, growth_mib, kMemMark, kMemSteps,
               rss_known ? "" : " RSS unavailable: growth gate not applied");
        EXPECT(st.dispatches == kMemSteps && mmis == 0, "long flushed chain result");
        EXPECT(!rss_known || growth_mib < 48.0, "RSS grew %.1f MiB over %u dispatches", growth_mib, kMemSteps - kMemMark);
    }

    // ---- VALIDATION ----
    AcBatchDesc small = {AC_ABI_VERSION, sizeof(AcBatchDesc), 1, AC_BATCH_FLAG_EXPLICIT_BARRIERS};
    REQUIRE_AC(ac_runtime_batch_begin_ex(rig.ctx, &small, &batch));
    // an offset that violates the device's minStorageBufferOffsetAlignment (Adreno 650: 4 bytes, so a fixed offset of 4
    // is legal there); with an alignment of 1 no offset is misaligned and the check does not apply
    const uint64_t align = lim.min_storage_buffer_offset_alignment;
    const bool align_testable = align >= 2;
    const AcBufferBinding misaligned[3] = {{rig.w, 0, kDim * kDim * sizeof(float)},
                                           {rig.x, align_testable ? align / 2 : 0, kDim * sizeof(float)},
                                           {rig.x, rig.stride, kDim * sizeof(float)}};
    const AcResult r_align = align_testable ? ac_runtime_batch_dispatch_bindings(batch, rig.kernel, misaligned, 3, kPc,
                                                                                 sizeof(kPc), {16, 1, 1})
                                            : AC_ERR_BUFFER_ALIGNMENT_INVALID;
    if (!align_testable) printf("VALIDATION misaligned: not applicable (alignment %llu)\n", (unsigned long long)align);
    const AcBufferBinding outside[3] = {{rig.w, 0, kDim * kDim * sizeof(float) + 4},
                                        {rig.x, 0, kDim * sizeof(float)},
                                        {rig.x, rig.stride, kDim * sizeof(float)}};
    const AcResult r_range = ac_runtime_batch_dispatch_bindings(batch, rig.kernel, outside, 3, kPc, sizeof(kPc), {16, 1, 1});
    const AcBufferBinding ok[3] = {{rig.w, 0, kDim * kDim * sizeof(float)},
                                   {rig.x, 0, kDim * sizeof(float)},
                                   {rig.x, rig.stride, kDim * sizeof(float)}};
    const AcResult r_first = ac_runtime_batch_dispatch_bindings(batch, rig.kernel, ok, 3, kPc, sizeof(kPc), {16, 1, 1});
    const AcResult r_full = ac_runtime_batch_dispatch_bindings(batch, rig.kernel, ok, 3, kPc, sizeof(kPc), {16, 1, 1});
    REQUIRE_AC(ac_runtime_batch_submit(batch, &ev));
    const AcResult r_late = ac_runtime_batch_barrier(batch);
    AcBatchDesc badflag = {AC_ABI_VERSION, sizeof(AcBatchDesc), 4, 0x80u};
    AcCommandBatch* nb = nullptr;
    const AcResult r_flag = ac_runtime_batch_begin_ex(rig.ctx, &badflag, &nb);
    printf("VALIDATION misaligned=%d out_of_range=%d first=%d over_capacity=%d barrier_after_submit=%d bad_flag=%d\n",
           (int)r_align, (int)r_range, (int)r_first, (int)r_full, (int)r_late, (int)r_flag);
    EXPECT(r_align == AC_ERR_BUFFER_ALIGNMENT_INVALID, "misaligned offset");
    EXPECT(r_range == AC_ERR_INVALID_BUFFER_SIZE, "range beyond the buffer");
    EXPECT(r_first == AC_SUCCESS && r_full == AC_ERR_DEVICE_BUSY, "capacity");
    EXPECT(r_late == AC_ERR_INVALID_ARGUMENT, "barrier after submit");
    EXPECT(r_flag == AC_ERR_INVALID_ARGUMENT && nb == nullptr, "unknown flag");

    // ---- LIFETIME: the event keeps the context alive ----
    ac_runtime_batch_destroy(batch);
    REQUIRE_AC(ac_runtime_event_wait(ev, 10000000000ull));
    REQUIRE_AC(ac_runtime_buffer_destroy(rig.w));
    REQUIRE_AC(ac_runtime_buffer_destroy(rig.x));
    REQUIRE_AC(ac_runtime_kernel_destroy(rig.kernel));
    const AcResult r_busy = ac_runtime_context_destroy(rig.ctx);
    REQUIRE_AC(ac_runtime_event_destroy(ev));
    const AcResult r_done = ac_runtime_context_destroy(rig.ctx);
    printf("LIFETIME destroy_with_live_event=%d destroy_after_release=%d\n", (int)r_busy, (int)r_done);
    EXPECT(r_busy == AC_ERR_RESOURCE_IN_USE, "context destroy with a live event");
    EXPECT(r_done == AC_SUCCESS, "context destroy after release");

    printf("AMUDA_BATCH_GRAPH_KAT=%s (%d failed checks)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
