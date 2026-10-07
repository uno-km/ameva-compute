/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* SPDX-License-Identifier: Apache-2.0 */
/*
 * TEST DOUBLE for the AMUDA C ABI (ac_runtime). It records what amuda_rt asks for and executes NOTHING.
 * It exists only to unit-test host-side plumbing (argument packing, binding order, specialization
 * constants, pipeline caching, flush/invalidate ordering, error propagation) in an environment
 * without a GPU. It is never linked into the product.
 */
#include "ac_runtime.h"
#include "mock_ac_runtime.h"
#include <stdlib.h>
#include <string.h>

struct AcContext { int magic; };
struct AcBuffer { void* mem; size_t size; int coherent; int flushes; int invalidates; };
struct AcKernel { AcKernelRuntimeMetadata md; AcSpecConstant spec[3]; uint32_t nspec; };
struct AcEvent { int dummy; };

MockLog g_mock;

AcResult ac_runtime_context_create(const AcContextDesc* d, AcContext** out) {
    if (g_mock.fail_context) { *out = NULL; return AC_ERR_DEVICE_NOT_SUPPORTED; }
    if (!d || d->abi_version != AC_ABI_VERSION || d->struct_size != sizeof(AcContextDesc) || d->backend_kind != AC_BACKEND_VULKAN)
        return AC_ERR_INVALID_ARGUMENT;
    *out = (AcContext*)calloc(1, sizeof(AcContext));
    return AC_SUCCESS;
}
AcResult ac_runtime_context_destroy(AcContext* c) { free(c); return AC_SUCCESS; }
AcResult ac_runtime_context_get_device_name(const AcContext* c, char* out, size_t n) {
    (void)c; strncpy(out, "MOCK (no execution)", n - 1); out[n - 1] = 0; return AC_SUCCESS;
}
AcResult ac_runtime_buffer_create(AcContext* c, const AcBufferDesc* d, AcBuffer** out) {
    (void)c;
    if (!d || d->abi_version != AC_ABI_VERSION || d->struct_size != sizeof(AcBufferDesc) || d->size_bytes == 0)
        return AC_ERR_INVALID_ARGUMENT;
    AcBuffer* b = (AcBuffer*)calloc(1, sizeof(AcBuffer));
    b->mem = calloc(1, d->size_bytes);
    b->size = d->size_bytes;
    b->coherent = !g_mock.non_coherent;
    g_mock.buffers_created++;
    *out = b;
    return AC_SUCCESS;
}
AcResult ac_runtime_buffer_destroy(AcBuffer* b) {
    free(b->mem); free(b); g_mock.buffers_destroyed++;
    return g_mock.fail_buffer_destroy ? AC_ERR_INVALID_ARGUMENT : AC_SUCCESS;
}
AcResult ac_runtime_buffer_get_mapped_ptr(const AcBuffer* b, void** p) { *p = b->mem; return AC_SUCCESS; }
AcResult ac_runtime_buffer_is_coherent(const AcBuffer* b, uint32_t* c) { *c = (uint32_t)b->coherent; return AC_SUCCESS; }
AcResult ac_runtime_buffer_flush(AcBuffer* b, size_t o, size_t s) {
    if (o + s > b->size) return AC_ERR_INVALID_ARGUMENT;
    b->flushes++; g_mock.flushes++; g_mock.seq[g_mock.nseq++ % MOCK_SEQ] = 'F'; return AC_SUCCESS;
}
AcResult ac_runtime_buffer_invalidate(AcBuffer* b, size_t o, size_t s) {
    if (o + s > b->size) return AC_ERR_INVALID_ARGUMENT;
    b->invalidates++; g_mock.invalidates++; g_mock.seq[g_mock.nseq++ % MOCK_SEQ] = 'I'; return AC_SUCCESS;
}
AcResult ac_runtime_kernel_create(AcContext* c, const AcKernelDesc* d, AcKernel** out) {
    (void)c;
    const AcKernelRuntimeMetadata* m = d->metadata;
    /* mirror the ac_runtime validation rules amuda_rt depends on */
    if (d->abi_version != AC_ABI_VERSION || d->struct_size != sizeof(AcKernelDesc)) return AC_ERR_ABI_MISMATCH;
    if (!d->spirv_code || d->code_size_bytes % 4 || !m) return AC_ERR_INVALID_ARGUMENT;
    if (!m->expected_spirv_sha256 || strlen(m->expected_spirv_sha256) != 64) return AC_ERR_INVALID_ARGUMENT;
    if (!m->local_size.x || !m->local_size.y || !m->local_size.z) return AC_ERR_INVALID_WORK_GROUP_SIZE;
    if (m->push_constant_size > 128 || m->binding_count > 16) return AC_ERR_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < m->binding_count; ++i) if (m->bindings[i].binding_index != i) return AC_ERR_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < m->specialization_schema_count; ++i) {
        const AcSpecConstantSchemaEntry* s = &m->specialization_schemas[i];
        uint32_t ax = s->overrides_local_size_axis;
        uint32_t want = ax == 1 ? m->local_size.x : ax == 2 ? m->local_size.y : m->local_size.z;
        if (ax && s->default_value_bits != want) return AC_ERR_INVALID_ARGUMENT;
        if (s->default_value_bits < s->min_value_bits || s->default_value_bits > s->max_value_bits) return AC_ERR_INVALID_ARGUMENT;
    }
    if (g_mock.fail_kernel) return AC_ERR_KERNEL_BUILD_FAILED;
    AcKernel* k = (AcKernel*)calloc(1, sizeof(AcKernel));
    k->md = *m;
    k->nspec = d->specialization_count;
    memcpy(k->spec, d->specializations, sizeof(AcSpecConstant) * (d->specialization_count > 3 ? 3 : d->specialization_count));
    g_mock.kernels_created++;
    g_mock.last_local[0] = m->local_size.x; g_mock.last_local[1] = m->local_size.y; g_mock.last_local[2] = m->local_size.z;
    for (uint32_t i = 0; i < 3 && i < d->specialization_count; ++i) g_mock.last_spec[i] = d->specializations[i].value_bits;
    *out = k;
    return AC_SUCCESS;
}
AcResult ac_runtime_kernel_destroy(AcKernel* k) {
    free(k); g_mock.kernels_destroyed++;
    return g_mock.fail_kernel_destroy ? AC_ERR_INVALID_ARGUMENT : AC_SUCCESS;
}
AcResult ac_runtime_dispatch(AcContext* c, const AcLaunchDesc* d, AcEvent** out) {
    (void)c;
    if (d->abi_version != AC_ABI_VERSION || d->struct_size != sizeof(AcLaunchDesc)) return AC_ERR_ABI_MISMATCH;
    if (d->buffer_count != d->kernel->md.binding_count) return AC_ERR_INVALID_ARGUMENT;
    if (d->push_constants_size != d->kernel->md.push_constant_size) return AC_ERR_INVALID_ARGUMENT;
    if (d->push_constants_size && !d->push_constants) return AC_ERR_NULL_POINTER;
    if (g_mock.fail_dispatch) return AC_ERR_DEVICE_BUSY;
    g_mock.dispatches++;
    g_mock.seq[g_mock.nseq++ % MOCK_SEQ] = 'D';
    g_mock.last_nbuf = d->buffer_count;
    for (uint32_t i = 0; i < d->buffer_count; ++i) g_mock.last_buf_mem[i] = d->buffers[i]->mem;
    g_mock.last_pc_size = (uint32_t)d->push_constants_size;
    if (d->push_constants_size) memcpy(g_mock.last_pc, d->push_constants, d->push_constants_size);
    g_mock.last_grid[0] = d->grid_dim.x; g_mock.last_grid[1] = d->grid_dim.y; g_mock.last_grid[2] = d->grid_dim.z;
    *out = (AcEvent*)calloc(1, sizeof(AcEvent));
    g_mock.events_live++;
    return AC_SUCCESS;
}
AcResult ac_runtime_event_wait(AcEvent* e, uint64_t t) {
    (void)e; (void)t;
    g_mock.seq[g_mock.nseq++ % MOCK_SEQ] = 'W';
    if (g_mock.timeout_next) { g_mock.timeout_next = 0; return AC_ERR_LAUNCH_TIMEOUT; }
    return AC_SUCCESS;
}
AcResult ac_runtime_event_get_timing(AcEvent* e, uint64_t* wall, uint64_t* gpu) { (void)e; *wall = 2000; *gpu = 1500; return AC_SUCCESS; }
AcResult ac_runtime_event_destroy(AcEvent* e) { free(e); g_mock.events_live--; return AC_SUCCESS; }
