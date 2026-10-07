/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_runtime.cpp
 * @brief AMUDA 2.0 Public Opaque C Runtime Implementation
 *
 * Implements strict ABI contract validation, internal object encapsulation,
 * context child tracking (including active events), and failsafe error handling.
 */

#include "ac_runtime.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <atomic>
#include "../backend/vulkan/ac_vk_sha256.h"
#include "../backend/opencl/ac_cl_backend.h"
#include "ac_security_guard.h"

#define AC_MAGIC_CONTEXT 0xAC010001
#define AC_MAGIC_BUFFER  0xAC010002
#define AC_MAGIC_KERNEL  0xAC010003
#define AC_MAGIC_EVENT   0xAC010004
#define AC_MAGIC_BATCH   0xAC010005

/* Every handle carries the object of exactly one backend: vk_* for AC_BACKEND_VULKAN, cl_* for AC_BACKEND_OPENCL. */
struct AcContext {
    uint32_t magic;
    uint32_t abi_version;
    AcBackendKind backend_kind;
    AcVkContext* vk_ctx;
    uint32_t active_child_buffers;
    uint32_t active_child_kernels;
    uint32_t active_child_events;
    uint32_t active_child_batches;
    AcClContext* cl_ctx;
};

struct AcBuffer {
    uint32_t magic;
    AcContext* owning_context;
    AcVkBuffer* vk_buf;
    AcClBuffer* cl_buf;
};

struct AcKernel {
    uint32_t magic;
    AcContext* owning_context;
    AcVkKernel* vk_kernel;
    AcClKernel* cl_kernel;
};

struct AcEvent {
    uint32_t magic;
    AcContext* owning_context;
    AcVkEvent* vk_event;
    AcClEvent* cl_event;
};

struct AcCommandBatch {
    uint32_t magic;
    AcContext* owning_context;
    AcVkCommandBatch* vk_batch;
    AcClBatch* cl_batch;
};

static AcEvent* wrap_cl_event(AcContext* ctx, AcClEvent* cl_ev) {
    AcEvent* ev = (AcEvent*)calloc(1, sizeof(AcEvent));
    if (!ev) {
        ac_cl_event_destroy(cl_ev);
        return NULL;
    }
    ev->magic = AC_MAGIC_EVENT;
    ev->owning_context = ctx;
    ev->cl_event = cl_ev;
    ctx->active_child_events++;
    return ev;
}


/* ========================================================================= */
/* Context Management Implementation                                         */
/* ========================================================================= */

AcResult ac_runtime_context_create(const AcContextDesc* desc, AcContext** out_ctx) {
    if (out_ctx) {
        *out_ctx = NULL;
    }
    if (!desc || !out_ctx) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->abi_version != AC_ABI_VERSION) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->struct_size != sizeof(AcContextDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->flags != 0) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    AcResult sec_res = ac_security_verify_host_environment();
    if (sec_res != AC_SUCCESS) {
        return sec_res;
    }
    if (desc->backend_kind == AC_BACKEND_OPENCL) {
        AcClContext* cl_ctx = NULL;
        AcResult res = ac_cl_context_create(&cl_ctx);
        if (res != AC_SUCCESS) {
            return res;
        }
        AcContext* ctx = (AcContext*)calloc(1, sizeof(AcContext));
        if (!ctx) {
            ac_cl_context_destroy(cl_ctx);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        ctx->magic = AC_MAGIC_CONTEXT;
        ctx->abi_version = desc->abi_version;
        ctx->backend_kind = AC_BACKEND_OPENCL;
        ctx->cl_ctx = cl_ctx;
        *out_ctx = ctx;
        return AC_SUCCESS;
    }
    if (desc->backend_kind != AC_BACKEND_VULKAN) {
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    AcVkContext* vk_ctx = NULL;
    AcResult res = ac_vk_context_create(&vk_ctx);
    if (res != AC_SUCCESS) {
        return res;
    }

    AcContext* ctx = (AcContext*)calloc(1, sizeof(AcContext));
    if (!ctx) {
        ac_vk_context_destroy(vk_ctx);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    ctx->magic = AC_MAGIC_CONTEXT;
    ctx->abi_version = desc->abi_version;
    ctx->backend_kind = desc->backend_kind;
    ctx->vk_ctx = vk_ctx;
    ctx->active_child_buffers = 0;
    ctx->active_child_kernels = 0;
    ctx->active_child_events = 0;

    *out_ctx = ctx;
    return AC_SUCCESS;
}

AcResult ac_runtime_context_destroy(AcContext* ctx) {
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }

    /* Strict rejection if any child buffer, kernel, event, or batch is still alive */
    if (ctx->active_child_buffers != 0 ||
        ctx->active_child_kernels != 0 ||
        ctx->active_child_events  != 0 ||
        ctx->active_child_batches != 0) {
        return AC_ERR_RESOURCE_IN_USE;
    }


    if (ctx->vk_ctx) {
        AcResult res = ac_vk_context_destroy(ctx->vk_ctx);
        if (res != AC_SUCCESS) {
            return res;
        }
        ctx->vk_ctx = NULL;
    }
    if (ctx->cl_ctx) {
        AcResult res = ac_cl_context_destroy(ctx->cl_ctx);
        ctx->cl_ctx = NULL; /* released even when the final clFinish reported an error */
        if (res != AC_SUCCESS) {
            ctx->magic = 0;
            free(ctx);
            return res;
        }
    }

    ctx->magic = 0;
    free(ctx);
    return AC_SUCCESS;
}

AcResult ac_runtime_context_get_backend_kind(const AcContext* ctx, AcBackendKind* out_kind) {
    if (out_kind) {
        *out_kind = AC_BACKEND_UNKNOWN;
    }
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!out_kind) {
        return AC_ERR_NULL_POINTER;
    }
    *out_kind = ctx->backend_kind;
    return AC_SUCCESS;
}

AcResult ac_runtime_context_get_device_name(
    const AcContext* ctx,
    char* out_name,
    size_t max_len
) {
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!out_name || max_len == 0) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (ctx->vk_ctx) {
        snprintf(out_name, max_len, "%s", ctx->vk_ctx->device_properties.deviceName);
        return AC_SUCCESS;
    }
    if (ctx->cl_ctx) {
        return ac_cl_context_get_device_name(ctx->cl_ctx, out_name, max_len);
    }
    return AC_ERR_DEVICE_NOT_SUPPORTED;
}

/* ========================================================================= */
/* Buffer Management Implementation                                          */
/* ========================================================================= */

static VkBufferUsageFlags translate_buffer_usage(AcBufferUsageFlags flags) {
    VkBufferUsageFlags vk = 0;
    if (flags & AC_BUFFER_USAGE_STORAGE)      vk |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (flags & AC_BUFFER_USAGE_TRANSFER_SRC) vk |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (flags & AC_BUFFER_USAGE_TRANSFER_DST) vk |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (flags & AC_BUFFER_USAGE_UNIFORM)      vk |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    return vk;
}

AcResult ac_runtime_buffer_create(AcContext* ctx, const AcBufferDesc* desc, AcBuffer** out_buf) {
    if (out_buf) {
        *out_buf = NULL;
    }
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!desc || !out_buf) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->abi_version != AC_ABI_VERSION) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->struct_size != sizeof(AcBufferDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->size_bytes == 0) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }

    /* Reject unknown usage flag bits */
    const AcBufferUsageFlags kKnownUsage =
        AC_BUFFER_USAGE_STORAGE |
        AC_BUFFER_USAGE_TRANSFER_SRC |
        AC_BUFFER_USAGE_TRANSFER_DST |
        AC_BUFFER_USAGE_UNIFORM;

    if (desc->usage == 0 || (desc->usage & ~kKnownUsage) != 0) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* Reject values other than 0 and 1 */
    if (desc->force_non_coherent > 1u) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (ctx->cl_ctx) {
        /* usage and coherence preferences have no OpenCL counterpart: every SVM buffer serves every use, and the
           host reaches it only through ac_runtime_buffer_write / read / fill */
        AcClBuffer* cl_buf = NULL;
        AcResult res = ac_cl_buffer_create(ctx->cl_ctx, desc->size_bytes, &cl_buf);
        if (res != AC_SUCCESS) {
            return res;
        }
        AcBuffer* buf = (AcBuffer*)calloc(1, sizeof(AcBuffer));
        if (!buf) {
            ac_cl_buffer_destroy(cl_buf);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        buf->magic = AC_MAGIC_BUFFER;
        buf->owning_context = ctx;
        buf->cl_buf = cl_buf;
        ctx->active_child_buffers++;
        *out_buf = buf;
        return AC_SUCCESS;
    }

    VkBufferUsageFlags vk_usage = translate_buffer_usage(desc->usage);
    const bool force_non_coherent = (desc->force_non_coherent == 1u);

    AcVkBuffer* vk_buf = NULL;
    AcResult res = ac_vk_buffer_create_explicit(
        ctx->vk_ctx,
        desc->size_bytes,
        vk_usage,
        force_non_coherent,
        &vk_buf
    );
    if (res != AC_SUCCESS) {
        return res;
    }

    AcBuffer* buf = (AcBuffer*)calloc(1, sizeof(AcBuffer));
    if (!buf) {
        ac_vk_buffer_destroy(vk_buf);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    buf->magic = AC_MAGIC_BUFFER;
    buf->owning_context = ctx;
    buf->vk_buf = vk_buf;

    ctx->active_child_buffers++;
    *out_buf = buf;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_destroy(AcBuffer* buf) {
    if (!buf || buf->magic != AC_MAGIC_BUFFER) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (buf->vk_buf) {
        AcResult res = ac_vk_buffer_destroy(buf->vk_buf);
        if (res != AC_SUCCESS) {
            return res;
        }
        buf->vk_buf = NULL;
    }
    AcResult cl_res = AC_SUCCESS;
    if (buf->cl_buf) {
        cl_res = ac_cl_buffer_destroy(buf->cl_buf); /* freed in any case; an error reports the queue's state */
        buf->cl_buf = NULL;
    }

    if (buf->owning_context) {
        if (buf->owning_context->active_child_buffers > 0) {
            buf->owning_context->active_child_buffers--;
        }
        buf->owning_context = NULL;
    }

    buf->magic = 0;
    free(buf);
    return cl_res;
}

static bool buffer_valid(const AcBuffer* buf) {
    return buf && buf->magic == AC_MAGIC_BUFFER && (buf->vk_buf || buf->cl_buf);
}

AcResult ac_runtime_buffer_get_mapped_ptr(const AcBuffer* buf, void** out_ptr) {
    if (out_ptr) {
        *out_ptr = NULL;
    }
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_ptr) {
        return AC_ERR_NULL_POINTER;
    }
    if (buf->cl_buf) {
        return AC_ERR_BUFFER_NOT_MAPPED; /* OpenCL SVM: ac_runtime_buffer_write / read / fill */
    }
    *out_ptr = buf->vk_buf->mapped_ptr;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_get_size(const AcBuffer* buf, size_t* out_size) {
    if (out_size) {
        *out_size = 0;
    }
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_size = buf->cl_buf ? ac_cl_buffer_size(buf->cl_buf) : buf->vk_buf->size_bytes;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_get_allocation_size(const AcBuffer* buf, size_t* out_alloc_size) {
    if (out_alloc_size) {
        *out_alloc_size = 0;
    }
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_alloc_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_alloc_size = buf->cl_buf ? ac_cl_buffer_size(buf->cl_buf) : buf->vk_buf->allocation_size_bytes;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_is_coherent(const AcBuffer* buf, uint32_t* out_is_coherent) {
    if (out_is_coherent) {
        *out_is_coherent = 0u;
    }
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_is_coherent) {
        return AC_ERR_NULL_POINTER;
    }
    if (buf->cl_buf) {
        return AC_ERR_BUFFER_NOT_MAPPED;
    }
    *out_is_coherent = buf->vk_buf->is_coherent ? 1u : 0u;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_flush(AcBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (buf->cl_buf) {
        return AC_ERR_BUFFER_NOT_MAPPED;
    }
    return ac_vk_buffer_flush(buf->vk_buf, offset, size_bytes);
}

AcResult ac_runtime_buffer_invalidate(AcBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (buf->cl_buf) {
        return AC_ERR_BUFFER_NOT_MAPPED;
    }
    return ac_vk_buffer_invalidate(buf->vk_buf, offset, size_bytes);
}

AcResult ac_runtime_buffer_write(AcBuffer* buf, size_t offset, const void* src, size_t size_bytes) {
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!src && size_bytes) {
        return AC_ERR_NULL_POINTER;
    }
    if (buf->cl_buf) {
        return ac_cl_buffer_write(buf->cl_buf, offset, src, size_bytes);
    }
    const size_t size = buf->vk_buf->size_bytes;
    if (offset > size || size_bytes > size - offset) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }
    if (size_bytes == 0) {
        return AC_SUCCESS;
    }
    memcpy((char*)buf->vk_buf->mapped_ptr + offset, src, size_bytes);
    return ac_vk_buffer_flush(buf->vk_buf, offset, size_bytes);
}

AcResult ac_runtime_buffer_read(AcBuffer* buf, size_t offset, void* dst, size_t size_bytes) {
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!dst && size_bytes) {
        return AC_ERR_NULL_POINTER;
    }
    if (buf->cl_buf) {
        return ac_cl_buffer_read(buf->cl_buf, offset, dst, size_bytes);
    }
    const size_t size = buf->vk_buf->size_bytes;
    if (offset > size || size_bytes > size - offset) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }
    if (size_bytes == 0) {
        return AC_SUCCESS;
    }
    AcResult res = ac_vk_buffer_invalidate(buf->vk_buf, offset, size_bytes);
    if (res != AC_SUCCESS) {
        return res;
    }
    memcpy(dst, (const char*)buf->vk_buf->mapped_ptr + offset, size_bytes);
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_fill(AcBuffer* buf, size_t offset, size_t size_bytes, uint8_t value) {
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (buf->cl_buf) {
        return ac_cl_buffer_fill(buf->cl_buf, offset, size_bytes, value);
    }
    const size_t size = buf->vk_buf->size_bytes;
    if (offset > size || size_bytes > size - offset) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }
    if (size_bytes == 0) {
        return AC_SUCCESS;
    }
    memset((char*)buf->vk_buf->mapped_ptr + offset, value, size_bytes);
    return ac_vk_buffer_flush(buf->vk_buf, offset, size_bytes);
}

AcResult ac_runtime_buffer_get_address(const AcBuffer* buf, void** out_address) {
    if (out_address) {
        *out_address = NULL;
    }
    if (!buffer_valid(buf)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_address) {
        return AC_ERR_NULL_POINTER;
    }
    *out_address = buf->cl_buf ? ac_cl_buffer_address(buf->cl_buf) : buf->vk_buf->mapped_ptr;
    return AC_SUCCESS;
}

/* ========================================================================= */
/* Specialization Bounds & Encoding Utilities                                */
/* ========================================================================= */

static inline void write_le32(uint8_t out[4], uint32_t val) {
    out[0] = (uint8_t)(val & 0xFF);
    out[1] = (uint8_t)((val >> 8) & 0xFF);
    out[2] = (uint8_t)((val >> 16) & 0xFF);
    out[3] = (uint8_t)((val >> 24) & 0xFF);
}

static bool validate_spec_bounds(
    AcSpecConstantType type,
    uint32_t value_bits,
    uint32_t min_bits,
    uint32_t max_bits
) {
    switch (type) {
    case AC_SPEC_CONSTANT_U32:
        return value_bits >= min_bits && value_bits <= max_bits;

    case AC_SPEC_CONSTANT_I32: {
        int32_t value;
        int32_t min_value;
        int32_t max_value;

        memcpy(&value, &value_bits, sizeof(value));
        memcpy(&min_value, &min_bits, sizeof(min_value));
        memcpy(&max_value, &max_bits, sizeof(max_value));

        return value >= min_value && value <= max_value;
    }

    case AC_SPEC_CONSTANT_F32: {
        float value;
        float min_value;
        float max_value;

        memcpy(&value, &value_bits, sizeof(value));
        memcpy(&min_value, &min_bits, sizeof(min_value));
        memcpy(&max_value, &max_bits, sizeof(max_value));

        return isfinite(value) &&
               isfinite(min_value) &&
               isfinite(max_value) &&
               value >= min_value &&
               value <= max_value;
    }

    default:
        return false;
    }
}

/* ========================================================================= */
/* Kernel Management Implementation                                          */
/* ========================================================================= */

AcResult ac_runtime_kernel_create(AcContext* ctx, const AcKernelDesc* desc, AcKernel** out_kernel) {
    if (out_kernel) {
        *out_kernel = NULL;
    }
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!desc || !out_kernel) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->abi_version != AC_ABI_VERSION) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->struct_size != sizeof(AcKernelDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (ctx->cl_ctx) {
        return AC_ERR_FEATURE_NOT_SUPPORTED; /* SPIR-V for Vulkan; OpenCL contexts take ac_runtime_kernel_create_from_source */
    }
    if (!desc->spirv_code) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->code_size_bytes == 0 || (desc->code_size_bytes % 4 != 0)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!desc->metadata) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->specialization_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->specialization_count > 0 && !desc->specializations) {
        return AC_ERR_NULL_POINTER;
    }

    const AcKernelRuntimeMetadata* rmeta = desc->metadata;
    if (!rmeta->kernel_name || rmeta->kernel_name[0] == '\0') {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!rmeta->entry_point || rmeta->entry_point[0] == '\0') {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!rmeta->expected_spirv_sha256 || strlen(rmeta->expected_spirv_sha256) != 64) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (rmeta->local_size.x == 0 || rmeta->local_size.y == 0 || rmeta->local_size.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    if (rmeta->push_constant_size > 128) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (rmeta->binding_count > 0 && !rmeta->bindings) {
        return AC_ERR_NULL_POINTER;
    }
    if (rmeta->binding_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* Verify duplicate binding indices and validate binding entries */
    for (uint32_t i = 0; i < rmeta->binding_count; ++i) {
        if (rmeta->bindings[i].stage_flags != AC_STAGE_COMPUTE) {
            return AC_ERR_INVALID_ARGUMENT;
        }
        if (rmeta->bindings[i].binding_type != AC_RUNTIME_BINDING_STORAGE_BUFFER &&
            rmeta->bindings[i].binding_type != AC_RUNTIME_BINDING_UNIFORM_BUFFER) {
            return AC_ERR_INVALID_ARGUMENT;
        }
        for (uint32_t j = i + 1; j < rmeta->binding_count; ++j) {
            if (rmeta->bindings[i].binding_index == rmeta->bindings[j].binding_index) {
                return AC_ERR_INVALID_ARGUMENT; /* Duplicate binding index rejected */
            }
        }
    }

    /* Verify schema declarations */
    if (rmeta->specialization_schema_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (rmeta->specialization_schema_count > 0 && !rmeta->specialization_schemas) {
        return AC_ERR_NULL_POINTER;
    }
    for (uint32_t i = 0; i < rmeta->specialization_schema_count; ++i) {
        const AcSpecConstantSchemaEntry* s_i = &rmeta->specialization_schemas[i];
        if (s_i->type > AC_SPEC_CONSTANT_F32) {
            return AC_ERR_INVALID_ARGUMENT; /* Invalid schema type */
        }
        for (uint32_t j = i + 1; j < rmeta->specialization_schema_count; ++j) {
            if (s_i->constant_id == rmeta->specialization_schemas[j].constant_id) {
                return AC_ERR_INVALID_ARGUMENT; /* Duplicate schema constant ID */
            }
            if (s_i->overrides_local_size_axis != 0 &&
                s_i->overrides_local_size_axis == rmeta->specialization_schemas[j].overrides_local_size_axis) {
                return AC_ERR_INVALID_ARGUMENT; /* Duplicate local-size axis override */
            }
        }
        /* Validate min <= max based on type */
        bool min_max_valid = false;
        switch (s_i->type) {
        case AC_SPEC_CONSTANT_U32:
            min_max_valid = (s_i->min_value_bits <= s_i->max_value_bits);
            break;
        case AC_SPEC_CONSTANT_I32: {
            int32_t min_v, max_v;
            memcpy(&min_v, &s_i->min_value_bits, sizeof(min_v));
            memcpy(&max_v, &s_i->max_value_bits, sizeof(max_v));
            min_max_valid = (min_v <= max_v);
            break;
        }
        case AC_SPEC_CONSTANT_F32: {
            float min_v, max_v;
            memcpy(&min_v, &s_i->min_value_bits, sizeof(min_v));
            memcpy(&max_v, &s_i->max_value_bits, sizeof(max_v));
            min_max_valid = isfinite(min_v) && isfinite(max_v) && (min_v <= max_v);
            break;
        }
        default:
            min_max_valid = false;
            break;
        }
        if (!min_max_valid) {
            return AC_ERR_INVALID_ARGUMENT; /* min > max or non-finite */
        }
        /* Validate default is within [min, max] */
        if (!validate_spec_bounds(s_i->type, s_i->default_value_bits, s_i->min_value_bits, s_i->max_value_bits)) {
            return AC_ERR_INVALID_ARGUMENT; /* Default out of bounds */
        }
        /* Validate local-size axis range */
        if (s_i->overrides_local_size_axis > 3) {
            return AC_ERR_INVALID_ARGUMENT; /* Invalid axis */
        }
        /* Local-size axis override constraints */
        if (s_i->overrides_local_size_axis != 0) {
            if (s_i->type != AC_SPEC_CONSTANT_U32) {
                return AC_ERR_INVALID_ARGUMENT; /* Local size override must be U32 */
            }
            if (s_i->default_value_bits == 0 || s_i->min_value_bits == 0) {
                return AC_ERR_INVALID_ARGUMENT; /* Local size must be >= 1 */
            }
            /* Runtime-enforced invariant: schema default must match metadata.local_size */
            if (s_i->overrides_local_size_axis == 1 && s_i->default_value_bits != rmeta->local_size.x) {
                return AC_ERR_INVALID_ARGUMENT;
            }
            if (s_i->overrides_local_size_axis == 2 && s_i->default_value_bits != rmeta->local_size.y) {
                return AC_ERR_INVALID_ARGUMENT;
            }
            if (s_i->overrides_local_size_axis == 3 && s_i->default_value_bits != rmeta->local_size.z) {
                return AC_ERR_INVALID_ARGUMENT;
            }
        }
    }

    /* Verify caller specialization overrides */
    for (uint32_t i = 0; i < desc->specialization_count; ++i) {
        if (desc->specializations[i].type > AC_SPEC_CONSTANT_F32) {
            return AC_ERR_INVALID_ARGUMENT; /* Unknown type */
        }
        for (uint32_t j = i + 1; j < desc->specialization_count; ++j) {
            if (desc->specializations[i].constant_id == desc->specializations[j].constant_id) {
                return AC_ERR_INVALID_ARGUMENT; /* Duplicate constant ID */
            }
        }
        const AcSpecConstantSchemaEntry* matched = NULL;
        for (uint32_t s = 0; s < rmeta->specialization_schema_count; ++s) {
            if (rmeta->specialization_schemas[s].constant_id == desc->specializations[i].constant_id) {
                matched = &rmeta->specialization_schemas[s];
                break;
            }
        }
        if (!matched) {
            return AC_ERR_KERNEL_ARG_INVALID; /* Undeclared constant ID */
        }
        if (desc->specializations[i].type != matched->type) {
            return AC_ERR_INVALID_ARGUMENT; /* Type mismatch */
        }
        if (matched->overrides_local_size_axis != 0 && desc->specializations[i].value_bits == 0) {
            return AC_ERR_INVALID_ARGUMENT; /* Caller override for local size cannot be 0 */
        }
        if (!validate_spec_bounds(matched->type, desc->specializations[i].value_bits, matched->min_value_bits, matched->max_value_bits)) {
            return AC_ERR_INVALID_ARGUMENT; /* Caller value out of bounds or invalid float (NaN/Inf) */
        }
    }

    if (rmeta->specialization_schema_count == 0 && desc->specialization_count > 0) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }

    /* Build Effective Specialization Set */
    AcSpecConstant canonical_specs[16] = {};
    AcDim3 effective_local_size = rmeta->local_size;

    for (uint32_t k = 0; k < rmeta->specialization_schema_count; ++k) {
        const AcSpecConstantSchemaEntry* schema = &rmeta->specialization_schemas[k];
        uint32_t val = schema->default_value_bits;
        for (uint32_t c = 0; c < desc->specialization_count; ++c) {
            if (desc->specializations[c].constant_id == schema->constant_id) {
                val = desc->specializations[c].value_bits;
                break;
            }
        }

        canonical_specs[k].constant_id = schema->constant_id;
        canonical_specs[k].type = schema->type;
        canonical_specs[k].value_bits = val;

        if (schema->overrides_local_size_axis == 1) {
            effective_local_size.x = val;
        } else if (schema->overrides_local_size_axis == 2) {
            effective_local_size.y = val;
        } else if (schema->overrides_local_size_axis == 3) {
            effective_local_size.z = val;
        }
    }

    /* Sort canonical_specs strictly by constant_id ascending */
    for (uint32_t i = 0; i < rmeta->specialization_schema_count; ++i) {
        for (uint32_t j = i + 1; j < rmeta->specialization_schema_count; ++j) {
            if (canonical_specs[i].constant_id > canonical_specs[j].constant_id) {
                AcSpecConstant tmp = canonical_specs[i];
                canonical_specs[i] = canonical_specs[j];
                canonical_specs[j] = tmp;
            }
        }
    }

    /* Calculate deterministic binary-serialized pipeline identity hash */
    char pipeline_identity[AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE] = {};
    {
        AcSha256Ctx id_ctx;
        ac_sha256_init(&id_ctx);

        /* 1. Domain separator (exactly 18 bytes) */
        static const uint8_t kPipelineIdentityDomain[] = {
            'A', 'M', 'E', 'V', 'A', '-',
            'P', 'I', 'P', 'E', 'L', 'I', 'N', 'E',
            '-', 'I', 'D', '\0'
        };
        static_assert(sizeof(kPipelineIdentityDomain) == 18,
            "Pipeline identity domain must be 18 bytes");
        ac_sha256_update(&id_ctx, kPipelineIdentityDomain, sizeof(kPipelineIdentityDomain));

        /* 2. Identity schema version as LE32 */
        uint8_t le[4];
        write_le32(le, AC_PIPELINE_IDENTITY_SCHEMA_VERSION);
        ac_sha256_update(&id_ctx, le, 4);

        /* 3. SPIR-V SHA-256 raw digest (32 bytes) */
        uint8_t spv_raw_hash[32];
        AcSha256Ctx spv_ctx;
        ac_sha256_init(&spv_ctx);
        ac_sha256_update(&spv_ctx, (const uint8_t*)desc->spirv_code, desc->code_size_bytes);
        ac_sha256_final(&spv_ctx, spv_raw_hash);
        ac_sha256_update(&id_ctx, spv_raw_hash, 32);

        /* 4. Entry point length as LE32 + UTF-8 bytes */
        if (!rmeta->entry_point) {
            return AC_ERR_NULL_POINTER;
        }
        size_t ep_len_sz = strlen(rmeta->entry_point);
        if (ep_len_sz == 0 || ep_len_sz > UINT32_MAX) {
            return AC_ERR_INVALID_ARGUMENT;
        }
        uint32_t ep_len = (uint32_t)ep_len_sz;
        write_le32(le, ep_len);
        ac_sha256_update(&id_ctx, le, 4);
        ac_sha256_update(&id_ctx, (const uint8_t*)rmeta->entry_point, ep_len);

        /* 5. Canonical specialization count as LE32 */
        uint32_t spec_count = rmeta->specialization_schema_count;
        if (spec_count > 16) {
            return AC_ERR_INVALID_ARGUMENT;
        }
        write_le32(le, spec_count);
        ac_sha256_update(&id_ctx, le, 4);

        /* 6. Canonical entries (constant_id, type, value_bits as LE32) */
        for (uint32_t i = 0; i < spec_count; ++i) {
            write_le32(le, canonical_specs[i].constant_id);
            ac_sha256_update(&id_ctx, le, 4);
            write_le32(le, canonical_specs[i].type);
            ac_sha256_update(&id_ctx, le, 4);
            write_le32(le, canonical_specs[i].value_bits);
            ac_sha256_update(&id_ctx, le, 4);
        }

        uint8_t final_hash[32];
        ac_sha256_final(&id_ctx, final_hash);
        for (int i = 0; i < 32; ++i) {
            sprintf(pipeline_identity + (i * 2), "%02x", final_hash[i]);
        }
        pipeline_identity[64] = '\0';
    }

    AcKernelBindingDesc vk_bindings[16];
    for (uint32_t i = 0; i < rmeta->binding_count; ++i) {
        vk_bindings[i].binding_index = rmeta->bindings[i].binding_index;
        vk_bindings[i].binding_type = (rmeta->bindings[i].binding_type == AC_RUNTIME_BINDING_STORAGE_BUFFER)
            ? AC_BINDING_TYPE_STORAGE_BUFFER : AC_BINDING_TYPE_UNIFORM_BUFFER;
        vk_bindings[i].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    AcKernelMetadata vk_meta = {};
    vk_meta.kernel_name = rmeta->kernel_name;
    vk_meta.entry_point = rmeta->entry_point;
    vk_meta.binding_count = rmeta->binding_count;
    vk_meta.bindings = vk_bindings;
    vk_meta.push_constant_size = rmeta->push_constant_size;
    vk_meta.local_size = rmeta->local_size;
    vk_meta.required_features = rmeta->required_features;
    vk_meta.expected_spirv_sha256 = rmeta->expected_spirv_sha256;
    vk_meta.specialization_schema_count = rmeta->specialization_schema_count;
    vk_meta.specialization_schemas = rmeta->specialization_schemas;

    AcVkKernel* vk_kernel = NULL;
    AcResult res = ac_vk_kernel_create_with_specialization(
        ctx->vk_ctx,
        desc->spirv_code,
        desc->code_size_bytes,
        &vk_meta,
        rmeta->specialization_schema_count,
        canonical_specs,
        effective_local_size,
        pipeline_identity,
        &vk_kernel
    );
    if (res != AC_SUCCESS) {
        return res;
    }

    AcKernel* kernel = (AcKernel*)calloc(1, sizeof(AcKernel));
    if (!kernel) {
        ac_vk_kernel_destroy(vk_kernel);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    kernel->magic = AC_MAGIC_KERNEL;
    kernel->owning_context = ctx;
    kernel->vk_kernel = vk_kernel;

    ctx->active_child_kernels++;
    *out_kernel = kernel;
    return AC_SUCCESS;
}

AcResult ac_runtime_kernel_create_from_source(AcContext* ctx, const AcKernelSourceDesc* desc, AcKernel** out_kernel) {
    if (out_kernel) {
        *out_kernel = NULL;
    }
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!desc || !out_kernel) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->abi_version != AC_ABI_VERSION || desc->struct_size != sizeof(AcKernelSourceDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->cl_ctx) {
        return AC_ERR_FEATURE_NOT_SUPPORTED; /* Vulkan contexts take SPIR-V (ac_runtime_kernel_create) */
    }
    if (!desc->source || !desc->metadata) {
        return AC_ERR_NULL_POINTER;
    }
    const AcKernelRuntimeMetadata* m = desc->metadata;
    if (!m->kernel_name || m->kernel_name[0] == '\0' || !m->entry_point || m->entry_point[0] == '\0') {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (m->local_size.x == 0 || m->local_size.y == 0 || m->local_size.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    if (m->push_constant_size > 128 || m->binding_count > 16 || (m->binding_count > 0 && !m->bindings)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < m->binding_count; ++i) {
        if (m->bindings[i].stage_flags != AC_STAGE_COMPUTE ||
            (m->bindings[i].binding_type != AC_RUNTIME_BINDING_STORAGE_BUFFER &&
             m->bindings[i].binding_type != AC_RUNTIME_BINDING_UNIFORM_BUFFER)) {
            return AC_ERR_INVALID_ARGUMENT;
        }
    }
    if (m->specialization_schema_count != 0) {
        return AC_ERR_FEATURE_NOT_SUPPORTED; /* OpenCL C takes compile-time values as build options */
    }
    if ((desc->flags & ~AC_KERNEL_SOURCE_BINDING_RANGES) != 0 ||
        ((desc->flags & AC_KERNEL_SOURCE_BINDING_RANGES) && m->binding_count == 0)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    AcClKernel* cl_kernel = NULL;
    AcResult res = ac_cl_kernel_create(ctx->cl_ctx, desc->source, desc->source_size, desc->build_options, m,
                                       (desc->flags & AC_KERNEL_SOURCE_BINDING_RANGES) != 0, &cl_kernel);
    if (res != AC_SUCCESS) {
        return res;
    }
    AcKernel* kernel = (AcKernel*)calloc(1, sizeof(AcKernel));
    if (!kernel) {
        ac_cl_kernel_destroy(cl_kernel);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    kernel->magic = AC_MAGIC_KERNEL;
    kernel->owning_context = ctx;
    kernel->cl_kernel = cl_kernel;
    ctx->active_child_kernels++;
    *out_kernel = kernel;
    return AC_SUCCESS;
}

AcResult ac_runtime_kernel_destroy(AcKernel* kernel) {
    if (!kernel || kernel->magic != AC_MAGIC_KERNEL) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (kernel->vk_kernel) {
        AcResult res = ac_vk_kernel_destroy(kernel->vk_kernel);
        if (res != AC_SUCCESS) {
            return res;
        }
        kernel->vk_kernel = NULL;
    }
    if (kernel->cl_kernel) {
        ac_cl_kernel_destroy(kernel->cl_kernel); /* released after the last recorded dispatch that uses it */
        kernel->cl_kernel = NULL;
    }

    if (kernel->owning_context) {
        if (kernel->owning_context->active_child_kernels > 0) {
            kernel->owning_context->active_child_kernels--;
        }
        kernel->owning_context = NULL;
    }

    kernel->magic = 0;
    free(kernel);
    return AC_SUCCESS;
}

AcResult ac_runtime_kernel_get_pipeline_identity(
    const AcKernel* kernel,
    char* out_hex,
    size_t out_hex_size
) {
    if (out_hex && out_hex_size > 0) {
        out_hex[0] = '\0';
    }
    if (!kernel || kernel->magic != AC_MAGIC_KERNEL || (!kernel->vk_kernel && !kernel->cl_kernel)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_hex) {
        return AC_ERR_NULL_POINTER;
    }
    if (out_hex_size < AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_hex, kernel->cl_kernel ? ac_cl_kernel_identity(kernel->cl_kernel) : kernel->vk_kernel->pipeline_identity,
           AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE);
    return AC_SUCCESS;
}

AcResult ac_runtime_kernel_get_effective_local_size(
    const AcKernel* kernel,
    AcDim3* out_local_size
) {
    if (out_local_size) {
        *out_local_size = (AcDim3){0, 0, 0};
    }
    if (!kernel || kernel->magic != AC_MAGIC_KERNEL || (!kernel->vk_kernel && !kernel->cl_kernel)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_local_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_local_size = kernel->cl_kernel ? ac_cl_kernel_local_size(kernel->cl_kernel) : kernel->vk_kernel->effective_local_size;
    return AC_SUCCESS;
}

AcResult ac_runtime_kernel_get_push_constant_size(
    const AcKernel* kernel,
    size_t* out_size
) {
    if (!out_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_size = 0;
    if (!kernel || kernel->magic != AC_MAGIC_KERNEL || (!kernel->vk_kernel && !kernel->cl_kernel)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    *out_size = kernel->cl_kernel ? ac_cl_kernel_push_constant_size(kernel->cl_kernel)
                                  : (size_t)kernel->vk_kernel->push_constant_size;
    return AC_SUCCESS;
}

/* ========================================================================= */
/* Execution & Synchronization Implementation                                */
/* ========================================================================= */

AcResult ac_runtime_dispatch(AcContext* ctx, const AcLaunchDesc* desc, AcEvent** out_event) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!desc) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->abi_version != AC_ABI_VERSION) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->struct_size != sizeof(AcLaunchDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!desc->kernel || desc->kernel->magic != AC_MAGIC_KERNEL) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->kernel->owning_context != ctx) {
        return AC_ERR_CONTEXT_MISMATCH;
    }
    /* Workgroup dimensions and runtime safety budget validation */
    if (desc->grid_dim.x == 0 || desc->grid_dim.y == 0 || desc->grid_dim.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    AcResult budget_res = ac_validate_dispatch_budget(
        desc->grid_dim,
        AC_MAX_SAFE_GROUPS_PER_AXIS,
        AC_MAX_SAFE_TOTAL_GROUPS
    );
    if (budget_res != AC_SUCCESS) {
        return budget_res;
    }

    if (desc->buffer_count > 0 && !desc->buffers) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->buffer_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (ctx->cl_ctx) {
        /* one batch of one dispatch, every buffer bound whole */
        if (desc->push_constants_size > 128 || (desc->push_constants_size > 0 && !desc->push_constants)) {
            return AC_ERR_INVALID_ARGUMENT;
        }
        AcClBuffer* cl_bufs[16];
        uint64_t offsets[16], ranges[16];
        for (uint32_t i = 0; i < desc->buffer_count; ++i) {
            const AcBuffer* b = desc->buffers[i];
            if (!b || b->magic != AC_MAGIC_BUFFER || !b->cl_buf) {
                return AC_ERR_NULL_POINTER;
            }
            if (b->owning_context != ctx) {
                return AC_ERR_CONTEXT_MISMATCH;
            }
            cl_bufs[i] = b->cl_buf;
            offsets[i] = 0;
            ranges[i] = ac_cl_buffer_size(b->cl_buf);
        }
        AcClBatch* cb = NULL;
        AcResult res = ac_cl_batch_begin(ctx->cl_ctx, 1, 0, &cb);
        if (res != AC_SUCCESS) {
            return res;
        }
        res = ac_cl_batch_dispatch(cb, desc->kernel->cl_kernel, cl_bufs, offsets, ranges, desc->buffer_count,
                                   desc->push_constants, desc->push_constants_size, desc->grid_dim);
        AcClEvent* cl_ev = NULL;
        if (res == AC_SUCCESS) {
            res = ac_cl_batch_submit(cb, &cl_ev);
        }
        ac_cl_batch_destroy(cb);
        if (res != AC_SUCCESS) {
            return res;
        }
        if (out_event == NULL) {
            AcResult wait_res = ac_cl_event_wait(cl_ev, UINT64_MAX);
            ac_cl_event_destroy(cl_ev);
            return wait_res;
        }
        AcEvent* ev = wrap_cl_event(ctx, cl_ev);
        if (!ev) {
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        *out_event = ev;
        return AC_SUCCESS;
    }

    AcVkBuffer* vk_bufs[16];
    for (uint32_t i = 0; i < desc->buffer_count; ++i) {
        if (!desc->buffers[i] || desc->buffers[i]->magic != AC_MAGIC_BUFFER) {
            return AC_ERR_NULL_POINTER;
        }
        if (desc->buffers[i]->owning_context != ctx) {
            return AC_ERR_CONTEXT_MISMATCH;
        }
        vk_bufs[i] = desc->buffers[i]->vk_buf;
    }

    if (desc->push_constants_size > 0 && !desc->push_constants) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->push_constants_size > 128) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (out_event == NULL) {
        /* Synchronous dispatch contract: block indefinitely until execution finishes and verify both wait and destroy */
        AcVkEvent* sync_event = NULL;
        AcResult dispatch_res = ac_vk_dispatch(
            ctx->vk_ctx,
            desc->kernel->vk_kernel,
            vk_bufs,
            desc->buffer_count,
            desc->push_constants,
            desc->push_constants_size,
            desc->grid_dim.x,
            desc->grid_dim.y,
            desc->grid_dim.z,
            &sync_event
        );
        if (dispatch_res != AC_SUCCESS) {
            return dispatch_res;
        }

        AcResult wait_res = ac_vk_event_wait(sync_event, UINT64_MAX);
        AcResult destroy_res = ac_vk_event_destroy(sync_event);

        if (wait_res != AC_SUCCESS) {
            return wait_res;
        }
        if (destroy_res != AC_SUCCESS) {
            return destroy_res;
        }
        return AC_SUCCESS;
    }

    /* Asynchronous dispatch contract */
    AcVkEvent* vk_event = NULL;
    AcResult res = ac_vk_dispatch(
        ctx->vk_ctx,
        desc->kernel->vk_kernel,
        vk_bufs,
        desc->buffer_count,
        desc->push_constants,
        desc->push_constants_size,
        desc->grid_dim.x,
        desc->grid_dim.y,
        desc->grid_dim.z,
        &vk_event
    );
    if (res != AC_SUCCESS) {
        return res;
    }

    AcEvent* ev = (AcEvent*)calloc(1, sizeof(AcEvent));
    if (!ev) {
        ac_vk_event_destroy(vk_event);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    ev->magic = AC_MAGIC_EVENT;
    ev->owning_context = ctx;
    ev->vk_event = vk_event;

    ctx->active_child_events++;
    *out_event = ev;
    return AC_SUCCESS;
}

AcResult ac_runtime_event_wait(AcEvent* event, uint64_t timeout_ns) {
    if (!event || event->magic != AC_MAGIC_EVENT || (!event->vk_event && !event->cl_event)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (event->cl_event) {
        return ac_cl_event_wait(event->cl_event, timeout_ns);
    }
    return ac_vk_event_wait(event->vk_event, timeout_ns);
}

AcResult ac_runtime_event_get_timing(AcEvent* event, uint64_t* out_wall_latency_ns, uint64_t* out_gpu_duration_ns) {
    if (!event || event->magic != AC_MAGIC_EVENT || (!event->vk_event && !event->cl_event)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (event->cl_event) {
        return ac_cl_event_get_timing(event->cl_event, out_wall_latency_ns, out_gpu_duration_ns);
    }
    return ac_vk_event_get_timing(event->vk_event, out_wall_latency_ns, out_gpu_duration_ns);
}

AcResult ac_runtime_event_destroy(AcEvent* event) {
    if (!event || event->magic != AC_MAGIC_EVENT) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (event->vk_event) {
        AcResult res = ac_vk_event_destroy(event->vk_event);
        if (res != AC_SUCCESS) {
            return res;
        }
        event->vk_event = NULL;
    }
    if (event->cl_event) {
        ac_cl_event_destroy(event->cl_event);
        event->cl_event = NULL;
    }

    if (event->owning_context) {
        if (event->owning_context->active_child_events > 0) {
            event->owning_context->active_child_events--;
        }
        event->owning_context = NULL;
    }

    event->magic = 0;
    free(event);
    return AC_SUCCESS;
}

/* ========================================================================= */
/* Command Batching Implementation (Backend-Neutral Public API)              */
/* ========================================================================= */

AcResult ac_runtime_batch_begin(
    AcContext* context,
    AcCommandBatch** out_batch
) {
    if (out_batch) {
        *out_batch = NULL;
    }
    if (!context || !out_batch) {
        return AC_ERR_NULL_POINTER;
    }
    if (context->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }

    AcVkCommandBatch* vk_batch = NULL;
    AcClBatch* cl_batch = NULL;
    AcResult res = context->cl_ctx ? ac_cl_batch_begin(context->cl_ctx, 8, 0, &cl_batch)
                                   : ac_vk_batch_begin(context->vk_ctx, 8, &vk_batch);
    if (res != AC_SUCCESS) {
        return res;
    }

    AcCommandBatch* batch = (AcCommandBatch*)calloc(1, sizeof(AcCommandBatch));
    if (!batch) {
        if (cl_batch) ac_cl_batch_destroy(cl_batch);
        else ac_vk_batch_destroy(vk_batch);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    batch->magic = AC_MAGIC_BATCH;
    batch->owning_context = context;
    batch->vk_batch = vk_batch;
    batch->cl_batch = cl_batch;
    context->active_child_batches++;

    *out_batch = batch;
    return AC_SUCCESS;
}

AcResult ac_runtime_batch_dispatch(
    AcCommandBatch* batch,
    AcKernel* kernel,
    const AcLaunchDesc* launch,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!batch || !kernel || !launch) {
        return AC_ERR_NULL_POINTER;
    }
    if (batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (kernel->magic != AC_MAGIC_KERNEL) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (launch->abi_version != AC_ABI_VERSION || launch->struct_size != sizeof(AcLaunchDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (launch->kernel != kernel) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (kernel->owning_context != batch->owning_context) {
        return AC_ERR_CONTEXT_MISMATCH;
    }

    if (launch->buffer_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        /* logical per-dispatch events are a Vulkan feature of this older entry point; the completion event of
           ac_runtime_batch_submit covers OpenCL batches */
        if (out_event) {
            return AC_ERR_FEATURE_NOT_SUPPORTED;
        }
        AcClBuffer* cl_bufs[16];
        uint64_t offsets[16], ranges[16];
        for (uint32_t i = 0; i < launch->buffer_count; ++i) {
            const AcBuffer* b = launch->buffers[i];
            if (!b || b->magic != AC_MAGIC_BUFFER || !b->cl_buf) {
                return AC_ERR_NULL_POINTER;
            }
            if (b->owning_context != batch->owning_context) {
                return AC_ERR_CONTEXT_MISMATCH;
            }
            cl_bufs[i] = b->cl_buf;
            offsets[i] = 0;
            ranges[i] = ac_cl_buffer_size(b->cl_buf);
        }
        return ac_cl_batch_dispatch(batch->cl_batch, kernel->cl_kernel, cl_bufs, offsets, ranges, launch->buffer_count,
                                    launch->push_constants, launch->push_constants_size, launch->grid_dim);
    }
    AcVkBuffer* vk_bufs[16];
    for (uint32_t i = 0; i < launch->buffer_count; ++i) {
        if (!launch->buffers[i] || launch->buffers[i]->magic != AC_MAGIC_BUFFER) {
            return AC_ERR_NULL_POINTER;
        }
        if (launch->buffers[i]->owning_context != batch->owning_context) {
            return AC_ERR_CONTEXT_MISMATCH;
        }
        vk_bufs[i] = launch->buffers[i]->vk_buf;
    }

    AcVkEvent* vk_event = NULL;
    AcResult res = ac_vk_batch_dispatch(
        batch->vk_batch,
        kernel->vk_kernel,
        vk_bufs,
        launch->buffer_count,
        launch->push_constants,
        launch->push_constants_size,
        launch->grid_dim.x,
        launch->grid_dim.y,
        launch->grid_dim.z,
        out_event ? &vk_event : NULL
    );
    if (res != AC_SUCCESS) {
        return res;
    }

    if (out_event && vk_event) {
        AcEvent* ev = (AcEvent*)calloc(1, sizeof(AcEvent));
        if (!ev) {
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        ev->magic = AC_MAGIC_EVENT;
        ev->owning_context = batch->owning_context;
        ev->vk_event = vk_event;
        batch->owning_context->active_child_events++;
        *out_event = ev;
    }

    return AC_SUCCESS;
}

AcResult ac_runtime_batch_submit(
    AcCommandBatch* batch,
    AcEvent** out_completion_event
) {
    if (out_completion_event) {
        *out_completion_event = NULL;
    }
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        AcClEvent* cl_ev = NULL;
        AcResult res = ac_cl_batch_submit(batch->cl_batch, out_completion_event ? &cl_ev : NULL);
        if (res != AC_SUCCESS) {
            return res;
        }
        if (out_completion_event && cl_ev) {
            AcEvent* ev = wrap_cl_event(batch->owning_context, cl_ev);
            if (!ev) {
                return AC_ERR_OUT_OF_HOST_MEMORY;
            }
            *out_completion_event = ev;
        }
        return AC_SUCCESS;
    }

    AcVkEvent* comp_vk_ev = NULL;
    AcResult res = ac_vk_batch_submit(batch->vk_batch, out_completion_event ? &comp_vk_ev : NULL);
    if (res != AC_SUCCESS) {
        return res;
    }

    if (out_completion_event && comp_vk_ev) {
        AcEvent* ev = (AcEvent*)calloc(1, sizeof(AcEvent));
        if (!ev) {
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        ev->magic = AC_MAGIC_EVENT;
        ev->owning_context = batch->owning_context;
        ev->vk_event = comp_vk_ev;
        batch->owning_context->active_child_events++;
        *out_completion_event = ev;
    }

    return AC_SUCCESS;
}

AcResult ac_runtime_batch_abort(
    AcCommandBatch* batch
) {
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        return ac_cl_batch_abort(batch->cl_batch);
    }
    return ac_vk_batch_abort(batch->vk_batch);
}

void ac_runtime_batch_destroy(
    AcCommandBatch* batch
) {
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return;
    }

    if (batch->vk_batch) {
        ac_vk_batch_destroy(batch->vk_batch);
        batch->vk_batch = NULL;
    }
    if (batch->cl_batch) {
        ac_cl_batch_destroy(batch->cl_batch);
        batch->cl_batch = NULL;
    }

    if (batch->owning_context) {
        if (batch->owning_context->active_child_batches > 0) {
            batch->owning_context->active_child_batches--;
        }
        batch->owning_context = NULL;
    }

    batch->magic = 0;
    free(batch);
}

/* ========================================================================= */
/* Graph Batch Additions                                                     */
/* ========================================================================= */

/* OpenCL contexts keep program binaries in the same calls (a format of their own, checked against device and driver) */
AcResult ac_runtime_pipeline_cache_merge(AcContext* ctx, const void* data, size_t size) {
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT || (!ctx->vk_ctx && !ctx->cl_ctx)) return AC_ERR_INVALID_CONTEXT;
    if (!data) return AC_ERR_NULL_POINTER;
    if (ctx->cl_ctx) return ac_cl_program_cache_merge(ctx->cl_ctx, data, size);
    return ac_vk_pipeline_cache_merge(ctx->vk_ctx, data, size);
}

AcResult ac_runtime_pipeline_cache_get_data(AcContext* ctx, void* data, size_t* inout_size) {
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT || (!ctx->vk_ctx && !ctx->cl_ctx)) return AC_ERR_INVALID_CONTEXT;
    if (!inout_size) return AC_ERR_NULL_POINTER;
    if (ctx->cl_ctx) return ac_cl_program_cache_get_data(ctx->cl_ctx, data, inout_size);
    return ac_vk_pipeline_cache_get_data(ctx->vk_ctx, data, inout_size);
}

AcResult ac_runtime_context_get_limits(const AcContext* ctx, AcDeviceLimits* out_limits) {
    if (!ctx || ctx->magic != AC_MAGIC_CONTEXT || (!ctx->vk_ctx && !ctx->cl_ctx)) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!out_limits) {
        return AC_ERR_NULL_POINTER;
    }
    /* the fields up to timestamp_period_ns form the first version of the struct; later fields are appended */
    const size_t base_size = offsetof(AcDeviceLimits, integer_dot_product);
    if (out_limits->abi_version != AC_ABI_VERSION || out_limits->struct_size < base_size ||
        out_limits->struct_size > sizeof(AcDeviceLimits)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (ctx->cl_ctx) {
        return ac_cl_context_get_limits(ctx->cl_ctx, out_limits);
    }
    const AcVkContext* vk = ctx->vk_ctx;
    const VkPhysicalDeviceLimits* l = &vk->device_properties.limits;
    out_limits->max_storage_buffer_range = l->maxStorageBufferRange;
    out_limits->min_storage_buffer_offset_alignment = l->minStorageBufferOffsetAlignment;
    out_limits->max_memory_allocation_size = vk->max_memory_allocation_size;
    for (int i = 0; i < 3; ++i) {
        out_limits->max_compute_workgroup_count[i] = l->maxComputeWorkGroupCount[i];
        out_limits->max_compute_workgroup_size[i] = l->maxComputeWorkGroupSize[i];
    }
    out_limits->max_compute_workgroup_invocations = l->maxComputeWorkGroupInvocations;
    out_limits->max_compute_shared_memory_size = l->maxComputeSharedMemorySize;
    out_limits->max_push_constants_size = l->maxPushConstantsSize;
    out_limits->subgroup_size = vk->subgroup_size;
    out_limits->timestamps_supported = (l->timestampComputeAndGraphics && vk->timestamp_valid_bits > 0) ? 1u : 0u;
    out_limits->timestamp_period_ns = l->timestampPeriod;
    if (out_limits->struct_size >= offsetof(AcDeviceLimits, integer_dot_product_accelerated) + sizeof(uint32_t)) {
        out_limits->integer_dot_product = vk->integer_dot_product;
        out_limits->integer_dot_product_accelerated = vk->integer_dot_product_accelerated;
    }
    return AC_SUCCESS;
}

AcResult ac_runtime_batch_begin_ex(
    AcContext* context,
    const AcBatchDesc* desc,
    AcCommandBatch** out_batch
) {
    if (out_batch) {
        *out_batch = NULL;
    }
    if (!context || !desc || !out_batch) {
        return AC_ERR_NULL_POINTER;
    }
    if (context->magic != AC_MAGIC_CONTEXT) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (desc->abi_version != AC_ABI_VERSION || desc->struct_size != sizeof(AcBatchDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->max_dispatches == 0 || desc->max_dispatches > AC_BATCH_MAX_DISPATCHES) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (desc->flags & ~(AC_BATCH_FLAG_EXPLICIT_BARRIERS | AC_BATCH_FLAG_TIMESTAMPS | AC_BATCH_FLAG_DISPATCH_TIMESTAMPS)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (context->cl_ctx) {
        AcClBatch* cl_batch = NULL;
        AcResult res = ac_cl_batch_begin(context->cl_ctx, desc->max_dispatches, desc->flags, &cl_batch);
        if (res != AC_SUCCESS) {
            return res;
        }
        AcCommandBatch* batch = (AcCommandBatch*)calloc(1, sizeof(AcCommandBatch));
        if (!batch) {
            ac_cl_batch_destroy(cl_batch);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        batch->magic = AC_MAGIC_BATCH;
        batch->owning_context = context;
        batch->cl_batch = cl_batch;
        context->active_child_batches++;
        *out_batch = batch;
        return AC_SUCCESS;
    }
    uint32_t vk_flags = 0;
    if (desc->flags & AC_BATCH_FLAG_EXPLICIT_BARRIERS) vk_flags |= AC_VK_BATCH_FLAG_EXPLICIT_BARRIERS;
    if (desc->flags & AC_BATCH_FLAG_TIMESTAMPS) vk_flags |= AC_VK_BATCH_FLAG_TIMESTAMPS;
    if (desc->flags & AC_BATCH_FLAG_DISPATCH_TIMESTAMPS) vk_flags |= AC_VK_BATCH_FLAG_DISPATCH_TIMESTAMPS;

    AcVkCommandBatch* vk_batch = NULL;
    AcResult res = ac_vk_batch_begin_ex(context->vk_ctx, desc->max_dispatches, vk_flags, &vk_batch);
    if (res != AC_SUCCESS) {
        return res;
    }
    AcCommandBatch* batch = (AcCommandBatch*)calloc(1, sizeof(AcCommandBatch));
    if (!batch) {
        ac_vk_batch_destroy(vk_batch);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    batch->magic = AC_MAGIC_BATCH;
    batch->owning_context = context;
    batch->vk_batch = vk_batch;
    context->active_child_batches++;
    *out_batch = batch;
    return AC_SUCCESS;
}

AcResult ac_runtime_batch_dispatch_bindings(
    AcCommandBatch* batch,
    AcKernel* kernel,
    const AcBufferBinding* bindings,
    uint32_t binding_count,
    const void* push_constants,
    size_t push_constants_size,
    AcDim3 grid_dim
) {
    if (!batch || !kernel || (binding_count > 0 && !bindings)) {
        return AC_ERR_NULL_POINTER;
    }
    if (batch->magic != AC_MAGIC_BATCH || kernel->magic != AC_MAGIC_KERNEL || (!kernel->vk_kernel && !kernel->cl_kernel)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (kernel->owning_context != batch->owning_context) {
        return AC_ERR_CONTEXT_MISMATCH;
    }
    if (binding_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (push_constants_size > 128 || (push_constants_size > 0 && !push_constants)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        AcClBuffer* cl_bufs[16];
        uint64_t offsets[16], ranges[16];
        for (uint32_t i = 0; i < binding_count; ++i) {
            const AcBuffer* b = bindings[i].buffer;
            if (!b || b->magic != AC_MAGIC_BUFFER || !b->cl_buf) {
                return AC_ERR_NULL_POINTER;
            }
            if (b->owning_context != batch->owning_context) {
                return AC_ERR_CONTEXT_MISMATCH;
            }
            cl_bufs[i] = b->cl_buf;
            offsets[i] = bindings[i].offset;
            ranges[i] = bindings[i].range;
        }
        return ac_cl_batch_dispatch(batch->cl_batch, kernel->cl_kernel, cl_bufs, offsets, ranges, binding_count,
                                    push_constants, push_constants_size, grid_dim);
    }
    AcVkBuffer* vk_bufs[16];
    uint64_t offsets[16];
    uint64_t ranges[16];
    for (uint32_t i = 0; i < binding_count; ++i) {
        const AcBuffer* b = bindings[i].buffer;
        if (!b || b->magic != AC_MAGIC_BUFFER || !b->vk_buf) {
            return AC_ERR_NULL_POINTER;
        }
        if (b->owning_context != batch->owning_context) {
            return AC_ERR_CONTEXT_MISMATCH;
        }
        vk_bufs[i] = b->vk_buf;
        offsets[i] = bindings[i].offset;
        ranges[i] = bindings[i].range;
    }
    return ac_vk_batch_dispatch_ranges(batch->vk_batch, kernel->vk_kernel, vk_bufs, offsets, ranges, binding_count,
                                       push_constants, push_constants_size, grid_dim.x, grid_dim.y, grid_dim.z);
}

AcResult ac_runtime_batch_barrier(AcCommandBatch* batch) {
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        return ac_cl_batch_barrier(batch->cl_batch);
    }
    return ac_vk_batch_barrier(batch->vk_batch);
}

AcResult ac_runtime_batch_flush(AcCommandBatch* batch) {
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        return ac_cl_batch_flush(batch->cl_batch);
    }
    return ac_vk_batch_flush(batch->vk_batch);
}

AcResult ac_runtime_batch_get_dispatch_times(const AcCommandBatch* batch, uint64_t* out_ns, uint32_t capacity,
                                             uint32_t* out_count) {
    if (out_count) {
        *out_count = 0;
    }
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        return ac_cl_batch_get_dispatch_times(batch->cl_batch, out_ns, capacity, out_count);
    }
    return ac_vk_batch_get_dispatch_times(batch->vk_batch, out_ns, capacity, out_count);
}

AcResult ac_runtime_batch_get_stats(const AcCommandBatch* batch, AcBatchStats* out_stats) {
    if (!batch || batch->magic != AC_MAGIC_BATCH) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_stats) {
        return AC_ERR_NULL_POINTER;
    }
    if (out_stats->abi_version != AC_ABI_VERSION || out_stats->struct_size != sizeof(AcBatchStats)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->cl_batch) {
        ac_cl_batch_get_stats(batch->cl_batch, out_stats);
        return AC_SUCCESS;
    }
    out_stats->dispatches = ac_vk_batch_get_dispatch_count(batch->vk_batch);
    out_stats->barriers = ac_vk_batch_get_inserted_barriers(batch->vk_batch);
    out_stats->elided_barriers = ac_vk_batch_get_elided_barriers(batch->vk_batch);
    out_stats->flushes = ac_vk_batch_get_flush_count(batch->vk_batch);
    ac_vk_batch_get_cache_stats(batch->vk_batch, &out_stats->descriptor_cache_hits,
                                &out_stats->descriptor_cache_misses);
    return AC_SUCCESS;
}


