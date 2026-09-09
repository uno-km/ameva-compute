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
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define AC_MAGIC_CONTEXT 0xAC010001
#define AC_MAGIC_BUFFER  0xAC010002
#define AC_MAGIC_KERNEL  0xAC010003
#define AC_MAGIC_EVENT   0xAC010004

struct AcContext {
    uint32_t magic;
    uint32_t abi_version;
    AcBackendKind backend_kind;
    AcVkContext* vk_ctx;
    uint32_t active_child_buffers;
    uint32_t active_child_kernels;
    uint32_t active_child_events;
};

struct AcBuffer {
    uint32_t magic;
    AcContext* owning_context;
    AcVkBuffer* vk_buf;
};

struct AcKernel {
    uint32_t magic;
    AcContext* owning_context;
    AcVkKernel* vk_kernel;
};

struct AcEvent {
    uint32_t magic;
    AcContext* owning_context;
    AcVkEvent* vk_event;
};

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

    /* Strict rejection if any child buffer, kernel, or event is still alive */
    if (ctx->active_child_buffers != 0 ||
        ctx->active_child_kernels != 0 ||
        ctx->active_child_events  != 0) {
        return AC_ERR_RESOURCE_IN_USE;
    }

    if (ctx->vk_ctx) {
        AcResult res = ac_vk_context_destroy(ctx->vk_ctx);
        if (res != AC_SUCCESS) {
            return res;
        }
        ctx->vk_ctx = NULL;
    }

    ctx->magic = 0;
    free(ctx);
    return AC_SUCCESS;
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

    if (buf->owning_context) {
        if (buf->owning_context->active_child_buffers > 0) {
            buf->owning_context->active_child_buffers--;
        }
        buf->owning_context = NULL;
    }

    buf->magic = 0;
    free(buf);
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_get_mapped_ptr(const AcBuffer* buf, void** out_ptr) {
    if (out_ptr) {
        *out_ptr = NULL;
    }
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_ptr) {
        return AC_ERR_NULL_POINTER;
    }
    *out_ptr = buf->vk_buf->mapped_ptr;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_get_size(const AcBuffer* buf, size_t* out_size) {
    if (out_size) {
        *out_size = 0;
    }
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_size = buf->vk_buf->size_bytes;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_get_allocation_size(const AcBuffer* buf, size_t* out_alloc_size) {
    if (out_alloc_size) {
        *out_alloc_size = 0;
    }
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_alloc_size) {
        return AC_ERR_NULL_POINTER;
    }
    *out_alloc_size = buf->vk_buf->allocation_size_bytes;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_is_coherent(const AcBuffer* buf, uint32_t* out_is_coherent) {
    if (out_is_coherent) {
        *out_is_coherent = 0u;
    }
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!out_is_coherent) {
        return AC_ERR_NULL_POINTER;
    }
    *out_is_coherent = buf->vk_buf->is_coherent ? 1u : 0u;
    return AC_SUCCESS;
}

AcResult ac_runtime_buffer_flush(AcBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    return ac_vk_buffer_flush(buf->vk_buf, offset, size_bytes);
}

AcResult ac_runtime_buffer_invalidate(AcBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buf || buf->magic != AC_MAGIC_BUFFER || !buf->vk_buf) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    return ac_vk_buffer_invalidate(buf->vk_buf, offset, size_bytes);
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
    if (!desc->spirv_code) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->code_size_bytes == 0 || (desc->code_size_bytes % 4 != 0)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (!desc->metadata) {
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

    AcVkKernel* vk_kernel = NULL;
    AcResult res = ac_vk_kernel_create_with_metadata(
        ctx->vk_ctx,
        desc->spirv_code,
        desc->code_size_bytes,
        &vk_meta,
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
    if (desc->grid_dim.x == 0 || desc->grid_dim.y == 0 || desc->grid_dim.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }

    if (desc->buffer_count > 0 && !desc->buffers) {
        return AC_ERR_NULL_POINTER;
    }
    if (desc->buffer_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
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
    if (!event || event->magic != AC_MAGIC_EVENT || !event->vk_event) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    return ac_vk_event_wait(event->vk_event, timeout_ns);
}

AcResult ac_runtime_event_get_timing(AcEvent* event, uint64_t* out_wall_latency_ns, uint64_t* out_gpu_duration_ns) {
    if (!event || event->magic != AC_MAGIC_EVENT || !event->vk_event) {
        return AC_ERR_INVALID_ARGUMENT;
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
