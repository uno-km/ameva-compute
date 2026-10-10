/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ac_dispatch.h"
#include "ac_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <new>
#include <unordered_map>
#include <algorithm>

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ========================================================================= */
/* Descriptor set cache (per context)                                        */
/* ========================================================================= */
/* A descriptor set is fully determined by its set layout and the (buffer, offset, range) bound to each slot. Sets
 * are immutable once written, so one set serves every dispatch -- single or batched, now or in a later batch --
 * that binds the same ranges with the same layout. Pools are created with FREE_DESCRIPTOR_SET_BIT so a set can be
 * returned when a buffer or a kernel it references is destroyed (their destroy functions reject destruction while
 * submissions are outstanding, so no pending command buffer can still use such a set). */
namespace {

const uint32_t kDescMaxBindings = 16;
const uint32_t kDescSetsPerPool = 256;
const uint32_t kDescStoragePerPool = 256 * 8;
const uint32_t kDescUniformPerPool = 256;
/* above this many sets the cache is emptied, but only when nothing is recording or in flight */
const size_t kDescCacheTrimThreshold = 16384;

/* Capacity for n elements, grown geometrically. reserve(n) alone allocates exactly n, so reserving size() + k before
 * every k pushes reallocates and copies the whole vector each time: quadratic in the dispatches of one batch (a
 * 19000-dispatch graph spent about 100 us per dispatch here). */
template <class V>
void reserve_geometric(V& v, size_t n) {
    if (v.capacity() < n) v.reserve(std::max(n, 2 * v.capacity()));
}

template <class H>
uint64_t handle_bits(H h) {
    uint64_t v = 0;
    memcpy(&v, &h, sizeof(h) < sizeof(v) ? sizeof(h) : sizeof(v));
    return v;
}

struct DescKey {
    VkDescriptorSetLayout layout;
    uint32_t count;
    VkBuffer buffer[kDescMaxBindings];
    VkDeviceSize offset[kDescMaxBindings];
    VkDeviceSize range[kDescMaxBindings];
};

struct DescKeyHash {
    size_t operator()(const DescKey& k) const noexcept {
        uint64_t h = 1469598103934665603ull;
        const auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix(handle_bits(k.layout));
        mix(k.count);
        for (uint32_t i = 0; i < k.count; ++i) {
            mix(handle_bits(k.buffer[i]));
            mix((uint64_t)k.offset[i]);
            mix((uint64_t)k.range[i]);
        }
        return (size_t)h;
    }
};

struct DescKeyEq {
    bool operator()(const DescKey& a, const DescKey& b) const noexcept {
        if (a.layout != b.layout || a.count != b.count) return false;
        for (uint32_t i = 0; i < a.count; ++i) {
            if (a.buffer[i] != b.buffer[i] || a.offset[i] != b.offset[i] || a.range[i] != b.range[i]) return false;
        }
        return true;
    }
};

struct DescEntry {
    VkDescriptorSet set;
    uint32_t pool;
};

struct DescCache {
    std::unordered_map<DescKey, DescEntry, DescKeyHash, DescKeyEq> map;
    std::vector<VkDescriptorPool> pools;
};

DescCache* desc_cache_of(AcVkContext* ctx, bool create) {
    if (!ctx->desc_cache && create) {
        ctx->desc_cache = new (std::nothrow) DescCache();
    }
    return (DescCache*)ctx->desc_cache;
}

void desc_cache_clear(AcVkContext* ctx, DescCache* c) {
    for (VkDescriptorPool p : c->pools) {
        vkDestroyDescriptorPool(ctx->device, p, NULL);
    }
    c->pools.clear();
    c->map.clear();
}

/* Empties an oversized cache when no command buffer can reference its sets. */
void desc_cache_trim(AcVkContext* ctx) {
    DescCache* c = desc_cache_of(ctx, false);
    if (c && c->map.size() > kDescCacheTrimThreshold && ctx->submission_outstanding_count == 0 &&
        ctx->recording_batches == 0) {
        desc_cache_clear(ctx, c);
    }
}

AcResult desc_cache_get(AcVkContext* ctx, const AcVkKernel* kernel, const VkDescriptorBufferInfo* infos,
                        uint32_t count, VkDescriptorSet* out_set, bool* out_hit) {
    *out_set = VK_NULL_HANDLE;
    *out_hit = false;
    if (count > kDescMaxBindings) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    DescCache* c = desc_cache_of(ctx, true);
    if (!c) {
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    DescKey key;
    memset(&key, 0, sizeof(key));
    key.layout = kernel->descriptor_set_layout;
    key.count = count;
    for (uint32_t i = 0; i < count; ++i) {
        key.buffer[i] = infos[i].buffer;
        key.offset[i] = infos[i].offset;
        key.range[i] = infos[i].range;
    }
    auto it = c->map.find(key);
    if (it != c->map.end()) {
        *out_set = it->second.set;
        *out_hit = true;
        return AC_SUCCESS;
    }

    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &kernel->descriptor_set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult r = VK_ERROR_OUT_OF_POOL_MEMORY;
    if (!c->pools.empty()) {
        alloc.descriptorPool = c->pools.back();
        r = vkAllocateDescriptorSets(ctx->device, &alloc, &set);
    }
    if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
        VkDescriptorPoolSize sizes[2] = {};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[0].descriptorCount = kDescStoragePerPool;
        sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[1].descriptorCount = kDescUniformPerPool;
        VkDescriptorPoolCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets = kDescSetsPerPool;
        pci.poolSizeCount = 2;
        pci.pPoolSizes = sizes;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(ctx->device, &pci, NULL, &pool) != VK_SUCCESS) {
            return AC_ERR_OUT_OF_DEVICE_MEMORY;
        }
        c->pools.push_back(pool);
        alloc.descriptorPool = pool;
        r = vkAllocateDescriptorSets(ctx->device, &alloc, &set);
    }
    if (r != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    VkWriteDescriptorSet writes[kDescMaxBindings] = {};
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = kernel->binding_index[i];
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (VkDescriptorType)kernel->binding_descriptor_type[i];
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, count, writes, 0, NULL);

    DescEntry e;
    e.set = set;
    e.pool = (uint32_t)(c->pools.size() - 1);
    try {
        c->map.emplace(key, e);
    } catch (...) {
        vkFreeDescriptorSets(ctx->device, c->pools[e.pool], 1, &set);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    *out_set = set;
    return AC_SUCCESS;
}

template <class Pred>
void desc_cache_purge_if(AcVkContext* ctx, Pred pred) {
    DescCache* c = ctx ? desc_cache_of(ctx, false) : NULL;
    if (!c || ctx->device == VK_NULL_HANDLE) return;
    for (auto it = c->map.begin(); it != c->map.end();) {
        if (pred(it->first)) {
            vkFreeDescriptorSets(ctx->device, c->pools[it->second.pool], 1, &it->second.set);
            it = c->map.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace

extern "C" {

void ac_vk_desc_cache_purge_buffer(AcVkContext* ctx, VkBuffer buffer) {
    desc_cache_purge_if(ctx, [buffer](const DescKey& k) {
        for (uint32_t i = 0; i < k.count; ++i) {
            if (k.buffer[i] == buffer) return true;
        }
        return false;
    });
}

void ac_vk_desc_cache_purge_layout(AcVkContext* ctx, VkDescriptorSetLayout layout) {
    desc_cache_purge_if(ctx, [layout](const DescKey& k) { return k.layout == layout; });
}

void ac_vk_desc_cache_destroy(AcVkContext* ctx) {
    DescCache* c = ctx ? desc_cache_of(ctx, false) : NULL;
    if (!c) return;
    if (ctx->device != VK_NULL_HANDLE) {
        desc_cache_clear(ctx, c);
    }
    delete c;
    ctx->desc_cache = NULL;
}

AcResult ac_validate_dispatch_grid(
    const AcDim3 grid,
    const uint32_t max_group_count[3]
) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    if (!max_group_count) {
        return AC_ERR_NULL_POINTER;
    }
    if (grid.x > max_group_count[0] ||
        grid.y > max_group_count[1] ||
        grid.z > max_group_count[2]) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    uint64_t total_groups = (uint64_t)grid.x * (uint64_t)grid.y * (uint64_t)grid.z;
    if (total_groups == 0 || total_groups > 0xFFFFFFFFULL) {
        return AC_ERR_NUMERICAL_OVERFLOW;
    }
    return AC_SUCCESS;
}

AcResult ac_validate_dispatch_budget(
    const AcDim3 grid,
    uint32_t max_safe_groups_per_axis,
    uint64_t max_safe_total_groups
) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    if (max_safe_groups_per_axis > 0) {
        if (grid.x > max_safe_groups_per_axis ||
            grid.y > max_safe_groups_per_axis ||
            grid.z > max_safe_groups_per_axis) {
            return AC_ERR_INVALID_WORK_GROUP_SIZE;
        }
    }
    /* Safe 64-bit product overflow detection */
    if ((uint64_t)grid.x > 0 && (uint64_t)grid.y > 0xFFFFFFFFFFFFFFFFULL / (uint64_t)grid.x) {
        return AC_ERR_NUMERICAL_OVERFLOW;
    }
    uint64_t xy = (uint64_t)grid.x * (uint64_t)grid.y;
    if (xy > 0 && (uint64_t)grid.z > 0xFFFFFFFFFFFFFFFFULL / xy) {
        return AC_ERR_NUMERICAL_OVERFLOW;
    }
    uint64_t total_groups = xy * (uint64_t)grid.z;
    if (max_safe_total_groups > 0 && total_groups > max_safe_total_groups) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    return AC_SUCCESS;
}

AcResult ac_vk_dispatch(
    AcVkContext* ctx,
    AcVkKernel* kernel,
    AcVkBuffer** buffers,
    uint32_t buffer_count,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    AcVkEvent** out_event
) {
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    bool has_timestamp = false;
    AcVkEvent* event = NULL;
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    VkFenceCreateInfo fenceInfo = {};
    VkCommandBufferBeginInfo beginInfo = {};
    VkSubmitInfo submitInfo = {};
    AcResult ret = AC_SUCCESS;
    VkResult vk_res = VK_SUCCESS;

    /* Output safety: Set to NULL immediately on entry */
    if (out_event) {
        *out_event = NULL;
    }

    const char* fail_stage = getenv("AMEVA_RUNTIME_FAIL_STAGE");
    if (!fail_stage) fail_stage = getenv("AMEVA_BACKEND_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "dispatch") == 0) {
        fprintf(stderr, "[AMEVA_RUNTIME_FAIL_INJECTION] Simulated failure at stage: dispatch\n");
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }

    if (!ctx || ctx->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!kernel) {
        return AC_ERR_NULL_POINTER;
    }
    /* Context Mismatch: Kernel must belong to this context */
    if (kernel->owning_context != ctx) {
        return AC_ERR_CONTEXT_MISMATCH;
    }
    if (!buffers) {
        return AC_ERR_NULL_POINTER;
    }
    /* Buffer count mismatch */
    if (buffer_count != kernel->buffer_count) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }
    /* Context Mismatch: All buffers must belong to this context */
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        if (!buffers[i]) {
            return AC_ERR_NULL_POINTER;
        }
        if (buffers[i]->owning_context != ctx) {
            return AC_ERR_CONTEXT_MISMATCH;
        }
    }
    /* Push constant size validation */
    if (push_constants_size != kernel->push_constant_size) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }

    /* Workgroup dimensions and hardware limit validation via pure function */
    {
        AcDim3 grid = { group_x, group_y, group_z };
        AcResult grid_res = ac_validate_dispatch_grid(grid, ctx->device_properties.limits.maxComputeWorkGroupCount);
        if (grid_res != AC_SUCCESS) {
            return grid_res;
        }
    }

    /* ------------------------------------------------------------------------
     * Pre-Submit Allocations: Complete all host tracking allocations BEFORE queue submit
     * ------------------------------------------------------------------------ */

    /* 1. Descriptor set: whole buffers (the legacy contract of this entry point), from the context cache */
    desc_cache_trim(ctx);
    {
        VkDescriptorBufferInfo bufferInfos[16] = {};
        for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
            bufferInfos[i].buffer = buffers[i]->buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = buffers[i]->size_bytes;
        }
        bool hit = false;
        ret = desc_cache_get(ctx, kernel, bufferInfos, kernel->buffer_count, &descriptor_set, &hit);
        if (ret != AC_SUCCESS) {
            return ret;
        }
    }

    /* 2. Allocate Command Buffer */
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx->command_pool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    vk_res = vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &command_buffer);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    /* 3. Create Completion Fence */
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_res = vkCreateFence(ctx->device, &fenceInfo, NULL, &fence);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_cmd;
    }

    /* 4. Optional Hardware Timestamp Query Pool */
    if (ctx->device_properties.limits.timestampComputeAndGraphics && ctx->timestamp_valid_bits > 0) {
        VkQueryPoolCreateInfo qpci = {};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        if (vkCreateQueryPool(ctx->device, &qpci, NULL, &query_pool) == VK_SUCCESS) {
            has_timestamp = true;
        }
    }

    /* 5. Record Command Buffer */
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vk_res = vkBeginCommandBuffer(command_buffer, &beginInfo);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_query;
    }

    if (has_timestamp) {
        vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
    }

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->compute_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);

    if (push_constants && push_constants_size > 0) {
        vkCmdPushConstants(command_buffer, kernel->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_constants_size, push_constants);
    }

    vkCmdDispatch(command_buffer, group_x, group_y, group_z);

    /* Device -> host visibility: shader writes must be made available to HOST_READ before the fence
       signals (Vulkan spec, host access to device memory). Without it, readback after the fence wait
       is only correct on drivers that happen to flush caches at submit boundaries. */
    {
        VkMemoryBarrier hostBarrier = {};
        hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &hostBarrier, 0, NULL, 0, NULL);
    }

    if (has_timestamp) {
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
    }

    vk_res = vkEndCommandBuffer(command_buffer);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_query;
    }

    /* 6. Pre-allocate AcVkEvent wrapper before queue submit */
    event = (AcVkEvent*)calloc(1, sizeof(AcVkEvent));
    if (!event) {
        ret = AC_ERR_OUT_OF_HOST_MEMORY;
        goto cleanup_query;
    }

    event->bound_buffers = (AcVkBuffer**)malloc(kernel->buffer_count * sizeof(AcVkBuffer*));
    if (!event->bound_buffers) {
        free(event);
        event = NULL;
        ret = AC_ERR_OUT_OF_HOST_MEMORY;
        goto cleanup_query;
    }
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        event->bound_buffers[i] = buffers[i];
    }
    event->fence = fence;
    event->command_buffer = command_buffer;
    event->descriptor_pool = VK_NULL_HANDLE; /* the set belongs to the context cache */
    event->descriptor_set = descriptor_set;
    event->query_pool = query_pool;
    event->has_timestamp = has_timestamp;
    event->submit_timestamp_ns = get_time_ns();
    event->completion_timestamp_ns = 0;
    event->wall_latency_ns = 0;
    event->gpu_duration_ns = 0;
    event->is_signaled = false;
    event->resources_retired = false;
    event->owning_context = ctx;
    event->bound_kernel = kernel;
    event->bound_buffer_count = kernel->buffer_count;
    event->shared = NULL;

    /* ------------------------------------------------------------------------
     * Increment Outstanding Submission Counters right before submit
     * ------------------------------------------------------------------------ */
    kernel->submission_outstanding_count++;
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        buffers[i]->submission_outstanding_count++;
    }
    ctx->submission_outstanding_count++;

    /* ------------------------------------------------------------------------
     * 7. Submit to Compute Queue
     * ------------------------------------------------------------------------ */
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer;

    vk_res = vkQueueSubmit(ctx->compute_queue, 1, &submitInfo, fence);
    if (vk_res != VK_SUCCESS) {
        /* Fail-Safe Rollback: Revert all incremented counters immediately! */
        kernel->submission_outstanding_count--;
        for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
            buffers[i]->submission_outstanding_count--;
        }
        ctx->submission_outstanding_count--;

        free(event->bound_buffers);
        free(event);
        event = NULL;
        ret = AC_ERR_DRIVER_CRASH_OR_LOST;
        goto cleanup_query;
    }

    /* 8. Return Event or Synchronous Wait */
    if (out_event) {
        *out_event = event;
        return AC_SUCCESS;
    } else {
        AcResult wait_res = ac_vk_event_wait(event, 5000000000ULL);
        ac_vk_event_destroy(event);
        return wait_res;
    }

cleanup_query:
    if (query_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(ctx->device, query_pool, NULL);
    }
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx->device, fence, NULL);
    }
cleanup_cmd:
    if (command_buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command_buffer);
    }
    return ret;
}

} // extern "C"

/* ========================================================================= */
/* Command batches                                                           */
/* ========================================================================= */
/* One command buffer, one fence, one submission for any number of dispatches.
 *
 * Ownership: the Vulkan objects of a batch live in an AcVkSubmission that is reference counted. The batch holds one
 * reference, every event handed out for it (completion event, per-dispatch logical events) holds one more. The
 * fence, command buffer and query pool are released with the last reference, after the GPU has finished, so the
 * order in which the caller destroys batch and events does not matter. (Events used to carry the batch fence and
 * destroy it, after which destroying the batch waited on and destroyed the same fence again.)
 *
 * Retention: every recorded dispatch increments the outstanding count of its kernel and of each bound buffer, so
 * none of them can be destroyed while recorded or in flight; the counts are returned exactly once, when the
 * submission is first observed complete, or when a recording batch is aborted.
 *
 * Hazards: by default (legacy mode) a barrier is inserted before a dispatch that reads or writes a buffer written
 * since the last barrier, or writes (last binding) a buffer read since the last barrier -- whole-buffer granularity.
 * With AC_VK_BATCH_FLAG_EXPLICIT_BARRIERS nothing is inferred: the caller, which knows the byte ranges, inserts
 * ac_vk_batch_barrier() where needed. Every batch ends with a SHADER_WRITE -> HOST_READ barrier. */

struct AcVkSubmission {
    AcVkContext* ctx;
    VkCommandBuffer cmd;                  /* command buffer being recorded (the last one once submitted) */
    std::vector<VkCommandBuffer> flushed; /* earlier command buffers of this batch, already submitted */
    VkFence fence;
    VkQueryPool query_pool;
    bool has_timestamps;
    bool dispatch_timestamps; /* query 2 + i is written after dispatch i */
    int state; /* 0 recording, 1 submitted, 2 aborted */
    bool signaled;
    bool device_lost;
    bool retired;
    bool in_flight;       /* a command buffer of this batch reached the queue (flush or submit) */
    bool drained;         /* the work of an aborted batch that had flushed command buffers is known complete */
    bool drain_submitted; /* the fence was submitted (empty batch) to wait for that work */
    uint32_t refs;
    uint64_t submit_ns;
    uint64_t wall_ns;
    uint64_t gpu_ns;
    std::vector<AcVkBuffer*> buffers; /* one entry per (dispatch, binding) */
    std::vector<AcVkKernel*> kernels; /* one entry per dispatch */
};

enum { SUB_RECORDING = 0, SUB_SUBMITTED = 1, SUB_ABORTED = 2 };

struct AcVkCommandBatch {
    AcVkContext* ctx;
    AcVkSubmission* sub;
    uint32_t flags;
    uint32_t max_dispatches;
    uint32_t dispatch_count;
    uint32_t inserted_barriers;
    uint32_t elided_barriers;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint32_t flushes;
    uint32_t dispatches_at_flush;
    bool recording_counted; /* contributes to ctx->recording_batches */
    std::vector<VkBuffer> written;
    std::vector<VkBuffer> read;
    VkPipeline last_pipeline;
    VkPipelineLayout last_layout;
    VkDescriptorSet last_set;
};

static void sub_retire(AcVkSubmission* s) {
    if (s->retired) return;
    s->retired = true;
    for (AcVkKernel* k : s->kernels) {
        if (k->submission_outstanding_count > 0) k->submission_outstanding_count--;
    }
    for (AcVkBuffer* b : s->buffers) {
        if (b->submission_outstanding_count > 0) b->submission_outstanding_count--;
    }
    if (s->in_flight && s->ctx->submission_outstanding_count > 0) {
        s->ctx->submission_outstanding_count--;
    }
}

/* An aborted batch may already have flushed command buffers on the queue. Nothing it references may be released
 * before they finish: an empty submission carrying the batch fence signals only after every earlier submission of the
 * queue (fence signal operations include all commands earlier in submission order). */
static bool sub_drain(AcVkSubmission* s) {
    if (s->drained || s->flushed.empty() || s->state == SUB_SUBMITTED) return true;
    VkResult r = VK_SUCCESS;
    if (!s->drain_submitted) { // a fence may be submitted only while unsignaled and not pending: once
        r = vkQueueSubmit(s->ctx->compute_queue, 0, NULL, s->fence);
        if (r == VK_SUCCESS) s->drain_submitted = true;
    }
    if (r == VK_SUCCESS) r = vkWaitForFences(s->ctx->device, 1, &s->fence, VK_TRUE, 5000000000ULL);
    if (r != VK_SUCCESS && r != VK_ERROR_DEVICE_LOST) {
        fprintf(stderr, "ac_vk_batch: flushed work of an aborted batch did not complete (VkResult %d)\n", (int)r);
        return false;
    }
    s->drained = true;
    return true;
}

static void sub_free_vk(AcVkSubmission* s) {
    VkDevice dev = s->ctx->device;
    if (s->query_pool != VK_NULL_HANDLE) vkDestroyQueryPool(dev, s->query_pool, NULL);
    if (s->fence != VK_NULL_HANDLE) vkDestroyFence(dev, s->fence, NULL);
    if (s->cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(dev, s->ctx->command_pool, 1, &s->cmd);
    if (!s->flushed.empty()) {
        vkFreeCommandBuffers(dev, s->ctx->command_pool, (uint32_t)s->flushed.size(), s->flushed.data());
        s->flushed.clear();
    }
    s->query_pool = VK_NULL_HANDLE;
    s->fence = VK_NULL_HANDLE;
    s->cmd = VK_NULL_HANDLE;
}

static void batch_end_recording(AcVkCommandBatch* b) {
    if (b->recording_counted) {
        b->recording_counted = false;
        if (b->ctx->recording_batches > 0) b->ctx->recording_batches--;
    }
}

static void batch_barrier_record(AcVkCommandBatch* b) {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(b->sub->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &mb, 0, NULL, 0, NULL);
    b->inserted_barriers++;
    b->written.clear();
    b->read.clear();
}

static bool contains(const std::vector<VkBuffer>& v, VkBuffer x) {
    for (VkBuffer y : v) {
        if (y == x) return true;
    }
    return false;
}

static AcVkEvent* shared_event_new(AcVkSubmission* s) {
    AcVkEvent* ev = (AcVkEvent*)calloc(1, sizeof(AcVkEvent));
    if (!ev) return NULL;
    ev->owning_context = s->ctx;
    ev->has_timestamp = s->has_timestamps;
    ev->shared = s;
    s->refs++;
    return ev;
}

/* Records one dispatch with fully specified descriptor ranges (validated by the caller). */
static AcResult batch_record(AcVkCommandBatch* b, AcVkKernel* kernel, AcVkBuffer** bufs,
                             const VkDescriptorBufferInfo* infos, uint32_t count, const void* push_constants,
                             size_t push_constants_size, uint32_t gx, uint32_t gy, uint32_t gz,
                             AcVkEvent** out_event) {
    if (out_event) *out_event = NULL;
    if (b->sub->state != SUB_RECORDING) return AC_ERR_INVALID_ARGUMENT;
    if (b->dispatch_count >= b->max_dispatches) return AC_ERR_DEVICE_BUSY;
    if (kernel->owning_context != b->ctx) return AC_ERR_CONTEXT_MISMATCH;
    if (count != kernel->buffer_count) return AC_ERR_KERNEL_ARG_INVALID;
    if (push_constants_size != kernel->push_constant_size) return AC_ERR_KERNEL_ARG_INVALID;
    if (push_constants_size > 0 && !push_constants) return AC_ERR_NULL_POINTER;
    for (uint32_t i = 0; i < count; ++i) {
        if (!bufs[i] || bufs[i]->owning_context != b->ctx) return AC_ERR_CONTEXT_MISMATCH;
    }
    AcDim3 grid = { gx, gy, gz };
    AcResult r = ac_validate_dispatch_grid(grid, b->ctx->device_properties.limits.maxComputeWorkGroupCount);
    if (r != AC_SUCCESS) return r;
    r = ac_validate_dispatch_budget(grid, AC_MAX_SAFE_GROUPS_PER_AXIS, AC_MAX_SAFE_TOTAL_GROUPS);
    if (r != AC_SUCCESS) return r;

    VkDescriptorSet set = VK_NULL_HANDLE;
    bool hit = false;
    r = desc_cache_get(b->ctx, kernel, infos, count, &set, &hit);
    if (r != AC_SUCCESS) return r;
    if (hit) b->cache_hits++;
    else b->cache_misses++;

    AcVkEvent* ev = NULL;
    if (out_event) {
        ev = shared_event_new(b->sub);
        if (!ev) return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    try {
        b->sub->kernels.push_back(kernel);
        reserve_geometric(b->sub->buffers, b->sub->buffers.size() + count);
    } catch (...) {
        if (ev) {
            b->sub->refs--;
            free(ev);
        }
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    if (!(b->flags & AC_VK_BATCH_FLAG_EXPLICIT_BARRIERS) && count > 0) {
        const VkBuffer w = infos[count - 1].buffer;
        bool hazard = contains(b->read, w);
        for (uint32_t i = 0; i < count && !hazard; ++i) hazard = contains(b->written, infos[i].buffer);
        if (hazard) {
            batch_barrier_record(b);
        } else {
            b->elided_barriers++;
        }
        for (uint32_t i = 0; i + 1 < count; ++i) {
            if (!contains(b->read, infos[i].buffer)) b->read.push_back(infos[i].buffer);
        }
        if (!contains(b->written, w)) b->written.push_back(w);
    }

    VkCommandBuffer cmd = b->sub->cmd;
    if (kernel->compute_pipeline != b->last_pipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->compute_pipeline);
        b->last_pipeline = kernel->compute_pipeline;
    }
    if (set != b->last_set || kernel->pipeline_layout != b->last_layout) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline_layout, 0, 1, &set, 0, NULL);
        b->last_set = set;
        b->last_layout = kernel->pipeline_layout;
    }
    if (push_constants_size > 0) {
        vkCmdPushConstants(cmd, kernel->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_constants_size,
                           push_constants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);
    if (b->sub->dispatch_timestamps) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b->sub->query_pool, 2 + b->dispatch_count);
    }

    kernel->submission_outstanding_count++;
    for (uint32_t i = 0; i < count; ++i) {
        bufs[i]->submission_outstanding_count++;
        b->sub->buffers.push_back(bufs[i]);
    }
    b->dispatch_count++;
    if (out_event) *out_event = ev;
    return AC_SUCCESS;
}

extern "C" {

bool ac_vk_submission_is_signaled(const AcVkSubmission* s) {
    return s && s->signaled;
}

bool ac_vk_submission_is_submitted(const AcVkSubmission* s) {
    return s && s->state == SUB_SUBMITTED;
}

AcResult ac_vk_submission_wait(AcVkSubmission* s, uint64_t timeout_ns) {
    if (!s) return AC_ERR_NULL_POINTER;
    if (s->state != SUB_SUBMITTED) return AC_ERR_INVALID_ARGUMENT;
    if (s->signaled) return AC_SUCCESS;
    if (s->device_lost) return AC_ERR_DRIVER_CRASH_OR_LOST;
    VkResult r = vkWaitForFences(s->ctx->device, 1, &s->fence, VK_TRUE, timeout_ns);
    if (r == VK_TIMEOUT) return AC_ERR_LAUNCH_TIMEOUT;
    if (r == VK_ERROR_DEVICE_LOST) {
        s->device_lost = true;
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    if (r != VK_SUCCESS) {
        return r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? AC_ERR_OUT_OF_DEVICE_MEMORY : AC_ERR_OUT_OF_HOST_MEMORY;
    }
    s->signaled = true;
    const uint64_t now = get_time_ns();
    s->wall_ns = now >= s->submit_ns ? now - s->submit_ns : 0;
    if (s->has_timestamps) {
        uint64_t ts[2] = {0, 0};
        if (vkGetQueryPoolResults(s->ctx->device, s->query_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            const uint32_t bits = s->ctx->timestamp_valid_bits;
            const uint64_t mask = (bits >= 64 || bits == 0) ? ~0ULL : ((1ULL << bits) - 1ULL);
            s->gpu_ns = (uint64_t)((double)((ts[1] - ts[0]) & mask) *
                                   (double)s->ctx->device_properties.limits.timestampPeriod);
        } else {
            s->has_timestamps = false;
        }
    }
    sub_retire(s);
    return AC_SUCCESS;
}

void ac_vk_submission_get_timing(const AcVkSubmission* s, uint64_t* wall, uint64_t* gpu, bool* has_ts) {
    if (wall) *wall = s ? s->wall_ns : 0;
    if (gpu) *gpu = s ? s->gpu_ns : 0;
    if (has_ts) *has_ts = s ? s->has_timestamps : false;
}

void ac_vk_submission_release(AcVkSubmission* s) {
    if (!s) return;
    if (s->refs > 0) s->refs--;
    if (s->refs > 0) return;
    if (s->state == SUB_SUBMITTED && !s->signaled && !s->device_lost) {
        AcResult r = ac_vk_submission_wait(s, 5000000000ULL);
        if (r != AC_SUCCESS && !s->device_lost) {
            /* still pending as far as anyone can tell: freeing it would be undefined behaviour */
            fprintf(stderr, "ac_vk_batch: submission not complete at release (code %d); its command buffer, fence "
                            "and %zu buffer references are leaked\n", (int)r, s->buffers.size());
            return;
        }
    }
    if (!sub_drain(s)) {
        fprintf(stderr, "ac_vk_batch: %zu flushed command buffers and %zu buffer references are leaked\n",
                s->flushed.size(), s->buffers.size());
        return;
    }
    sub_retire(s);
    sub_free_vk(s);
    delete s;
}

AcResult ac_vk_batch_begin_ex(
    AcVkContext* ctx,
    uint32_t max_dispatches,
    uint32_t flags,
    AcVkCommandBatch** out_batch
) {
    if (out_batch) *out_batch = NULL;
    if (!ctx || !out_batch) return AC_ERR_NULL_POINTER;
    if (ctx->device == VK_NULL_HANDLE || ctx->command_pool == VK_NULL_HANDLE) return AC_ERR_INVALID_CONTEXT;
    if (max_dispatches == 0 || max_dispatches > AC_VK_BATCH_MAX_DISPATCHES) return AC_ERR_INVALID_ARGUMENT;
    if (flags & ~(AC_VK_BATCH_FLAG_EXPLICIT_BARRIERS | AC_VK_BATCH_FLAG_TIMESTAMPS |
                  AC_VK_BATCH_FLAG_DISPATCH_TIMESTAMPS)) return AC_ERR_INVALID_ARGUMENT;

    desc_cache_trim(ctx);

    AcVkSubmission* s = new (std::nothrow) AcVkSubmission();
    AcVkCommandBatch* b = new (std::nothrow) AcVkCommandBatch();
    if (!s || !b) {
        delete s;
        delete b;
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    s->ctx = ctx;
    s->cmd = VK_NULL_HANDLE;
    s->fence = VK_NULL_HANDLE;
    s->query_pool = VK_NULL_HANDLE;
    s->has_timestamps = false;
    s->dispatch_timestamps = false;
    s->state = SUB_RECORDING;
    s->signaled = false;
    s->device_lost = false;
    s->retired = false;
    s->in_flight = false;
    s->drained = false;
    s->drain_submitted = false;
    s->refs = 1;
    s->submit_ns = 0;
    s->wall_ns = 0;
    s->gpu_ns = 0;

    VkCommandBufferAllocateInfo cai = {};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = ctx->command_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkAllocateCommandBuffers(ctx->device, &cai, &s->cmd) != VK_SUCCESS ||
        vkCreateFence(ctx->device, &fci, NULL, &s->fence) != VK_SUCCESS) {
        sub_free_vk(s);
        delete s;
        delete b;
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }
    const bool per_dispatch = (flags & AC_VK_BATCH_FLAG_DISPATCH_TIMESTAMPS) != 0;
    uint32_t query_count = 2; // batch begin, batch end
    if (((flags & AC_VK_BATCH_FLAG_TIMESTAMPS) || per_dispatch) &&
        ctx->device_properties.limits.timestampComputeAndGraphics && ctx->timestamp_valid_bits > 0) {
        if (per_dispatch) query_count += max_dispatches; // then one after every dispatch
        VkQueryPoolCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = query_count;
        if (vkCreateQueryPool(ctx->device, &qci, NULL, &s->query_pool) == VK_SUCCESS) {
            s->has_timestamps = true;
            s->dispatch_timestamps = per_dispatch;
        }
    } else if (per_dispatch) {
        sub_free_vk(s);
        delete s;
        delete b;
        return AC_ERR_FEATURE_NOT_SUPPORTED; // asked for, not available: no silent profile without data
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(s->cmd, &bi) != VK_SUCCESS) {
        sub_free_vk(s);
        delete s;
        delete b;
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    if (s->has_timestamps) {
        vkCmdResetQueryPool(s->cmd, s->query_pool, 0, query_count);
        vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s->query_pool, 0);
    }

    b->ctx = ctx;
    b->sub = s;
    b->flags = flags;
    b->max_dispatches = max_dispatches;
    b->dispatch_count = 0;
    b->inserted_barriers = 0;
    b->elided_barriers = 0;
    b->cache_hits = 0;
    b->cache_misses = 0;
    b->flushes = 0;
    b->dispatches_at_flush = 0;
    b->recording_counted = true;
    b->last_pipeline = VK_NULL_HANDLE;
    b->last_layout = VK_NULL_HANDLE;
    b->last_set = VK_NULL_HANDLE;
    ctx->recording_batches++;
    *out_batch = b;
    return AC_SUCCESS;
}

AcResult ac_vk_batch_begin(
    AcVkContext* ctx,
    uint32_t max_dispatches,
    AcVkCommandBatch** out_batch
) {
    return ac_vk_batch_begin_ex(ctx, max_dispatches > 0 ? max_dispatches : 8, 0, out_batch);
}

AcResult ac_vk_batch_dispatch(
    AcVkCommandBatch* batch,
    AcVkKernel* kernel,
    AcVkBuffer** buffers,
    uint32_t buffer_count,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    AcVkEvent** out_event
) {
    if (out_event) *out_event = NULL;
    if (!batch || !kernel || (buffer_count > 0 && !buffers)) return AC_ERR_NULL_POINTER;
    if (buffer_count > kDescMaxBindings) return AC_ERR_INVALID_ARGUMENT;
    /* legacy contract: every buffer bound whole */
    VkDescriptorBufferInfo infos[kDescMaxBindings] = {};
    for (uint32_t i = 0; i < buffer_count; ++i) {
        if (!buffers[i]) return AC_ERR_NULL_POINTER;
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->size_bytes;
    }
    return batch_record(batch, kernel, buffers, infos, buffer_count, push_constants, push_constants_size, group_x,
                        group_y, group_z, out_event);
}

AcResult ac_vk_batch_dispatch_ranges(
    AcVkCommandBatch* batch,
    AcVkKernel* kernel,
    AcVkBuffer** buffers,
    const uint64_t* offsets,
    const uint64_t* ranges,
    uint32_t buffer_count,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z
) {
    if (!batch || !kernel || !buffers || !offsets || !ranges) return AC_ERR_NULL_POINTER;
    if (buffer_count > kDescMaxBindings) return AC_ERR_INVALID_ARGUMENT;
    const VkPhysicalDeviceLimits& lim = batch->ctx->device_properties.limits;
    VkDescriptorBufferInfo infos[kDescMaxBindings] = {};
    for (uint32_t i = 0; i < buffer_count; ++i) {
        const AcVkBuffer* bf = buffers[i];
        if (!bf) return AC_ERR_NULL_POINTER;
        const VkDeviceSize align = kernel->binding_descriptor_type[i] == (uint32_t)VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                       ? lim.minUniformBufferOffsetAlignment
                                       : lim.minStorageBufferOffsetAlignment;
        const VkDeviceSize max_range = kernel->binding_descriptor_type[i] == (uint32_t)VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                           ? lim.maxUniformBufferRange
                                           : lim.maxStorageBufferRange;
        if (align > 0 && offsets[i] % align != 0) return AC_ERR_BUFFER_ALIGNMENT_INVALID;
        if (ranges[i] == 0 || offsets[i] >= bf->size_bytes || ranges[i] > bf->size_bytes - offsets[i])
            return AC_ERR_INVALID_BUFFER_SIZE;
        if (ranges[i] > max_range) return AC_ERR_INVALID_BUFFER_SIZE;
        infos[i].buffer = bf->buffer;
        infos[i].offset = offsets[i];
        infos[i].range = ranges[i];
    }
    return batch_record(batch, kernel, buffers, infos, buffer_count, push_constants, push_constants_size, group_x,
                        group_y, group_z, NULL);
}

AcResult ac_vk_batch_barrier(AcVkCommandBatch* batch) {
    if (!batch) return AC_ERR_NULL_POINTER;
    if (batch->sub->state != SUB_RECORDING) return AC_ERR_INVALID_ARGUMENT;
    batch_barrier_record(batch);
    return AC_SUCCESS;
}

/* Sends what has been recorded so far to the queue and continues recording into a fresh command buffer. Long graphs
 * are cut into submissions of bounded GPU time this way (mobile drivers reset the GPU when one submission runs for
 * seconds), and the GPU starts while the rest is still being recorded. Ordering needs nothing extra: a barrier recorded
 * later has every earlier command of the queue in its first synchronization scope, and the final fence signals after
 * all earlier submissions. The hazard state of the batch carries over unchanged. */
AcResult ac_vk_batch_flush(AcVkCommandBatch* batch) {
    if (!batch) return AC_ERR_NULL_POINTER;
    AcVkSubmission* s = batch->sub;
    if (s->state != SUB_RECORDING) return AC_ERR_INVALID_ARGUMENT;
    if (batch->dispatch_count == batch->dispatches_at_flush) return AC_SUCCESS; // nothing new to send
    try {
        reserve_geometric(s->flushed, s->flushed.size() + 1);
    } catch (...) {
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    VkCommandBufferAllocateInfo cai = {};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = batch->ctx->command_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer next = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(batch->ctx->device, &cai, &next) != VK_SUCCESS) return AC_ERR_OUT_OF_DEVICE_MEMORY;

    VkResult r = vkEndCommandBuffer(s->cmd);
    if (r == VK_SUCCESS) {
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s->cmd;
        if (!s->in_flight) s->submit_ns = get_time_ns();
        r = vkQueueSubmit(batch->ctx->compute_queue, 1, &si, VK_NULL_HANDLE);
    }
    if (r != VK_SUCCESS) {
        /* the batch cannot continue: earlier flushed work may be running, the caller must destroy the batch */
        vkFreeCommandBuffers(batch->ctx->device, batch->ctx->command_pool, 1, &next);
        s->state = SUB_ABORTED;
        batch_end_recording(batch);
        if (sub_drain(s)) sub_retire(s);
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    if (!s->in_flight) {
        s->in_flight = true;
        batch->ctx->submission_outstanding_count++;
    }
    s->flushed.push_back(s->cmd);
    s->cmd = next;
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(s->cmd, &bi) != VK_SUCCESS) {
        s->state = SUB_ABORTED;
        batch_end_recording(batch);
        if (sub_drain(s)) sub_retire(s);
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    /* bound state does not survive a command buffer boundary */
    batch->last_pipeline = VK_NULL_HANDLE;
    batch->last_layout = VK_NULL_HANDLE;
    batch->last_set = VK_NULL_HANDLE;
    batch->dispatches_at_flush = batch->dispatch_count;
    batch->flushes++;
    return AC_SUCCESS;
}

AcResult ac_vk_batch_submit(
    AcVkCommandBatch* batch,
    AcVkEvent** out_completion_event
) {
    if (out_completion_event) *out_completion_event = NULL;
    if (!batch) return AC_ERR_NULL_POINTER;
    AcVkSubmission* s = batch->sub;
    if (s->state != SUB_RECORDING) return AC_ERR_INVALID_ARGUMENT;

    AcVkEvent* ev = NULL;
    if (out_completion_event) {
        ev = shared_event_new(s);
        if (!ev) return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    VkMemoryBarrier hb = {};
    hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(s->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL,
                         0, NULL);
    if (s->has_timestamps) {
        vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->query_pool, 1);
    }
    VkResult r = vkEndCommandBuffer(s->cmd);
    if (r == VK_SUCCESS) {
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s->cmd;
        if (!s->in_flight) s->submit_ns = get_time_ns();
        r = vkQueueSubmit(batch->ctx->compute_queue, 1, &si, s->fence);
    }
    batch_end_recording(batch);
    if (r != VK_SUCCESS) {
        /* the last command buffer did not reach the GPU; flushed ones may still run and are waited for before the
           retained counts are returned. The Vulkan objects go with the batch. */
        s->state = SUB_ABORTED;
        if (sub_drain(s)) sub_retire(s);
        if (ev) {
            s->refs--;
            free(ev);
        }
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    s->state = SUB_SUBMITTED;
    if (!s->in_flight) {
        s->in_flight = true;
        batch->ctx->submission_outstanding_count++;
    }
    if (out_completion_event) *out_completion_event = ev;
    return AC_SUCCESS;
}

AcResult ac_vk_batch_abort(AcVkCommandBatch* batch) {
    if (!batch) return AC_ERR_NULL_POINTER;
    AcVkSubmission* s = batch->sub;
    if (s->state == SUB_ABORTED) return AC_SUCCESS;
    if (s->state != SUB_RECORDING) return AC_ERR_INVALID_ARGUMENT;
    s->state = SUB_ABORTED;
    /* flushed command buffers may be running: their resources stay retained until they are done */
    if (sub_drain(s)) sub_retire(s);
    batch_end_recording(batch);
    return AC_SUCCESS;
}

uint32_t ac_vk_batch_get_flush_count(const AcVkCommandBatch* batch) {
    return batch ? batch->flushes : 0;
}

AcResult ac_vk_batch_get_dispatch_times(const AcVkCommandBatch* batch, uint64_t* out_ns, uint32_t capacity,
                                        uint32_t* out_count) {
    if (out_count) *out_count = 0;
    if (!batch || !out_ns || !out_count) return AC_ERR_NULL_POINTER;
    const AcVkSubmission* s = batch->sub;
    if (!s->dispatch_timestamps) return AC_ERR_FEATURE_NOT_SUPPORTED;
    if (!s->signaled) return AC_ERR_DEVICE_BUSY;
    const uint32_t n = batch->dispatch_count;
    if (capacity < n) return AC_ERR_INVALID_ARGUMENT;
    if (n == 0) return AC_SUCCESS;
    std::vector<uint64_t> ts;
    try {
        ts.resize((size_t)n + 1);
    } catch (...) {
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    VkResult r = vkGetQueryPoolResults(s->ctx->device, s->query_pool, 0, 1, sizeof(uint64_t), &ts[0], sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS) {
        r = vkGetQueryPoolResults(s->ctx->device, s->query_pool, 2, n, (size_t)n * sizeof(uint64_t), &ts[1],
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }
    if (r != VK_SUCCESS) return AC_ERR_DRIVER_CRASH_OR_LOST;
    const uint32_t bits = s->ctx->timestamp_valid_bits;
    const uint64_t mask = (bits >= 64 || bits == 0) ? ~0ULL : ((1ULL << bits) - 1ULL);
    const double period = (double)s->ctx->device_properties.limits.timestampPeriod;
    for (uint32_t i = 0; i < n; ++i) {
        out_ns[i] = (uint64_t)((double)((ts[i + 1] - ts[i]) & mask) * period);
    }
    *out_count = n;
    return AC_SUCCESS;
}

void ac_vk_batch_destroy(AcVkCommandBatch* batch) {
    if (!batch) return;
    if (batch->sub->state == SUB_RECORDING) {
        ac_vk_batch_abort(batch);
    }
    batch_end_recording(batch);
    ac_vk_submission_release(batch->sub);
    batch->sub = NULL;
    delete batch;
}

uint32_t ac_vk_batch_get_inserted_barriers(const AcVkCommandBatch* batch) {
    return batch ? batch->inserted_barriers : 0;
}

uint32_t ac_vk_batch_get_elided_barriers(const AcVkCommandBatch* batch) {
    return batch ? batch->elided_barriers : 0;
}

uint32_t ac_vk_batch_get_dispatch_count(const AcVkCommandBatch* batch) {
    return batch ? batch->dispatch_count : 0;
}

void ac_vk_batch_get_cache_stats(const AcVkCommandBatch* batch, uint64_t* hits, uint64_t* misses) {
    if (hits) *hits = batch ? batch->cache_hits : 0;
    if (misses) *misses = batch ? batch->cache_misses : 0;
}

} // extern "C"
