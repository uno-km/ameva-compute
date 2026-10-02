#include "ac_dispatch.h"
#include "ac_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <new>

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

extern "C" {


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
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    bool has_timestamp = false;
    AcVkEvent* event = NULL;
    VkDescriptorPoolSize poolSize = {};
    VkDescriptorPoolCreateInfo poolInfo = {};
    VkDescriptorSetAllocateInfo allocSetInfo = {};
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

    /* 1. Allocate Descriptor Pool */
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = kernel->buffer_count;

    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vk_res = vkCreateDescriptorPool(ctx->device, &poolInfo, NULL, &descriptor_pool);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    /* 2. Allocate Descriptor Set */
    allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSetInfo.descriptorPool = descriptor_pool;
    allocSetInfo.descriptorSetCount = 1;
    allocSetInfo.pSetLayouts = &kernel->descriptor_set_layout;

    vk_res = vkAllocateDescriptorSets(ctx->device, &allocSetInfo, &descriptor_set);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_pool;
    }

    /* 3. Update Descriptor Sets */
    {
        VkDescriptorBufferInfo bufferInfos[16] = {};
        VkWriteDescriptorSet descriptorWrites[16] = {};

        for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
            bufferInfos[i].buffer = buffers[i]->buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = buffers[i]->size_bytes;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptor_set;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }

        vkUpdateDescriptorSets(ctx->device, kernel->buffer_count, descriptorWrites, 0, NULL);
    }

    /* 4. Allocate Command Buffer */
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx->command_pool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    vk_res = vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &command_buffer);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_pool;
    }

    /* 5. Create Completion Fence */
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_res = vkCreateFence(ctx->device, &fenceInfo, NULL, &fence);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup_cmd;
    }

    /* 6. Optional Hardware Timestamp Query Pool */
    if (ctx->device_properties.limits.timestampComputeAndGraphics && ctx->timestamp_valid_bits > 0) {
        VkQueryPoolCreateInfo qpci = {};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        if (vkCreateQueryPool(ctx->device, &qpci, NULL, &query_pool) == VK_SUCCESS) {
            has_timestamp = true;
        }
    }

    /* 7. Record Command Buffer */
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

    /* 8. Pre-allocate AcVkEvent wrapper before queue submit */
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
    event->descriptor_pool = descriptor_pool;
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

    /* ------------------------------------------------------------------------
     * Increment Outstanding Submission Counters right before submit
     * ------------------------------------------------------------------------ */
    kernel->submission_outstanding_count++;
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        buffers[i]->submission_outstanding_count++;
    }
    ctx->submission_outstanding_count++;

    /* ------------------------------------------------------------------------
     * 9. Submit to Compute Queue
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

    /* 10. Return Event or Synchronous Wait */
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
cleanup_pool:
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, descriptor_pool, NULL);
    }
    return ret;
}

/* ========================================================================= */
/* Vulkan Batch Internal Implementation (Phase B1 Batch Architecture)        */
/* ========================================================================= */

enum AcVkBatchState {
    AC_VK_BATCH_RECORDING = 0,
    AC_VK_BATCH_SUBMITTED = 1,
    AC_VK_BATCH_ABORTED   = 2,
    AC_VK_BATCH_COMPLETED = 3
};

struct AcVkCommandBatch {
    AcVkContext* ctx;
    AcVkBatchState state;
    uint32_t dispatch_count;
    uint32_t max_dispatches;
    VkCommandBuffer command_buffer;
    VkFence fence;
    bool fence_signaled;

    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<AcVkBuffer*> retained_buffers;
    std::vector<AcVkKernel*> retained_kernels;
    std::vector<AcVkEvent*> logical_events;
    std::vector<AcVkBuffer*> written_buffers;

    uint32_t inserted_barriers;
    uint32_t elided_barriers;
};

AcResult ac_vk_batch_begin(
    AcVkContext* ctx,
    uint32_t max_dispatches,
    AcVkCommandBatch** out_batch
) {
    if (out_batch) {
        *out_batch = NULL;
    }
    if (!ctx || !out_batch) {
        return AC_ERR_NULL_POINTER;
    }
    if (ctx->device == VK_NULL_HANDLE || ctx->command_pool == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx->command_pool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult res = vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &cmd);
    if (res != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0; // Unsignaled
    VkFence fence = VK_NULL_HANDLE;
    res = vkCreateFence(ctx->device, &fenceInfo, NULL, &fence);
    if (res != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmd, &beginInfo);
    if (res != VK_SUCCESS) {
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }

    AcVkCommandBatch* batch = new (std::nothrow) AcVkCommandBatch();
    if (!batch) {
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    batch->ctx = ctx;
    batch->state = AC_VK_BATCH_RECORDING;
    batch->dispatch_count = 0;
    batch->max_dispatches = (max_dispatches > 0) ? max_dispatches : 8;
    batch->command_buffer = cmd;
    batch->fence = fence;
    batch->fence_signaled = false;
    batch->inserted_barriers = 0;
    batch->elided_barriers = 0;

    *out_batch = batch;
    return AC_SUCCESS;
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
    if (out_event) {
        *out_event = NULL;
    }
    if (!batch || !kernel) {
        return AC_ERR_NULL_POINTER;
    }
    if (batch->state != AC_VK_BATCH_RECORDING) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (batch->dispatch_count >= batch->max_dispatches) {
        return AC_ERR_DEVICE_BUSY;
    }
    if (kernel->owning_context != batch->ctx) {
        return AC_ERR_CONTEXT_MISMATCH;
    }
    for (uint32_t i = 0; i < buffer_count; ++i) {
        if (!buffers[i] || buffers[i]->owning_context != batch->ctx) {
            return AC_ERR_CONTEXT_MISMATCH;
        }
    }

    AcDim3 grid = { group_x, group_y, group_z };
    AcResult grid_res = ac_validate_dispatch_grid(grid, batch->ctx->device_properties.limits.maxComputeWorkGroupCount);
    if (grid_res != AC_SUCCESS) return grid_res;
    AcResult budget_res = ac_validate_dispatch_budget(grid, AC_MAX_SAFE_GROUPS_PER_AXIS, AC_MAX_SAFE_TOTAL_GROUPS);
    if (budget_res != AC_SUCCESS) return budget_res;

    // Buffer dependency analysis: Minimal Compute-to-Compute barrier
    bool dependency_detected = false;
    for (uint32_t i = 0; i < buffer_count; ++i) {
        for (size_t w = 0; w < batch->written_buffers.size(); ++w) {
            if (buffers[i] == batch->written_buffers[w]) {
                dependency_detected = true;
                break;
            }
        }
        if (dependency_detected) break;
    }

    if (dependency_detected) {
        VkMemoryBarrier memBarrier = {};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(batch->command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &memBarrier, 0, NULL, 0, NULL);
        batch->inserted_barriers++;
    } else {
        batch->elided_barriers++;
    }

    // Mark destination buffer (last buffer) as written
    if (buffer_count > 0) {
        AcVkBuffer* dst_buf = buffers[buffer_count - 1];
        bool already_tracked = false;
        for (size_t w = 0; w < batch->written_buffers.size(); ++w) {
            if (batch->written_buffers[w] == dst_buf) {
                already_tracked = true;
                break;
            }
        }
        if (!already_tracked) {
            batch->written_buffers.push_back(dst_buf);
        }
    }

    // Per-dispatch Descriptor Pool and Set (Zero pool reuse hazard)
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = (buffer_count > 0) ? buffer_count : 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkResult res = vkCreateDescriptorPool(batch->ctx->device, &poolInfo, NULL, &desc_pool);
    if (res != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    VkDescriptorSetAllocateInfo allocSetInfo = {};
    allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSetInfo.descriptorPool = desc_pool;
    allocSetInfo.descriptorSetCount = 1;
    allocSetInfo.pSetLayouts = &kernel->descriptor_set_layout;

    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    res = vkAllocateDescriptorSets(batch->ctx->device, &allocSetInfo, &desc_set);
    if (res != VK_SUCCESS) {
        vkDestroyDescriptorPool(batch->ctx->device, desc_pool, NULL);
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    batch->descriptor_pools.push_back(desc_pool);

    // Update Descriptor Sets
    if (buffer_count > 0) {
        VkDescriptorBufferInfo bufferInfos[16] = {};
        VkWriteDescriptorSet descriptorWrites[16] = {};
        for (uint32_t i = 0; i < buffer_count && i < 16; ++i) {
            bufferInfos[i].buffer = buffers[i]->buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = buffers[i]->size_bytes;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = desc_set;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(batch->ctx->device, buffer_count, descriptorWrites, 0, NULL);
    }

    // Record Commands
    vkCmdBindPipeline(batch->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->compute_pipeline);
    vkCmdBindDescriptorSets(batch->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        kernel->pipeline_layout, 0, 1, &desc_set, 0, NULL);

    if (push_constants_size > 0 && push_constants != NULL) {
        vkCmdPushConstants(batch->command_buffer, kernel->pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_constants_size, push_constants);
    }

    vkCmdDispatch(batch->command_buffer, group_x, group_y, group_z);

    // Retain resources (increment outstanding counter)
    kernel->submission_outstanding_count++;
    batch->retained_kernels.push_back(kernel);
    for (uint32_t i = 0; i < buffer_count; ++i) {
        buffers[i]->submission_outstanding_count++;
        batch->retained_buffers.push_back(buffers[i]);
    }
    batch->dispatch_count++;

    // Create logical event if requested
    if (out_event) {
        AcVkEvent* log_ev = (AcVkEvent*)calloc(1, sizeof(AcVkEvent));
        if (log_ev) {
            log_ev->fence = batch->fence;
            log_ev->owning_context = batch->ctx;
            log_ev->bound_kernel = kernel;
            log_ev->is_signaled = false;
            log_ev->resources_retired = false;
            batch->logical_events.push_back(log_ev);
            *out_event = log_ev;
        }
    }

    return AC_SUCCESS;
}

AcResult ac_vk_batch_submit(
    AcVkCommandBatch* batch,
    AcVkEvent** out_completion_event
) {
    if (out_completion_event) {
        *out_completion_event = NULL;
    }
    if (!batch) {
        return AC_ERR_NULL_POINTER;
    }
    if (batch->state != AC_VK_BATCH_RECORDING) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    VkResult vk_res = vkEndCommandBuffer(batch->command_buffer);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }

    batch->ctx->submission_outstanding_count++;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &batch->command_buffer;

    vk_res = vkQueueSubmit(batch->ctx->compute_queue, 1, &submitInfo, batch->fence);
    if (vk_res != VK_SUCCESS) {
        // Fail-safe rollback of all outstanding counters!
        batch->ctx->submission_outstanding_count--;
        for (size_t i = 0; i < batch->retained_kernels.size(); ++i) {
            if (batch->retained_kernels[i]->submission_outstanding_count > 0) {
                batch->retained_kernels[i]->submission_outstanding_count--;
            }
        }
        for (size_t i = 0; i < batch->retained_buffers.size(); ++i) {
            if (batch->retained_buffers[i]->submission_outstanding_count > 0) {
                batch->retained_buffers[i]->submission_outstanding_count--;
            }
        }
        batch->state = AC_VK_BATCH_ABORTED;
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }

    batch->state = AC_VK_BATCH_SUBMITTED;

    AcVkEvent* comp_ev = (AcVkEvent*)calloc(1, sizeof(AcVkEvent));
    if (!comp_ev) {
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    comp_ev->fence = batch->fence;
    comp_ev->owning_context = batch->ctx;
    comp_ev->is_signaled = false;
    comp_ev->resources_retired = false;
    comp_ev->submit_timestamp_ns = get_time_ns();

    if (out_completion_event) {
        *out_completion_event = comp_ev;
    }
    return AC_SUCCESS;
}

AcResult ac_vk_batch_abort(AcVkCommandBatch* batch) {
    if (!batch) return AC_ERR_NULL_POINTER;
    if (batch->state == AC_VK_BATCH_ABORTED) return AC_SUCCESS;

    if (batch->state == AC_VK_BATCH_RECORDING) {
        for (size_t i = 0; i < batch->retained_kernels.size(); ++i) {
            if (batch->retained_kernels[i]->submission_outstanding_count > 0) {
                batch->retained_kernels[i]->submission_outstanding_count--;
            }
        }
        for (size_t i = 0; i < batch->retained_buffers.size(); ++i) {
            if (batch->retained_buffers[i]->submission_outstanding_count > 0) {
                batch->retained_buffers[i]->submission_outstanding_count--;
            }
        }
        for (size_t i = 0; i < batch->descriptor_pools.size(); ++i) {
            if (batch->descriptor_pools[i] != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(batch->ctx->device, batch->descriptor_pools[i], NULL);
            }
        }
        batch->descriptor_pools.clear();

        if (batch->command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(batch->ctx->device, batch->ctx->command_pool, 1, &batch->command_buffer);
            batch->command_buffer = VK_NULL_HANDLE;
        }
        if (batch->fence != VK_NULL_HANDLE) {
            vkDestroyFence(batch->ctx->device, batch->fence, NULL);
            batch->fence = VK_NULL_HANDLE;
        }
        batch->state = AC_VK_BATCH_ABORTED;
        return AC_SUCCESS;
    }

    return AC_ERR_INVALID_ARGUMENT;
}

void ac_vk_batch_destroy(AcVkCommandBatch* batch) {
    if (!batch) return;

    if (batch->state == AC_VK_BATCH_RECORDING) {
        ac_vk_batch_abort(batch);
    } else if (batch->state == AC_VK_BATCH_SUBMITTED) {
        // Guarantee GPU completion before releasing batch resources!
        if (!batch->fence_signaled && batch->fence != VK_NULL_HANDLE) {
            vkWaitForFences(batch->ctx->device, 1, &batch->fence, VK_TRUE, 5000000000ULL);
            batch->fence_signaled = true;
        }

        for (size_t i = 0; i < batch->retained_kernels.size(); ++i) {
            if (batch->retained_kernels[i]->submission_outstanding_count > 0) {
                batch->retained_kernels[i]->submission_outstanding_count--;
            }
        }
        for (size_t i = 0; i < batch->retained_buffers.size(); ++i) {
            if (batch->retained_buffers[i]->submission_outstanding_count > 0) {
                batch->retained_buffers[i]->submission_outstanding_count--;
            }
        }
        if (batch->ctx->submission_outstanding_count > 0) {
            batch->ctx->submission_outstanding_count--;
        }

        for (size_t i = 0; i < batch->descriptor_pools.size(); ++i) {
            if (batch->descriptor_pools[i] != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(batch->ctx->device, batch->descriptor_pools[i], NULL);
            }
        }
        batch->descriptor_pools.clear();

        if (batch->command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(batch->ctx->device, batch->ctx->command_pool, 1, &batch->command_buffer);
            batch->command_buffer = VK_NULL_HANDLE;
        }
        if (batch->fence != VK_NULL_HANDLE) {
            vkDestroyFence(batch->ctx->device, batch->fence, NULL);
            batch->fence = VK_NULL_HANDLE;
        }
    }

    for (size_t i = 0; i < batch->logical_events.size(); ++i) {
        if (batch->logical_events[i]) {
            free(batch->logical_events[i]);
        }
    }
    batch->logical_events.clear();

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

} // extern "C"

