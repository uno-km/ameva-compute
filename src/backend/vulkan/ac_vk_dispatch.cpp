#include "ac_dispatch.h"
#include "ac_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

extern "C" {

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
    VkDescriptorPoolSize poolSize = {};
    VkDescriptorPoolCreateInfo poolInfo = {};
    VkDescriptorSetAllocateInfo allocSetInfo = {};
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    VkFenceCreateInfo fenceInfo = {};
    VkCommandBufferBeginInfo beginInfo = {};
    VkSubmitInfo submitInfo = {};
    AcResult ret = AC_SUCCESS;
    VkResult vk_res = VK_SUCCESS;

    if (out_event) {
        *out_event = NULL;
    }

    if (!ctx || ctx->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!kernel || kernel->owning_context != ctx) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!buffers || buffer_count < kernel->buffer_count) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }
    if (push_constants_size != kernel->push_constant_size) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }
    if (group_x == 0 || group_y == 0 || group_z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }

    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        if (!buffers[i] || buffers[i]->owning_context != ctx) {
            return AC_ERR_INVALID_CONTEXT;
        }
    }

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
        goto cleanup;
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
        goto cleanup;
    }

    /* 5. Create Completion Fence */
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_res = vkCreateFence(ctx->device, &fenceInfo, NULL, &fence);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup;
    }

    /* 6. Record Command Buffer */
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &beginInfo);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->compute_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);

    if (push_constants && push_constants_size > 0) {
        vkCmdPushConstants(command_buffer, kernel->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_constants_size, push_constants);
    }

    vkCmdDispatch(command_buffer, group_x, group_y, group_z);
    vkEndCommandBuffer(command_buffer);

    /* 7. Submit to Compute Queue */
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer;

    vk_res = vkQueueSubmit(ctx->compute_queue, 1, &submitInfo, fence);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_DRIVER_CRASH_OR_LOST;
        goto cleanup;
    }

    /* 8. Increment In-Flight Usage Counters */
    kernel->in_flight_count++;
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        buffers[i]->in_flight_count++;
    }
    ctx->in_flight_dispatches++;

    /* 9. Construct Event Handle */
    {
        AcVkEvent* event = (AcVkEvent*)calloc(1, sizeof(AcVkEvent));
        if (!event) {
            ret = AC_ERR_OUT_OF_HOST_MEMORY;
            goto cleanup_counters;
        }

        event->fence = fence;
        event->command_buffer = command_buffer;
        event->descriptor_pool = descriptor_pool;
        event->descriptor_set = descriptor_set;
        event->submit_timestamp_ns = get_time_ns();
        event->is_completed = false;
        event->owning_context = ctx;
        event->bound_kernel = kernel;
        event->bound_buffer_count = kernel->buffer_count;
        event->bound_buffers = (AcVkBuffer**)malloc(kernel->buffer_count * sizeof(AcVkBuffer*));
        if (!event->bound_buffers) {
            free(event);
            ret = AC_ERR_OUT_OF_HOST_MEMORY;
            goto cleanup_counters;
        }
        for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
            event->bound_buffers[i] = buffers[i];
        }

        if (out_event) {
            *out_event = event;
            return AC_SUCCESS;
        } else {
            AcResult wait_res = ac_vk_event_wait(event, 5000000000ULL);
            ac_vk_event_destroy(event);
            return wait_res;
        }
    }

cleanup_counters:
    kernel->in_flight_count--;
    for (uint32_t i = 0; i < kernel->buffer_count; ++i) {
        buffers[i]->in_flight_count--;
    }
    ctx->in_flight_dispatches--;

cleanup:
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx->device, fence, NULL);
    }
    if (command_buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command_buffer);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, descriptor_pool, NULL);
    }
    return ret;
}

} // extern "C"
