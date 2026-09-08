#include "ac_buffer.h"
#include "ac_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

extern "C" {

AcResult ac_vk_buffer_create(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    AcVkBuffer** out_buf
) {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped_ptr = NULL;
    VkBufferCreateInfo bufferInfo = {};
    VkMemoryRequirements memReqs = {};
    VkMemoryAllocateInfo allocInfo = {};
    AcResult ret = AC_SUCCESS;
    VkResult vk_res = VK_SUCCESS;
    uint32_t memTypeIndex = 0xFFFFFFFFu;
    VkMemoryPropertyFlags required_props = 0;

    if (!out_buf) {
        return AC_ERR_NULL_POINTER;
    }
    *out_buf = NULL;

    if (!ctx || ctx->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (size_bytes == 0) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }

    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_bytes;
    bufferInfo.usage = usage_flags ? usage_flags : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vk_res = vkCreateBuffer(ctx->device, &bufferInfo, NULL, &buffer);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    vkGetBufferMemoryRequirements(ctx->device, buffer, &memReqs);

    required_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memTypeIndex = find_memory_type(ctx->physical_device, memReqs.memoryTypeBits, required_props);
    if (memTypeIndex == 0xFFFFFFFFu) {
        ret = AC_ERR_DEVICE_NOT_SUPPORTED;
        goto cleanup;
    }

    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    vk_res = vkAllocateMemory(ctx->device, &allocInfo, NULL, &memory);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup;
    }

    vk_res = vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_OUT_OF_DEVICE_MEMORY;
        goto cleanup;
    }

    vk_res = vkMapMemory(ctx->device, memory, 0, size_bytes, 0, &mapped_ptr);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_BUFFER_NOT_MAPPED;
        goto cleanup;
    }

    {
        AcVkBuffer* buf = (AcVkBuffer*)calloc(1, sizeof(AcVkBuffer));
        if (!buf) {
            ret = AC_ERR_OUT_OF_HOST_MEMORY;
            goto cleanup;
        }

        buf->buffer = buffer;
        buf->memory = memory;
        buf->mapped_ptr = mapped_ptr;
        buf->size_bytes = size_bytes;
        buf->memory_property_flags = required_props;
        buf->owning_context = ctx;
        buf->in_flight_count = 0;

        *out_buf = buf;
        return AC_SUCCESS;
    }

cleanup:
    if (mapped_ptr) {
        vkUnmapMemory(ctx->device, memory);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx->device, memory, NULL);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
    }
    return ret;
}

AcResult ac_vk_buffer_destroy(AcVkBuffer* buf) {
    if (!buf) {
        return AC_SUCCESS;
    }

    if (buf->in_flight_count > 0) {
        return AC_ERR_DEVICE_BUSY;
    }

    if (!buf->owning_context || buf->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkDevice device = buf->owning_context->device;

    if (buf->mapped_ptr) {
        vkUnmapMemory(device, buf->memory);
        buf->mapped_ptr = NULL;
    }
    if (buf->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buf->buffer, NULL);
        buf->buffer = VK_NULL_HANDLE;
    }
    if (buf->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buf->memory, NULL);
        buf->memory = VK_NULL_HANDLE;
    }

    buf->owning_context = NULL;
    buf->size_bytes = 0;
    free(buf);
    return AC_SUCCESS;
}

} // extern "C"
