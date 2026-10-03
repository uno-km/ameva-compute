#include "ac_buffer.h"
#include "ac_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags required_props,
    VkMemoryPropertyFlags forbidden_props
) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            ((mem_props.memoryTypes[i].propertyFlags & required_props) == required_props) &&
            ((mem_props.memoryTypes[i].propertyFlags & forbidden_props) == 0)) {
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

extern "C" {

AcResult ac_vk_buffer_create_explicit(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    bool force_non_coherent,
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
    VkMemoryPropertyFlags selected_props = 0;
    bool is_coherent = false;

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

    if (force_non_coherent) {
        /* Strictly require host visible without coherent bit */
        VkMemoryPropertyFlags req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        VkMemoryPropertyFlags forb = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        memTypeIndex = find_memory_type(ctx->physical_device, memReqs.memoryTypeBits, req, forb);
        if (memTypeIndex == 0xFFFFFFFFu) {
            ret = AC_ERR_FEATURE_NOT_SUPPORTED;
            goto cleanup;
        }
        selected_props = req;
        is_coherent = false;
    } else {
        /* Try host-visible coherent first */
        VkMemoryPropertyFlags req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        memTypeIndex = find_memory_type(ctx->physical_device, memReqs.memoryTypeBits, req, 0);
        if (memTypeIndex != 0xFFFFFFFFu) {
            selected_props = req;
            is_coherent = true;
        } else {
            /* Fall back to host-visible non-coherent */
            req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            memTypeIndex = find_memory_type(ctx->physical_device, memReqs.memoryTypeBits, req, 0);
            if (memTypeIndex == 0xFFFFFFFFu) {
                ret = AC_ERR_FEATURE_NOT_SUPPORTED;
                goto cleanup;
            }
            selected_props = req;
            is_coherent = false;
        }
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

    vk_res = vkMapMemory(ctx->device, memory, 0, memReqs.size, 0, &mapped_ptr);
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
        buf->allocation_size_bytes = memReqs.size;
        buf->memory_property_flags = selected_props;
        buf->memory_type_index = memTypeIndex;
        buf->owning_context = ctx;
        buf->submission_outstanding_count = 0;
        buf->is_coherent = is_coherent;

        ctx->active_child_buffers++;
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

AcResult ac_vk_buffer_create(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    AcVkBuffer** out_buf
) {
    return ac_vk_buffer_create_explicit(ctx, size_bytes, usage_flags, false, out_buf);
}

AcResult ac_vk_buffer_destroy(AcVkBuffer* buf) {
    if (!buf) {
        return AC_SUCCESS;
    }

    if (buf->submission_outstanding_count > 0) {
        return AC_ERR_DEVICE_BUSY;
    }

    if (!buf->owning_context || buf->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkDevice device = buf->owning_context->device;

    /* cached descriptor sets that reference this buffer must go before the buffer does */
    ac_vk_desc_cache_purge_buffer(buf->owning_context, buf->buffer);

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

    if (buf->owning_context->active_child_buffers > 0) {
        buf->owning_context->active_child_buffers--;
    }

    buf->owning_context = NULL;
    buf->size_bytes = 0;
    buf->allocation_size_bytes = 0;
    free(buf);
    return AC_SUCCESS;
}

AcResult ac_vk_buffer_flush(AcVkBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buf) {
        return AC_ERR_NULL_POINTER;
    }
    if (!buf->owning_context || buf->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (offset >= buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (size_bytes == 0) {
        size_bytes = buf->size_bytes - offset;
    } else if (offset + size_bytes > buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (buf->is_coherent) {
        return AC_SUCCESS;
    }

    VkDeviceSize atom_size = buf->owning_context->device_properties.limits.nonCoherentAtomSize;
    if (atom_size == 0) atom_size = 1;

    VkDeviceSize aligned_offset = (offset / atom_size) * atom_size;
    VkDeviceSize end = offset + size_bytes;
    VkDeviceSize aligned_end = ((end + atom_size - 1) / atom_size) * atom_size;
    VkDeviceSize final_size = (aligned_end >= buf->allocation_size_bytes) ? VK_WHOLE_SIZE : (aligned_end - aligned_offset);

    if (aligned_offset >= buf->allocation_size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buf->memory;
    range.offset = aligned_offset;
    range.size = final_size;

    VkResult vk_res = vkFlushMappedMemoryRanges(buf->owning_context->device, 1, &range);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    return AC_SUCCESS;
}

AcResult ac_vk_buffer_invalidate(AcVkBuffer* buf, size_t offset, size_t size_bytes) {
    if (!buf) {
        return AC_ERR_NULL_POINTER;
    }
    if (!buf->owning_context || buf->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (offset >= buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (size_bytes == 0) {
        size_bytes = buf->size_bytes - offset;
    } else if (offset + size_bytes > buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    if (buf->is_coherent) {
        return AC_SUCCESS;
    }

    VkDeviceSize atom_size = buf->owning_context->device_properties.limits.nonCoherentAtomSize;
    if (atom_size == 0) atom_size = 1;

    VkDeviceSize aligned_offset = (offset / atom_size) * atom_size;
    VkDeviceSize end = offset + size_bytes;
    VkDeviceSize aligned_end = ((end + atom_size - 1) / atom_size) * atom_size;
    VkDeviceSize final_size = (aligned_end >= buf->allocation_size_bytes) ? VK_WHOLE_SIZE : (aligned_end - aligned_offset);

    if (aligned_offset >= buf->allocation_size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buf->memory;
    range.offset = aligned_offset;
    range.size = final_size;

    VkResult vk_res = vkInvalidateMappedMemoryRanges(buf->owning_context->device, 1, &range);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
    return AC_SUCCESS;
}

// ac_vk_buffer_map / upload / readback: ported unchanged from track/stt (46aaddd); the whisper AMEVA backend
// (adapters/whisper/ggml-backend-ameva.cpp) depends on them. AMEVA_RUNTIME_FAIL_STAGE=map|readback is the
// fault-injection hook its tests use.
AcResult ac_vk_buffer_map(AcVkBuffer* buf, void** out_ptr) {
    if (!out_ptr) {
        return AC_ERR_NULL_POINTER;
    }
    *out_ptr = NULL;

    if (!buf) {
        return AC_ERR_NULL_POINTER;
    }

    const char* fail_stage = getenv("AMEVA_RUNTIME_FAIL_STAGE");
    if (!fail_stage) fail_stage = getenv("AMEVA_BACKEND_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "map") == 0) {
        fprintf(stderr, "[AMEVA_RUNTIME_FAIL_INJECTION] Simulated failure at stage: map\n");
        return AC_ERR_BUFFER_NOT_MAPPED;
    }

    if (!buf->mapped_ptr) {
        return AC_ERR_BUFFER_NOT_MAPPED;
    }

    *out_ptr = buf->mapped_ptr;
    return AC_SUCCESS;
}

AcResult ac_vk_buffer_upload(AcVkBuffer* buf, const void* src, size_t offset, size_t size_bytes) {
    if (!buf || !src) {
        return AC_ERR_NULL_POINTER;
    }
    if (offset + size_bytes < offset || offset + size_bytes > buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    void* dst_ptr = NULL;
    AcResult res = ac_vk_buffer_map(buf, &dst_ptr);
    if (res != AC_SUCCESS) {
        return res;
    }
    memcpy((char*)dst_ptr + offset, src, size_bytes);
    return ac_vk_buffer_flush(buf, offset, size_bytes);
}

AcResult ac_vk_buffer_readback(AcVkBuffer* buf, void* dst, size_t offset, size_t size_bytes) {
    if (!buf || !dst) {
        return AC_ERR_NULL_POINTER;
    }
    if (offset + size_bytes < offset || offset + size_bytes > buf->size_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    const char* fail_stage = getenv("AMEVA_RUNTIME_FAIL_STAGE");
    if (!fail_stage) fail_stage = getenv("AMEVA_BACKEND_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "readback") == 0) {
        fprintf(stderr, "[AMEVA_RUNTIME_FAIL_INJECTION] Simulated failure at stage: readback\n");
        return AC_ERR_BUFFER_NOT_MAPPED;
    }

    AcResult res = ac_vk_buffer_invalidate(buf, offset, size_bytes);
    if (res != AC_SUCCESS) {
        return res;
    }

    void* src_ptr = NULL;
    res = ac_vk_buffer_map(buf, &src_ptr);
    if (res != AC_SUCCESS) {
        return res;
    }

    memcpy(dst, (const char*)src_ptr + offset, size_bytes);
    return AC_SUCCESS;
}

} // extern "C"
