/**
 * @file ac_buffer.h
 * @brief Vulkan Buffer Memory Management with Explicit Context Ownership
 */
#ifndef AC_BUFFER_H
#define AC_BUFFER_H

#include "ac_types.h"
#include "ac_context.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AcVkBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_ptr;
    size_t size_bytes;
    VkMemoryPropertyFlags memory_property_flags;
    AcVkContext* owning_context;
    uint32_t in_flight_count;
};

AcResult ac_vk_buffer_create(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    AcVkBuffer** out_buf
);

AcResult ac_vk_buffer_destroy(AcVkBuffer* buf);

#ifdef __cplusplus
}
#endif

#endif /* AC_BUFFER_H */
