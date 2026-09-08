/**
 * @file ac_buffer.h
 * @brief Vulkan Buffer Memory Management with Explicit Context Ownership
 *
 * Contract:
 *   - Persistently Mapped Host-Visible Buffer: Requires both VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
 *     and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT. Returns AC_ERR_FEATURE_NOT_SUPPORTED if unavailable.
 *   - Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *   - Enforces context match against owning_context.
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
    uint32_t submission_outstanding_count;
};

/**
 * @brief Allocate a host-visible coherent storage buffer with persistently mapped virtual address.
 *
 * @param ctx Owning Vulkan compute context.
 * @param size_bytes Buffer allocation size in bytes (> 0).
 * @param usage_flags Vulkan buffer usage flags (e.g. VK_BUFFER_USAGE_STORAGE_BUFFER_BIT).
 * @param out_buf Output buffer pointer (set to NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_create(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    AcVkBuffer** out_buf
);

/**
 * @brief Safely destroy a Vulkan buffer.
 * Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *
 * @param buf Buffer to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_destroy(AcVkBuffer* buf);

#ifdef __cplusplus
}
#endif

#endif /* AC_BUFFER_H */
