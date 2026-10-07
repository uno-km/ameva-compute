/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_buffer.h
 * @brief Vulkan Buffer Memory Management with Explicit Context Ownership & Cache Coherency
 *
 * Contract:
 *   - Logical vs Allocation Size: Explicitly tracks both size_bytes (logical requested)
 *     and allocation_size_bytes (physical Vulkan memory allocation).
 *   - Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *   - Enforces context match against owning_context.
 *   - Provides explicit memory range flush and invalidate operations aligned to nonCoherentAtomSize.
 *   - Supports explicit coherent vs non-coherent memory allocation via ac_vk_buffer_create_explicit.
 */
#ifndef AC_BUFFER_H
#define AC_BUFFER_H

#include "ac_base.h"
#include "ac_context.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AcVkBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_ptr;
    size_t size_bytes;              /* Logical requested byte size */
    size_t allocation_size_bytes;   /* Physical Vulkan memory allocation size (memReqs.size) */
    VkMemoryPropertyFlags memory_property_flags;
    uint32_t memory_type_index;
    AcVkContext* owning_context;
    uint32_t submission_outstanding_count;
    bool is_coherent;
};

/**
 * @brief Allocate a host-visible storage buffer with persistently mapped virtual address.
 * Defaults to coherent memory if available, falling back to non-coherent.
 *
 * @param ctx Owning Vulkan compute context.
 * @param size_bytes Logical buffer allocation size in bytes (> 0).
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
 * @brief Allocate a buffer with explicit coherency preference.
 *
 * @param ctx Owning Vulkan compute context.
 * @param size_bytes Logical buffer allocation size in bytes (> 0).
 * @param usage_flags Vulkan buffer usage flags.
 * @param force_non_coherent If true, strictly requires HOST_VISIBLE memory without HOST_COHERENT bit.
 * @param out_buf Output buffer pointer.
 * @return AC_SUCCESS on success, AC_ERR_FEATURE_NOT_SUPPORTED if requested type unavailable.
 */
AcResult ac_vk_buffer_create_explicit(
    AcVkContext* ctx,
    size_t size_bytes,
    VkBufferUsageFlags usage_flags,
    bool force_non_coherent,
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

/**
 * @brief Flush mapped memory range for host writes to be visible to device.
 * For coherent memory, this performs parameter validation and returns AC_SUCCESS.
 * For non-coherent memory, range is aligned to nonCoherentAtomSize and flushed.
 *
 * @param buf Buffer to flush.
 * @param offset Byte offset from buffer start.
 * @param size_bytes Size of range in bytes (0 flushes from offset to end of buffer).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_flush(AcVkBuffer* buf, size_t offset, size_t size_bytes);

/**
 * @brief Invalidate mapped memory range for device writes to be visible to host.
 * For coherent memory, this performs parameter validation and returns AC_SUCCESS.
 * For non-coherent memory, range is aligned to nonCoherentAtomSize and invalidated.
 *
 * @param buf Buffer to invalidate.
 * @param offset Byte offset from buffer start.
 * @param size_bytes Size of range in bytes (0 invalidates from offset to end of buffer).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_invalidate(AcVkBuffer* buf, size_t offset, size_t size_bytes);

/**
 * @brief Retrieve direct pointer to persistently mapped memory.
 *
 * @param buf Target buffer.
 * @param out_ptr Pointer to mapped virtual address.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_map(AcVkBuffer* buf, void** out_ptr);

/**
 * @brief Upload data from host into buffer with cache flush.
 *
 * @param buf Target buffer.
 * @param src Host source buffer.
 * @param offset Byte offset into buffer.
 * @param size_bytes Number of bytes to upload.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_upload(AcVkBuffer* buf, const void* src, size_t offset, size_t size_bytes);

/**
 * @brief Read data from buffer into host memory with cache invalidation.
 *
 * @param buf Target buffer.
 * @param dst Host destination buffer.
 * @param offset Byte offset into buffer.
 * @param size_bytes Number of bytes to read back.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_buffer_readback(AcVkBuffer* buf, void* dst, size_t offset, size_t size_bytes);

#ifdef __cplusplus
}
#endif

#endif /* AC_BUFFER_H */
