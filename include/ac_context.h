/**
 * @file ac_context.h
 * @brief Vulkan Compute Context Lifecycle & Hardware Selection (Zero Global State)
 *
 * Contract:
 *   - Strictly non-CPU GPU device selection (Fail-Fast against CPU fallbacks).
 *   - Rejects destruction with AC_ERR_RESOURCE_IN_USE if child buffers or kernels remain alive.
 *   - Rejects destruction with AC_ERR_DEVICE_BUSY if unretired submissions remain outstanding.
 */
#ifndef AC_CONTEXT_H
#define AC_CONTEXT_H

#include "ac_base.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AcVkContext {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_index;
    VkPhysicalDeviceProperties device_properties;
    VkCommandPool command_pool;
    uint32_t active_child_buffers;
    uint32_t active_child_kernels;
    uint32_t submission_outstanding_count;
    uint32_t timestamp_valid_bits;
    /* Vulkan 1.1 properties not in VkPhysicalDeviceLimits (0 when the driver does not report them) */
    uint64_t max_memory_allocation_size;
    uint32_t subgroup_size;
    /* VK_KHR_shader_integer_dot_product: 1 when the extension and its feature are enabled on the device (packed
       int8 dot products in shaders), and whether the driver reports the signed 4x8 packed form as accelerated */
    uint32_t integer_dot_product;
    uint32_t integer_dot_product_accelerated;
    /* Compiled pipelines of this device, used by every kernel creation; filled from saved data with
       ac_vk_pipeline_cache_merge and read back with ac_vk_pipeline_cache_get_data. */
    VkPipelineCache pipeline_cache;
    /* Descriptor sets keyed by (set layout, bound buffer ranges), shared by every dispatch path of this context.
       Owned by ac_vk_dispatch.cpp; NULL until the first dispatch. */
    void* desc_cache;
    /* Command batches between begin and submit/abort: the descriptor cache is only trimmed when this and
       submission_outstanding_count are both 0. */
    uint32_t recording_batches;
};

/**
 * @brief Create an isolated Vulkan compute context.
 * Enforces selection of a non-CPU integrated/discrete compute GPU.
 *
 * @param out_ctx Output context pointer (set to NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_context_create(AcVkContext** out_ctx);

/**
 * @brief Safely destroy a Vulkan compute context.
 * Rejects destruction with AC_ERR_RESOURCE_IN_USE if child buffers/kernels exist.
 * Rejects destruction with AC_ERR_DEVICE_BUSY if submissions are outstanding.
 *
 * @param ctx Context to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_context_destroy(AcVkContext* ctx);

/**
 * @brief Merge saved pipeline cache data (vkGetPipelineCacheData of an earlier context) into the context's cache.
 * The Vulkan header of the data must name this device and driver (vendor ID, device ID, pipelineCacheUUID); data of
 * another device or driver version, or malformed data, is rejected with AC_ERR_INVALID_ARGUMENT and the cache is
 * left as it was.
 */
AcResult ac_vk_pipeline_cache_merge(AcVkContext* ctx, const void* data, size_t size);

/**
 * @brief Copy the context's pipeline cache data. With data == NULL, *inout_size receives the size; otherwise up to
 * *inout_size bytes are written and *inout_size is set to the bytes written (AC_ERR_INVALID_BUFFER_SIZE when the
 * buffer is too small for the whole cache).
 */
AcResult ac_vk_pipeline_cache_get_data(AcVkContext* ctx, void* data, size_t* inout_size);

/**
 * @brief Descriptor set cache maintenance (implemented in ac_vk_dispatch.cpp).
 * The purge functions free every cached set that references the buffer / uses the layout; callers invoke them
 * only after the object has no outstanding submission (its destroy already enforces that).
 */
void ac_vk_desc_cache_purge_buffer(AcVkContext* ctx, VkBuffer buffer);
void ac_vk_desc_cache_purge_layout(AcVkContext* ctx, VkDescriptorSetLayout layout);
void ac_vk_desc_cache_destroy(AcVkContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* AC_CONTEXT_H */
