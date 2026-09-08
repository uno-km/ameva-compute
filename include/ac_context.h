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

#include "ac_types.h"
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

#ifdef __cplusplus
}
#endif

#endif /* AC_CONTEXT_H */
