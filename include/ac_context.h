/**
 * @file ac_context.h
 * @brief Vulkan Compute Context Lifecycle & Hardware Selection (Zero Global State)
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
    uint32_t in_flight_dispatches;
};

AcResult ac_vk_context_create(AcVkContext** out_ctx);
AcResult ac_vk_context_destroy(AcVkContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* AC_CONTEXT_H */
