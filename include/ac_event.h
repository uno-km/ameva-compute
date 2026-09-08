/**
 * @file ac_event.h
 * @brief Vulkan Event, Fence Synchronization & Lifecycle Tracking
 */
#ifndef AC_EVENT_H
#define AC_EVENT_H

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AcVkEvent {
    VkFence fence;
    VkCommandBuffer command_buffer;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    uint64_t submit_timestamp_ns;
    bool is_completed;
    AcVkContext* owning_context;
    AcVkKernel* bound_kernel;
    AcVkBuffer** bound_buffers;
    uint32_t bound_buffer_count;
};

AcResult ac_vk_event_wait(AcVkEvent* event, uint64_t timeout_ns);
AcResult ac_vk_event_destroy(AcVkEvent* event);

#ifdef __cplusplus
}
#endif

#endif /* AC_EVENT_H */
