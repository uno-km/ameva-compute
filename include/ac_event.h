/**
 * @file ac_event.h
 * @brief Vulkan Event, Fence Synchronization & Lifecycle Tracking
 *
 * Lifecycle & Idempotency Contract:
 *   - States: CREATED -> SUBMITTED -> SIGNALED -> RETIRED
 *   - Release-Once: Decrementing bound kernel/buffers submission_outstanding_count occurs
 *     strictly once across multiple ac_vk_event_wait or ac_vk_event_destroy invocations.
 *   - Double-wait is idempotent and returns AC_SUCCESS.
 *   - Destruction without explicit wait automatically waits for fence completion before retiring resources.
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
    bool is_signaled;
    bool resources_retired;
    AcVkContext* owning_context;
    AcVkKernel* bound_kernel;
    AcVkBuffer** bound_buffers;
    uint32_t bound_buffer_count;
};

/**
 * @brief Wait for GPU execution to complete on this event.
 * Idempotent: subsequent calls return AC_SUCCESS without double-retiring resources.
 *
 * @param event Event to wait on.
 * @param timeout_ns Timeout in nanoseconds.
 * @return AC_SUCCESS on completion, AC_ERR_LAUNCH_TIMEOUT on timeout, error code on failure.
 */
AcResult ac_vk_event_wait(AcVkEvent* event, uint64_t timeout_ns);

/**
 * @brief Destroy an event, ensuring GPU completion and strictly single resource retirement.
 *
 * @param event Event to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_event_destroy(AcVkEvent* event);

#ifdef __cplusplus
}
#endif

#endif /* AC_EVENT_H */
