/**
 * @file ac_event.h
 * @brief Vulkan Event, Fence Synchronization & Lifecycle Tracking with Hardware Timestamps
 *
 * Lifecycle & Idempotency Contract:
 *   - States: CREATED -> SUBMITTED -> SIGNALED -> RETIRED
 *   - Release-Once: Decrementing bound kernel/buffers submission_outstanding_count occurs
 *     strictly once across multiple ac_vk_event_wait or ac_vk_event_destroy invocations.
 *   - Double-wait is idempotent and returns AC_SUCCESS.
 *   - Destruction without explicit wait automatically waits for fence completion before retiring resources.
 *   - Provides fine-grained hardware timestamp query alongside host wall latency.
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
    VkQueryPool query_pool;
    uint64_t submit_timestamp_ns;
    uint64_t completion_timestamp_ns;
    uint64_t wall_latency_ns;
    uint64_t gpu_duration_ns;
    bool has_timestamp;
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

/**
 * @brief Retrieve execution latency metrics for a signaled event.
 *
 * @param event Signaled event.
 * @param out_wall_latency_ns Pointer to receive host wall latency in nanoseconds (optional, can be NULL).
 * @param out_gpu_duration_ns Pointer to receive GPU duration in nanoseconds (optional, can be NULL).
 * @return AC_SUCCESS on success, AC_ERR_DEVICE_BUSY if not signaled, AC_ERR_FEATURE_NOT_SUPPORTED if GPU timestamps unsupported.
 */
AcResult ac_vk_event_get_timing(
    AcVkEvent* event,
    uint64_t* out_wall_latency_ns,
    uint64_t* out_gpu_duration_ns
);

#ifdef __cplusplus
}
#endif

#endif /* AC_EVENT_H */
