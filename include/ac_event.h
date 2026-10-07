/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
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
 *     Timestamp calculations incorporate wrap-safe masking based on timestampValidBits.
 */
#ifndef AC_EVENT_H
#define AC_EVENT_H

#include "ac_base.h"
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
    /* Events of a command batch (completion event, per-dispatch logical events) own none of the Vulkan objects
       above: they hold a reference on the batch's submission, which owns the fence, command buffer, query pool and
       resource retention and is released with its last reference (batch or event). NULL for a single dispatch. */
    struct AcVkSubmission* shared;
};

/* Shared batch submissions (implemented in ac_vk_dispatch.cpp) */
struct AcVkSubmission;
/* AC_ERR_INVALID_ARGUMENT if the batch was never submitted (still recording, or aborted) */
AcResult ac_vk_submission_wait(struct AcVkSubmission* submission, uint64_t timeout_ns);
bool ac_vk_submission_is_signaled(const struct AcVkSubmission* submission);
bool ac_vk_submission_is_submitted(const struct AcVkSubmission* submission);
void ac_vk_submission_get_timing(const struct AcVkSubmission* submission, uint64_t* out_wall_latency_ns,
                                 uint64_t* out_gpu_duration_ns, bool* out_has_timestamps);
/* Drops one reference; the last one releases the Vulkan objects. A submitted, unsignaled submission is waited for
   first; if that wait fails the objects are deliberately leaked (freeing pending work is undefined behaviour). */
void ac_vk_submission_release(struct AcVkSubmission* submission);

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
 * @param out_wall_latency_ns Host synchronous wall latency in nanoseconds (optional, can be NULL).
 * @param out_gpu_duration_ns GPU timestamp command interval duration in nanoseconds (optional, can be NULL).
 *        Represents interval between TOP_OF_PIPE and BOTTOM_OF_PIPE stages.
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
