#include "ac_event.h"
#include "ac_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void ac_vk_event_retire_resources(AcVkEvent* event) {
    if (!event || event->resources_retired) {
        return; /* Idempotency: Retire strictly once */
    }
    event->resources_retired = true;

    if (event->bound_kernel) {
        if (event->bound_kernel->submission_outstanding_count > 0) {
            event->bound_kernel->submission_outstanding_count--;
        }
    }

    if (event->bound_buffers) {
        for (uint32_t i = 0; i < event->bound_buffer_count; ++i) {
            if (event->bound_buffers[i] && event->bound_buffers[i]->submission_outstanding_count > 0) {
                event->bound_buffers[i]->submission_outstanding_count--;
            }
        }
    }

    if (event->owning_context && event->owning_context->submission_outstanding_count > 0) {
        event->owning_context->submission_outstanding_count--;
    }
}

extern "C" {

AcResult ac_vk_event_wait(AcVkEvent* event, uint64_t timeout_ns) {
    if (!event) {
        return AC_ERR_NULL_POINTER;
    }

    const char* fail_stage = getenv("AMEVA_RUNTIME_FAIL_STAGE");
    if (!fail_stage) fail_stage = getenv("AMEVA_BACKEND_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "wait") == 0) {
        fprintf(stderr, "[AMEVA_RUNTIME_FAIL_INJECTION] Simulated failure at stage: wait\n");
        return AC_ERR_LAUNCH_TIMEOUT;
    }

    /* Batch events: the submission owns the fence and retires the batch's resources exactly once */
    if (event->shared) {
        AcResult sr = ac_vk_submission_wait(event->shared, timeout_ns);
        if (sr == AC_SUCCESS && !event->is_signaled) {
            bool has_ts = false;
            ac_vk_submission_get_timing(event->shared, &event->wall_latency_ns, &event->gpu_duration_ns, &has_ts);
            event->is_signaled = true;
        }
        return sr;
    }

    /* Idempotent double-wait */
    if (event->is_signaled) {
        ac_vk_event_retire_resources(event);
        return AC_SUCCESS;
    }
    if (!event->owning_context || event->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkResult res = vkWaitForFences(event->owning_context->device, 1, &event->fence, VK_TRUE, timeout_ns);
    if (res == VK_SUCCESS) {
        event->is_signaled = true;
        event->completion_timestamp_ns = get_time_ns();
        if (event->completion_timestamp_ns >= event->submit_timestamp_ns) {
            event->wall_latency_ns = event->completion_timestamp_ns - event->submit_timestamp_ns;
        }

        /* Read back GPU hardware timestamps if available */
        if (event->has_timestamp && event->query_pool != VK_NULL_HANDLE) {
            uint64_t timestamps[2] = {0, 0};
            VkResult qr = vkGetQueryPoolResults(
                event->owning_context->device,
                event->query_pool,
                0, 2,
                sizeof(timestamps),
                timestamps,
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
            );
            if (qr == VK_SUCCESS) {
                uint32_t valid_bits = event->owning_context->timestamp_valid_bits;
                uint64_t mask = (valid_bits >= 64 || valid_bits == 0) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << valid_bits) - 1ULL);
                uint64_t raw_delta = (timestamps[1] - timestamps[0]) & mask;
                double period = (double)event->owning_context->device_properties.limits.timestampPeriod;
                event->gpu_duration_ns = (uint64_t)((double)raw_delta * period);
            }
        }

        ac_vk_event_retire_resources(event);
        return AC_SUCCESS;
    } else if (res == VK_TIMEOUT) {
        return AC_ERR_LAUNCH_TIMEOUT;
    } else {
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
}

AcResult ac_vk_event_destroy(AcVkEvent* event) {
    if (!event) {
        return AC_SUCCESS;
    }

    if (event->shared) {
        /* Same contract as a single-dispatch event: destroying a pending event waits for it, and on timeout
           nothing is released so the caller may retry. An unsubmitted or aborted batch has nothing to wait for. */
        if (ac_vk_submission_is_submitted(event->shared) && !ac_vk_submission_is_signaled(event->shared)) {
            AcResult wr = ac_vk_submission_wait(event->shared, 5000000000ULL);
            if (wr == AC_ERR_LAUNCH_TIMEOUT || wr == AC_ERR_OUT_OF_HOST_MEMORY || wr == AC_ERR_OUT_OF_DEVICE_MEMORY) {
                return wr;
            }
        }
        ac_vk_submission_release(event->shared);
        event->shared = NULL;
        event->owning_context = NULL;
        free(event);
        return AC_SUCCESS;
    }

    if (!event->owning_context || event->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkDevice device = event->owning_context->device;

    /* Guarantee completion before releasing resources. If the fence is still unsignaled the command buffer is
     * pending: freeing it, the fence, the query pool or the descriptor pool is undefined behaviour
     * (VUID-vkFreeCommandBuffers-pCommandBuffers-00047, VUID-vkDestroyFence-fence-01120,
     *  VUID-vkDestroyDescriptorPool-descriptorPool-00303). On timeout nothing is released and the caller gets
     * AC_ERR_LAUNCH_TIMEOUT; destroy may be called again later. Never mark an unfinished event as signaled.
     * Return value contract: AC_SUCCESS <=> the event was released (callers drop their pointer only then). */
    if (!event->is_signaled) {
        VkResult wr = vkWaitForFences(device, 1, &event->fence, VK_TRUE, 5000000000ULL);
        if (wr == VK_SUCCESS) {
            event->is_signaled = true;
        } else if (wr == VK_TIMEOUT) {
            return AC_ERR_LAUNCH_TIMEOUT;
        } else if (wr != VK_ERROR_DEVICE_LOST) {
            /* OUT_OF_HOST/DEVICE_MEMORY: completion unknown, so the work may still be pending */
            return wr == VK_ERROR_OUT_OF_DEVICE_MEMORY ? AC_ERR_OUT_OF_DEVICE_MEMORY : AC_ERR_OUT_OF_HOST_MEMORY;
        }
        /* VK_ERROR_DEVICE_LOST: the lost device executes nothing more, destroying is allowed; the loss itself is
         * reported by ac_vk_event_wait and by every later submission. */
    }

    /* Strictly single resource retirement */
    ac_vk_event_retire_resources(event);

    if (event->query_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, event->query_pool, NULL);
        event->query_pool = VK_NULL_HANDLE;
    }

    if (event->fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, event->fence, NULL);
        event->fence = VK_NULL_HANDLE;
    }

    if (event->command_buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, event->owning_context->command_pool, 1, &event->command_buffer);
        event->command_buffer = VK_NULL_HANDLE;
    }

    if (event->descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, event->descriptor_pool, NULL);
        event->descriptor_pool = VK_NULL_HANDLE;
    }

    if (event->bound_buffers) {
        free(event->bound_buffers);
        event->bound_buffers = NULL;
    }
    event->bound_kernel = NULL;
    event->owning_context = NULL;
    free(event);
    return AC_SUCCESS;
}

AcResult ac_vk_event_get_timing(
    AcVkEvent* event,
    uint64_t* out_wall_latency_ns,
    uint64_t* out_gpu_duration_ns
) {
    if (!event) {
        return AC_ERR_NULL_POINTER;
    }
    if (event->shared && !event->is_signaled && ac_vk_submission_is_signaled(event->shared)) {
        /* signaled through another event of the same batch, or through the batch itself */
        bool has_ts = false;
        ac_vk_submission_get_timing(event->shared, &event->wall_latency_ns, &event->gpu_duration_ns, &has_ts);
        event->is_signaled = true;
    }
    if (!event->is_signaled) {
        return AC_ERR_DEVICE_BUSY;
    }
    if (out_wall_latency_ns) {
        *out_wall_latency_ns = event->wall_latency_ns;
    }
    if (out_gpu_duration_ns) {
        if (event->has_timestamp) {
            *out_gpu_duration_ns = event->gpu_duration_ns;
        } else {
            *out_gpu_duration_ns = 0;
            return AC_ERR_FEATURE_NOT_SUPPORTED;
        }
    }
    return AC_SUCCESS;
}

} // extern "C"
