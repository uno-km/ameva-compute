#include "ac_event.h"
#include "ac_types.h"
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

    if (!event->owning_context || event->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkDevice device = event->owning_context->device;

    /* Guarantee completion before releasing resources */
    if (!event->is_signaled) {
        vkWaitForFences(device, 1, &event->fence, VK_TRUE, 5000000000ULL);
        event->is_signaled = true;
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
