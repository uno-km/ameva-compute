#include "ac_event.h"
#include "ac_types.h"
#include <stdio.h>
#include <stdlib.h>

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

} // extern "C"
