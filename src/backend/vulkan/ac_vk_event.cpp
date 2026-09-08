#include "ac_event.h"
#include "ac_types.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {

AcResult ac_vk_event_wait(AcVkEvent* event, uint64_t timeout_ns) {
    if (!event) {
        return AC_ERR_NULL_POINTER;
    }
    if (event->is_completed) {
        return AC_SUCCESS;
    }
    if (!event->owning_context || event->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkResult res = vkWaitForFences(event->owning_context->device, 1, &event->fence, VK_TRUE, timeout_ns);
    if (res == VK_SUCCESS) {
        event->is_completed = true;
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

    if (!event->is_completed) {
        vkWaitForFences(device, 1, &event->fence, VK_TRUE, 5000000000ULL);
        event->is_completed = true;
    }

    if (event->bound_kernel) {
        if (event->bound_kernel->in_flight_count > 0) {
            event->bound_kernel->in_flight_count--;
        }
        event->bound_kernel = NULL;
    }

    if (event->bound_buffers) {
        for (uint32_t i = 0; i < event->bound_buffer_count; ++i) {
            if (event->bound_buffers[i] && event->bound_buffers[i]->in_flight_count > 0) {
                event->bound_buffers[i]->in_flight_count--;
            }
        }
        free(event->bound_buffers);
        event->bound_buffers = NULL;
    }

    if (event->owning_context->in_flight_dispatches > 0) {
        event->owning_context->in_flight_dispatches--;
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

    event->owning_context = NULL;
    free(event);
    return AC_SUCCESS;
}

} // extern "C"
