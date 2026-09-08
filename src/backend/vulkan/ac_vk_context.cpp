#include "ac_context.h"
#include "ac_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

AcResult ac_vk_context_create(AcVkContext** out_ctx) {
    if (!out_ctx) {
        return AC_ERR_NULL_POINTER;
    }
    *out_ctx = NULL;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "AmevaComputeCore";
    appInfo.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.pEngineName = "AMUDA";
    appInfo.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&instanceInfo, NULL, &instance);
    if (res != VK_SUCCESS) {
        return AC_ERR_DRIVER_INIT_FAILED;
    }

    uint32_t deviceCount = 0;
    res = vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (res != VK_SUCCESS || deviceCount == 0) {
        vkDestroyInstance(instance, NULL);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    VkPhysicalDevice physicalDevices[16];
    if (deviceCount > 16) deviceCount = 16;
    res = vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties selectedProps = {};
    uint32_t computeQueueFamilyIndex = 0xFFFFFFFFu;
    uint32_t timestampValidBits = 0;

    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

        /* Ground Truth: Physical device MUST NOT be CPU fallback */
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[i], &queueFamilyCount, NULL);
            VkQueueFamilyProperties queueFamilies[16];
            if (queueFamilyCount > 16) queueFamilyCount = 16;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[i], &queueFamilyCount, queueFamilies);

            for (uint32_t q = 0; q < queueFamilyCount; ++q) {
                if (queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    selectedDevice = physicalDevices[i];
                    selectedProps = props;
                    computeQueueFamilyIndex = q;
                    timestampValidBits = queueFamilies[q].timestampValidBits;
                    break;
                }
            }
            if (selectedDevice != VK_NULL_HANDLE) break;
        }
    }

    if (selectedDevice == VK_NULL_HANDLE) {
        vkDestroyInstance(instance, NULL);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device = VK_NULL_HANDLE;
    res = vkCreateDevice(selectedDevice, &deviceCreateInfo, NULL, &device);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        return AC_ERR_DRIVER_INIT_FAILED;
    }

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    res = vkCreateCommandPool(device, &cmdPoolInfo, NULL, &commandPool);
    if (res != VK_SUCCESS) {
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return AC_ERR_DRIVER_INIT_FAILED;
    }

    AcVkContext* ctx = (AcVkContext*)calloc(1, sizeof(AcVkContext));
    if (!ctx) {
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    ctx->instance = instance;
    ctx->physical_device = selectedDevice;
    ctx->device = device;
    ctx->compute_queue = computeQueue;
    ctx->queue_family_index = computeQueueFamilyIndex;
    ctx->device_properties = selectedProps;
    ctx->command_pool = commandPool;
    ctx->active_child_buffers = 0;
    ctx->active_child_kernels = 0;
    ctx->submission_outstanding_count = 0;
    ctx->timestamp_valid_bits = timestampValidBits;

    *out_ctx = ctx;
    return AC_SUCCESS;
}

AcResult ac_vk_context_destroy(AcVkContext* ctx) {
    if (!ctx) {
        return AC_SUCCESS;
    }

    /* Strict Hierarchy: Reject destruction if child buffers or kernels are still alive */
    if (ctx->active_child_buffers > 0 || ctx->active_child_kernels > 0) {
        return AC_ERR_RESOURCE_IN_USE;
    }

    /* Reject destruction if submissions are currently outstanding */
    if (ctx->submission_outstanding_count > 0) {
        return AC_ERR_DEVICE_BUSY;
    }

    if (ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        if (ctx->command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
            ctx->command_pool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(ctx->device, NULL);
        ctx->device = VK_NULL_HANDLE;
    }

    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
    }

    free(ctx);
    return AC_SUCCESS;
}

} // extern "C"
