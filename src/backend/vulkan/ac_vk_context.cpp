/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ac_context.h"
#include "ac_base.h"
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

    /* VK_KHR_shader_integer_dot_product (packed int8 dot products, used by integer matrix kernels): enabled when the
       device offers the extension and its feature; reported through AcDeviceLimits, never assumed. */
    uint32_t intDot = 0, intDotAccelerated = 0;
    const char* deviceExtensions[1];
    uint32_t deviceExtensionCount = 0;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR intDotFeatures = {};
    intDotFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    if (selectedProps.apiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
        uint32_t extCount = 0;
        bool offered = false;
        if (vkEnumerateDeviceExtensionProperties(selectedDevice, NULL, &extCount, NULL) == VK_SUCCESS && extCount > 0) {
            VkExtensionProperties* exts = (VkExtensionProperties*)calloc(extCount, sizeof(VkExtensionProperties));
            if (exts && vkEnumerateDeviceExtensionProperties(selectedDevice, NULL, &extCount, exts) == VK_SUCCESS) {
                for (uint32_t e = 0; e < extCount; ++e) {
                    if (strcmp(exts[e].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) {
                        offered = true;
                    }
                }
            }
            free(exts);
        }
        if (offered) {
            VkPhysicalDeviceFeatures2 features2 = {};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &intDotFeatures;
            vkGetPhysicalDeviceFeatures2(selectedDevice, &features2);
            if (intDotFeatures.shaderIntegerDotProduct) {
                VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR intDotProps = {};
                intDotProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
                VkPhysicalDeviceProperties2 props2 = {};
                props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                props2.pNext = &intDotProps;
                vkGetPhysicalDeviceProperties2(selectedDevice, &props2);
                intDot = 1;
                intDotAccelerated = intDotProps.integerDotProduct4x8BitPackedSignedAccelerated ? 1u : 0u;
                deviceExtensions[deviceExtensionCount++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
                intDotFeatures.pNext = NULL;
                intDotFeatures.shaderIntegerDotProduct = VK_TRUE;
                deviceCreateInfo.pNext = &intDotFeatures;
            }
        }
    }
    deviceCreateInfo.enabledExtensionCount = deviceExtensionCount;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionCount ? deviceExtensions : NULL;

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
    ctx->max_memory_allocation_size = 0;
    ctx->subgroup_size = 0;
    ctx->desc_cache = NULL;
    ctx->recording_batches = 0;
    ctx->integer_dot_product = intDot;
    ctx->integer_dot_product_accelerated = intDotAccelerated;
    ctx->pipeline_cache = VK_NULL_HANDLE;
    {
        VkPipelineCacheCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device, &pci, NULL, &ctx->pipeline_cache) != VK_SUCCESS) {
            vkDestroyCommandPool(device, commandPool, NULL);
            vkDestroyDevice(device, NULL);
            vkDestroyInstance(instance, NULL);
            free(ctx);
            return AC_ERR_DRIVER_INIT_FAILED;
        }
    }

    /* Vulkan 1.1 core queries (the instance requests 1.1): maxMemoryAllocationSize and the subgroup size. A device
       that only implements 1.0 reports neither; the fields then stay 0 ("not reported"), they are never guessed. */
    if (selectedProps.apiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
        VkPhysicalDeviceSubgroupProperties subgroup = {};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceMaintenance3Properties maint3 = {};
        maint3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
        maint3.pNext = &subgroup;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &maint3;
        vkGetPhysicalDeviceProperties2(selectedDevice, &props2);
        ctx->max_memory_allocation_size = (uint64_t)maint3.maxMemoryAllocationSize;
        ctx->subgroup_size = subgroup.subgroupSize;
    }

    *out_ctx = ctx;
    return AC_SUCCESS;
}

AcResult ac_vk_pipeline_cache_merge(AcVkContext* ctx, const void* data, size_t size) {
    if (!ctx || !data) return AC_ERR_NULL_POINTER;
    if (ctx->pipeline_cache == VK_NULL_HANDLE) return AC_ERR_INVALID_CONTEXT;
    /* VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID, deviceID, pipelineCacheUUID[16] */
    if (size < 32) return AC_ERR_INVALID_ARGUMENT;
    const uint8_t* h = (const uint8_t*)data;
    uint32_t header_size, header_version, vendor, device;
    memcpy(&header_size, h, 4);
    memcpy(&header_version, h + 4, 4);
    memcpy(&vendor, h + 8, 4);
    memcpy(&device, h + 12, 4);
    if (header_size < 32 || header_size > size || header_version != (uint32_t)VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
        vendor != ctx->device_properties.vendorID || device != ctx->device_properties.deviceID ||
        memcmp(h + 16, ctx->device_properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    VkPipelineCacheCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pci.initialDataSize = size;
    pci.pInitialData = data;
    VkPipelineCache loaded = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(ctx->device, &pci, NULL, &loaded) != VK_SUCCESS) return AC_ERR_INVALID_ARGUMENT;
    const VkResult r = vkMergePipelineCaches(ctx->device, ctx->pipeline_cache, 1, &loaded);
    vkDestroyPipelineCache(ctx->device, loaded, NULL);
    return r == VK_SUCCESS ? AC_SUCCESS : AC_ERR_OUT_OF_HOST_MEMORY;
}

AcResult ac_vk_pipeline_cache_get_data(AcVkContext* ctx, void* data, size_t* inout_size) {
    if (!ctx || !inout_size) return AC_ERR_NULL_POINTER;
    if (ctx->pipeline_cache == VK_NULL_HANDLE) return AC_ERR_INVALID_CONTEXT;
    size_t n = data ? *inout_size : 0;
    const VkResult r = vkGetPipelineCacheData(ctx->device, ctx->pipeline_cache, &n, data);
    if (r == VK_INCOMPLETE) return AC_ERR_INVALID_BUFFER_SIZE;
    if (r != VK_SUCCESS) return AC_ERR_OUT_OF_HOST_MEMORY;
    *inout_size = n;
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
    if (ctx->recording_batches > 0) {
        return AC_ERR_RESOURCE_IN_USE;
    }

    if (ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        ac_vk_desc_cache_destroy(ctx);
        if (ctx->pipeline_cache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, NULL);
            ctx->pipeline_cache = VK_NULL_HANDLE;
        }
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
