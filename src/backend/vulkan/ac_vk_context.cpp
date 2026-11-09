/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ac_context.h"
#include "ac_base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_vk_int_dot_probe_spv.h"

/* Known-answer test of the signed packed int8 dot product on this device: compiles a small kernel, runs 16 products
   through it (the extremes of int8 and pseudo-random patterns) and compares them with a scalar reference. Returns 1
   when every product is right; otherwise 0 and the reason in `why`. Used when the operation is called without
   VK_KHR_shader_integer_dot_product being offered, so that a driver that refuses it or computes it wrong is detected. */
static int ac_vk_int_dot_probe(VkPhysicalDevice phys, VkDevice dev, VkQueue queue, VkCommandPool pool, char* why,
                               size_t why_size) {
    int ok = 0;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkShaderModule sm = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    do {
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = 48 * sizeof(int32_t);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(dev, &bi, NULL, &buf) != VK_SUCCESS) { snprintf(why, why_size, "vkCreateBuffer"); break; }
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t ti = mp.memoryTypeCount;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { ti = i; break; }
        }
        if (ti == mp.memoryTypeCount) { snprintf(why, why_size, "no host-visible coherent memory"); break; }
        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = ti;
        if (vkAllocateMemory(dev, &ai, NULL, &mem) != VK_SUCCESS || vkBindBufferMemory(dev, buf, mem, 0) != VK_SUCCESS) {
            snprintf(why, why_size, "buffer memory");
            break;
        }
        int32_t* d = NULL;
        if (vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void**)&d) != VK_SUCCESS) { snprintf(why, why_size, "vkMapMemory"); break; }
        static const uint32_t edge[4][2] = { { 0x80808080u, 0x80808080u }, { 0x7F7F7F7Fu, 0x80808080u },
                                             { 0x7F7F7F7Fu, 0x7F7F7F7Fu }, { 0x01FF80FEu, 0xFF7F02C3u } };
        uint32_t x = 0x12345678u;
        int32_t expect[16];
        for (int i = 0; i < 16; ++i) {
            uint32_t a, b;
            if (i < 4) { a = edge[i][0]; b = edge[i][1]; }
            else { x = x * 1664525u + 1013904223u; a = x; x = x * 1664525u + 1013904223u; b = x; }
            d[i] = (int32_t)a;
            d[16 + i] = (int32_t)b;
            d[32 + i] = 0x5A5A5A5A;
            int32_t e = 0;
            for (int k = 0; k < 4; ++k) e += (int32_t)(int8_t)(a >> (8 * k)) * (int32_t)(int8_t)(b >> (8 * k));
            expect[i] = e;
        }
        VkShaderModuleCreateInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        si.codeSize = sizeof(kAcVkIntDotProbeSpv);
        si.pCode = kAcVkIntDotProbeSpv;
        VkResult r = vkCreateShaderModule(dev, &si, NULL, &sm);
        if (r != VK_SUCCESS) { snprintf(why, why_size, "vkCreateShaderModule refused the kernel (%d)", (int)r); break; }
        VkDescriptorSetLayoutBinding lb = {};
        lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb.descriptorCount = 1;
        lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &lb;
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorSetLayout(dev, &li, NULL, &dsl) != VK_SUCCESS || vkCreateDescriptorPool(dev, &pi, NULL, &dp) != VK_SUCCESS) {
            snprintf(why, why_size, "descriptor layout or pool");
            break;
        }
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        if (vkCreatePipelineLayout(dev, &pli, NULL, &pl) != VK_SUCCESS) { snprintf(why, why_size, "pipeline layout"); break; }
        VkDescriptorSetAllocateInfo dai = {};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = dp;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev, &dai, &ds) != VK_SUCCESS) { snprintf(why, why_size, "descriptor set"); break; }
        VkDescriptorBufferInfo dbi = { buf, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        ci.layout = pl;
        r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
        if (r != VK_SUCCESS) { snprintf(why, why_size, "the driver refused the kernel (vkCreateComputePipelines %d)", (int)r); break; }
        VkCommandBufferAllocateInfo cai = {};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBufferBeginInfo bgi = {};
        bgi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkAllocateCommandBuffers(dev, &cai, &cb) != VK_SUCCESS || vkBeginCommandBuffer(cb, &bgi) != VK_SUCCESS) {
            snprintf(why, why_size, "command buffer");
            break;
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cb, 1, 1, 1);
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        VkFenceCreateInfo fi = {};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkSubmitInfo sub = {};
        sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;
        if (vkEndCommandBuffer(cb) != VK_SUCCESS || vkCreateFence(dev, &fi, NULL, &fence) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &sub, fence) != VK_SUCCESS) {
            snprintf(why, why_size, "submission");
            break;
        }
        r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull);
        if (r != VK_SUCCESS) { snprintf(why, why_size, "the kernel did not finish (%d)", (int)r); break; }
        int bad = -1;
        for (int i = 0; i < 16 && bad < 0; ++i) if (d[32 + i] != expect[i]) bad = i;
        if (bad >= 0) { snprintf(why, why_size, "wrong result at %d: %d, expected %d", bad, d[32 + bad], expect[bad]); break; }
        ok = 1;
    } while (0);
    if (cb) vkFreeCommandBuffers(dev, pool, 1, &cb);
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (pipe) vkDestroyPipeline(dev, pipe, NULL);
    if (pl) vkDestroyPipelineLayout(dev, pl, NULL);
    if (dp) vkDestroyDescriptorPool(dev, dp, NULL);
    if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    if (sm) vkDestroyShaderModule(dev, sm, NULL);
    if (mem) vkFreeMemory(dev, mem, NULL);
    if (buf) vkDestroyBuffer(dev, buf, NULL);
    return ok;
}

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

    /* Packed int8 dot product on a device that does not offer VK_KHR_shader_integer_dot_product (outside the Vulkan
       specification). Mali-G78 with driver 38 exposes the operation through OpenCL only, yet its Vulkan compiler accepts
       it: Galaxy S21 r38p0, prompt 1.8-2.3x, generation +10-16 % (2026-11-09); Mali-G68 r32p1 refuses the kernels.
       Unset: on for that driver only, after a known-answer test, announced on stderr, and off (with the reason) when the
       test fails. AC_VK_FORCE_INT_DOT=1: any device, failure of the test is an error. =0: off. */
    if (!intDot) {
        const char* forceIntDot = getenv("AC_VK_FORCE_INT_DOT");
        const bool forceOn = forceIntDot && strcmp(forceIntDot, "1") == 0;
        const bool forceOff = forceIntDot && strcmp(forceIntDot, "0") == 0;
        const bool listed = selectedProps.vendorID == 0x13B5u && strstr(selectedProps.deviceName, "Mali-G78") != NULL &&
                            VK_VERSION_MAJOR(selectedProps.driverVersion) == 38;
        if (forceOn || (listed && !forceOff)) {
            char why[160] = "";
            if (ac_vk_int_dot_probe(selectedDevice, device, computeQueue, commandPool, why, sizeof(why))) {
                intDot = 1;
                fprintf(stderr, "ac_runtime: int8 dot product on Vulkan without VK_KHR_shader_integer_dot_product (%s, "
                                "known-answer test passed, outside the Vulkan specification; AC_VK_FORCE_INT_DOT=0 turns it off)\n",
                        forceOn ? "AC_VK_FORCE_INT_DOT=1" : selectedProps.deviceName);
            } else if (forceOn) {
                fprintf(stderr, "ac_runtime: AC_VK_FORCE_INT_DOT=1: known-answer test failed: %s\n", why);
                vkDestroyCommandPool(device, commandPool, NULL);
                vkDestroyDevice(device, NULL);
                vkDestroyInstance(instance, NULL);
                return AC_ERR_DRIVER_INIT_FAILED;
            } else {
                fprintf(stderr, "ac_runtime: int8 dot product stays off on Vulkan: known-answer test failed: %s\n", why);
            }
        }
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
