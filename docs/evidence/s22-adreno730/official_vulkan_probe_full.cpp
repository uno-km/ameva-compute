#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>

static const char* get_device_type_str(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "INTEGRATED_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "DISCRETE_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "VIRTUAL_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
        default: return "OTHER";
    }
}

int main(void) {
    printf("=== OFFICIAL VULKAN DEVICE LIMITS PROBE ===\n");

    void* lib = dlopen("/system/lib64/libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
        fprintf(stderr, "FAIL: Cannot dlopen libvulkan.so: %s\n", dlerror());
        return 1;
    }

    PFN_vkCreateInstance pfn_vkCreateInstance = (PFN_vkCreateInstance)dlsym(lib, "vkCreateInstance");
    PFN_vkDestroyInstance pfn_vkDestroyInstance = (PFN_vkDestroyInstance)dlsym(lib, "vkDestroyInstance");
    PFN_vkEnumeratePhysicalDevices pfn_vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)dlsym(lib, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties pfn_vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)dlsym(lib, "vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties pfn_vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)dlsym(lib, "vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties pfn_vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)dlsym(lib, "vkGetPhysicalDeviceMemoryProperties");

    if (!pfn_vkCreateInstance || !pfn_vkDestroyInstance || !pfn_vkEnumeratePhysicalDevices || !pfn_vkGetPhysicalDeviceProperties) {
        fprintf(stderr, "FAIL: Cannot resolve core Vulkan symbols\n");
        dlclose(lib);
        return 2;
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "DeviceLimitProbeFull";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AMEVA";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = pfn_vkCreateInstance(&instInfo, NULL, &instance);
    if (res != VK_SUCCESS || instance == VK_NULL_HANDLE) {
        fprintf(stderr, "FAIL: vkCreateInstance returned %d\n", res);
        dlclose(lib);
        return 3;
    }

    uint32_t deviceCount = 0;
    res = pfn_vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (res != VK_SUCCESS || deviceCount == 0) {
        fprintf(stderr, "FAIL: vkEnumeratePhysicalDevices count returned %d (count=%u)\n", res, deviceCount);
        pfn_vkDestroyInstance(instance, NULL);
        dlclose(lib);
        return 4;
    }

    VkPhysicalDevice* physicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
    res = pfn_vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "FAIL: vkEnumeratePhysicalDevices query returned %d\n", res);
        free(physicalDevices);
        pfn_vkDestroyInstance(instance, NULL);
        dlclose(lib);
        return 5;
    }

    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDevice dev = physicalDevices[i];
        VkPhysicalDeviceProperties props = {};
        pfn_vkGetPhysicalDeviceProperties(dev, &props);

        printf("PhysicalDeviceIndex: %u\n", i);
        printf("deviceName: %s\n", props.deviceName);
        printf("vendorID: 0x%04X\n", props.vendorID);
        printf("deviceID: 0x%04X\n", props.deviceID);
        printf("deviceType: %s (%u)\n", get_device_type_str(props.deviceType), props.deviceType);
        printf("driverVersion: 0x%08X\n", props.driverVersion);
        printf("apiVersion: %u.%u.%u (0x%08X)\n",
               VK_VERSION_MAJOR(props.apiVersion),
               VK_VERSION_MINOR(props.apiVersion),
               VK_VERSION_PATCH(props.apiVersion),
               props.apiVersion);
        printf("maxComputeWorkGroupCount: [%u, %u, %u]\n",
               props.limits.maxComputeWorkGroupCount[0],
               props.limits.maxComputeWorkGroupCount[1],
               props.limits.maxComputeWorkGroupCount[2]);
        printf("maxComputeWorkGroupSize: [%u, %u, %u]\n",
               props.limits.maxComputeWorkGroupSize[0],
               props.limits.maxComputeWorkGroupSize[1],
               props.limits.maxComputeWorkGroupSize[2]);
        printf("maxComputeWorkGroupInvocations: %u\n",
               props.limits.maxComputeWorkGroupInvocations);
        printf("nonCoherentAtomSize: %llu bytes\n", (unsigned long long)props.limits.nonCoherentAtomSize);
        printf("timestampPeriod: %f ns/tick\n", props.limits.timestampPeriod);

        // Queue Families
        if (pfn_vkGetPhysicalDeviceQueueFamilyProperties) {
            uint32_t qfCount = 0;
            pfn_vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, NULL);
            printf("queueFamily count: %u\n", qfCount);
            if (qfCount > 0) {
                VkQueueFamilyProperties* qfProps = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * qfCount);
                pfn_vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfProps);
                for (uint32_t q = 0; q < qfCount; ++q) {
                    printf("  QueueFamily[%u]: flags=0x%04X (compute=%s, timestampValidBits=%u)\n",
                           q, qfProps[q].queueFlags,
                           (qfProps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "TRUE" : "FALSE",
                           qfProps[q].timestampValidBits);
                }
                free(qfProps);
            }
        }

        // Memory Properties
        if (pfn_vkGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties memProps = {};
            pfn_vkGetPhysicalDeviceMemoryProperties(dev, &memProps);
            printf("memoryTypeCount: %u\n", memProps.memoryTypeCount);
            for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
                VkMemoryPropertyFlags flags = memProps.memoryTypes[m].propertyFlags;
                int is_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                int is_visible  = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                printf("  MemoryType[%u]: heapIndex=%u, flags=0x%04X (hostVisible=%d, hostCoherent=%d)\n",
                       m, memProps.memoryTypes[m].heapIndex, flags, is_visible, is_coherent);
            }
        }
    }

    free(physicalDevices);
    pfn_vkDestroyInstance(instance, NULL);
    dlclose(lib);
    printf("=== PROBE COMPLETE ===\n");
    return 0;
}
