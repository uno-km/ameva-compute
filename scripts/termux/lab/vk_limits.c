#include <vulkan/vulkan.h>
#include <stdio.h>
int main(void) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "limits", 1, "x", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai, 0, 0, 0, 0};
    VkInstance inst;
    if (vkCreateInstance(&ci, 0, &inst) != VK_SUCCESS) { printf("instance failed\n"); return 1; }
    uint32_t n = 4; VkPhysicalDevice pd[4];
    vkEnumeratePhysicalDevices(inst, &n, pd);
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[i], &p);
        VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd[i], &mp);
        printf("%s api=%u.%u driver=0x%x\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), p.driverVersion);
        printf("maxStorageBufferRange=%u maxPushConstantsSize=%u minStorageBufferOffsetAlignment=%llu maxComputeSharedMemorySize=%u maxComputeWorkGroupInvocations=%u maxComputeWorkGroupCount=%u,%u,%u timestampPeriod=%.3f timestampComputeAndGraphics=%u maxBoundDescriptorSets=%u maxDescriptorSetStorageBuffers=%u\n",
               p.limits.maxStorageBufferRange, p.limits.maxPushConstantsSize, (unsigned long long)p.limits.minStorageBufferOffsetAlignment,
               p.limits.maxComputeSharedMemorySize, p.limits.maxComputeWorkGroupInvocations, p.limits.maxComputeWorkGroupCount[0],
               p.limits.maxComputeWorkGroupCount[1], p.limits.maxComputeWorkGroupCount[2], p.limits.timestampPeriod,
               p.limits.timestampComputeAndGraphics, p.limits.maxBoundDescriptorSets, p.limits.maxPerStageDescriptorStorageBuffers);
        printf("subgroupSize=%u subgroupStages=0x%x subgroupOps=0x%x\n", sg.subgroupSize, sg.supportedStages, sg.supportedOperations);
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
            printf("memtype %u heap %u flags 0x%x heapsize %llu MiB\n", t, mp.memoryTypes[t].heapIndex, mp.memoryTypes[t].propertyFlags,
                   (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[t].heapIndex].size >> 20));
    }
    vkDestroyInstance(inst, 0);
    return 0;
}
