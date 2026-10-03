#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "ext", 1, "x", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai, 0, 0, 0, 0};
    VkInstance inst;
    if (vkCreateInstance(&ci, 0, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1; VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, 0, &ne, 0);
    VkExtensionProperties* e = malloc(ne * sizeof(*e)); vkEnumerateDeviceExtensionProperties(pd, 0, &ne, e);
    const char* want[] = {"VK_KHR_push_descriptor", "VK_KHR_maintenance3", "VK_KHR_maintenance4", "VK_KHR_shader_float16_int8",
                          "VK_KHR_16bit_storage", "VK_KHR_8bit_storage", "VK_EXT_subgroup_size_control", "VK_KHR_shader_integer_dot_product",
                          "VK_KHR_cooperative_matrix", "VK_KHR_buffer_device_address", "VK_KHR_timeline_semaphore", "VK_KHR_synchronization2",
                          "VK_EXT_robustness2", "VK_KHR_shader_subgroup_extended_types"};
    for (unsigned w = 0; w < sizeof(want)/sizeof(want[0]); ++w) {
        int found = 0; for (uint32_t i = 0; i < ne; ++i) if (!strcmp(e[i].extensionName, want[w])) found = 1;
        printf("%-40s %s\n", want[w], found ? "yes" : "NO");
    }
    VkPhysicalDeviceMaintenance3Properties m3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDevicePushDescriptorPropertiesKHR pdp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR, &m3};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pdp};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("maxPushDescriptors=%u maxMemoryAllocationSize=%llu MiB maxPerSetDescriptors=%u\n", pdp.maxPushDescriptors,
           (unsigned long long)(m3.maxMemoryAllocationSize >> 20), m3.maxPerSetDescriptors);
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("shaderFloat16=%u shaderInt8=%u storageBuffer16BitAccess=%u storageBuffer8BitAccess=%u shaderInt64=%u shaderInt16=%u robustBufferAccess=%u\n",
           f12.shaderFloat16, f12.shaderInt8, f11.storageBuffer16BitAccess, f12.storageBuffer8BitAccess, f2.features.shaderInt64,
           f2.features.shaderInt16, f2.features.robustBufferAccess);
    vkDestroyInstance(inst, 0);
    return 0;
}
