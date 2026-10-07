/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// vk_specs: properties of the first Vulkan device that could steer kernel selection (lab probe).
//   device, vendor/device ID, driver version and driver info, subgroup size and its controllable range, shared memory,
//   workgroup limits, integer dot product, and on Arm GPUs the shader core count (VK_ARM_shader_core_builtins /
//   VK_ARM_shader_core_properties when the driver offers them).
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has_ext(VkExtensionProperties* e, uint32_t n, const char* name) {
    for (uint32_t i = 0; i < n; ++i) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

int main(void) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "specs", 1, "x", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai, 0, 0, 0, 0};
    VkInstance inst;
    if (vkCreateInstance(&ci, 0, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || n == 0) return 1;
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, 0, &ne, 0);
    VkExtensionProperties* e = calloc(ne, sizeof(*e));
    vkEnumerateDeviceExtensionProperties(pd, 0, &ne, e);

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pd, &p);
    void* chain = 0;
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, chain};
    chain = &sg;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT, chain};
    if (has_ext(e, ne, "VK_EXT_subgroup_size_control")) chain = &sgc;
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, chain};
    if (VK_VERSION_MINOR(p.apiVersion) >= 2 || has_ext(e, ne, "VK_KHR_driver_properties")) chain = &drv;
    VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR idp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR, chain};
    if (has_ext(e, ne, "VK_KHR_shader_integer_dot_product")) chain = &idp;
#ifdef VK_ARM_shader_core_builtins
    VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM cb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM, chain};
    if (has_ext(e, ne, "VK_ARM_shader_core_builtins")) chain = &cb;
#endif
#ifdef VK_ARM_shader_core_properties
    VkPhysicalDeviceShaderCorePropertiesARM cp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM, chain};
    if (has_ext(e, ne, "VK_ARM_shader_core_properties")) chain = &cp;
#endif
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, chain};
    vkGetPhysicalDeviceProperties2(pd, &p2);

    printf("device=%s vendor=0x%x deviceID=0x%x api=%u.%u.%u driverVersion=0x%x\n", p.deviceName, p.vendorID, p.deviceID,
           VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion), p.driverVersion);
    if (chain && (VK_VERSION_MINOR(p.apiVersion) >= 2 || has_ext(e, ne, "VK_KHR_driver_properties")))
        printf("driverName=%s driverInfo=%s\n", drv.driverName, drv.driverInfo);
    printf("subgroupSize=%u subgroupOps=0x%x", sg.subgroupSize, sg.supportedOperations);
    if (has_ext(e, ne, "VK_EXT_subgroup_size_control"))
        printf(" subgroupSizeRange=%u..%u", sgc.minSubgroupSize, sgc.maxSubgroupSize);
    printf("\nsharedMemory=%u maxWorkgroupInvocations=%u maxWorkgroupSize=%u,%u,%u\n", p.limits.maxComputeSharedMemorySize,
           p.limits.maxComputeWorkGroupInvocations, p.limits.maxComputeWorkGroupSize[0], p.limits.maxComputeWorkGroupSize[1],
           p.limits.maxComputeWorkGroupSize[2]);
    if (has_ext(e, ne, "VK_KHR_shader_integer_dot_product"))
        printf("intDot4x8PackedSignedAccelerated=%u\n", idp.integerDotProduct4x8BitPackedSignedAccelerated);
    else
        printf("intDot=absent\n");
#ifdef VK_ARM_shader_core_builtins
    if (has_ext(e, ne, "VK_ARM_shader_core_builtins"))
        printf("armShaderCoreCount=%u armWarpsPerCore=%u\n", cb.shaderCoreCount, cb.shaderWarpsPerCore);
    else
        printf("VK_ARM_shader_core_builtins=absent\n");
#endif
#ifdef VK_ARM_shader_core_properties
    if (has_ext(e, ne, "VK_ARM_shader_core_properties"))
        printf("armPixelRate=%u armTexelRate=%u armFmaRate=%u\n", cp.pixelRate, cp.texelRate, cp.fmaRate);
    else
        printf("VK_ARM_shader_core_properties=absent\n");
#endif
    free(e);
    vkDestroyInstance(inst, 0);
    return 0;
}
