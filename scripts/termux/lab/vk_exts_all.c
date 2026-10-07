/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// vk_exts_all: device name, driver version and every device extension of the first Vulkan device (lab probe).
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "exts", 1, "x", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai, 0, 0, 0, 0};
    VkInstance inst;
    if (vkCreateInstance(&ci, 0, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || n == 0) return 1;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pd, &p);
    printf("device %s driver 0x%x api %u.%u.%u\n", p.deviceName, p.driverVersion, VK_VERSION_MAJOR(p.apiVersion),
           VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, 0, &ne, 0);
    VkExtensionProperties* e = malloc(ne * sizeof(*e));
    vkEnumerateDeviceExtensionProperties(pd, 0, &ne, e);
    for (uint32_t i = 0; i < ne; ++i) printf("ext %s %u\n", e[i].extensionName, e[i].specVersion);
    free(e);
    vkDestroyInstance(inst, 0);
    return 0;
}
