/**
 * @file ac_kernel.h
 * @brief Vulkan Compute Pipeline & Kernel Management
 */
#ifndef AC_KERNEL_H
#define AC_KERNEL_H

#include "ac_types.h"
#include "ac_context.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AcVkKernel {
    VkShaderModule shader_module;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    size_t push_constant_size;
    uint32_t buffer_count;
    AcVkContext* owning_context;
    uint32_t in_flight_count;
};

AcResult ac_vk_kernel_create_spirv(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    size_t push_constant_size,
    uint32_t buffer_count,
    AcVkKernel** out_kernel
);

AcResult ac_vk_kernel_destroy(AcVkKernel* kernel);

#ifdef __cplusplus
}
#endif

#endif /* AC_KERNEL_H */
