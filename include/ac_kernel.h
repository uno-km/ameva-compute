/**
 * @file ac_kernel.h
 * @brief Vulkan Compute Pipeline & Kernel Management
 *
 * Contract:
 *   - Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *   - Validates SPIR-V bytecode size alignment and word count.
 *   - Enforces explicit owning_context relationship.
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
    uint32_t submission_outstanding_count;
};

/**
 * @brief Create a compute kernel from SPIR-V bytecode.
 *
 * @param ctx Owning Vulkan compute context.
 * @param spirv_code Pointer to 32-bit aligned SPIR-V bytecode words.
 * @param code_size_bytes Byte size of the SPIR-V bytecode.
 * @param push_constant_size Size of push constant block in bytes (0 if unused).
 * @param buffer_count Number of storage buffer bindings (0 defaults to 3).
 * @param out_kernel Output kernel pointer (set to NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_kernel_create_spirv(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    size_t push_constant_size,
    uint32_t buffer_count,
    AcVkKernel** out_kernel
);

/**
 * @brief Safely destroy a Vulkan compute kernel.
 * Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *
 * @param kernel Kernel to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_kernel_destroy(AcVkKernel* kernel);

#ifdef __cplusplus
}
#endif

#endif /* AC_KERNEL_H */
