/**
 * @file ac_kernel.h
 * @brief Universal Vulkan Compute Pipeline & Kernel Management with Metadata Authentication
 *
 * Supported Descriptor Scope (Phase 3 & 4 Contract):
 *   - Descriptor Set 0 only (Single descriptor set architecture; multi-set is planned for subsequent phases).
 *   - Resource Bindings: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER and VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER.
 *   - Push Constants: Up to 128 bytes (enforced by hardware limit check).
 *   - Workgroup Local Dimensions: 1D, 2D, 3D up to device limits.
 *   - Bytecode Verification: Cryptographically authenticated against expected_spirv_sha256.
 *
 * Contract:
 *   - Rejects destruction with AC_ERR_DEVICE_BUSY if submission_outstanding_count > 0.
 *   - Validates SPIR-V bytecode size alignment, word count, and optional SHA-256 hash.
 *   - Enforces explicit owning_context relationship.
 */
#ifndef AC_KERNEL_H
#define AC_KERNEL_H

#include "ac_base.h"
#include "ac_context.h"
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AC_BINDING_TYPE_STORAGE_BUFFER = 0,
    AC_BINDING_TYPE_UNIFORM_BUFFER = 1
} AcBindingType;

typedef struct {
    uint32_t binding_index;
    AcBindingType binding_type;
    VkShaderStageFlags stage_flags;
} AcKernelBindingDesc;

typedef struct {
    const char* kernel_name;
    const char* entry_point;
    uint32_t binding_count;
    const AcKernelBindingDesc* bindings;
    size_t push_constant_size;
    AcDim3 local_size;
    uint32_t required_features;
    const char* expected_spirv_sha256;
    uint32_t specialization_schema_count;
    const AcSpecConstantSchemaEntry* specialization_schemas;
} AcKernelMetadata;

struct AcVkKernel {
    VkShaderModule shader_module;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    size_t push_constant_size;
    uint32_t buffer_count;
    AcVkContext* owning_context;
    uint32_t submission_outstanding_count;
    AcKernelMetadata metadata;
    uint32_t specialization_count;
    AcSpecConstant specializations[16];
    AcDim3 effective_local_size;
    char pipeline_identity[AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE];
    /* descriptor set layout as created: binding numbers and VkDescriptorType per buffer slot */
    uint32_t binding_index[16];
    uint32_t binding_descriptor_type[16];
};

/**
 * @brief Create a compute kernel with explicit specialization constants and metadata.
 */
AcResult ac_vk_kernel_create_with_specialization(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const AcKernelMetadata* metadata,
    uint32_t specialization_count,
    const AcSpecConstant* specializations,
    AcDim3 effective_local_size,
    const char* pipeline_identity,
    AcVkKernel** out_kernel
);

/**
 * @brief Create a compute kernel with explicit universal metadata and validation.
 *
 * @param ctx Owning Vulkan compute context.
 * @param spirv_code Pointer to 32-bit aligned SPIR-V bytecode words.
 * @param code_size_bytes Byte size of the SPIR-V bytecode.
 * @param metadata Kernel metadata descriptor containing entry point, bindings, push constant size, SHA-256.
 * @param out_kernel Output kernel pointer (set to NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vk_kernel_create_with_metadata(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const AcKernelMetadata* metadata,
    AcVkKernel** out_kernel
);

/**
 * @brief Backwards-compatible SPIR-V kernel creation wrapper.
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
