#include "ac_kernel.h"
#include "ac_base.h"
#include "ac_vk_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

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
) {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci = {};
    VkDescriptorSetLayoutBinding bindings[16] = {};
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    VkPushConstantRange pushRange = {};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    VkComputePipelineCreateInfo pipelineInfo = {};
    VkSpecializationMapEntry map_entries[16] = {};
    uint32_t payload[16] = {};
    VkSpecializationInfo spec_info = {};
    AcResult ret = AC_SUCCESS;
    VkResult vk_res = VK_SUCCESS;
    uint32_t buffer_count = 0;
    size_t push_constant_size = 0;
    const char* entry_name = "main";

    if (!out_kernel) {
        return AC_ERR_NULL_POINTER;
    }
    *out_kernel = NULL;

    if (!ctx || ctx->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }
    if (!spirv_code || code_size_bytes < 4 || (code_size_bytes % 4 != 0)) {
        return AC_ERR_BINARY_CORRUPTED;
    }
    if (!metadata) {
        return AC_ERR_NULL_POINTER;
    }
    if (specialization_count > 16) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (specialization_count > 0 && !specializations) {
        return AC_ERR_NULL_POINTER;
    }

    /* 0. Device Workgroup Size and Invocation Limits Check */
    if (effective_local_size.x == 0 || effective_local_size.y == 0 || effective_local_size.z == 0) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    {
        const VkPhysicalDeviceLimits* limits = &ctx->device_properties.limits;
        if (effective_local_size.x > limits->maxComputeWorkGroupSize[0] ||
            effective_local_size.y > limits->maxComputeWorkGroupSize[1] ||
            effective_local_size.z > limits->maxComputeWorkGroupSize[2]) {
            return AC_ERR_INVALID_WORK_GROUP_SIZE;
        }
        const uint64_t invocation_count = (uint64_t)effective_local_size.x *
                                          (uint64_t)effective_local_size.y *
                                          (uint64_t)effective_local_size.z;
        if (invocation_count > limits->maxComputeWorkGroupInvocations) {
            return AC_ERR_INVALID_WORK_GROUP_SIZE;
        }
    }

    /* 1. Cryptographic Authentication against expected SHA-256 */
    if (metadata->expected_spirv_sha256 && metadata->expected_spirv_sha256[0] != '\0') {
        char actual_sha256[65];
        ac_sha256_compute_hex((const uint8_t*)spirv_code, code_size_bytes, actual_sha256);
        if (strcmp(actual_sha256, metadata->expected_spirv_sha256) != 0) {
            return AC_ERR_BINARY_CORRUPTED;
        }
    }

    /* 2. Feature Requirements Validation */
    if (metadata->required_features & AC_FEATURE_TIMESTAMPS) {
        if (!ctx->device_properties.limits.timestampComputeAndGraphics || ctx->timestamp_valid_bits == 0) {
            return AC_ERR_FEATURE_NOT_SUPPORTED;
        }
    }
    if ((metadata->required_features & ~(AC_FEATURE_TIMESTAMPS | AC_FEATURE_NON_COHERENT_ATOM)) != 0) {
        return AC_ERR_FEATURE_NOT_SUPPORTED;
    }

    buffer_count = metadata->binding_count;
    if (buffer_count == 0) {
        buffer_count = 3; /* Default for ternary GEMV */
    }
    if (buffer_count > 16) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }

    push_constant_size = metadata->push_constant_size;
    if (push_constant_size > 128) {
        return AC_ERR_KERNEL_ARG_INVALID;
    }

    if (metadata->entry_point && metadata->entry_point[0] != '\0') {
        entry_name = metadata->entry_point;
    }

    /* 3. Shader Module Creation */
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code_size_bytes;
    smci.pCode = spirv_code;

    vk_res = vkCreateShaderModule(ctx->device, &smci, NULL, &shader_module);
    if (vk_res != VK_SUCCESS) {
        return AC_ERR_KERNEL_BUILD_FAILED;
    }

    /* 4. Descriptor Set Layout */
    for (uint32_t i = 0; i < buffer_count; ++i) {
        bindings[i].binding = (metadata->bindings != NULL) ? metadata->bindings[i].binding_index : i;
        bindings[i].descriptorType = (metadata->bindings != NULL && metadata->bindings[i].binding_type == AC_BINDING_TYPE_UNIFORM_BUFFER)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = (metadata->bindings != NULL && metadata->bindings[i].stage_flags != 0)
            ? metadata->bindings[i].stage_flags
            : VK_SHADER_STAGE_COMPUTE_BIT;
    }

    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = buffer_count;
    layoutInfo.pBindings = bindings;

    vk_res = vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, NULL, &descriptor_set_layout);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_KERNEL_BUILD_FAILED;
        goto cleanup;
    }

    /* 5. Pipeline Layout with Push Constants */
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = (uint32_t)push_constant_size;

    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout;
    if (push_constant_size > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    }

    vk_res = vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, NULL, &pipeline_layout);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_KERNEL_BUILD_FAILED;
        goto cleanup;
    }

    /* 6. Compute Pipeline Creation with Specialization Constants */
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shader_module;
    pipelineInfo.stage.pName = entry_name;
    pipelineInfo.layout = pipeline_layout;

    if (specialization_count > 0) {
        for (uint32_t i = 0; i < specialization_count; ++i) {
            map_entries[i].constantID = specializations[i].constant_id;
            map_entries[i].offset = i * sizeof(uint32_t);
            map_entries[i].size = sizeof(uint32_t);
            payload[i] = specializations[i].value_bits;
        }
        spec_info.mapEntryCount = specialization_count;
        spec_info.pMapEntries = map_entries;
        spec_info.dataSize = specialization_count * sizeof(uint32_t);
        spec_info.pData = payload;
        pipelineInfo.stage.pSpecializationInfo = &spec_info;
    } else {
        pipelineInfo.stage.pSpecializationInfo = NULL;
    }

    vk_res = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, &pipelineInfo, NULL, &compute_pipeline);
    if (vk_res != VK_SUCCESS) {
        ret = AC_ERR_KERNEL_BUILD_FAILED;
        goto cleanup;
    }

    {
        AcVkKernel* k = (AcVkKernel*)calloc(1, sizeof(AcVkKernel));
        if (!k) {
            ret = AC_ERR_OUT_OF_HOST_MEMORY;
            goto cleanup;
        }

        k->shader_module = shader_module;
        k->descriptor_set_layout = descriptor_set_layout;
        k->pipeline_layout = pipeline_layout;
        k->compute_pipeline = compute_pipeline;
        k->push_constant_size = push_constant_size;
        k->buffer_count = buffer_count;
        k->owning_context = ctx;
        k->submission_outstanding_count = 0;
        k->metadata = *metadata;
        k->specialization_count = specialization_count;
        for (uint32_t i = 0; i < specialization_count; ++i) {
            k->specializations[i] = specializations[i];
        }
        k->effective_local_size = effective_local_size;
        if (pipeline_identity) {
            strncpy(k->pipeline_identity, pipeline_identity, sizeof(k->pipeline_identity) - 1);
        }
        /* The caller's metadata->bindings array need not outlive this call (k->metadata keeps only the pointer):
           the layout the descriptor writes must match is captured here. */
        for (uint32_t i = 0; i < buffer_count; ++i) {
            k->binding_index[i] = bindings[i].binding;
            k->binding_descriptor_type[i] = (uint32_t)bindings[i].descriptorType;
        }

        ctx->active_child_kernels++;
        *out_kernel = k;
        return AC_SUCCESS;
    }

cleanup:
    if (compute_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx->device, compute_pipeline, NULL);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx->device, pipeline_layout, NULL);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx->device, descriptor_set_layout, NULL);
    }
    if (shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(ctx->device, shader_module, NULL);
    }
    return ret;
}

AcResult ac_vk_kernel_create_with_metadata(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const AcKernelMetadata* metadata,
    AcVkKernel** out_kernel
) {
    AcDim3 local_size = {0, 0, 0};
    if (metadata) {
        local_size = metadata->local_size;
    }
    return ac_vk_kernel_create_with_specialization(
        ctx,
        spirv_code,
        code_size_bytes,
        metadata,
        0,
        NULL,
        local_size,
        NULL,
        out_kernel
    );
}

AcResult ac_vk_kernel_create_spirv(
    AcVkContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    size_t push_constant_size,
    uint32_t buffer_count,
    AcVkKernel** out_kernel
) {
    AcKernelMetadata default_meta = {};
    default_meta.kernel_name = "spirv_kernel";
    default_meta.entry_point = "main";
    default_meta.binding_count = buffer_count;
    default_meta.bindings = NULL;
    default_meta.push_constant_size = push_constant_size;
    default_meta.local_size = {0, 0, 0};
    default_meta.required_features = AC_FEATURE_NONE;
    default_meta.expected_spirv_sha256 = NULL;
    return ac_vk_kernel_create_with_metadata(ctx, spirv_code, code_size_bytes, &default_meta, out_kernel);
}

AcResult ac_vk_kernel_destroy(AcVkKernel* kernel) {
    if (!kernel) {
        return AC_SUCCESS;
    }

    if (kernel->submission_outstanding_count > 0) {
        return AC_ERR_DEVICE_BUSY;
    }

    if (!kernel->owning_context || kernel->owning_context->device == VK_NULL_HANDLE) {
        return AC_ERR_INVALID_CONTEXT;
    }

    VkDevice device = kernel->owning_context->device;

    /* cached descriptor sets of this layout must be freed before the layout is destroyed */
    ac_vk_desc_cache_purge_layout(kernel->owning_context, kernel->descriptor_set_layout);

    if (kernel->compute_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, kernel->compute_pipeline, NULL);
        kernel->compute_pipeline = VK_NULL_HANDLE;
    }
    if (kernel->pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, kernel->pipeline_layout, NULL);
        kernel->pipeline_layout = VK_NULL_HANDLE;
    }
    if (kernel->descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, kernel->descriptor_set_layout, NULL);
        kernel->descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (kernel->shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, kernel->shader_module, NULL);
        kernel->shader_module = VK_NULL_HANDLE;
    }

    if (kernel->owning_context->active_child_kernels > 0) {
        kernel->owning_context->active_child_kernels--;
    }

    kernel->owning_context = NULL;
    free(kernel);
    return AC_SUCCESS;
}

} // extern "C"
