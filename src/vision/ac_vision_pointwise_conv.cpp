/**
 * @file ac_vision_pointwise_conv.cpp
 * @brief AMUDA Vision Pointwise Conv2D 1x1 Lowered to MatMul Implementation
 *
 * Enforces Zero-Silent-Fallback, strict ABI checks, and hardware boundary safety.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ac_vision_ops.h"
#include "ac_runtime.h"
#include "../shaders/pointwise_conv_spv.h"
#include "../backend/vulkan/ac_vk_sha256.h"

static const AcRuntimeBindingDesc kPointwiseConvBindings[4] = {
    { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
    { 3, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
};

static const AcKernelRuntimeMetadata kDefaultPointwiseConvMetadata = {
    "pointwise_conv",
    "main",
    4,
    kPointwiseConvBindings,
    sizeof(uint32_t) * 4, /* push_constant_size: M, K, N, has_bias */
    { 16, 16, 1 },
    0,
    kPointwiseConvSpvSha256,
    0,
    NULL
};

AcResult ac_vision_pointwise_conv_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
) {
    if (!out_kernel) {
        return AC_ERR_NULL_POINTER;
    }
    *out_kernel = NULL;

    if (!ctx) {
        return AC_ERR_NULL_POINTER;
    }

    const uint32_t* code = spirv_code ? spirv_code : kPointwiseConvSpv;
    size_t code_size = code_size_bytes ? code_size_bytes : kPointwiseConvSpvByteSize;

    char computed_sha256[65] = {0};
    AcKernelRuntimeMetadata meta = kDefaultPointwiseConvMetadata;
    if (expected_sha256) {
        meta.expected_spirv_sha256 = expected_sha256;
    } else if (spirv_code && spirv_code != kPointwiseConvSpv) {
        ac_sha256_compute_hex((const uint8_t*)code, code_size, computed_sha256);
        meta.expected_spirv_sha256 = computed_sha256;
    } else {
        meta.expected_spirv_sha256 = kPointwiseConvSpvSha256;
    }

    AcKernelDesc kdesc = {};
    kdesc.abi_version = AC_ABI_VERSION;
    kdesc.struct_size = sizeof(AcKernelDesc);
    kdesc.spirv_code = code;
    kdesc.code_size_bytes = code_size;
    kdesc.metadata = &meta;
    kdesc.specialization_count = 0;
    kdesc.specializations = NULL;

    return ac_runtime_kernel_create(ctx, &kdesc, out_kernel);
}

AcResult ac_vision_pointwise_conv_kernel_destroy(
    AcKernel* kernel
) {
    return ac_runtime_kernel_destroy(kernel);
}

AcResult ac_vision_pointwise_conv_with_kernel_f32(
    AcContext* ctx,
    AcKernel* kernel,
    const AcVisionPointwiseConvParams* params,
    AcBuffer* in_buf,
    const AcVisionTensorDesc* in_desc,
    AcBuffer* weight_buf,
    const AcVisionTensorDesc* weight_desc,
    AcBuffer* bias_buf,
    AcBuffer* out_buf,
    const AcVisionTensorDesc* out_desc,
    AcEvent** out_event
) {
    /* 1. Output safety invariant */
    if (out_event) {
        *out_event = NULL;
    }

    /* 2. Null pointer validation */
    if (!ctx || !kernel || !params) {
        return AC_ERR_NULL_POINTER;
    }
    if (!in_buf || !in_desc) {
        return AC_ERR_NULL_POINTER;
    }
    if (!weight_buf || !weight_desc) {
        return AC_ERR_NULL_POINTER;
    }
    if (!out_buf || !out_desc) {
        return AC_ERR_NULL_POINTER;
    }
    if (params->has_bias != 0 && !bias_buf) {
        return AC_ERR_NULL_POINTER;
    }

    /* 3. ABI version validation */
    if (params->abi_version != AC_VISION_ABI_VERSION || params->struct_size != sizeof(AcVisionPointwiseConvParams)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (in_desc->abi_version != AC_VISION_ABI_VERSION || in_desc->struct_size != sizeof(AcVisionTensorDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (weight_desc->abi_version != AC_VISION_ABI_VERSION || weight_desc->struct_size != sizeof(AcVisionTensorDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (out_desc->abi_version != AC_VISION_ABI_VERSION || out_desc->struct_size != sizeof(AcVisionTensorDesc)) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 4. Dtype validation */
    if (in_desc->dtype != AC_VISION_DTYPE_F32 ||
        weight_desc->dtype != AC_VISION_DTYPE_F32 ||
        out_desc->dtype != AC_VISION_DTYPE_F32) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 5. Layout validation */
    if (in_desc->layout != AC_VISION_LAYOUT_NHWC ||
        weight_desc->layout != AC_VISION_LAYOUT_OHWI ||
        out_desc->layout != AC_VISION_LAYOUT_NHWC) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 6. Batch and Contract validation */
    if (in_desc->n != 1 || out_desc->n != 1) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (params->kernel_h != 1 || params->kernel_w != 1) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (params->stride_h != 1 || params->stride_w != 1) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (params->dilation_h != 1 || params->dilation_w != 1) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 7. Zero dimension check */
    if (params->input_height == 0 || params->input_width == 0 ||
        params->input_channels == 0 || params->output_channels == 0) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 8. Shape dimension consistency */
    if (in_desc->h != params->input_height || in_desc->w != params->input_width ||
        in_desc->c != params->input_channels) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (weight_desc->n != params->output_channels || weight_desc->h != 1 ||
        weight_desc->w != 1 || weight_desc->c != params->input_channels) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (out_desc->h != params->input_height || out_desc->w != params->input_width ||
        out_desc->c != params->output_channels) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 9. Multiplication overflow checks */
    uint64_t M64 = (uint64_t)params->input_height * (uint64_t)params->input_width;
    if (M64 > 104857600ull) { /* 100M spatial elements ceiling */
        return AC_ERR_INVALID_ARGUMENT;
    }

    uint64_t in_elems = M64 * (uint64_t)params->input_channels;
    uint64_t w_elems = (uint64_t)params->output_channels * (uint64_t)params->input_channels;
    uint64_t out_elems = M64 * (uint64_t)params->output_channels;

    if (in_elems > (SIZE_MAX / sizeof(float)) ||
        w_elems > (SIZE_MAX / sizeof(float)) ||
        out_elems > (SIZE_MAX / sizeof(float))) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    size_t in_req_bytes = (size_t)in_elems * sizeof(float);
    size_t w_req_bytes = (size_t)w_elems * sizeof(float);
    size_t out_req_bytes = (size_t)out_elems * sizeof(float);

    /* 10. Dispatch budget checks */
    uint32_t grid_x = (params->output_channels + 15) / 16;
    uint32_t grid_y = ((uint32_t)M64 + 15) / 16;
    if (grid_x == 0 || grid_y == 0 ||
        grid_x > 1048576u || grid_y > 1048576u ||
        ((uint64_t)grid_x * (uint64_t)grid_y > 16777216ull)) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }

    /* 11. Buffer size validation */
    size_t in_buf_size = 0, w_buf_size = 0, out_buf_size = 0;
    if (ac_runtime_buffer_get_size(in_buf, &in_buf_size) != AC_SUCCESS || in_buf_size < in_req_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (ac_runtime_buffer_get_size(weight_buf, &w_buf_size) != AC_SUCCESS || w_buf_size < w_req_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (ac_runtime_buffer_get_size(out_buf, &out_buf_size) != AC_SUCCESS || out_buf_size < out_req_bytes) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (params->has_bias != 0) {
        size_t bias_buf_size = 0;
        size_t bias_req_bytes = (size_t)params->output_channels * sizeof(float);
        if (ac_runtime_buffer_get_size(bias_buf, &bias_buf_size) != AC_SUCCESS || bias_buf_size < bias_req_bytes) {
            return AC_ERR_INVALID_ARGUMENT;
        }
    }

    /* 12. Contiguous layout validation */
    if (in_desc->row_stride_bytes != 0 &&
        in_desc->row_stride_bytes != (size_t)in_desc->w * in_desc->c * sizeof(float)) {
        return AC_ERR_INVALID_ARGUMENT;
    }
    if (out_desc->row_stride_bytes != 0 &&
        out_desc->row_stride_bytes != (size_t)out_desc->w * out_desc->c * sizeof(float)) {
        return AC_ERR_INVALID_ARGUMENT;
    }

    /* 13. Forced failure injection point */
    const char* fail_stage = getenv("AMEVA_VISION_FAIL_STAGE");
    if (fail_stage && strcmp(fail_stage, "dispatch") == 0) {
        return AC_ERR_LAUNCH_TIMEOUT;
    }

    /* 14. Dispatch configuration */
    AcBuffer* bound_bias = (params->has_bias && bias_buf) ? bias_buf : out_buf;
    AcBuffer* bound_buffers[4] = { in_buf, weight_buf, bound_bias, out_buf };

    uint32_t push_constants[4] = {
        (uint32_t)M64,
        params->input_channels,
        params->output_channels,
        params->has_bias
    };

    AcLaunchDesc ldesc = {};
    ldesc.abi_version = AC_ABI_VERSION;
    ldesc.struct_size = sizeof(AcLaunchDesc);
    ldesc.kernel = kernel;
    ldesc.buffer_count = 4;
    ldesc.buffers = bound_buffers;
    ldesc.push_constants = push_constants;
    ldesc.push_constants_size = sizeof(push_constants);
    ldesc.grid_dim = (AcDim3){ grid_x, grid_y, 1 };

    return ac_runtime_dispatch(ctx, &ldesc, out_event);
}

AcResult ac_vision_pointwise_conv_f32(
    AcContext* ctx,
    const AcVisionPointwiseConvParams* params,
    AcBuffer* in_buf,
    const AcVisionTensorDesc* in_desc,
    AcBuffer* weight_buf,
    const AcVisionTensorDesc* weight_desc,
    AcBuffer* bias_buf,
    AcBuffer* out_buf,
    const AcVisionTensorDesc* out_desc,
    AcEvent** out_event
) {
    if (!ctx) {
        if (out_event) *out_event = NULL;
        return AC_ERR_NULL_POINTER;
    }

    AcKernel* kernel = NULL;
    AcResult res = ac_vision_pointwise_conv_kernel_create(ctx, NULL, 0, NULL, &kernel);
    if (res != AC_SUCCESS) {
        if (out_event) *out_event = NULL;
        return res;
    }

    res = ac_vision_pointwise_conv_with_kernel_f32(
        ctx,
        kernel,
        params,
        in_buf,
        in_desc,
        weight_buf,
        weight_desc,
        bias_buf,
        out_buf,
        out_desc,
        out_event
    );

    /* If synchronous execution (out_event == NULL), destroy kernel immediately.
     * If async execution was requested, wait for event before destroying kernel to avoid lifetime hazard. */
    if (out_event && *out_event) {
        /* Note: For async caller wishing to retain kernel, caller should use ac_vision_pointwise_conv_with_kernel_f32 */
        ac_runtime_event_wait(*out_event, 10000000000ull); /* 10s safety timeout */
    }

    ac_vision_pointwise_conv_kernel_destroy(kernel);
    return res;
}
