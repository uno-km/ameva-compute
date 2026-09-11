/**
 * @file ac_vision_ops.h
 * @brief AMUDA Vision Public Opaque C Runtime ABI
 *
 * Ground Truth Engineering Principles:
 *   - Strictly Opaque C ABI: Zero Vulkan types or headers exposed.
 *   - Layout Lowering: NHWC Input lowered to 2D row-major matrix,
 *     OHWI Weight mapped for GEMM multiplication.
 *   - Fail-Fast Verification: Rejects unsupported layouts, kernels != 1x1,
 *     strides != 1, batch > 1, invalid dtypes, and buffer overflows.
 *   - Zero Silent Fallback: No silent CPU routing or mock results.
 */

#ifndef AC_VISION_OPS_H
#define AC_VISION_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ac_base.h"
#include "ac_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AC_VISION_ABI_VERSION 0x00010000u

/**
 * @brief Vision Supported Data Types
 */
typedef enum {
    AC_VISION_DTYPE_F32 = 0,
    AC_VISION_DTYPE_UNKNOWN = 0xFFFFFFFFu
} AcVisionDtype;

/**
 * @brief Vision Tensor Memory Layouts
 */
typedef enum {
    AC_VISION_LAYOUT_NHWC = 0,
    AC_VISION_LAYOUT_NCHW = 1,
    AC_VISION_LAYOUT_OHWI = 2,
    AC_VISION_LAYOUT_UNKNOWN = 0xFFFFFFFFu
} AcVisionLayout;

/**
 * @brief Vision Tensor Descriptor
 */
typedef struct {
    uint32_t abi_version;       /* Must match AC_VISION_ABI_VERSION */
    uint32_t struct_size;       /* Must match sizeof(AcVisionTensorDesc) */
    AcVisionDtype dtype;        /* Must be AC_VISION_DTYPE_F32 */
    AcVisionLayout layout;      /* AC_VISION_LAYOUT_NHWC for activation, OHWI for weight */
    uint32_t n;                 /* Batch dimension (must be 1 for V1) */
    uint32_t h;                 /* Height dimension */
    uint32_t w;                 /* Width dimension */
    uint32_t c;                 /* Channel dimension */
    size_t row_stride_bytes;    /* 0 for default contiguous, or exact row stride */
} AcVisionTensorDesc;

/**
 * @brief Pointwise Conv2D 1x1 Parameters
 */
typedef struct {
    uint32_t abi_version;       /* Must match AC_VISION_ABI_VERSION */
    uint32_t struct_size;       /* Must match sizeof(AcVisionPointwiseConvParams) */
    uint32_t input_channels;    /* C_IN */
    uint32_t output_channels;   /* C_OUT */
    uint32_t input_height;      /* H */
    uint32_t input_width;       /* W */
    uint32_t has_bias;          /* 1 if bias buffer present, 0 otherwise */
    uint32_t kernel_h;          /* Must be 1 */
    uint32_t kernel_w;          /* Must be 1 */
    uint32_t stride_h;          /* Must be 1 */
    uint32_t stride_w;          /* Must be 1 */
    uint32_t dilation_h;        /* Must be 1 */
    uint32_t dilation_w;        /* Must be 1 */
} AcVisionPointwiseConvParams;

/**
 * @brief Create a Pointwise Conv2D 1x1 Compute Kernel.
 *
 * If spirv_code is NULL, the default canonical embedded SPIR-V shader is used.
 * Caller can pass custom/mutated SPIR-V for anti-fake sensitivity tests.
 *
 * @param ctx Context handle
 * @param spirv_code Optional custom SPIR-V bytecode words (or NULL for default)
 * @param code_size_bytes Byte size of custom SPIR-V code (or 0 for default)
 * @param expected_sha256 Optional expected SHA-256 hex string (or NULL)
 * @param out_kernel Output kernel handle (guaranteed NULL on failure)
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_vision_pointwise_conv_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

/**
 * @brief Destroy a Pointwise Conv2D 1x1 Compute Kernel.
 */
AcResult ac_vision_pointwise_conv_kernel_destroy(
    AcKernel* kernel
);

/**
 * @brief Execute 1x1 Pointwise Convolution lowered to Vulkan F32 SGEMM with explicit kernel.
 */
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
);

/**
 * @brief Execute 1x1 Pointwise Convolution lowered to Vulkan F32 SGEMM.
 *
 * Automatically manages kernel lifecycle if kernel is not pre-created.
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* AC_VISION_OPS_H */
