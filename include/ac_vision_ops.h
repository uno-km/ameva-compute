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

/**
 * @brief Execute 1x1 Pointwise Convolution on Device-Resident buffers.
 *
 * Core execution engine operating directly on caller-managed device memory handles.
 * Pure device-resident: zero CPU staging, zero readbacks.
 */
AcResult ac_vision_pointwise_conv_device_f32(
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
 * @brief Create a Fused Pointwise Conv2D 1x1 + ReLU Compute Kernel.
 */
AcResult ac_vision_pointwise_conv_relu_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

/**
 * @brief Execute Fused 1x1 Pointwise Convolution + ReLU with explicit kernel on Device-Resident buffers.
 */
AcResult ac_vision_pointwise_conv_relu_with_kernel_f32(
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
 * @brief Execute Fused 1x1 Pointwise Convolution + ReLU with automatic kernel lifecycle.
 */
AcResult ac_vision_pointwise_conv_relu_f32(
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

/**
 * @brief Create an Elementwise ReLU Compute Kernel.
 */
AcResult ac_vision_relu_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

AcResult ac_vision_relu_kernel_destroy(
    AcKernel* kernel
);

/**
 * @brief Execute Elementwise ReLU on Device-Resident buffer.
 */
AcResult ac_vision_relu_device_f32(
    AcContext* ctx,
    AcKernel* kernel,
    AcBuffer* in_buf,
    AcBuffer* out_buf,
    size_t num_elements,
    AcEvent** out_event
);

/* =============================================================================
 * Context-Scoped Weight Cache (Strict Non-Global Architecture)
 * ============================================================================= */

/**
 * @brief Weight Cache Key
 */
typedef struct {
    uint8_t sha256[32];
    uint64_t byte_size;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    AcVisionDtype dtype;
} AcVisionWeightKey;

typedef struct AcVisionWeightCache AcVisionWeightCache;

AcResult ac_vision_weight_cache_create(
    AcContext* ctx,
    AcVisionWeightCache** out_cache
);

AcResult ac_vision_weight_cache_get_or_upload(
    AcVisionWeightCache* cache,
    const AcVisionWeightKey* key,
    const void* host_data,
    size_t size_bytes,
    AcBuffer** out_buffer,
    bool* out_cache_hit
);

uint32_t ac_vision_weight_cache_entry_count(
    const AcVisionWeightCache* cache
);

AcResult ac_vision_weight_cache_destroy(
    AcVisionWeightCache* cache
);

/* =============================================================================
 * Conv3x3 + ReLU Direct Convolution Operations
 * ============================================================================= */

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t has_bias;
    uint32_t kernel_h;  /* 3 */
    uint32_t kernel_w;  /* 3 */
    uint32_t stride_h;  /* 1 */
    uint32_t stride_w;  /* 1 */
    uint32_t pad_h;     /* 1 */
    uint32_t pad_w;     /* 1 */
} AcVisionConv3x3Params;

AcResult ac_vision_conv3x3_relu_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

AcResult ac_vision_conv3x3_relu_kernel_destroy(
    AcKernel* kernel
);

AcResult ac_vision_conv3x3_relu_device_f32(
    AcContext* ctx,
    AcKernel* kernel,
    const AcVisionConv3x3Params* params,
    AcBuffer* in_buf,
    AcBuffer* weight_buf,
    AcBuffer* bias_buf,
    AcBuffer* out_buf,
    AcEvent** out_event
);

/* =============================================================================
 * GPU Concat2 Operations (Along Channel Dimension)
 * ============================================================================= */

AcResult ac_vision_concat2_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

AcResult ac_vision_concat2_kernel_destroy(
    AcKernel* kernel
);

AcResult ac_vision_concat2_device_f32(
    AcContext* ctx,
    AcKernel* kernel,
    AcBuffer* branch0_buf,
    uint32_t c0,
    AcBuffer* branch1_buf,
    uint32_t c1,
    uint32_t spatial_elements, /* H * W */
    AcBuffer* out_buf,
    AcEvent** out_event
);

/* =============================================================================
 * GPU MaxPool2d Operations (NHWC Layout)
 * ============================================================================= */

#ifdef __cplusplus
enum class AcVisionPoolPadMode : uint32_t {
    FULL = 0,
    VALID = 1,
    SAME_UPPER = 2,
    SAME_LOWER = 3,
};
#else
typedef enum {
    AC_VISION_POOL_PAD_FULL = 0,
    AC_VISION_POOL_PAD_VALID = 1,
    AC_VISION_POOL_PAD_SAME_UPPER = 2,
    AC_VISION_POOL_PAD_SAME_LOWER = 3,
} AcVisionPoolPadMode;
#endif

typedef struct {
    uint32_t in_h;
    uint32_t in_w;
    uint32_t channels;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t pad_mode;
} AcVisionMaxPoolPushConstants;

#if defined(__cplusplus)
static_assert(sizeof(AcVisionMaxPoolPushConstants) == 56, "AcVisionMaxPoolPushConstants must be exactly 56 bytes");
#endif

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t channels;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t pad_mode;
} AcVisionMaxPool2dParams;

AcResult ac_vision_maxpool2d_kernel_create(
    AcContext* ctx,
    const uint32_t* spirv_code,
    size_t code_size_bytes,
    const char* expected_sha256,
    AcKernel** out_kernel
);

AcResult ac_vision_maxpool2d_kernel_destroy(
    AcKernel* kernel
);

AcResult ac_vision_maxpool2d_device_f32(
    AcContext* ctx,
    AcKernel* kernel,
    const AcVisionMaxPool2dParams* params,
    AcBuffer* in_buf,
    AcBuffer* out_buf,
    AcEvent** out_event
);

#ifdef __cplusplus
AcResult ac_vision_checked_pool_output_size(
    uint32_t input,
    uint32_t pad_before,
    uint32_t pad_after,
    uint32_t kernel,
    uint32_t stride,
    uint32_t pad_mode,
    uint32_t* output,
    uint32_t* out_tail_pad = nullptr,
    uint32_t* out_tail_rem = nullptr
);
#else
AcResult ac_vision_checked_pool_output_size(
    uint32_t input,
    uint32_t pad_before,
    uint32_t pad_after,
    uint32_t kernel,
    uint32_t stride,
    uint32_t pad_mode,
    uint32_t* output,
    uint32_t* out_tail_pad,
    uint32_t* out_tail_rem
);
#endif


/* =============================================================================
 * SqueezeNet Fire2 Subgraph Operations (Device-Resident & Host Convenience)
 * ============================================================================= */

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t H;                 /* 56 */
    uint32_t W;                 /* 56 */
    uint32_t C_in;              /* 64 */
    uint32_t C_squeeze;         /* 16 */
    uint32_t C_expand1x1;       /* 64 */
    uint32_t C_expand3x3;       /* 64 */
} AcVisionFire2Params;

typedef struct {
    uint64_t context_init_time_us;
    uint64_t kernel_create_time_us;
    uint64_t buffer_alloc_time_us;
    uint64_t buffer_map_time_us;
    uint64_t weight_hash_time_us;
    uint64_t weight_cache_lookup_time_us;
    uint64_t weight_upload_time_us;
    uint64_t input_upload_time_us;
    uint64_t each_dispatch_submit_time_us;
    uint64_t each_event_wait_time_us;
    uint64_t dispatch_call_wall_time_us;
    uint64_t post_dispatch_event_wait_us;
    uint64_t output_invalidate_time_us;
    uint64_t output_copy_time_us;
    uint64_t buffer_destroy_time_us;

    uint64_t host_to_device_time_us;
    uint64_t command_record_time_ns;        /* Legacy alias for launch_desc_build_time_ns */
    uint64_t launch_desc_build_time_ns;     /* CPU time to populate AcLaunchDesc */
    double command_record_time_us;
    uint64_t dispatch_to_completion_time_us; /* KERNEL_ONLY */
    uint64_t device_to_host_time_us;
    uint64_t total_measured_components_us;
    uint64_t total_offload_time_us;          /* END_TO_END_OFFLOAD */
    uint64_t unaccounted_time_us;
    double unaccounted_ratio;
    bool timing_reconciled;

    uint32_t subgraph_input_upload_count;    /* 1 */
    uint32_t internal_host_upload_count;     /* 0 */
    uint32_t internal_host_readback_count;   /* 0 */
    uint32_t subgraph_output_readback_count; /* 1 */
    uint32_t device_resident_layer_count;    /* 4 */
    uint32_t later_dispatches_executed_after_failure; /* 0 on error */

    /* MaxPool Effective Padding & Mode Evidence */
    uint32_t effective_pad_right;
    uint32_t effective_pad_bottom;
    uint32_t pool_pad_mode;
} AcVisionFire2ExecutionProfile;

typedef AcVisionFire2Params AcVisionFire3Params;
typedef AcVisionFire2Params AcVisionFire4Params;
typedef AcVisionFire2Params AcVisionFire5Params;
typedef AcVisionFire2Params AcVisionFire6Params;
typedef AcVisionFire2Params AcVisionFire7Params;
typedef AcVisionFire2Params AcVisionFire8Params;
typedef AcVisionFire2ExecutionProfile AcVisionChainExecutionProfile;

/* =============================================================================
 * Failure Stage Reached Evidence & Lifecycle Tracking
 * ============================================================================= */

typedef struct {
    uint32_t squeeze_fail_context_created;
    uint32_t squeeze_fail_kernel_created;
    uint32_t weight_fail_buffer_created;
    uint32_t weight_fail_buffer_destroyed;
    uint32_t dispatch_fail_event_created;
    uint32_t wait_fail_event_created;
    uint32_t wait_fail_event_destroyed;
    uint32_t invalidate_fail_dispatch_succeeded;
    uint32_t invalidate_fail_wait_succeeded;
} AcVisionFailureStageEvidence;

void ac_vision_reset_failure_stage_evidence(void);
const AcVisionFailureStageEvidence* ac_vision_get_failure_stage_evidence(void);

/**
 * @brief Pure Device Execution Core for SqueezeNet Fire2 Subgraph
 *
 * Operates strictly on pre-allocated device resident memory handles.
 * ZERO host staging, ZERO host pointer access, ZERO intermediate readbacks.
 */
AcResult ac_vision_fire2_subgraph_device_f32(
    AcContext* ctx,
    AcKernel* kernel_pointwise_relu,
    AcKernel* kernel_conv3x3_relu,
    AcKernel* kernel_concat2,
    const AcVisionFire2Params* params,
    AcBuffer* buf_in_pool1,           /* Device resident input */
    AcBuffer* buf_weight_sq,
    AcBuffer* buf_bias_sq,
    AcBuffer* buf_weight_ex1,
    AcBuffer* buf_bias_ex1,
    AcBuffer* buf_weight_ex3,
    AcBuffer* buf_bias_ex3,
    AcBuffer* buf_inter_sq_relu,      /* Device resident intermediate */
    AcBuffer* buf_inter_ex1_relu,     /* Device resident intermediate */
    AcBuffer* buf_inter_ex3_relu,     /* Device resident intermediate */
    AcBuffer* buf_out_concat,         /* Device resident final output */
    AcVisionFire2ExecutionProfile* profile
);

/**
 * @brief Host Convenience Wrapper for SqueezeNet Fire2 Subgraph
 *
 * Performs exactly 1 host-to-device upload, delegates to pure device core,
 * and performs exactly 1 device-to-host readback.
 */
AcResult ac_vision_fire2_subgraph_f32(
    AcContext* ctx,
    AcVisionWeightCache* weight_cache,
    const AcVisionFire2Params* params,
    const float* host_in_pool1,
    const float* host_weight_sq,
    const float* host_bias_sq,
    const float* host_weight_ex1,
    const float* host_bias_ex1,
    const float* host_weight_ex3,
    const float* host_bias_ex3,
    float* host_out_concat,
    AcVisionFire2ExecutionProfile* profile
);

#ifdef __cplusplus
}
#endif

#endif /* AC_VISION_OPS_H */
