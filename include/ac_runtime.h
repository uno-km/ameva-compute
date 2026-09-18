/**
 * @file ac_runtime.h
 * @brief AMUDA 2.0 Public Opaque C Runtime ABI
 *
 * Ground Truth Engineering Principles:
 *   - Strictly Opaque Handles (AcContext, AcBuffer, AcKernel, AcEvent).
 *   - Completely Decoupled: Zero Vulkan types or <vulkan/vulkan.h> header exposure.
 *   - Standalone Buffer Usage and Stage Flags translated internally to backend primitives.
 *   - Fail-Fast Verification: Mandatory abi_version and struct_size validation on descriptors.
 *   - Output Safety Invariant: Every creation and output pointer is set to NULL on entry.
 *   - Active Child Tracking: Context destruction is rejected if child buffers, kernels,
 *     or events remain alive.
 *
 * ABI Contract Specification (AMUDA 2.1.0):
 *   In AC_ABI_VERSION 0x00020100, all descriptor struct_size fields require an
 *   exact size match (desc->struct_size == sizeof(DescriptorType)).
 *   Specialization constants are fully integrated into kernel metadata and descriptors.
 */

#ifndef AC_RUNTIME_H
#define AC_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ac_base.h"
#include "ac_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Opaque Handle Types (Zero Internal Field Exposure)                        */
/* ========================================================================= */

typedef struct AcContext AcContext;
typedef struct AcBuffer  AcBuffer;
typedef struct AcKernel  AcKernel;
typedef struct AcEvent   AcEvent;

/* ========================================================================= */
/* Public Flags & Enumerations                                               */
/* ========================================================================= */

/**
 * @brief Public Buffer Usage Flags (Zero Vulkan Exposure)
 */
typedef uint32_t AcBufferUsageFlags;
#define AC_BUFFER_USAGE_STORAGE      (1u << 0)
#define AC_BUFFER_USAGE_TRANSFER_SRC (1u << 1)
#define AC_BUFFER_USAGE_TRANSFER_DST (1u << 2)
#define AC_BUFFER_USAGE_UNIFORM      (1u << 3)

/**
 * @brief Public Shader Stage Flags
 */
typedef uint32_t AcShaderStage;
#define AC_STAGE_COMPUTE ((AcShaderStage)(1u << 5))

/**
 * @brief Public Binding Types
 */
typedef uint32_t AcRuntimeBindingType;
#define AC_RUNTIME_BINDING_STORAGE_BUFFER ((AcRuntimeBindingType)0u)
#define AC_RUNTIME_BINDING_UNIFORM_BUFFER ((AcRuntimeBindingType)1u)

/**
 * @brief Public Kernel Binding Descriptor
 */
typedef struct {
    uint32_t binding_index;
    AcRuntimeBindingType binding_type;
    AcShaderStage stage_flags;
} AcRuntimeBindingDesc;


/**
 * @brief Public Kernel Metadata Descriptor
 */
typedef struct {
    const char* kernel_name;
    const char* entry_point;
    uint32_t binding_count;
    const AcRuntimeBindingDesc* bindings;
    size_t push_constant_size;
    AcDim3 local_size;
    uint32_t required_features;
    const char* expected_spirv_sha256;
    uint32_t specialization_schema_count;
    const AcSpecConstantSchemaEntry* specialization_schemas;
} AcKernelRuntimeMetadata;

/* ========================================================================= */
/* Descriptors for Initialization and Launch                                 */
/* ========================================================================= */

/**
 * @brief Context Creation Descriptor
 */
typedef struct {
    uint32_t abi_version;       /* Must match AC_ABI_VERSION */
    uint32_t struct_size;       /* Must match sizeof(AcContextDesc) */
    AcBackendKind backend_kind; /* Must be AC_BACKEND_VULKAN */
    uint32_t flags;             /* Reserved for future expansion; must be 0 */
} AcContextDesc;

/**
 * @brief Buffer Creation Descriptor
 */
typedef struct {
    uint32_t abi_version;        /* Must match AC_ABI_VERSION */
    uint32_t struct_size;        /* Must match sizeof(AcBufferDesc) */
    size_t size_bytes;           /* Logical allocation size in bytes (> 0) */
    AcBufferUsageFlags usage;    /* AC_BUFFER_USAGE_* flags (unknown bits rejected) */
    uint32_t force_non_coherent; /* Explicit non-coherent preference: 0 (false) or 1 (true) */
} AcBufferDesc;

/**
 * @brief Kernel Creation Descriptor
 */
typedef struct {
    uint32_t abi_version;       /* Must match AC_ABI_VERSION */
    uint32_t struct_size;       /* Must match sizeof(AcKernelDesc) */
    const uint32_t* spirv_code; /* Pointer to 32-bit aligned SPIR-V bytecode words */
    size_t code_size_bytes;     /* Byte size of SPIR-V code (must be multiple of 4) */
    const AcKernelRuntimeMetadata* metadata;
    uint32_t specialization_count;
    const AcSpecConstant* specializations;
} AcKernelDesc;

/**
 * @brief Kernel Launch / Dispatch Descriptor
 */
typedef struct {
    uint32_t abi_version;       /* Must match AC_ABI_VERSION */
    uint32_t struct_size;       /* Must match sizeof(AcLaunchDesc) */
    AcKernel* kernel;           /* Kernel to execute (must match context) */
    AcBuffer** buffers;         /* Array of bound buffer handles (must match context) */
    uint32_t buffer_count;      /* Number of bound buffers */
    const void* push_constants; /* Optional push constant payload */
    size_t push_constants_size; /* Push constant byte size (<= 128) */
    AcDim3 grid_dim;            /* Dispatch workgroup dimensions (x, y, z > 0) */
} AcLaunchDesc;

/* ========================================================================= */
/* Context Management APIs                                                   */
/* ========================================================================= */

/**
 * @brief Create an isolated compute context.
 *
 * @param desc Pointer to AcContextDesc.
 * @param out_ctx Output context pointer (guaranteed NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_context_create(
    const AcContextDesc* desc,
    AcContext** out_ctx
);

/**
 * @brief Destroy a compute context.
 * Rejects destruction with AC_ERR_RESOURCE_IN_USE if any child buffer,
 * kernel, or unretired event remains alive.
 *
 * @param ctx Context to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_context_destroy(
    AcContext* ctx
);

/**
 * @brief Retrieve physical GPU device name.
 */
AcResult ac_runtime_context_get_device_name(
    const AcContext* ctx,
    char* out_name,
    size_t max_len
);

/* ========================================================================= */
/* Buffer Management APIs                                                    */
/* ========================================================================= */

/**
 * @brief Allocate a buffer with explicit usage and coherency controls.
 *
 * @param ctx Owning context.
 * @param desc Pointer to AcBufferDesc.
 * @param out_buf Output buffer pointer (guaranteed NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_buffer_create(
    AcContext* ctx,
    const AcBufferDesc* desc,
    AcBuffer** out_buf
);

/**
 * @brief Destroy a buffer, releasing underlying device allocation.
 *
 * @param buf Buffer to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_buffer_destroy(
    AcBuffer* buf
);

/**
 * @brief Retrieve persistently mapped host virtual address.
 */
AcResult ac_runtime_buffer_get_mapped_ptr(
    const AcBuffer* buf,
    void** out_ptr
);

/**
 * @brief Retrieve logical requested buffer byte size.
 */
AcResult ac_runtime_buffer_get_size(
    const AcBuffer* buf,
    size_t* out_size
);

/**
 * @brief Retrieve underlying physical driver allocation size (atom-aligned).
 */
AcResult ac_runtime_buffer_get_allocation_size(
    const AcBuffer* buf,
    size_t* out_alloc_size
);

/**
 * @brief Query whether buffer memory is host-coherent (1: coherent, 0: non-coherent).
 */
AcResult ac_runtime_buffer_is_coherent(
    const AcBuffer* buf,
    uint32_t* out_is_coherent
);

/**
 * @brief Explicitly flush mapped memory range for host writes.
 */
AcResult ac_runtime_buffer_flush(
    AcBuffer* buf,
    size_t offset,
    size_t size_bytes
);

/**
 * @brief Explicitly invalidate mapped memory range for device writes.
 */
AcResult ac_runtime_buffer_invalidate(
    AcBuffer* buf,
    size_t offset,
    size_t size_bytes
);

/* ========================================================================= */
/* Kernel Management APIs                                                    */
/* ========================================================================= */

/**
 * @brief Create a compute kernel from SPIR-V bytecode with metadata authentication.
 *
 * @param ctx Owning context.
 * @param desc Pointer to AcKernelDesc.
 * @param out_kernel Output kernel pointer (guaranteed NULL on failure).
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_kernel_create(
    AcContext* ctx,
    const AcKernelDesc* desc,
    AcKernel** out_kernel
);

/**
 * @brief Destroy a compute kernel.
 *
 * @param kernel Kernel to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_kernel_destroy(
    AcKernel* kernel
);

#define AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE 65u

/**
 * @brief Retrieve deterministic canonical pipeline identity SHA-256 hex string.
 *
 * @param kernel Kernel handle.
 * @param out_hex Destination buffer (must be at least 65 bytes).
 * @param out_hex_size Size of out_hex buffer.
 * @return AC_SUCCESS on success, AC_ERR_INVALID_ARGUMENT if out_hex_size < 65, error code on failure.
 */
AcResult ac_runtime_kernel_get_pipeline_identity(
    const AcKernel* kernel,
    char* out_hex,
    size_t out_hex_size
);

/**
 * @brief Retrieve effective workgroup local size after specialization overrides.
 *
 * @param kernel Kernel handle.
 * @param out_local_size Output pointer for effective local dimensions.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_kernel_get_effective_local_size(
    const AcKernel* kernel,
    AcDim3* out_local_size
);

/**
 * @brief Retrieve push constant size of a compiled kernel.
 *
 * @param kernel Kernel handle.
 * @param out_size Output pointer for push constant size in bytes.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_kernel_get_push_constant_size(
    const AcKernel* kernel,
    size_t* out_size
);

/* ========================================================================= */
/* Execution & Synchronization APIs                                          */
/* ========================================================================= */

/**
 * @brief Dispatch a compute kernel over a 3D grid.
 * If out_event is NULL, the call blocks synchronously until GPU execution and event destruction complete.
 * If out_event is non-NULL, returns an asynchronous event handle tracked by context.
 *
 * @param ctx Owning context.
 * @param desc Pointer to AcLaunchDesc.
 * @param out_event Optional output event pointer for async synchronization.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_dispatch(
    AcContext* ctx,
    const AcLaunchDesc* desc,
    AcEvent** out_event
);

/**
 * @brief Wait for event completion (Idempotent: double wait returns AC_SUCCESS).
 *
 * @param event Event to wait on.
 * @param timeout_ns Timeout in nanoseconds.
 * @return AC_SUCCESS on completion, AC_ERR_LAUNCH_TIMEOUT on timeout, error code on failure.
 */
AcResult ac_runtime_event_wait(
    AcEvent* event,
    uint64_t timeout_ns
);

/**
 * @brief Retrieve execution latency metrics for a signaled event.
 */
AcResult ac_runtime_event_get_timing(
    AcEvent* event,
    uint64_t* out_wall_latency_ns,
    uint64_t* out_gpu_duration_ns
);

/**
 * @brief Destroy an event handle and retire tracked child references.
 *
 * @param event Event to destroy.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_event_destroy(
    AcEvent* event
);

/* ========================================================================= */
/* Command Batching APIs (AMUDA 2.1.0 Backend-Neutral Batch Architecture)    */
/* ========================================================================= */

typedef struct AcCommandBatch AcCommandBatch;

/**
 * @brief Begin a new command batch for recording multiple compute dispatches.
 *
 * @param context Owning context.
 * @param out_batch Output pointer to created command batch.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_batch_begin(
    AcContext* context,
    AcCommandBatch** out_batch
);

/**
 * @brief Record a compute kernel dispatch into an open command batch.
 * Retains all buffer and kernel references until batch completion.
 *
 * @param batch Open command batch in recording state.
 * @param kernel Kernel to dispatch.
 * @param launch Launch descriptor.
 * @param out_event Optional logical event pointer associated with this dispatch.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_batch_dispatch(
    AcCommandBatch* batch,
    AcKernel* kernel,
    const AcLaunchDesc* launch,
    AcEvent** out_event
);

/**
 * @brief Submit all recorded dispatches in the batch to the hardware compute queue.
 *
 * @param batch Recorded command batch.
 * @param out_completion_event Optional output event signaling completion of the entire batch.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_batch_submit(
    AcCommandBatch* batch,
    AcEvent** out_completion_event
);

/**
 * @brief Abort an open or unsubmitted batch, releasing all retained resources.
 *
 * @param batch Batch to abort.
 * @return AC_SUCCESS on success, error code on failure.
 */
AcResult ac_runtime_batch_abort(
    AcCommandBatch* batch
);

/**
 * @brief Destroy a command batch handle.
 *
 * @param batch Batch handle to destroy.
 */
void ac_runtime_batch_destroy(
    AcCommandBatch* batch
);

/* ========================================================================= */
/* Error State Management APIs (AMUDA Generation-Based Error Tracking)       */
/* ========================================================================= */

/**
 * @brief Record a backend error into the active generation error state.
 * Implements first-error-wins semantics within the current generation.
 */
void ac_runtime_set_error(
    int32_t error_code,
    AmevaBackendOpKind op,
    size_t offset,
    size_t requested_bytes,
    size_t capacity_bytes
);

/**
 * @brief Retrieve the current generation error state snapshot.
 */
AmevaBackendErrorState ac_runtime_get_error_state(void);

/**
 * @brief Consume and return the last error, clearing first_error if generation matches.
 */
int32_t ac_runtime_consume_error(void);

/**
 * @brief Advance the error tracking generation epoch, resetting first_error.
 */
uint64_t ac_runtime_advance_generation(void);

/**
 * @brief Get current generation epoch.
 */
uint64_t ac_runtime_get_generation(void);

/**
 * @brief Explicitly reset error state.
 */
void ac_runtime_reset_error_state(void);

#ifdef __cplusplus
}
#endif

#endif /* AC_RUNTIME_H */


