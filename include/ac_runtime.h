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
    AcBackendKind backend_kind; /* AC_BACKEND_VULKAN or AC_BACKEND_OPENCL */
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
 * A recording batch is aborted. A submitted batch and the events obtained from it share one submission: the
 * fence and command buffer are released with the last of them, after the GPU finished, in any destruction order.
 *
 * @param batch Batch handle to destroy.
 */
void ac_runtime_batch_destroy(
    AcCommandBatch* batch
);

/* ========================================================================= */
/* Graph Batches (AMUDA 2.2 additions; descriptor layouts of 2.1 unchanged)  */
/* ========================================================================= */

/** Upper bound for AcBatchDesc.max_dispatches. */
#define AC_BATCH_MAX_DISPATCHES 65536u
/** No automatic hazard analysis: the caller inserts ac_runtime_batch_barrier() between dependent dispatches. */
#define AC_BATCH_FLAG_EXPLICIT_BARRIERS (1u << 0)
/** Record GPU timestamps around the batch; read them from the completion event (ac_runtime_event_get_timing). */
#define AC_BATCH_FLAG_TIMESTAMPS        (1u << 1)
/** Also record a GPU timestamp after every dispatch (profiling); read with ac_runtime_batch_get_dispatch_times.
 *  Begin fails with AC_ERR_FEATURE_NOT_SUPPORTED when the queue has no timestamps. */
#define AC_BATCH_FLAG_DISPATCH_TIMESTAMPS (1u << 2)

typedef struct {
    uint32_t abi_version;    /* Must match AC_ABI_VERSION */
    uint32_t struct_size;    /* Must match sizeof(AcBatchDesc) */
    uint32_t max_dispatches; /* 1 .. AC_BATCH_MAX_DISPATCHES */
    uint32_t flags;          /* AC_BATCH_FLAG_* (unknown bits rejected) */
} AcBatchDesc;

/**
 * @brief One buffer range bound to one kernel binding.
 * offset: multiple of AcDeviceLimits.min_storage_buffer_offset_alignment.
 * range:  > 0, offset + range <= buffer size, range <= AcDeviceLimits.max_storage_buffer_range.
 */
typedef struct {
    AcBuffer* buffer;
    uint64_t offset;
    uint64_t range;
} AcBufferBinding;

typedef struct {
    uint32_t abi_version;     /* Set by the caller: AC_ABI_VERSION */
    uint32_t struct_size;     /* Set by the caller: sizeof(AcBatchStats) */
    uint32_t dispatches;
    uint32_t barriers;        /* barriers recorded (explicit, or inserted by the automatic analysis) */
    uint32_t elided_barriers; /* automatic analysis only: dispatches recorded without a barrier */
    uint32_t flushes;         /* intermediate submissions made by ac_runtime_batch_flush */
    uint64_t descriptor_cache_hits;
    uint64_t descriptor_cache_misses;
} AcBatchStats;

typedef struct {
    uint32_t abi_version;     /* Set by the caller: AC_ABI_VERSION */
    uint32_t struct_size;     /* Set by the caller: sizeof(AcDeviceLimits) */
    uint64_t max_storage_buffer_range;
    uint64_t min_storage_buffer_offset_alignment;
    uint64_t max_memory_allocation_size; /* 0: not reported by the driver */
    uint32_t max_compute_workgroup_count[3];
    uint32_t max_compute_workgroup_size[3];
    uint32_t max_compute_workgroup_invocations;
    uint32_t max_compute_shared_memory_size;
    uint32_t max_push_constants_size;
    uint32_t subgroup_size;              /* 0: not reported by the driver */
    uint32_t timestamps_supported;
    float timestamp_period_ns;
    /* Appended fields: filled only when struct_size covers them (a caller built with an older header passes the
       smaller size and receives the fields up to timestamp_period_ns). */
    uint32_t integer_dot_product;             /* 1: shaders may use packed int8 dot products
                                                 (VK_KHR_shader_integer_dot_product enabled) */
    uint32_t integer_dot_product_accelerated; /* 1: the driver reports the signed 4x8 packed form as accelerated */
} AcDeviceLimits;

/**
 * @brief Merge saved pipeline cache data into the context's pipeline cache, which every kernel creation uses (a
 * kernel whose pipeline is in the cache is created without compiling its shader again).
 * @param data Data from ac_runtime_pipeline_cache_get_data of an earlier context on the same device and driver.
 * @return AC_SUCCESS; AC_ERR_INVALID_ARGUMENT when the data names another device or driver version (Vulkan header:
 *         vendor ID, device ID, pipelineCacheUUID) or is malformed (the cache is left as it was).
 */
AcResult ac_runtime_pipeline_cache_merge(AcContext* ctx, const void* data, size_t size);

/**
 * @brief Copy the context's pipeline cache data (to be saved and merged into a later context).
 * @param data NULL: *inout_size receives the size. Otherwise up to *inout_size bytes are written and *inout_size is
 *             set to the bytes written; AC_ERR_INVALID_BUFFER_SIZE when the buffer cannot hold the whole cache.
 */
AcResult ac_runtime_pipeline_cache_get_data(AcContext* ctx, void* data, size_t* inout_size);

/**
 * @brief Report the device limits that govern buffer binding and dispatch geometry.
 */
AcResult ac_runtime_context_get_limits(
    const AcContext* ctx,
    AcDeviceLimits* out_limits
);

/**
 * @brief Begin a batch with an explicit capacity and mode (see AC_BATCH_FLAG_*).
 */
AcResult ac_runtime_batch_begin_ex(
    AcContext* context,
    const AcBatchDesc* desc,
    AcCommandBatch** out_batch
);

/**
 * @brief Record a dispatch that binds buffer ranges instead of whole buffers.
 * binding_count must equal the kernel's binding count; descriptor sets are cached per context, so a range
 * combination recorded again (in this or a later batch) costs no descriptor allocation.
 */
AcResult ac_runtime_batch_dispatch_bindings(
    AcCommandBatch* batch,
    AcKernel* kernel,
    const AcBufferBinding* bindings,
    uint32_t binding_count,
    const void* push_constants,
    size_t push_constants_size,
    AcDim3 grid_dim
);

/**
 * @brief Record a full compute-to-compute memory barrier.
 */
AcResult ac_runtime_batch_barrier(
    AcCommandBatch* batch
);

/**
 * @brief Submit the dispatches recorded so far and keep recording into the same batch.
 * Bounds the GPU time of each submission (mobile drivers reset the GPU when a single submission runs for seconds) and
 * lets the GPU start while the rest is recorded. Order and visibility are those of one submission: barriers recorded
 * later cover the flushed commands, and the completion event of ac_runtime_batch_submit signals after all of them.
 * A no-op when nothing was recorded since the previous flush. On failure the batch is aborted (destroy it).
 */
AcResult ac_runtime_batch_flush(
    AcCommandBatch* batch
);

/**
 * @brief GPU time of every dispatch of a completed batch begun with AC_BATCH_FLAG_DISPATCH_TIMESTAMPS.
 * out_ns[i] is the time from the end of dispatch i-1 (or the batch start) to the end of dispatch i, in ns; dispatches
 * recorded without a barrier in between may overlap, and the overlap is attributed to the later one.
 * Requires the completion event to have been waited on (AC_ERR_DEVICE_BUSY otherwise) and capacity >= dispatches.
 */
AcResult ac_runtime_batch_get_dispatch_times(
    const AcCommandBatch* batch,
    uint64_t* out_ns,
    uint32_t capacity,
    uint32_t* out_count
);

/**
 * @brief Recording statistics of a batch (valid in any state).
 */
AcResult ac_runtime_batch_get_stats(
    const AcCommandBatch* batch,
    AcBatchStats* out_stats
);

/* ========================================================================= */
/* AMUDA 2.3 additions: OpenCL backend and backend-neutral host access       */
/* (descriptor layouts of 2.1 unchanged)                                     */
/* ========================================================================= */
/*
 * AC_BACKEND_OPENCL (docs/design/opencl-prompt-path.md, stage B):
 *   - The vendor OpenCL driver is loaded at context creation (AMEVA_OPENCL_LIBRARY=<path> names it); a missing driver
 *     fails with AC_ERR_DRIVER_NOT_FOUND / AC_ERR_DRIVER_SYMBOL_MISSING.
 *   - Buffers are coarse-grained SVM: the host reaches them only with ac_runtime_buffer_write / read / fill.
 *     ac_runtime_buffer_get_mapped_ptr, _flush, _invalidate and _is_coherent return AC_ERR_BUFFER_NOT_MAPPED.
 *     ac_runtime_buffer_get_address gives a stable address for offset arithmetic, not for host access.
 *   - Kernels come from OpenCL C (ac_runtime_kernel_create_from_source). Argument i < binding_count is the pointer of
 *     binding i (the bound range starts there); with push_constant_size > 0 the next argument is a struct of exactly
 *     that size passed by value (shorter push data is zero-padded). With AC_KERNEL_SOURCE_BINDING_RANGES the kernel
 *     declares one more argument after these, a struct of binding_count uint: it receives the byte size of every
 *     bound range (what a GLSL shader reads with buffer.length()). The argument count is checked at creation.
 *   - Batches: nothing runs before ac_runtime_batch_flush / _submit; the dispatches of a context run in recording
 *     order on one in-order queue, so every dispatch sees the writes of the earlier ones (barriers are implied).
 *   - Host access APIs are ordered after every dispatch already flushed or submitted on the context, and before the
 *     dispatches of a batch that is still recording.
 */

/**
 * @brief OpenCL C kernel creation descriptor (AC_BACKEND_OPENCL contexts).
 * metadata: as for SPIR-V kernels; entry_point is the __kernel function, expected_spirv_sha256 is not used.
 */
/** The kernel takes, after the bindings and the push-constant struct, the struct of bound range sizes (needs
 *  binding_count > 0). */
#define AC_KERNEL_SOURCE_BINDING_RANGES (1u << 0)

typedef struct {
    uint32_t abi_version;                    /* Must match AC_ABI_VERSION */
    uint32_t struct_size;                    /* Must match sizeof(AcKernelSourceDesc) */
    const char* source;                      /* OpenCL C source */
    size_t source_size;                      /* bytes of source; 0: NUL-terminated */
    const char* build_options;               /* NULL or options for the OpenCL C compiler */
    const AcKernelRuntimeMetadata* metadata;
    uint32_t flags;                          /* AC_KERNEL_SOURCE_* (unknown bits rejected) */
} AcKernelSourceDesc;

/** @brief The backend the context was created for. */
AcResult ac_runtime_context_get_backend_kind(const AcContext* ctx, AcBackendKind* out_kind);

/**
 * @brief Create a kernel from OpenCL C source. AC_ERR_FEATURE_NOT_SUPPORTED on a Vulkan context (and
 * ac_runtime_kernel_create, which takes SPIR-V for Vulkan, returns it on an OpenCL context).
 */
AcResult ac_runtime_kernel_create_from_source(AcContext* ctx, const AcKernelSourceDesc* desc, AcKernel** out_kernel);

/**
 * @brief Copy size_bytes from host memory into the buffer at offset; complete on return.
 * Vulkan: copy through the mapping and flush the range. OpenCL: a blocking copy on the context's queue.
 */
AcResult ac_runtime_buffer_write(AcBuffer* buf, size_t offset, const void* src, size_t size_bytes);

/** @brief Copy size_bytes of the buffer at offset into host memory; complete on return (Vulkan: invalidate first). */
AcResult ac_runtime_buffer_read(AcBuffer* buf, size_t offset, void* dst, size_t size_bytes);

/** @brief Set size_bytes of the buffer at offset to value; complete on return. */
AcResult ac_runtime_buffer_fill(AcBuffer* buf, size_t offset, size_t size_bytes, uint8_t value);

/**
 * @brief A stable address of the buffer's first byte, for offset arithmetic (Vulkan: the mapping; OpenCL: the SVM
 * pointer, which the host must not dereference).
 */
AcResult ac_runtime_buffer_get_address(const AcBuffer* buf, void** out_address);

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


