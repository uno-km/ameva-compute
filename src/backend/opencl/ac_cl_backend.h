/**
 * @file ac_cl_backend.h
 * @brief OpenCL backend of ac_runtime (AC_BACKEND_OPENCL), internal interface used by src/runtime/ac_runtime.cpp.
 *
 * Contract (docs/design/opencl-prompt-path.md, stage B):
 *   - The vendor OpenCL driver is opened with dlopen at context creation; libac_runtime.so.2 does not link OpenCL.
 *     A missing driver or symbol fails the context creation with the libraries tried (no other backend is chosen).
 *   - Buffers are coarse-grained SVM allocations. The host reaches them only through ac_cl_buffer_write / read / fill
 *     (the SVM pointer is not host-accessible without a map); ac_cl_buffer_address is for offset arithmetic only.
 *   - Kernels are OpenCL C sources. Argument i < binding_count is the global pointer of binding i (buffer address +
 *     binding offset); when the metadata declares push constants, the next argument is a struct of exactly that size
 *     passed by value (shorter push data is zero-padded). A kernel created with takes_ranges has one more argument
 *     after these: the struct of binding_count uint that receives the byte size of every bound range (clamped to
 *     2^32 - 1). The kernel's argument count is checked at creation.
 *   - Batches keep Vulkan's semantics: nothing is enqueued before ac_cl_batch_flush / ac_cl_batch_submit, and the
 *     dispatches of a batch run in recording order on one in-order queue (barriers are implied by that order).
 */
#ifndef AC_CL_BACKEND_H
#define AC_CL_BACKEND_H

#include "ac_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AcClContext AcClContext;
typedef struct AcClBuffer AcClBuffer;
typedef struct AcClKernel AcClKernel;
typedef struct AcClEvent AcClEvent;
typedef struct AcClBatch AcClBatch;

AcResult ac_cl_context_create(AcClContext** out_ctx);
AcResult ac_cl_context_destroy(AcClContext* ctx);
AcResult ac_cl_context_get_device_name(const AcClContext* ctx, char* out_name, size_t max_len);
/* fills the fields of out_limits that its struct_size covers (validated by the caller) */
AcResult ac_cl_context_get_limits(const AcClContext* ctx, AcDeviceLimits* out_limits);

AcResult ac_cl_buffer_create(AcClContext* ctx, size_t size_bytes, AcClBuffer** out_buf);
AcResult ac_cl_buffer_destroy(AcClBuffer* buf);
size_t ac_cl_buffer_size(const AcClBuffer* buf);
void* ac_cl_buffer_address(const AcClBuffer* buf);
AcResult ac_cl_buffer_write(AcClBuffer* buf, size_t offset, const void* src, size_t size_bytes);
AcResult ac_cl_buffer_read(AcClBuffer* buf, size_t offset, void* dst, size_t size_bytes);
AcResult ac_cl_buffer_fill(AcClBuffer* buf, size_t offset, size_t size_bytes, uint8_t value);

AcResult ac_cl_kernel_create(AcClContext* ctx, const char* source, size_t source_size, const char* build_options,
                             const AcKernelRuntimeMetadata* metadata, bool takes_ranges, AcClKernel** out_kernel);
AcResult ac_cl_kernel_destroy(AcClKernel* kernel);
AcDim3 ac_cl_kernel_local_size(const AcClKernel* kernel);
size_t ac_cl_kernel_push_constant_size(const AcClKernel* kernel);
/* sha256 hex of (entry point, build options, source) with the terminating NUL: 65 bytes */
const char* ac_cl_kernel_identity(const AcClKernel* kernel);

/* flags: AC_BATCH_FLAG_* (validated by the caller) */
AcResult ac_cl_batch_begin(AcClContext* ctx, uint32_t max_dispatches, uint32_t flags, AcClBatch** out_batch);
AcResult ac_cl_batch_dispatch(AcClBatch* batch, AcClKernel* kernel, AcClBuffer* const* buffers, const uint64_t* offsets,
                              const uint64_t* ranges, uint32_t binding_count, const void* push_constants,
                              size_t push_constants_size, AcDim3 grid_dim);
AcResult ac_cl_batch_barrier(AcClBatch* batch);
AcResult ac_cl_batch_flush(AcClBatch* batch);
AcResult ac_cl_batch_submit(AcClBatch* batch, AcClEvent** out_completion_event);
AcResult ac_cl_batch_abort(AcClBatch* batch);
void ac_cl_batch_destroy(AcClBatch* batch);
AcResult ac_cl_batch_get_dispatch_times(const AcClBatch* batch, uint64_t* out_ns, uint32_t capacity, uint32_t* out_count);
void ac_cl_batch_get_stats(const AcClBatch* batch, AcBatchStats* out_stats);

AcResult ac_cl_event_wait(AcClEvent* event, uint64_t timeout_ns);
AcResult ac_cl_event_get_timing(AcClEvent* event, uint64_t* out_wall_latency_ns, uint64_t* out_gpu_duration_ns);
AcResult ac_cl_event_destroy(AcClEvent* event);

AcResult ac_cl_program_cache_merge(AcClContext* ctx, const void* data, size_t size);
AcResult ac_cl_program_cache_get_data(AcClContext* ctx, void* data, size_t* inout_size);

#ifdef __cplusplus
}
#endif

#endif /* AC_CL_BACKEND_H */
