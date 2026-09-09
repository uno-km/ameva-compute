/**
 * @file ac_dispatch.h
 * @brief Explicit Vulkan Compute Kernel Dispatch
 *
 * Contract:
 *   - Context Mismatch: Returns AC_ERR_CONTEXT_MISMATCH if kernel or any buffer
 *     belongs to a different context than ctx.
 *   - Pre-Submit Allocation: All host-side tracking and descriptor allocations complete
 *     before queue submission.
 *   - Fail-Safe Rollback: If vkQueueSubmit fails, all submission_outstanding_count
 *     counters rollback to pre-submission values immediately.
 *   - Output Safety: out_event is set to NULL on entry and only assigned upon full success.
 *   - Workgroup Validation: Rejects group dimensions of 0 or multiplication overflow.
 */
#ifndef AC_DISPATCH_H
#define AC_DISPATCH_H

#include "ac_base.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"

#ifdef __cplusplus
extern "C" {
#endif

AcResult ac_vk_dispatch(
    AcVkContext* ctx,
    AcVkKernel* kernel,
    AcVkBuffer** buffers,
    uint32_t buffer_count,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    AcVkEvent** out_event
);

#ifdef __cplusplus
}
#endif

#endif /* AC_DISPATCH_H */
