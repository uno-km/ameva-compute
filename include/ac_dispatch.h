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

/**
 * @brief Maximum safe compute workgroups per axis (X, Y, or Z).
 * Default: 1,048,576 (2^20).
 *
 * Rationale:
 * Mobile GPU architectures (e.g. ARM Mali-G68) report theoretical hardware maximums
 * up to UINT32_MAX (4,294,967,295). Submitting unconstrained workloads to physical
 * queues leads to GPU driver lockups, kernel TDRs, and process hangs.
 * Setting a defensive per-axis limit of 1,048,576 accommodates massive 1D and 2D
 * compute grids (e.g., 1024x1024 tiles or 67M float elements at local_size=64)
 * while rejecting runaway or corrupted grid parameters before hardware submission.
 */
#define AC_MAX_SAFE_GROUPS_PER_AXIS   1048576u
#define AC_MAX_SAFE_DISPATCH_GROUPS_X AC_MAX_SAFE_GROUPS_PER_AXIS

/**
 * @brief Maximum safe total workgroups across all dimensions (X * Y * Z).
 * Default: 16,777,216 (2^24).
 *
 * Rationale:
 * Restricts total 3D workgroup product to prevent memory bandwidth starvation,
 * device queue timeout, and integer arithmetic overflow.
 */
#define AC_MAX_SAFE_TOTAL_GROUPS      16777216ull

/**
 * @brief Pure function to validate compute dispatch grid dimensions against device hardware limits.
 *
 * @param grid Requested 3D dispatch grid dimensions.
 * @param max_group_count Physical device workgroup limits (maxComputeWorkGroupCount[3]).
 * @return AC_SUCCESS on valid grid, AC_ERR_INVALID_WORK_GROUP_SIZE if zero or exceeding device limit,
 *         AC_ERR_NUMERICAL_OVERFLOW if total workgroups exceed 32-bit uint capacity.
 */
AcResult ac_validate_dispatch_grid(
    const AcDim3 grid,
    const uint32_t max_group_count[3]
);

/**
 * @brief Pure function to validate compute dispatch grid dimensions against AMEVA runtime safety budget.
 *
 * Validates that:
 *   1. No axis is zero (grid.x == 0 || grid.y == 0 || grid.z == 0).
 *   2. No single axis exceeds max_safe_groups_per_axis (checked uniformly on X, Y, and Z).
 *   3. 64-bit multiplication does not overflow.
 *   4. Total workgroups (grid.x * grid.y * grid.z) does not exceed max_safe_total_groups.
 *
 * @param grid Requested 3D dispatch grid dimensions.
 * @param max_safe_groups_per_axis Maximum allowed workgroups per axis (0 disables check).
 * @param max_safe_total_groups Maximum allowed total workgroups (0 disables check).
 * @return AC_SUCCESS on valid budget, AC_ERR_INVALID_WORK_GROUP_SIZE if zero or exceeding budget,
 *         AC_ERR_NUMERICAL_OVERFLOW on 64-bit product overflow.
 */
AcResult ac_validate_dispatch_budget(
    const AcDim3 grid,
    uint32_t max_safe_groups_per_axis,
    uint64_t max_safe_total_groups
);

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
