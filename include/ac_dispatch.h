/**
 * @file ac_dispatch.h
 * @brief Explicit Vulkan Compute Kernel Dispatch
 */
#ifndef AC_DISPATCH_H
#define AC_DISPATCH_H

#include "ac_types.h"
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
