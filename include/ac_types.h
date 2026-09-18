/**
 * @file ac_types.h
 * @brief AMEVA-COMPUTE (AMUDA) Core Type Definitions and Status Codes
 *
 * Ground Truth Engineering: Zero-Silent-Fallback & Fail-Fast Doctrine.
 *
 * Thread Safety Specification:
 *   AMUDA Core ABI is strictly single-thread-owned per context.
 *   Concurrent API invocations on the same AcVkContext, AcVkBuffer, AcVkKernel,
 *   or AcVkEvent instances without external synchronization are unsupported.
 *   Multi-threaded workloads must instantiate separate AcVkContext instances.
 */

#ifndef AC_TYPES_H
#define AC_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ac_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for Vulkan Compute Core Types */
typedef struct AcVkContext AcVkContext;
typedef struct AcVkBuffer  AcVkBuffer;
typedef struct AcVkKernel  AcVkKernel;
typedef struct AcVkEvent   AcVkEvent;

#include "ac_error.h"

/**
 * @brief Legacy OpenCL Device Context (Maintained for backwards compatibility)
 */
typedef struct {
    void*        driver_handle;
    void*        platform_id;
    void*        device_id;
    void*        context;
    void*        command_queue;
    char         device_name[128];
    char         vendor_name[64];
    uint32_t     vendor_id;
    uint32_t     max_compute_units;
    uint64_t     max_work_group_size;
    uint64_t     max_work_item_sizes[3];
    uint64_t     global_mem_size;
    uint64_t     local_mem_size;
    uint32_t     active_quirks;
    bool         is_uma_supported;
} AcContext;

/**
 * @brief Legacy OpenCL UMA Zero-Copy Memory Buffer
 */
typedef struct {
    void*        host_ptr;
    void*        native_mem;
    size_t       size_bytes;
    uint32_t     flags;
    bool         is_mapped;
    bool         is_user_allocated;
} AcBuffer;

/**
 * @brief Legacy OpenCL Executable Kernel
 */
typedef struct {
    void*        program;
    void*        kernel;
    char         kernel_name[64];
    uint64_t     work_group_size;
    bool         is_builtin;
} AcKernel;

/**
 * @brief Legacy OpenCL Hardware Event & Profiler
 */
typedef struct {
    void*        native_event;
    uint64_t     submit_time_ns;
    uint64_t     start_time_ns;
    uint64_t     end_time_ns;
    uint64_t     duration_us;
} AcEvent;

#ifdef __cplusplus
}
#endif

#endif /* AC_TYPES_H */

