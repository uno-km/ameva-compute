/**
 * @file ac_types.h
 * @brief AMEVA-COMPUTE (AMUDA) Core Type Definitions and Status Codes
 *
 * Ground Truth Engineering: Zero-Silent-Fallback & Fail-Fast Doctrine.
 */

#ifndef AC_TYPES_H
#define AC_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Strict Status and Error Codes (Zero Silent Failures)
 */
typedef enum {
    AC_SUCCESS                          =  0,
    
    /* Driver & Environment Errors (100-199) */
    AC_ERR_INVALID_CONTEXT              = -101,
    AC_ERR_DRIVER_NOT_FOUND             = -102,
    AC_ERR_DRIVER_SYMBOL_MISSING        = -103,
    AC_ERR_DEVICE_NOT_SUPPORTED         = -104,
    AC_ERR_DRIVER_INIT_FAILED           = -105,
    
    /* Memory Management Errors (200-299) */
    AC_ERR_OUT_OF_HOST_MEMORY           = -201,
    AC_ERR_OUT_OF_DEVICE_MEMORY         = -202,
    AC_ERR_BUFFER_ALIGNMENT_INVALID     = -203,
    AC_ERR_BUFFER_NOT_MAPPED            = -204,
    AC_ERR_NULL_POINTER                 = -205,
    AC_ERR_INVALID_BUFFER_SIZE          = -206,
    
    /* Kernel Compilation Errors (300-399) */
    AC_ERR_KERNEL_BUILD_FAILED          = -301,
    AC_ERR_KERNEL_NOT_FOUND             = -302,
    AC_ERR_KERNEL_ARG_INVALID           = -303,
    AC_ERR_BINARY_CORRUPTED             = -304,
    
    /* Hardware Safety & Dispatch Errors (400-499) */
    AC_ERR_INVALID_WORK_GROUP_SIZE      = -401,
    AC_ERR_INVALID_WORK_ITEM_DIMS       = -402,
    AC_ERR_MALI_INFINITE_LOOP_DETECTED  = -403,
    AC_ERR_ADRENO_UNSAFE_SUBGROUP_OP    = -404,
    AC_ERR_LAUNCH_TIMEOUT               = -405,
    AC_ERR_DEVICE_BUSY                  = -406,
    
    /* Numerical Integrity Errors (500-599) */
    AC_ERR_NAN_OR_INF_DETECTED          = -501,
    AC_ERR_NUMERICAL_OVERFLOW           = -502,
    
    /* Hardware & OS Governance (600-699) */
    AC_ERR_DRIVER_CRASH_OR_LOST         = -601,
    AC_ERR_OS_PROCESS_KILL_SIGNAL       = -602,
    AC_ERR_THERMAL_THROTTLING_TRIPPED   = -603
} AcResult;

/**
 * @brief Hardware Quirk Flags
 */
typedef enum {
    AC_QUIRK_NONE                       = 0,
    AC_QUIRK_MALI_SUBGROUP_16           = 1 << 0,  /* Subgroup 16 stride zero infinite loop fix */
    AC_QUIRK_MALI_MODULO_128            = 1 << 1,  /* Strict 128-byte alignment enforcement */
    AC_QUIRK_ADRENO_PNEXT_BYPASS        = 1 << 2,  /* Bypass pNext when requested == native */
    AC_QUIRK_ADRENO_FP16_REDUCE         = 1 << 3,  /* Bypass broken FP16 subgroup intrinsics */
    AC_QUIRK_HEADLESS_EGL               = 1 << 4   /* Headless EGL dummy display shim */
} AcQuirkFlags;

/**
 * @brief 3D Dimensions for Grid and Block
 */
typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} AcDim3;

/**
 * @brief Device Context
 */
typedef struct {
    void*        driver_handle;         /* dlopen handle */
    void*        platform_id;           /* Native platform */
    void*        device_id;             /* Native device */
    void*        context;               /* Native context */
    void*        command_queue;         /* Native queue */
    char         device_name[128];
    char         vendor_name[64];
    uint32_t     vendor_id;
    uint32_t     max_compute_units;
    uint64_t     max_work_group_size;
    uint64_t     max_work_item_sizes[3];
    uint64_t     global_mem_size;
    uint64_t     local_mem_size;
    uint32_t     active_quirks;         /* Bitmask of AcQuirkFlags */
    bool         is_uma_supported;
} AcContext;

/**
 * @brief UMA Zero-Copy Memory Buffer
 */
typedef struct {
    void*        host_ptr;              /* 64-byte cache-line aligned CPU memory */
    void*        native_mem;            /* Native GPU buffer handle */
    size_t       size_bytes;
    uint32_t     flags;
    bool         is_mapped;
    bool         is_user_allocated;
} AcBuffer;

/**
 * @brief Executable Kernel
 */
typedef struct {
    void*        program;               /* Native program handle */
    void*        kernel;                /* Native kernel handle */
    char         kernel_name[64];
    uint64_t     work_group_size;
    bool         is_builtin;
} AcKernel;

/**
 * @brief Hardware Event & Profiler
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
