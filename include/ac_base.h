/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_base.h
 * @brief AMUDA Base Types, Status Codes, and ABI Versioning
 *
 * Provides vendor-neutral, hardware-neutral primitive types, dimensions,
 * status codes, and ABI definitions shared across all AMUDA subsystems.
 * Contains ZERO Vulkan or OpenCL header dependencies.
 */

#ifndef AC_BASE_H
#define AC_BASE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief AMUDA ABI Version Specification (Major.Minor.Patch -> 0xMMNNPP00)
 */
#define AC_ABI_VERSION_MAJOR 2u
#define AC_ABI_VERSION_MINOR 1u
#define AC_ABI_VERSION_PATCH 0u

#define AC_ABI_VERSION 0x00020100u

/**
 * @brief Supported Hardware Compute Backend Targets
 */
typedef uint32_t AcBackendKind;

#define AC_BACKEND_UNKNOWN ((AcBackendKind)0u)
#define AC_BACKEND_VULKAN  ((AcBackendKind)1u)
#define AC_BACKEND_OPENCL  ((AcBackendKind)2u) /* AMUDA 2.3: OpenCL 2.0+ with coarse-grained SVM (ac_runtime.h) */

/**
 * @brief Specialization Constant Types
 */
typedef uint32_t AcSpecConstantType;
#define AC_SPEC_CONSTANT_U32 ((AcSpecConstantType)0u)
#define AC_SPEC_CONSTANT_I32 ((AcSpecConstantType)1u)
#define AC_SPEC_CONSTANT_F32 ((AcSpecConstantType)2u)

/**
 * @brief Specialization Constant Value Override
 */
typedef struct {
    uint32_t constant_id;
    AcSpecConstantType type;
    uint32_t value_bits; /* Bit-preserving 32-bit payload */
} AcSpecConstant;

/**
 * @brief Specialization Constant Metadata Schema Entry
 */
typedef struct {
    uint32_t constant_id;
    AcSpecConstantType type;
    uint32_t default_value_bits;
    uint32_t min_value_bits;
    uint32_t max_value_bits;
    uint32_t overrides_local_size_axis; /* 0: None, 1: X, 2: Y, 3: Z */
} AcSpecConstantSchemaEntry;

#define AC_PIPELINE_IDENTITY_SCHEMA_VERSION 1u
#define AC_PIPELINE_IDENTITY_SHA256_HEX_SIZE 65u

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
    AC_ERR_INVALID_ARGUMENT             = -106,
    AC_ERR_CONTEXT_MISMATCH             = -107,
    AC_ERR_FEATURE_NOT_SUPPORTED        = -108,
    AC_ERR_INVALID_SHAPE                = -109,
    AC_ERR_ABI_MISMATCH                 = -110,
    AC_ERR_BACKEND_UNAVAILABLE          = -111, /* context creation refused for this host process (ac_security_guard) */

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
    AC_ERR_RESOURCE_IN_USE              = -407,
    
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
    AC_QUIRK_MALI_SUBGROUP_16           = 1 << 0,
    AC_QUIRK_MALI_MODULO_128            = 1 << 1,
    AC_QUIRK_ADRENO_PNEXT_BYPASS        = 1 << 2,
    AC_QUIRK_ADRENO_FP16_REDUCE         = 1 << 3,
    AC_QUIRK_HEADLESS_EGL               = 1 << 4
} AcQuirkFlags;

/**
 * @brief Universal Backend Feature Flags
 */
typedef enum {
    AC_FEATURE_NONE                     = 0,
    AC_FEATURE_TIMESTAMPS               = 1 << 0,
    AC_FEATURE_NON_COHERENT_ATOM        = 1 << 1
} AcFeatureFlags;

/**
 * @brief 3D Dimensions for Grid and Workgroup Blocks
 */
typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} AcDim3;

#ifdef __cplusplus
}
#endif

#endif /* AC_BASE_H */
