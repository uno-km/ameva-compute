/**
 * @file amuda_context.c
 * @brief Driver Loading, Context Management, and Hardware Quirk Detection
 */

#include "amuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define DLOPEN(path) LoadLibraryA(path)
#define DLSYM(h, sym) GetProcAddress((HMODULE)(h), sym)
#define DLCLOSE(h) FreeLibrary((HMODULE)(h))
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define DLSYM(h, sym) dlsym(h, sym)
#define DLCLOSE(h) dlclose(h)
#endif

/* Standard OpenCL Minimal ABI Definitions */
typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef uint64_t  cl_bitfield;
typedef void*     cl_platform_id;
typedef void*     cl_device_id;
typedef void*     cl_context;
typedef void*     cl_command_queue;
typedef void*     cl_mem;
typedef void*     cl_program;
typedef void*     cl_kernel;
typedef void*     cl_event;

#define CL_SUCCESS                                  0
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_VENDOR                            0x102C
#define CL_DEVICE_VENDOR_ID                         0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS                 0x1002
#define CL_DEVICE_MAX_WORK_ITEM_SIZES               0x1005
#define CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE                   0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE                    0x1023
#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)

/* Dynamic Function Pointers */
typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*fn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
typedef cl_int (*fn_clReleaseContext)(cl_context);
typedef cl_int (*fn_clReleaseCommandQueue)(cl_command_queue);

typedef struct {
    fn_clGetPlatformIDs       clGetPlatformIDs;
    fn_clGetDeviceIDs         clGetDeviceIDs;
    fn_clGetDeviceInfo        clGetDeviceInfo;
    fn_clCreateContext        clCreateContext;
    fn_clCreateCommandQueue   clCreateCommandQueue;
    fn_clReleaseContext       clReleaseContext;
    fn_clReleaseCommandQueue  clReleaseCommandQueue;
} OpenCL_Symbols;

static OpenCL_Symbols g_cl = {0};

static const char* g_driver_search_paths[] = {
    "/vendor/lib64/libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    "/vendor/lib64/egl/libGLES_mali.so",
    "/vendor/lib64/libmali.so",
    "/system/lib64/libOpenCL.so",
    "libOpenCL.so",
#if defined(_WIN32)
    "OpenCL.dll",
#endif
    NULL
};

AcResult ac_init(AcContext** out_ctx) {
    if (!out_ctx) return AC_ERR_NULL_POINTER;

    void* driver_handle = NULL;
    for (int i = 0; g_driver_search_paths[i] != NULL; ++i) {
        driver_handle = DLOPEN(g_driver_search_paths[i]);
        if (driver_handle != NULL) {
            break;
        }
    }

    if (!driver_handle) {
        return AC_ERR_DRIVER_NOT_FOUND;
    }

    /* Resolve core symbols */
    g_cl.clGetPlatformIDs = (fn_clGetPlatformIDs)DLSYM(driver_handle, "clGetPlatformIDs");
    g_cl.clGetDeviceIDs = (fn_clGetDeviceIDs)DLSYM(driver_handle, "clGetDeviceIDs");
    g_cl.clGetDeviceInfo = (fn_clGetDeviceInfo)DLSYM(driver_handle, "clGetDeviceInfo");
    g_cl.clCreateContext = (fn_clCreateContext)DLSYM(driver_handle, "clCreateContext");
    g_cl.clCreateCommandQueue = (fn_clCreateCommandQueue)DLSYM(driver_handle, "clCreateCommandQueue");
    g_cl.clReleaseContext = (fn_clReleaseContext)DLSYM(driver_handle, "clReleaseContext");
    g_cl.clReleaseCommandQueue = (fn_clReleaseCommandQueue)DLSYM(driver_handle, "clReleaseCommandQueue");

    if (!g_cl.clGetPlatformIDs || !g_cl.clGetDeviceIDs || !g_cl.clGetDeviceInfo ||
        !g_cl.clCreateContext || !g_cl.clCreateCommandQueue) {
        DLCLOSE(driver_handle);
        return AC_ERR_DRIVER_SYMBOL_MISSING;
    }

    /* Query Platform and GPU Device */
    cl_platform_id platform = NULL;
    cl_uint num_platforms = 0;
    cl_int err = g_cl.clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        DLCLOSE(driver_handle);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    cl_device_id device = NULL;
    cl_uint num_devices = 0;
    err = g_cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        DLCLOSE(driver_handle);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    /* Allocate AcContext */
    AcContext* ctx = (AcContext*)calloc(1, sizeof(AcContext));
    if (!ctx) {
        DLCLOSE(driver_handle);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    ctx->driver_handle = driver_handle;
    ctx->platform_id = platform;
    ctx->device_id = device;

    /* Query hardware specifications */
    g_cl.clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(ctx->device_name), ctx->device_name, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(ctx->vendor_name), ctx->vendor_name, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(ctx->vendor_id), &ctx->vendor_id, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ctx->max_compute_units), &ctx->max_compute_units, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ctx->max_work_group_size), &ctx->max_work_group_size, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(ctx->max_work_item_sizes), ctx->max_work_item_sizes, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ctx->global_mem_size), &ctx->global_mem_size, NULL);
    g_cl.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ctx->local_mem_size), &ctx->local_mem_size, NULL);

    /* Hardware Quirk Evaluation */
    ctx->active_quirks = AC_QUIRK_NONE;
    if (strstr(ctx->device_name, "Adreno") != NULL || strstr(ctx->vendor_name, "Qualcomm") != NULL) {
        ctx->active_quirks |= (AC_QUIRK_ADRENO_PNEXT_BYPASS | AC_QUIRK_ADRENO_FP16_REDUCE);
        ctx->is_uma_supported = true;
    } else if (strstr(ctx->device_name, "Mali") != NULL || strstr(ctx->vendor_name, "ARM") != NULL) {
        ctx->active_quirks |= (AC_QUIRK_MALI_SUBGROUP_16 | AC_QUIRK_MALI_MODULO_128);
        ctx->is_uma_supported = true;
    } else if (strstr(ctx->device_name, "Xclipse") != NULL || strstr(ctx->vendor_name, "Samsung") != NULL) {
        ctx->active_quirks |= AC_QUIRK_MALI_MODULO_128;
        ctx->is_uma_supported = true;
    } else {
        ctx->is_uma_supported = false;
    }

    /* Create Context & Command Queue */
    cl_int status = 0;
    ctx->context = g_cl.clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS || !ctx->context) {
        free(ctx);
        DLCLOSE(driver_handle);
        return AC_ERR_DRIVER_INIT_FAILED;
    }

    ctx->command_queue = g_cl.clCreateCommandQueue((cl_context)ctx->context, (cl_device_id)ctx->device_id, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS || !ctx->command_queue) {
        g_cl.clReleaseContext((cl_context)ctx->context);
        free(ctx);
        DLCLOSE(driver_handle);
        return AC_ERR_DRIVER_INIT_FAILED;
    }

    *out_ctx = ctx;
    return AC_SUCCESS;
}

AcResult ac_destroy(AcContext* ctx) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;

    if (ctx->command_queue && g_cl.clReleaseCommandQueue) {
        g_cl.clReleaseCommandQueue((cl_command_queue)ctx->command_queue);
    }
    if (ctx->context && g_cl.clReleaseContext) {
        g_cl.clReleaseContext((cl_context)ctx->context);
    }
    /* On Android, dlclose on vendor GPU drivers deadlocks internal driver worker threads */
#if !defined(__ANDROID__)
    if (ctx->driver_handle) {
        DLCLOSE(ctx->driver_handle);
    }
#endif

    free(ctx);
    return AC_SUCCESS;
}
