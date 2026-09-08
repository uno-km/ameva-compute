/**
 * @file amuda_dispatch.c
 * @brief Dynamic Kernel Compilation, 3D Grid/Block Indexing, and Hardware Profiling
 */

#include "amuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define DLSYM(h, sym) GetProcAddress((HMODULE)(h), sym)
#else
#include <dlfcn.h>
#define DLSYM(h, sym) dlsym(h, sym)
#endif

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef void*     cl_context;
typedef void*     cl_command_queue;
typedef void*     cl_program;
typedef void*     cl_kernel;
typedef void*     cl_event;

#define CL_SUCCESS                          0
#define CL_PROGRAM_BUILD_LOG                0x1183
#define CL_PROFILING_COMMAND_START          0x1282
#define CL_PROFILING_COMMAND_END            0x1283

typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_program (*fn_clCreateProgramWithBinary)(cl_context, cl_uint, const void*, const size_t*, const unsigned char**, cl_int*, cl_int*);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const void*, const char*, void*, void*);
typedef cl_int (*fn_clGetProgramBuildInfo)(cl_program, void*, uint32_t, size_t, void*, size_t*);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clWaitForEvents)(cl_uint, const cl_event*);
typedef cl_int (*fn_clGetEventProfilingInfo)(cl_event, uint32_t, size_t, void*, size_t*);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program);
typedef cl_int (*fn_clReleaseEvent)(cl_event);

static fn_clCreateProgramWithSource g_clCreateProgramWithSource = NULL;
static fn_clCreateProgramWithBinary g_clCreateProgramWithBinary = NULL;
static fn_clBuildProgram            g_clBuildProgram = NULL;
static fn_clGetProgramBuildInfo     g_clGetProgramBuildInfo = NULL;
static fn_clCreateKernel            g_clCreateKernel = NULL;
static fn_clSetKernelArg            g_clSetKernelArg = NULL;
static fn_clEnqueueNDRangeKernel    g_clEnqueueNDRangeKernel = NULL;
static fn_clWaitForEvents           g_clWaitForEvents = NULL;
static fn_clGetEventProfilingInfo   g_clGetEventProfilingInfo = NULL;
static fn_clReleaseKernel           g_clReleaseKernel = NULL;
static fn_clReleaseProgram          g_clReleaseProgram = NULL;
static fn_clReleaseEvent            g_clReleaseEvent = NULL;

static void ensure_dispatch_symbols(AcContext* ctx) {
    if (!g_clCreateKernel && ctx->driver_handle) {
        g_clCreateProgramWithSource = (fn_clCreateProgramWithSource)DLSYM(ctx->driver_handle, "clCreateProgramWithSource");
        g_clCreateProgramWithBinary = (fn_clCreateProgramWithBinary)DLSYM(ctx->driver_handle, "clCreateProgramWithBinary");
        g_clBuildProgram = (fn_clBuildProgram)DLSYM(ctx->driver_handle, "clBuildProgram");
        g_clGetProgramBuildInfo = (fn_clGetProgramBuildInfo)DLSYM(ctx->driver_handle, "clGetProgramBuildInfo");
        g_clCreateKernel = (fn_clCreateKernel)DLSYM(ctx->driver_handle, "clCreateKernel");
        g_clSetKernelArg = (fn_clSetKernelArg)DLSYM(ctx->driver_handle, "clSetKernelArg");
        g_clEnqueueNDRangeKernel = (fn_clEnqueueNDRangeKernel)DLSYM(ctx->driver_handle, "clEnqueueNDRangeKernel");
        g_clWaitForEvents = (fn_clWaitForEvents)DLSYM(ctx->driver_handle, "clWaitForEvents");
        g_clGetEventProfilingInfo = (fn_clGetEventProfilingInfo)DLSYM(ctx->driver_handle, "clGetEventProfilingInfo");
        g_clReleaseKernel = (fn_clReleaseKernel)DLSYM(ctx->driver_handle, "clReleaseKernel");
        g_clReleaseProgram = (fn_clReleaseProgram)DLSYM(ctx->driver_handle, "clReleaseProgram");
        g_clReleaseEvent = (fn_clReleaseEvent)DLSYM(ctx->driver_handle, "clReleaseEvent");
    }
}

static const char* k_amuda_device_preamble =
"/* AMUDA SIMT Kernel Device Preamble */\n"
"#define __global__      __kernel\n"
"#define __device__      static inline\n"
"#define __host__        \n"
"#define __constant__    __constant\n"
"#define __shared__      __local\n"
"\n"
"/* 1D Index Accessors */\n"
"#define threadIdx_x     ((int)get_local_id(0))\n"
"#define threadIdx_y     ((int)get_local_id(1))\n"
"#define threadIdx_z     ((int)get_local_id(2))\n"
"\n"
"#define blockIdx_x      ((int)get_group_id(0))\n"
"#define blockIdx_y      ((int)get_group_id(1))\n"
"#define blockIdx_z      ((int)get_group_id(2))\n"
"\n"
"#define blockDim_x      ((int)get_local_size(0))\n"
"#define blockDim_y      ((int)get_local_size(1))\n"
"#define blockDim_z      ((int)get_local_size(2))\n"
"\n"
"#define gridDim_x       ((int)get_num_groups(0))\n"
"#define gridDim_y       ((int)get_num_groups(1))\n"
"#define gridDim_z       ((int)get_num_groups(2))\n"
"\n"
"/* 3D Struct Syntax Emulation */\n"
"typedef struct { uint x; uint y; uint z; } _AmudaDim3Accessor;\n"
"#define threadIdx       ((_AmudaDim3Accessor){ (uint)get_local_id(0), (uint)get_local_id(1), (uint)get_local_id(2) })\n"
"#define blockIdx        ((_AmudaDim3Accessor){ (uint)get_group_id(0), (uint)get_group_id(1), (uint)get_group_id(2) })\n"
"#define blockDim        ((_AmudaDim3Accessor){ (uint)get_local_size(0), (uint)get_local_size(1), (uint)get_local_size(2) })\n"
"#define gridDim         ((_AmudaDim3Accessor){ (uint)get_num_groups(0), (uint)get_num_groups(1), (uint)get_num_groups(2) })\n"
"\n"
"/* Thread Synchronization & Hardware Fences */\n"
"#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)\n"
"\n"
"/* Hardware Atomic Primitives */\n"
"#define atomicAdd(p, v) atomic_add((volatile __global int*)(p), (int)(v))\n"
"#define atomicSub(p, v) atomic_sub((volatile __global int*)(p), (int)(v))\n"
"#define atomicMin(p, v) atomic_min((volatile __global int*)(p), (int)(v))\n"
"#define atomicMax(p, v) atomic_max((volatile __global int*)(p), (int)(v))\n"
"\n";

static char* sanitize_kernel_source(const char* src) {
    char* copy = strdup(src);
    if (!copy) return NULL;
    char* p = strstr(copy, "#include \"amuda.h\"");
    if (p) {
        memset(p, ' ', strlen("#include \"amuda.h\""));
    }
    p = strstr(copy, "#include <amuda.h>");
    if (p) {
        memset(p, ' ', strlen("#include <amuda.h>"));
    }
    return copy;
}

AcResult ac_kernel_load_source(AcContext* ctx, const char* source_code, const char* func_name, AcKernel** out_kernel) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!source_code || !func_name || !out_kernel) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (!g_clCreateProgramWithSource || !g_clBuildProgram || !g_clCreateKernel) {
        return AC_ERR_DRIVER_SYMBOL_MISSING;
    }

    char* clean_src = sanitize_kernel_source(source_code);
    const char* sources[2] = { k_amuda_device_preamble, clean_src ? clean_src : source_code };
    cl_int err = CL_SUCCESS;
    cl_program prog = g_clCreateProgramWithSource((cl_context)ctx->context, 2, sources, NULL, &err);
    if (clean_src) free(clean_src);
    if (err != CL_SUCCESS || !prog) {
        return AC_ERR_KERNEL_BUILD_FAILED;
    }

    /* Compile with high-performance relaxed math flags */
    const char* options = "-cl-fast-relaxed-math -cl-mad-enable";
    err = g_clBuildProgram(prog, 1, &ctx->device_id, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        if (g_clGetProgramBuildInfo) {
            char log[4096] = {0};
            g_clGetProgramBuildInfo(prog, ctx->device_id, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
            fprintf(stderr, "[AMUDA_BUILD_ERROR]\n%s\n", log);
        }
        g_clReleaseProgram(prog);
        return AC_ERR_KERNEL_BUILD_FAILED;
    }

    cl_kernel k = g_clCreateKernel(prog, func_name, &err);
    if (err != CL_SUCCESS || !k) {
        g_clReleaseProgram(prog);
        return AC_ERR_KERNEL_NOT_FOUND;
    }

    AcKernel* kernel = (AcKernel*)calloc(1, sizeof(AcKernel));
    if (!kernel) {
        g_clReleaseKernel(k);
        g_clReleaseProgram(prog);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    kernel->program = prog;
    kernel->kernel = k;
    strncpy(kernel->kernel_name, func_name, sizeof(kernel->kernel_name) - 1);
    kernel->is_builtin = false;

    *out_kernel = kernel;
    return AC_SUCCESS;
}

AcResult ac_kernel_load_binary(AcContext* ctx, const void* binary_data, size_t binary_size, const char* func_name, AcKernel** out_kernel) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!binary_data || !func_name || !out_kernel) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (!g_clCreateProgramWithBinary || !g_clCreateKernel) {
        return AC_ERR_DRIVER_SYMBOL_MISSING;
    }

    cl_int binary_status = 0;
    cl_int err = CL_SUCCESS;
    const unsigned char* bin_ptr = (const unsigned char*)binary_data;

    cl_program prog = g_clCreateProgramWithBinary((cl_context)ctx->context,
                                                  1,
                                                  &ctx->device_id,
                                                  &binary_size,
                                                  &bin_ptr,
                                                  &binary_status,
                                                  &err);

    if (err != CL_SUCCESS || binary_status != CL_SUCCESS || !prog) {
        return AC_ERR_BINARY_CORRUPTED;
    }

    err = g_clBuildProgram(prog, 1, &ctx->device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        g_clReleaseProgram(prog);
        return AC_ERR_KERNEL_BUILD_FAILED;
    }

    cl_kernel k = g_clCreateKernel(prog, func_name, &err);
    if (err != CL_SUCCESS || !k) {
        g_clReleaseProgram(prog);
        return AC_ERR_KERNEL_NOT_FOUND;
    }

    AcKernel* kernel = (AcKernel*)calloc(1, sizeof(AcKernel));
    if (!kernel) {
        g_clReleaseKernel(k);
        g_clReleaseProgram(prog);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    kernel->program = prog;
    kernel->kernel = k;
    strncpy(kernel->kernel_name, func_name, sizeof(kernel->kernel_name) - 1);
    kernel->is_builtin = true;

    *out_kernel = kernel;
    return AC_SUCCESS;
}

AcResult ac_kernel_free(AcContext* ctx, AcKernel* kernel) {
    (void)ctx;
    if (!kernel) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (kernel->kernel && g_clReleaseKernel) {
        g_clReleaseKernel((cl_kernel)kernel->kernel);
    }
    if (kernel->program && g_clReleaseProgram) {
        g_clReleaseProgram((cl_program)kernel->program);
    }

    free(kernel);
    return AC_SUCCESS;
}

AcResult ac_launch(
    AcContext* ctx,
    AcKernel*  kernel,
    AcDim3     grid,
    AcDim3     block,
    uint32_t   argc,
    void**     argv,
    size_t*    arg_sizes,
    AcEvent**  out_event
) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!kernel || !kernel->kernel) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (!g_clEnqueueNDRangeKernel || !g_clSetKernelArg) {
        return AC_ERR_DRIVER_SYMBOL_MISSING;
    }

    /* 1. Dimension Validation & Quirk Enforcement */
    uint64_t total_threads = (uint64_t)block.x * block.y * block.z;
    if (total_threads == 0 || total_threads > ctx->max_work_group_size) {
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }

    /* Mali Quirk: Prevent loadstride_b = 0 infinite loop */
    if ((ctx->active_quirks & AC_QUIRK_MALI_SUBGROUP_16) && (block.x < 32 && block.y == 1 && block.z == 1)) {
        return AC_ERR_MALI_INFINITE_LOOP_DETECTED;
    }

    /* 2. Bind Arguments */
    for (uint32_t i = 0; i < argc; ++i) {
        cl_int err = g_clSetKernelArg((cl_kernel)kernel->kernel, i, arg_sizes[i], argv[i]);
        if (err != CL_SUCCESS) {
            return AC_ERR_KERNEL_ARG_INVALID;
        }
    }

    /* 3. 3D Grid/Block to OpenCL Global/Local Size Mapping */
    size_t global_size[3] = {
        (size_t)grid.x * block.x,
        (size_t)grid.y * block.y,
        (size_t)grid.z * block.z
    };
    size_t local_size[3] = {
        (size_t)block.x,
        (size_t)block.y,
        (size_t)block.z
    };

    /* 4. Enqueue Non-blocking Kernel Launch */
    cl_event native_ev = NULL;
    cl_int err = g_clEnqueueNDRangeKernel((cl_command_queue)ctx->command_queue,
                                          (cl_kernel)kernel->kernel,
                                          3,
                                          NULL,
                                          global_size,
                                          local_size,
                                          0,
                                          NULL,
                                          out_event ? &native_ev : NULL);

    if (err != CL_SUCCESS) {
        return AC_ERR_DRIVER_CRASH_OR_LOST;
    }

    if (out_event && native_ev) {
        AcEvent* ev = (AcEvent*)calloc(1, sizeof(AcEvent));
        ev->native_event = native_ev;
        *out_event = ev;
    }

    return AC_SUCCESS;
}

AcResult ac_event_wait(AcContext* ctx, AcEvent* event) {
    (void)ctx;
    if (!event || !event->native_event) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (!g_clWaitForEvents) return AC_ERR_DRIVER_SYMBOL_MISSING;

    cl_int err = g_clWaitForEvents(1, (const cl_event*)&event->native_event);
    if (err != CL_SUCCESS) {
        return AC_ERR_LAUNCH_TIMEOUT;
    }

    if (g_clGetEventProfilingInfo) {
        cl_ulong start = 0, end = 0;
        g_clGetEventProfilingInfo((cl_event)event->native_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        g_clGetEventProfilingInfo((cl_event)event->native_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        event->start_time_ns = (uint64_t)start;
        event->end_time_ns = (uint64_t)end;
        event->duration_us = (uint64_t)((end - start) / 1000);
    }

    return AC_SUCCESS;
}

uint64_t ac_event_get_duration_us(AcEvent* event) {
    return event ? event->duration_us : 0;
}

AcResult ac_event_free(AcContext* ctx, AcEvent* event) {
    (void)ctx;
    if (!event) return AC_ERR_NULL_POINTER;

    ensure_dispatch_symbols(ctx);
    if (event->native_event && g_clReleaseEvent) {
        g_clReleaseEvent((cl_event)event->native_event);
    }
    free(event);
    return AC_SUCCESS;
}
