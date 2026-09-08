/**
 * @file amuda_memory.c
 * @brief UMA Zero-Copy Memory Management, 64-Byte Alignment, and Cache Coherency
 */

#include "amuda.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define DLSYM(h, sym) GetProcAddress((HMODULE)(h), sym)
static void* aligned_alloc_compat(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void aligned_free_compat(void* ptr) {
    _aligned_free(ptr);
}
#else
#include <dlfcn.h>
#define DLSYM(h, sym) dlsym(h, sym)
static void* aligned_alloc_compat(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}
static void aligned_free_compat(void* ptr) {
    free(ptr);
}
#endif

typedef int32_t   cl_int;
typedef uint64_t  cl_bitfield;
typedef void*     cl_context;
typedef void*     cl_command_queue;
typedef void*     cl_mem;
typedef void*     cl_event;

#define CL_SUCCESS                          0
#define CL_MEM_READ_WRITE                   (1 << 0)
#define CL_MEM_USE_HOST_PTR                 (1 << 3)
#define CL_MAP_READ                         (1 << 0)
#define CL_MAP_WRITE                        (1 << 1)
#define CL_TRUE                             1

typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*);
typedef void* (*fn_clEnqueueMapBuffer)(cl_command_queue, cl_mem, uint32_t, cl_bitfield, size_t, size_t, uint32_t, const cl_event*, cl_event*, cl_int*);
typedef cl_int (*fn_clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, uint32_t, const cl_event*, cl_event*);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);

static fn_clCreateBuffer          g_clCreateBuffer = NULL;
static fn_clEnqueueMapBuffer      g_clEnqueueMapBuffer = NULL;
static fn_clEnqueueUnmapMemObject g_clEnqueueUnmapMemObject = NULL;
static fn_clReleaseMemObject      g_clReleaseMemObject = NULL;

static void ensure_mem_symbols(AcContext* ctx) {
    if (!g_clCreateBuffer && ctx->driver_handle) {
        g_clCreateBuffer = (fn_clCreateBuffer)DLSYM(ctx->driver_handle, "clCreateBuffer");
        g_clEnqueueMapBuffer = (fn_clEnqueueMapBuffer)DLSYM(ctx->driver_handle, "clEnqueueMapBuffer");
        g_clEnqueueUnmapMemObject = (fn_clEnqueueUnmapMemObject)DLSYM(ctx->driver_handle, "clEnqueueUnmapMemObject");
        g_clReleaseMemObject = (fn_clReleaseMemObject)DLSYM(ctx->driver_handle, "clReleaseMemObject");
    }
}

AcResult ac_buffer_alloc(AcContext* ctx, size_t size_bytes, AcBuffer** out_buf) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!out_buf) return AC_ERR_NULL_POINTER;
    if (size_bytes == 0) return AC_ERR_INVALID_BUFFER_SIZE;

    ensure_mem_symbols(ctx);
    if (!g_clCreateBuffer) return AC_ERR_DRIVER_SYMBOL_MISSING;

    /* 64-Byte Cache-Line Aligned Allocation */
    void* host_ptr = aligned_alloc_compat(64, size_bytes);
    if (!host_ptr) {
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }
    memset(host_ptr, 0, size_bytes);

    /* Bind to GPU Physical Address Space via Zero-Copy UMA */
    cl_int err = CL_SUCCESS;
    cl_mem native_mem = g_clCreateBuffer((cl_context)ctx->context,
                                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                         size_bytes,
                                         host_ptr,
                                         &err);

    if (err != CL_SUCCESS || !native_mem) {
        aligned_free_compat(host_ptr);
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    /* Initial map to establish host write access and permit subsequent unmap */
    void* mapped = g_clEnqueueMapBuffer((cl_command_queue)ctx->command_queue,
                                        native_mem,
                                        CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE,
                                        0,
                                        size_bytes,
                                        0, NULL, NULL,
                                        &err);
    if (err != CL_SUCCESS || !mapped) {
        g_clReleaseMemObject(native_mem);
        aligned_free_compat(host_ptr);
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    AcBuffer* buf = (AcBuffer*)calloc(1, sizeof(AcBuffer));
    if (!buf) {
        g_clReleaseMemObject(native_mem);
        aligned_free_compat(host_ptr);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    buf->host_ptr = mapped;
    buf->native_mem = native_mem;
    buf->size_bytes = size_bytes;
    buf->is_mapped = true;
    buf->is_user_allocated = false;

    *out_buf = buf;
    return AC_SUCCESS;
}

AcResult ac_buffer_from_host(AcContext* ctx, void* aligned_ptr, size_t size_bytes, AcBuffer** out_buf) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!aligned_ptr || !out_buf) return AC_ERR_NULL_POINTER;
    if (size_bytes == 0) return AC_ERR_INVALID_BUFFER_SIZE;

    /* Enforce 64-Byte Alignment */
    if (((uintptr_t)aligned_ptr % 64) != 0) {
        return AC_ERR_BUFFER_ALIGNMENT_INVALID;
    }

    /* Enforce Mali 128-Byte Boundary Rule if Active */
    if ((ctx->active_quirks & AC_QUIRK_MALI_MODULO_128) && ((uintptr_t)aligned_ptr % 128 != 0)) {
        return AC_ERR_BUFFER_ALIGNMENT_INVALID;
    }

    ensure_mem_symbols(ctx);
    if (!g_clCreateBuffer) return AC_ERR_DRIVER_SYMBOL_MISSING;

    cl_int err = CL_SUCCESS;
    cl_mem native_mem = g_clCreateBuffer((cl_context)ctx->context,
                                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                         size_bytes,
                                         aligned_ptr,
                                         &err);

    if (err != CL_SUCCESS || !native_mem) {
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    void* mapped = g_clEnqueueMapBuffer((cl_command_queue)ctx->command_queue,
                                        native_mem,
                                        CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE,
                                        0,
                                        size_bytes,
                                        0, NULL, NULL,
                                        &err);
    if (err != CL_SUCCESS || !mapped) {
        g_clReleaseMemObject(native_mem);
        return AC_ERR_OUT_OF_DEVICE_MEMORY;
    }

    AcBuffer* buf = (AcBuffer*)calloc(1, sizeof(AcBuffer));
    if (!buf) {
        g_clReleaseMemObject(native_mem);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    buf->host_ptr = mapped;
    buf->native_mem = native_mem;
    buf->size_bytes = size_bytes;
    buf->is_mapped = true;
    buf->is_user_allocated = true;

    *out_buf = buf;
    return AC_SUCCESS;
}

AcResult ac_buffer_sync_to_device(AcContext* ctx, AcBuffer* buf) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!buf || !buf->native_mem) return AC_ERR_NULL_POINTER;

    ensure_mem_symbols(ctx);
    if (!g_clEnqueueUnmapMemObject) return AC_ERR_DRIVER_SYMBOL_MISSING;

    /* Unmap/Flush Host Cache Line to Physical DRAM */
    if (buf->is_mapped) {
        cl_int err = g_clEnqueueUnmapMemObject((cl_command_queue)ctx->command_queue,
                                               (cl_mem)buf->native_mem,
                                               buf->host_ptr,
                                               0, NULL, NULL);
        if (err != CL_SUCCESS) {
            return AC_ERR_BUFFER_NOT_MAPPED;
        }
        buf->is_mapped = false;
    }
    return AC_SUCCESS;
}

AcResult ac_buffer_sync_to_host(AcContext* ctx, AcBuffer* buf) {
    if (!ctx) return AC_ERR_INVALID_CONTEXT;
    if (!buf || !buf->native_mem) return AC_ERR_NULL_POINTER;

    ensure_mem_symbols(ctx);
    if (!g_clEnqueueMapBuffer) return AC_ERR_DRIVER_SYMBOL_MISSING;

    /* Invalidate Cache and Map Physical DRAM to Host Virtual Space */
    if (!buf->is_mapped) {
        cl_int err = CL_SUCCESS;
        void* mapped = g_clEnqueueMapBuffer((cl_command_queue)ctx->command_queue,
                                            (cl_mem)buf->native_mem,
                                            CL_TRUE, /* Blocking call */
                                            CL_MAP_READ | CL_MAP_WRITE,
                                            0,
                                            buf->size_bytes,
                                            0, NULL, NULL,
                                            &err);
        if (err != CL_SUCCESS || !mapped) {
            return AC_ERR_BUFFER_NOT_MAPPED;
        }
        buf->host_ptr = mapped;
        buf->is_mapped = true;
    }
    return AC_SUCCESS;
}

AcResult ac_buffer_free(AcContext* ctx, AcBuffer* buf) {
    if (!buf) return AC_ERR_NULL_POINTER;

    ensure_mem_symbols(ctx);
    if (ctx && buf->is_mapped && buf->native_mem && g_clEnqueueUnmapMemObject) {
        g_clEnqueueUnmapMemObject((cl_command_queue)ctx->command_queue,
                                  (cl_mem)buf->native_mem,
                                  buf->host_ptr,
                                  0, NULL, NULL);
        buf->is_mapped = false;
    }
    if (buf->native_mem && g_clReleaseMemObject) {
        g_clReleaseMemObject((cl_mem)buf->native_mem);
    }
    if (!buf->is_user_allocated && buf->host_ptr) {
        aligned_free_compat(buf->host_ptr);
    }

    free(buf);
    return AC_SUCCESS;
}
