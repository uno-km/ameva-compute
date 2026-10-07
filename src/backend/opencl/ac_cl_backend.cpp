/**
 * @file ac_cl_backend.cpp
 * @brief OpenCL backend of ac_runtime (AC_BACKEND_OPENCL). Contract: ac_cl_backend.h.
 *
 * The OpenCL headers are used for types and prototypes only (decltype of the entry points); every entry point is
 * resolved with dlsym from the vendor driver, which is opened once per process and never closed (GPU drivers are not
 * built to be unloaded).
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "ac_cl_backend.h"
#include "../vulkan/ac_vk_sha256.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

/* ------------------------------------------------------------------------------------------------------------------ */
/* Driver entry points                                                                                                */
/* ------------------------------------------------------------------------------------------------------------------ */

#define AC_CL_ENTRY_POINTS(X)                                                                                         \
    X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext)                 \
    X(clReleaseContext) X(clCreateCommandQueueWithProperties) X(clReleaseCommandQueue) X(clSVMAlloc) X(clSVMFree)     \
    X(clEnqueueSVMMemcpy) X(clEnqueueSVMMemFill) X(clCreateProgramWithSource) X(clCreateProgramWithBinary)            \
    X(clBuildProgram) X(clGetProgramBuildInfo) X(clGetProgramInfo) X(clReleaseProgram) X(clCreateKernel)              \
    X(clReleaseKernel) X(clGetKernelInfo) X(clGetKernelWorkGroupInfo) X(clSetKernelArg)                              \
    X(clSetKernelArgSVMPointer) X(clEnqueueNDRangeKernel) X(clEnqueueMarkerWithWaitList) X(clFlush) X(clFinish)       \
    X(clWaitForEvents) X(clGetEventInfo) X(clGetEventProfilingInfo) X(clRetainEvent) X(clReleaseEvent)
/* OpenCL 2.1+, only for reporting the subgroup size (0 without it; Adreno 650's OpenCL 2.0 driver lacks it) */
#define AC_CL_OPTIONAL_ENTRY_POINTS(X) X(clGetKernelSubGroupInfo)

struct ClApi {
#define AC_CL_MEMBER(name) decltype(&::name) name = nullptr;
    AC_CL_ENTRY_POINTS(AC_CL_MEMBER)
    AC_CL_OPTIONAL_ENTRY_POINTS(AC_CL_MEMBER)
#undef AC_CL_MEMBER
    char library[256] = { 0 };
};

ClApi g_api;
std::once_flag g_api_once;
AcResult g_api_result = AC_ERR_DRIVER_NOT_FOUND;
char g_api_error[1536] = { 0 };

void api_append(size_t* n, const char* fmt, const char* a, const char* b) {
    if (*n + 1 >= sizeof(g_api_error)) return;
    const int w = snprintf(g_api_error + *n, sizeof(g_api_error) - *n, fmt, a, b);
    if (w > 0) *n = (*n + (size_t)w < sizeof(g_api_error)) ? *n + (size_t)w : sizeof(g_api_error) - 1;
}

void api_load() {
    /* Where the driver comes from, in this order:
       1. AMEVA_OPENCL_LIBRARY=<path>: that library and nothing else;
       2. Android: the vendor's libOpenCL.so through libvndksupport's android_load_sphal_library, the way the system's
          Vulkan loader opens the vendor Vulkan driver. The driver and its dependencies then live in the vendor ("sphal")
          linker namespace and the process keeps its own libraries; no LD_LIBRARY_PATH is involved (putting
          /system/lib64 there to make /vendor reachable replaces the libc++_shared.so of a Termux process: Galaxy A53,
          2026-10-07, "cannot locate symbol ...filesystem..." in the llama bundle);
       3. the dynamic linker's own search: vendor paths (reachable only when LD_LIBRARY_PATH names their directories)
          and libOpenCL.so of a Linux system. */
    const char* forced = getenv("AMEVA_OPENCL_LIBRARY");
    const char* const candidates[] = { "/vendor/lib64/libOpenCL.so", "/vendor/lib64/egl/libGLES_mali.so",
                                       "/system/vendor/lib64/libOpenCL.so", "/vendor/lib64/libmali.so",
                                       "libOpenCL.so", "libOpenCL.so.1", nullptr };
    size_t n = 0;
    void* lib = nullptr;
    if (forced && *forced) {
        lib = dlopen(forced, RTLD_NOW | RTLD_LOCAL);
        if (!lib) api_append(&n, " [%s: %s]", forced, dlerror());
        else snprintf(g_api.library, sizeof(g_api.library), "%s", forced);
    } else {
        if (void* vndk = dlopen("libvndksupport.so", RTLD_NOW | RTLD_LOCAL)) {
            typedef void* (*load_sphal_fn)(const char*, int);
            load_sphal_fn load_sphal = nullptr;
            void* sym = dlsym(vndk, "android_load_sphal_library");
            if (sym) memcpy(&load_sphal, &sym, sizeof(sym));
            if (load_sphal) {
                lib = load_sphal("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
                if (lib) {
                    snprintf(g_api.library, sizeof(g_api.library), "libOpenCL.so (vendor namespace)");
                } else {
                    const char* e = dlerror();
                    api_append(&n, " [%s: %.160s]", "android_load_sphal_library(libOpenCL.so)", e ? e : "?");
                }
            }
        }
        for (int i = 0; candidates[i] && !lib; ++i) {
            lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
            if (!lib) {
                const char* e = dlerror();
                api_append(&n, " [%s: %.160s]", candidates[i], e ? e : "?");
            } else {
                snprintf(g_api.library, sizeof(g_api.library), "%s", candidates[i]);
            }
        }
    }
    if (!lib) {
        g_api_result = AC_ERR_DRIVER_NOT_FOUND;
        return;
    }
    bool missing = false;
#define AC_CL_RESOLVE(name)                                                                                           \
    {                                                                                                                 \
        void* sym = dlsym(lib, #name);                                                                                \
        if (!sym) {                                                                                                   \
            api_append(&n, " [missing %s in %s]", #name, g_api.library);                                              \
            missing = true;                                                                                           \
        } else {                                                                                                      \
            memcpy(&g_api.name, &sym, sizeof(sym));                                                                   \
        }                                                                                                             \
    }
    AC_CL_ENTRY_POINTS(AC_CL_RESOLVE)
#undef AC_CL_RESOLVE
#define AC_CL_RESOLVE_OPTIONAL(name)                                                                                  \
    {                                                                                                                 \
        void* sym = dlsym(lib, #name);                                                                                \
        if (sym) memcpy(&g_api.name, &sym, sizeof(sym));                                                              \
    }
    AC_CL_OPTIONAL_ENTRY_POINTS(AC_CL_RESOLVE_OPTIONAL)
#undef AC_CL_RESOLVE_OPTIONAL
    g_api_result = missing ? AC_ERR_DRIVER_SYMBOL_MISSING : AC_SUCCESS;
}

AcResult api() {
    std::call_once(g_api_once, api_load);
    if (g_api_result == AC_ERR_DRIVER_NOT_FOUND) {
        fprintf(stderr,
                "ac_runtime(opencl): no OpenCL driver could be opened:%s. AMEVA_OPENCL_LIBRARY=<path> selects the "
                "driver library.\n",
                g_api_error);
    } else if (g_api_result != AC_SUCCESS) {
        fprintf(stderr, "ac_runtime(opencl): the OpenCL driver %s lacks entry points this backend needs:%s\n",
                g_api.library, g_api_error);
    }
    return g_api_result;
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

std::string device_string(cl_device_id dev, cl_device_info what) {
    size_t sz = 0;
    if (g_api.clGetDeviceInfo(dev, what, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) return std::string();
    std::string s(sz, '\0');
    if (g_api.clGetDeviceInfo(dev, what, sz, &s[0], nullptr) != CL_SUCCESS) return std::string();
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

bool has_extension(const std::string& exts, const char* name) {
    const size_t len = strlen(name);
    for (size_t pos = exts.find(name); pos != std::string::npos; pos = exts.find(name, pos + 1)) {
        const bool start = pos == 0 || exts[pos - 1] == ' ';
        const bool end = pos + len == exts.size() || exts[pos + len] == ' ';
        if (start && end) return true;
    }
    return false;
}

void sha256_hex(const std::string& data, char out[65]) {
    ac_sha256_compute_hex((const uint8_t*)data.data(), data.size(), out);
}

AcResult map_cl_error(cl_int e) {
    switch (e) {
    case CL_SUCCESS: return AC_SUCCESS;
    case CL_OUT_OF_HOST_MEMORY: return AC_ERR_OUT_OF_HOST_MEMORY;
    case CL_OUT_OF_RESOURCES:
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return AC_ERR_OUT_OF_DEVICE_MEMORY;
    case CL_INVALID_WORK_GROUP_SIZE:
    case CL_INVALID_WORK_ITEM_SIZE:
    case CL_INVALID_GLOBAL_WORK_SIZE: return AC_ERR_INVALID_WORK_GROUP_SIZE;
    case CL_INVALID_ARG_INDEX:
    case CL_INVALID_ARG_VALUE:
    case CL_INVALID_ARG_SIZE:
    case CL_INVALID_KERNEL_ARGS: return AC_ERR_KERNEL_ARG_INVALID;
    case CL_BUILD_PROGRAM_FAILURE:
    case CL_INVALID_PROGRAM_EXECUTABLE: return AC_ERR_KERNEL_BUILD_FAILED;
    default: return AC_ERR_DRIVER_CRASH_OR_LOST;
    }
}

} // namespace

/* ------------------------------------------------------------------------------------------------------------------ */
/* Objects                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

struct AcClContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::mutex mutex; /* the queue and kernel arguments (set and enqueued together) */
    std::string name, driver, extensions;
    uint64_t max_alloc = 0, local_mem = 0;
    size_t max_wg = 0, max_items[3] = { 0, 0, 0 };
    uint32_t subgroup_size = 0;
    uint32_t integer_dot = 0, integer_dot_accelerated = 0;
    bool wait_spin = true; /* AMEVA_CL_WAIT, see ac_cl_event_wait */
    /* program binaries by sha256(device, driver, options, entry point, source) */
    std::mutex cache_mutex;
    std::unordered_map<std::string, std::vector<unsigned char>> binaries;
};

struct AcClBuffer {
    AcClContext* ctx = nullptr;
    void* ptr = nullptr;
    size_t size = 0;
};

struct AcClKernel {
    AcClContext* ctx = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    AcDim3 local = { 1, 1, 1 };
    uint32_t binding_count = 0;
    size_t push_size = 0;
    bool takes_ranges = false; /* one more argument: the byte size of every bound range (binding_count uints) */
    std::string name;          /* metadata kernel_name, for messages and the dispatch trace */
    char identity[65] = { 0 };
    std::atomic<int> refs{ 1 }; /* the owner plus recorded, not yet enqueued dispatches */
};

struct AcClEvent {
    AcClContext* ctx = nullptr;
    cl_event marker = nullptr;
    cl_event first = nullptr, last = nullptr; /* first and last kernel of the batch, when it recorded timestamps */
    uint64_t submit_ns = 0, done_ns = 0;
};

namespace {

void kernel_release(AcClKernel* k) {
    if (k->refs.fetch_sub(1) != 1) return;
    if (k->kernel) g_api.clReleaseKernel(k->kernel);
    if (k->program) g_api.clReleaseProgram(k->program);
    delete k;
}

struct PendingDispatch {
    AcClKernel* kernel;
    void* args[16];
    uint32_t ranges[16];
    unsigned char push[128];
    size_t global[3];
    size_t local[3];
};

} // namespace

struct AcClBatch {
    AcClContext* ctx = nullptr;
    uint32_t max_dispatches = 0, flags = 0;
    enum State { RECORDING, SUBMITTED, ABORTED } state = RECORDING;
    std::vector<PendingDispatch> pending;
    std::vector<cl_event> kernel_events; /* every enqueued kernel, when the batch records timestamps */
    std::vector<std::string> trace_names; /* kernel name and work-items per event, when AMEVA_CL_TRACE is set */
    cl_event marker = nullptr;
    uint32_t dispatches = 0, barriers = 0, flushes = 0;
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Context                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

AcResult ac_cl_context_create(AcClContext** out_ctx) {
    *out_ctx = nullptr;
    AcResult r = api();
    if (r != AC_SUCCESS) return r;
    cl_uint np = 0;
    cl_int e = g_api.clGetPlatformIDs(0, nullptr, &np);
    if (e != CL_SUCCESS || np == 0) {
        fprintf(stderr, "ac_runtime(opencl): %s has no platform (clGetPlatformIDs %d, %u platforms)\n", g_api.library,
                (int)e, (unsigned)np);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }
    std::vector<cl_platform_id> plats(np);
    g_api.clGetPlatformIDs(np, plats.data(), nullptr);
    AcClContext* c = new AcClContext();
    if (const char* w = getenv("AMEVA_CL_WAIT")) {
        if (!strcmp(w, "spin")) c->wait_spin = true;
        else if (!strcmp(w, "sleep")) c->wait_spin = false;
        else {
            fprintf(stderr, "ac_runtime(opencl): AMEVA_CL_WAIT must be spin or sleep (got \"%s\")\n", w);
            delete c;
            return AC_ERR_INVALID_ARGUMENT;
        }
    }
    for (cl_platform_id p : plats) {
        cl_device_id d = nullptr;
        if (g_api.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, nullptr) == CL_SUCCESS && d) {
            c->platform = p;
            c->device = d;
            break;
        }
    }
    if (!c->device) {
        delete c;
        fprintf(stderr, "ac_runtime(opencl): no GPU device on the %u platform(s) of %s\n", (unsigned)np, g_api.library);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }
    cl_device_svm_capabilities svm = 0;
    g_api.clGetDeviceInfo(c->device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);
    c->name = device_string(c->device, CL_DEVICE_NAME);
    // What identifies the driver build differs by vendor: Mali reports CL_DRIVER_VERSION "3.0" for every DDK and the
    // DDK build in CL_DEVICE_VERSION ("OpenCL 3.0 v1.r32p1-01bet2-...<hash>"); Adreno reports the compiler build in
    // CL_DRIVER_VERSION. Both strings go into the identity the program cache and its users key on.
    c->driver = device_string(c->device, CL_DRIVER_VERSION) + " / " + device_string(c->device, CL_DEVICE_VERSION);
    c->extensions = device_string(c->device, CL_DEVICE_EXTENSIONS);
    if (!(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        fprintf(stderr, "ac_runtime(opencl): %s has no coarse-grained SVM buffers (OpenCL 2.0+ required)\n",
                c->name.c_str());
        delete c;
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }
    cl_ulong ul = 0;
    g_api.clGetDeviceInfo(c->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(ul), &ul, nullptr);
    c->max_alloc = ul;
    g_api.clGetDeviceInfo(c->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, nullptr);
    c->local_mem = ul;
    g_api.clGetDeviceInfo(c->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(c->max_wg), &c->max_wg, nullptr);
    g_api.clGetDeviceInfo(c->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(c->max_items), c->max_items, nullptr);
    c->integer_dot = (has_extension(c->extensions, "cl_arm_integer_dot_product_accumulate_int8") ||
                      has_extension(c->extensions, "cl_khr_integer_dot_product")) ? 1u : 0u;
#ifdef CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR
    if (has_extension(c->extensions, "cl_khr_integer_dot_product")) {
        cl_device_integer_dot_product_acceleration_properties_khr acc;
        memset(&acc, 0, sizeof(acc));
        if (g_api.clGetDeviceInfo(c->device, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR,
                                  sizeof(acc), &acc, nullptr) == CL_SUCCESS)
            c->integer_dot_accelerated = acc.signed_accelerated ? 1u : 0u;
    }
#endif
    const cl_context_properties cprops[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)c->platform, 0 };
    c->context = g_api.clCreateContext(cprops, 1, &c->device, nullptr, nullptr, &e);
    if (e != CL_SUCCESS || !c->context) {
        fprintf(stderr, "ac_runtime(opencl): clCreateContext failed (%d) on %s\n", (int)e, c->name.c_str());
        delete c;
        return AC_ERR_DRIVER_INIT_FAILED;
    }
    const cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    c->queue = g_api.clCreateCommandQueueWithProperties(c->context, c->device, qprops, &e);
    if (e != CL_SUCCESS || !c->queue) {
        fprintf(stderr, "ac_runtime(opencl): clCreateCommandQueueWithProperties failed (%d) on %s\n", (int)e,
                c->name.c_str());
        g_api.clReleaseContext(c->context);
        delete c;
        return AC_ERR_DRIVER_INIT_FAILED;
    }
    /* subgroup size: a property of a kernel in OpenCL; asked of an empty one with a 64-item group */
    if (g_api.clGetKernelSubGroupInfo) {
        const char* src = "__kernel void ac_subgroup_probe(void) {}";
        cl_program p = g_api.clCreateProgramWithSource(c->context, 1, &src, nullptr, &e);
        if (e == CL_SUCCESS && g_api.clBuildProgram(p, 1, &c->device, "", nullptr, nullptr) == CL_SUCCESS) {
            cl_kernel k = g_api.clCreateKernel(p, "ac_subgroup_probe", &e);
            if (e == CL_SUCCESS) {
                const size_t local = 64;
                size_t sg = 0;
                if (g_api.clGetKernelSubGroupInfo(k, c->device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof(local),
                                                  &local, sizeof(sg), &sg, nullptr) == CL_SUCCESS)
                    c->subgroup_size = (uint32_t)sg;
                g_api.clReleaseKernel(k);
            }
        }
        if (p) g_api.clReleaseProgram(p);
    }
    *out_ctx = c;
    return AC_SUCCESS;
}

AcResult ac_cl_context_destroy(AcClContext* c) {
    if (!c) return AC_ERR_INVALID_CONTEXT;
    cl_int e = g_api.clFinish(c->queue);
    g_api.clReleaseCommandQueue(c->queue);
    g_api.clReleaseContext(c->context);
    delete c;
    return e == CL_SUCCESS ? AC_SUCCESS : AC_ERR_DRIVER_CRASH_OR_LOST;
}

AcResult ac_cl_context_get_device_name(const AcClContext* c, char* out_name, size_t max_len) {
    snprintf(out_name, max_len, "%s", c->name.c_str());
    return AC_SUCCESS;
}

AcResult ac_cl_context_get_limits(const AcClContext* c, AcDeviceLimits* l) {
    /* A binding is a pointer into an SVM allocation: its range is bounded by the allocation, kept below 2^31 bytes so
       that byte offsets inside one binding fit the 32-bit arithmetic of the kernels. 64-byte binding offsets keep
       every vector load of the kernels (at most 16 bytes) aligned. The workgroup count mirrors Vulkan's 65535 per
       axis so that callers split grids the same way on both backends. */
    l->max_storage_buffer_range = c->max_alloc < 0x7FFFFFFFull ? c->max_alloc : 0x7FFFFFFFull;
    l->min_storage_buffer_offset_alignment = 64;
    l->max_memory_allocation_size = c->max_alloc;
    for (int i = 0; i < 3; ++i) {
        l->max_compute_workgroup_count[i] = 65535u;
        l->max_compute_workgroup_size[i] = (uint32_t)c->max_items[i];
    }
    l->max_compute_workgroup_invocations = (uint32_t)c->max_wg;
    l->max_compute_shared_memory_size = (uint32_t)c->local_mem;
    l->max_push_constants_size = 128;
    l->subgroup_size = c->subgroup_size;
    l->timestamps_supported = 1;
    l->timestamp_period_ns = 1.0f; /* OpenCL profiling counters are in nanoseconds */
    if (l->struct_size >= offsetof(AcDeviceLimits, integer_dot_product_accelerated) + sizeof(uint32_t)) {
        l->integer_dot_product = c->integer_dot;
        l->integer_dot_product_accelerated = c->integer_dot_accelerated;
    }
    return AC_SUCCESS;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Buffers                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

AcResult ac_cl_buffer_create(AcClContext* c, size_t size, AcClBuffer** out_buf) {
    *out_buf = nullptr;
    if (size > c->max_alloc) return AC_ERR_INVALID_BUFFER_SIZE;
    void* p = g_api.clSVMAlloc(c->context, CL_MEM_READ_WRITE, size, 128);
    if (!p) return AC_ERR_OUT_OF_DEVICE_MEMORY;
    AcClBuffer* b = new AcClBuffer();
    b->ctx = c;
    b->ptr = p;
    b->size = size;
    *out_buf = b;
    return AC_SUCCESS;
}

AcResult ac_cl_buffer_destroy(AcClBuffer* b) {
    AcClContext* c = b->ctx;
    cl_int e;
    {
        /* commands already enqueued may still use the memory: wait for them before freeing it */
        std::lock_guard<std::mutex> lock(c->mutex);
        e = g_api.clFinish(c->queue);
    }
    g_api.clSVMFree(c->context, b->ptr);
    delete b;
    return e == CL_SUCCESS ? AC_SUCCESS : AC_ERR_DRIVER_CRASH_OR_LOST;
}

size_t ac_cl_buffer_size(const AcClBuffer* b) { return b->size; }
void* ac_cl_buffer_address(const AcClBuffer* b) { return b->ptr; }

AcResult ac_cl_buffer_write(AcClBuffer* b, size_t offset, const void* src, size_t n) {
    if (offset > b->size || n > b->size - offset) return AC_ERR_INVALID_BUFFER_SIZE;
    if (n == 0) return AC_SUCCESS;
    std::lock_guard<std::mutex> lock(b->ctx->mutex);
    const cl_int e = g_api.clEnqueueSVMMemcpy(b->ctx->queue, CL_TRUE, (char*)b->ptr + offset, src, n, 0, nullptr, nullptr);
    return map_cl_error(e);
}

AcResult ac_cl_buffer_read(AcClBuffer* b, size_t offset, void* dst, size_t n) {
    if (offset > b->size || n > b->size - offset) return AC_ERR_INVALID_BUFFER_SIZE;
    if (n == 0) return AC_SUCCESS;
    std::lock_guard<std::mutex> lock(b->ctx->mutex);
    const cl_int e = g_api.clEnqueueSVMMemcpy(b->ctx->queue, CL_TRUE, dst, (const char*)b->ptr + offset, n, 0, nullptr, nullptr);
    return map_cl_error(e);
}

AcResult ac_cl_buffer_fill(AcClBuffer* b, size_t offset, size_t n, uint8_t value) {
    if (offset > b->size || n > b->size - offset) return AC_ERR_INVALID_BUFFER_SIZE;
    if (n == 0) return AC_SUCCESS;
    std::lock_guard<std::mutex> lock(b->ctx->mutex);
    cl_event ev = nullptr;
    cl_int e = g_api.clEnqueueSVMMemFill(b->ctx->queue, (char*)b->ptr + offset, &value, 1, n, 0, nullptr, &ev);
    if (e == CL_SUCCESS) e = g_api.clWaitForEvents(1, &ev);
    if (ev) g_api.clReleaseEvent(ev);
    return map_cl_error(e);
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Kernels                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

namespace {

cl_int build(AcClContext* c, cl_program p, const char* options, const char* name, bool report) {
    const cl_int e = g_api.clBuildProgram(p, 1, &c->device, options ? options : "", nullptr, nullptr);
    if (e != CL_SUCCESS && report) {
        size_t sz = 0;
        g_api.clGetProgramBuildInfo(p, c->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::string log(sz, '\0');
        if (sz) g_api.clGetProgramBuildInfo(p, c->device, CL_PROGRAM_BUILD_LOG, sz, &log[0], nullptr);
        fprintf(stderr, "ac_runtime(opencl): building kernel '%s' failed (%d), options '%s':\n%s\n", name, (int)e,
                options ? options : "", log.c_str());
    }
    return e;
}

} // namespace

AcResult ac_cl_kernel_create(AcClContext* c, const char* source, size_t source_size, const char* options,
                             const AcKernelRuntimeMetadata* m, bool takes_ranges, AcClKernel** out_kernel) {
    *out_kernel = nullptr;
    const std::string src(source, source_size ? source_size : strlen(source));
    const std::string opts = options ? options : "";
    const std::string ident_data = std::string(m->entry_point) + '\0' + opts + '\0' + src;
    char ident[65], key[65];
    sha256_hex(ident_data, ident);
    sha256_hex(c->name + '\0' + c->driver + '\0' + ident_data, key);

    cl_program prog = nullptr;
    cl_int e = CL_SUCCESS;
    {
        std::vector<unsigned char> bin;
        {
            std::lock_guard<std::mutex> lock(c->cache_mutex);
            auto it = c->binaries.find(key);
            if (it != c->binaries.end()) bin = it->second;
        }
        if (!bin.empty()) {
            const unsigned char* bp = bin.data();
            const size_t bs = bin.size();
            cl_int st = CL_SUCCESS;
            prog = g_api.clCreateProgramWithBinary(c->context, 1, &c->device, &bs, &bp, &st, &e);
            if (e != CL_SUCCESS || st != CL_SUCCESS || build(c, prog, opts.c_str(), m->kernel_name, false) != CL_SUCCESS) {
                /* a stale or foreign binary: drop it and compile the source, which is what the binary was made of */
                if (prog) g_api.clReleaseProgram(prog);
                prog = nullptr;
                std::lock_guard<std::mutex> lock(c->cache_mutex);
                c->binaries.erase(key);
            }
        }
    }
    if (!prog) {
        const char* s = src.c_str();
        const size_t l = src.size();
        prog = g_api.clCreateProgramWithSource(c->context, 1, &s, &l, &e);
        if (e != CL_SUCCESS || !prog) return map_cl_error(e);
        if (build(c, prog, opts.c_str(), m->kernel_name, true) != CL_SUCCESS) {
            g_api.clReleaseProgram(prog);
            return AC_ERR_KERNEL_BUILD_FAILED;
        }
        size_t bs = 0;
        if (g_api.clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bs), &bs, nullptr) == CL_SUCCESS && bs > 0) {
            std::vector<unsigned char> bin(bs);
            unsigned char* bp = bin.data();
            if (g_api.clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr) == CL_SUCCESS) {
                std::lock_guard<std::mutex> lock(c->cache_mutex);
                c->binaries[key] = std::move(bin);
            }
        }
    }
    cl_kernel k = g_api.clCreateKernel(prog, m->entry_point, &e);
    if (e != CL_SUCCESS || !k) {
        fprintf(stderr, "ac_runtime(opencl): kernel '%s' has no entry point '%s' (%d)\n", m->kernel_name,
                m->entry_point, (int)e);
        g_api.clReleaseProgram(prog);
        return AC_ERR_KERNEL_NOT_FOUND;
    }
    cl_uint nargs = 0;
    g_api.clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr);
    const cl_uint want = m->binding_count + (m->push_constant_size > 0 ? 1u : 0u) + (takes_ranges ? 1u : 0u);
    if (nargs != want) {
        fprintf(stderr, "ac_runtime(opencl): kernel '%s' takes %u arguments; its description asks for %u (%u bindings%s%s)\n",
                m->kernel_name, (unsigned)nargs, (unsigned)want, (unsigned)m->binding_count,
                m->push_constant_size > 0 ? ", the push-constant struct" : "",
                takes_ranges ? ", the binding ranges" : "");
        g_api.clReleaseKernel(k);
        g_api.clReleaseProgram(prog);
        return AC_ERR_KERNEL_ARG_INVALID;
    }
    size_t kwg = 0;
    g_api.clGetKernelWorkGroupInfo(k, c->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kwg), &kwg, nullptr);
    const size_t local = (size_t)m->local_size.x * m->local_size.y * m->local_size.z;
    if (kwg != 0 && local > kwg) {
        fprintf(stderr, "ac_runtime(opencl): kernel '%s' allows %zu work-items per group; its metadata asks for %zu\n",
                m->kernel_name, kwg, local);
        g_api.clReleaseKernel(k);
        g_api.clReleaseProgram(prog);
        return AC_ERR_INVALID_WORK_GROUP_SIZE;
    }
    AcClKernel* ak = new AcClKernel();
    ak->ctx = c;
    ak->program = prog;
    ak->kernel = k;
    ak->local = m->local_size;
    ak->binding_count = m->binding_count;
    ak->push_size = m->push_constant_size;
    ak->takes_ranges = takes_ranges;
    ak->name = m->kernel_name ? m->kernel_name : "?";
    memcpy(ak->identity, ident, sizeof(ident));
    *out_kernel = ak;
    return AC_SUCCESS;
}

AcResult ac_cl_kernel_destroy(AcClKernel* k) {
    kernel_release(k);
    return AC_SUCCESS;
}

AcDim3 ac_cl_kernel_local_size(const AcClKernel* k) { return k->local; }
size_t ac_cl_kernel_push_constant_size(const AcClKernel* k) { return k->push_size; }
const char* ac_cl_kernel_identity(const AcClKernel* k) { return k->identity; }

/* ------------------------------------------------------------------------------------------------------------------ */
/* Batches                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

namespace {

bool batch_timestamps(const AcClBatch* b) {
    return (b->flags & (AC_BATCH_FLAG_TIMESTAMPS | AC_BATCH_FLAG_DISPATCH_TIMESTAMPS)) != 0;
}

/* AMEVA_CL_TRACE=<file>: every batch that records dispatch timestamps appends one line per dispatch when its times
   are read: batch number, dispatch number, kernel, work-items, and the driver's four profiling counters in
   microseconds from the first dispatch's QUEUED (queued, submitted, started, ended). The per-dispatch time of the
   C ABI is end minus the previous end; the trace shows what of it the kernel ran and what the queue waited. */
const char* trace_path() {
    static const char* p = [] {
        const char* e = getenv("AMEVA_CL_TRACE");
        return (e && *e) ? e : (const char*)nullptr;
    }();
    return p;
}

void pending_drop(AcClBatch* b) {
    for (PendingDispatch& d : b->pending) kernel_release(d.kernel);
    b->pending.clear();
}

/* enqueue every recorded dispatch in recording order; caller holds ctx->mutex */
AcResult pending_enqueue(AcClBatch* b) {
    const bool ts = batch_timestamps(b);
    AcResult r = AC_SUCCESS;
    size_t i = 0;
    for (; i < b->pending.size(); ++i) {
        PendingDispatch& d = b->pending[i];
        AcClKernel* k = d.kernel;
        cl_int e = CL_SUCCESS;
        for (uint32_t a = 0; a < k->binding_count && e == CL_SUCCESS; ++a)
            e = g_api.clSetKernelArgSVMPointer(k->kernel, a, d.args[a]);
        cl_uint next = k->binding_count;
        if (e == CL_SUCCESS && k->push_size > 0) e = g_api.clSetKernelArg(k->kernel, next++, k->push_size, d.push);
        if (e == CL_SUCCESS && k->takes_ranges)
            e = g_api.clSetKernelArg(k->kernel, next, sizeof(uint32_t) * k->binding_count, d.ranges);
        cl_event ev = nullptr;
        if (e == CL_SUCCESS)
            e = g_api.clEnqueueNDRangeKernel(b->ctx->queue, k->kernel, 3, nullptr, d.global, d.local, 0, nullptr,
                                             ts ? &ev : nullptr);
        if (e != CL_SUCCESS) {
            fprintf(stderr, "ac_runtime(opencl): enqueue of a dispatch failed (%d), global %zu,%zu,%zu local %zu,%zu,%zu\n",
                    (int)e, d.global[0], d.global[1], d.global[2], d.local[0], d.local[1], d.local[2]);
            r = map_cl_error(e);
            break;
        }
        if (ev) {
            b->kernel_events.push_back(ev);
            if (trace_path())
                b->trace_names.push_back(k->name + " " + std::to_string(d.global[0] * d.global[1] * d.global[2]));
        }
        kernel_release(k);
    }
    /* on failure dispatch i and the rest are not enqueued, and the batch is aborted */
    for (size_t j = i; j < b->pending.size(); ++j) kernel_release(b->pending[j].kernel);
    b->pending.clear();
    if (r != AC_SUCCESS) b->state = AcClBatch::ABORTED;
    return r;
}

} // namespace

AcResult ac_cl_batch_begin(AcClContext* c, uint32_t max_dispatches, uint32_t flags, AcClBatch** out_batch) {
    AcClBatch* b = new AcClBatch();
    b->ctx = c;
    b->max_dispatches = max_dispatches;
    b->flags = flags;
    *out_batch = b;
    return AC_SUCCESS;
}

AcResult ac_cl_batch_dispatch(AcClBatch* b, AcClKernel* k, AcClBuffer* const* bufs, const uint64_t* offsets,
                              const uint64_t* ranges, uint32_t n, const void* push, size_t push_size, AcDim3 grid) {
    if (b->state != AcClBatch::RECORDING) return AC_ERR_INVALID_ARGUMENT;
    if (k->ctx != b->ctx) return AC_ERR_CONTEXT_MISMATCH;
    if (n != k->binding_count) return AC_ERR_KERNEL_ARG_INVALID;
    if (push_size > k->push_size) return AC_ERR_INVALID_ARGUMENT;
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return AC_ERR_INVALID_WORK_GROUP_SIZE;
    if (b->dispatches >= b->max_dispatches) return AC_ERR_INVALID_ARGUMENT;
    PendingDispatch d;
    memset(&d, 0, sizeof(d));
    for (uint32_t i = 0; i < n; ++i) {
        const AcClBuffer* buf = bufs[i];
        if (buf->ctx != b->ctx) return AC_ERR_CONTEXT_MISMATCH;
        if (ranges[i] == 0 || offsets[i] > buf->size || ranges[i] > buf->size - offsets[i]) return AC_ERR_INVALID_BUFFER_SIZE;
        if (offsets[i] % 64 != 0) return AC_ERR_BUFFER_ALIGNMENT_INVALID;
        d.args[i] = (char*)buf->ptr + offsets[i];
        d.ranges[i] = ranges[i] > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)ranges[i];
    }
    if (push_size) memcpy(d.push, push, push_size); /* the rest of the declared struct stays zero */
    const uint32_t lsz[3] = { k->local.x, k->local.y, k->local.z }, gsz[3] = { grid.x, grid.y, grid.z };
    for (int a = 0; a < 3; ++a) {
        d.local[a] = lsz[a];
        d.global[a] = (size_t)gsz[a] * lsz[a];
    }
    k->refs.fetch_add(1);
    d.kernel = k;
    b->pending.push_back(d);
    b->dispatches++;
    return AC_SUCCESS;
}

AcResult ac_cl_batch_barrier(AcClBatch* b) {
    if (b->state != AcClBatch::RECORDING) return AC_ERR_INVALID_ARGUMENT;
    b->barriers++; /* the in-order queue already orders every dispatch after the previous one */
    return AC_SUCCESS;
}

AcResult ac_cl_batch_flush(AcClBatch* b) {
    if (b->state != AcClBatch::RECORDING) return AC_ERR_INVALID_ARGUMENT;
    if (b->pending.empty()) return AC_SUCCESS;
    std::lock_guard<std::mutex> lock(b->ctx->mutex);
    AcResult r = pending_enqueue(b);
    if (r != AC_SUCCESS) return r;
    b->flushes++;
    return map_cl_error(g_api.clFlush(b->ctx->queue));
}

AcResult ac_cl_batch_submit(AcClBatch* b, AcClEvent** out_event) {
    if (out_event) *out_event = nullptr;
    if (b->state != AcClBatch::RECORDING) return AC_ERR_INVALID_ARGUMENT;
    AcClContext* c = b->ctx;
    std::lock_guard<std::mutex> lock(c->mutex);
    AcResult r = pending_enqueue(b);
    if (r != AC_SUCCESS) return r;
    cl_int e = g_api.clEnqueueMarkerWithWaitList(c->queue, 0, nullptr, &b->marker);
    if (e == CL_SUCCESS) e = g_api.clFlush(c->queue);
    if (e != CL_SUCCESS) {
        b->state = AcClBatch::ABORTED;
        return map_cl_error(e);
    }
    b->state = AcClBatch::SUBMITTED;
    if (out_event) {
        AcClEvent* ev = new AcClEvent();
        ev->ctx = c;
        ev->marker = b->marker;
        g_api.clRetainEvent(ev->marker);
        if (!b->kernel_events.empty()) {
            ev->first = b->kernel_events.front();
            ev->last = b->kernel_events.back();
            g_api.clRetainEvent(ev->first);
            g_api.clRetainEvent(ev->last);
        }
        ev->submit_ns = now_ns();
        *out_event = ev;
    }
    return AC_SUCCESS;
}

AcResult ac_cl_batch_abort(AcClBatch* b) {
    pending_drop(b);
    if (b->state == AcClBatch::RECORDING) b->state = AcClBatch::ABORTED;
    return AC_SUCCESS;
}

void ac_cl_batch_destroy(AcClBatch* b) {
    pending_drop(b);
    for (cl_event e : b->kernel_events) g_api.clReleaseEvent(e);
    if (b->marker) g_api.clReleaseEvent(b->marker);
    delete b;
}

AcResult ac_cl_batch_get_dispatch_times(const AcClBatch* b, uint64_t* out_ns, uint32_t capacity, uint32_t* out_count) {
    if (!(b->flags & AC_BATCH_FLAG_DISPATCH_TIMESTAMPS)) return AC_ERR_FEATURE_NOT_SUPPORTED;
    if (b->state != AcClBatch::SUBMITTED || !b->marker) return AC_ERR_DEVICE_BUSY;
    cl_int st = CL_QUEUED;
    g_api.clGetEventInfo(b->marker, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr);
    if (st != CL_COMPLETE) return st < 0 ? AC_ERR_DRIVER_CRASH_OR_LOST : AC_ERR_DEVICE_BUSY;
    const uint32_t n = (uint32_t)b->kernel_events.size();
    if (capacity < n || (!out_ns && n)) return AC_ERR_INVALID_BUFFER_SIZE;
    cl_ulong prev_end = 0;
    for (uint32_t i = 0; i < n; ++i) {
        cl_ulong s = 0, en = 0;
        if (g_api.clGetEventProfilingInfo(b->kernel_events[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr) != CL_SUCCESS ||
            g_api.clGetEventProfilingInfo(b->kernel_events[i], CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr) != CL_SUCCESS)
            return AC_ERR_FEATURE_NOT_SUPPORTED;
        const cl_ulong from = i == 0 ? s : prev_end;
        out_ns[i] = en > from ? (uint64_t)(en - from) : 0;
        prev_end = en;
    }
    if (out_count) *out_count = n;
    if (trace_path() && n && b->trace_names.size() == n) {
        static std::atomic<uint32_t> batch_seq{ 0 };
        const uint32_t seq = batch_seq.fetch_add(1);
        if (FILE* f = fopen(trace_path(), "a")) {
            cl_ulong t0 = 0;
            g_api.clGetEventProfilingInfo(b->kernel_events[0], CL_PROFILING_COMMAND_QUEUED, sizeof(t0), &t0, nullptr);
            for (uint32_t i = 0; i < n; ++i) {
                cl_ulong t[4] = { 0, 0, 0, 0 };
                const cl_profiling_info what[4] = { CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT,
                                                    CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END };
                for (int j = 0; j < 4; ++j)
                    g_api.clGetEventProfilingInfo(b->kernel_events[i], what[j], sizeof(t[j]), &t[j], nullptr);
                fprintf(f, "%u %u %s %.1f %.1f %.1f %.1f\n", (unsigned)seq, (unsigned)i, b->trace_names[i].c_str(),
                        (double)(int64_t)(t[0] - t0) / 1e3, (double)(int64_t)(t[1] - t0) / 1e3,
                        (double)(int64_t)(t[2] - t0) / 1e3, (double)(int64_t)(t[3] - t0) / 1e3);
            }
            fclose(f);
        } else {
            fprintf(stderr, "ac_runtime(opencl): AMEVA_CL_TRACE file %s could not be opened\n", trace_path());
        }
    }
    return AC_SUCCESS;
}

void ac_cl_batch_get_stats(const AcClBatch* b, AcBatchStats* s) {
    s->dispatches = b->dispatches;
    s->barriers = b->barriers;
    s->elided_barriers = 0;
    s->flushes = b->flushes;
    s->descriptor_cache_hits = 0;
    s->descriptor_cache_misses = 0;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Events                                                                                                             */
/* ------------------------------------------------------------------------------------------------------------------ */

/* How the waiting thread passes the time until its batch is done (AMEVA_CL_WAIT, read when the context is created):
     spin (default)  it reads the batch's status in a loop without sleeping: one CPU core stays busy while the GPU works
     sleep           it sleeps 50 us between reads, or blocks in clWaitForEvents when there is no timeout: no CPU use
   Why spin: with every CPU core idle, the same kernels on the same data took up to twice as long on Galaxy A53
   (Mali-G68, DDK r32p1; 2026-10-07, Qwen2.5-0.5B token generation, AMEVA_CL_TRACE): the one-row product q5_0
   4864x896 520 us on average against 326 us, the output projection 26.9 ms against 12.4 ms. The slow stretches last
   tens of milliseconds, there is no time between the kernels, and the GPU clock stays at its maximum. One busy
   core of either cluster removes them (kernel time per token 75 ms -> 46 ms, 11.9 -> 19.2 tokens/s; the same
   plugin on Vulkan 14.5 -> 15.0). What below the driver slows down is not visible from an application; holding a
   core awake is the means an application has. The price is that core's power for the duration of every batch. */
AcResult ac_cl_event_wait(AcClEvent* ev, uint64_t timeout_ns) {
    if (ev->done_ns) return AC_SUCCESS;
    cl_int st = CL_QUEUED;
    if (ev->ctx->wait_spin) {
        const uint64_t until = timeout_ns == UINT64_MAX ? UINT64_MAX : now_ns() + timeout_ns;
        for (;;) {
            if (g_api.clGetEventInfo(ev->marker, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr) != CL_SUCCESS)
                return AC_ERR_DRIVER_CRASH_OR_LOST;
            if (st == CL_COMPLETE) break;
            if (st < 0) {
                fprintf(stderr, "ac_runtime(opencl): batch failed on the device (status %d)\n", (int)st);
                return AC_ERR_DRIVER_CRASH_OR_LOST;
            }
            /* about 20 us of clock reads (no system call) between two reads of the status */
            const uint64_t t = now_ns();
            if (t >= until) return AC_ERR_LAUNCH_TIMEOUT;
            while (now_ns() - t < 20000) {}
        }
    } else if (timeout_ns == UINT64_MAX) {
        const cl_int e = g_api.clWaitForEvents(1, &ev->marker);
        g_api.clGetEventInfo(ev->marker, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr);
        if (e != CL_SUCCESS || st != CL_COMPLETE) {
            fprintf(stderr, "ac_runtime(opencl): batch failed on the device (wait %d, status %d)\n", (int)e, (int)st);
            return AC_ERR_DRIVER_CRASH_OR_LOST;
        }
    } else {
        const uint64_t until = now_ns() + timeout_ns;
        for (;;) {
            if (g_api.clGetEventInfo(ev->marker, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr) != CL_SUCCESS)
                return AC_ERR_DRIVER_CRASH_OR_LOST;
            if (st == CL_COMPLETE) break;
            if (st < 0) {
                fprintf(stderr, "ac_runtime(opencl): batch failed on the device (status %d)\n", (int)st);
                return AC_ERR_DRIVER_CRASH_OR_LOST;
            }
            if (now_ns() >= until) return AC_ERR_LAUNCH_TIMEOUT;
            struct timespec ts = { 0, 50000 };
            nanosleep(&ts, nullptr);
        }
    }
    ev->done_ns = now_ns();
    return AC_SUCCESS;
}

AcResult ac_cl_event_get_timing(AcClEvent* ev, uint64_t* out_wall, uint64_t* out_gpu) {
    if (out_wall) *out_wall = 0;
    if (out_gpu) *out_gpu = 0;
    if (!ev->done_ns) return AC_ERR_DEVICE_BUSY;
    if (out_wall) *out_wall = ev->done_ns - ev->submit_ns;
    if (out_gpu && ev->first && ev->last) {
        cl_ulong s = 0, en = 0;
        if (g_api.clGetEventProfilingInfo(ev->first, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr) == CL_SUCCESS &&
            g_api.clGetEventProfilingInfo(ev->last, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr) == CL_SUCCESS &&
            en > s)
            *out_gpu = (uint64_t)(en - s);
    }
    return AC_SUCCESS;
}

AcResult ac_cl_event_destroy(AcClEvent* ev) {
    if (ev->marker) g_api.clReleaseEvent(ev->marker);
    if (ev->first) g_api.clReleaseEvent(ev->first);
    if (ev->last) g_api.clReleaseEvent(ev->last);
    delete ev;
    return AC_SUCCESS;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Program binary cache                                                                                               */
/* Format: "ACCLBIN1", u32 length + device name, u32 length + driver version, u32 count, then per entry the 64-char  */
/* key and u64 length + the binary. Data of another device or driver is rejected, like Vulkan pipeline cache data.   */
/* ------------------------------------------------------------------------------------------------------------------ */

namespace {

const char kCacheMagic[8] = { 'A', 'C', 'C', 'L', 'B', 'I', 'N', '1' };

void put_u32(std::vector<unsigned char>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back((unsigned char)(x >> (8 * i)));
}
void put_u64(std::vector<unsigned char>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back((unsigned char)(x >> (8 * i)));
}
bool get_u32(const unsigned char*& p, const unsigned char* end, uint32_t* x) {
    if (end - p < 4) return false;
    *x = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    p += 4;
    return true;
}
bool get_u64(const unsigned char*& p, const unsigned char* end, uint64_t* x) {
    if (end - p < 8) return false;
    *x = 0;
    for (int i = 0; i < 8; ++i) *x |= (uint64_t)p[i] << (8 * i);
    p += 8;
    return true;
}

std::vector<unsigned char> cache_serialize(AcClContext* c) {
    std::vector<unsigned char> v(kCacheMagic, kCacheMagic + 8);
    put_u32(v, (uint32_t)c->name.size());
    v.insert(v.end(), c->name.begin(), c->name.end());
    put_u32(v, (uint32_t)c->driver.size());
    v.insert(v.end(), c->driver.begin(), c->driver.end());
    std::lock_guard<std::mutex> lock(c->cache_mutex);
    put_u32(v, (uint32_t)c->binaries.size());
    for (const auto& kv : c->binaries) {
        v.insert(v.end(), kv.first.begin(), kv.first.end());
        put_u64(v, kv.second.size());
        v.insert(v.end(), kv.second.begin(), kv.second.end());
    }
    return v;
}

} // namespace

AcResult ac_cl_program_cache_merge(AcClContext* c, const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;
    const unsigned char* end = p + size;
    if (size < 8 || memcmp(p, kCacheMagic, 8) != 0) return AC_ERR_INVALID_ARGUMENT;
    p += 8;
    uint32_t len = 0;
    if (!get_u32(p, end, &len) || (size_t)(end - p) < len || std::string((const char*)p, len) != c->name)
        return AC_ERR_INVALID_ARGUMENT;
    p += len;
    if (!get_u32(p, end, &len) || (size_t)(end - p) < len || std::string((const char*)p, len) != c->driver)
        return AC_ERR_INVALID_ARGUMENT;
    p += len;
    uint32_t count = 0;
    if (!get_u32(p, end, &count)) return AC_ERR_INVALID_ARGUMENT;
    std::unordered_map<std::string, std::vector<unsigned char>> in;
    for (uint32_t i = 0; i < count; ++i) {
        if (end - p < 64) return AC_ERR_INVALID_ARGUMENT;
        std::string key((const char*)p, 64);
        p += 64;
        uint64_t bl = 0;
        if (!get_u64(p, end, &bl) || (uint64_t)(end - p) < bl) return AC_ERR_INVALID_ARGUMENT;
        in[key] = std::vector<unsigned char>(p, p + bl);
        p += bl;
    }
    if (p != end) return AC_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(c->cache_mutex);
    for (auto& kv : in) c->binaries.emplace(kv.first, std::move(kv.second));
    return AC_SUCCESS;
}

AcResult ac_cl_program_cache_get_data(AcClContext* c, void* data, size_t* inout_size) {
    const std::vector<unsigned char> v = cache_serialize(c);
    if (!data) {
        *inout_size = v.size();
        return AC_SUCCESS;
    }
    if (*inout_size < v.size()) return AC_ERR_INVALID_BUFFER_SIZE;
    memcpy(data, v.data(), v.size());
    *inout_size = v.size();
    return AC_SUCCESS;
}
