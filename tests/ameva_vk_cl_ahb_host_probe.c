// SPDX-License-Identifier: Apache-2.0
// ameva_vk_cl_ahb_host_probe: zero-copy sharing of Vulkan memory with OpenCL on Mali through an AHardwareBuffer, with
// the data written by the host (one round; the multi-round GPU handoff is ameva_vk_cl_gpu_handoff_probe.c).
//
// The first probe (ameva_vk_cl_interop_probe.c) showed that the Mali OpenCL driver refuses to import the host mapping
// of an ac_runtime (Vulkan) buffer with CL_IMPORT_TYPE_HOST_ARM (CL_OUT_OF_HOST_MEMORY). This probe prints what the
// Vulkan driver can export for a storage buffer (S21 r38p0, A53 r32p1: DMA_BUF is import-only, AHB is exportable).
//
// 1. Vulkan: a storage buffer whose memory is allocated exportable as an AHB, host visible and mapped.
// 2. The host writes a float pattern into the first half through the Vulkan mapping.
// 3. vkGetMemoryAndroidHardwareBufferANDROID exports the AHB; clImportMemoryARM(AHB) imports it.
// 4. An OpenCL kernel reads the first half and writes 2x + 1 into the second half; clFinish.
// 5. The host checks every value through the Vulkan mapping.
// Prints AHB_INTEROP=PASS / FAIL with the step and code. Exit status 0 only on PASS.
// Result (2026-10-06): the S21 intermittently shows 16-32 values (whole 64-byte lines) that the OpenCL kernel read
// stale, at 128 MB and at 256 MB, with or without AMEVA_PROBE_CLEAN. See docs/design/opencl-prompt-path.md (stage B).
// Build: clang -O2 -o ahb tests/ameva_vk_cl_ahb_host_probe.c -lvulkan -lOpenCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <vulkan/vulkan.h>

#ifndef CL_IMPORT_TYPE_ARM
#define CL_IMPORT_TYPE_ARM 0x40B2
#endif
#ifndef CL_IMPORT_TYPE_DMA_BUF_ARM
#define CL_IMPORT_TYPE_DMA_BUF_ARM 0x40B4
#endif
#ifndef CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM
#define CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM 0x41E2
#endif
#ifndef CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM
#define CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM SIZE_MAX
#endif
typedef intptr_t probe_import_prop;
typedef cl_mem (*clImportMemoryARM_fn)(cl_context, cl_mem_flags, const probe_import_prop*, void*, size_t, cl_int*);

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static int fail(const char* step, long code) {
    printf("AHB_INTEROP=FAIL (%s, code %ld)\n", step, code);
    return 1;
}

static const char* kSrc =
    "__kernel void twice_plus_one(__global float* a, int nhalf) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < nhalf) a[nhalf + i] = a[i] * 2.0f + 1.0f;\n"
    "}\n";

int main(int argc, char** argv) {
    const size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 16;
    const VkDeviceSize bytes = (VkDeviceSize)mb << 20;
    const size_t n = (size_t)bytes / sizeof(float), half = n / 2;

    // ---- Vulkan: instance, device with the external-memory extensions -------------------------------------------
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmabuf-probe", 1, "ameva", 1, VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    VkInstance inst;
    VkResult vr = vkCreateInstance(&ici, NULL, &inst);
    if (vr != VK_SUCCESS) return fail("vkCreateInstance", vr);
    uint32_t npd = 1;
    VkPhysicalDevice pd;
    vr = vkEnumeratePhysicalDevices(inst, &npd, &pd);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return fail("vkEnumeratePhysicalDevices", vr);
    VkPhysicalDeviceProperties pprops;
    vkGetPhysicalDeviceProperties(pd, &pprops);
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio };
    // The Mali drivers export Vulkan memory as an AHardwareBuffer (not as a dma_buf: DMA_BUF is import-only there)
    const char* exts[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                           VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
                           VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 4, exts, NULL };
    VkDevice dev;
    vr = vkCreateDevice(pd, &dci, NULL, &dev);
    if (vr != VK_SUCCESS) return fail("vkCreateDevice with the external-memory and AHB extensions", vr);
    PFN_vkGetMemoryAndroidHardwareBufferANDROID getAhb =
        (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(dev, "vkGetMemoryAndroidHardwareBufferANDROID");
    if (!getAhb) return fail("vkGetMemoryAndroidHardwareBufferANDROID not found", 0);
    PFN_vkGetMemoryFdKHR getMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    if (!getMemoryFd) return fail("vkGetMemoryFdKHR not found", 0);

    // what the driver says it can export / import for a storage buffer, per handle type
    {
        const struct { VkExternalMemoryHandleTypeFlagBits t; const char* name; } hts[] = {
            { VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, "OPAQUE_FD" },
            { VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, "DMA_BUF" },
            { VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID, "AHB" },
        };
        for (unsigned i = 0; i < sizeof(hts) / sizeof(hts[0]); ++i) {
            VkPhysicalDeviceExternalBufferInfo ebi = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, NULL, 0,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hts[i].t };
            VkExternalBufferProperties ebp = { VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES, NULL, { 0, 0, 0 } };
            vkGetPhysicalDeviceExternalBufferProperties(pd, &ebi, &ebp);
            const VkExternalMemoryFeatureFlags ft = ebp.externalMemoryProperties.externalMemoryFeatures;
            printf("external buffer %-9s features:%s%s%s exportFrom=0x%x compatible=0x%x\n", hts[i].name,
                   (ft & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) ? " DEDICATED_ONLY" : "",
                   (ft & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ? " EXPORTABLE" : "",
                   (ft & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) ? " IMPORTABLE" : "",
                   ebp.externalMemoryProperties.exportFromImportedHandleTypes,
                   ebp.externalMemoryProperties.compatibleHandleTypes);
        }
    }

    // ---- 1. exportable, host-visible storage buffer ------------------------------------------------------------
    const VkExternalMemoryHandleTypeFlagBits HT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkExternalMemoryBufferCreateInfo ebci = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, HT };
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebci, 0, bytes,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    VkBuffer buf;
    vr = vkCreateBuffer(dev, &bci, NULL, &buf);
    if (vr != VK_SUCCESS) return fail("vkCreateBuffer (external AHB)", vr);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
    }
    if (mt == UINT32_MAX) return fail("no host-visible coherent memory type for an exportable buffer", 0);
    VkMemoryDedicatedAllocateInfo dai = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, buf };
    VkExportMemoryAllocateInfo eai = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dai, HT };
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &eai, req.size, mt };
    VkDeviceMemory mem;
    vr = vkAllocateMemory(dev, &mai, NULL, &mem);
    if (vr != VK_SUCCESS) return fail("vkAllocateMemory (export AHB)", vr);
    vr = vkBindBufferMemory(dev, buf, mem, 0);
    if (vr != VK_SUCCESS) return fail("vkBindBufferMemory", vr);
    void* p = NULL;
    vr = vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p);
    if (vr != VK_SUCCESS || !p) return fail("vkMapMemory", vr);
    printf("vulkan device: %s, %zu MB exportable memory (type %u, flags 0x%x) mapped at %p\n", pprops.deviceName, mb, mt,
           mp.memoryTypes[mt].propertyFlags, p);

    // ---- 2. pattern through the Vulkan mapping -----------------------------------------------------------------
    float* f = (float*)p;
    for (size_t i = 0; i < half; ++i) f[i] = (float)(i % 1000003) * 0.5f;
    for (size_t i = half; i < n; ++i) f[i] = -1.0f;
    // Drain the CPU's write buffers before another API's GPU work reads the memory (an OpenCL enqueue on imported
    // memory is not a Vulkan queue submission).
#if defined(__aarch64__)
    // AMEVA_PROBE_CLEAN=1 also cleans every line to the point of coherency. On the S21 neither variant removed the
    // stale lines at 256 MB; the GPU handoff probe later showed the same with no host access at all, so the CPU
    // caches are not the cause there.
    if (getenv("AMEVA_PROBE_CLEAN") && atoi(getenv("AMEVA_PROBE_CLEAN"))) {
        uint64_t ctr;
        __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
        const uintptr_t line = (uintptr_t)4 << ((ctr >> 16) & 0xF); // DminLine: words -> bytes
        const double tc = now_ms();
        for (uintptr_t a = (uintptr_t)p & ~(line - 1); a < (uintptr_t)p + (uintptr_t)bytes; a += line)
            __asm__ volatile("dc cvac, %0" ::"r"(a) : "memory");
        __asm__ volatile("dsb sy" ::: "memory");
        printf("cache clean of %zu MB (line %lu B): %.3f ms\n", mb, (unsigned long)line, now_ms() - tc);
    }
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif

    // ---- 3. export as an AHardwareBuffer, import into OpenCL ----------------------------------------------------
    VkMemoryGetAndroidHardwareBufferInfoANDROID gai = { VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                        NULL, mem };
    struct AHardwareBuffer* ahb = NULL;
    double t0 = now_ms();
    vr = getAhb(dev, &gai, &ahb);
    const double t_export = now_ms() - t0;
    printf("vkGetMemoryAndroidHardwareBufferANDROID -> VkResult %d, AHardwareBuffer %p, %.3f ms\n", (int)vr, (void*)ahb,
           t_export);
    if (vr != VK_SUCCESS || !ahb) return fail("vkGetMemoryAndroidHardwareBufferANDROID", vr);
    cl_platform_id plat;
    cl_device_id cdev;
    cl_int e = clGetPlatformIDs(1, &plat, NULL);
    if (e != CL_SUCCESS) return fail("clGetPlatformIDs", e);
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cdev, NULL);
    if (e != CL_SUCCESS) return fail("clGetDeviceIDs", e);
    cl_context cctx = clCreateContext(NULL, 1, &cdev, NULL, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateContext", e);
    cl_command_queue q = clCreateCommandQueueWithProperties(cctx, cdev, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateCommandQueueWithProperties", e);
    clImportMemoryARM_fn import = (clImportMemoryARM_fn)clGetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM");
    if (!import) return fail("clImportMemoryARM not exported", 0);
    const probe_import_prop props[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM, 0 };
    t0 = now_ms();
    cl_mem m = import(cctx, CL_MEM_READ_WRITE, props, ahb, CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM, &e);
    const double t_import = now_ms() - t0;
    if (e != CL_SUCCESS || !m) return fail("clImportMemoryARM(CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM)", e);

    // ---- 4. OpenCL kernel on the shared memory -----------------------------------------------------------------
    cl_program prog = clCreateProgramWithSource(cctx, 1, &kSrc, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateProgramWithSource", e);
    e = clBuildProgram(prog, 1, &cdev, "", NULL, NULL);
    if (e != CL_SUCCESS) return fail("clBuildProgram", e);
    cl_kernel k = clCreateKernel(prog, "twice_plus_one", &e);
    if (e != CL_SUCCESS) return fail("clCreateKernel", e);
    const cl_int h = (cl_int)half;
    clSetKernelArg(k, 0, sizeof(cl_mem), &m);
    clSetKernelArg(k, 1, sizeof(cl_int), &h);
    const size_t g = (half + 63) / 64 * 64, l = 64;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        t0 = now_ms();
        e = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
        if (e != CL_SUCCESS) return fail("clEnqueueNDRangeKernel", e);
        e = clFinish(q);
        if (e != CL_SUCCESS) return fail("clFinish", e);
        const double t = now_ms() - t0;
        if (t < best) best = t;
    }

    // ---- 5. check through the Vulkan mapping -------------------------------------------------------------------
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < half; ++i) {
        if (f[half + i] != f[i] * 2.0f + 1.0f) {
            if (!bad) first_bad = i;
            ++bad;
        }
    }
    const double got_v = bad ? (double)f[half + first_bad] : 0.0, want_v = bad ? (double)(f[first_bad] * 2.0f + 1.0f) : 0.0;
    printf("export %.3f ms, import %.3f ms, kernel round trip (best of 5, %zu MB read + written) %.3f ms\n", t_export,
           t_import, mb / 2, best);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseMemObject(m);
    clReleaseCommandQueue(q);
    clReleaseContext(cctx);
    vkUnmapMemory(dev, mem);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeMemory(dev, mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    if (bad) {
        printf("AHB_INTEROP=FAIL (%zu of %zu values wrong, first at %zu: %g, want %g)\n", bad, half, first_bad, got_v,
               want_v);
        return 1;
    }
    printf("AHB_INTEROP=PASS (%zu values written by OpenCL read back through the Vulkan mapping)\n", half);
    return 0;
}
