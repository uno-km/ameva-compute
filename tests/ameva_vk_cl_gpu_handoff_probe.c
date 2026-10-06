// SPDX-License-Identifier: Apache-2.0
// ameva_vk_cl_gpu_handoff_probe: handoff between a Vulkan compute queue and an OpenCL queue on one memory, as stage B
// of docs/design/opencl-prompt-path.md would need it (Vulkan kernels write the activations, OpenCL block-integer
// kernels read them and write the product, Vulkan kernels read the product).
//
// Setup once: Vulkan buffer whose memory is exported as an AHardwareBuffer (the Mali drivers export AHB, not
// dma_buf), imported into OpenCL with clImportMemoryARM(CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM).
// Each of R rounds, with e = (i % 1000003) * 0.5 + round (a new value every round, so data left from an earlier
// round is caught):
//   1. the first half gets a[i] = e (by the produce mode below); queue fence waited
//   2. OpenCL kernel writes a[half + i] = 2 a[i] + round + 1; clFinish
//   3. Vulkan kernel (mode 1) checks a[i] and a[half + i] and writes the verdict into a separate result buffer that
//      OpenCL never sees; fence waited. In the gpu mode the host never reads or writes the shared memory.
//   4. the host counts the passes and classifies each failure by the value the check saw:
//      "OpenCL read a stale input"          OpenCL wrote 2 (e - 1) + round + 1 (the previous round's input)
//      "Vulkan read a stale OpenCL result"  the check saw 2 (e - 1) + round (the previous round's OpenCL result)
//      "Vulkan read a stale input"          the check itself saw a[i] != e
// The handoff is ordered by waiting for each queue on the host (Vulkan fence, clFinish) and nothing else; this
// measures whether that is enough on the driver pair. Prints GPU_HANDOFF=PASS / FAIL and per-round times.
//
// argv: [MB] [rounds] [produce], produce one of
//   gpu      a Vulkan kernel writes the first half (shader mode 0; the activation case of a graph)
//   cpu      the host writes the first half through the Vulkan mapping, then one Vulkan submission over zero
//            elements (the order of a ggml-ameva graph, where Vulkan kernels run before the first OpenCL MUL_MAT)
//   launder  as cpu, but the Vulkan submission reads and rewrites every value (shader mode 2)
//   staged   the host writes a staging buffer (never shared with OpenCL), vkCmdCopyBuffer copies it in
//   compute  as staged, but a Vulkan kernel (shader mode 3) copies it in
// Synchronisation variants (environment):
//   AMEVA_PROBE_SLEEP_US=n  wait n microseconds on the host between the produce fence and the OpenCL enqueue
//   AMEVA_PROBE_FOREIGN=1   queue family ownership release to / acquire from VK_QUEUE_FAMILY_FOREIGN_EXT
//   AMEVA_PROBE_REIMPORT=1  release the OpenCL import and import the AHB again every round
//   AMEVA_PROBE_MIGRATE=1   clEnqueueMigrateMemObjects(flags 0) of the import before the OpenCL kernel
//   AMEVA_PROBE_ACQUIRE=1   the OpenCL kernel starts with a device-scope acquire fence (OpenCL C 3.0)
//   AMEVA_PROBE_DUMMYMAP=1  map and unmap a 64-byte ordinary OpenCL buffer for writing before the kernel
//   AMEVA_PROBE_MEMTYPE=n   Vulkan memory type n for the shared memory (AHB-exportable, host-visible)
//   AMEVA_PROBE_VOLATILE=1  the OpenCL kernel reads through a volatile pointer (diagnostic: is a cache in the way)
//   AMEVA_PROBE_DMABUF=1    import the AHB's dma_buf (native handle fd 0) with CL_IMPORT_TYPE_DMA_BUF_ARM and
//                           CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM = CL_TRUE instead of the AHB itself
//   AMEVA_PROBE_SEMAPHORE=1 the produce submission signals a Vulkan semaphore exported as a sync_fd; the OpenCL queue
//                           imports it (cl_khr_external_semaphore_sync_fd) and waits on it before the kernel, and the
//                           host does not wait for the produce fence first (needs a Vulkan produce submission)
// Not offered because the drivers refuse them (S21 r38p0, A53 r32p1): clEnqueueMapBuffer and clEnqueueWriteBuffer on
// the AHB import (CL_INVALID_OPERATION, -59); a device-local memory type that is not host-visible (none is
// AHB-exportable).
//
// Build: glslangValidator -V --vn kProbeSpv -o probe_spv.h tests/ameva_vk_cl_gpu_probe.comp
//        clang -O2 -I. -o handoff tests/ameva_vk_cl_gpu_handoff_probe.c -lvulkan -lOpenCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <vulkan/vulkan.h>

#include "probe_spv.h"

#ifndef CL_IMPORT_TYPE_ARM
#define CL_IMPORT_TYPE_ARM 0x40B2
#endif
#ifndef CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM
#define CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM 0x41E2
#endif
#ifndef CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM
#define CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM SIZE_MAX
#endif
#ifndef CL_IMPORT_TYPE_DMA_BUF_ARM
#define CL_IMPORT_TYPE_DMA_BUF_ARM 0x40B4
#endif
#ifndef CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM
#define CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM 0x41E3
#endif
typedef intptr_t probe_import_prop;
typedef cl_mem (*clImportMemoryARM_fn)(cl_context, cl_mem_flags, const probe_import_prop*, void*, size_t, cl_int*);

// libnativewindow's AHardwareBuffer_getNativeHandle (API 26), looked up at run time
typedef struct {
    int version, numFds, numInts;
    int data[];
} probe_native_handle;
typedef const probe_native_handle* (*getNativeHandle_fn)(const struct AHardwareBuffer*);

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static int fail(const char* step, long code) {
    printf("GPU_HANDOFF=FAIL (%s, code %ld)\n", step, code);
    return 1;
}

static int env_int(const char* name) {
    const char* v = getenv(name);
    return v ? atoi(v) : 0;
}

static const char* kSrc =
    "#ifdef VOLATILE\n"
    "#define PTR volatile __global float*\n"
    "#else\n"
    "#define PTR __global float*\n"
    "#endif\n"
    "__kernel void twice_plus_c(PTR a, int nhalf, float c) {\n"
    "#ifdef ACQUIRE\n"
    "    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);\n"
    "#endif\n"
    "    int i = get_global_id(0);\n"
    "    if (i < nhalf) a[nhalf + i] = a[i] * 2.0f + c;\n"
    "}\n";

typedef struct {
    VkDevice dev;
    VkQueue q;
    VkCommandBuffer cb;
    VkPipeline pipe;
    VkPipelineLayout pl;
    VkDescriptorSet ds;
    VkFence fence;
    VkBuffer buf;   // the memory shared with OpenCL
    int foreign;    // AMEVA_PROBE_FOREIGN
    VkSemaphore sem;   // AMEVA_PROBE_SEMAPHORE: signalled by produce submissions, which the host then does not wait for
} VkRun;

// Ownership of the shared buffer between this queue family (0) and the foreign agent (the OpenCL queue): a release
// after Vulkan wrote it, an acquire before Vulkan reads what OpenCL wrote. The Vulkan specification asks for these
// for memory used by an external agent.
static void vk_foreign_release(const VkRun* v, VkPipelineStageFlags stage, VkAccessFlags access) {
    VkBufferMemoryBarrier b = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL, access, 0, 0, VK_QUEUE_FAMILY_FOREIGN_EXT,
                                v->buf, 0, VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(v->cb, stage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 1, &b, 0, NULL);
}
static void vk_foreign_acquire(const VkRun* v) {
    VkBufferMemoryBarrier b = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL, 0,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_FOREIGN_EXT, 0,
                                v->buf, 0, VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(v->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1,
                         &b, 0, NULL);
}

static VkResult vk_begin(const VkRun* v) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    VkResult r = vkResetCommandBuffer(v->cb, 0);
    return r == VK_SUCCESS ? vkBeginCommandBuffer(v->cb, &bi) : r;
}

// end and submit, signalling `signal` when it is not null
static VkResult vk_submit(const VkRun* v, VkSemaphore signal) {
    VkResult r = vkEndCommandBuffer(v->cb);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &v->cb, signal ? 1u : 0u, &signal };
    if (r == VK_SUCCESS) r = vkResetFences(v->dev, 1, &v->fence);
    if (r == VK_SUCCESS) r = vkQueueSubmit(v->q, 1, &si, v->fence);
    return r;
}

static VkResult vk_wait(const VkRun* v) { return vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX); }

// end, submit, wait on the host
static VkResult vk_submit_wait(const VkRun* v) {
    VkResult r = vk_submit(v, VK_NULL_HANDLE);
    return r == VK_SUCCESS ? vk_wait(v) : r;
}

// a produce submission: in the semaphore mode it signals v->sem and the host does not wait
static VkResult vk_submit_produce(const VkRun* v) { return v->sem ? vk_submit(v, v->sem) : vk_submit_wait(v); }

// make writes of `stage` visible to the host and to later (other-API) device accesses
static void vk_writes_out(const VkRun* v, VkPipelineStageFlags stage, VkAccessFlags access) {
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, access,
                           VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT };
    vkCmdPipelineBarrier(v->cb, stage, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 1, &mb, 0,
                         NULL, 0, NULL);
}

// one dispatch of the probe shader in `mode`, waited for on the host
static VkResult vk_run(const VkRun* v, uint32_t nhalf, uint32_t mode, uint32_t round) {
    VkResult r = vk_begin(v);
    if (r != VK_SUCCESS) return r;
    // the check (mode 1) reads what OpenCL wrote: take the buffer back from the foreign agent first
    if (v->foreign && mode == 1u) vk_foreign_acquire(v);
    vkCmdBindPipeline(v->cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe);
    vkCmdBindDescriptorSets(v->cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0, 1, &v->ds, 0, NULL);
    const uint32_t pcv[3] = { nhalf, mode, round };
    vkCmdPushConstants(v->cb, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), pcv);
    const uint32_t groups = (nhalf + 63) / 64;
    const uint32_t gx = groups < 65535 ? groups : 65535, gy = (groups + 65534) / 65535;
    vkCmdDispatch(v->cb, gx, gy, 1);
    vk_writes_out(v, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    // the other modes hand the buffer to OpenCL: release it to the foreign agent
    if (v->foreign && mode != 1u) vk_foreign_release(v, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    return mode == 1u ? vk_submit_wait(v) : vk_submit_produce(v);
}

// a host-visible, host-coherent buffer of `bytes`, mapped
static VkResult host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties* mp, VkDeviceSize bytes,
                            VkBufferUsageFlags usage, VkBuffer* b, void** map) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes, usage, VK_SHARING_MODE_EXCLUSIVE, 0,
                               NULL };
    VkResult r = vkCreateBuffer(dev, &bci, NULL, b);
    if (r != VK_SUCCESS) return r;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *b, &req);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t t = UINT32_MAX;
    for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) { t = i; break; }
    }
    if (t == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, t };
    VkDeviceMemory m;
    r = vkAllocateMemory(dev, &mai, NULL, &m);
    if (r == VK_SUCCESS) r = vkBindBufferMemory(dev, *b, m, 0);
    if (r == VK_SUCCESS) r = vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, map);
    return r;
}

int main(int argc, char** argv) {
    const size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 64;
    const int rounds = argc > 2 ? atoi(argv[2]) : 10;
    const VkDeviceSize bytes = (VkDeviceSize)mb << 20;
    const size_t n = (size_t)bytes / sizeof(float), half = n / 2;
    const char* produce = argc > 3 ? argv[3] : "gpu";
    const int staged = !strcmp(produce, "staged"), compute = !strcmp(produce, "compute");
    const int launder = !strcmp(produce, "launder"), cpu_produce = launder || !strcmp(produce, "cpu");
    if (!staged && !compute && !cpu_produce && strcmp(produce, "gpu")) return fail("unknown produce mode", 0);
    const long sleep_us = getenv("AMEVA_PROBE_SLEEP_US") ? atol(getenv("AMEVA_PROBE_SLEEP_US")) : 0;
    const int reimport = env_int("AMEVA_PROBE_REIMPORT"), migrate = env_int("AMEVA_PROBE_MIGRATE");
    const int acquire = env_int("AMEVA_PROBE_ACQUIRE"), dummymap = env_int("AMEVA_PROBE_DUMMYMAP");
    const int memtype = getenv("AMEVA_PROBE_MEMTYPE") ? env_int("AMEVA_PROBE_MEMTYPE") : -1;
    const int vol = env_int("AMEVA_PROBE_VOLATILE"), dmabuf = env_int("AMEVA_PROBE_DMABUF");
    const int use_sem = env_int("AMEVA_PROBE_SEMAPHORE");

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "handoff-probe", 1, "ameva", 1, VK_API_VERSION_1_1 };
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
    const char* exts[] = { VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
                           VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, use_sem ? 3u : 2u, exts,
                               NULL };
    VkDevice dev;
    vr = vkCreateDevice(pd, &dci, NULL, &dev);
    if (vr != VK_SUCCESS) return fail("vkCreateDevice (AHB extensions)", vr);
    PFN_vkGetMemoryAndroidHardwareBufferANDROID getAhb =
        (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(dev, "vkGetMemoryAndroidHardwareBufferANDROID");
    if (!getAhb) return fail("vkGetMemoryAndroidHardwareBufferANDROID not found", 0);

    // exportable (AHB), host-visible storage buffer
    const VkExternalMemoryHandleTypeFlagBits HT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkExternalMemoryBufferCreateInfo ebci = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, HT };
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebci, 0, bytes,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    VkBuffer buf;
    vr = vkCreateBuffer(dev, &bci, NULL, &buf);
    if (vr != VK_SUCCESS) return fail("vkCreateBuffer", vr);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i)) || !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            continue;
        if (memtype < 0 || (int)i == memtype) { mt = i; break; }
    }
    if (mt == UINT32_MAX) return fail("no host-visible AHB-exportable memory type (AMEVA_PROBE_MEMTYPE)", memtype);
    const int host_coherent = (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryDedicatedAllocateInfo dai = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, buf };
    VkExportMemoryAllocateInfo eai = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dai, HT };
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &eai, req.size, mt };
    VkDeviceMemory mem;
    vr = vkAllocateMemory(dev, &mai, NULL, &mem);
    if (vr != VK_SUCCESS) return fail("vkAllocateMemory (export AHB)", vr);
    vr = vkBindBufferMemory(dev, buf, mem, 0);
    if (vr != VK_SUCCESS) return fail("vkBindBufferMemory", vr);
    float* fw = NULL;   // the shared memory through the Vulkan mapping (cpu / launder modes only)
    vr = vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void**)&fw);
    if (vr != VK_SUCCESS || !fw) return fail("vkMapMemory", vr);

    // result buffer (binding 2) and, for staged / compute, the staging buffer (binding 1); neither is shared
    VkBuffer res, stg = VK_NULL_HANDLE;
    float* f = NULL;
    float* fs = NULL;
    vr = host_buffer(dev, &mp, (VkDeviceSize)half * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &res, (void**)&f);
    if (vr != VK_SUCCESS || !f) return fail("result buffer", vr);
    if (staged || compute) {
        vr = host_buffer(dev, &mp, (VkDeviceSize)half * 4,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &stg, (void**)&fs);
        if (vr != VK_SUCCESS || !fs) return fail("staging buffer", vr);
    }

    // compute pipeline of the probe shader
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(kProbeSpv), kProbeSpv };
    VkShaderModule sm;
    vr = vkCreateShaderModule(dev, &smci, NULL, &sm);
    if (vr != VK_SUCCESS) return fail("vkCreateShaderModule", vr);
    VkDescriptorSetLayoutBinding dslb[3];
    for (uint32_t b = 0; b < 3; ++b) {
        dslb[b] = (VkDescriptorSetLayoutBinding){ b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, dslb };
    VkDescriptorSetLayout dsl;
    vr = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl);
    if (vr != VK_SUCCESS) return fail("vkCreateDescriptorSetLayout", vr);
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    VkRun v = { 0 };
    v.dev = dev;
    v.buf = buf;
    v.foreign = env_int("AMEVA_PROBE_FOREIGN");
    vr = vkCreatePipelineLayout(dev, &plci, NULL, &v.pl);
    if (vr != VK_SUCCESS) return fail("vkCreatePipelineLayout", vr);
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                         { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                                           VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                                         v.pl, VK_NULL_HANDLE, -1 };
    vr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &v.pipe);
    if (vr != VK_SUCCESS) return fail("vkCreateComputePipelines", vr);
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &dps };
    VkDescriptorPool dp;
    vr = vkCreateDescriptorPool(dev, &dpci, NULL, &dp);
    if (vr != VK_SUCCESS) return fail("vkCreateDescriptorPool", vr);
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
    vr = vkAllocateDescriptorSets(dev, &dsai, &v.ds);
    if (vr != VK_SUCCESS) return fail("vkAllocateDescriptorSets", vr);
    // binding 1 (read by shader mode 3 only) is the staging buffer when there is one, else the result buffer
    VkDescriptorBufferInfo dbi[3] = { { buf, 0, VK_WHOLE_SIZE }, { stg ? stg : res, 0, VK_WHOLE_SIZE },
                                      { res, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wds[3];
    for (uint32_t b = 0; b < 3; ++b) {
        wds[b] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, v.ds, b, 0, 1,
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[b], NULL };
    }
    vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);
    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0 };
    VkCommandPool cpool;
    vr = vkCreateCommandPool(dev, &cpi, NULL, &cpool);
    if (vr != VK_SUCCESS) return fail("vkCreateCommandPool", vr);
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    vr = vkAllocateCommandBuffers(dev, &cbai, &v.cb);
    if (vr != VK_SUCCESS) return fail("vkAllocateCommandBuffers", vr);
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    vr = vkCreateFence(dev, &fci, NULL, &v.fence);
    if (vr != VK_SUCCESS) return fail("vkCreateFence", vr);
    vkGetDeviceQueue(dev, 0, 0, &v.q);
    PFN_vkGetSemaphoreFdKHR getSemFd = NULL;
    if (use_sem) {
        if (cpu_produce && !launder) return fail("AMEVA_PROBE_SEMAPHORE with the cpu mode orders nothing", 0);
        VkExportSemaphoreCreateInfo esci = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, NULL,
                                             VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &esci, 0 };
        vr = vkCreateSemaphore(dev, &sci, NULL, &v.sem);
        if (vr != VK_SUCCESS) return fail("vkCreateSemaphore (exportable sync_fd)", vr);
        getSemFd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");
        if (!getSemFd) return fail("vkGetSemaphoreFdKHR not found", 0);
    }

    // AHB export, OpenCL import
    VkMemoryGetAndroidHardwareBufferInfoANDROID gai = { VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                        NULL, mem };
    struct AHardwareBuffer* ahb = NULL;
    vr = getAhb(dev, &gai, &ahb);
    if (vr != VK_SUCCESS || !ahb) return fail("vkGetMemoryAndroidHardwareBufferANDROID", vr);
    cl_platform_id plat;
    cl_device_id cdev;
    cl_int e = clGetPlatformIDs(1, &plat, NULL);
    if (e != CL_SUCCESS) return fail("clGetPlatformIDs", e);
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cdev, NULL);
    if (e != CL_SUCCESS) return fail("clGetDeviceIDs", e);
    cl_context cctx = clCreateContext(NULL, 1, &cdev, NULL, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateContext", e);
    cl_command_queue cq = clCreateCommandQueueWithProperties(cctx, cdev, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateCommandQueueWithProperties", e);
    clImportMemoryARM_fn import = (clImportMemoryARM_fn)clGetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM");
    if (!import) return fail("clImportMemoryARM not exported", 0);
    const probe_import_prop props_ahb[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM, 0 };
    const probe_import_prop props_dmabuf[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM,
                                               CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM, CL_TRUE, 0 };
    const probe_import_prop* props = dmabuf ? props_dmabuf : props_ahb;
    void* import_mem = ahb;
    size_t import_size = CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM;
    int dmabuf_fd = -1;
    if (dmabuf) {
        void* nw = dlopen("libnativewindow.so", RTLD_NOW);
        getNativeHandle_fn getnh = nw ? (getNativeHandle_fn)dlsym(nw, "AHardwareBuffer_getNativeHandle") : NULL;
        if (!getnh) return fail("AHardwareBuffer_getNativeHandle not found", 0);
        const probe_native_handle* nh = getnh(ahb);
        if (!nh || nh->numFds < 1) return fail("AHB native handle without fds", nh ? nh->numFds : -1);
        dmabuf_fd = nh->data[0];
        import_mem = &dmabuf_fd;
        import_size = (size_t)bytes;
        printf("AHB native handle: %d fds, %d ints, dma_buf fd %d\n", nh->numFds, nh->numInts, dmabuf_fd);
    }
    cl_mem m = import(cctx, CL_MEM_READ_WRITE, props, import_mem, import_size, &e);
    if (e != CL_SUCCESS || !m) return fail(dmabuf ? "clImportMemoryARM(dma_buf, consistent with host)" : "clImportMemoryARM(AHB)", e);
    cl_program prog = clCreateProgramWithSource(cctx, 1, &kSrc, NULL, &e);
    if (e != CL_SUCCESS) return fail("clCreateProgramWithSource", e);
    char opts[96];
    snprintf(opts, sizeof(opts), "%s%s", acquire ? "-cl-std=CL3.0 -DACQUIRE " : "", vol ? "-DVOLATILE" : "");
    e = clBuildProgram(prog, 1, &cdev, opts, NULL, NULL);
    if (e != CL_SUCCESS) {
        char log[4096] = { 0 };
        clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        printf("%s\n", log);
        return fail("clBuildProgram", e);
    }
    cl_kernel k = clCreateKernel(prog, "twice_plus_c", &e);
    if (e != CL_SUCCESS) return fail("clCreateKernel", e);
    const cl_int h = (cl_int)half;
    clSetKernelArg(k, 0, sizeof(cl_mem), &m);
    clSetKernelArg(k, 1, sizeof(cl_int), &h);
    const size_t g = (half + 63) / 64 * 64, l = 64;
    clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR_p = NULL;
    clEnqueueWaitSemaphoresKHR_fn clEnqueueWaitSemaphoresKHR_p = NULL;
    clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR_p = NULL;
    int sem_signalled = 0;
    if (use_sem) {
        clCreateSemaphoreWithPropertiesKHR_p = (clCreateSemaphoreWithPropertiesKHR_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clCreateSemaphoreWithPropertiesKHR");
        clEnqueueWaitSemaphoresKHR_p =
            (clEnqueueWaitSemaphoresKHR_fn)clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueWaitSemaphoresKHR");
        clReleaseSemaphoreKHR_p =
            (clReleaseSemaphoreKHR_fn)clGetExtensionFunctionAddressForPlatform(plat, "clReleaseSemaphoreKHR");
        if (!clCreateSemaphoreWithPropertiesKHR_p || !clEnqueueWaitSemaphoresKHR_p || !clReleaseSemaphoreKHR_p)
            return fail("cl_khr_semaphore entry points not exported", 0);
    }
    cl_mem dummy = NULL;
    if (dummymap) {
        dummy = clCreateBuffer(cctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &e);
        if (e != CL_SUCCESS || !dummy) return fail("clCreateBuffer (dummy)", e);
    }
    printf("device %s, %zu MB shared (AHB, memory type %u flags 0x%x), %d rounds of %s -> OpenCL -> Vulkan check "
           "(sleep %ld us, foreign %d, reimport %d, migrate %d, acquire %d, dummymap %d, volatile %d, dmabuf %d, "
           "semaphore %d)\n",
           pprops.deviceName, mb, mt, (unsigned)mp.memoryTypes[mt].propertyFlags, rounds, produce, sleep_us, v.foreign,
           reimport, migrate, acquire, dummymap, vol, dmabuf, use_sem);

    size_t total_bad = 0, n_cl_in = 0, n_vk_out = 0, n_vk_in = 0, n_other = 0, n_line = 0;
    double t_vk0 = 0, t_cl = 0, t_vk1 = 0;
    for (int round = 0; round < rounds; ++round) {
        // 1. produce
        double t0 = now_ms();
        if (compute || staged) {
            for (size_t i = 0; i < half; ++i) fs[i] = (float)(i % 1000003) * 0.5f + (float)round;
        }
        if (compute) {
            vr = vk_run(&v, (uint32_t)half, 3u, (uint32_t)round);
        } else if (staged) {
            vr = vk_begin(&v);
            if (vr != VK_SUCCESS) return fail("staging copy begin", vr);
            const VkBufferCopy region = { 0, 0, (VkDeviceSize)half * 4 };
            vkCmdCopyBuffer(v.cb, stg, buf, 1, &region);
            vk_writes_out(&v, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            if (v.foreign) vk_foreign_release(&v, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            vr = vk_submit_produce(&v);
        } else if (cpu_produce) {
            for (size_t i = 0; i < half; ++i) fw[i] = (float)(i % 1000003) * 0.5f + (float)round;
            __asm__ volatile("dsb sy" ::: "memory");
            if (!host_coherent) {
                const VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, mem, 0, VK_WHOLE_SIZE };
                vr = vkFlushMappedMemoryRanges(dev, 1, &range);
                if (vr != VK_SUCCESS) return fail("vkFlushMappedMemoryRanges", vr);
            }
            vr = launder ? vk_run(&v, (uint32_t)half, 2u, 0u) : vk_run(&v, 0u, 0u, (uint32_t)round);
        } else {
            vr = vk_run(&v, (uint32_t)half, 0u, (uint32_t)round);
        }
        if (vr != VK_SUCCESS) return fail("produce", vr);
        t_vk0 += now_ms() - t0;
        if (sleep_us > 0) {
            struct timespec ts = { sleep_us / 1000000, (sleep_us % 1000000) * 1000 };
            nanosleep(&ts, NULL);
        }
        // 2. OpenCL
        t0 = now_ms();
        if (reimport) {
            clReleaseMemObject(m);
            m = import(cctx, CL_MEM_READ_WRITE, props, import_mem, import_size, &e);
            if (e != CL_SUCCESS || !m) return fail("clImportMemoryARM again", e);
            clSetKernelArg(k, 0, sizeof(cl_mem), &m);
        }
        if (dummymap) {
            float* dpm = (float*)clEnqueueMapBuffer(cq, dummy, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &e);
            if (e != CL_SUCCESS || !dpm) return fail("clEnqueueMapBuffer (dummy)", e);
            dpm[0] = (float)round;
            e = clEnqueueUnmapMemObject(cq, dummy, dpm, 0, NULL, NULL);
            if (e != CL_SUCCESS) return fail("clEnqueueUnmapMemObject (dummy)", e);
        }
        if (migrate) {
            e = clEnqueueMigrateMemObjects(cq, 1, &m, 0, 0, NULL, NULL);
            if (e != CL_SUCCESS) return fail("clEnqueueMigrateMemObjects", e);
        }
        cl_semaphore_khr csem = NULL;
        if (v.sem) {
            // the produce submission's completion, as a sync_fd, becomes an OpenCL semaphore the queue waits on
            int fd = -1;
            VkSemaphoreGetFdInfoKHR gfi = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, NULL, v.sem,
                                            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
            vr = getSemFd(dev, &gfi, &fd);
            if (vr != VK_SUCCESS) return fail("vkGetSemaphoreFdKHR", vr);
            if (fd >= 0) {
                const cl_semaphore_properties_khr sp[] = { CL_SEMAPHORE_TYPE_KHR, CL_SEMAPHORE_TYPE_BINARY_KHR,
                                                           CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, (cl_semaphore_properties_khr)fd,
                                                           0 };
                csem = clCreateSemaphoreWithPropertiesKHR_p(cctx, sp, &e);
                if (e != CL_SUCCESS || !csem) return fail("clCreateSemaphoreWithPropertiesKHR (sync_fd)", e);
                e = clEnqueueWaitSemaphoresKHR_p(cq, 1, &csem, NULL, 0, NULL, NULL);
                if (e != CL_SUCCESS) return fail("clEnqueueWaitSemaphoresKHR", e);
            } else {
                ++sem_signalled;   // already signalled when exported: nothing for OpenCL to wait on
            }
        }
        const cl_float c = (cl_float)(round + 1);
        clSetKernelArg(k, 2, sizeof(cl_float), &c);
        e = clEnqueueNDRangeKernel(cq, k, 1, NULL, &g, &l, 0, NULL, NULL);
        if (e == CL_SUCCESS) e = clFinish(cq);
        if (e != CL_SUCCESS) return fail("OpenCL kernel", e);
        if (csem) clReleaseSemaphoreKHR_p(csem);
        if (v.sem) {
            vr = vk_wait(&v);   // the produce fence, before the check reuses it
            if (vr != VK_SUCCESS) return fail("produce fence", vr);
        }
        t_cl += now_ms() - t0;
        // 3. check
        t0 = now_ms();
        vr = vk_run(&v, (uint32_t)half, 1u, (uint32_t)round);
        if (vr != VK_SUCCESS) return fail("Vulkan check", vr);
        t_vk1 += now_ms() - t0;
        // 4. classify
        size_t bad = 0, last_line = SIZE_MAX;
        int shown = 0;
        for (size_t i = 0; i < half; ++i) {
            const float r = f[i];
            if (r == 1.0f) continue;
            ++bad;
            if (i / 16 != last_line) { ++n_line; last_line = i / 16; }
            const float want_in = (float)(i % 1000003) * 0.5f + (float)round, out = -r;
            const char* what;
            if (r == 0.25f) {
                ++n_vk_in;
                what = "Vulkan read a stale input";
            } else if (round > 0 ? out == 2.0f * (want_in - 1.0f) + (float)(round + 1) : out == 1.0f) {
                ++n_cl_in;   // round 0: the input was still the zero of the fresh allocation
                what = "OpenCL read a stale input";
            } else if (round > 0 ? out == 2.0f * (want_in - 1.0f) + (float)round : out == 0.0f) {
                ++n_vk_out;  // round 0: the output was still the zero of the fresh allocation
                what = "Vulkan read a stale OpenCL result";
            } else {
                ++n_other;
                what = "other";
            }
            if (shown < 3) {
                printf("  round %d i=%zu (byte %zu): %s (check %g, want input %g)\n", round, i, i * 4, what, r, want_in);
                ++shown;
            }
        }
        total_bad += bad;
        if (bad) printf("round %d: %zu of %zu values not handed over\n", round, bad, half);
    }
    printf("per round: %s produce %.3f ms, OpenCL %.3f ms, Vulkan check %.3f ms\n", produce, t_vk0 / rounds,
           t_cl / rounds, t_vk1 / rounds);
    if (use_sem) printf("semaphore: %d of %d rounds already signalled when exported\n", sem_signalled, rounds);
    if (total_bad) {
        printf("GPU_HANDOFF=FAIL (%zu values in %zu 64-byte lines over %d rounds: OpenCL stale input %zu, Vulkan stale "
               "OpenCL result %zu, Vulkan stale input %zu, other %zu)\n",
               total_bad, n_line, rounds, n_cl_in, n_vk_out, n_vk_in, n_other);
        return 1;
    }
    printf("GPU_HANDOFF=PASS (%d rounds x %zu values, %s -> OpenCL -> Vulkan through one AHB)\n", rounds, half, produce);
    return 0;
}
