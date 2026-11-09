// Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 OR Commercial-License
// re_dump: what the GPU driver made of one of OUR kernels (research tool, R2 of the research handover).
//   re_dump vk kernel.spv out.bin      compute pipeline; pipeline statistics if the driver offers them; pipeline cache to out.bin
//   re_dump cl kernel.cl out.bin [opts] OpenCL program binary to out.bin; private/local memory of each kernel
// Fails with a message and exit code 2 on any driver error; nothing is reported that the driver did not return.
#define VK_NO_PROTOTYPES
#define CL_TARGET_OPENCL_VERSION 300
#include <vulkan/vulkan.h>
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char* what, long code) { fprintf(stderr, "re_dump: %s failed (%ld)\n", what, code); exit(2); }
static void* slurp(const char* p, size_t* n) {
    FILE* f = fopen(p, "rb"); if (!f) die(p, -1);
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char* d = malloc(*n + 1); if (fread(d, 1, *n, f) != *n) die("read", -1); d[*n] = 0; fclose(f); return d;
}
static void spit(const char* p, const void* d, size_t n) {
    FILE* f = fopen(p, "wb"); if (!f || fwrite(d, 1, n, f) != n) die(p, -1); fclose(f);
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static int run_vk(const char* spv, const char* out) {
    const char* vkl = getenv("RE_VK_LIB"); /* e.g. Termux's loader for Mesa Turnip */
    void* lib = dlopen(vkl ? vkl : "/system/lib64/libvulkan.so", RTLD_NOW);
    if (!lib) die(dlerror(), 0);
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#define IP(n) PFN_##n n = (PFN_##n)gipa(inst, #n); if (!n) die(#n, 0)
#define DP(n) PFN_##n n = (PFN_##n)gdpa(dev, #n); if (!n) die(#n, 0)
    VkInstance inst = VK_NULL_HANDLE;
    IP(vkCreateInstance);
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "re_dump", 1, "x", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai, 0, 0, 0, 0};
    VkResult r = vkCreateInstance(&ici, 0, &inst); if (r) die("vkCreateInstance", r);
    IP(vkEnumeratePhysicalDevices); IP(vkEnumerateDeviceExtensionProperties); IP(vkGetPhysicalDeviceProperties);
    IP(vkCreateDevice); IP(vkGetDeviceProcAddr);
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd); if (!n) die("no device", 0);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, 0, &ne, 0);
    VkExtensionProperties* ex = calloc(ne, sizeof(*ex)); vkEnumerateDeviceExtensionProperties(pd, 0, &ne, ex);
    int has_stats = 0, has_int = 0;
    for (uint32_t i = 0; i < ne; ++i) {
        if (!strcmp(ex[i].extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) has_stats = 1;
        if (!strcmp(ex[i].extensionName, "VK_KHR_shader_integer_dot_product")) has_int = 1;
    }
    printf("device=%s api=%u.%u driver=%u.%u.%u exts=%u pipeline_executable_properties=%d int_dot_ext=%d\n", pp.deviceName,
           VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion), VK_VERSION_MAJOR(pp.driverVersion),
           VK_VERSION_MINOR(pp.driverVersion), VK_VERSION_PATCH(pp.driverVersion), ne, has_stats, has_int);
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &prio};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR f = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, 0, VK_TRUE};
    const char* en[1] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, has_stats ? &f : 0, 0, 1, &q, 0, 0, has_stats ? 1u : 0u, en, 0};
    VkDevice dev; r = vkCreateDevice(pd, &dci, 0, &dev); if (r) die("vkCreateDevice", r);
    PFN_vkGetDeviceProcAddr gdpa = vkGetDeviceProcAddr;
    DP(vkCreateShaderModule); DP(vkCreateDescriptorSetLayout); DP(vkCreatePipelineLayout); DP(vkCreatePipelineCache);
    DP(vkCreateComputePipelines); DP(vkGetPipelineCacheData);
    size_t sz; void* code = slurp(spv, &sz);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, sz, code};
    VkShaderModule sm; r = vkCreateShaderModule(dev, &smi, 0, &sm); if (r) die("vkCreateShaderModule", r);
    VkDescriptorSetLayoutBinding b[8];
    for (uint32_t i = 0; i < 8; ++i) { b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}; }
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 8, b};
    VkDescriptorSetLayout dsl; r = vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl); if (r) die("descriptor set layout", r);
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dsl, 1, &pc};
    VkPipelineLayout pl; r = vkCreatePipelineLayout(dev, &pli, 0, &pl); if (r) die("pipeline layout", r);
    VkPipelineCacheCreateInfo cci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache cache; r = vkCreatePipelineCache(dev, &cci, 0, &cache); if (r) die("pipeline cache", r);
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sm; ci.stage.pName = "main"; ci.layout = pl;
    if (has_stats) ci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    VkPipeline pipe; double t0 = now_ms();
    r = vkCreateComputePipelines(dev, cache, 1, &ci, 0, &pipe); if (r) die("vkCreateComputePipelines", r);
    printf("spv_bytes=%zu compile_ms=%.1f\n", sz, now_ms() - t0);
    if (has_stats) {
        DP(vkGetPipelineExecutablePropertiesKHR); DP(vkGetPipelineExecutableStatisticsKHR);
        VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, 0, pipe};
        uint32_t ne2 = 0; vkGetPipelineExecutablePropertiesKHR(dev, &pi, &ne2, 0);
        for (uint32_t e = 0; e < ne2; ++e) {
            VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, 0, pipe, e};
            uint32_t ns = 0; vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, 0);
            VkPipelineExecutableStatisticKHR* st = calloc(ns, sizeof(*st));
            for (uint32_t i = 0; i < ns; ++i) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, st);
            DP(vkGetPipelineExecutableInternalRepresentationsKHR);
            uint32_t nr = 0; vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, 0);
            VkPipelineExecutableInternalRepresentationKHR* ir = calloc(nr, sizeof(*ir));
            for (uint32_t i = 0; i < nr; ++i) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
            vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, ir);
            for (uint32_t i = 0; i < nr; ++i) { ir[i].pData = calloc(1, ir[i].dataSize + 1); }
            vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, ir);
            for (uint32_t i = 0; i < nr; ++i) {
                char fn[512]; snprintf(fn, sizeof(fn), "%s.ir%u.txt", out, i);
                spit(fn, ir[i].pData, ir[i].dataSize ? ir[i].dataSize - (ir[i].isText ? 1 : 0) : 0);
                printf("ir[%u] %s: %zu bytes -> %s\n", i, ir[i].name, ir[i].dataSize, fn);
            }
            for (uint32_t i = 0; i < ns; ++i) {
                printf("stat[%u] %s = ", e, st[i].name);
                switch (st[i].format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: printf("%llu\n", (unsigned long long)st[i].value.u64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: printf("%lld\n", (long long)st[i].value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: printf("%f\n", st[i].value.f64); break;
                default: printf("%u\n", st[i].value.b32);
                }
            }
        }
    }
    size_t cn = 0; r = vkGetPipelineCacheData(dev, cache, &cn, 0); if (r) die("vkGetPipelineCacheData", r);
    void* cd = malloc(cn); r = vkGetPipelineCacheData(dev, cache, &cn, cd); if (r) die("vkGetPipelineCacheData", r);
    spit(out, cd, cn); printf("pipeline_cache_bytes=%zu\n", cn);
    return 0;
}

static int run_cl(const char* src, const char* out, const char* opts) {
    void* lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) lib = dlopen("/vendor/lib64/egl/libGLES_mali.so", RTLD_NOW);
    if (!lib) die(dlerror(), 0);
#define CL(n) __typeof__(&n) p_##n = (__typeof__(&n))dlsym(lib, #n); if (!p_##n) die(#n, 0)
    CL(clGetPlatformIDs); CL(clGetDeviceIDs); CL(clGetDeviceInfo); CL(clCreateContext); CL(clCreateProgramWithSource);
    CL(clBuildProgram); CL(clGetProgramBuildInfo); CL(clGetProgramInfo); CL(clCreateKernelsInProgram);
    CL(clGetKernelInfo); CL(clGetKernelWorkGroupInfo);
    cl_platform_id pf; cl_device_id d; cl_int e;
    if ((e = p_clGetPlatformIDs(1, &pf, 0))) die("clGetPlatformIDs", e);
    if ((e = p_clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &d, 0))) die("clGetDeviceIDs", e);
    char name[128] = "", ver[128] = "";
    p_clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, 0); p_clGetDeviceInfo(d, CL_DRIVER_VERSION, sizeof(ver), ver, 0);
    printf("device=%s driver=%s\n", name, ver);
    cl_context c = p_clCreateContext(0, 1, &d, 0, 0, &e); if (e) die("clCreateContext", e);
    size_t n; const char* s = slurp(src, &n);
    cl_program p = p_clCreateProgramWithSource(c, 1, &s, &n, &e); if (e) die("clCreateProgramWithSource", e);
    double t0 = now_ms();
    e = p_clBuildProgram(p, 1, &d, opts, 0, 0);
    if (e) {
        static char log[65536]; p_clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, 0);
        fprintf(stderr, "%s\n", log); die("clBuildProgram", e);
    }
    printf("source_bytes=%zu build_ms=%.1f\n", n, now_ms() - t0);
    size_t bn = 0; if ((e = p_clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, 0))) die("binary size", e);
    unsigned char* b = malloc(bn); unsigned char* bp[1] = {b};
    if ((e = p_clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bp), bp, 0))) die("binary", e);
    spit(out, b, bn); printf("program_binary_bytes=%zu\n", bn);
    cl_uint nk = 0; p_clCreateKernelsInProgram(p, 0, 0, &nk);
    cl_kernel k[64]; if (nk > 64) nk = 64; p_clCreateKernelsInProgram(p, nk, k, 0);
    for (cl_uint i = 0; i < nk; ++i) {
        char kn[128] = ""; cl_ulong pm = 0, lm = 0; size_t wg = 0, pref = 0;
        p_clGetKernelInfo(k[i], CL_KERNEL_FUNCTION_NAME, sizeof(kn), kn, 0);
        p_clGetKernelWorkGroupInfo(k[i], d, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm), &pm, 0);
        p_clGetKernelWorkGroupInfo(k[i], d, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lm), &lm, 0);
        p_clGetKernelWorkGroupInfo(k[i], d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, 0);
        p_clGetKernelWorkGroupInfo(k[i], d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(pref), &pref, 0);
        printf("kernel %s private=%llu local=%llu max_wg=%zu pref_multiple=%zu\n", kn, (unsigned long long)pm, (unsigned long long)lm, wg, pref);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 4 && !strcmp(argv[1], "vk")) return run_vk(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "cl")) return run_cl(argv[2], argv[3], argc > 4 ? argv[4] : "");
    fprintf(stderr, "usage: re_dump vk|cl input output [options]\n");
    return 2;
}
