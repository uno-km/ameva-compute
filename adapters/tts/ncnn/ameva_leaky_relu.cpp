#include "ameva_leaky_relu.h"
#include "ameva_tts_counters.h"
#include "ac_runtime.h"
#include "../../src/shaders/tts/leaky_relu_spv.h"

#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

namespace ameva {
namespace tts {

namespace {

struct LeakyReluPushConstants {
    uint32_t n_elements;
    float negative_slope;
};

class AmevaComputeManager {
public:
    static AmevaComputeManager& Instance() {
        static AmevaComputeManager inst;
        return inst;
    }

    bool EnsureInitialized() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (initialized_) return true;

        const char* fail_stage = getenv("AMEVA_TTS_FAIL_STAGE");

        fprintf(stderr, "[AMEVA-TTS-LIFECYCLE] Initializing AMEVA Compute Runtime Context...\n");

        // Gate 1: Context failure injection
        if (fail_stage && strcmp(fail_stage, "context") == 0) {
            fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'context'\n");
            g_counters.gpu_failure_count++;
            DumpCounters();
            return false;
        }

        AcContextDesc ctx_desc = {};
        ctx_desc.abi_version = AC_ABI_VERSION;
        ctx_desc.struct_size = sizeof(AcContextDesc);
        ctx_desc.backend_kind = AC_BACKEND_VULKAN;
        ctx_desc.flags = 0;

        AcResult res = ac_runtime_context_create(&ctx_desc, &ctx_);
        if (res != AC_SUCCESS || !ctx_) {
            fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_context_create failed with error %d\n", res);
            g_counters.gpu_failure_count++;
            DumpCounters();
            return false;
        }
        g_counters.context_create_count++;

        // Gate 2: Kernel failure injection
        if (fail_stage && strcmp(fail_stage, "kernel") == 0) {
            fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'kernel'\n");
            g_counters.gpu_failure_count++;
            ac_runtime_context_destroy(ctx_);
            ctx_ = nullptr;
            g_counters.context_destroy_count++;
            DumpCounters();
            return false;
        }

        AcRuntimeBindingDesc bindings[2] = {
            { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
            { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
        };

        AcKernelRuntimeMetadata meta = {};
        meta.kernel_name = "leaky_relu";
        meta.entry_point = "main";
        meta.binding_count = 2;
        meta.bindings = bindings;
        meta.push_constant_size = sizeof(LeakyReluPushConstants);
        meta.local_size = { 256, 1, 1 };
        meta.expected_spirv_sha256 = AC_LEAKY_RELU_SPV_SHA256;

        AcKernelDesc kern_desc = {};
        kern_desc.abi_version = AC_ABI_VERSION;
        kern_desc.struct_size = sizeof(AcKernelDesc);
        kern_desc.spirv_code = kLeakyReluSpv;
        kern_desc.code_size_bytes = kLeakyReluSpvByteSize;
        kern_desc.metadata = &meta;

        res = ac_runtime_kernel_create(ctx_, &kern_desc, &kernel_);
        if (res != AC_SUCCESS || !kernel_) {
            fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_kernel_create failed with error %d\n", res);
            ac_runtime_context_destroy(ctx_);
            ctx_ = nullptr;
            g_counters.context_destroy_count++;
            g_counters.gpu_failure_count++;
            DumpCounters();
            return false;
        }
        g_counters.kernel_create_count++;
        initialized_ = true;
        fprintf(stderr, "[AMEVA-TTS-LIFECYCLE] AMEVA Compute Runtime Context & LeakyReLU Kernel INITIALIZED (1 Context, 1 Kernel).\n");
        return true;
    }

    AcContext* GetContext() { return ctx_; }
    AcKernel* GetKernel() { return kernel_; }
    std::recursive_mutex& GetMutex() { return mutex_; }

    ~AmevaComputeManager() {
        Teardown();
    }

    void Teardown() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!initialized_) return;

        fprintf(stderr, "[AMEVA-TTS-LIFECYCLE] Tearing down AMEVA Compute Runtime Context...\n");
        if (kernel_) {
            ac_runtime_kernel_destroy(kernel_);
            kernel_ = nullptr;
            g_counters.kernel_destroy_count++;
        }
        if (ctx_) {
            ac_runtime_context_destroy(ctx_);
            ctx_ = nullptr;
            g_counters.context_destroy_count++;
        }
        initialized_ = false;
        DumpCounters();
    }

private:
    AmevaComputeManager() = default;
    std::recursive_mutex mutex_;
    AcContext* ctx_{nullptr};
    AcKernel* kernel_{nullptr};
    bool initialized_{false};
};

} // namespace

AmevaLeakyReLU::AmevaLeakyReLU() {
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = false;
    negative_slope_ = 0.1f;
}

AmevaLeakyReLU::~AmevaLeakyReLU() {
}

int AmevaLeakyReLU::load_param(const ncnn::ParamDict& pd) {
    negative_slope_ = pd.get(0, 0.1f);
    fprintf(stderr, "[AMEVA-TTS-CUSTOM-LAYER] Param loaded: negative_slope=%f\n", negative_slope_);
    return 0;
}

int AmevaLeakyReLU::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
    g_counters.custom_layer_exec_count++;

    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    size_t cstep = bottom_blob.cstep;
    size_t total_elements = bottom_blob.total();
    size_t total_bytes = total_elements * elemsize;

    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "[AMEVA-TTS-CUSTOM-LAYER] FORWARD(Mat) INVOKED:\n");
    fprintf(stderr, "  CUSTOM_LAYER_FORWARD_SIGNATURE=Mat\n");
    fprintf(stderr, "  CUSTOM_LAYER_FORWARD_VKMAT_CALLED=false\n");
    fprintf(stderr, "  BOTTOM_BLOB_STORAGE=CPU_MAT\n");
    fprintf(stderr, "  BOTTOM_BLOB_DATA_PTR_NON_NULL=%s\n", bottom_blob.data ? "true" : "false");
    fprintf(stderr, "  TOP_BLOB_STORAGE=CPU_MAT\n");
    fprintf(stderr, "  DIMS=%d, W=%d, H=%d, D=%d, C=%d, ELEMSIZE=%zu, ELEMPACK=%d, CSTEP=%zu\n",
            dims, w, h, d, c, elemsize, elempack, cstep);
    fprintf(stderr, "  TOTAL_ELEMENTS=%zu, TOTAL_BYTES=%zu\n", total_elements, total_bytes);

    if (total_elements == 0 || !bottom_blob.data) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] Empty or null bottom_blob!\n");
        g_counters.gpu_failure_count++;
        DumpCounters();
        return -1;
    }

    top_blob.create(w, h, c, elemsize, elempack, opt.blob_allocator);
    if (!top_blob.data) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] Failed to allocate top_blob with NCNN allocator!\n");
        g_counters.gpu_failure_count++;
        DumpCounters();
        return -1;
    }

    auto& mgr = AmevaComputeManager::Instance();
    if (!mgr.EnsureInitialized()) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] Failed to initialize AMEVA context/kernel!\n");
        top_blob.release();
        return -1;
    }

    const char* fail_stage = getenv("AMEVA_TTS_FAIL_STAGE");

    // Gate 3: Buffer allocation failure injection
    if (fail_stage && strcmp(fail_stage, "buffer") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'buffer'\n");
        g_counters.gpu_failure_count++;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }

    std::lock_guard<std::recursive_mutex> lock(mgr.GetMutex());
    AcContext* ctx = mgr.GetContext();
    AcKernel* kernel = mgr.GetKernel();

    AcBufferDesc buf_desc = {};
    buf_desc.abi_version = AC_ABI_VERSION;
    buf_desc.struct_size = sizeof(AcBufferDesc);
    buf_desc.size_bytes = total_bytes;
    buf_desc.usage = AC_BUFFER_USAGE_STORAGE | AC_BUFFER_USAGE_TRANSFER_SRC | AC_BUFFER_USAGE_TRANSFER_DST;

    AcBuffer* buf_in = nullptr;
    AcBuffer* buf_out = nullptr;
    AcResult res = ac_runtime_buffer_create(ctx, &buf_desc, &buf_in);
    if (res != AC_SUCCESS || !buf_in) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_buffer_create in failed (%d)\n", res);
        g_counters.gpu_failure_count++;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }
    g_counters.buffer_create_count++;

    res = ac_runtime_buffer_create(ctx, &buf_desc, &buf_out);
    if (res != AC_SUCCESS || !buf_out) {
        ac_runtime_buffer_destroy(buf_in);
        g_counters.buffer_destroy_count++;
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_buffer_create out failed (%d)\n", res);
        g_counters.gpu_failure_count++;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }
    g_counters.buffer_create_count++;

    // Gate 4: Dispatch failure injection
    if (fail_stage && strcmp(fail_stage, "dispatch") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'dispatch'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }

    void* mapped_in = nullptr;
    void* mapped_out = nullptr;
    ac_runtime_buffer_get_mapped_ptr(buf_in, &mapped_in);
    ac_runtime_buffer_get_mapped_ptr(buf_out, &mapped_out);

    memcpy(mapped_in, bottom_blob.data, total_bytes);
    ac_runtime_buffer_flush(buf_in, 0, total_bytes);
    g_counters.staging_h2d_bytes += total_bytes;

    AcLaunchDesc launch = {};
    launch.abi_version = AC_ABI_VERSION;
    launch.struct_size = sizeof(AcLaunchDesc);
    launch.kernel = kernel;
    AcBuffer* bufs[2] = { buf_in, buf_out };
    launch.buffers = bufs;
    launch.buffer_count = 2;

    LeakyReluPushConstants pc = {};
    pc.n_elements = (uint32_t)total_elements;
    pc.negative_slope = negative_slope_;
    launch.push_constants = &pc;
    launch.push_constants_size = sizeof(pc);
    launch.grid_dim = { (pc.n_elements + 255) / 256, 1, 1 };

    res = ac_runtime_dispatch(ctx, &launch, nullptr);
    if (res != AC_SUCCESS) {
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_dispatch failed (%d)\n", res);
        g_counters.gpu_failure_count++;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }
    g_counters.gpu_dispatch_count++;

    // Gate 5: Wait failure injection
    if (fail_stage && strcmp(fail_stage, "wait") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'wait'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }

    // Gate 6: Readback failure injection
    if (fail_stage && strcmp(fail_stage, "readback") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'readback'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        top_blob.release();
        mgr.Teardown();
        return -1;
    }

    ac_runtime_buffer_invalidate(buf_out, 0, total_bytes);
    memcpy(top_blob.data, mapped_out, total_bytes);
    g_counters.staging_d2h_bytes += total_bytes;

    ac_runtime_buffer_destroy(buf_in);
    ac_runtime_buffer_destroy(buf_out);
    g_counters.buffer_destroy_count += 2;

    fprintf(stderr, "[AMEVA-TTS-CUSTOM-LAYER] FORWARD(Mat) SUCCESSFUL: AMEVA GPU Dispatched %zu elements.\n", total_elements);
    fprintf(stderr, "----------------------------------------------------------------------\n");
    return 0;
}

#if NCNN_VULKAN
int AmevaLeakyReLU::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const {
    g_counters.custom_layer_exec_count++;

    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    size_t cstep = bottom_blob.cstep;
    size_t total_elements = bottom_blob.total();
    size_t total_bytes = total_elements * elemsize;

    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "[AMEVA-TTS-CUSTOM-LAYER] FORWARD(VkMat) INVOKED:\n");
    fprintf(stderr, "  CUSTOM_LAYER_FORWARD_SIGNATURE=VkMat\n");
    fprintf(stderr, "  CUSTOM_LAYER_FORWARD_VKMAT_CALLED=true\n");
    fprintf(stderr, "  BOTTOM_BLOB_STORAGE=VK_MAT\n");
    fprintf(stderr, "  DIMS=%d, W=%d, H=%d, D=%d, C=%d, ELEMSIZE=%zu, ELEMPACK=%d, CSTEP=%zu\n",
            dims, w, h, d, c, elemsize, elempack, cstep);
    fprintf(stderr, "  TOTAL_ELEMENTS=%zu, TOTAL_BYTES=%zu\n", total_elements, total_bytes);

    cmd.submit_and_wait();

    ncnn::Mat host_in;
    cmd.record_download(bottom_blob, host_in, opt);
    cmd.submit_and_wait();

    ncnn::Mat host_out;
    host_out.create(w, h, c, elemsize, elempack, opt.blob_allocator);

    auto& mgr = AmevaComputeManager::Instance();
    if (!mgr.EnsureInitialized()) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] Failed to initialize AMEVA context/kernel!\n");
        return -1;
    }

    const char* fail_stage = getenv("AMEVA_TTS_FAIL_STAGE");

    // Gate 3: Buffer allocation failure injection
    if (fail_stage && strcmp(fail_stage, "buffer") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'buffer'\n");
        g_counters.gpu_failure_count++;
        mgr.Teardown();
        return -1;
    }

    std::lock_guard<std::recursive_mutex> lock(mgr.GetMutex());
    AcContext* ctx = mgr.GetContext();
    AcKernel* kernel = mgr.GetKernel();

    AcBufferDesc buf_desc = {};
    buf_desc.abi_version = AC_ABI_VERSION;
    buf_desc.struct_size = sizeof(AcBufferDesc);
    buf_desc.size_bytes = total_bytes;
    buf_desc.usage = AC_BUFFER_USAGE_STORAGE | AC_BUFFER_USAGE_TRANSFER_SRC | AC_BUFFER_USAGE_TRANSFER_DST;

    AcBuffer* buf_in = nullptr;
    AcBuffer* buf_out = nullptr;
    AcResult res = ac_runtime_buffer_create(ctx, &buf_desc, &buf_in);
    if (res != AC_SUCCESS || !buf_in) {
        g_counters.gpu_failure_count++;
        mgr.Teardown();
        return -1;
    }
    g_counters.buffer_create_count++;

    res = ac_runtime_buffer_create(ctx, &buf_desc, &buf_out);
    if (res != AC_SUCCESS || !buf_out) {
        ac_runtime_buffer_destroy(buf_in);
        g_counters.buffer_destroy_count++;
        g_counters.gpu_failure_count++;
        mgr.Teardown();
        return -1;
    }
    g_counters.buffer_create_count++;

    // Gate 4: Dispatch failure injection
    if (fail_stage && strcmp(fail_stage, "dispatch") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'dispatch'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        mgr.Teardown();
        return -1;
    }

    void* mapped_in = nullptr;
    void* mapped_out = nullptr;
    ac_runtime_buffer_get_mapped_ptr(buf_in, &mapped_in);
    ac_runtime_buffer_get_mapped_ptr(buf_out, &mapped_out);

    memcpy(mapped_in, host_in.data, total_bytes);
    ac_runtime_buffer_flush(buf_in, 0, total_bytes);
    g_counters.staging_h2d_bytes += total_bytes;

    AcLaunchDesc launch = {};
    launch.abi_version = AC_ABI_VERSION;
    launch.struct_size = sizeof(AcLaunchDesc);
    launch.kernel = kernel;
    AcBuffer* bufs[2] = { buf_in, buf_out };
    launch.buffers = bufs;
    launch.buffer_count = 2;

    LeakyReluPushConstants pc = {};
    pc.n_elements = (uint32_t)total_elements;
    pc.negative_slope = negative_slope_;
    launch.push_constants = &pc;
    launch.push_constants_size = sizeof(pc);
    launch.grid_dim = { (pc.n_elements + 255) / 256, 1, 1 };

    res = ac_runtime_dispatch(ctx, &launch, nullptr);
    if (res != AC_SUCCESS) {
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        fprintf(stderr, "[AMEVA-TTS-FAIL-FAST] ac_runtime_dispatch failed (%d)\n", res);
        g_counters.gpu_failure_count++;
        mgr.Teardown();
        return -1;
    }
    g_counters.gpu_dispatch_count++;

    // Gate 5: Wait failure injection
    if (fail_stage && strcmp(fail_stage, "wait") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'wait'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        mgr.Teardown();
        return -1;
    }

    // Gate 6: Readback failure injection
    if (fail_stage && strcmp(fail_stage, "readback") == 0) {
        fprintf(stderr, "[AMEVA-TTS-FAIL-INJECT] Simulating failure at stage 'readback'\n");
        g_counters.gpu_failure_count++;
        ac_runtime_buffer_destroy(buf_in);
        ac_runtime_buffer_destroy(buf_out);
        g_counters.buffer_destroy_count += 2;
        mgr.Teardown();
        return -1;
    }

    ac_runtime_buffer_invalidate(buf_out, 0, total_bytes);
    memcpy(host_out.data, mapped_out, total_bytes);
    g_counters.staging_d2h_bytes += total_bytes;

    ac_runtime_buffer_destroy(buf_in);
    ac_runtime_buffer_destroy(buf_out);
    g_counters.buffer_destroy_count += 2;

    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    cmd.record_upload(host_out, top_blob, opt);

    fprintf(stderr, "[AMEVA-TTS-CUSTOM-LAYER] FORWARD(VkMat) SUCCESSFUL: AMEVA GPU Dispatched %zu elements.\n", total_elements);
    fprintf(stderr, "----------------------------------------------------------------------\n");
    return 0;
}
#endif

ncnn::Layer* AmevaLeakyReLU_layer_creator(void* userdata) {
    (void)userdata;
    return new AmevaLeakyReLU();
}

void AmevaLeakyReLU_layer_destroyer(ncnn::Layer* layer, void* userdata) {
    (void)userdata;
    delete layer;
}

} // namespace tts
} // namespace ameva
