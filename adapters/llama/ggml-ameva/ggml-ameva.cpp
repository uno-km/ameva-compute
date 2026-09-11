#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ac_runtime.h"
#include "ac_dispatch.h"
#include "ggml_mul_mat_spv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <vector>

void ameva_increment_gpu_op_count(void);

// Internal AcContext & AcKernel Singletons
AcContext* ameva_get_ac_context() {
    static AcContext* s_ctx = nullptr;
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_ctx) {
        AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
        AcResult res = ac_runtime_context_create(&cdesc, &s_ctx);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create Vulkan AcContext (code %d)\n", res);
            return nullptr;
        }
    }
    return s_ctx;
}

static AcKernel* ameva_get_mul_mat_kernel() {
    static AcKernel* s_kernel = nullptr;
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_kernel) {
        AcContext* ctx = ameva_get_ac_context();
        if (!ctx) return nullptr;

        static const AcRuntimeBindingDesc kBindings[3] = {
            { 0, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
            { 1, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE },
            { 2, AC_RUNTIME_BINDING_STORAGE_BUFFER, AC_STAGE_COMPUTE }
        };

        AcKernelRuntimeMetadata meta = {
            "ggml_mul_mat",
            "main",
            3,
            kBindings,
            24, // 6 x uint32: M, K, N, offset_src0, offset_src1, offset_dst
            { 16, 16, 1 },
            0,
            kGgmlMulMatSpvSha256,
            0,
            nullptr
        };

        AcKernelDesc kdesc = {
            AC_ABI_VERSION,
            sizeof(AcKernelDesc),
            kGgmlMulMatSpv,
            kGgmlMulMatSpvByteSize,
            &meta,
            0,
            nullptr
        };

        AcResult res = ac_runtime_kernel_create(ctx, &kdesc, &s_kernel);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create mul_mat AcKernel (code %d)\n", res);
            return nullptr;
        }
    }
    return s_kernel;
}

// GUID definition
static ggml_guid_t ggml_backend_ameva_guid() {
    static ggml_guid guid = {
        0x41, 0x4d, 0x45, 0x56, 0x41, 0x2d, 0x43, 0x4f,
        0x4d, 0x50, 0x55, 0x54, 0x45, 0x30, 0x31, 0x00
    };
    return &guid;
}

bool ggml_backend_is_ameva(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_ameva_guid());
}

struct ggml_backend_ameva_context {
    int device_id;
};

static const char* ggml_backend_ameva_name(ggml_backend_t backend) {
    (void)backend;
    return GGML_AMEVA_NAME;
}

static void ggml_backend_ameva_free(ggml_backend_t backend) {
    if (!backend) return;
    delete (ggml_backend_ameva_context*)backend->context;
    delete backend;
}

static void ggml_backend_ameva_synchronize(ggml_backend_t backend) {
    (void)backend;
    // Dispatches are synchronously synchronized on completion
}

struct ggml_backend_ameva_buffer_context {
    AcBuffer* ac_buf;
    void* mapped_ptr;
    size_t size;
};

static enum ggml_status ggml_backend_ameva_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    (void)backend;
    if (!cgraph) return GGML_STATUS_FAILED;

    AcContext* ctx = ameva_get_ac_context();
    AcKernel* kernel = ameva_get_mul_mat_kernel();
    if (!ctx || !kernel) {
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;

        if (node->op != GGML_OP_MUL_MAT) {
            fprintf(stderr, "ggml-ameva: node %d is unsupported op %d (FAIL-FAST)\n", i, node->op);
            return GGML_STATUS_FAILED;
        }

        struct ggml_tensor* src0 = node->src[0];
        struct ggml_tensor* src1 = node->src[1];
        if (!src0 || !src1) return GGML_STATUS_FAILED;

        if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
            fprintf(stderr, "ggml-ameva: node %d unsupported dtype (FAIL-FAST)\n", i);
            return GGML_STATUS_FAILED;
        }

        uint32_t K = (uint32_t)src0->ne[0];
        uint32_t N = (uint32_t)src0->ne[1];
        uint32_t M = (uint32_t)src1->ne[1];

        // Resolve AcBuffers
        AcBuffer* buf_src0 = nullptr;
        AcBuffer* buf_src1 = nullptr;
        AcBuffer* buf_dst  = nullptr;
        uint32_t off_src0 = 0;
        uint32_t off_src1 = 0;
        uint32_t off_dst  = 0;

        AcBuffer* temp_src0 = nullptr;
        AcBuffer* temp_src1 = nullptr;
        AcBuffer* temp_dst  = nullptr;

        // Helper to resolve or wrap buffer
        auto resolve_buf = [&](struct ggml_tensor* t, AcBuffer*& out_buf, uint32_t& out_off, AcBuffer*& temp_buf, bool is_input) -> bool {
            if (t->buffer && strcmp(ggml_backend_buffer_name(t->buffer), GGML_AMEVA_NAME) == 0) {
                auto* bctx = (ggml_backend_ameva_buffer_context*)t->buffer->context;
                out_buf = bctx->ac_buf;
                out_off = (uint32_t)(((const char*)t->data - (const char*)bctx->mapped_ptr) / sizeof(float));
                return true;
            }
            // Allocate staging buffer for host-resident tensors
            size_t nbytes = ggml_nbytes(t);
            AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (nbytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
            if (ac_runtime_buffer_create(ctx, &bdesc, &temp_buf) != AC_SUCCESS) return false;
            void* map_ptr = nullptr;
            ac_runtime_buffer_get_mapped_ptr(temp_buf, &map_ptr);
            if (is_input) {
                memcpy(map_ptr, t->data, nbytes);
                ac_runtime_buffer_flush(temp_buf, 0, nbytes);
            }
            out_buf = temp_buf;
            out_off = 0;
            return true;
        };

        if (!resolve_buf(src0, buf_src0, off_src0, temp_src0, true) ||
            !resolve_buf(src1, buf_src1, off_src1, temp_src1, true) ||
            !resolve_buf(node, buf_dst,  off_dst,  temp_dst,  false)) {
            if (temp_src0) ac_runtime_buffer_destroy(temp_src0);
            if (temp_src1) ac_runtime_buffer_destroy(temp_src1);
            if (temp_dst)  ac_runtime_buffer_destroy(temp_dst);
            return GGML_STATUS_FAILED;
        }

        // Flush input buffers if they came from AMEVA buffer
        if (!temp_src0) ac_runtime_buffer_flush(buf_src0, off_src0 * sizeof(float), (size_t)K * N * sizeof(float));
        if (!temp_src1) ac_runtime_buffer_flush(buf_src1, off_src1 * sizeof(float), (size_t)M * K * sizeof(float));

        // Push constants: M, K, N, off_src0, off_src1, off_dst
        uint32_t push_constants[6] = { M, K, N, off_src0, off_src1, off_dst };
        AcBuffer* bound_bufs[3] = { buf_src0, buf_src1, buf_dst };

        uint32_t grid_x = (N + 15) / 16;
        uint32_t grid_y = (M + 15) / 16;

        AcLaunchDesc ldesc = {
            AC_ABI_VERSION,
            sizeof(AcLaunchDesc),
            kernel,
            bound_bufs,
            3,
            push_constants,
            sizeof(push_constants),
            { grid_x, grid_y, 1 }
        };

        AcEvent* ev = nullptr;
        AcResult dres = ac_runtime_dispatch(ctx, &ldesc, &ev);
        if (dres != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: dispatch failed with code %d\n", dres);
            if (temp_src0) ac_runtime_buffer_destroy(temp_src0);
            if (temp_src1) ac_runtime_buffer_destroy(temp_src1);
            if (temp_dst)  ac_runtime_buffer_destroy(temp_dst);
            return GGML_STATUS_FAILED;
        }

        ac_runtime_event_wait(ev, 5000000000ULL);
        ac_runtime_event_destroy(ev);

        // Readback / Invalidate
        if (temp_dst) {
            ac_runtime_buffer_invalidate(temp_dst, 0, ggml_nbytes(node));
            void* map_ptr = nullptr;
            ac_runtime_buffer_get_mapped_ptr(temp_dst, &map_ptr);
            memcpy(node->data, map_ptr, ggml_nbytes(node));
            ac_runtime_buffer_destroy(temp_dst);
        } else {
            ac_runtime_buffer_invalidate(buf_dst, off_dst * sizeof(float), (size_t)M * N * sizeof(float));
        }

        if (temp_src0) ac_runtime_buffer_destroy(temp_src0);
        if (temp_src1) ac_runtime_buffer_destroy(temp_src1);

        // Increment verified GPU op counter
        ameva_increment_gpu_op_count();
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_ameva_i = {
    /* .get_name                = */ ggml_backend_ameva_name,
    /* .free                    = */ ggml_backend_ameva_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_ameva_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_ameva_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

ggml_backend_t ggml_backend_ameva_init(size_t dev_num) {
    (void)dev_num;
    auto* ctx = new ggml_backend_ameva_context{ 0 };
    auto* backend = new ggml_backend{
        /* .guid    = */ ggml_backend_ameva_guid(),
        /* .iface   = */ ggml_backend_ameva_i,
        /* .device  = */ ggml_backend_dev_by_name(GGML_AMEVA_DEVICE_NAME),
        /* .context = */ ctx,
    };
    return backend;
}
