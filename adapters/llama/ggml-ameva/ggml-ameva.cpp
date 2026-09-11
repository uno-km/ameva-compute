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
#include <cmath>
#include <mutex>
#include <vector>

void ameva_increment_gpu_op_count(void);
void ameva_increment_q5_0_cpu_dequant_count(void);
void ameva_increment_f32_gpu_sgemm_dispatch_count(void);
void ameva_increment_context_create_count(void);
void ameva_increment_context_destroy_count(void);
void ameva_increment_kernel_create_count(void);
void ameva_increment_kernel_destroy_count(void);
void ameva_increment_buffer_create_count(void);
void ameva_increment_buffer_destroy_count(void);
void ameva_increment_event_create_count(void);
void ameva_increment_event_destroy_count(void);
void ameva_increment_q5_0_staging_alloc_count(void);
void ameva_increment_q5_0_staging_free_count(void);

// =============================================================================
// Standalone Minimal Q5_0 Decoder (Zero Upstream Private Symbol Dependency)
// =============================================================================
#define AMEVA_QK5_0 32

#pragma pack(push, 2)
typedef struct {
    uint16_t d;                 // 2 bytes fp16 delta
    uint8_t  qh[4];             // 4 bytes 5th bits
    uint8_t  qs[AMEVA_QK5_0 / 2]; // 16 bytes nibbles
} ameva_block_q5_0;
#pragma pack(pop)

static_assert(sizeof(ameva_block_q5_0) == 22, "ameva_block_q5_0 size mismatch");
static_assert(alignof(ameva_block_q5_0) == 2, "ameva_block_q5_0 alignment mismatch");

static inline float ameva_fp16_to_fp32(uint16_t h) {
#if defined(__ARM_NEON) || defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f;
    memcpy(&f, &h, sizeof(h));
    return (float)f;
#else
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000;
    uint32_t two_w = w + w;
    uint32_t exp = (two_w >> 27) & 0x1f;
    if (exp == 0) {
        if (two_w == 0) {
            float res = 0.0f;
            if (sign) res = -0.0f;
            return res;
        }
        while ((two_w & 0x04000000) == 0) {
            two_w <<= 1;
            exp--;
        }
        exp++;
        two_w &= ~0x04000000;
    } else if (exp == 31) {
        uint32_t f_bits = sign | 0x7f800000 | (two_w >> 13);
        float res;
        memcpy(&res, &f_bits, 4);
        return res;
    }
    uint32_t f_bits = sign | ((exp + (127 - 15)) << 23) | (two_w >> 13);
    float res;
    memcpy(&res, &f_bits, 4);
    return res;
#endif
}

static inline void ameva_dequantize_row_q5_0(const ameva_block_q5_0* vx, float* vy, int64_t k) {
    const int nb = (int)(k / AMEVA_QK5_0);
    for (int i = 0; i < nb; i++) {
        const float d = ameva_fp16_to_fp32(vx[i].d);
        uint32_t qh;
        memcpy(&qh, vx[i].qh, sizeof(qh));

        for (int j = 0; j < AMEVA_QK5_0 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = ((vx[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((vx[i].qs[j] >> 4) | xh_1) - 16;

            vy[i * AMEVA_QK5_0 + j + 0] = (float)x0 * d;
            vy[i * AMEVA_QK5_0 + j + AMEVA_QK5_0 / 2] = (float)x1 * d;
        }
    }
}

// =============================================================================
// Internal AcContext & AcKernel Lifecycle Singletons
// =============================================================================
static AcContext* s_ctx = nullptr;
static AcKernel*  s_kernel = nullptr;
static std::mutex s_mutex;

AcContext* ameva_get_ac_context() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_ctx) {
        AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
        AcResult res = ac_runtime_context_create(&cdesc, &s_ctx);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create Vulkan AcContext (code %d)\n", res);
            return nullptr;
        }
        ameva_increment_context_create_count();
    }
    return s_ctx;
}

static AcKernel* ameva_get_mul_mat_kernel() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_kernel) {
        if (!s_ctx) {
            AcContextDesc cdesc = { AC_ABI_VERSION, sizeof(AcContextDesc), AC_BACKEND_VULKAN, 0 };
            AcResult res = ac_runtime_context_create(&cdesc, &s_ctx);
            if (res != AC_SUCCESS) {
                fprintf(stderr, "ggml-ameva: failed to create Vulkan AcContext (code %d)\n", res);
                return nullptr;
            }
            ameva_increment_context_create_count();
        }

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

        AcResult res = ac_runtime_kernel_create(s_ctx, &kdesc, &s_kernel);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create mul_mat AcKernel (code %d)\n", res);
            return nullptr;
        }
        ameva_increment_kernel_create_count();
    }
    return s_kernel;
}

void ggml_backend_ameva_teardown(void) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_kernel) {
        ac_runtime_kernel_destroy(s_kernel);
        s_kernel = nullptr;
        ameva_increment_kernel_destroy_count();
    }
    if (s_ctx) {
        ac_runtime_context_destroy(s_ctx);
        s_ctx = nullptr;
        ameva_increment_context_destroy_count();
    }
}

__attribute__((destructor)) static void ameva_plugin_fini(void) {
    ggml_backend_ameva_teardown();
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

    AmevaFailureInjectionPoint fi = ggml_backend_ameva_get_failure_injection();

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

        // Supported: (F32 x F32 -> F32) and (Q5_0 x F32 -> F32)
        if (node->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
            fprintf(stderr, "ggml-ameva: node %d unsupported dtype combination (FAIL-FAST)\n", i);
            return GGML_STATUS_FAILED;
        }
        if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q5_0) {
            fprintf(stderr, "ggml-ameva: node %d unsupported src0 dtype %d (FAIL-FAST)\n", i, src0->type);
            return GGML_STATUS_FAILED;
        }

        uint32_t K = (uint32_t)src0->ne[0];
        uint32_t N = (uint32_t)src0->ne[1];
        uint32_t M = (uint32_t)src1->ne[1];

        // Failure Injection: TENSOR_RANGE_OOB
        if (fi == AMEVA_FAIL_TENSOR_RANGE_OOB) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: TENSOR_RANGE_OOB triggered\n");
            return GGML_STATUS_FAILED;
        }

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
        bool is_q5_0_staging = (src0->type == GGML_TYPE_Q5_0);

        // Cleanup lambda
        auto cleanup_temps = [&]() {
            if (temp_src0) {
                ac_runtime_buffer_destroy(temp_src0);
                temp_src0 = nullptr;
                ameva_increment_buffer_destroy_count();
                if (is_q5_0_staging) {
                    ameva_increment_q5_0_staging_free_count();
                }
            }
            if (temp_src1) {
                ac_runtime_buffer_destroy(temp_src1);
                temp_src1 = nullptr;
                ameva_increment_buffer_destroy_count();
            }
            if (temp_dst) {
                ac_runtime_buffer_destroy(temp_dst);
                temp_dst = nullptr;
                ameva_increment_buffer_destroy_count();
            }
        };

        // 1. Resolve SRC0
        if (src0->type == GGML_TYPE_Q5_0) {
            // Failure Injection: MAP_SRC0_FAIL
            if (fi == AMEVA_FAIL_MAP_SRC0) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: MAP_SRC0_FAIL triggered\n");
                return GGML_STATUS_FAILED;
            }

            // Q5_0 CPU Dequantization Staging -> F32
            ameva_increment_q5_0_cpu_dequant_count();
            size_t f32_bytes = (size_t)K * N * sizeof(float);
            AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (f32_bytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
            if (ac_runtime_buffer_create(ctx, &bdesc, &temp_src0) != AC_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            ameva_increment_buffer_create_count();
            ameva_increment_q5_0_staging_alloc_count();
            void* map_ptr = nullptr;
            AcResult mres = ac_runtime_buffer_get_mapped_ptr(temp_src0, &map_ptr);
            if (mres != AC_SUCCESS || !map_ptr) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }

            size_t q5_row_bytes = (size_t)(K / 32) * 22;
            const char* q5_src_bytes = (const char*)src0->data;
            float* f32_dst_ptr = (float*)map_ptr;

            for (uint32_t row = 0; row < N; ++row) {
                const ameva_block_q5_0* row_blocks = (const ameva_block_q5_0*)(q5_src_bytes + row * q5_row_bytes);
                ameva_dequantize_row_q5_0(row_blocks, f32_dst_ptr + row * K, K);
            }

            // Failure Injection: FLUSH_SRC0_FAIL
            if (fi == AMEVA_FAIL_FLUSH_SRC0) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC0_FAIL triggered\n");
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }

            AcResult fres = ac_runtime_buffer_flush(temp_src0, 0, f32_bytes);
            if (fres != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            buf_src0 = temp_src0;
            off_src0 = 0;
        } else {
            // F32 SRC0
            if (fi == AMEVA_FAIL_MAP_SRC0) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: MAP_SRC0_FAIL triggered\n");
                return GGML_STATUS_FAILED;
            }
            if (src0->buffer && strcmp(ggml_backend_buffer_name(src0->buffer), GGML_AMEVA_NAME) == 0) {
                auto* bctx = (ggml_backend_ameva_buffer_context*)src0->buffer->context;
                size_t byte_off = (size_t)((const char*)src0->data - (const char*)bctx->mapped_ptr);
                if (byte_off % sizeof(float) != 0 || (byte_off / sizeof(float)) > UINT32_MAX) {
                    return GGML_STATUS_FAILED;
                }
                buf_src0 = bctx->ac_buf;
                off_src0 = (uint32_t)(byte_off / sizeof(float));
            } else {
                size_t nbytes = ggml_nbytes(src0);
                AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (nbytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
                if (ac_runtime_buffer_create(ctx, &bdesc, &temp_src0) != AC_SUCCESS) return GGML_STATUS_FAILED;
                ameva_increment_buffer_create_count();
                void* map_ptr = nullptr;
                if (ac_runtime_buffer_get_mapped_ptr(temp_src0, &map_ptr) != AC_SUCCESS || !map_ptr) {
                    cleanup_temps();
                    return GGML_STATUS_FAILED;
                }
                memcpy(map_ptr, src0->data, nbytes);
                if (fi == AMEVA_FAIL_FLUSH_SRC0) {
                    fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC0_FAIL triggered\n");
                    cleanup_temps();
                    return GGML_STATUS_FAILED;
                }
                if (ac_runtime_buffer_flush(temp_src0, 0, nbytes) != AC_SUCCESS) {
                    cleanup_temps();
                    return GGML_STATUS_FAILED;
                }
                buf_src0 = temp_src0;
                off_src0 = 0;
            }
        }

        // 2. Resolve SRC1 (F32 Activation)
        if (fi == AMEVA_FAIL_MAP_SRC1) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: MAP_SRC1_FAIL triggered\n");
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }
        if (src1->buffer && strcmp(ggml_backend_buffer_name(src1->buffer), GGML_AMEVA_NAME) == 0) {
            auto* bctx = (ggml_backend_ameva_buffer_context*)src1->buffer->context;
            size_t byte_off = (size_t)((const char*)src1->data - (const char*)bctx->mapped_ptr);
            if (byte_off % sizeof(float) != 0 || (byte_off / sizeof(float)) > UINT32_MAX) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            buf_src1 = bctx->ac_buf;
            off_src1 = (uint32_t)(byte_off / sizeof(float));
        } else {
            size_t nbytes = ggml_nbytes(src1);
            AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (nbytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
            if (ac_runtime_buffer_create(ctx, &bdesc, &temp_src1) != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            ameva_increment_buffer_create_count();
            void* map_ptr = nullptr;
            if (ac_runtime_buffer_get_mapped_ptr(temp_src1, &map_ptr) != AC_SUCCESS || !map_ptr) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            memcpy(map_ptr, src1->data, nbytes);
            if (fi == AMEVA_FAIL_FLUSH_SRC1) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC1_FAIL triggered\n");
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            if (ac_runtime_buffer_flush(temp_src1, 0, nbytes) != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            buf_src1 = temp_src1;
            off_src1 = 0;
        }

        // 3. Resolve DST
        if (fi == AMEVA_FAIL_MAP_DST) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: MAP_DST_FAIL triggered\n");
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }
        if (node->buffer && strcmp(ggml_backend_buffer_name(node->buffer), GGML_AMEVA_NAME) == 0) {
            auto* bctx = (ggml_backend_ameva_buffer_context*)node->buffer->context;
            size_t byte_off = (size_t)((const char*)node->data - (const char*)bctx->mapped_ptr);
            if (byte_off % sizeof(float) != 0 || (byte_off / sizeof(float)) > UINT32_MAX) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            buf_dst = bctx->ac_buf;
            off_dst = (uint32_t)(byte_off / sizeof(float));
        } else {
            size_t nbytes = ggml_nbytes(node);
            AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (nbytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
            if (ac_runtime_buffer_create(ctx, &bdesc, &temp_dst) != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            ameva_increment_buffer_create_count();
            void* map_ptr = nullptr;
            if (ac_runtime_buffer_get_mapped_ptr(temp_dst, &map_ptr) != AC_SUCCESS || !map_ptr) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            buf_dst = temp_dst;
            off_dst = 0;
        }

        // Failure Injection: OFFSET_OVERFLOW
        if (fi == AMEVA_FAIL_OFFSET_OVERFLOW) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: OFFSET_OVERFLOW triggered\n");
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }

        // Flush resident buffers if needed
        if (!temp_src0) {
            if (fi == AMEVA_FAIL_FLUSH_SRC0) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC0_FAIL triggered\n");
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            if (ac_runtime_buffer_flush(buf_src0, off_src0 * sizeof(float), (size_t)K * N * sizeof(float)) != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
        }
        if (!temp_src1) {
            if (fi == AMEVA_FAIL_FLUSH_SRC1) {
                fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC1_FAIL triggered\n");
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            if (ac_runtime_buffer_flush(buf_src1, off_src1 * sizeof(float), (size_t)M * K * sizeof(float)) != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
        }

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

        // Failure Injection: DISPATCH_FAIL
        if (fi == AMEVA_FAIL_DISPATCH) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: DISPATCH_FAIL triggered\n");
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }

        AcEvent* ev = nullptr;
        AcResult dres = ac_runtime_dispatch(ctx, &ldesc, &ev);
        if (dres != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: dispatch failed with code %d\n", dres);
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }
        if (ev) {
            ameva_increment_event_create_count();
        }

        // Failure Injection: WAIT_TIMEOUT
        uint64_t wait_timeout_ns = (fi == AMEVA_FAIL_WAIT_TIMEOUT) ? 1ULL : 5000000000ULL;
        AcResult wres = ac_runtime_event_wait(ev, wait_timeout_ns);
        ac_runtime_event_destroy(ev);
        ameva_increment_event_destroy_count();

        if (wres != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: event wait failed with code %d (FAIL-FAST)\n", wres);
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }

        // Failure Injection: INVALIDATE_DST_FAIL
        if (fi == AMEVA_FAIL_INVALIDATE_DST) {
            fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: INVALIDATE_DST_FAIL triggered\n");
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }

        // Readback / Invalidate
        if (temp_dst) {
            AcResult ires = ac_runtime_buffer_invalidate(temp_dst, 0, ggml_nbytes(node));
            if (ires != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            void* map_ptr = nullptr;
            AcResult mres = ac_runtime_buffer_get_mapped_ptr(temp_dst, &map_ptr);
            if (mres != AC_SUCCESS || !map_ptr) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
            memcpy(node->data, map_ptr, ggml_nbytes(node));
        } else {
            AcResult ires = ac_runtime_buffer_invalidate(buf_dst, off_dst * sizeof(float), (size_t)M * N * sizeof(float));
            if (ires != AC_SUCCESS) {
                cleanup_temps();
                return GGML_STATUS_FAILED;
            }
        }

        cleanup_temps();

        // Increment verified execution counters ONLY on verified completion
        ameva_increment_f32_gpu_sgemm_dispatch_count();
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

// Dynamic Loader Entrypoints
extern "C" {
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
    return ggml_backend_ameva_reg();
}

GGML_BACKEND_API int ggml_backend_score(void) {
    return 100;
}
}
