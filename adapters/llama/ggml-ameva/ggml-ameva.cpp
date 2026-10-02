#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ac_runtime.h"
#include "ac_dispatch.h"
#include "ggml_mul_mat_spv.h"
#include "ggml_mul_mat_vec_spv.h"
#include "ggml_mul_mat_q5_0_vec_spv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <ctime>

// Weight GPU staging cache: avoids repeated CPU dequant & buffer allocation per token
static std::mutex g_weight_cache_mutex;
static std::unordered_map<const void*, AcBuffer*> g_weight_cache;
// Raw (still quantized) copies of Q5_0 weights that live outside AMEVA buffers, for the native Q5_0 kernel.
// Weights in AMEVA buffers are bound in place and never copied.
static std::unordered_map<const void*, AcBuffer*> g_raw_weight_cache;

// Q4_K dequantization — exported by libggml-base.so
extern "C" void dequantize_row_q4_K(const void * __restrict x, float * __restrict y, int64_t k);

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
// Per-stage timing of graph_compute (AMEVA_PROFILE=1 prints the totals when the plugin unloads)
// =============================================================================
enum AmevaStage { ST_SRC0, ST_SRC1, ST_DST, ST_FLUSH, ST_DISPATCH, ST_WAIT, ST_READBACK, ST_CLEANUP, ST_COUNT };
static const char* const k_stage_names[ST_COUNT] = {
    "src0 (weight cache/dequant)", "src1 (activation staging)", "dst (output buffer)", "resident flush",
    "dispatch (record+submit)", "wait (fence)", "readback/invalidate", "cleanup (temp destroy)"};
static std::atomic<uint64_t> g_stage_ns[ST_COUNT];
static std::atomic<uint64_t> g_prof_ops{0}, g_prof_graphs{0}, g_prof_temp_src1{0}, g_prof_temp_dst{0},
    g_prof_src0_miss{0}, g_prof_flush_bytes{0};
// GPU execution time of each dispatch (timestamp queries); wait minus this is submission + fence latency
static std::atomic<uint64_t> g_prof_gpu_ns{0}, g_prof_gpu_samples{0}, g_prof_gpu_unavailable{0};

static inline uint64_t ameva_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void ameva_profile_report(void) {
    const char* e = getenv("AMEVA_PROFILE");
    if (!e || strcmp(e, "1") != 0) return;
    const uint64_t ops = g_prof_ops.load();
    uint64_t total = 0;
    for (int i = 0; i < ST_COUNT; ++i) total += g_stage_ns[i].load();
    fprintf(stderr, "AMEVA_PROFILE graphs=%llu ops=%llu total_ms=%.1f per_op_us=%.1f temp_src1=%llu temp_dst=%llu "
                    "src0_cache_miss=%llu resident_flush_MiB=%.1f\n",
            (unsigned long long)g_prof_graphs.load(), (unsigned long long)ops, total / 1e6,
            ops ? total / 1e3 / ops : 0.0, (unsigned long long)g_prof_temp_src1.load(),
            (unsigned long long)g_prof_temp_dst.load(), (unsigned long long)g_prof_src0_miss.load(),
            g_prof_flush_bytes.load() / 1048576.0);
    const uint64_t gpu = g_prof_gpu_ns.load(), wait = g_stage_ns[ST_WAIT].load(), gs = g_prof_gpu_samples.load();
    fprintf(stderr, "AMEVA_PROFILE gpu_kernel_ms=%.1f samples=%llu unavailable=%llu per_op_gpu_us=%.1f "
                    "wait_minus_gpu_per_op_us=%.1f\n", gpu / 1e6, (unsigned long long)gs,
            (unsigned long long)g_prof_gpu_unavailable.load(), gs ? gpu / 1e3 / gs : 0.0,
            gs == ops && ops ? (wait > gpu ? (wait - gpu) / 1e3 / ops : 0.0) : -1.0);
    for (int i = 0; i < ST_COUNT; ++i) {
        const uint64_t ns = g_stage_ns[i].load();
        fprintf(stderr, "AMEVA_PROFILE stage=\"%s\" total_ms=%.1f share=%.1f%% per_op_us=%.1f\n", k_stage_names[i],
                ns / 1e6, total ? 100.0 * ns / total : 0.0, ops ? ns / 1e3 / ops : 0.0);
    }
}

// =============================================================================
// Internal AcContext & AcKernel Lifecycle Singletons
// =============================================================================
static AcContext* s_ctx = nullptr;
static AcKernel*  s_kernel = nullptr;     // ggml_mul_mat.comp: tiled GEMM, any M
static AcKernel*  s_kernel_vec = nullptr; // ggml_mul_mat_vec.comp: M == 1 (decode)
static AcKernel*  s_kernel_q5 = nullptr;  // ggml_mul_mat_q5_0_vec.comp: Q5_0 src0 read in place, any M
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

// Both kernels share bindings (src0, src1, dst) and push constants (M, K, N, offset_src0/src1/dst).
static AcKernel* ameva_create_mul_mat_kernel(AcKernel** slot, const char* name, const uint32_t* spv, size_t spv_bytes,
                                             const char* spv_sha256, uint32_t lx, uint32_t ly) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!*slot) {
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
            name,
            "main",
            3,
            kBindings,
            24, // 6 x uint32: M, K, N, offset_src0, offset_src1, offset_dst
            { lx, ly, 1 },
            0,
            spv_sha256,
            0,
            nullptr
        };

        AcKernelDesc kdesc = {
            AC_ABI_VERSION,
            sizeof(AcKernelDesc),
            spv,
            spv_bytes,
            &meta,
            0,
            nullptr
        };

        AcResult res = ac_runtime_kernel_create(s_ctx, &kdesc, slot);
        if (res != AC_SUCCESS) {
            fprintf(stderr, "ggml-ameva: failed to create %s AcKernel (code %d)\n", name, res);
            return nullptr;
        }
        ameva_increment_kernel_create_count();
    }
    return *slot;
}

static AcKernel* ameva_get_mul_mat_kernel() {
    return ameva_create_mul_mat_kernel(&s_kernel, "ggml_mul_mat", kGgmlMulMatSpv, kGgmlMulMatSpvByteSize,
                                       kGgmlMulMatSpvSha256, 16, 16);
}

static AcKernel* ameva_get_mul_mat_vec_kernel() {
    return ameva_create_mul_mat_kernel(&s_kernel_vec, "ggml_mul_mat_vec", kGgmlMulMatVecSpv, kGgmlMulMatVecSpvByteSize,
                                       kGgmlMulMatVecSpvSha256, 64, 1);
}

static AcKernel* ameva_get_mul_mat_q5_0_vec_kernel() {
    return ameva_create_mul_mat_kernel(&s_kernel_q5, "ggml_mul_mat_q5_0_vec", kGgmlMulMatQ50VecSpv,
                                       kGgmlMulMatQ50VecSpvByteSize, kGgmlMulMatQ50VecSpvSha256, 64, 1);
}

// ggml_mul_mat_vec.comp and ggml_mul_mat_q5_0_vec.comp compute 4 rows per workgroup; Vulkan guarantees 65535
// workgroups per dimension.
static constexpr uint32_t kMulMatVecRows = 4;

// AMEVA_Q5_0_NATIVE=0 sends Q5_0 weights through the CPU-dequantized F32 cache instead (A/B measurement); it is
// never chosen implicitly.
static bool ameva_q5_0_native_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("AMEVA_Q5_0_NATIVE");
        return !(e && strcmp(e, "0") == 0);
    }();
    return enabled;
}

// AMEVA_MUL_MAT_VEC=0 forces the tiled GEMM for M == 1 as well (A/B measurement); it is never chosen implicitly.
static bool ameva_mul_mat_vec_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("AMEVA_MUL_MAT_VEC");
        return !(e && strcmp(e, "0") == 0);
    }();
    return enabled;
}

void ggml_backend_ameva_teardown(void) {
    std::lock_guard<std::mutex> lock(s_mutex);
    {
        std::lock_guard<std::mutex> clock(g_weight_cache_mutex);
        for (auto& pair : g_weight_cache) {
            if (pair.second) {
                ac_runtime_buffer_destroy(pair.second);
                ameva_increment_buffer_destroy_count();
            }
        }
        g_weight_cache.clear();
        for (auto& pair : g_raw_weight_cache) {
            if (pair.second) {
                ac_runtime_buffer_destroy(pair.second);
                ameva_increment_buffer_destroy_count();
            }
        }
        g_raw_weight_cache.clear();
    }
    if (s_kernel) {
        ac_runtime_kernel_destroy(s_kernel);
        s_kernel = nullptr;
        ameva_increment_kernel_destroy_count();
    }
    if (s_kernel_vec) {
        ac_runtime_kernel_destroy(s_kernel_vec);
        s_kernel_vec = nullptr;
        ameva_increment_kernel_destroy_count();
    }
    if (s_kernel_q5) {
        ac_runtime_kernel_destroy(s_kernel_q5);
        s_kernel_q5 = nullptr;
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
    g_prof_graphs.fetch_add(1);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;
        const uint64_t t_node0 = ameva_now_ns();

        if (node->op != GGML_OP_MUL_MAT) {
            fprintf(stderr, "ggml-ameva: node %d is unsupported op %d (FAIL-FAST)\n", i, node->op);
            return GGML_STATUS_FAILED;
        }

        struct ggml_tensor* src0 = node->src[0];
        struct ggml_tensor* src1 = node->src[1];
        if (!src0 || !src1) return GGML_STATUS_FAILED;

        // Supported: (F32 x F32 -> F32), (Q5_0 x F32 -> F32), and (Q4_K x F32 -> F32)
        if (node->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
            fprintf(stderr, "ggml-ameva: node %d unsupported dtype combination (FAIL-FAST)\n", i);
            return GGML_STATUS_FAILED;
        }
        if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q5_0 && src0->type != GGML_TYPE_Q4_K) {
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
        bool is_quant_staging = (src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q4_K);

        // Cleanup lambda
        auto cleanup_temps = [&]() {
            if (temp_src0) {
                ac_runtime_buffer_destroy(temp_src0);
                temp_src0 = nullptr;
                ameva_increment_buffer_destroy_count();
                if (is_quant_staging) {
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
        // Native Q5_0: the kernel reads the quantized blocks itself; src0 offsets are then in bytes.
        const bool src0_q5_native = src0->type == GGML_TYPE_Q5_0 && ameva_q5_0_native_enabled() &&
                                    (N + kMulMatVecRows - 1) / kMulMatVecRows <= 65535u && M <= 65535u;
        uint32_t off_src0_bytes = 0;
        if (src0_q5_native) {
            if (src0->buffer && strcmp(ggml_backend_buffer_name(src0->buffer), GGML_AMEVA_NAME) == 0) {
                auto* bctx = (ggml_backend_ameva_buffer_context*)src0->buffer->context;
                const size_t byte_off = (size_t)((const char*)src0->data - (const char*)bctx->mapped_ptr);
                if (byte_off % 4 != 0 || byte_off > UINT32_MAX) {
                    fprintf(stderr, "ggml-ameva: node %d Q5_0 weight at byte offset %zu is not 4-byte aligned or "
                                    "exceeds 32 bits (FAIL-FAST)\n", i, byte_off);
                    return GGML_STATUS_FAILED;
                }
                buf_src0 = bctx->ac_buf;
                off_src0_bytes = (uint32_t)byte_off;
            } else {
                AcBuffer* cached_raw = nullptr;
                {
                    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
                    auto it = g_raw_weight_cache.find(src0->data);
                    if (it != g_raw_weight_cache.end()) cached_raw = it->second;
                }
                if (!cached_raw) {
                    const size_t qbytes = ggml_nbytes(src0);
                    AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (qbytes + 63) & ~(size_t)63,
                                           AC_BUFFER_USAGE_STORAGE, 0 };
                    if (ac_runtime_buffer_create(ctx, &bdesc, &cached_raw) != AC_SUCCESS) {
                        fprintf(stderr, "ggml-ameva: node %d raw Q5_0 staging buffer (%zu bytes) failed\n", i, qbytes);
                        return GGML_STATUS_FAILED;
                    }
                    ameva_increment_buffer_create_count();
                    void* map_ptr = nullptr;
                    if (ac_runtime_buffer_get_mapped_ptr(cached_raw, &map_ptr) != AC_SUCCESS || !map_ptr) {
                        ac_runtime_buffer_destroy(cached_raw);
                        return GGML_STATUS_FAILED;
                    }
                    memcpy(map_ptr, src0->data, qbytes);
                    if (ac_runtime_buffer_flush(cached_raw, 0, qbytes) != AC_SUCCESS) {
                        ac_runtime_buffer_destroy(cached_raw);
                        return GGML_STATUS_FAILED;
                    }
                    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
                    g_raw_weight_cache[src0->data] = cached_raw;
                }
                buf_src0 = cached_raw;
                off_src0_bytes = 0;
            }
        } else if (src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q4_K) {
            // Check weight cache first: instant GPU lookup, zero CPU dequant
            AcBuffer* cached_buf = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
                auto it = g_weight_cache.find(src0->data);
                if (it != g_weight_cache.end()) {
                    cached_buf = it->second;
                }
            }

            if (cached_buf) {
                // Cache HIT: Re-use pre-dequantized GPU weight buffer
                buf_src0 = cached_buf;
                off_src0 = 0;
            } else {
                // Cache MISS: Dequantize once on CPU, upload to GPU, then cache forever
                if (fi == AMEVA_FAIL_MAP_SRC0) {
                    fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: MAP_SRC0_FAIL triggered\n");
                    return GGML_STATUS_FAILED;
                }

                ameva_increment_q5_0_cpu_dequant_count();
                g_prof_src0_miss.fetch_add(1);
                size_t f32_bytes = (size_t)K * N * sizeof(float);
                AcBufferDesc bdesc = { AC_ABI_VERSION, sizeof(AcBufferDesc), (f32_bytes + 63) & ~63, AC_BUFFER_USAGE_STORAGE, 0 };
                AcBuffer* new_buf = nullptr;
                if (ac_runtime_buffer_create(ctx, &bdesc, &new_buf) != AC_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                ameva_increment_buffer_create_count();
                ameva_increment_q5_0_staging_alloc_count();
                void* map_ptr = nullptr;
                AcResult mres = ac_runtime_buffer_get_mapped_ptr(new_buf, &map_ptr);
                if (mres != AC_SUCCESS || !map_ptr) {
                    ac_runtime_buffer_destroy(new_buf);
                    return GGML_STATUS_FAILED;
                }

                size_t quant_row_bytes = ggml_row_size(src0->type, K);
                const char* quant_src_bytes = (const char*)src0->data;
                float* f32_dst_ptr = (float*)map_ptr;

                if (src0->type == GGML_TYPE_Q5_0) {
                    for (uint32_t row = 0; row < N; ++row) {
                        const ameva_block_q5_0* row_blocks = (const ameva_block_q5_0*)(quant_src_bytes + row * quant_row_bytes);
                        ameva_dequantize_row_q5_0(row_blocks, f32_dst_ptr + row * K, K);
                    }
                } else {
                    for (uint32_t row = 0; row < N; ++row) {
                        dequantize_row_q4_K(quant_src_bytes + row * quant_row_bytes, f32_dst_ptr + row * K, K);
                    }
                }

                if (fi == AMEVA_FAIL_FLUSH_SRC0) {
                    fprintf(stderr, "ggml-ameva [FAILURE_INJECTION]: FLUSH_SRC0_FAIL triggered\n");
                    ac_runtime_buffer_destroy(new_buf);
                    return GGML_STATUS_FAILED;
                }

                AcResult fres = ac_runtime_buffer_flush(new_buf, 0, f32_bytes);
                if (fres != AC_SUCCESS) {
                    ac_runtime_buffer_destroy(new_buf);
                    return GGML_STATUS_FAILED;
                }

                {
                    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
                    g_weight_cache[src0->data] = new_buf;
                }
                buf_src0 = new_buf;
                off_src0 = 0;
            }
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

        const uint64_t t_src0 = ameva_now_ns();
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

        const uint64_t t_src1 = ameva_now_ns();
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

        const uint64_t t_dst = ameva_now_ns();
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
            const size_t f_off = src0_q5_native ? (size_t)off_src0_bytes : (size_t)off_src0 * sizeof(float);
            const size_t f_len = src0_q5_native ? ggml_nbytes(src0) : (size_t)K * N * sizeof(float);
            if (ac_runtime_buffer_flush(buf_src0, f_off, f_len) != AC_SUCCESS) {
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

        const uint64_t t_flush = ameva_now_ns();
        if (!temp_src0) g_prof_flush_bytes.fetch_add(src0_q5_native ? (uint64_t)ggml_nbytes(src0) : (uint64_t)K * N * sizeof(float));
        if (!temp_src1) g_prof_flush_bytes.fetch_add((uint64_t)M * K * sizeof(float));
        // Push constants: M, K, N, off_src0, off_src1, off_dst
        uint32_t push_constants[6] = { M, K, N, src0_q5_native ? off_src0_bytes : off_src0, off_src1, off_dst };
        AcBuffer* bound_bufs[3] = { buf_src0, buf_src1, buf_dst };

        // decode (M == 1) runs the matrix-vector kernel; it is created on first use and a failure is fatal
        const bool use_vec = M == 1 && (N + kMulMatVecRows - 1) / kMulMatVecRows <= 65535u && ameva_mul_mat_vec_enabled();
        AcKernel* op_kernel = kernel;
        if (src0_q5_native) {
            op_kernel = ameva_get_mul_mat_q5_0_vec_kernel();
        } else if (use_vec) {
            op_kernel = ameva_get_mul_mat_vec_kernel();
        }
        if (!op_kernel) {
            cleanup_temps();
            return GGML_STATUS_FAILED;
        }
        uint32_t grid_x = (src0_q5_native || use_vec) ? (N + kMulMatVecRows - 1) / kMulMatVecRows : (N + 15) / 16;
        uint32_t grid_y = src0_q5_native ? M : (use_vec ? 1u : (M + 15) / 16);

        AcLaunchDesc ldesc = {
            AC_ABI_VERSION,
            sizeof(AcLaunchDesc),
            op_kernel,
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
        const uint64_t t_disp = ameva_now_ns();

        // Failure Injection: WAIT_TIMEOUT
        uint64_t wait_timeout_ns = (fi == AMEVA_FAIL_WAIT_TIMEOUT) ? 1ULL : 5000000000ULL;
        AcResult wres = ac_runtime_event_wait(ev, wait_timeout_ns);
        if (wres == AC_SUCCESS) {
            uint64_t wall_ns = 0, gpu_ns = 0;
            if (ac_runtime_event_get_timing(ev, &wall_ns, &gpu_ns) == AC_SUCCESS && gpu_ns > 0) {
                g_prof_gpu_ns.fetch_add(gpu_ns);
                g_prof_gpu_samples.fetch_add(1);
            } else {
                g_prof_gpu_unavailable.fetch_add(1);
            }
        }
        ac_runtime_event_destroy(ev);
        ameva_increment_event_destroy_count();
        const uint64_t t_wait = ameva_now_ns();

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

        const uint64_t t_rb = ameva_now_ns();
        if (temp_src1) g_prof_temp_src1.fetch_add(1);
        if (temp_dst) g_prof_temp_dst.fetch_add(1);
        cleanup_temps();
        const uint64_t t_end = ameva_now_ns();
        g_stage_ns[ST_SRC0].fetch_add(t_src0 - t_node0);
        g_stage_ns[ST_SRC1].fetch_add(t_src1 - t_src0);
        g_stage_ns[ST_DST].fetch_add(t_dst - t_src1);
        g_stage_ns[ST_FLUSH].fetch_add(t_flush - t_dst);
        g_stage_ns[ST_DISPATCH].fetch_add(t_disp - t_flush);
        g_stage_ns[ST_WAIT].fetch_add(t_wait - t_disp);
        g_stage_ns[ST_READBACK].fetch_add(t_rb - t_wait);
        g_stage_ns[ST_CLEANUP].fetch_add(t_end - t_rb);
        g_prof_ops.fetch_add(1);

        // Increment verified execution counters ONLY on verified completion
        ameva_increment_f32_gpu_sgemm_dispatch_count();
        ameva_increment_gpu_op_count();
    }

    return GGML_STATUS_SUCCESS;
}

// Designated initializers: see ggml-ameva-device.cpp. synchronize stays NULL (ggml treats it as optional)
// because graph_compute returns only after every dispatch it issued has completed; there is never pending work.
static const struct ggml_backend_i ggml_backend_ameva_i = {
    .get_name      = ggml_backend_ameva_name,
    .free          = ggml_backend_ameva_free,
    .synchronize   = nullptr,
    .graph_compute = ggml_backend_ameva_graph_compute,
};

const char* ameva_abi_check(void);             // ggml-ameva-device.cpp
extern struct ggml_backend_device g_ameva_device; // ggml-ameva-device.cpp

// The backend carries the device it was created for. It used to look the device up by name in the global
// registry, which yields NULL whenever the backend is created before (or without) registration.
ggml_backend_t ameva_backend_create(ggml_backend_dev_t dev) {
    if (const char* why = ameva_abi_check()) {
        fprintf(stderr, "ggml-ameva: cannot create the AMEVA backend: %s\n", why);
        return nullptr;
    }
    if (dev != &g_ameva_device) {
        fprintf(stderr, "ggml-ameva: cannot create the AMEVA backend for foreign device %p\n", (void*)dev);
        return nullptr;
    }
    // the plugin destructor does not run when the loader never dlcloses the backend (llama.cpp b8471 on
    // Android), so the profile report is tied to process exit instead
    static std::once_flag s_report_once;
    std::call_once(s_report_once, [] { std::atexit(ameva_profile_report); });
    auto* backend = new ggml_backend{
        .guid    = ggml_backend_ameva_guid(),
        .iface   = ggml_backend_ameva_i,
        .device  = dev,
        .context = new ggml_backend_ameva_context{ 0 },
    };
    return backend;
}

ggml_backend_t ggml_backend_ameva_init(size_t dev_num) {
    if (dev_num != 0) {
        fprintf(stderr, "ggml-ameva: device %zu does not exist (AMEVA exposes one device, AMEVA0)\n", dev_num);
        return nullptr;
    }
    return ameva_backend_create(&g_ameva_device);
}

// Dynamic Loader Entrypoints
extern "C" {
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
    // NULL makes the loader print "failed to initialize backend from <path>"; a later --device AMEVA0 then
    // fails as an unknown device instead of crashing inside the scheduler.
    if (const char* why = ameva_abi_check()) {
        fprintf(stderr, "ggml-ameva: refusing to load: %s\n", why);
        return nullptr;
    }
    return ggml_backend_ameva_reg();
}

GGML_BACKEND_API int ggml_backend_score(void) {
    return 100;
}
}
