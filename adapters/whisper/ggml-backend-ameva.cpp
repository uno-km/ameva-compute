/**
 * @file ggml-backend-ameva.cpp
 * @brief Formal AMEVA Compute Hardware Backend Implementation for GGML / whisper.cpp
 */

#include "ggml-backend-ameva.h"
#include "../../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h"

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <mutex>

#define UNUSED(x) (void)(x)

static AmevaBackendCounters g_counters = {};
static bool s_bounds_violation = false;

// Geometry constants for target Whisper Tiny encoder projection
static const uint32_t TARGET_M = 1500;
static const uint32_t TARGET_N = 384;
static const uint32_t TARGET_K = 384;

static const size_t WEIGHT_BYTES = (size_t)TARGET_K * TARGET_N * sizeof(uint16_t); // 294,912
static const size_t ACT_BYTES    = (size_t)TARGET_K * TARGET_M * sizeof(float);    // 2,304,000
static const size_t OUT_BYTES    = (size_t)TARGET_N * TARGET_M * sizeof(float);    // 2,304,000

struct MatMulPushConstants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

// Hardware Vulkan resources
static AcVkContext * s_vk_ctx = nullptr;
static AcVkKernel  * s_vk_kernel = nullptr;
static AcVkBuffer  * s_buf_w = nullptr;
static AcVkBuffer  * s_buf_a = nullptr;
static AcVkBuffer  * s_buf_o = nullptr;
static std::mutex    s_hw_mutex;

// ============================================================================
// Hardware Vulkan Pipeline Management
// ============================================================================

static bool init_hardware_pipeline() {
    std::lock_guard<std::mutex> lock(s_hw_mutex);
    if (s_vk_ctx != nullptr) return true;

    AcResult res = ac_vk_context_create(&s_vk_ctx);
    if (res != AC_SUCCESS || !s_vk_ctx) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_context_create failed with code %d\n", res);
        return false;
    }

    if (s_vk_ctx->device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Vulkan context bound to CPU fallback! Rejected.\n");
        ac_vk_context_destroy(s_vk_ctx);
        s_vk_ctx = nullptr;
        return false;
    }

    const char * env_path = getenv("AMEVA_SPIRV_PATH");
    const char * spv_path = env_path ? env_path : "/data/data/com.termux/files/home/matmul_f16_f32.spv";
    std::ifstream spv_file(spv_path, std::ios::binary | std::ios::ate);
    if (!spv_file) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Cannot open SPIR-V binary %s\n", spv_path);
        ac_vk_context_destroy(s_vk_ctx);
        s_vk_ctx = nullptr;
        return false;
    }
    size_t spv_size = spv_file.tellg();
    spv_file.seekg(0, std::ios::beg);
    std::vector<uint32_t> spv_code(spv_size / sizeof(uint32_t));
    spv_file.read(reinterpret_cast<char*>(spv_code.data()), spv_size);
    spv_file.close();

    AcKernelMetadata meta = {};
    meta.kernel_name = "matmul_f16_f32";
    meta.entry_point = "main";
    meta.binding_count = 3;
    meta.bindings = nullptr;
    meta.push_constant_size = sizeof(MatMulPushConstants);
    meta.local_size = {16, 16, 1};
    meta.required_features = AC_FEATURE_NONE;
    meta.expected_spirv_sha256 = nullptr;

    res = ac_vk_kernel_create_with_metadata(s_vk_ctx, spv_code.data(), spv_size, &meta, &s_vk_kernel);
    if (res != AC_SUCCESS || !s_vk_kernel) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: ac_vk_kernel_create_with_metadata failed with code %d\n", res);
        ac_vk_context_destroy(s_vk_ctx);
        s_vk_ctx = nullptr;
        return false;
    }

    res = ac_vk_buffer_create(s_vk_ctx, WEIGHT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &s_buf_w);
    if (res != AC_SUCCESS) return false;
    res = ac_vk_buffer_create(s_vk_ctx, ACT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &s_buf_a);
    if (res != AC_SUCCESS) return false;
    res = ac_vk_buffer_create(s_vk_ctx, OUT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &s_buf_o);
    if (res != AC_SUCCESS) return false;

    return true;
}

static bool execute_vulkan_matmul(struct ggml_tensor * t) {
    if (!t || !s_vk_ctx || !s_vk_kernel || !s_buf_w || !s_buf_a || !s_buf_o) {
        return false;
    }

    // 1. Upload input weights and activations to UMA buffers
    if (t->src[0]->buffer) {
        ggml_backend_tensor_get(t->src[0], s_buf_w->mapped_ptr, 0, WEIGHT_BYTES);
    } else if (t->src[0]->data) {
        memcpy(s_buf_w->mapped_ptr, t->src[0]->data, WEIGHT_BYTES);
    } else {
        return false;
    }

    if (t->src[1]->buffer) {
        ggml_backend_tensor_get(t->src[1], s_buf_a->mapped_ptr, 0, ACT_BYTES);
    } else if (t->src[1]->data) {
        memcpy(s_buf_a->mapped_ptr, t->src[1]->data, ACT_BYTES);
    } else {
        return false;
    }

    // 2. Dispatch Vulkan compute pipeline
    MatMulPushConstants pc = { TARGET_M, TARGET_N, TARGET_K };
    AcVkBuffer * bufs[3] = { s_buf_w, s_buf_a, s_buf_o };
    uint32_t group_x = (TARGET_N + 15) / 16; // 24
    uint32_t group_y = (TARGET_M + 15) / 16; // 94
    uint32_t group_z = 1;

    AcVkEvent * event = nullptr;
    AcResult res = ac_vk_dispatch(s_vk_ctx, s_vk_kernel, bufs, 3, &pc, sizeof(pc), group_x, group_y, group_z, &event);
    if (res != AC_SUCCESS || !event) {
        return false;
    }

    // 3. Wait on GPU event
    res = ac_vk_event_wait(event, 5000000000ULL);
    ac_vk_event_destroy(event);
    if (res != AC_SUCCESS) {
        return false;
    }

    // 4. Store result back into destination tensor
    if (t->buffer) {
        ggml_backend_tensor_set(t, s_buf_o->mapped_ptr, 0, OUT_BYTES);
    }
    if (t->data && t->data != s_buf_o->mapped_ptr) {
        memcpy(t->data, s_buf_o->mapped_ptr, OUT_BYTES);
    }

    return true;
}

// ============================================================================
// 1. AMEVA Buffer Implementation with Strict Bounds Defense
// ============================================================================

struct ameva_buffer_context {
    void * ptr;
    size_t size;
};

static bool validate_buffer_bounds(struct ameva_buffer_context * ctx, struct ggml_tensor * tensor, size_t offset, size_t size) {
    if (!ctx || !ctx->ptr || !tensor) {
        s_bounds_violation = true;
        return false;
    }
    if (offset + size < offset) { // Integer overflow
        s_bounds_violation = true;
        return false;
    }
    size_t nbytes = ggml_nbytes(tensor);
    if (offset + size > nbytes) { // Tensor overflow
        s_bounds_violation = true;
        return false;
    }
    if (tensor->data) {
        uintptr_t t_start = (uintptr_t)tensor->data + offset;
        uintptr_t t_end   = t_start + size;
        uintptr_t b_start = (uintptr_t)ctx->ptr;
        uintptr_t b_end   = b_start + ctx->size;
        if (t_start < b_start || t_end > b_end) { // Buffer bounds violation
            s_bounds_violation = true;
            return false;
        }
    }
    return true;
}

static void ameva_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer && buffer->context) {
        struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
        if (ctx->ptr) {
            free(ctx->ptr);
            ctx->ptr = NULL;
        }
        free(ctx);
        buffer->context = NULL;
        g_counters.buffer_free_count++;
    }
}

static void * ameva_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return NULL;
    return ((struct ameva_buffer_context *)buffer->context)->ptr;
}

static void ameva_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (!buffer || !buffer->context) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, tensor, offset, size)) return;
    memset((char *)tensor->data + offset, value, size);
}

static void ameva_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !data) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, tensor, offset, size)) return;
    memcpy((char *)tensor->data + offset, data, size);
}

static void ameva_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !data) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (!validate_buffer_bounds(ctx, const_cast<struct ggml_tensor*>(tensor), offset, size)) return;
    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ameva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) return;
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)buffer->context;
    if (ctx->ptr && ctx->size > 0) {
        memset(ctx->ptr, value, ctx->size);
    }
}

static const struct ggml_backend_buffer_i ameva_backend_buffer_i = {
    /* .free_buffer     = */ ameva_buffer_free_buffer,
    /* .get_base        = */ ameva_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ameva_buffer_memset_tensor,
    /* .set_tensor      = */ ameva_buffer_set_tensor,
    /* .get_tensor      = */ ameva_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ameva_buffer_clear,
    /* .reset           = */ NULL,
};

// ============================================================================
// 2. AMEVA Buffer Type Implementation
// ============================================================================

static const char * ameva_buft_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "AMEVA";
}

static ggml_backend_buffer_t ameva_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ameva_buffer_context * ctx = (struct ameva_buffer_context *)malloc(sizeof(struct ameva_buffer_context));
    if (!ctx) return NULL;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, size != 0 ? size : 64) != 0 || !ptr) {
        free(ctx);
        return NULL;
    }

    ctx->ptr = ptr;
    ctx->size = size;
    g_counters.buffer_alloc_count++;

    return ggml_backend_buffer_init(buft, ameva_backend_buffer_i, ctx, size);
}

static size_t ameva_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ameva_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return (size_t)256 * 1024 * 1024; // 256 MB conservative mobile device upper bound
}

static size_t ameva_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ameva_buft_is_host(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return true; // ARM UMA shared memory architecture
}

static const struct ggml_backend_buffer_type_i ameva_backend_buffer_type_i = {
    /* .get_name       = */ ameva_buft_get_name,
    /* .alloc_buffer   = */ ameva_buft_alloc_buffer,
    /* .get_alignment  = */ ameva_buft_get_alignment,
    /* .get_max_size   = */ ameva_buft_get_max_size,
    /* .get_alloc_size = */ ameva_buft_get_alloc_size,
    /* .is_host        = */ ameva_buft_is_host,
};

ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(void) {
    static struct ggml_backend_buffer_type s_buft = {
        /* .iface   = */ ameva_backend_buffer_type_i,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &s_buft;
}

// ============================================================================
// 3. AMEVA Backend (Stream) Implementation with Real Vulkan Execution
// ============================================================================

struct ameva_backend_context {
    const char * name;
};

static const char * ameva_backend_get_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "AMEVA";
}

static void ameva_backend_free(ggml_backend_t backend) {
    if (backend) {
        if (backend->context) {
            free(backend->context);
            backend->context = NULL;
        }
        free(backend);
    }
}

static void ameva_backend_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    g_counters.sync_count++;
}

static enum ggml_status ameva_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    UNUSED(backend);
    g_counters.graph_compute_call_count++;
    if (!cgraph) return GGML_STATUS_FAILED;
    if (s_bounds_violation) {
        fprintf(stderr, "[AMEVA_BACKEND] FATAL: Prior buffer bounds violation detected. Aborting.\n");
        return GGML_STATUS_FAILED;
    }

    const int n_nodes = ggml_graph_n_nodes(cgraph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (!node) continue;

        if (ggml_backend_dev_supports_op(ggml_backend_ameva_device(), node)) {
            g_counters.ameva_dispatch_attempt_count++;

            // 1. Initialize hardware pipeline if not already active
            if (!init_hardware_pipeline()) {
                g_counters.ameva_dispatch_failure_count++;
                fprintf(stderr, "[AMEVA_BACKEND] FATAL: Hardware pipeline initialization failed\n");
                return GGML_STATUS_FAILED;
            }

            // 2. Perform real Vulkan compute dispatch
            if (!execute_vulkan_matmul(node)) {
                g_counters.ameva_dispatch_failure_count++;
                fprintf(stderr, "[AMEVA_BACKEND] FATAL: Vulkan MatMul dispatch failed\n");
                return GGML_STATUS_FAILED;
            }

            // 3. Strict invariant: ONLY increment on successful GPU execution
            g_counters.ameva_dispatch_success_count++;
            g_counters.ameva_gpu_op_count++;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ameva_backend_i = {
    /* .get_name                = */ ameva_backend_get_name,
    /* .free                    = */ ameva_backend_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ameva_backend_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ameva_backend_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_t ggml_backend_ameva_init(const char * params) {
    UNUSED(params);
    struct ameva_backend_context * ctx = (struct ameva_backend_context *)calloc(1, sizeof(struct ameva_backend_context));
    if (!ctx) return NULL;
    ctx->name = "AMEVA";

    ggml_backend_t backend = (ggml_backend_t)calloc(1, sizeof(struct ggml_backend));
    if (!backend) {
        free(ctx);
        return NULL;
    }

    backend->iface = ameva_backend_i;
    backend->device = ggml_backend_ameva_device();
    backend->context = ctx;

    return backend;
}

// ============================================================================
// 4. AMEVA Device Implementation with Strict Dtype/Shape Validation
// ============================================================================

static const char * ameva_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "AMEVA0";
}

static const char * ameva_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "AMEVA Vulkan STT Hardware Accelerator";
}

static void ameva_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    if (free)  *free  = 0;
    if (total) *total = 0;
}

static enum ggml_backend_dev_type ameva_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ameva_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    UNUSED(dev);
    if (!props) return;
    props->name = "AMEVA0";
    props->description = "AMEVA Vulkan STT Hardware Accelerator";
    props->memory_free = 0;
    props->memory_total = 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = NULL;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ameva_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    return ggml_backend_ameva_init(params);
}

static ggml_backend_buffer_type_t ameva_device_get_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_ameva_buffer_type();
}

static bool ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    g_counters.supports_op_query_count++;
    if (!op) return false;

    // 1. Operation must be GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) return false;

    // 2. Strict Dtype Validation: SRC0=F16, SRC1=F32, DST=F32
    if (!op->src[0] || !op->src[1]) return false;
    if (op->src[0]->type != GGML_TYPE_F16) return false;
    if (op->src[1]->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F32) return false;

    // 3. Strict Shape Validation:
    // SRC0 (Weights):     [384, 384, 1, 1]
    // SRC1 (Activations): [384, 1500, 1, 1]
    // DST  (Output):      [384, 1500, 1, 1]
    if (op->src[0]->ne[0] != 384 || op->src[0]->ne[1] != 384 || op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1) return false;
    if (op->src[1]->ne[0] != 384 || op->src[1]->ne[1] != 1500 || op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1) return false;
    if (op->ne[0] != 384 || op->ne[1] != 1500 || op->ne[2] != 1 || op->ne[3] != 1) return false;

    // 4. Strict Stride & Contiguous Layout Validation
    if (op->src[0]->nb[0] != sizeof(ggml_fp16_t)) return false;
    if (op->src[0]->nb[1] != 384 * sizeof(ggml_fp16_t)) return false;
    if (op->src[1]->nb[0] != sizeof(float)) return false;
    if (op->src[1]->nb[1] != 384 * sizeof(float)) return false;
    if (op->nb[0] != sizeof(float)) return false;
    if (op->nb[1] != 384 * sizeof(float)) return false;

    g_counters.supported_pattern_match_count++;
    return true;
}

static bool ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    UNUSED(dev);
    // Strict ownership: only allow genuine AMEVA buffer type
    return buft == ggml_backend_ameva_buffer_type();
}

static bool ameva_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ameva_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ameva_backend_device_i = {
    /* .get_name              = */ ameva_device_get_name,
    /* .get_description       = */ ameva_device_get_description,
    /* .get_memory            = */ ameva_device_get_memory,
    /* .get_type              = */ ameva_device_get_type,
    /* .get_props             = */ ameva_device_get_props,
    /* .init_backend          = */ ameva_device_init_backend,
    /* .get_buffer_type       = */ ameva_device_get_buffer_type,
    /* .get_host_buffer_type  = */ NULL,
    /* .buffer_from_host_ptr  = */ NULL,
    /* .supports_op           = */ ameva_device_supports_op,
    /* .supports_buft         = */ ameva_device_supports_buft,
    /* .offload_op            = */ ameva_device_offload_op,
    /* .event_new             = */ NULL,
    /* .event_free            = */ NULL,
    /* .event_synchronize     = */ NULL,
};

ggml_backend_dev_t ggml_backend_ameva_device(void) {
    static struct ggml_backend_device s_dev = {
        /* .iface   = */ ameva_backend_device_i,
        /* .reg     = */ NULL,
        /* .context = */ NULL,
    };
    return &s_dev;
}

// ============================================================================
// 5. AMEVA Backend Registration Implementation
// ============================================================================

static const char * ameva_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "AMEVA";
}

static size_t ameva_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ameva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    UNUSED(reg);
    if (index == 0) {
        return ggml_backend_ameva_device();
    }
    return NULL;
}

static const struct ggml_backend_reg_i ameva_backend_reg_i = {
    /* .get_name         = */ ameva_reg_get_name,
    /* .get_device_count = */ ameva_reg_get_device_count,
    /* .get_device       = */ ameva_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_ameva_reg(void) {
    static struct ggml_backend_reg s_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ameva_backend_reg_i,
        /* .context     = */ NULL,
    };
    return &s_reg;
}

void ameva_backend_get_counters(AmevaBackendCounters * out_counters) {
    if (out_counters) {
        *out_counters = g_counters;
    }
}

void ameva_backend_reset_counters(void) {
    g_counters = {};
    s_bounds_violation = false;
}

void ameva_backend_register(void) {
    static std::once_flag s_flag;
    std::call_once(s_flag, []() {
        ggml_backend_reg_t reg = ggml_backend_ameva_reg();
        ggml_backend_dev_t dev = ggml_backend_ameva_device();
        dev->reg = reg;
        ggml_backend_register(reg);
        ggml_backend_device_register(dev);
    });
}
