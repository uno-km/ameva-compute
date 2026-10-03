#include "ggml-ameva.h"
#include "ggml-ameva-impl.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <mutex>

// Diagnostics Counters
static std::atomic<int64_t> g_ameva_supports_query_count{0};
static std::atomic<int64_t> g_ameva_supported_op_count{0};
static std::atomic<int64_t> g_ameva_gpu_op_count{0};
static std::atomic<int64_t> g_ameva_q5_0_cpu_dequant_count{0};
static std::atomic<int64_t> g_ameva_f32_gpu_sgemm_dispatch_count{0};
static std::atomic<int64_t> g_ameva_context_create_count{0};
static std::atomic<int64_t> g_ameva_context_destroy_count{0};
static std::atomic<int64_t> g_ameva_kernel_create_count{0};
static std::atomic<int64_t> g_ameva_kernel_destroy_count{0};
static std::atomic<int64_t> g_ameva_buffer_create_count{0};
static std::atomic<int64_t> g_ameva_buffer_destroy_count{0};
static std::atomic<int64_t> g_ameva_event_create_count{0};
static std::atomic<int64_t> g_ameva_event_destroy_count{0};
static std::atomic<int64_t> g_ameva_q5_0_staging_alloc_count{0};
static std::atomic<int64_t> g_ameva_q5_0_staging_free_count{0};
static std::atomic<int64_t> g_ameva_repacked_tensor_count{0};
static std::atomic<int64_t> g_ameva_repacked_bytes{0};
static std::atomic<int64_t> g_ameva_qmv_mul_mat_count{0};
static std::atomic<int64_t> g_ameva_mmq_mul_mat_count{0};
static std::atomic<int>     g_ameva_failure_injection{0};

#ifndef GGML_AMEVA_BUILD_COMMIT
#define GGML_AMEVA_BUILD_COMMIT "bcbf3d1a7a47179e279f1085702dcbd1908b877d"
#endif

const char* ggml_backend_ameva_get_build_commit(void) {
    return GGML_AMEVA_BUILD_COMMIT;
}

int64_t ggml_backend_ameva_get_supports_query_count(void) {
    return g_ameva_supports_query_count.load();
}

int64_t ggml_backend_ameva_get_supported_op_count(void) {
    return g_ameva_supported_op_count.load();
}

int64_t ggml_backend_ameva_get_gpu_op_count(void) {
    return g_ameva_gpu_op_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_cpu_dequant_count(void) {
    return g_ameva_q5_0_cpu_dequant_count.load();
}

int64_t ggml_backend_ameva_get_f32_gpu_sgemm_dispatch_count(void) {
    return g_ameva_f32_gpu_sgemm_dispatch_count.load();
}

int64_t ggml_backend_ameva_get_context_create_count(void) {
    return g_ameva_context_create_count.load();
}

int64_t ggml_backend_ameva_get_context_destroy_count(void) {
    return g_ameva_context_destroy_count.load();
}

int64_t ggml_backend_ameva_get_kernel_create_count(void) {
    return g_ameva_kernel_create_count.load();
}

int64_t ggml_backend_ameva_get_kernel_destroy_count(void) {
    return g_ameva_kernel_destroy_count.load();
}

int64_t ggml_backend_ameva_get_buffer_create_count(void) {
    return g_ameva_buffer_create_count.load();
}

int64_t ggml_backend_ameva_get_buffer_destroy_count(void) {
    return g_ameva_buffer_destroy_count.load();
}

int64_t ggml_backend_ameva_get_event_create_count(void) {
    return g_ameva_event_create_count.load();
}

int64_t ggml_backend_ameva_get_event_destroy_count(void) {
    return g_ameva_event_destroy_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_staging_alloc_count(void) {
    return g_ameva_q5_0_staging_alloc_count.load();
}

int64_t ggml_backend_ameva_get_q5_0_staging_free_count(void) {
    return g_ameva_q5_0_staging_free_count.load();
}

int64_t ggml_backend_ameva_get_repacked_tensor_count(void) {
    return g_ameva_repacked_tensor_count.load();
}

int64_t ggml_backend_ameva_get_repacked_bytes(void) {
    return g_ameva_repacked_bytes.load();
}

int64_t ggml_backend_ameva_get_mmq_mul_mat_count(void) {
    return g_ameva_mmq_mul_mat_count.load();
}

int64_t ggml_backend_ameva_get_qmv_mul_mat_count(void) {
    return g_ameva_qmv_mul_mat_count.load();
}

void ggml_backend_ameva_reset_counters(void) {
    g_ameva_supports_query_count.store(0);
    g_ameva_supported_op_count.store(0);
    g_ameva_gpu_op_count.store(0);
    g_ameva_q5_0_cpu_dequant_count.store(0);
    g_ameva_f32_gpu_sgemm_dispatch_count.store(0);
    g_ameva_buffer_create_count.store(0);
    g_ameva_buffer_destroy_count.store(0);
    g_ameva_event_create_count.store(0);
    g_ameva_event_destroy_count.store(0);
    g_ameva_q5_0_staging_alloc_count.store(0);
    g_ameva_q5_0_staging_free_count.store(0);
    g_ameva_qmv_mul_mat_count.store(0);
    g_ameva_mmq_mul_mat_count.store(0);
    // the repacked tensor and byte counts describe stored weights, not activity, and are not reset
}

void ameva_increment_gpu_op_count(void) {
    g_ameva_gpu_op_count.fetch_add(1);
}

void ameva_increment_q5_0_cpu_dequant_count(void) {
    g_ameva_q5_0_cpu_dequant_count.fetch_add(1);
}

void ameva_increment_f32_gpu_sgemm_dispatch_count(void) {
    g_ameva_f32_gpu_sgemm_dispatch_count.fetch_add(1);
}

void ameva_count_mul_mat(int64_t n) {
    g_ameva_f32_gpu_sgemm_dispatch_count.fetch_add(n);
}

void ameva_count_mmq_mul_mat(int64_t n) {
    g_ameva_mmq_mul_mat_count.fetch_add(n);
}

void ameva_count_qmv_mul_mat(int64_t n) {
    g_ameva_qmv_mul_mat_count.fetch_add(n);
}

void ameva_count_repacked_tensor(uint64_t bytes) {
    g_ameva_repacked_tensor_count.fetch_add(1);
    g_ameva_repacked_bytes.fetch_add((int64_t)bytes);
}

void ameva_increment_context_create_count(void) {
    g_ameva_context_create_count.fetch_add(1);
}

void ameva_increment_context_destroy_count(void) {
    g_ameva_context_destroy_count.fetch_add(1);
}

void ameva_increment_kernel_create_count(void) {
    g_ameva_kernel_create_count.fetch_add(1);
}

void ameva_increment_kernel_destroy_count(void) {
    g_ameva_kernel_destroy_count.fetch_add(1);
}

void ameva_increment_buffer_create_count(void) {
    g_ameva_buffer_create_count.fetch_add(1);
}

void ameva_increment_buffer_destroy_count(void) {
    g_ameva_buffer_destroy_count.fetch_add(1);
}

void ameva_increment_event_create_count(void) {
    g_ameva_event_create_count.fetch_add(1);
}

void ameva_increment_event_destroy_count(void) {
    g_ameva_event_destroy_count.fetch_add(1);
}

void ameva_increment_q5_0_staging_alloc_count(void) {
    g_ameva_q5_0_staging_alloc_count.fetch_add(1);
}

void ameva_increment_q5_0_staging_free_count(void) {
    g_ameva_q5_0_staging_free_count.fetch_add(1);
}

void ggml_backend_ameva_set_failure_injection(AmevaFailureInjectionPoint point) {
    g_ameva_failure_injection.store((int)point);
}

AmevaFailureInjectionPoint ggml_backend_ameva_get_failure_injection(void) {
    return (AmevaFailureInjectionPoint)g_ameva_failure_injection.load();
}

ggml_backend_t ameva_backend_create(ggml_backend_dev_t dev); // ggml-ameva.cpp

static const char* ggml_backend_ameva_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_NAME;
}

static const char* ggml_backend_ameva_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_DESCRIPTION;
}

// UMA: the GPU allocates from system RAM, so system memory is the device memory. MemAvailable (not MemFree)
// includes reclaimable page cache. Both values are 0 only when /proc/meminfo cannot be read; llama.cpp treats
// 0/0 as "unknown" and uses host memory instead, and the failure is printed once.
static void ggml_backend_ameva_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    (void)dev;
    unsigned long long total_kb = 0, avail_kb = 0;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long v = 0;
            if (sscanf(line, "MemTotal: %llu kB", &v) == 1) total_kb = v;
            else if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
        }
        fclose(f);
    }
    if (total_kb == 0 || avail_kb == 0) {
        // never report total without free: free=0 with total>0 would make llama.cpp's split points 0/0
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            fprintf(stderr, "ggml-ameva: /proc/meminfo has no MemTotal/MemAvailable; device memory reported as unknown\n");
        total_kb = avail_kb = 0;
    }
    if (free)  *free  = (size_t)(avail_kb * 1024);
    if (total) *total = (size_t)(total_kb * 1024);
}

static enum ggml_backend_dev_type ggml_backend_ameva_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ameva_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    if (!props) return;
    props->name = GGML_AMEVA_DEVICE_NAME;
    props->description = GGML_AMEVA_DEVICE_DESCRIPTION;
    ggml_backend_ameva_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_backend_ameva_device_init(ggml_backend_dev_t dev, const char* params) {
    (void)params;
    return ameva_backend_create(dev);
}

static ggml_backend_buffer_type_t ggml_backend_ameva_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return ggml_backend_ameva_buffer_type(0);
}

const char* ameva_op_unsupported_reason(const struct ggml_tensor* op); // ggml-ameva.cpp
void ameva_audit_decline(const struct ggml_tensor* op, const char* why); // ggml-ameva.cpp

// One contract for planning and execution: graph_compute rejects exactly what this rejects. Metadata-only nodes
// (graph inputs, views) are accepted: they compute nothing on any backend, and declining them made the scheduler
// put views on the CPU between AMEVA nodes (extra splits) and made ggml's test-backend-ops skip every case, because
// it asks about each tensor of a test graph, inputs included.
static bool ggml_backend_ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    g_ameva_supports_query_count.fetch_add(1);
    if (const char* why = ameva_op_unsupported_reason(op)) {
        ameva_audit_decline(op, why);
        return false;
    }
    g_ameva_supported_op_count.fetch_add(1);
    return true;
}

// AMEVA buffers only. The kernels bind Vulkan buffers; a tensor in host memory of another backend cannot be bound,
// so the scheduler must copy such inputs into AMEVA buffers before a split runs (which it does exactly when this
// returns false). The previous engine accepted plain "CPU" buffers and staged them itself, one copy per node.
static bool ggml_backend_ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return buft == ggml_backend_ameva_buffer_type(0);
}

static bool ggml_backend_ameva_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    (void)op;
    return false; // Zero unsolicited offload
}

// Designated initializers throughout: ggml-backend-impl.h inserts members into these vtables between releases,
// and a positional list silently shifts every later entry into the wrong slot when the headers change.
static const struct ggml_backend_device_i ggml_backend_ameva_device_i = {
    .get_name        = ggml_backend_ameva_device_get_name,
    .get_description = ggml_backend_ameva_device_get_description,
    .get_memory      = ggml_backend_ameva_device_get_memory,
    .get_type        = ggml_backend_ameva_device_get_type,
    .get_props       = ggml_backend_ameva_device_get_props,
    .init_backend    = ggml_backend_ameva_device_init,
    .get_buffer_type = ggml_backend_ameva_device_get_buffer_type,
    .supports_op     = ggml_backend_ameva_device_supports_op,
    .supports_buft   = ggml_backend_ameva_device_supports_buft,
    .offload_op      = ggml_backend_ameva_device_offload_op,
};

// Backend Registry Interface
static const char* ggml_backend_ameva_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return GGML_AMEVA_NAME;
}

static size_t ggml_backend_ameva_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static ggml_backend_dev_t ggml_backend_ameva_reg_get_device(ggml_backend_reg_t reg, size_t index);

static const struct ggml_backend_reg_i ggml_backend_ameva_reg_i = {
    .get_name         = ggml_backend_ameva_reg_get_name,
    .get_device_count = ggml_backend_ameva_reg_get_device_count,
    .get_device       = ggml_backend_ameva_reg_get_device,
    .get_proc_address = nullptr,
};

static struct ggml_backend_reg g_ameva_reg = {
    .api_version = GGML_BACKEND_API_VERSION,
    .iface       = ggml_backend_ameva_reg_i,
    .context     = nullptr,
};

// The device is linked to its registry statically: a backend created without going through the registry
// (ggml_backend_ameva_init) must still report a device whose reg is valid.
__attribute__((visibility("hidden"))) struct ggml_backend_device g_ameva_device = {
    .iface   = ggml_backend_ameva_device_i,
    .reg     = &g_ameva_reg,
    .context = nullptr,
};

static ggml_backend_dev_t ggml_backend_ameva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    return index == 0 ? &g_ameva_device : nullptr;
}

// ggml-backend-impl.h is not ABI-versioned: GGML_BACKEND_API_VERSION stayed 2 while ggml_backend_i and
// ggml_backend_buffer_i each gained two members (set/get_tensor_2d[_async]). A backend compiled against one
// ggml and loaded by a process running another reads every field behind those vtables at a shifted offset.
// Galaxy S25: libggml-ameva.so compiled against newer headers, loaded by llama.cpp b8471 (ggml 0.9.8), which
// read backend->device from the slot of iface.event_wait (NULL) and stopped in GGML_ASSERT(device).
// Before any AMEVA object reaches libggml-base, check through libggml-base's own accessors that both sides
// agree on every layout and enum value this backend relies on. Returns NULL when they agree.
const char* ameva_abi_check(void) {
    static std::once_flag once;
    static char msg[320];
    static const char* result = nullptr;
    std::call_once(once, [] {
        const char* what = nullptr;

        struct ggml_backend pb = {};
        pb.device = &g_ameva_device;
        struct ggml_backend_buffer_type pbt = {};
        pbt.device = &g_ameva_device;
        struct ggml_backend_buffer pbuf = {};
        pbuf.buft = &pbt;
        pbuf.size = 0x5A17;
        struct ggml_backend_device pd = {};
        pd.reg = &g_ameva_reg;

        if (ggml_backend_get_device(&pb) != &g_ameva_device) {
            what = "struct ggml_backend (ggml_backend_i member count)";
        } else if (ggml_backend_buffer_get_size(&pbuf) != 0x5A17 || ggml_backend_buffer_get_type(&pbuf) != &pbt) {
            what = "struct ggml_backend_buffer (ggml_backend_buffer_i member count)";
        } else if (ggml_backend_buft_get_device(&pbt) != &g_ameva_device) {
            what = "struct ggml_backend_buffer_type";
        } else if (ggml_backend_dev_backend_reg(&pd) != &g_ameva_reg) {
            what = "struct ggml_backend_device";
        }
        // every enum value the graph engine switches on must mean the same op/type in the loaded libggml-base
        static const struct { enum ggml_op op; const char* name; } kOps[] = {
            {GGML_OP_NONE, "NONE"}, {GGML_OP_DUP, "DUP"}, {GGML_OP_ADD, "ADD"}, {GGML_OP_SUB, "SUB"},
            {GGML_OP_MUL, "MUL"}, {GGML_OP_DIV, "DIV"}, {GGML_OP_RMS_NORM, "RMS_NORM"}, {GGML_OP_MUL_MAT, "MUL_MAT"},
            {GGML_OP_SCALE, "SCALE"}, {GGML_OP_CPY, "CPY"}, {GGML_OP_CONT, "CONT"}, {GGML_OP_RESHAPE, "RESHAPE"},
            {GGML_OP_VIEW, "VIEW"}, {GGML_OP_PERMUTE, "PERMUTE"}, {GGML_OP_TRANSPOSE, "TRANSPOSE"},
            {GGML_OP_GET_ROWS, "GET_ROWS"}, {GGML_OP_SET_ROWS, "SET_ROWS"}, {GGML_OP_ROPE, "ROPE"},
            {GGML_OP_FLASH_ATTN_EXT, "FLASH_ATTN_EXT"}, {GGML_OP_UNARY, "UNARY"}, {GGML_OP_GLU, "GLU"}};
        static const struct { enum ggml_type type; const char* name; } kTypes[] = {
            {GGML_TYPE_F32, "f32"}, {GGML_TYPE_F16, "f16"}, {GGML_TYPE_I32, "i32"}, {GGML_TYPE_I64, "i64"},
            {GGML_TYPE_Q5_0, "q5_0"}, {GGML_TYPE_Q4_K, "q4_K"}};
        for (const auto& o : kOps) {
            if (!what && strcmp(ggml_op_name(o.op), o.name) != 0) what = "enum ggml_op";
        }
        for (const auto& t : kTypes) {
            if (!what && strcmp(ggml_type_name(t.type), t.name) != 0) what = "enum ggml_type";
        }
        if (!what) {
            // struct ggml_tensor: build a MUL_MAT node with libggml-base and read it back with our layout
            struct ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
            struct ggml_context* tctx = ggml_init(ip);
            if (!tctx) {
                what = "ggml_init for the tensor layout probe";
            } else {
                struct ggml_tensor* a = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, 8, 3);
                struct ggml_tensor* b = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, 8, 2);
                struct ggml_tensor* c = ggml_mul_mat(tctx, a, b);
                if (!c || c->op != GGML_OP_MUL_MAT || c->src[0] != a || c->src[1] != b || c->type != GGML_TYPE_F32 ||
                    c->ne[0] != 3 || c->ne[1] != 2 || a->nb[1] != 32 || ggml_nbytes(c) != 24)
                    what = "struct ggml_tensor";
                ggml_free(tctx);
            }
        }
        if (what) {
            snprintf(msg, sizeof(msg),
                     "ABI mismatch with the loaded libggml-base: %s differs from the ggml headers libggml-ameva.so "
                     "was compiled with (offsetof(ggml_backend, device)=%zu). Rebuild it against the ggml of the "
                     "llama.cpp build that loads it.", what, offsetof(struct ggml_backend, device));
            result = msg;
        }
    });
    return result;
}

ggml_backend_reg_t ggml_backend_ameva_reg(void) {
    return &g_ameva_reg;
}

void ggml_backend_ameva_register(void) {
    if (const char* why = ameva_abi_check()) {
        fprintf(stderr, "ggml-ameva: not registering the AMEVA backend: %s\n", why);
        return;
    }
    ggml_backend_register(ggml_backend_ameva_reg());
}
