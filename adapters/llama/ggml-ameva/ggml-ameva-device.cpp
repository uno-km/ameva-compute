#include "ggml-ameva.h"
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

// STRICT CONTRACT: Supports F32xF32->F32 and Q5_0xF32->F32 2D contiguous non-batched MatMul
static bool ggml_backend_ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    g_ameva_supports_query_count.fetch_add(1);
    if (!op) return false;

    // 1. Must be GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    // 2. Strict DTYPE check:
    // Supported:
    // (SRC0 = F32, SRC1 = F32, DST = F32)
    // (SRC0 = Q5_0, SRC1 = F32, DST = F32)
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q5_0 && src0->type != GGML_TYPE_Q4_K) {
        return false;
    }

    // 3. CONTIGUOUS_ONLY = true
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }

    // 4. BATCH_DIMENSIONS_SUPPORTED = false (ne[2] == 1 && ne[3] == 1)
    if (op->ne[2] != 1 || op->ne[3] != 1 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }

    // 5. Inner dimension match: K dimension must match
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    int64_t K = src0->ne[0];
    int64_t N = src0->ne[1];

    // 6. Q5_0 alignment and shape contract checks
    if (src0->type == GGML_TYPE_Q5_0) {
        if (K % 32 != 0) {
            return false;
        }
        size_t expected_q5_row_bytes = (size_t)(K / 32) * 22;
        if (ggml_row_size(src0->type, K) != expected_q5_row_bytes) {
            return false;
        }
    }

    // 6b. Q4_K alignment: super-block size is 256 elements
    if (src0->type == GGML_TYPE_Q4_K) {
        if (K % 256 != 0) {
            return false;
        }
    }

    // 7. F32 row bytes check
    if (ggml_row_size(src1->type, K) != (size_t)K * sizeof(float)) {
        return false;
    }
    if (ggml_row_size(op->type, N) != (size_t)N * sizeof(float)) {
        return false;
    }

    // Approved: increment supported op count
    g_ameva_supported_op_count.fetch_add(1);
    return true;
}

static bool ggml_backend_ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    // Accept AMEVA native buffers
    if (buft == ggml_backend_ameva_buffer_type(0)) return true;
    // Accept CPU host buffers — enables hybrid execution where CPU-resident
    // tensors (quantized weights) are dequantized on CPU, then GPU matmul
    // operates on the resulting F32 data via temporary staging buffers.
    const char* name = ggml_backend_buft_name(buft);
    if (name && (strcmp(name, "CPU") == 0 || strncmp(name, "CPU", 3) == 0)) return true;
    return false;
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
        } else if (strcmp(ggml_op_name(GGML_OP_MUL_MAT), "MUL_MAT") != 0) {
            what = "enum ggml_op (GGML_OP_MUL_MAT)";
        } else if (strcmp(ggml_type_name(GGML_TYPE_F32), "f32") != 0 || strcmp(ggml_type_name(GGML_TYPE_Q5_0), "q5_0") != 0 ||
                   strcmp(ggml_type_name(GGML_TYPE_Q4_K), "q4_K") != 0) {
            what = "enum ggml_type";
        } else {
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
