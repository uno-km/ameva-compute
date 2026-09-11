#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>

// Diagnostics Counters
static std::atomic<int64_t> g_ameva_supported_op_count{0};
static std::atomic<int64_t> g_ameva_gpu_op_count{0};

int64_t ggml_backend_ameva_get_supported_op_count(void) {
    return g_ameva_supported_op_count.load();
}

int64_t ggml_backend_ameva_get_gpu_op_count(void) {
    return g_ameva_gpu_op_count.load();
}

void ggml_backend_ameva_reset_counters(void) {
    g_ameva_supported_op_count.store(0);
    g_ameva_gpu_op_count.store(0);
}

void ameva_increment_gpu_op_count(void) {
    g_ameva_gpu_op_count.fetch_add(1);
}

static const char* ggml_backend_ameva_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_NAME;
}

static const char* ggml_backend_ameva_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_AMEVA_DEVICE_DESCRIPTION;
}

static void ggml_backend_ameva_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    (void)dev;
    if (free)  *free  = (size_t)3 * 1024 * 1024 * 1024ULL;
    if (total) *total = (size_t)4 * 1024 * 1024 * 1024ULL;
}

static enum ggml_backend_dev_type ggml_backend_ameva_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ameva_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    (void)dev;
    if (!props) return;
    props->name = GGML_AMEVA_DEVICE_NAME;
    props->description = GGML_AMEVA_DEVICE_DESCRIPTION;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_backend_ameva_device_init(ggml_backend_dev_t dev, const char* params) {
    (void)dev;
    (void)params;
    return ggml_backend_ameva_init(0);
}

static ggml_backend_buffer_type_t ggml_backend_ameva_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return ggml_backend_ameva_buffer_type(0);
}

// STRICT CONTRACT: Only approved single combination returns true
static bool ggml_backend_ameva_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
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

    // 2. Strict DTYPE check: SRC0 = F32, SRC1 = F32, DST = F32
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
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

    // 5. TRANSPOSE_SUPPORTED = false (standard 2D layout where inner dim matches)
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    // Approved: increment supported op count
    g_ameva_supported_op_count.fetch_add(1);
    return true;
}

static bool ggml_backend_ameva_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return buft == ggml_backend_ameva_buffer_type(0);
}

static bool ggml_backend_ameva_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    (void)op;
    return false; // Zero unsolicited offload
}

static const struct ggml_backend_device_i ggml_backend_ameva_device_i = {
    /* .get_name             = */ ggml_backend_ameva_device_get_name,
    /* .get_description      = */ ggml_backend_ameva_device_get_description,
    /* .get_memory           = */ ggml_backend_ameva_device_get_memory,
    /* .get_type             = */ ggml_backend_ameva_device_get_type,
    /* .get_props            = */ ggml_backend_ameva_device_get_props,
    /* .init_backend         = */ ggml_backend_ameva_device_init,
    /* .get_buffer_type      = */ ggml_backend_ameva_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_ameva_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ameva_device_supports_buft,
    /* .offload_op           = */ ggml_backend_ameva_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static struct ggml_backend_device g_ameva_device = {
    /* .iface   = */ ggml_backend_ameva_device_i,
    /* .reg     = */ nullptr,
    /* .context = */ nullptr,
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

static ggml_backend_dev_t ggml_backend_ameva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    if (index == 0) {
        g_ameva_device.reg = reg;
        return &g_ameva_device;
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_ameva_reg_i = {
    /* .get_name         = */ ggml_backend_ameva_reg_get_name,
    /* .get_device_count = */ ggml_backend_ameva_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ameva_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static struct ggml_backend_reg g_ameva_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_ameva_reg_i,
    /* .context     = */ nullptr,
};

ggml_backend_reg_t ggml_backend_ameva_reg(void) {
    return &g_ameva_reg;
}

void ggml_backend_ameva_register(void) {
    ggml_backend_register(ggml_backend_ameva_reg());
}
