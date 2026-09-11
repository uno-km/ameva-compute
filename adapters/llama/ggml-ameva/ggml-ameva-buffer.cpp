#include "ggml-ameva.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Forward declaration of internal context accessor
AcContext* ameva_get_ac_context();
void ameva_increment_buffer_create_count(void);
void ameva_increment_buffer_destroy_count(void);

struct ggml_backend_ameva_buffer_context {
    AcBuffer* ac_buf;
    void* mapped_ptr;
    size_t size;
};

static void ggml_backend_ameva_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return;
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    if (ctx->ac_buf) {
        ac_runtime_buffer_destroy(ctx->ac_buf);
        ctx->ac_buf = nullptr;
        ameva_increment_buffer_destroy_count();
    }
    delete ctx;
    buffer->context = nullptr;
}

static void* ggml_backend_ameva_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return nullptr;
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    return ctx->mapped_ptr;
}

static enum ggml_status ggml_backend_ameva_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor) {
    (void)buffer;
    (void)tensor;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_ameva_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data) {
        return;
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        fprintf(stderr, "ggml-ameva-buffer: memset_tensor range violation (offset=%zu, size=%zu, tensor=%zu)\n",
                offset, size, tensor_nbytes);
        return;
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    if (dst < (char*)ctx->mapped_ptr || (size_t)(dst - (char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(dst - (char*)ctx->mapped_ptr)) {
        fprintf(stderr, "ggml-ameva-buffer: memset_tensor buffer limit violation\n");
        return;
    }
    memset(dst, value, size);
    size_t byte_offset = (size_t)(dst - (char*)ctx->mapped_ptr);
    AcResult fres = ac_runtime_buffer_flush(ctx->ac_buf, byte_offset, size);
    if (fres != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva-buffer: flush failed in memset_tensor (code %d)\n", fres);
    }
}

static void ggml_backend_ameva_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        return;
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        fprintf(stderr, "ggml-ameva-buffer: set_tensor range violation (offset=%zu, size=%zu, tensor=%zu)\n",
                offset, size, tensor_nbytes);
        return;
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    if (dst < (char*)ctx->mapped_ptr || (size_t)(dst - (char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(dst - (char*)ctx->mapped_ptr)) {
        fprintf(stderr, "ggml-ameva-buffer: set_tensor buffer limit violation\n");
        return;
    }
    memcpy(dst, data, size);
    size_t byte_offset = (size_t)(dst - (char*)ctx->mapped_ptr);
    AcResult fres = ac_runtime_buffer_flush(ctx->ac_buf, byte_offset, size);
    if (fres != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva-buffer: flush failed in set_tensor (code %d)\n", fres);
    }
}

static void ggml_backend_ameva_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        return;
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        fprintf(stderr, "ggml-ameva-buffer: get_tensor range violation (offset=%zu, size=%zu, tensor=%zu)\n",
                offset, size, tensor_nbytes);
        return;
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    const char* src = (const char*)tensor->data + offset;
    if (src < (const char*)ctx->mapped_ptr || (size_t)(src - (const char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(src - (const char*)ctx->mapped_ptr)) {
        fprintf(stderr, "ggml-ameva-buffer: get_tensor buffer limit violation\n");
        return;
    }
    size_t byte_offset = (size_t)(src - (const char*)ctx->mapped_ptr);
    AcResult ires = ac_runtime_buffer_invalidate(ctx->ac_buf, byte_offset, size);
    if (ires != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva-buffer: invalidate failed in get_tensor (code %d)\n", ires);
    }
    memcpy(data, src, size);
}

static bool ggml_backend_ameva_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* src, struct ggml_tensor* dst) {
    (void)buffer;
    if (src && dst && src->data && dst->data) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_ameva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) return;
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    memset(ctx->mapped_ptr, value, buffer->size);
    ac_runtime_buffer_flush(ctx->ac_buf, 0, buffer->size);
}

static void ggml_backend_ameva_buffer_reset(ggml_backend_buffer_t buffer) {
    (void)buffer;
}

static const struct ggml_backend_buffer_i ggml_backend_ameva_buffer_i = {
    /* .free_buffer     = */ ggml_backend_ameva_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ameva_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ameva_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_ameva_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ameva_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ameva_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_ameva_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_ameva_buffer_clear,
    /* .reset           = */ ggml_backend_ameva_buffer_reset,
};

static const char* ggml_backend_ameva_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return GGML_AMEVA_NAME;
}

static ggml_backend_buffer_t ggml_backend_ameva_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    AcContext* ctx = ameva_get_ac_context();
    if (!ctx) return nullptr;

    size_t aligned_size = (size + 63) & ~63;
    if (aligned_size < 256) aligned_size = 256;

    AcBufferDesc bdesc = {
        AC_ABI_VERSION,
        sizeof(AcBufferDesc),
        aligned_size,
        AC_BUFFER_USAGE_STORAGE,
        0
    };

    AcBuffer* ac_buf = nullptr;
    AcResult res = ac_runtime_buffer_create(ctx, &bdesc, &ac_buf);
    if (res != AC_SUCCESS) {
        fprintf(stderr, "ggml-ameva: failed to allocate AcBuffer size %zu (code %d)\n", aligned_size, res);
        return nullptr;
    }
    ameva_increment_buffer_create_count();

    void* mapped_ptr = nullptr;
    res = ac_runtime_buffer_get_mapped_ptr(ac_buf, &mapped_ptr);
    if (res != AC_SUCCESS || !mapped_ptr) {
        ac_runtime_buffer_destroy(ac_buf);
        ameva_increment_buffer_destroy_count();
        return nullptr;
    }

    auto* buf_ctx = new ggml_backend_ameva_buffer_context{ ac_buf, mapped_ptr, aligned_size };
    return ggml_backend_buffer_init(buft, ggml_backend_ameva_buffer_i, buf_ctx, aligned_size);
}

static size_t ggml_backend_ameva_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64;
}

static size_t ggml_backend_ameva_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    // Realistic maximum contiguous buffer size on mobile Vulkan (512 MiB)
    return 512ULL * 1024 * 1024;
}

static size_t ggml_backend_ameva_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

static bool ggml_backend_ameva_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true; // Host-visible unified memory architecture (UMA)
}

static const struct ggml_backend_buffer_type_i ggml_backend_ameva_buffer_type_i = {
    /* .get_name       = */ ggml_backend_ameva_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_ameva_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_ameva_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_ameva_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_backend_ameva_buffer_type_get_alloc_size,
    /* .is_host        = */ ggml_backend_ameva_buffer_type_is_host,
};

static struct ggml_backend_buffer_type g_ameva_buft = {
    /* .iface   = */ ggml_backend_ameva_buffer_type_i,
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(size_t dev_num) {
    (void)dev_num;
    return &g_ameva_buft;
}
