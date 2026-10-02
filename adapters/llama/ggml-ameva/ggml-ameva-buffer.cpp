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
        AcResult r = ac_runtime_buffer_destroy(ctx->ac_buf);
        if (r != AC_SUCCESS)
            fprintf(stderr, "ggml-ameva-buffer: ac_runtime_buffer_destroy failed (code %d); the GPU allocation leaked\n", r);
        ctx->ac_buf = nullptr;
        ameva_increment_buffer_destroy_count();
    }
    delete ctx;
    buffer->context = nullptr;
}

static void* ggml_backend_ameva_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer) {
        fprintf(stderr, "ggml-ameva-buffer: get_base called with NULL buffer!\n");
        return nullptr;
    }
    if (!buffer->context) {
        fprintf(stderr, "ggml-ameva-buffer: get_base called with NULL buffer->context! (buffer=%p, size=%zu, buft=%p)\n",
                buffer, buffer->size, buffer->buft);
        return nullptr;
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    if (!ctx->mapped_ptr) {
        fprintf(stderr, "ggml-ameva-buffer: get_base called but ctx->mapped_ptr is NULL! (size=%zu)\n", ctx->size);
    }
    return ctx->mapped_ptr;
}

static void ggml_backend_ameva_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
    // The ggml buffer interface has no error return: a write that cannot be completed or made visible to the
    // GPU aborts, because continuing would compute on data the caller believes was written.
    if (!buffer || !buffer->context || !tensor || !tensor->data) {
        GGML_ABORT("ggml-ameva-buffer: memset_tensor on an unallocated tensor or buffer");
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        GGML_ABORT("ggml-ameva-buffer: memset_tensor range violation (offset=%zu, size=%zu, tensor=%zu)",
                   offset, size, tensor_nbytes);
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    if (dst < (char*)ctx->mapped_ptr || (size_t)(dst - (char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(dst - (char*)ctx->mapped_ptr)) {
        GGML_ABORT("ggml-ameva-buffer: memset_tensor outside the buffer mapping");
    }
    memset(dst, value, size);
    size_t byte_offset = (size_t)(dst - (char*)ctx->mapped_ptr);
    AcResult fres = ac_runtime_buffer_flush(ctx->ac_buf, byte_offset, size);
    if (fres != AC_SUCCESS) {
        GGML_ABORT("ggml-ameva-buffer: flush failed in memset_tensor (code %d)", fres);
    }
}

static void ggml_backend_ameva_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        GGML_ABORT("ggml-ameva-buffer: set_tensor on an unallocated tensor or buffer, or from NULL");
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        GGML_ABORT("ggml-ameva-buffer: set_tensor range violation (offset=%zu, size=%zu, tensor=%zu)",
                   offset, size, tensor_nbytes);
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    if (dst < (char*)ctx->mapped_ptr || (size_t)(dst - (char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(dst - (char*)ctx->mapped_ptr)) {
        GGML_ABORT("ggml-ameva-buffer: set_tensor outside the buffer mapping");
    }
    memcpy(dst, data, size);
    size_t byte_offset = (size_t)(dst - (char*)ctx->mapped_ptr);
    AcResult fres = ac_runtime_buffer_flush(ctx->ac_buf, byte_offset, size);
    if (fres != AC_SUCCESS) {
        GGML_ABORT("ggml-ameva-buffer: flush failed in set_tensor (code %d)", fres);
    }
}

static void ggml_backend_ameva_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        GGML_ABORT("ggml-ameva-buffer: get_tensor on an unallocated tensor or buffer, or into NULL");
    }
    size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        GGML_ABORT("ggml-ameva-buffer: get_tensor range violation (offset=%zu, size=%zu, tensor=%zu)",
                   offset, size, tensor_nbytes);
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    const char* src = (const char*)tensor->data + offset;
    if (src < (const char*)ctx->mapped_ptr || (size_t)(src - (const char*)ctx->mapped_ptr) > ctx->size || size > ctx->size - (size_t)(src - (const char*)ctx->mapped_ptr)) {
        GGML_ABORT("ggml-ameva-buffer: get_tensor outside the buffer mapping");
    }
    size_t byte_offset = (size_t)(src - (const char*)ctx->mapped_ptr);
    AcResult ires = ac_runtime_buffer_invalidate(ctx->ac_buf, byte_offset, size);
    if (ires != AC_SUCCESS) {
        GGML_ABORT("ggml-ameva-buffer: invalidate failed in get_tensor (code %d)", ires);
    }
    memcpy(data, src, size);
}

static bool ggml_backend_ameva_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* src, struct ggml_tensor* dst) {
    (void)buffer;
    // src->data is a CPU address only for host buffers; for any other buffer it must not be read here.
    // false tells ggml to copy through the host.
    if (src && dst && src->data && dst->data && src->buffer && ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_ameva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) {
        GGML_ABORT("ggml-ameva-buffer: clear on a buffer without context");
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    memset(ctx->mapped_ptr, value, buffer->size);
    AcResult fres = ac_runtime_buffer_flush(ctx->ac_buf, 0, buffer->size);
    if (fres != AC_SUCCESS) {
        GGML_ABORT("ggml-ameva-buffer: flush failed in clear (code %d)", fres);
    }
}

// Designated initializers (see ggml-ameva-device.cpp). init_tensor and reset are optional in ggml and left NULL:
// tensors in a flat, host-mapped Vulkan buffer need no per-tensor setup and the buffer keeps no state to reset.
static const struct ggml_backend_buffer_i ggml_backend_ameva_buffer_i = {
    .free_buffer   = ggml_backend_ameva_buffer_free_buffer,
    .get_base      = ggml_backend_ameva_buffer_get_base,
    .init_tensor   = nullptr,
    .memset_tensor = ggml_backend_ameva_buffer_memset_tensor,
    .set_tensor    = ggml_backend_ameva_buffer_set_tensor,
    .get_tensor    = ggml_backend_ameva_buffer_get_tensor,
    .cpy_tensor    = ggml_backend_ameva_buffer_cpy_tensor,
    .clear         = ggml_backend_ameva_buffer_clear,
    .reset         = nullptr,
};

static const char* ggml_backend_ameva_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return GGML_AMEVA_NAME;
}

static ggml_backend_buffer_t ggml_backend_ameva_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    fprintf(stderr, "ggml-ameva-buffer: alloc_buffer called (requested_size=%zu)\n", size);
    AcContext* ctx = ameva_get_ac_context();
    if (!ctx) {
        fprintf(stderr, "ggml-ameva-buffer: alloc_buffer failed - ameva_get_ac_context returned NULL!\n");
        return nullptr;
    }

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
        fprintf(stderr, "ggml-ameva: failed to get mapped ptr for AcBuffer size %zu (res=%d, ptr=%p)\n", aligned_size, res, mapped_ptr);
        ac_runtime_buffer_destroy(ac_buf);
        ameva_increment_buffer_destroy_count();
        return nullptr;
    }

    fprintf(stderr, "ggml-ameva-buffer: alloc_buffer SUCCESS (aligned_size=%zu, mapped_ptr=%p)\n", aligned_size, mapped_ptr);
    auto* buf_ctx = new ggml_backend_ameva_buffer_context{ ac_buf, mapped_ptr, aligned_size };
    ggml_backend_buffer_t ret_buf = ggml_backend_buffer_init(buft, ggml_backend_ameva_buffer_i, buf_ctx, aligned_size);
    fprintf(stderr, "ggml-ameva-buffer: alloc_buffer returned buffer=%p, buf->context=%p\n", ret_buf, ret_buf ? ret_buf->context : nullptr);
    return ret_buf;
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
    .get_name       = ggml_backend_ameva_buffer_type_get_name,
    .alloc_buffer   = ggml_backend_ameva_buffer_type_alloc_buffer,
    .get_alignment  = ggml_backend_ameva_buffer_type_get_alignment,
    .get_max_size   = ggml_backend_ameva_buffer_type_get_max_size,
    .get_alloc_size = ggml_backend_ameva_buffer_type_get_alloc_size,
    .is_host        = ggml_backend_ameva_buffer_type_is_host,
};

extern struct ggml_backend_device g_ameva_device; // ggml-ameva-device.cpp

// .device links the buffer type to AMEVA0; ggml and llama.cpp attribute buffers to a device through it
// (scheduler, memory accounting). It was NULL, so AMEVA allocations belonged to no device.
static struct ggml_backend_buffer_type g_ameva_buft = {
    .iface   = ggml_backend_ameva_buffer_type_i,
    .device  = &g_ameva_device,
    .context = nullptr,
};

ggml_backend_buffer_type_t ggml_backend_ameva_buffer_type(size_t dev_num) {
    if (dev_num != 0) {
        fprintf(stderr, "ggml-ameva-buffer: device %zu does not exist (AMEVA exposes one device)\n", dev_num);
        return nullptr;
    }
    return &g_ameva_buft;
}
