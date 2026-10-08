/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#include "ggml-ameva.h"
#include "ggml-ameva-impl.h"
#include "ggml-ameva-repack.h"
#include "ggml-backend-impl.h"
#include "ac_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <vector>

void ameva_increment_buffer_create_count(void);
void ameva_increment_buffer_destroy_count(void);

// =============================================================================
// Repacked weights
// =============================================================================
// A quantized weight is stored in the AMEVA repacked layout (ggml-ameva-repack.h) so that ameva_qmv reads every
// 32-weight unit with aligned vector loads. The layout of a tensor is decided by its first write and kept for its
// lifetime. Every access through the buffer interface converts, so ggml always sees ggml-layout bytes; the bytes in
// memory are not in the ggml layout, which is why AMEVA buffers are not host buffers while repacking is enabled
// (the CPU backend would otherwise read them in place).
//
//   AMEVA_REPACK=0      no tensor is repacked; buffers are host buffers again (the layout before repacking)
//   AMEVA_REPACK_ALL=1  tests only: repack eligible tensors of every buffer, not only of weight buffers

static bool env_flag(const char* name, bool dflt) {
    const char* v = getenv(name);
    if (!v || !*v) return dflt;
    if (strcmp(v, "0") == 0) return false;
    if (strcmp(v, "1") == 0) return true;
    GGML_ABORT("ggml-ameva: %s must be 0 or 1 (got \"%s\")", name, v);
}

bool ameva_repack_enabled(void) {
    static const bool on = env_flag("AMEVA_REPACK", true);
    return on;
}

static bool ameva_repack_all(void) {
    static const bool all = env_flag("AMEVA_REPACK_ALL", false);
    return all;
}

bool ameva_repack_all_enabled(void) {
    return ameva_repack_enabled() && ameva_repack_all();
}

// tensor->extra of a repacked tensor; only the address is used
static char s_repacked_tag;

const struct ggml_tensor* ameva_repacked_base(const struct ggml_tensor* t) {
    if (!t) return nullptr;
    if (t->extra == &s_repacked_tag) return t;
    if (t->view_src && t->view_src->extra == &s_repacked_tag) return t->view_src;
    return nullptr;
}

// Byte offset of [p, p + size) inside the buffer. The ggml buffer interface has no error return: an access that cannot
// be completed aborts, because continuing would compute on data the caller believes was transferred.
static size_t base_offset(const ggml_backend_ameva_buffer_context* ctx, const void* p, size_t size, const char* what) {
    const char* base = (const char*)ctx->base;
    const char* c = (const char*)p;
    if (!base || c < base || (size_t)(c - base) > ctx->size || size > ctx->size - (size_t)(c - base)) {
        GGML_ABORT("ggml-ameva-buffer: %s outside the buffer", what);
    }
    return (size_t)(c - base);
}

// Host access to the buffer bytes: complete on return (Vulkan: copy through the mapping and flush / invalidate the
// range; OpenCL: a blocking copy on the context's queue).
static void host_write(ggml_backend_ameva_buffer_context* ctx, size_t off, const void* src, size_t size, const char* what) {
    if (size == 0) return;
    const AcResult r = ac_runtime_buffer_write(ctx->ac_buf, off, src, size);
    if (r != AC_SUCCESS) GGML_ABORT("ggml-ameva-buffer: writing %zu bytes at %zu failed in %s (code %d)", size, off, what, r);
}

static void host_read(ggml_backend_ameva_buffer_context* ctx, size_t off, void* dst, size_t size, const char* what) {
    if (size == 0) return;
    const AcResult r = ac_runtime_buffer_read(ctx->ac_buf, off, dst, size);
    if (r != AC_SUCCESS) GGML_ABORT("ggml-ameva-buffer: reading %zu bytes at %zu failed in %s (code %d)", size, off, what, r);
}

static void host_fill(ggml_backend_ameva_buffer_context* ctx, size_t off, size_t size, uint8_t value, const char* what) {
    if (size == 0) return;
    const AcResult r = ac_runtime_buffer_fill(ctx->ac_buf, off, size, value);
    if (r != AC_SUCCESS) GGML_ABORT("ggml-ameva-buffer: filling %zu bytes at %zu failed in %s (code %d)", size, off, what, r);
}

enum class Access { Read, Write, Fill };

// The repack codec reads and writes the fields of every block with small accesses spread over the three regions. The
// buffer mapping is GPU memory that several drivers map uncached (the first HOST_VISIBLE | HOST_COHERENT type: Mali,
// Turnip), where such accesses ran at 6 to 9 MB/s: Llama-3.2-1B Q4_K_M took 81 to 134 s to load and Gemma-2-2B up to
// 294 s (Galaxy A35, A53, S21, S20, 2026-10-04; 5 and 12 s on the Galaxy S25). The codec therefore works on a staging
// copy in ordinary memory, at most this many bytes of rows at a time, and the buffer is only written and read in whole
// region ranges (one host access per region).
static const uint64_t kStagingBytes = 4ull << 20;

// Rows [r0, r0 + nr) of the repacked tensor at buffer offset `t_off`, between the buffer and `stage`, which holds those
// rows as a repacked tensor of nr rows. In both, every region is row-major, so the rows are one range per region.
static void stage_rows(ggml_backend_ameva_buffer_context* ctx, const AmevaRepackRegions& R, int64_t r0, int64_t nr,
                       uint64_t t_off, uint8_t* stage, bool to_buffer, const char* what) {
    const uint64_t n = (uint64_t)nr, r = (uint64_t)r0;
    const struct { uint64_t in_buffer, staged, len; } parts[3] = {
        { R.q + r * R.q_row, 0, n * R.q_row },
        { R.h + r * R.h_row, n * R.q_row, n * R.h_row },
        { R.s + r * R.s_row, n * (R.q_row + R.h_row), n * R.s_row },
    };
    for (const auto& p : parts) {
        if (p.len == 0) continue;
        if (to_buffer) host_write(ctx, (size_t)(t_off + p.in_buffer), stage + p.staged, (size_t)p.len, what);
        else host_read(ctx, (size_t)(t_off + p.in_buffer), stage + p.staged, (size_t)p.len, what);
    }
}

static int64_t staging_rows(uint64_t row_bytes) {
    return row_bytes >= kStagingBytes ? 1 : (int64_t)(kStagingBytes / row_bytes);
}

// Rows [r0, r0 + nr) of the repacked tensor at buffer offset `t_off` (`rows` rows of K weights of `type`) from `ggml`
// (those rows in the ggml layout).
static void write_repacked(ggml_backend_ameva_buffer_context* ctx, enum ggml_type type, int64_t K, int64_t rows,
                           uint64_t t_off, int64_t r0, int64_t nr, const uint8_t* ggml, const char* what,
                           const char* name) {
    const uint64_t row_bytes = ggml_row_size(type, K);
    AmevaRepackRegions R;
    if (!ameva_repack_regions(type, K, rows, &R)) {
        GGML_ABORT("ggml-ameva-buffer: %s: no repacked layout for %s", what, name);
    }
    const int64_t chunk = staging_rows(row_bytes);
    std::vector<uint8_t> stage((size_t)((uint64_t)std::min(chunk, nr) * row_bytes));
    for (int64_t r = r0; r < r0 + nr; r += chunk) {
        const int64_t n = std::min(chunk, r0 + nr - r);
        if (!ameva_repack_rows(type, K, n, 0, n, ggml + (uint64_t)(r - r0) * row_bytes, stage.data())) {
            GGML_ABORT("ggml-ameva-buffer: %s: repacking rows of %s failed", what, name);
        }
        stage_rows(ctx, R, r, n, t_off, stage.data(), true, what);
    }
}

static void write_rows(ggml_backend_ameva_buffer_context* ctx, const ggml_tensor* base, int64_t r0, int64_t nr,
                       const uint8_t* ggml, const char* what) {
    const uint64_t t_off = base_offset(ctx, base->data, ggml_nbytes(base), what);
    write_repacked(ctx, base->type, base->ne[0], ggml_nrows(base), t_off, r0, nr, ggml, what, base->name);
}

// Rows [r0, r0 + nr) of the repacked tensor `base` into `ggml` in the ggml layout.
static void read_rows(ggml_backend_ameva_buffer_context* ctx, const ggml_tensor* base, int64_t r0, int64_t nr,
                      uint8_t* ggml, const char* what) {
    const int64_t K = base->ne[0], rows = ggml_nrows(base);
    const uint64_t row_bytes = ggml_row_size(base->type, K);
    const uint64_t t_off = base_offset(ctx, base->data, ggml_nbytes(base), what);
    AmevaRepackRegions R;
    if (!ameva_repack_regions(base->type, K, rows, &R)) {
        GGML_ABORT("ggml-ameva-buffer: %s: no repacked layout for %s", what, base->name);
    }
    const int64_t chunk = staging_rows(row_bytes);
    std::vector<uint8_t> stage((size_t)((uint64_t)std::min(chunk, nr) * row_bytes));
    for (int64_t r = r0; r < r0 + nr; r += chunk) {
        const int64_t n = std::min(chunk, r0 + nr - r);
        stage_rows(ctx, R, r, n, t_off, stage.data(), false, what);
        if (!ameva_unrepack_rows(base->type, K, n, 0, n, stage.data(), ggml + (uint64_t)(r - r0) * row_bytes)) {
            GGML_ABORT("ggml-ameva-buffer: %s: unpacking rows of %s failed", what, base->name);
        }
    }
}

// Bytes [off, off + size) of the ggml-layout image of the repacked tensor `base`: copied to `out` (Read), replaced by
// `in` (Write) or set to `value` (Fill). Whole rows are converted; a range that does not cover whole rows is merged
// into the rows it touches.
static void repacked_access(ggml_backend_ameva_buffer_context* ctx, const ggml_tensor* base, uint64_t off, size_t size,
                            Access mode, const void* in, void* out, uint8_t value, const char* what) {
    if (size == 0) return;
    const int64_t K = base->ne[0];
    const uint64_t row_bytes = ggml_row_size(base->type, K);
    if (off > ggml_nbytes(base) || size > ggml_nbytes(base) - off) {
        GGML_ABORT("ggml-ameva-buffer: %s range outside the repacked tensor %s", what, base->name);
    }
    const int64_t r0 = (int64_t)(off / row_bytes);
    const int64_t r1 = (int64_t)((off + size + row_bytes - 1) / row_bytes);
    const uint64_t lo = off - (uint64_t)r0 * row_bytes;
    if (mode == Access::Write && lo == 0 && size % row_bytes == 0) {
        write_rows(ctx, base, r0, r1 - r0, (const uint8_t*)in, what);
        return;
    }
    std::vector<uint8_t> img((size_t)((uint64_t)(r1 - r0) * row_bytes));
    read_rows(ctx, base, r0, r1 - r0, img.data(), what);
    if (mode == Access::Read) {
        memcpy(out, img.data() + lo, size);
        return;
    }
    if (mode == Access::Write) memcpy(img.data() + lo, in, size);
    else memset(img.data() + lo, value, size);
    write_rows(ctx, base, r0, r1 - r0, img.data(), what);
}

// The repacked tensor a write to `tensor` goes to, deciding the layout on the tensor's first write: repacked when
// repacking is enabled, the buffer holds weights (or AMEVA_REPACK_ALL=1), and the tensor is a contiguous tensor (not a
// view) of a type with a repacked layout and an ameva_qmv kernel. A first write that does not cover the whole tensor
// converts the bytes already there, so earlier writes made in the ggml layout keep their meaning.
static const ggml_tensor* resolve_for_write(ggml_backend_buffer_t buffer, ggml_tensor* tensor, size_t offset,
                                            size_t size, const char* what) {
    if (const ggml_tensor* base = ameva_repacked_base(tensor)) return base;
    if (!ameva_repack_enabled() || tensor->view_src || tensor->extra) return nullptr;
    if (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS && !ameva_repack_all()) return nullptr;
    AmevaRepackLayout L;
    if (!ameva_repack_layout(tensor->type, &L) || !ameva_qmv_has_type(tensor->type)) return nullptr;
    const size_t nbytes = ggml_nbytes(tensor);
    if (!ggml_is_contiguous(tensor) || tensor->ne[0] <= 0 || tensor->ne[0] % L.block_elems != 0 || nbytes == 0) {
        return nullptr;
    }
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    const size_t t0 = base_offset(ctx, tensor->data, nbytes, what);
    const int64_t K = tensor->ne[0], rows = ggml_nrows(tensor);
    if (offset != 0 || size != nbytes) {
        std::vector<uint8_t> img(nbytes);
        host_read(ctx, t0, img.data(), nbytes, what);
        write_rows(ctx, tensor, 0, rows, img.data(), what);
    }
    tensor->extra = &s_repacked_tag;
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ctx->repacked.push_back({ t0, nbytes, tensor->type, K, rows });
    }
    ameva_count_repacked_tensor(nbytes);
    return tensor;
}

// Called by graph_compute for each MUL_MAT src0: repack a tensor that was uploaded before its buffer was marked
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS (e.g. whisper.cpp sets usage after the model upload). Safe to call repeatedly:
// returns immediately if already repacked or not eligible.
bool ameva_ensure_weight_repacked(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor) {
    if (!buffer || !buffer->context || !tensor) return false;
    if (ameva_repacked_base(tensor)) return true;
    if (!ameva_repack_enabled() || tensor->view_src || tensor->extra) return false;
    if (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
    AmevaRepackLayout L;
    if (!ameva_repack_layout(tensor->type, &L) || !ameva_qmv_has_type(tensor->type)) return false;
    const size_t nbytes = ggml_nbytes(tensor);
    // Row-padded tensors have nb[0] == ggml_type_size (element-contiguous) but nb[1] > ggml_row_size.
    // set_tensor strips the inter-row padding, so data in the AMEVA buffer is contiguous. Repacking reads
    // the buffer via host_read at cont strides, which is correct. Allow element-contiguous (rows_contiguous).
    const bool elem_cont = (tensor->nb[0] == ggml_type_size(tensor->type));
    if (!elem_cont || tensor->ne[0] <= 0 || tensor->ne[0] % L.block_elems != 0 || nbytes == 0)
        return false;
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    const size_t t0 = base_offset(ctx, tensor->data, nbytes, "ensure_weight_repacked");
    const int64_t K = tensor->ne[0], rows = ggml_nrows(tensor);
    // Read ggml-layout bytes from GPU into staging memory
    std::vector<uint8_t> img(nbytes);
    host_read(ctx, t0, img.data(), nbytes, "ensure_weight_repacked");
    // Mark as repacked before write_rows (write_rows calls base_offset which needs the tag set)
    tensor->extra = &s_repacked_tag;
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ctx->repacked.push_back({ t0, nbytes, tensor->type, K, rows });
    }
    ameva_count_repacked_tensor(nbytes);
    // Write back in repacked layout
    write_rows(ctx, tensor, 0, rows, img.data(), "ensure_weight_repacked");
    return true;
}

// =============================================================================
// Buffer interface
// =============================================================================
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
    if (!ctx->base) {
        fprintf(stderr, "ggml-ameva-buffer: get_base called but ctx->base is NULL! (size=%zu)\n", ctx->size);
    }
    return ctx->base;
}

static void check_range(const ggml_tensor* tensor, size_t offset, size_t size, const char* what) {
    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (offset > tensor_nbytes || size > tensor_nbytes - offset) {
        GGML_ABORT("ggml-ameva-buffer: %s range violation (offset=%zu, size=%zu, tensor=%zu)", what, offset, size,
                   tensor_nbytes);
    }
}

static void ggml_backend_ameva_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data) {
        GGML_ABORT("ggml-ameva-buffer: memset_tensor on an unallocated tensor or buffer");
    }
    check_range(tensor, offset, size, "memset_tensor");
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    const size_t byte_offset = base_offset(ctx, dst, size, "memset_tensor");
    if (const ggml_tensor* base = resolve_for_write(buffer, tensor, offset, size, "memset_tensor")) {
        repacked_access(ctx, base, (uint64_t)(dst - (char*)base->data), size, Access::Fill, nullptr, nullptr, value,
                        "memset_tensor");
        return;
    }
    host_fill(ctx, byte_offset, size, value, "memset_tensor");
}

static void ggml_backend_ameva_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        GGML_ABORT("ggml-ameva-buffer: set_tensor on an unallocated tensor or buffer, or from NULL");
    }
    check_range(tensor, offset, size, "set_tensor");
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    char* dst = (char*)tensor->data + offset;
    const size_t byte_offset = base_offset(ctx, dst, size, "set_tensor");
    if (const ggml_tensor* base = resolve_for_write(buffer, tensor, offset, size, "set_tensor")) {
        repacked_access(ctx, base, (uint64_t)(dst - (char*)base->data), size, Access::Write, data, nullptr, 0,
                        "set_tensor");
        return;
    }
    // Full-tensor write of a row-padded (but element-contiguous) tensor: strip inter-row padding so the GPU
    // buffer stores rows contiguously. All kernels address weight rows via ggml_row_size(type, K) (the
    // contiguous stride), so the stored layout must match. GGUF aligns rows to page/cache boundaries, which
    // produces nb[1] > ggml_row_size for small filters (e.g. whisper conv1/conv2 f16 with filter_size=3).
    if (offset == 0 && size == ggml_nbytes(tensor) &&
        tensor->ne[0] > 0 &&
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] > ggml_row_size(tensor->type, tensor->ne[0])) {
        const size_t cont_row = ggml_row_size(tensor->type, tensor->ne[0]);
        const int64_t nrows = ggml_nrows(tensor);
        std::vector<uint8_t> stripped((size_t)nrows * cont_row);
        for (int64_t r = 0; r < nrows; r++)
            memcpy(stripped.data() + (size_t)r * cont_row,
                   (const char*)data + (size_t)r * tensor->nb[1], cont_row);
        host_write(ctx, byte_offset, stripped.data(), stripped.size(), "set_tensor");
        return;
    }
    host_write(ctx, byte_offset, data, size, "set_tensor");
}

static void ggml_backend_ameva_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data || !data) {
        GGML_ABORT("ggml-ameva-buffer: get_tensor on an unallocated tensor or buffer, or into NULL");
    }
    check_range(tensor, offset, size, "get_tensor");
    auto* ctx = (ggml_backend_ameva_buffer_context*)buffer->context;
    const char* src = (const char*)tensor->data + offset;
    const size_t byte_offset = base_offset(ctx, src, size, "get_tensor");
    if (const ggml_tensor* base = ameva_repacked_base(tensor)) {
        repacked_access(ctx, base, (uint64_t)(src - (const char*)base->data), size, Access::Read, nullptr, data, 0,
                        "get_tensor");
        return;
    }
    host_read(ctx, byte_offset, data, size, "get_tensor");
}

static bool ggml_backend_ameva_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* src, struct ggml_tensor* dst) {
    (void)buffer;
    // src->data is a CPU address only for host buffers; for any other buffer (AMEVA buffers included while weights
    // are repacked) it must not be read here. false tells ggml to copy through the host (get_tensor + set_tensor).
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
    host_fill(ctx, 0, buffer->size, value, "clear");
    // Repacking permutes the bits of a tensor, so all-zero and all-one bytes stay what they are; any other value is
    // written into each repacked tensor through its layout.
    if (value != 0x00 && value != 0xFF) {
        std::lock_guard<std::mutex> lock(ctx->mu);
        std::vector<uint8_t> img;
        for (const AmevaRepackSpan& s : ctx->repacked) {
            img.assign((size_t)s.bytes, value);
            write_repacked(ctx, s.type, s.K, s.rows, s.off, 0, s.rows, img.data(), "clear",
                           ggml_type_name(s.type));
        }
    }
}

// Designated initializers (see ggml-ameva-device.cpp). init_tensor and reset are optional in ggml and left NULL:
// tensors in a flat Vulkan buffer need no per-tensor setup (the layout of a tensor is decided by its first write)
// and the buffer keeps no state to reset.
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

    void* base = nullptr;
    res = ac_runtime_buffer_get_address(ac_buf, &base);
    if (res != AC_SUCCESS || !base) {
        fprintf(stderr, "ggml-ameva: failed to get address for AcBuffer size %zu (res=%d, ptr=%p)\n", aligned_size, res, base);
        ac_runtime_buffer_destroy(ac_buf);
        ameva_increment_buffer_destroy_count();
        return nullptr;
    }

    auto* buf_ctx = new ggml_backend_ameva_buffer_context{ ac_buf, base, aligned_size };
    ggml_backend_buffer_t ret_buf = ggml_backend_buffer_init(buft, ggml_backend_ameva_buffer_i, buf_ctx, aligned_size);
    if (!ret_buf) {
        fprintf(stderr, "ggml-ameva-buffer: ggml_backend_buffer_init failed for %zu bytes\n", aligned_size);
        ac_runtime_buffer_destroy(ac_buf);
        ameva_increment_buffer_destroy_count();
        delete buf_ctx;
        return nullptr;
    }
    return ret_buf;
}

static size_t ggml_backend_ameva_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64;
}

// 512 MiB, or the device's maxMemoryAllocationSize when smaller. Buffers may exceed maxStorageBufferRange: kernels
// bind tensor-sized ranges, never whole buffers. 0 only when no Vulkan context can be created; ggml then fails the
// allocation instead of trying one.
static size_t ggml_backend_ameva_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return (size_t)ameva_max_buffer_size();
}

// The repacked layout takes exactly ggml_nbytes, so allocation sizes do not depend on it.
static size_t ggml_backend_ameva_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

// The memory is host-visible (UMA) in either case; "host buffer" in ggml means that the CPU may read and write tensor
// data in place, which holds only while no tensor is repacked, and never on OpenCL (SVM memory is reached through
// ac_runtime_buffer_write / read only).
static bool ggml_backend_ameva_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return !ameva_repack_enabled() && !ameva_backend_is_opencl();
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
