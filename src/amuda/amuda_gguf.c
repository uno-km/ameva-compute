/**
 * @file amuda_gguf.c
 * @brief Zero-Copy Bounded GGUF Model Parser & Tensor Binder
 *
 * Implements strict boundary checking, fail-closed validation,
 * resource budget enforcement, and safe dimension parsing
 * per AMEVA P0-3 stabilization standards.
 */

#include "amuda_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define GGUF_MAX_METADATA_KV_COUNT         65536ULL
#define GGUF_MAX_TENSOR_COUNT              65536ULL
#define GGUF_MAX_TOTAL_METADATA_BYTES      (16 * 1024 * 1024ULL) /* 16 MB max for metadata section */
#define GGUF_MAX_TOTAL_TENSOR_ALLOC_BYTES  (16 * 1024 * 1024ULL) /* 16 MB max for GgufTensor array */
#define GGUF_MAX_STRING_LEN                65536ULL
#define GGUF_MAX_RECURSION_DEPTH           16

typedef struct {
    const uint8_t* cur;
    const uint8_t* end;
} GgufReader;

static inline void reader_init(GgufReader* r, const uint8_t* data, size_t size) {
    if (!r) return;
    r->cur = data;
    r->end = (data && size <= (SIZE_MAX - (size_t)data)) ? (data + size) : NULL;
}

static inline size_t reader_remaining(const GgufReader* r) {
    if (!r || !r->cur || !r->end || r->cur > r->end) return 0;
    return (size_t)(r->end - r->cur);
}

static inline bool reader_can_read(const GgufReader* r, size_t bytes) {
    return reader_remaining(r) >= bytes;
}

static inline bool reader_read_u32(GgufReader* r, uint32_t* val) {
    if (!reader_can_read(r, 4)) return false;
    *val = ((uint32_t)r->cur[0]) |
           (((uint32_t)r->cur[1]) << 8) |
           (((uint32_t)r->cur[2]) << 16) |
           (((uint32_t)r->cur[3]) << 24);
    r->cur += 4;
    return true;
}

static inline bool reader_read_u64(GgufReader* r, uint64_t* val) {
    if (!reader_can_read(r, 8)) return false;
    *val = (((uint64_t)r->cur[0])) |
           (((uint64_t)r->cur[1]) << 8) |
           (((uint64_t)r->cur[2]) << 16) |
           (((uint64_t)r->cur[3]) << 24) |
           (((uint64_t)r->cur[4]) << 32) |
           (((uint64_t)r->cur[5]) << 40) |
           (((uint64_t)r->cur[6]) << 48) |
           (((uint64_t)r->cur[7]) << 56);
    r->cur += 8;
    return true;
}

static inline bool reader_read_f32(GgufReader* r, float* val) {
    uint32_t u32 = 0;
    if (!reader_read_u32(r, &u32)) return false;
    memcpy(val, &u32, sizeof(float));
    return true;
}

static inline bool reader_skip(GgufReader* r, size_t bytes) {
    if (!reader_can_read(r, bytes)) return false;
    r->cur += bytes;
    return true;
}

static inline bool reader_read_string(GgufReader* r, char* dst, size_t max_dst_len, uint64_t* out_len) {
    uint64_t slen = 0;
    if (!reader_read_u64(r, &slen)) return false;
    if (slen > GGUF_MAX_STRING_LEN || !reader_can_read(r, (size_t)slen)) return false;
    if (out_len) *out_len = slen;
    if (dst && max_dst_len > 0) {
        size_t copy_len = ((size_t)slen < max_dst_len - 1) ? (size_t)slen : max_dst_len - 1;
        memcpy(dst, r->cur, copy_len);
        dst[copy_len] = '\0';
    }
    r->cur += (size_t)slen;
    return true;
}

static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (UINT64_MAX / a < b) {
        return false;
    }
    *out = a * b;
    return true;
}

static inline bool checked_align_up(size_t val, size_t align, size_t* out) {
    if (align == 0 || (align & (align - 1)) != 0) return false;
    size_t rem = val % align;
    if (rem == 0) {
        *out = val;
        return true;
    }
    size_t add = align - rem;
    if (SIZE_MAX - val < add) return false;
    *out = val + add;
    return true;
}

static bool skip_val_bounded(GgufReader* r, uint32_t vtype, uint32_t depth) {
    if (depth > GGUF_MAX_RECURSION_DEPTH) return false;
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return reader_skip(r, 1);
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return reader_skip(r, 2);
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return reader_skip(r, 4);
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return reader_skip(r, 8);
        case GGUF_TYPE_STRING: {
            uint64_t slen = 0;
            if (!reader_read_u64(r, &slen)) return false;
            if (slen > GGUF_MAX_STRING_LEN || !reader_can_read(r, (size_t)slen)) return false;
            r->cur += (size_t)slen;
            return true;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t atype = 0;
            uint64_t alen = 0;
            if (!reader_read_u32(r, &atype)) return false;
            if (!reader_read_u64(r, &alen)) return false;
            size_t elem_size = 0;
            switch (atype) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:
                case GGUF_TYPE_BOOL:
                    elem_size = 1; break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:
                    elem_size = 2; break;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32:
                case GGUF_TYPE_FLOAT32:
                    elem_size = 4; break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:
                case GGUF_TYPE_FLOAT64:
                    elem_size = 8; break;
                default:
                    elem_size = 0; break;
            }
            if (elem_size > 0) {
                if (alen > SIZE_MAX / elem_size) return false;
                size_t total_bytes = (size_t)alen * elem_size;
                return reader_skip(r, total_bytes);
            }
            for (uint64_t i = 0; i < alen; i++) {
                if (!skip_val_bounded(r, atype, depth + 1)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

static AcResult calculate_tensor_size(uint32_t type, const uint64_t* ne, uint32_t n_dims, size_t* out_size) {
    if (!ne || n_dims < 1 || n_dims > 4 || !out_size) return AC_ERR_INVALID_ARGUMENT;

    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        if (ne[i] == 0) return AC_ERR_BINARY_CORRUPTED;
        if (!checked_mul_u64(n_elements, ne[i], &n_elements)) {
            return AC_ERR_BINARY_CORRUPTED;
        }
    }

    switch (type) {
        case GGML_TYPE_F32: {
            uint64_t bytes = 0;
            if (!checked_mul_u64(n_elements, 4, &bytes) || bytes > SIZE_MAX) {
                return AC_ERR_BINARY_CORRUPTED;
            }
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_F16: {
            uint64_t bytes = 0;
            if (!checked_mul_u64(n_elements, 2, &bytes) || bytes > SIZE_MAX) {
                return AC_ERR_BINARY_CORRUPTED;
            }
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q4_0: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 18, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q4_1: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 20, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q5_0: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 22, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q5_1: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 24, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q8_0: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 34, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q8_1: {
            uint64_t blocks = (n_elements + 31) / 32;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 40, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q2_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 84, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q3_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 110, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q4_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 144, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q5_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 176, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q6_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 210, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_Q8_K: {
            uint64_t blocks = (n_elements + 255) / 256;
            uint64_t bytes = 0;
            if (!checked_mul_u64(blocks, 292, &bytes) || bytes > SIZE_MAX) return AC_ERR_BINARY_CORRUPTED;
            *out_size = (size_t)bytes;
            return AC_SUCCESS;
        }
        case GGML_TYPE_I2_S: {
            uint64_t nrow = 1;
            for (uint32_t d = 1; d < n_dims; d++) {
                if (!checked_mul_u64(nrow, ne[d], &nrow)) return AC_ERR_BINARY_CORRUPTED;
            }
            uint64_t n_per_row = ne[0];
            uint64_t row_bytes = (n_per_row / 4) + 32;
            uint64_t total_bytes = 0;
            if (!checked_mul_u64(nrow, row_bytes, &total_bytes) || total_bytes > SIZE_MAX) {
                return AC_ERR_BINARY_CORRUPTED;
            }
            *out_size = (size_t)total_bytes;
            return AC_SUCCESS;
        }
        default:
            return AC_ERR_DEVICE_NOT_SUPPORTED;
    }
}

AcResult ac_gguf_open(const char* file_path, GgufModel** out_model) {
    if (!file_path || !out_model) return AC_ERR_NULL_POINTER;
    *out_model = NULL;

    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        return AC_ERR_BINARY_CORRUPTED;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }
    size_t file_size = (size_t)st.st_size;

    void* mmap_addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_addr == MAP_FAILED) {
        close(fd);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    GgufReader r;
    reader_init(&r, (const uint8_t*)mmap_addr, file_size);

    uint32_t magic = 0;
    if (!reader_read_u32(&r, &magic) || magic != GGUF_MAGIC) {
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }

    uint32_t version = 0;
    if (!reader_read_u32(&r, &version) || (version != 2 && version != 3)) {
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    uint64_t tensor_count = 0;
    if (!reader_read_u64(&r, &tensor_count) || tensor_count > GGUF_MAX_TENSOR_COUNT) {
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }

    uint64_t metadata_kv_count = 0;
    if (!reader_read_u64(&r, &metadata_kv_count) || metadata_kv_count > GGUF_MAX_METADATA_KV_COUNT) {
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }

    GgufModel* model = (GgufModel*)calloc(1, sizeof(GgufModel));
    if (!model) {
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    model->fd = fd;
    model->mmap_addr = mmap_addr;
    model->mmap_size = file_size;
    model->version = version;
    model->tensor_count = tensor_count;
    model->metadata_kv_count = metadata_kv_count;

    uint64_t alignment = 32;

    // Parse Metadata KV Pairs with total metadata byte budget guard
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        size_t current_meta_offset = (size_t)(r.cur - (const uint8_t*)mmap_addr);
        if (current_meta_offset > GGUF_MAX_TOTAL_METADATA_BYTES) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        char key[128] = {0};
        uint64_t klen = 0;
        if (!reader_read_string(&r, key, sizeof(key), &klen)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        uint32_t vtype = 0;
        if (!reader_read_u32(&r, &vtype)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        if (strcmp(key, "general.architecture") == 0 && vtype == GGUF_TYPE_STRING) {
            uint64_t slen = 0;
            if (!reader_read_string(&r, model->config.arch, sizeof(model->config.arch), &slen)) {
                ac_gguf_close(model);
                return AC_ERR_BINARY_CORRUPTED;
            }
        } else if (strcmp(key, "general.alignment") == 0) {
            if (vtype == GGUF_TYPE_UINT32) {
                uint32_t a32 = 0;
                if (!reader_read_u32(&r, &a32)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                alignment = a32;
            } else if (vtype == GGUF_TYPE_UINT64) {
                if (!reader_read_u64(&r, &alignment)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
            } else {
                if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
            }
            if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > (1ULL << 30)) {
                ac_gguf_close(model);
                return AC_ERR_BINARY_CORRUPTED;
            }
        } else {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "%s.", model->config.arch);
            size_t plen = strlen(prefix);

            if (plen > 1 && strncmp(key, prefix, plen) == 0) {
                const char* subkey = key + plen;
                if (strcmp(subkey, "context_length") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.context_length)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.context_length = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "embedding_length") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.embedding_length)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.embedding_length = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "block_count") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.block_count)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.block_count = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "feed_forward_length") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.feed_forward_length)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.feed_forward_length = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "attention.head_count") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.head_count)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.head_count = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "attention.head_count_kv") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.head_count_kv)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.head_count_kv = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "attention.layer_norm_rms_epsilon") == 0) {
                    if (vtype == GGUF_TYPE_FLOAT32) {
                        if (!reader_read_f32(&r, &model->config.rms_norm_eps)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "rope.freq_base") == 0) {
                    if (vtype == GGUF_TYPE_FLOAT32) {
                        if (!reader_read_f32(&r, &model->config.rope_freq_base)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else if (strcmp(subkey, "vocab_size") == 0) {
                    if (vtype == GGUF_TYPE_UINT32) {
                        if (!reader_read_u32(&r, &model->config.vocab_size)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    } else if (vtype == GGUF_TYPE_UINT64) {
                        uint64_t v = 0;
                        if (!reader_read_u64(&r, &v)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                        model->config.vocab_size = (uint32_t)v;
                    } else {
                        if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                    }
                } else {
                    if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
                }
            } else {
                if (!skip_val_bounded(&r, vtype, 0)) { ac_gguf_close(model); return AC_ERR_BINARY_CORRUPTED; }
            }
        }
    }

    // Allocate Tensors Array with memory budget guard
    if (tensor_count > 0) {
        if (tensor_count > SIZE_MAX / sizeof(GgufTensor)) {
            ac_gguf_close(model);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        size_t alloc_bytes = (size_t)tensor_count * sizeof(GgufTensor);
        if (alloc_bytes > GGUF_MAX_TOTAL_TENSOR_ALLOC_BYTES) {
            ac_gguf_close(model);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
        model->tensors = (GgufTensor*)calloc(tensor_count, sizeof(GgufTensor));
        if (!model->tensors) {
            ac_gguf_close(model);
            return AC_ERR_OUT_OF_HOST_MEMORY;
        }
    }

    // Parse Tensor Headers
    for (uint64_t i = 0; i < tensor_count; i++) {
        uint64_t nlen = 0;
        if (!reader_read_string(&r, model->tensors[i].name, sizeof(model->tensors[i].name), &nlen)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        uint32_t n_dims = 0;
        if (!reader_read_u32(&r, &n_dims)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        // Strict dimension bounds enforcement: 1 <= n_dims <= 4
        if (n_dims < 1 || n_dims > 4) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }
        model->tensors[i].n_dims = n_dims;

        for (uint32_t d = 0; d < n_dims; d++) {
            if (!reader_read_u64(&r, &model->tensors[i].ne[d])) {
                ac_gguf_close(model);
                return AC_ERR_BINARY_CORRUPTED;
            }
            if (model->tensors[i].ne[d] == 0) {
                ac_gguf_close(model);
                return AC_ERR_BINARY_CORRUPTED;
            }
        }

        if (!reader_read_u32(&r, &model->tensors[i].type)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        if (!reader_read_u64(&r, &model->tensors[i].offset)) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        AcResult sz_res = calculate_tensor_size(
            model->tensors[i].type,
            model->tensors[i].ne,
            model->tensors[i].n_dims,
            &model->tensors[i].size_bytes
        );
        if (sz_res != AC_SUCCESS) {
            ac_gguf_close(model);
            return sz_res;
        }
    }

    // Align to data offset
    size_t current_offset = (size_t)(r.cur - (const uint8_t*)mmap_addr);
    size_t data_base_offset = 0;
    if (!checked_align_up(current_offset, (size_t)alignment, &data_base_offset)) {
        ac_gguf_close(model);
        return AC_ERR_BINARY_CORRUPTED;
    }
    if (data_base_offset > file_size) {
        ac_gguf_close(model);
        return AC_ERR_BINARY_CORRUPTED;
    }

    size_t available_data_bytes = file_size - data_base_offset;
    const uint8_t* data_base = (const uint8_t*)mmap_addr + data_base_offset;

    // Direct pointer binding with strict bounds checks:
    // Invariant: offset <= available && size <= available - offset
    for (uint64_t i = 0; i < tensor_count; i++) {
        uint64_t toff = model->tensors[i].offset;
        size_t tsz = model->tensors[i].size_bytes;

        if (toff > available_data_bytes || tsz > available_data_bytes - (size_t)toff) {
            ac_gguf_close(model);
            return AC_ERR_BINARY_CORRUPTED;
        }

        model->tensors[i].data_ptr = (void*)(data_base + toff);
    }

    *out_model = model;
    return AC_SUCCESS;
}

GgufTensor* ac_gguf_get_tensor(GgufModel* model, const char* tensor_name) {
    if (!model || !tensor_name || !model->tensors) return NULL;
    for (uint64_t i = 0; i < model->tensor_count; i++) {
        if (strcmp(model->tensors[i].name, tensor_name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

AcResult ac_gguf_get_tensor_scale(const GgufTensor* tensor, float* out_scale) {
    if (!tensor || !out_scale) return AC_ERR_NULL_POINTER;
    if (tensor->type == GGML_TYPE_I2_S) {
        if (!tensor->data_ptr) return AC_ERR_BINARY_CORRUPTED;
        uint64_t n_elements = 0;
        if (!checked_mul_u64(tensor->ne[0], (tensor->n_dims > 1 ? tensor->ne[1] : 1), &n_elements)) {
            return AC_ERR_BINARY_CORRUPTED;
        }
        size_t packed_bytes = (size_t)(n_elements / 4);
        if (packed_bytes > tensor->size_bytes || sizeof(float) > tensor->size_bytes - packed_bytes) {
            return AC_ERR_BINARY_CORRUPTED;
        }
        const uint8_t* base = (const uint8_t*)tensor->data_ptr;
        float scale = 0.0f;
        memcpy(&scale, base + packed_bytes, sizeof(float));
        *out_scale = scale;
        return AC_SUCCESS;
    } else if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16) {
        *out_scale = 1.0f;
        return AC_SUCCESS;
    }
    return AC_ERR_DEVICE_NOT_SUPPORTED;
}

void ac_gguf_close(GgufModel* model) {
    if (!model) return;
    if (model->tensors) {
        free(model->tensors);
        model->tensors = NULL;
    }
    if (model->mmap_addr && model->mmap_size > 0) {
        munmap(model->mmap_addr, model->mmap_size);
        model->mmap_addr = NULL;
    }
    if (model->fd >= 0) {
        close(model->fd);
        model->fd = -1;
    }
    free(model);
}
