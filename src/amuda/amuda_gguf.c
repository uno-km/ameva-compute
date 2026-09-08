/**
 * @file amuda_gguf.c
 * @brief Zero-Copy GGUF Model Parser & Tensor Binder
 */

#include "amuda_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define ALIGN_UP(offset, align) (((offset) + (align) - 1) & ~((align) - 1))

static size_t get_type_size(uint32_t type, const uint64_t* ne, uint32_t n_dims) {
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        n_elements *= ne[i];
    }
    switch (type) {
        case GGML_TYPE_F32:
            return n_elements * 4;
        case GGML_TYPE_F16:
            return n_elements * 2;
        case GGML_TYPE_Q4_0:
            return (n_elements / 32) * 18;
        case GGML_TYPE_Q8_0:
            return (n_elements / 32) * 34;
        case GGML_TYPE_I2_S: {
            uint64_t nrow = (n_dims >= 2) ? ne[1] : 1;
            uint64_t n_per_row = ne[0];
            return nrow * (n_per_row / 4 + 32);
        }
        default:
            return n_elements;
    }
}

static void skip_val(uint32_t vtype, const uint8_t** cursor) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            *cursor += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            *cursor += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            *cursor += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            *cursor += 8;
            break;
        case GGUF_TYPE_STRING: {
            uint64_t slen = *(const uint64_t*)(*cursor);
            *cursor += 8 + slen;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t atype = *(const uint32_t*)(*cursor);
            *cursor += 4;
            uint64_t alen = *(const uint64_t*)(*cursor);
            *cursor += 8;
            for (uint64_t i = 0; i < alen; i++) {
                skip_val(atype, cursor);
            }
            break;
        }
        default:
            break;
    }
}

AcResult ac_gguf_open(const char* file_path, GgufModel** out_model) {
    if (!file_path || !out_model) return AC_ERR_NULL_POINTER;

    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[AMUDA-GGUF] Failed to open file: %s\n", file_path);
        return AC_ERR_BINARY_CORRUPTED;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }
    size_t file_size = (size_t)st.st_size;

    void* mmap_addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_addr == MAP_FAILED) {
        perror("[AMUDA-GGUF] mmap failed");
        close(fd);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    const uint8_t* ptr = (const uint8_t*)mmap_addr;
    const uint8_t* end = ptr + file_size;

    uint32_t magic = *(const uint32_t*)ptr;
    ptr += 4;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[AMUDA-GGUF] Invalid GGUF magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_BINARY_CORRUPTED;
    }

    uint32_t version = *(const uint32_t*)ptr;
    ptr += 4;
    if (version != 2 && version != 3) {
        fprintf(stderr, "[AMUDA-GGUF] Unsupported GGUF version: %u\n", version);
        munmap(mmap_addr, file_size);
        close(fd);
        return AC_ERR_DEVICE_NOT_SUPPORTED;
    }

    uint64_t tensor_count = *(const uint64_t*)ptr;
    ptr += 8;
    uint64_t metadata_kv_count = *(const uint64_t*)ptr;
    ptr += 8;

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

    // Parse Metadata KV Pairs
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        if (ptr + 12 > end) break;
        uint64_t klen = *(const uint64_t*)ptr;
        ptr += 8;
        if (ptr + klen > end) break;

        char key[128] = {0};
        size_t copy_len = (klen < sizeof(key) - 1) ? klen : sizeof(key) - 1;
        memcpy(key, ptr, copy_len);
        ptr += klen;

        uint32_t vtype = *(const uint32_t*)ptr;
        ptr += 4;

        if (strcmp(key, "general.architecture") == 0 && vtype == GGUF_TYPE_STRING) {
            uint64_t slen = *(const uint64_t*)ptr;
            size_t scopy = (slen < sizeof(model->config.arch) - 1) ? slen : sizeof(model->config.arch) - 1;
            memcpy(model->config.arch, ptr + 8, scopy);
        } else if (strcmp(key, "general.alignment") == 0) {
            if (vtype == GGUF_TYPE_UINT32) alignment = *(const uint32_t*)ptr;
            else if (vtype == GGUF_TYPE_UINT64) alignment = *(const uint64_t*)ptr;
        }

        // Check arch-specific configurations
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "%s.", model->config.arch);
        size_t plen = strlen(prefix);

        if (strncmp(key, prefix, plen) == 0) {
            const char* subkey = key + plen;
            if (strcmp(subkey, "context_length") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.context_length = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.context_length = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "embedding_length") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.embedding_length = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.embedding_length = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "block_count") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.block_count = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.block_count = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "feed_forward_length") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.feed_forward_length = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.feed_forward_length = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "attention.head_count") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.head_count = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.head_count = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "attention.head_count_kv") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.head_count_kv = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.head_count_kv = (uint32_t)*(const uint64_t*)ptr;
            } else if (strcmp(subkey, "attention.layer_norm_rms_epsilon") == 0) {
                if (vtype == GGUF_TYPE_FLOAT32) model->config.rms_norm_eps = *(const float*)ptr;
            } else if (strcmp(subkey, "rope.freq_base") == 0) {
                if (vtype == GGUF_TYPE_FLOAT32) model->config.rope_freq_base = *(const float*)ptr;
            } else if (strcmp(subkey, "vocab_size") == 0) {
                if (vtype == GGUF_TYPE_UINT32) model->config.vocab_size = *(const uint32_t*)ptr;
                else if (vtype == GGUF_TYPE_UINT64) model->config.vocab_size = (uint32_t)*(const uint64_t*)ptr;
            }
        }

        skip_val(vtype, &ptr);
    }

    // Allocate Tensors Array
    model->tensors = (GgufTensor*)calloc(tensor_count, sizeof(GgufTensor));
    if (!model->tensors) {
        ac_gguf_close(model);
        return AC_ERR_OUT_OF_HOST_MEMORY;
    }

    // Parse Tensor Headers
    for (uint64_t i = 0; i < tensor_count; i++) {
        if (ptr >= end) break;
        uint64_t nlen = *(const uint64_t*)ptr;
        ptr += 8;

        size_t ncopy = (nlen < sizeof(model->tensors[i].name) - 1) ? nlen : sizeof(model->tensors[i].name) - 1;
        memcpy(model->tensors[i].name, ptr, ncopy);
        ptr += nlen;

        uint32_t n_dims = *(const uint32_t*)ptr;
        ptr += 4;
        model->tensors[i].n_dims = n_dims;

        for (uint32_t d = 0; d < n_dims; d++) {
            model->tensors[i].ne[d] = *(const uint64_t*)ptr;
            ptr += 8;
        }

        model->tensors[i].type = *(const uint32_t*)ptr;
        ptr += 4;

        model->tensors[i].offset = *(const uint64_t*)ptr;
        ptr += 8;

        model->tensors[i].size_bytes = get_type_size(model->tensors[i].type, model->tensors[i].ne, model->tensors[i].n_dims);
    }

    // Align to data offset
    size_t current_offset = (size_t)(ptr - (const uint8_t*)mmap_addr);
    size_t data_base_offset = ALIGN_UP(current_offset, alignment);
    const uint8_t* data_base = (const uint8_t*)mmap_addr + data_base_offset;

    // Direct pointer binding
    for (uint64_t i = 0; i < tensor_count; i++) {
        model->tensors[i].data_ptr = (void*)(data_base + model->tensors[i].offset);
    }

    *out_model = model;
    return AC_SUCCESS;
}

GgufTensor* ac_gguf_get_tensor(GgufModel* model, const char* tensor_name) {
    if (!model || !tensor_name) return NULL;
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
        uint64_t n_elements = tensor->ne[0] * (tensor->n_dims > 1 ? tensor->ne[1] : 1);
        size_t packed_bytes = (size_t)(n_elements / 4);
        const uint8_t* base = (const uint8_t*)tensor->data_ptr;
        *out_scale = *(const float*)(base + packed_bytes);
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
