/**
 * @file amuda_gguf.h
 * @brief Native Zero-Copy GGUF Model Parser & Tensor Binder
 */

#ifndef AMUDA_GGUF_H
#define AMUDA_GGUF_H

#include "amuda.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC 0x46554747 /* 'GGUF' */

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GgufType;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I2_S = 36, /* BitNet 1.58b */
} GgmlType;

typedef struct {
    char     name[64];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;       /* GgmlType */
    uint64_t offset;
    void*    data_ptr;   /* Direct host virtual memory pointer into mmap */
    size_t   size_bytes;
    AcBuffer* ac_buf;    /* Optional bound zero-copy UMA buffer */
} GgufTensor;

typedef struct {
    char     arch[32];
    uint32_t context_length;
    uint32_t embedding_length; /* hidden_dim */
    uint32_t block_count;      /* num_layers */
    uint32_t feed_forward_length;
    uint32_t head_count;
    uint32_t head_count_kv;
    float    rms_norm_eps;
    float    rope_freq_base;
    uint32_t vocab_size;
} GgufModelConfig;

typedef struct {
    void*           mmap_addr;
    size_t          mmap_size;
    int             fd;
    uint32_t        version;
    uint64_t        tensor_count;
    uint64_t        metadata_kv_count;
    GgufModelConfig config;
    GgufTensor*     tensors;
} GgufModel;

/**
 * @brief Open and memory-map a GGUF file without memory copying.
 */
AcResult ac_gguf_open(const char* file_path, GgufModel** out_model);

/**
 * @brief Find tensor by exact name. Returns NULL if not found.
 */
GgufTensor* ac_gguf_get_tensor(GgufModel* model, const char* tensor_name);

/**
 * @brief Get official FP32 scale factor for quantized tensors (e.g. I2_S)
 */
AcResult ac_gguf_get_tensor_scale(const GgufTensor* tensor, float* out_scale);

/**
 * @brief Close and unmap GGUF model.
 */
void ac_gguf_close(GgufModel* model);

#ifdef __cplusplus
}
#endif

#endif /* AMUDA_GGUF_H */
