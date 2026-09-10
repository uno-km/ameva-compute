/**
 * @file amuda_ops.c
 * @brief High-Performance Deep Learning and BLAS Operations Implementation
 */

#include "amuda_ops.h"
#include "amuda_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* 1. Inlined Kernel Source Definitions                                      */
/* ========================================================================= */

static const char* k_gemm_src =
"#define TILE_DIM 16\n"
"__kernel void gemm_tiled_fp32(\n"
"    const int M, const int N, const int K,\n"
"    const float alpha,\n"
"    const __global float* A, const __global float* B,\n"
"    const float beta, __global float* C\n"
") {\n"
"    int col = get_global_id(0);\n"
"    int row = get_global_id(1);\n"
"    int localCol = get_local_id(0);\n"
"    int localRow = get_local_id(1);\n"
"    __local float Asub[TILE_DIM][TILE_DIM];\n"
"    __local float Bsub[TILE_DIM][TILE_DIM];\n"
"    float sum = 0.0f;\n"
"    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
"    for (int t = 0; t < numTiles; ++t) {\n"
"        int tK = t * TILE_DIM;\n"
"        if (row < M && (tK + localCol) < K) {\n"
"            Asub[localRow][localCol] = A[row * K + (tK + localCol)];\n"
"        } else {\n"
"            Asub[localRow][localCol] = 0.0f;\n"
"        }\n"
"        if ((tK + localRow) < K && col < N) {\n"
"            Bsub[localRow][localCol] = B[(tK + localRow) * N + col];\n"
"        } else {\n"
"            Bsub[localRow][localCol] = 0.0f;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int k = 0; k < TILE_DIM; ++k) {\n"
"            sum += Asub[localRow][k] * Bsub[k][localCol];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (row < M && col < N) {\n"
"        int idx = row * N + col;\n"
"        if (beta != 0.0f) {\n"
"            C[idx] = alpha * sum + beta * C[idx];\n"
"        } else {\n"
"            C[idx] = alpha * sum;\n"
"        }\n"
"    }\n"
"}\n";

static const char* k_bitnet_src =
"static inline float decode_ternary(uchar code) {\n"
"    return (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);\n"
"}\n"
"__kernel void bitnet_gemv_i2(\n"
"    const int M, const int K,\n"
"    const __global float* x,\n"
"    const __global uchar* W_packed,\n"
"    const __global float* scales,\n"
"    __global float* y\n"
") {\n"
"    int row = get_global_id(0);\n"
"    if (row >= M) return;\n"
"    int num_bytes = K / 4;\n"
"    int row_offset = row * num_bytes;\n"
"    float acc = 0.0f;\n"
"    for (int b = 0; b < num_bytes; ++b) {\n"
"        uchar packed = W_packed[row_offset + b];\n"
"        int k_base = b * 4;\n"
"        acc += decode_ternary(packed & 0x03) * x[k_base];\n"
"        acc += decode_ternary((packed >> 2) & 0x03) * x[k_base + 1];\n"
"        acc += decode_ternary((packed >> 4) & 0x03) * x[k_base + 2];\n"
"        acc += decode_ternary((packed >> 6) & 0x03) * x[k_base + 3];\n"
"    }\n"
"    y[row] = acc * scales[row];\n"
"}\n";

static const char* k_bitnet_i2s_exact_src =
"__kernel void bitnet_gemv_i2_s_exact(\n"
"    const int M, const int K,\n"
"    const __global float* x_vec,\n"
"    const __global uchar* W_packed,\n"
"    const float tensor_scale,\n"
"    __global float* y_out\n"
") {\n"
"    int row = get_global_id(0);\n"
"    if (row >= M) return;\n"
"    constant float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };\n"
"    int row_byte_offset = row * (K / 4);\n"
"    int num_blocks = K / 128;\n"
"    float acc = 0.0f;\n"
"    for (int b = 0; b < num_blocks; ++b) {\n"
"        int blk_byte_base = row_byte_offset + b * 32;\n"
"        int blk_k_base = b * 128;\n"
"        for (int gp = 0; gp < 32; ++gp) {\n"
"            uchar byte = W_packed[blk_byte_base + gp];\n"
"            uchar c0 = (byte >> 6) & 0x03;\n"
"            uchar c1 = (byte >> 4) & 0x03;\n"
"            uchar c2 = (byte >> 2) & 0x03;\n"
"            uchar c3 = (byte >> 0) & 0x03;\n"
"            acc += map2bit[c0] * x_vec[blk_k_base + 0 + gp];\n"
"            acc += map2bit[c1] * x_vec[blk_k_base + 32 + gp];\n"
"            acc += map2bit[c2] * x_vec[blk_k_base + 64 + gp];\n"
"            acc += map2bit[c3] * x_vec[blk_k_base + 96 + gp];\n"
"        }\n"
"    }\n"
"    y_out[row] = acc * tensor_scale;\n"
"}\n";

static const char* k_transformer_src =
"__kernel void rmsnorm_forward(\n"
"    const int num_rows, const int hidden_dim, const float eps,\n"
"    const __global float* x, const __global float* weight, __global float* y\n"
") {\n"
"    int row = get_global_id(0);\n"
"    if (row >= num_rows) return;\n"
"    int offset = row * hidden_dim;\n"
"    float sum_sq = 0.0f;\n"
"    for (int i = 0; i < hidden_dim; ++i) {\n"
"        float val = x[offset + i];\n"
"        sum_sq += val * val;\n"
"    }\n"
"    float inv_rms = rsqrt((sum_sq / (float)hidden_dim) + eps);\n"
"    for (int i = 0; i < hidden_dim; ++i) {\n"
"        y[offset + i] = x[offset + i] * inv_rms * weight[i];\n"
"    }\n"
"}\n"
"__kernel void swiglu_forward(\n"
"    const int num_elements,\n"
"    const __global float* gate, const __global float* up, __global float* y\n"
") {\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= num_elements) return;\n"
"    float g = gate[idx];\n"
"    float silu = g / (1.0f + exp(-g));\n"
"    y[idx] = silu * up[idx];\n"
"}\n"
"__kernel void rope_forward_inplace(\n"
"    const int total_pairs, const int num_heads, const int half_dim,\n"
"    __global float* qk, const __global float* cos_sin\n"
") {\n"
"    int pair_idx = get_global_id(0);\n"
"    if (pair_idx >= total_pairs) return;\n"
"    int d = pair_idx % half_dim;\n"
"    int temp = pair_idx / half_dim;\n"
"    int h = temp % num_heads;\n"
"    int s = temp / num_heads;\n"
"    int head_dim = half_dim * 2;\n"
"    int qk_idx0 = s * (num_heads * head_dim) + h * head_dim + (d * 2);\n"
"    int qk_idx1 = qk_idx0 + 1;\n"
"    int cs_idx_cos = s * head_dim + d;\n"
"    int cs_idx_sin = s * head_dim + half_dim + d;\n"
"    float cos_val = cos_sin[cs_idx_cos];\n"
"    float sin_val = cos_sin[cs_idx_sin];\n"
"    float x0 = qk[qk_idx0];\n"
"    float x1 = qk[qk_idx1];\n"
"    qk[qk_idx0] = x0 * cos_val - x1 * sin_val;\n"
"    qk[qk_idx1] = x0 * sin_val + x1 * cos_val;\n"
"}\n";

/* ========================================================================= */
/* 2. Operations Dispatch Implementations                                    */
/* ========================================================================= */

AcResult ac_ops_gemm_fp32(
    AcContext* ctx,
    int M, int N, int K,
    float alpha,
    AcBuffer* A,
    AcBuffer* B,
    float beta,
    AcBuffer* C,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !A || !B || !C) return AC_ERR_NULL_POINTER;
    if (M <= 0 || N <= 0 || K <= 0) return AC_ERR_KERNEL_ARG_INVALID;

    /* Dimension sanity and arithmetic overflow checks */
    if ((uint64_t)M > 0x10000000ULL || (uint64_t)N > 0x10000000ULL || (uint64_t)K > 0x10000000ULL) {
        return AC_ERR_NUMERICAL_OVERFLOW;
    }

    uint64_t req_a = (uint64_t)M * (uint64_t)K * sizeof(float);
    uint64_t req_b = (uint64_t)K * (uint64_t)N * sizeof(float);
    uint64_t req_c = (uint64_t)M * (uint64_t)N * sizeof(float);

    if (A->size_bytes < req_a || B->size_bytes < req_b || C->size_bytes < req_c) {
        return AC_ERR_INVALID_BUFFER_SIZE;
    }

    // Synchronize host writes to device DRAM
    ac_buffer_sync_to_device(ctx, A);
    ac_buffer_sync_to_device(ctx, B);
    ac_buffer_sync_to_device(ctx, C);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_gemm_src, "gemm_tiled_fp32", &kernel);
    if (res != AC_SUCCESS) return res;

    // 2D Workgroup Grid
    uint32_t block_dim = 16;
    uint32_t grid_x = ((uint32_t)N + block_dim - 1) / block_dim;
    uint32_t grid_y = ((uint32_t)M + block_dim - 1) / block_dim;

    void* args[8] = {
        &M, &N, &K, &alpha,
        &A->native_mem, &B->native_mem,
        &beta, &C->native_mem
    };
    size_t arg_sizes[8] = {
        sizeof(int), sizeof(int), sizeof(int), sizeof(float),
        sizeof(void*), sizeof(void*),
        sizeof(float), sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim3(grid_x, grid_y, 1),
                    ac_dim3(block_dim, block_dim, 1),
                    8, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}

AcResult ac_ops_bitnet_gemv_i2(
    AcContext* ctx,
    int M, int K,
    AcBuffer* x,
    AcBuffer* W_packed,
    AcBuffer* scales,
    AcBuffer* y,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !x || !W_packed || !scales || !y) return AC_ERR_NULL_POINTER;
    if (M <= 0 || K <= 0) return AC_ERR_KERNEL_ARG_INVALID;
    if (K % 4 != 0) return AC_ERR_INVALID_BUFFER_SIZE;

    ac_buffer_sync_to_device(ctx, x);
    ac_buffer_sync_to_device(ctx, W_packed);
    ac_buffer_sync_to_device(ctx, scales);
    ac_buffer_sync_to_device(ctx, y);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_bitnet_src, "bitnet_gemv_i2", &kernel);
    if (res != AC_SUCCESS) return res;

    uint32_t block_size = 64;
    uint32_t grid_size = ((uint32_t)M + block_size - 1) / block_size;

    void* args[6] = {
        &M, &K,
        &x->native_mem, &W_packed->native_mem, &scales->native_mem,
        &y->native_mem
    };
    size_t arg_sizes[6] = {
        sizeof(int), sizeof(int),
        sizeof(void*), sizeof(void*), sizeof(void*),
        sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim1(grid_size),
                    ac_dim1(block_size),
                    6, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}

AcResult ac_ops_bitnet_gemv_i2_s_exact(
    AcContext* ctx,
    int M, int K,
    AcBuffer* x,
    AcBuffer* W_packed,
    float tensor_scale,
    AcBuffer* y,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !x || !W_packed || !y) return AC_ERR_NULL_POINTER;
    if (M <= 0 || K <= 0) return AC_ERR_KERNEL_ARG_INVALID;
    if (K % 128 != 0) return AC_ERR_KERNEL_ARG_INVALID;

    ac_buffer_sync_to_device(ctx, x);
    ac_buffer_sync_to_device(ctx, W_packed);
    ac_buffer_sync_to_device(ctx, y);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_bitnet_i2s_exact_src, "bitnet_gemv_i2_s_exact", &kernel);
    if (res != AC_SUCCESS) return res;

    uint32_t block_size = 64;
    uint32_t grid_size = ((uint32_t)M + block_size - 1) / block_size;

    void* args[6] = {
        &M, &K,
        &x->native_mem, &W_packed->native_mem, &tensor_scale,
        &y->native_mem
    };
    size_t arg_sizes[6] = {
        sizeof(int), sizeof(int),
        sizeof(void*), sizeof(void*), sizeof(float),
        sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim1(grid_size),
                    ac_dim1(block_size),
                    6, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}

AcResult ac_ops_rmsnorm(
    AcContext* ctx,
    int num_rows,
    int hidden_dim,
    float eps,
    AcBuffer* x,
    AcBuffer* weight,
    AcBuffer* y,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !x || !weight || !y) return AC_ERR_NULL_POINTER;
    if (num_rows <= 0 || hidden_dim <= 0) return AC_ERR_KERNEL_ARG_INVALID;

    ac_buffer_sync_to_device(ctx, x);
    ac_buffer_sync_to_device(ctx, weight);
    ac_buffer_sync_to_device(ctx, y);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_transformer_src, "rmsnorm_forward", &kernel);
    if (res != AC_SUCCESS) return res;

    uint32_t block_size = 64;
    uint32_t grid_size = ((uint32_t)num_rows + block_size - 1) / block_size;

    void* args[6] = {
        &num_rows, &hidden_dim, &eps,
        &x->native_mem, &weight->native_mem, &y->native_mem
    };
    size_t arg_sizes[6] = {
        sizeof(int), sizeof(int), sizeof(float),
        sizeof(void*), sizeof(void*), sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim1(grid_size),
                    ac_dim1(block_size),
                    6, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}

AcResult ac_ops_swiglu(
    AcContext* ctx,
    int num_elements,
    AcBuffer* gate,
    AcBuffer* up,
    AcBuffer* y,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !gate || !up || !y) return AC_ERR_NULL_POINTER;
    if (num_elements <= 0) return AC_ERR_KERNEL_ARG_INVALID;

    ac_buffer_sync_to_device(ctx, gate);
    ac_buffer_sync_to_device(ctx, up);
    ac_buffer_sync_to_device(ctx, y);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_transformer_src, "swiglu_forward", &kernel);
    if (res != AC_SUCCESS) return res;

    uint32_t block_size = 64;
    uint32_t grid_size = ((uint32_t)num_elements + block_size - 1) / block_size;

    void* args[4] = {
        &num_elements,
        &gate->native_mem, &up->native_mem, &y->native_mem
    };
    size_t arg_sizes[4] = {
        sizeof(int),
        sizeof(void*), sizeof(void*), sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim1(grid_size),
                    ac_dim1(block_size),
                    4, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}

AcResult ac_ops_rope(
    AcContext* ctx,
    int seq_len,
    int num_heads,
    int head_dim,
    AcBuffer* qk,
    AcBuffer* cos_sin,
    AcEvent** out_event
) {
    if (out_event) {
        *out_event = NULL;
    }
    if (!ctx || !qk || !cos_sin) return AC_ERR_NULL_POINTER;
    if (seq_len <= 0 || num_heads <= 0 || head_dim <= 0) return AC_ERR_KERNEL_ARG_INVALID;
    if (head_dim % 2 != 0) return AC_ERR_INVALID_BUFFER_SIZE;

    ac_buffer_sync_to_device(ctx, qk);
    ac_buffer_sync_to_device(ctx, cos_sin);

    AcKernel* kernel = NULL;
    AcResult res = ac_kernel_load_cached(ctx, k_transformer_src, "rope_forward_inplace", &kernel);
    if (res != AC_SUCCESS) return res;

    int half_dim = head_dim / 2;
    int total_pairs = seq_len * num_heads * half_dim;

    uint32_t block_size = 64;
    uint32_t grid_size = ((uint32_t)total_pairs + block_size - 1) / block_size;

    void* args[5] = {
        &total_pairs, &num_heads, &half_dim,
        &qk->native_mem, &cos_sin->native_mem
    };
    size_t arg_sizes[5] = {
        sizeof(int), sizeof(int), sizeof(int),
        sizeof(void*), sizeof(void*)
    };

    res = ac_launch(ctx, kernel,
                    ac_dim1(grid_size),
                    ac_dim1(block_size),
                    5, args, arg_sizes, out_event);

    ac_kernel_free(ctx, kernel);
    return res;
}
