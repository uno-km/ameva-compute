/**
 * @file gemm.cl
 * @brief 2D Shared-Memory Tiled General Matrix Multiplication (GEMM)
 *
 * Computes: C = alpha * (A @ B) + beta * C
 * Matrix A: M x K (row-major)
 * Matrix B: K x N (row-major)
 * Matrix C: M x N (row-major)
 *
 * Tile size 16x16: 256 threads per workgroup.
 * Fits within Mali/Adreno local memory constraints with zero bank conflicts.
 */

#define TILE_DIM 16

__kernel void gemm_tiled_fp32(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const __global float* A,
    const __global float* B,
    const float beta,
    __global float* C
) {
    int col = get_global_id(0); // N dimension
    int row = get_global_id(1); // M dimension

    int localCol = get_local_id(0);
    int localRow = get_local_id(1);

    __local float Asub[TILE_DIM][TILE_DIM];
    __local float Bsub[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int tK = t * TILE_DIM;

        // Load tile from matrix A
        if (row < M && (tK + localCol) < K) {
            Asub[localRow][localCol] = A[row * K + (tK + localCol)];
        } else {
            Asub[localRow][localCol] = 0.0f;
        }

        // Load tile from matrix B
        if ((tK + localRow) < K && col < N) {
            Bsub[localRow][localCol] = B[(tK + localRow) * N + col];
        } else {
            Bsub[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial inner product
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta != 0.0f) {
            C[idx] = alpha * sum + beta * C[idx];
        } else {
            C[idx] = alpha * sum;
        }
    }
}
