/**
 * @file bitnet_i2s.cl
 * @brief Official GGML_TYPE_I2_S (Type 36) Native SIMT Dequantization & GEMV Kernel
 *
 * Implements exact 32-stride 4-way interleaved ternary (-1, 0, 1) matrix-vector product.
 * Layout: 128 elements per block packed into 32 bytes.
 * Byte gp (0..31):
 *   bits 7..6: element gp + 0
 *   bits 5..4: element gp + 32
 *   bits 3..2: element gp + 64
 *   bits 1..0: element gp + 96
 * Bit mapping: 0 -> -1.0, 1 -> 0.0, 2 -> +1.0, 3 -> 0.0
 */

__kernel void k_bitnet_gemv_i2_s_exact(
    const int M,                    // Number of rows (out dim, e.g. 2560)
    const int K,                    // Number of columns (in dim, e.g. 6912)
    __global const uchar* W_packed, // Packed ternary matrix (M * (K/4) bytes)
    __global const float* x_vec,    // Input activation vector (K elements)
    const float tensor_scale,       // Official FP32 tensor scale from GGUF
    __global float* y_out           // Output activation vector (M elements)
) {
    int row = get_global_id(0);
    if (row >= M) return;

    constant float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };

    int row_byte_offset = row * (K / 4);
    int num_blocks = K / 128;

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        int blk_byte_base = row_byte_offset + b * 32;
        int blk_k_base = b * 128;

        #pragma unroll 8
        for (int gp = 0; gp < 32; ++gp) {
            uchar byte = W_packed[blk_byte_base + gp];

            uchar c0 = (byte >> 6) & 0x03;
            uchar c1 = (byte >> 4) & 0x03;
            uchar c2 = (byte >> 2) & 0x03;
            uchar c3 = (byte >> 0) & 0x03;

            acc += map2bit[c0] * x_vec[blk_k_base + 0 + gp];
            acc += map2bit[c1] * x_vec[blk_k_base + 32 + gp];
            acc += map2bit[c2] * x_vec[blk_k_base + 64 + gp];
            acc += map2bit[c3] * x_vec[blk_k_base + 96 + gp];
        }
    }

    y_out[row] = acc * tensor_scale;
}
