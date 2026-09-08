/**
 * @file bitnet_gemv.cl
 * @brief BitNet 1.58b Ternary {-1, 0, +1} High-Speed GEMV Kernel
 *
 * Matrix W is packed with 4 ternary weights per unsigned char (byte):
 *   bits [1:0] : weight 0 (0: 0, 1: +1, 2: -1)
 *   bits [3:2] : weight 1
 *   bits [5:4] : weight 2
 *   bits [7:6] : weight 3
 *
 * Computes: y[i] = scales[i] * sum_{k=0}^{K-1} W[i, k] * x[k]
 */

static inline float decode_ternary(uchar code) {
    // 0 -> 0.0f, 1 -> 1.0f, 2 -> -1.0f
    return (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
}

__kernel void bitnet_gemv_i2(
    const int M,
    const int K,
    const __global float* x,
    const __global uchar* W_packed,
    const __global float* scales,
    __global float* y
) {
    int row = get_global_id(0); // Output row index in [0, M-1]
    if (row >= M) return;

    int num_bytes = K / 4;
    int row_offset = row * num_bytes;

    float acc = 0.0f;

    for (int b = 0; b < num_bytes; ++b) {
        uchar packed = W_packed[row_offset + b];
        int k_base = b * 4;

        uchar c0 = packed & 0x03;
        uchar c1 = (packed >> 2) & 0x03;
        uchar c2 = (packed >> 4) & 0x03;
        uchar c3 = (packed >> 6) & 0x03;

        acc += decode_ternary(c0) * x[k_base];
        acc += decode_ternary(c1) * x[k_base + 1];
        acc += decode_ternary(c2) * x[k_base + 2];
        acc += decode_ternary(c3) * x[k_base + 3];
    }

    y[row] = acc * scales[row];
}
