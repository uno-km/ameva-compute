"""
Sprint 3 FP16 E2E Verification Script
=======================================
FP16 MatMul SPIR-V 셰이더를 실기기에서 GPU 디스패치한 뒤
FP32 레퍼런스와 비교하여 수치 정밀도를 검증한다.
"""

import sys
sys.path.insert(0, 'python')
import ameva_compute as ac
import numpy as np
import struct
import time

def load_spv(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()

def run_fp16_matmul_test():
    print("=" * 70)
    print("AMUDA Sprint 3: FP16 MatMul E2E Hardware Verification")
    print("=" * 70)

    M, K, N = 64, 128, 64
    print(f"\nMatrix dims: M={M}, K={K}, N={N}")

    # Generate random test data in FP32
    rng = np.random.default_rng(42)
    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B_f32 = rng.standard_normal((K, N)).astype(np.float32) * 0.1

    # FP32 reference result (CPU)
    C_ref_f32 = A_f32 @ B_f32

    # Convert to FP16 for GPU
    A_f16 = A_f32.astype(np.float16)
    B_f16 = B_f32.astype(np.float16)

    # FP16 reference (CPU) for fair comparison
    C_ref_f16 = (A_f16.astype(np.float32) @ B_f16.astype(np.float32)).astype(np.float16)

    print(f"A range: [{A_f16.min():.4f}, {A_f16.max():.4f}]")
    print(f"B range: [{B_f16.min():.4f}, {B_f16.max():.4f}]")

    # Load FP16 SPIR-V
    spv = load_spv('build/matmul_f16.spv')
    print(f"SPIR-V size: {len(spv)} bytes")

    with ac.init() as ctx:
        print(f"GPU Device: {ctx.device_name}")

        # Create FP16 buffers
        buf_a = ctx.buffer(shape=(M * K,), dtype='float16', data=A_f16.flatten())
        buf_b = ctx.buffer(shape=(K * N,), dtype='float16', data=B_f16.flatten())
        buf_c = ctx.buffer(shape=(M * N,), dtype='float16')

        # Create kernel with 3 storage bindings
        kernel = ctx.kernel(
            spirv_code=spv,
            name='matmul_f16',
            entry_point='main',
            bindings=[(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                      (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                      (2, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)],
            local_size=(16, 16, 1),
            push_constant_size=12,  # 3x uint32: M, K, N
        )

        # Push constants: M, K, N as uint32
        pc = struct.pack('<III', M, K, N)

        # Warmup dispatch
        event = ctx.dispatch(kernel, buffers=[buf_a, buf_b, buf_c],
                           grid=((N + 15) // 16, (M + 15) // 16, 1),
                           push_constants=pc)
        event.wait()
        print(f"Warmup: wall={event.wall_latency_us:.1f} µs, gpu={event.gpu_duration_us:.1f} µs")

        # Timed dispatch
        times_wall = []
        times_gpu = []
        for i in range(10):
            event = ctx.dispatch(kernel, buffers=[buf_a, buf_b, buf_c],
                               grid=((N + 15) // 16, (M + 15) // 16, 1),
                               push_constants=pc)
            event.wait()
            times_wall.append(event.wall_latency_us)
            times_gpu.append(event.gpu_duration_us)
            event.destroy()

        # Read result
        C_gpu = buf_c.numpy().reshape(M, N).copy()

        # Numerical comparison
        # Compare GPU FP16 output vs CPU FP16 reference
        abs_diff = np.abs(C_gpu.astype(np.float32) - C_ref_f16.astype(np.float32))
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        rmse = np.sqrt(np.mean(abs_diff ** 2))

        # Cosine similarity
        c_flat = C_gpu.astype(np.float32).flatten()
        r_flat = C_ref_f16.astype(np.float32).flatten()
        cos_sim = np.dot(c_flat, r_flat) / (np.linalg.norm(c_flat) * np.linalg.norm(r_flat) + 1e-10)

        # Also compare with FP32 reference
        abs_diff_32 = np.abs(C_gpu.astype(np.float32) - C_ref_f32)
        cos_sim_32 = np.dot(c_flat, C_ref_f32.flatten()) / (np.linalg.norm(c_flat) * np.linalg.norm(C_ref_f32.flatten()) + 1e-10)

        print(f"\n--- Results (10 dispatches) ---")
        print(f"Wall latency: avg={np.mean(times_wall):.1f} µs, min={np.min(times_wall):.1f} µs")
        print(f"GPU duration:  avg={np.mean(times_gpu):.1f} µs, min={np.min(times_gpu):.1f} µs")
        print(f"\n--- FP16 vs FP16 CPU Reference ---")
        print(f"Max abs diff:  {max_diff:.6f}")
        print(f"Mean abs diff: {mean_diff:.6f}")
        print(f"RMSE:          {rmse:.6f}")
        print(f"Cosine Sim:    {cos_sim:.10f}")
        print(f"\n--- FP16 GPU vs FP32 CPU Reference ---")
        print(f"Max abs diff:  {abs_diff_32.max():.6f}")
        print(f"Cosine Sim:    {cos_sim_32:.10f}")

        # Pass/Fail gates
        gates = []
        gates.append(("Cosine Sim vs FP16 ref >= 0.9999", cos_sim >= 0.9999))
        gates.append(("Cosine Sim vs FP32 ref >= 0.9990", cos_sim_32 >= 0.9990))
        gates.append(("RMSE < 0.01", rmse < 0.01))
        gates.append(("Max diff < 0.1", max_diff < 0.1))

        print(f"\n--- Gates ---")
        all_pass = True
        for name, passed in gates:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{status}] {name}")

        # Cleanup
        buf_a.destroy()
        buf_b.destroy()
        buf_c.destroy()
        kernel.destroy()

    print(f"\n{'=' * 70}")
    result = "PASSED" if all_pass else "FAILED"
    print(f"FP16 MatMul E2E Verification: {result}")
    print(f"{'=' * 70}")
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(run_fp16_matmul_test())
