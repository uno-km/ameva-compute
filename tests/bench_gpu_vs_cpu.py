"""
AMUDA ops.matmul GPU vs NumPy CPU Benchmark
=============================================
Sprint 4 verification: measure actual GPU speedup.
"""
import sys
import time
import numpy as np

sys.path.insert(0, 'python')

# --- GPU Test ---
print("=" * 60)
print("AMUDA GPU vs NumPy CPU MatMul Benchmark")
print("=" * 60)

import ameva_compute as ac

sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]

with ac.init() as ctx:
    print(f"GPU: {ctx.device_name}")
    print(f"AMUDA version: {ac.__version__}\n")

    for M, N in sizes:
        K = M  # square matrices
        a = np.random.randn(M, K).astype(np.float32) * 0.1
        b = np.random.randn(K, N).astype(np.float32) * 0.1

        # CPU warmup + timing
        _ = np.matmul(a, b)
        t0 = time.perf_counter()
        for _ in range(20):
            c_cpu = np.matmul(a, b)
        cpu_ms = (time.perf_counter() - t0) / 20 * 1000

        # GPU warmup + timing
        _ = ac.ops.matmul(ctx, a, b)
        t0 = time.perf_counter()
        for _ in range(20):
            c_gpu = ac.ops.matmul(ctx, a, b, dtype='float32')
        gpu_ms = (time.perf_counter() - t0) / 20 * 1000

        # Accuracy check
        diff = np.abs(c_gpu - c_cpu).max()
        cos = np.dot(c_gpu.flatten(), c_cpu.flatten()) / (
            np.linalg.norm(c_gpu.flatten()) * np.linalg.norm(c_cpu.flatten()) + 1e-10
        )

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        marker = "GPU WIN" if speedup > 1.0 else "CPU WIN"

        print(f"  {M}x{K} @ {K}x{N}:")
        print(f"    CPU: {cpu_ms:.3f} ms   GPU: {gpu_ms:.3f} ms   "
              f"Speedup: {speedup:.2f}x  [{marker}]")
        print(f"    Max diff: {diff:.6f}   Cosine: {cos:.8f}")
        print()

    print("Benchmark complete.")
    ac.ops.cleanup_cache(ctx)
