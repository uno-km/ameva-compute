# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMUDA High-Level Operations API
================================
Provides easy-to-use GPU compute primitives built on ac_runtime.
Each function loads the pre-compiled SPIR-V shader, dispatches on GPU,
and returns results as numpy arrays.

This is the primary interface for integrating AMUDA into ameva-runtime
adapters (bitnet, train, etc.)

Usage:
    import amuda
    ctx = amuda.init()
    c = amuda.ops.matmul(ctx, a, b, dtype='float16')
    probs = amuda.ops.softmax(ctx, logits)
    normed = amuda.ops.rmsnorm(ctx, x, weight, eps=1e-5)
"""

import os
import struct
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

# Lazy import to avoid circular dependencies
_ac = None
_lib = None
_shader_cache = {}
_kernel_cache = {}   # (ctx_id, shader_name, bindings_key, local_size, pc_size) -> Kernel
_buffer_pool = {}    # (ctx_id, size_bytes, dtype) -> [Buffer, ...]


def _get_ac():
    """Lazy import of ameva_compute core module."""
    global _ac
    if _ac is None:
        import ameva_compute as ac
        _ac = ac
    return _ac


def _find_shader_dir() -> Path:
    """Find the directory containing pre-compiled .spv files."""
    # Priority: env var > relative to this file > build/ in project root
    env_path = os.environ.get('AMUDA_SHADER_DIR')
    if env_path and Path(env_path).is_dir():
        return Path(env_path)

    # Relative to ameva-compute project root
    candidates = [
        Path(__file__).parent.parent / 'build',           # python/amuda/ops.py -> build/
        Path(__file__).parent.parent.parent / 'build',     # one more level up
        Path.home() / 'ameva-compute' / 'build',          # Termux default
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob('*.spv')):
            return p

    raise FileNotFoundError(
        "AMUDA SPIR-V shader directory not found. "
        "Set AMUDA_SHADER_DIR or build shaders first."
    )


def _load_shader(name: str) -> bytes:
    """Load a SPIR-V shader by name, with caching."""
    if name in _shader_cache:
        return _shader_cache[name]

    shader_dir = _find_shader_dir()
    path = shader_dir / f'{name}.spv'
    if not path.exists():
        raise FileNotFoundError(f"Shader not found: {path}")

    with open(path, 'rb') as f:
        data = f.read()

    _shader_cache[name] = data
    return data


def _get_or_create_kernel(ctx, shader_name, bindings, local_size, pc_size):
    """Get cached kernel or create new one. Avoids expensive pipeline recreation."""
    ac = _get_ac()
    ctx_id = id(ctx)
    cache_key = (ctx_id, shader_name, tuple(bindings), local_size, pc_size)

    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    spv = _load_shader(shader_name)
    kernel = ctx.kernel(
        spirv_code=spv,
        name=shader_name,
        entry_point='main',
        bindings=bindings,
        local_size=local_size,
        push_constant_size=pc_size,
    )
    _kernel_cache[cache_key] = kernel
    return kernel


def cleanup_cache(ctx=None):
    """Destroy all cached kernels for a context (or all contexts).
    Must be called before context.destroy() to avoid AC_ERR_RESOURCE_IN_USE."""
    global _kernel_cache
    ctx_id = id(ctx) if ctx is not None else None
    to_remove = []
    failures = []
    for key, kernel in _kernel_cache.items():
        if ctx_id is None or key[0] == ctx_id:
            # keep destroying the rest, then report every failure: a kernel that failed to destroy keeps its
            # context busy, and hiding that would only resurface later as AC_ERR_RESOURCE_IN_USE
            try:
                kernel.destroy()
            except Exception as e:
                failures.append("%s: %s" % (key[1] if len(key) > 1 else key, e))
            to_remove.append(key)
    for key in to_remove:
        del _kernel_cache[key]
    if failures:
        raise RuntimeError("cleanup_cache: %d kernel(s) failed to destroy: %s" % (len(failures), "; ".join(failures)))

def _ensure_f16(arr: np.ndarray) -> np.ndarray:
    """Convert array to float16 if not already."""
    if arr.dtype != np.float16:
        return arr.astype(np.float16)
    return arr


# ============================================================================
# Core Operations
# ============================================================================

def matmul(ctx, a: np.ndarray, b: np.ndarray,
           dtype: str = 'float16') -> np.ndarray:
    """
    GPU Matrix Multiplication: C = A @ B

    Args:
        ctx: AMUDA context (from amuda.init())
        a: Input matrix (M, K)
        b: Input matrix (K, N)
        dtype: Output dtype ('float16' or 'float32')

    Returns:
        Result matrix (M, N)
    """
    ac = _get_ac()
    assert a.ndim == 2 and b.ndim == 2, f"Expected 2D, got {a.ndim}D and {b.ndim}D"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: ({M},{K}) @ ({K2},{N})"

    a16 = _ensure_f16(a)
    b16 = _ensure_f16(b)

    buf_a = ctx.buffer(shape=(M * K,), dtype='float16', data=a16.flatten())
    buf_b = ctx.buffer(shape=(K * N,), dtype='float16', data=b16.flatten())
    buf_c = ctx.buffer(shape=(M * N,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (2, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(
        ctx, 'matmul_f16', bindings, (16, 16, 1), 12
    )

    pc = struct.pack('<III', M, K, N)
    event = ctx.dispatch(kernel, buffers=[buf_a, buf_b, buf_c],
                        grid=((N + 15) // 16, (M + 15) // 16, 1),
                        push_constants=pc)
    event.wait()

    result = buf_c.numpy().reshape(M, N).copy()

    event.destroy()
    # kernel is cached — do NOT destroy
    buf_a.destroy()
    buf_b.destroy()
    buf_c.destroy()

    if dtype == 'float32':
        return result.astype(np.float32)
    return result


def softmax(ctx, x: np.ndarray) -> np.ndarray:
    """
    GPU Softmax along last dimension.

    Args:
        ctx: AMUDA context
        x: Input tensor (..., D) — last dim is softmaxed

    Returns:
        Softmax output, same shape
    """
    ac = _get_ac()
    orig_shape = x.shape
    row_size = orig_shape[-1]
    num_rows = int(np.prod(orig_shape[:-1])) if x.ndim > 1 else 1

    x16 = _ensure_f16(x.reshape(num_rows, row_size))

    buf_in = ctx.buffer(shape=(num_rows * row_size,), dtype='float16',
                       data=x16.flatten())
    buf_out = ctx.buffer(shape=(num_rows * row_size,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'softmax_f16', bindings, (256, 1, 1), 8)

    pc = struct.pack('<II', num_rows, row_size)
    event = ctx.dispatch(kernel, buffers=[buf_in, buf_out],
                        grid=(num_rows, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_out.numpy().reshape(orig_shape).copy()

    event.destroy()
    buf_in.destroy()
    buf_out.destroy()

    return result


def rmsnorm(ctx, x: np.ndarray, weight: np.ndarray,
            eps: float = 1e-5) -> np.ndarray:
    """
    GPU RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)

    Args:
        ctx: AMUDA context
        x: Input tensor (num_tokens, hidden_dim)
        weight: Scale vector (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized output, same shape as x
    """
    ac = _get_ac()
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    num_tokens, hidden_dim = x.shape
    assert weight.shape == (hidden_dim,), f"Weight shape mismatch"

    x16 = _ensure_f16(x)
    w16 = _ensure_f16(weight)

    buf_x = ctx.buffer(shape=(num_tokens * hidden_dim,), dtype='float16',
                      data=x16.flatten())
    buf_w = ctx.buffer(shape=(hidden_dim,), dtype='float16',
                      data=w16.flatten())
    buf_out = ctx.buffer(shape=(num_tokens * hidden_dim,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (2, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'rmsnorm_f16', bindings, (256, 1, 1), 12)

    pc = struct.pack('<IIf', num_tokens, hidden_dim, eps)
    event = ctx.dispatch(kernel, buffers=[buf_x, buf_w, buf_out],
                        grid=(num_tokens, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_out.numpy().reshape(num_tokens, hidden_dim).copy()

    event.destroy()
    buf_x.destroy()
    buf_w.destroy()
    buf_out.destroy()

    return result


def silu(ctx, x: np.ndarray) -> np.ndarray:
    """
    GPU SiLU (Swish) activation: f(x) = x * sigmoid(x)

    Args:
        ctx: AMUDA context
        x: Input tensor (any shape)

    Returns:
        SiLU output, same shape
    """
    ac = _get_ac()
    orig_shape = x.shape
    count = int(np.prod(orig_shape))
    x16 = _ensure_f16(x.flatten())
    spv = _load_shader('silu_f16')

    buf_in = ctx.buffer(shape=(count,), dtype='float16', data=x16)
    buf_out = ctx.buffer(shape=(count,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'silu_f16', bindings, (256, 1, 1), 4)

    pc = struct.pack('<I', count)
    event = ctx.dispatch(kernel, buffers=[buf_in, buf_out],
                        grid=((count + 255) // 256, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_out.numpy().reshape(orig_shape).copy()

    event.destroy()
    buf_in.destroy()
    buf_out.destroy()

    return result


def gelu(ctx, x: np.ndarray) -> np.ndarray:
    """
    GPU GeLU (Tanh approximation) activation.

    Args:
        ctx: AMUDA context
        x: Input tensor (any shape)

    Returns:
        GeLU output, same shape
    """
    ac = _get_ac()
    orig_shape = x.shape
    count = int(np.prod(orig_shape))
    x16 = _ensure_f16(x.flatten())
    spv = _load_shader('gelu_tanh_f16')

    buf_in = ctx.buffer(shape=(count,), dtype='float16', data=x16)
    buf_out = ctx.buffer(shape=(count,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'gelu_tanh_f16', bindings, (256, 1, 1), 4)

    pc = struct.pack('<I', count)
    event = ctx.dispatch(kernel, buffers=[buf_in, buf_out],
                        grid=((count + 255) // 256, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_out.numpy().reshape(orig_shape).copy()

    event.destroy()
    buf_in.destroy()
    buf_out.destroy()

    return result


def elementwise(ctx, a: np.ndarray, b: np.ndarray,
                op: str = 'add') -> np.ndarray:
    """
    GPU element-wise binary operation.

    Args:
        ctx: AMUDA context
        a, b: Input tensors (same shape)
        op: Operation name ('add', 'sub', 'mul', 'div', 'max', 'min')

    Returns:
        Result tensor, same shape
    """
    ac = _get_ac()
    op_map = {'add': 0, 'sub': 1, 'mul': 2, 'div': 3, 'max': 4, 'min': 5}
    if op not in op_map:
        raise ValueError(f"Unknown op '{op}'. Supported: {list(op_map.keys())}")

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    orig_shape = a.shape
    count = int(np.prod(orig_shape))
    a16 = _ensure_f16(a.flatten())
    b16 = _ensure_f16(b.flatten())

    buf_a = ctx.buffer(shape=(count,), dtype='float16', data=a16)
    buf_b = ctx.buffer(shape=(count,), dtype='float16', data=b16)
    buf_c = ctx.buffer(shape=(count,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (2, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'elementwise_f16', bindings, (256, 1, 1), 12)

    pc = struct.pack('<IIf', count, op_map[op], 0.0)
    event = ctx.dispatch(kernel, buffers=[buf_a, buf_b, buf_c],
                        grid=((count + 255) // 256, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_c.numpy().reshape(orig_shape).copy()

    event.destroy()
    buf_a.destroy()
    buf_b.destroy()
    buf_c.destroy()

    return result


def scale(ctx, x: np.ndarray, scalar: float) -> np.ndarray:
    """
    GPU scalar multiplication: out = x * scalar

    Args:
        ctx: AMUDA context
        x: Input tensor (any shape)
        scalar: Scale factor

    Returns:
        Scaled tensor, same shape
    """
    ac = _get_ac()
    orig_shape = x.shape
    count = int(np.prod(orig_shape))
    x16 = _ensure_f16(x.flatten())
    dummy = np.zeros(count, dtype=np.float16)

    buf_a = ctx.buffer(shape=(count,), dtype='float16', data=x16)
    buf_b = ctx.buffer(shape=(count,), dtype='float16', data=dummy)
    buf_c = ctx.buffer(shape=(count,), dtype='float16')

    bindings = [(0, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (1, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER),
                (2, ac.AC_RUNTIME_BINDING_STORAGE_BUFFER)]
    kernel = _get_or_create_kernel(ctx, 'elementwise_f16', bindings, (256, 1, 1), 12)

    pc = struct.pack('<IIf', count, 6, scalar)
    event = ctx.dispatch(kernel, buffers=[buf_a, buf_b, buf_c],
                        grid=((count + 255) // 256, 1, 1),
                        push_constants=pc)
    event.wait()

    result = buf_c.numpy().reshape(orig_shape).copy()

    event.destroy()
    buf_a.destroy()
    buf_b.destroy()
    buf_c.destroy()

    return result
