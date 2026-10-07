# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA-COMPUTE (AMUDA) Python Package
=====================================
Ultra-lightweight CUDA-like GPGPU Runtime for Mobile & Termux.
Powered by the AMUDA Core Engine v2.1.0.

ABI: ac_runtime.h v2.1.0 (0x00020100)
Backend: Vulkan 1.3 SPIR-V Compute

Usage:
    import ameva_compute as ac

    with ac.init() as ctx:
        a = ctx.buffer(shape=(1024,), dtype='float32', data=np.ones(1024))
        b = ctx.buffer(shape=(1024,), dtype='float32', data=np.ones(1024) * 2)
        c = ctx.buffer(shape=(1024,), dtype='float32')

        kernel = ctx.kernel(spirv_code=spirv_bytes, name='vector_add',
                            binding_count=3)
        event = ctx.dispatch(kernel, buffers=[a, b, c], grid=(4,))
        event.wait()
        result = c.numpy()  # UMA zero-copy, no data transfer
"""

from .core import (
    # Entry point
    init,
    # Handle classes
    Context,
    Buffer,
    Kernel,
    Event,
    # Exceptions
    AmevaComputeError,
    AmevaDriverNotFoundError,
    # ABI Constants
    AC_ABI_VERSION,
    AC_BACKEND_VULKAN,
    AC_BUFFER_USAGE_STORAGE,
    AC_BUFFER_USAGE_TRANSFER_SRC,
    AC_BUFFER_USAGE_TRANSFER_DST,
    AC_BUFFER_USAGE_UNIFORM,
    AC_STAGE_COMPUTE,
    AC_RUNTIME_BINDING_STORAGE_BUFFER,
    AC_RUNTIME_BINDING_UNIFORM_BUFFER,
    AC_SPEC_CONSTANT_U32,
    AC_SPEC_CONSTANT_I32,
    AC_SPEC_CONSTANT_F32,
    AC_DEFAULT_TIMEOUT_NS,
    AC_SUCCESS,
)

__version__ = "2.1.0"  # must equal pyproject.toml [project].version (python/tests/test_core.py)

# High-level operations API
from . import ops

__all__ = [
    "init",
    "Context",
    "Buffer",
    "Kernel",
    "Event",
    "AmevaComputeError",
    "AmevaDriverNotFoundError",
    "ops",
    "AC_ABI_VERSION",
    "AC_BACKEND_VULKAN",
    "AC_BUFFER_USAGE_STORAGE",
    "AC_BUFFER_USAGE_TRANSFER_SRC",
    "AC_BUFFER_USAGE_TRANSFER_DST",
    "AC_BUFFER_USAGE_UNIFORM",
    "AC_STAGE_COMPUTE",
    "AC_RUNTIME_BINDING_STORAGE_BUFFER",
    "AC_RUNTIME_BINDING_UNIFORM_BUFFER",
    "AC_SPEC_CONSTANT_U32",
    "AC_SPEC_CONSTANT_I32",
    "AC_SPEC_CONSTANT_F32",
    "AC_DEFAULT_TIMEOUT_NS",
    "AC_SUCCESS",
]

