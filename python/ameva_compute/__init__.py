"""
AMEVA-COMPUTE (AMUDA) Python Package
====================================
Ultra-lightweight CUDA-like GPGPU Runtime for Mobile & Termux.
Powered by the AMUDA Core Engine.
"""

from .core import (
    init,
    Context,
    Buffer,
    Kernel,
    Event,
    AmevaComputeError,
)

__version__ = "0.1.0"
__all__ = [
    "init",
    "Context",
    "Buffer",
    "Kernel",
    "Event",
    "AmevaComputeError",
]
