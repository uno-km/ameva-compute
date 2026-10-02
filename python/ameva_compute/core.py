"""
AMEVA-COMPUTE (AMUDA) Python Native Bridge — ac_runtime.h ABI v2.1.0 Direct Binding
===================================================================================
Ground Truth Engineering: Zero-Silent-Fallback, RAII Resource Management,
UMA Zero-Copy NumPy Interface, 31 C ABI Function Direct Binding.

ABI Contract: AC_ABI_VERSION = 0x00020100 (AMUDA 2.1.0)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


def _report_del_failure(obj: object, err: Exception) -> None:
    """An exception cannot leave __del__; print it so a failed release is never silent."""
    stream = getattr(sys, "stderr", None)  # may already be gone during interpreter shutdown
    if stream is not None:
        stream.write("[ameva_compute] %s garbage-collected without destroy(); release failed: %s\n"
                     % (type(obj).__name__, err))

# =============================================================================
# ABI Constants (Mirror of ac_base.h)
# =============================================================================

AC_ABI_VERSION: int = 0x00020100

# Backend Kinds
AC_BACKEND_VULKAN: int = 1

# Buffer Usage Flags
AC_BUFFER_USAGE_STORAGE: int = 1 << 0
AC_BUFFER_USAGE_TRANSFER_SRC: int = 1 << 1
AC_BUFFER_USAGE_TRANSFER_DST: int = 1 << 2
AC_BUFFER_USAGE_UNIFORM: int = 1 << 3

# Shader Stage Flags
AC_STAGE_COMPUTE: int = 1 << 5

# Binding Types
AC_RUNTIME_BINDING_STORAGE_BUFFER: int = 0
AC_RUNTIME_BINDING_UNIFORM_BUFFER: int = 1

# Specialization Constant Types
AC_SPEC_CONSTANT_U32: int = 0
AC_SPEC_CONSTANT_I32: int = 1
AC_SPEC_CONSTANT_F32: int = 2

# Default Timeout (5 seconds in nanoseconds)
AC_DEFAULT_TIMEOUT_NS: int = 5_000_000_000

# AcResult Status Codes
AC_SUCCESS: int = 0

# =============================================================================
# Error Code Registry (Mirror of ac_base.h AcResult enum)
# =============================================================================

_ERROR_MESSAGES = {
    -101: "AC_ERR_INVALID_CONTEXT",
    -102: "AC_ERR_DRIVER_NOT_FOUND",
    -103: "AC_ERR_DRIVER_SYMBOL_MISSING",
    -104: "AC_ERR_DEVICE_NOT_SUPPORTED",
    -105: "AC_ERR_DRIVER_INIT_FAILED",
    -106: "AC_ERR_INVALID_ARGUMENT",
    -107: "AC_ERR_CONTEXT_MISMATCH",
    -108: "AC_ERR_FEATURE_NOT_SUPPORTED",
    -109: "AC_ERR_INVALID_SHAPE",
    -110: "AC_ERR_ABI_MISMATCH",
    -201: "AC_ERR_OUT_OF_HOST_MEMORY",
    -202: "AC_ERR_OUT_OF_DEVICE_MEMORY",
    -203: "AC_ERR_BUFFER_ALIGNMENT_INVALID",
    -204: "AC_ERR_BUFFER_NOT_MAPPED",
    -205: "AC_ERR_NULL_POINTER",
    -206: "AC_ERR_INVALID_BUFFER_SIZE",
    -301: "AC_ERR_KERNEL_BUILD_FAILED",
    -302: "AC_ERR_KERNEL_NOT_FOUND",
    -303: "AC_ERR_KERNEL_ARG_INVALID",
    -304: "AC_ERR_BINARY_CORRUPTED",
    -401: "AC_ERR_INVALID_WORK_GROUP_SIZE",
    -402: "AC_ERR_INVALID_WORK_ITEM_DIMS",
    -403: "AC_ERR_MALI_INFINITE_LOOP_DETECTED",
    -404: "AC_ERR_ADRENO_UNSAFE_SUBGROUP_OP",
    -405: "AC_ERR_LAUNCH_TIMEOUT",
    -406: "AC_ERR_DEVICE_BUSY",
    -407: "AC_ERR_RESOURCE_IN_USE",
    -501: "AC_ERR_NAN_OR_INF_DETECTED",
    -502: "AC_ERR_NUMERICAL_OVERFLOW",
    -601: "AC_ERR_DRIVER_CRASH_OR_LOST",
    -602: "AC_ERR_OS_PROCESS_KILL_SIGNAL",
    -603: "AC_ERR_THERMAL_THROTTLING_TRIPPED",
}


# =============================================================================
# Exception Hierarchy (Zero-Silent-Fallback)
# =============================================================================

class AmevaComputeError(Exception):
    """
    Strict hardware/kernel exception.
    Raised on ANY non-zero AcResult return from the native ABI.
    No silent fallback, no dummy data, no swallowed errors.
    """
    def __init__(self, code: int, message: str = "") -> None:
        error_name = _ERROR_MESSAGES.get(code, f"UNKNOWN_ERROR({code})")
        full_msg = f"[AMUDA AcResult {code}] {error_name}"
        if message:
            full_msg += f": {message}"
        super().__init__(full_msg)
        self.code = code
        self.error_name = error_name


class AmevaDriverNotFoundError(AmevaComputeError):
    """Raised when the native shared library cannot be located or loaded."""
    def __init__(self, search_paths: List[str]) -> None:
        msg = (
            "Failed to locate libac_runtime shared library. "
            f"Searched paths: {search_paths}. "
            "Ensure the AMUDA native library is built and accessible. "
            "Set AMEVA_COMPUTE_LIB_PATH environment variable to override."
        )
        super().__init__(-102, msg)


# =============================================================================
# ctypes Structure Definitions (Mirror of ac_runtime.h descriptors)
# =============================================================================

class AcDim3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("z", ctypes.c_uint32),
    ]


class AcContextDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("backend_kind", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class AcBufferDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("size_bytes", ctypes.c_size_t),
        ("usage", ctypes.c_uint32),
        ("force_non_coherent", ctypes.c_uint32),
    ]


class AcRuntimeBindingDesc(ctypes.Structure):
    _fields_ = [
        ("binding_index", ctypes.c_uint32),
        ("binding_type", ctypes.c_uint32),
        ("stage_flags", ctypes.c_uint32),
    ]


class AcSpecConstant(ctypes.Structure):
    _fields_ = [
        ("constant_id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("value_bits", ctypes.c_uint32),
    ]


class AcSpecConstantSchemaEntry(ctypes.Structure):
    _fields_ = [
        ("constant_id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("default_value_bits", ctypes.c_uint32),
        ("min_value_bits", ctypes.c_uint32),
        ("max_value_bits", ctypes.c_uint32),
        ("overrides_local_size_axis", ctypes.c_uint32),
    ]


class AcKernelRuntimeMetadata(ctypes.Structure):
    _fields_ = [
        ("kernel_name", ctypes.c_char_p),
        ("entry_point", ctypes.c_char_p),
        ("binding_count", ctypes.c_uint32),
        ("bindings", ctypes.POINTER(AcRuntimeBindingDesc)),
        ("push_constant_size", ctypes.c_size_t),
        ("local_size", AcDim3),
        ("required_features", ctypes.c_uint32),
        ("expected_spirv_sha256", ctypes.c_char_p),
        ("specialization_schema_count", ctypes.c_uint32),
        ("specialization_schemas", ctypes.POINTER(AcSpecConstantSchemaEntry)),
    ]


class AcKernelDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("spirv_code", ctypes.POINTER(ctypes.c_uint32)),
        ("code_size_bytes", ctypes.c_size_t),
        ("metadata", ctypes.POINTER(AcKernelRuntimeMetadata)),
        ("specialization_count", ctypes.c_uint32),
        ("specializations", ctypes.POINTER(AcSpecConstant)),
    ]


class AcLaunchDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("kernel", ctypes.c_void_p),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("buffer_count", ctypes.c_uint32),
        ("push_constants", ctypes.c_void_p),
        ("push_constants_size", ctypes.c_size_t),
        ("grid_dim", AcDim3),
    ]


# =============================================================================
# Native Library Loader (Fail-Fast, Zero-Silent-Fallback)
# =============================================================================

def _resolve_library_path() -> str:
    """
    Resolve the native shared library path.
    Search order:
      1. AMEVA_COMPUTE_LIB_PATH environment variable (explicit override)
      2. Adjacent to this Python file (wheel-bundled)
      3. Build directory relative to package root
      4. Termux system paths
      5. System library paths (ldconfig / PATH)
    """
    env_path = os.environ.get("AMEVA_COMPUTE_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    is_windows = platform.system() == "Windows"
    lib_name = "ac_runtime.dll" if is_windows else "libac_runtime.so"

    search_dirs = [
        Path(__file__).parent,
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "Release",
        Path(__file__).parent.parent.parent / "build" / "Debug",
    ]

    if not is_windows:
        search_dirs.extend([
            Path("/data/data/com.termux/files/usr/lib"),
            Path("/usr/local/lib"),
            Path("/usr/lib"),
        ])

    searched = []
    for d in search_dirs:
        candidate = d / lib_name
        searched.append(str(candidate))
        if candidate.is_file():
            return str(candidate)

    # Try system-wide search as last resort
    found = ctypes.util.find_library("ac_runtime")
    if found:
        return found

    raise AmevaDriverNotFoundError(searched)


def _load_library() -> ctypes.CDLL:
    """Load the native library and bind all 31 C ABI function signatures."""
    lib_path = _resolve_library_path()
    lib = ctypes.CDLL(lib_path)

    # --- Context Management (3 functions) ---
    lib.ac_runtime_context_create.argtypes = [
        ctypes.POINTER(AcContextDesc),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_context_create.restype = ctypes.c_int32

    lib.ac_runtime_context_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_context_destroy.restype = ctypes.c_int32

    lib.ac_runtime_context_get_device_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ac_runtime_context_get_device_name.restype = ctypes.c_int32

    # --- Buffer Management (8 functions) ---
    lib.ac_runtime_buffer_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AcBufferDesc),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_buffer_create.restype = ctypes.c_int32

    lib.ac_runtime_buffer_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_buffer_destroy.restype = ctypes.c_int32

    lib.ac_runtime_buffer_get_mapped_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_buffer_get_mapped_ptr.restype = ctypes.c_int32

    lib.ac_runtime_buffer_get_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ac_runtime_buffer_get_size.restype = ctypes.c_int32

    lib.ac_runtime_buffer_get_allocation_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ac_runtime_buffer_get_allocation_size.restype = ctypes.c_int32

    lib.ac_runtime_buffer_is_coherent.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.ac_runtime_buffer_is_coherent.restype = ctypes.c_int32

    lib.ac_runtime_buffer_flush.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.ac_runtime_buffer_flush.restype = ctypes.c_int32

    lib.ac_runtime_buffer_invalidate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.ac_runtime_buffer_invalidate.restype = ctypes.c_int32

    # --- Kernel Management (5 functions) ---
    lib.ac_runtime_kernel_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AcKernelDesc),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_kernel_create.restype = ctypes.c_int32

    lib.ac_runtime_kernel_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_kernel_destroy.restype = ctypes.c_int32

    lib.ac_runtime_kernel_get_pipeline_identity.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ac_runtime_kernel_get_pipeline_identity.restype = ctypes.c_int32

    lib.ac_runtime_kernel_get_effective_local_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AcDim3),
    ]
    lib.ac_runtime_kernel_get_effective_local_size.restype = ctypes.c_int32

    lib.ac_runtime_kernel_get_push_constant_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ac_runtime_kernel_get_push_constant_size.restype = ctypes.c_int32

    # --- Execution & Synchronization (4 functions) ---
    lib.ac_runtime_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AcLaunchDesc),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_dispatch.restype = ctypes.c_int32

    lib.ac_runtime_event_wait.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    lib.ac_runtime_event_wait.restype = ctypes.c_int32

    lib.ac_runtime_event_get_timing.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ac_runtime_event_get_timing.restype = ctypes.c_int32

    lib.ac_runtime_event_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_event_destroy.restype = ctypes.c_int32

    # --- Command Batching (5 functions) ---
    lib.ac_runtime_batch_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_batch_begin.restype = ctypes.c_int32

    lib.ac_runtime_batch_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(AcLaunchDesc),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_batch_dispatch.restype = ctypes.c_int32

    lib.ac_runtime_batch_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ac_runtime_batch_submit.restype = ctypes.c_int32

    lib.ac_runtime_batch_abort.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_batch_abort.restype = ctypes.c_int32

    lib.ac_runtime_batch_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_runtime_batch_destroy.restype = None

    # --- Error State Management (6 functions) ---
    lib.ac_runtime_set_error.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.ac_runtime_set_error.restype = None

    lib.ac_runtime_get_error_state.argtypes = []
    lib.ac_runtime_get_error_state.restype = ctypes.c_int32

    lib.ac_runtime_consume_error.argtypes = []
    lib.ac_runtime_consume_error.restype = ctypes.c_int32

    lib.ac_runtime_advance_generation.argtypes = []
    lib.ac_runtime_advance_generation.restype = ctypes.c_uint64

    lib.ac_runtime_get_generation.argtypes = []
    lib.ac_runtime_get_generation.restype = ctypes.c_uint64

    lib.ac_runtime_reset_error_state.argtypes = []
    lib.ac_runtime_reset_error_state.restype = None

    return lib


# Deferred library loading — fail at first use, not at import
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    """Get or lazily load the native library. Raises on failure (Fail-Fast)."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def _check(result: int, operation: str = "") -> None:
    """Check AcResult return code and raise on any non-zero value."""
    if result != AC_SUCCESS:
        raise AmevaComputeError(result, operation)


# =============================================================================
# NumPy dtype ↔ element size mapping
# =============================================================================

_DTYPE_SIZES = {
    np.dtype("float32"): 4,
    np.dtype("float16"): 2,
    np.dtype("int32"): 4,
    np.dtype("int16"): 2,
    np.dtype("int8"): 1,
    np.dtype("uint8"): 1,
    np.dtype("uint32"): 4,
    np.dtype("uint16"): 2,
}


# =============================================================================
# Public API Classes (RAII Resource Management)
# =============================================================================

class Event:
    """
    GPU execution event handle with timing query support.
    Wraps an opaque AcEvent* handle with automatic cleanup.
    """
    __slots__ = ("_handle", "_ctx_handle", "_destroyed")

    def __init__(self, handle: ctypes.c_void_p, ctx_handle: ctypes.c_void_p) -> None:
        self._handle = handle
        self._ctx_handle = ctx_handle
        self._destroyed = False

    def wait(self, timeout_ns: int = AC_DEFAULT_TIMEOUT_NS) -> None:
        """Block until GPU execution completes."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Event already destroyed")
        _check(_get_lib().ac_runtime_event_wait(self._handle, ctypes.c_uint64(timeout_ns)),
               "event_wait")

    @property
    def timing(self) -> Tuple[int, int]:
        """Returns (wall_latency_ns, gpu_duration_ns) tuple."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Event already destroyed")
        wall_ns = ctypes.c_uint64(0)
        gpu_ns = ctypes.c_uint64(0)
        _check(_get_lib().ac_runtime_event_get_timing(
            self._handle, ctypes.byref(wall_ns), ctypes.byref(gpu_ns)),
            "event_get_timing")
        return (wall_ns.value, gpu_ns.value)

    @property
    def wall_latency_us(self) -> float:
        """Wall-clock latency in microseconds."""
        wall_ns, _ = self.timing
        return wall_ns / 1000.0

    @property
    def gpu_duration_us(self) -> float:
        """Pure GPU execution duration in microseconds."""
        _, gpu_ns = self.timing
        return gpu_ns / 1000.0

    def destroy(self) -> None:
        """Explicitly destroy the event handle."""
        if not self._destroyed and self._handle:
            _check(_get_lib().ac_runtime_event_destroy(self._handle), "event_destroy")
            self._destroyed = True

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception as e:
            _report_del_failure(self, e)

    def __repr__(self) -> str:
        status = "destroyed" if self._destroyed else "active"
        return f"<Event handle=0x{self._handle or 0:x} status={status}>"


class Buffer:
    """
    GPU buffer with UMA zero-copy NumPy interface.
    Wraps an opaque AcBuffer* handle with automatic cleanup.
    """
    __slots__ = ("_handle", "_ctx_handle", "_size_bytes", "_shape",
                 "_dtype", "_destroyed")

    def __init__(
        self,
        handle: ctypes.c_void_p,
        ctx_handle: ctypes.c_void_p,
        size_bytes: int,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._handle = handle
        self._ctx_handle = ctx_handle
        self._size_bytes = size_bytes
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._destroyed = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def is_coherent(self) -> bool:
        """Query whether this buffer uses host-coherent memory."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Buffer already destroyed")
        out = ctypes.c_uint32(0)
        _check(_get_lib().ac_runtime_buffer_is_coherent(
            self._handle, ctypes.byref(out)), "buffer_is_coherent")
        return out.value != 0

    def numpy(self) -> np.ndarray:
        """
        Zero-copy view of GPU buffer as NumPy array.
        Returns an ndarray backed by the UMA-mapped device memory.
        No data copy occurs — this is the actual GPU memory.
        """
        if self._destroyed:
            raise AmevaComputeError(-106, "Buffer already destroyed")
        ptr = ctypes.c_void_p(0)
        _check(_get_lib().ac_runtime_buffer_get_mapped_ptr(
            self._handle, ctypes.byref(ptr)), "buffer_get_mapped_ptr")
        if not ptr.value:
            raise AmevaComputeError(-204, "Buffer mapped pointer is NULL")

        element_count = 1
        for dim in self._shape:
            element_count *= dim

        # Cast raw pointer to typed ctypes array, then wrap as NumPy
        # FP16 and other non-standard dtypes are not supported by ctypes directly,
        # so we fall back to reading raw bytes and reinterpreting via np.frombuffer.
        try:
            c_type = np.ctypeslib.as_ctypes_type(self._dtype)
            arr_type = c_type * element_count
            arr = arr_type.from_address(ptr.value)
            return np.ctypeslib.as_array(arr).reshape(self._shape)
        except NotImplementedError:
            # FP16, BF16 etc: read as raw bytes, then view-cast
            byte_count = element_count * self._dtype.itemsize
            byte_arr = (ctypes.c_uint8 * byte_count).from_address(ptr.value)
            raw = np.ctypeslib.as_array(byte_arr)
            return raw.view(self._dtype).reshape(self._shape)

    def flush(self, offset: int = 0, size: int = 0) -> None:
        """Explicitly flush host writes to device (for non-coherent memory)."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Buffer already destroyed")
        if size == 0:
            size = self._size_bytes
        _check(_get_lib().ac_runtime_buffer_flush(
            self._handle, ctypes.c_size_t(offset), ctypes.c_size_t(size)),
            "buffer_flush")

    def invalidate(self, offset: int = 0, size: int = 0) -> None:
        """Explicitly invalidate cache for device writes (for non-coherent memory)."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Buffer already destroyed")
        if size == 0:
            size = self._size_bytes
        _check(_get_lib().ac_runtime_buffer_invalidate(
            self._handle, ctypes.c_size_t(offset), ctypes.c_size_t(size)),
            "buffer_invalidate")

    def destroy(self) -> None:
        """Explicitly destroy the buffer and release device memory."""
        if not self._destroyed and self._handle:
            _check(_get_lib().ac_runtime_buffer_destroy(self._handle),
                   "buffer_destroy")
            self._destroyed = True

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception as e:
            _report_del_failure(self, e)

    def __repr__(self) -> str:
        status = "destroyed" if self._destroyed else "active"
        return (f"<Buffer shape={self._shape} dtype={self._dtype} "
                f"size={self._size_bytes}B status={status}>")


class Kernel:
    """
    Compiled SPIR-V compute kernel handle.
    Wraps an opaque AcKernel* handle with automatic cleanup.
    """
    __slots__ = ("_handle", "_ctx_handle", "_name", "_destroyed")

    def __init__(
        self,
        handle: ctypes.c_void_p,
        ctx_handle: ctypes.c_void_p,
        name: str,
    ) -> None:
        self._handle = handle
        self._ctx_handle = ctx_handle
        self._name = name
        self._destroyed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def pipeline_identity(self) -> str:
        """Retrieve deterministic SHA-256 pipeline identity hex string."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Kernel already destroyed")
        buf = ctypes.create_string_buffer(65)
        _check(_get_lib().ac_runtime_kernel_get_pipeline_identity(
            self._handle, buf, ctypes.c_size_t(65)),
            "kernel_get_pipeline_identity")
        return buf.value.decode("ascii")

    @property
    def effective_local_size(self) -> Tuple[int, int, int]:
        """Retrieve effective workgroup local size after specialization."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Kernel already destroyed")
        dim = AcDim3()
        _check(_get_lib().ac_runtime_kernel_get_effective_local_size(
            self._handle, ctypes.byref(dim)),
            "kernel_get_effective_local_size")
        return (dim.x, dim.y, dim.z)

    @property
    def push_constant_size(self) -> int:
        """Retrieve push constant size in bytes."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Kernel already destroyed")
        out = ctypes.c_size_t(0)
        _check(_get_lib().ac_runtime_kernel_get_push_constant_size(
            self._handle, ctypes.byref(out)),
            "kernel_get_push_constant_size")
        return out.value

    def destroy(self) -> None:
        """Explicitly destroy the kernel handle."""
        if not self._destroyed and self._handle:
            _check(_get_lib().ac_runtime_kernel_destroy(self._handle),
                   "kernel_destroy")
            self._destroyed = True

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception as e:
            _report_del_failure(self, e)

    def __repr__(self) -> str:
        status = "destroyed" if self._destroyed else "active"
        return f"<Kernel name='{self._name}' status={status}>"


class Context:
    """
    AMUDA GPU compute context.
    Manages the lifecycle of all child resources (buffers, kernels, events).
    Supports RAII pattern via context manager (with statement).
    """
    __slots__ = ("_handle", "_destroyed")

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._handle = handle
        self._destroyed = False

    @property
    def device_name(self) -> str:
        """Retrieve physical GPU device name string."""
        if self._destroyed:
            raise AmevaComputeError(-106, "Context already destroyed")
        buf = ctypes.create_string_buffer(256)
        _check(_get_lib().ac_runtime_context_get_device_name(
            self._handle, buf, ctypes.c_size_t(256)),
            "context_get_device_name")
        return buf.value.decode("utf-8", errors="replace")

    def buffer(
        self,
        *,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype] = "float32",
        data: Optional[np.ndarray] = None,
        usage: int = AC_BUFFER_USAGE_STORAGE,
        force_non_coherent: bool = False,
    ) -> Buffer:
        """
        Allocate a GPU buffer.

        Args:
            shape: Tensor dimensions.
            dtype: NumPy-compatible dtype string or object.
            data: Optional initial data (copied via UMA zero-copy memcpy).
            usage: Buffer usage flags (default: STORAGE).
            force_non_coherent: Force non-coherent memory allocation.

        Returns:
            Buffer handle with zero-copy NumPy interface.
        """
        if self._destroyed:
            raise AmevaComputeError(-106, "Context already destroyed")

        dt = np.dtype(dtype)
        element_count = 1
        for dim in shape:
            element_count *= dim
        size_bytes = element_count * dt.itemsize

        if size_bytes == 0:
            raise AmevaComputeError(-206, "Buffer size must be > 0")

        desc = AcBufferDesc(
            abi_version=AC_ABI_VERSION,
            struct_size=ctypes.sizeof(AcBufferDesc),
            size_bytes=size_bytes,
            usage=usage,
            force_non_coherent=1 if force_non_coherent else 0,
        )

        buf_ptr = ctypes.c_void_p(0)
        _check(_get_lib().ac_runtime_buffer_create(
            self._handle, ctypes.byref(desc), ctypes.byref(buf_ptr)),
            "buffer_create")

        buf = Buffer(buf_ptr.value, self._handle, size_bytes, shape, dt)

        # If initial data is provided, copy via UMA zero-copy memcpy
        if data is not None:
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            mapped = buf.numpy()
            np.copyto(mapped, data.reshape(shape).astype(dt))
            if not buf.is_coherent:
                buf.flush()

        return buf

    def kernel(
        self,
        *,
        spirv_code: bytes,
        name: str = "compute",
        entry_point: str = "main",
        binding_count: int = 0,
        bindings: Optional[List[Tuple[int, int]]] = None,
        local_size: Tuple[int, int, int] = (256, 1, 1),
        push_constant_size: int = 0,
        specializations: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Kernel:
        """
        Create a compute kernel from SPIR-V bytecode.

        Args:
            spirv_code: Raw SPIR-V bytecode (bytes object, must be 4-byte aligned).
            name: Kernel name for identification.
            entry_point: Shader entry point name (default: "main").
            binding_count: Number of bindings (auto-detected from bindings list if provided).
            bindings: List of (binding_index, binding_type) tuples.
            local_size: Workgroup local size (x, y, z).
            push_constant_size: Push constant payload size in bytes.
            specializations: List of (constant_id, type, value_bits) tuples.

        Returns:
            Kernel handle.
        """
        if self._destroyed:
            raise AmevaComputeError(-106, "Context already destroyed")

        if len(spirv_code) % 4 != 0:
            raise AmevaComputeError(-304, "SPIR-V bytecode size must be a multiple of 4 bytes")

        # Prepare SPIR-V code array
        word_count = len(spirv_code) // 4
        spirv_array = (ctypes.c_uint32 * word_count).from_buffer_copy(spirv_code)

        # Prepare bindings
        if bindings:
            binding_count = len(bindings)
            binding_arr = (AcRuntimeBindingDesc * binding_count)()
            for i, (idx, btype) in enumerate(bindings):
                binding_arr[i].binding_index = idx
                binding_arr[i].binding_type = btype
                binding_arr[i].stage_flags = AC_STAGE_COMPUTE
            binding_ptr = ctypes.cast(binding_arr, ctypes.POINTER(AcRuntimeBindingDesc))
        else:
            binding_ptr = None

        # Compute SPIR-V SHA-256 (required by C runtime validation)
        import hashlib
        spirv_sha256 = hashlib.sha256(spirv_code).hexdigest().encode("utf-8")

        # Prepare metadata
        metadata = AcKernelRuntimeMetadata(
            kernel_name=name.encode("utf-8"),
            entry_point=entry_point.encode("utf-8"),
            binding_count=binding_count,
            bindings=binding_ptr,
            push_constant_size=push_constant_size,
            local_size=AcDim3(local_size[0], local_size[1], local_size[2]),
            required_features=0,
            expected_spirv_sha256=spirv_sha256,
            specialization_schema_count=0,
            specialization_schemas=None,
        )

        # Prepare specialization constants
        spec_count = 0
        spec_ptr = None
        if specializations:
            spec_count = len(specializations)
            spec_arr = (AcSpecConstant * spec_count)()
            for i, (cid, ctype, cval) in enumerate(specializations):
                spec_arr[i].constant_id = cid
                spec_arr[i].type = ctype
                spec_arr[i].value_bits = cval
            spec_ptr = ctypes.cast(spec_arr, ctypes.POINTER(AcSpecConstant))

        desc = AcKernelDesc(
            abi_version=AC_ABI_VERSION,
            struct_size=ctypes.sizeof(AcKernelDesc),
            spirv_code=spirv_array,
            code_size_bytes=len(spirv_code),
            metadata=ctypes.pointer(metadata),
            specialization_count=spec_count,
            specializations=spec_ptr,
        )

        kernel_ptr = ctypes.c_void_p(0)
        _check(_get_lib().ac_runtime_kernel_create(
            self._handle, ctypes.byref(desc), ctypes.byref(kernel_ptr)),
            "kernel_create")

        return Kernel(kernel_ptr.value, self._handle, name)

    def dispatch(
        self,
        kernel: Kernel,
        *,
        buffers: List[Buffer],
        grid: Tuple[int, ...] = (1,),
        push_constants: Optional[bytes] = None,
        synchronous: bool = False,
    ) -> Optional[Event]:
        """
        Dispatch a compute kernel over a 3D grid.

        Args:
            kernel: Compiled kernel handle.
            buffers: List of bound buffer handles.
            grid: Workgroup grid dimensions (1D, 2D, or 3D).
            push_constants: Optional push constant payload bytes.
            synchronous: If True, blocks until GPU execution completes.

        Returns:
            Event handle for async tracking, or None if synchronous.
        """
        if self._destroyed:
            raise AmevaComputeError(-106, "Context already destroyed")

        # Normalize grid to 3D
        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1

        # Build buffer handle array
        buf_arr = (ctypes.c_void_p * len(buffers))()
        for i, b in enumerate(buffers):
            buf_arr[i] = b._handle

        # Push constants
        pc_ptr = None
        pc_size = 0
        if push_constants:
            pc_size = len(push_constants)
            pc_buf = ctypes.create_string_buffer(push_constants)
            pc_ptr = ctypes.cast(pc_buf, ctypes.c_void_p)

        desc = AcLaunchDesc(
            abi_version=AC_ABI_VERSION,
            struct_size=ctypes.sizeof(AcLaunchDesc),
            kernel=kernel._handle,
            buffers=buf_arr,
            buffer_count=len(buffers),
            push_constants=pc_ptr,
            push_constants_size=pc_size,
            grid_dim=AcDim3(gx, gy, gz),
        )

        if synchronous:
            _check(_get_lib().ac_runtime_dispatch(
                self._handle, ctypes.byref(desc), None),
                "dispatch_sync")
            return None
        else:
            event_ptr = ctypes.c_void_p(0)
            _check(_get_lib().ac_runtime_dispatch(
                self._handle, ctypes.byref(desc), ctypes.byref(event_ptr)),
                "dispatch_async")
            return Event(event_ptr.value, self._handle)

    def destroy(self) -> None:
        """Explicitly destroy the context and all owned resources."""
        if not self._destroyed and self._handle:
            _check(_get_lib().ac_runtime_context_destroy(self._handle),
                   "context_destroy")
            self._destroyed = True

    def __enter__(self) -> Context:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception as e:
            _report_del_failure(self, e)

    def __repr__(self) -> str:
        status = "destroyed" if self._destroyed else "active"
        return f"<Context status={status}>"


# =============================================================================
# Module-Level Initialization (Fail-Fast Entry Point)
# =============================================================================

def init(*, backend: int = AC_BACKEND_VULKAN) -> Context:
    """
    Initialize an AMUDA hardware compute context.

    Args:
        backend: Compute backend kind (default: AC_BACKEND_VULKAN).

    Returns:
        Context handle with GPU resource management.

    Raises:
        AmevaDriverNotFoundError: If the native library is not found.
        AmevaComputeError: If context creation fails for any reason.
    """
    desc = AcContextDesc(
        abi_version=AC_ABI_VERSION,
        struct_size=ctypes.sizeof(AcContextDesc),
        backend_kind=backend,
        flags=0,
    )

    ctx_ptr = ctypes.c_void_p(0)
    _check(_get_lib().ac_runtime_context_create(
        ctypes.byref(desc), ctypes.byref(ctx_ptr)),
        "context_create")

    if not ctx_ptr.value:
        raise AmevaComputeError(-105, "Context creation returned NULL handle")

    return Context(ctx_ptr.value)
