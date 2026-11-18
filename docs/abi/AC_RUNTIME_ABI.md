# ac_runtime C ABI

`libac_runtime.so.2` is the GPU runtime under every AMUDA engine: the ggml backend `libggml-ameva.so`, the BitNet,
STT, TTS, vision and diffusion engines, the Android SDK and the Python binding (`ameva_compute`). This document is the
contract between that library and its callers. The build enforces it (`scripts/termux/build_runtime_backend.sh`,
`tests/ac_abi_check.c`).

## Version

`AC_ABI_VERSION` in `include/ac_base.h` is `0x00MMNN00`: major `MM`, minor `NN`, low byte zero. The current version
is **2.3.0** (`0x00020300`). The major version is also the SONAME (`libac_runtime.so.2`).

A caller writes the `AC_ABI_VERSION` of the header it was built with into the `abi_version` field of every descriptor.
The runtime accepts the descriptor when `AC_ABI_COMPATIBLE(abi_version)` holds:

| Caller version | Runtime 2.3 | Why |
| :--- | :--- | :--- |
| 2.1, 2.2, 2.3 | accepted | a minor version only adds; an older caller keeps working with a newer runtime |
| 2.4 and later | refused | the caller may rely on something this runtime does not have |
| 2.0 | refused | below `AC_ABI_MIN_MINOR` (descriptor layouts changed in 2.1) |
| 1.x, 3.x | refused | another major version |
| low byte not zero | refused | not a version this scheme produces |

Refusal is `AC_ERR_INVALID_ARGUMENT` with every output pointer set to NULL; nothing falls back silently.
`ac_runtime_abi_version()` returns the version of the loaded library, so a caller that loads it at run time can check
before it creates a context. Functions introduced in a minor version accept only callers of that minor or later
(`ac_runtime_kernel_create_from_source` requires 2.3).

## What a version change means

| Change | Version |
| :--- | :--- |
| new function, new flag bit, new enum value, new error code | minor + 1 |
| field appended to a struct whose documentation says appended fields are honoured by `struct_size` (today only `AcDeviceLimits`) | minor + 1 |
| any other struct layout change, a removed or renamed function, a changed meaning of an existing value | major + 1 (new SONAME) |
| bug fix with no change to declarations | none |

Every other descriptor is checked with `struct_size == sizeof(Descriptor)`; a layout change there is a major change.

## What the build checks

`build_runtime_backend.sh` builds `tests/ac_abi_check.c` against the new library and stops the build when:

1. `ac_runtime_abi_version()` differs from the header, or the accept/refuse table above does not hold on a real
   context (`AC_ABI_CHECK=PASS` is required);
2. the size or a recorded field offset of a public struct differs from `docs/abi/ac_runtime_layout.txt`;
3. the library stops exporting a function listed in `docs/abi/ac_runtime_symbols.txt` (a removal), or exports an
   `ac_runtime_*` function the list does not contain (an undocumented addition).

So an ABI change is made by editing the header, `AC_ABI_VERSION`, and these two files in the same commit. The result
is recorded in `backend-manifest.json` (`ac_abi_version`, `tests.ac_abi_check`).

The layout file is the arm64 (LP64) layout, the only target the runtime is built for.

## Distribution

`libggml-ameva.so` and `libac_runtime.so.2` ship together in one bundle; the bundle manifest pins the SHA-256 of
both. A plugin finds the runtime beside itself (`RUNPATH` starts with `$ORIGIN`). Mixing a plugin and a runtime from
different bundles is not supported even when the versions are compatible: the compatibility rule protects callers
that use the C API directly (Android SDK, Python binding, third-party engines).

## History

| Version | Contents |
| :--- | :--- |
| 2.0 | first descriptor-based API |
| 2.1 | specialization constants in kernel metadata and descriptors; exact `struct_size` checks |
| (2.2) | never set in the header: batches (`ac_runtime_batch_*`), events with timestamps and device limits with appended fields were added while `AC_ABI_VERSION` stayed 2.1, and shipped that way in ameva-runtime 2.8.x. 2.3 accepts 2.1 callers, so those builds keep working |
| 2.3 | `AC_BACKEND_OPENCL`, `ac_runtime_kernel_create_from_source` (`AcKernelSourceDesc`, `AC_KERNEL_SOURCE_BINDING_RANGES`), backend-neutral host access (`ac_runtime_buffer_write/read/fill/get_address`), `ac_runtime_context_get_backend_kind`, `ac_runtime_abi_version`; version check relaxed from exact match to `AC_ABI_COMPATIBLE` |
