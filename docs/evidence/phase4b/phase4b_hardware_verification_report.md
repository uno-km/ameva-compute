# AMUDA 2.x Heterogeneous Hardware Verification Evidence Report

**Date**: 2026-09-10
**Target Repository**: `dev/ameva-compute` (Public C ABI 2.1.0, Encoded `0x00020100`)
**Working Tree Status**: Modified, unstaged | Index: No staged changes
**Verification Verdict**: **30 distinct verification gates executed on 2 physical devices (Total 60 device-gate executions PASSED)**

### Hardware Target Matrix
- **Galaxy S25**: Qualcomm Snapdragon 8 Elite / Adreno (TM) 830 (Driver: `0x80320040`, Vulkan 1.3.284, Android API 30)
- **Galaxy A35**: Samsung Exynos 1380 / ARM Mali-G68 (Driver: `0x09801000`, Vulkan 1.3.219, Android API 24)

### Strict Execution & Safety Parameters
- **Isolation Harness**: `timeout 120s` enforced per test binary
- **Android Linker Guard**: `unset LD_LIBRARY_PATH` enforced to prevent Bionic/Termux symbol conflicts
- **Build Isolation**: Clean re-build directory `build-audit` with fresh CMake configure (`-DCMAKE_BUILD_TYPE=Release`)
- **Numerical Fidelity**: Exact match (0 mismatches across non-power-of-2 boundaries and IEEE-754 specials)
- **Memory Stability Contract**: 1,000 continuous pipeline cycles; tolerance `|delta| <= 128 KB` (No monotonic RSS growth beyond tolerance observed)

### Safety Budget Policy & Release Rationale
- **Constant**: `AC_MAX_SAFE_GROUPS_PER_AXIS = 1,048,576u` (Uniformly enforced on X, Y, and Z axes)
- **Constant**: `AC_MAX_SAFE_TOTAL_GROUPS = 16,777,216ull` (Total 3D product limit with safe 64-bit overflow guard)
- **Behavioral Release Note**: Oversized dispatches that were previously device-valid on architectures reporting `UINT32_MAX` (e.g. ARM Mali) are now rejected defensively by AMEVA runtime dispatch-budget policy before physical queue submission to prevent kernel TDR lockups and driver stalls.

---

## Index of Bundled Evidence Artifacts

- [1. Toolchain & Build Fingerprints](#1-toolchain-&-build-fingerprints) (`build_environment.txt`)
- [2. Source Code Manifest (SHA-256)](#2-source-code-manifest-sha-256) (`source-manifest.sha256`)
- [3. Clean-Build Binary Manifest (SHA-256)](#3-clean-build-binary-manifest-sha-256) (`binary-manifest.sha256`)
- [4. Galaxy S25 Official Vulkan Device Limits](#4-galaxy-s25-official-vulkan-device-limits) (`s25_limits_official.txt`)
- [5. Galaxy A35 Official Vulkan Device Limits](#5-galaxy-a35-official-vulkan-device-limits) (`a35_limits_official.txt`)
- [6. Galaxy S25 Phase 4B Specialization Raw Log (build-audit)](#6-galaxy-s25-phase-4b-specialization-raw-log-build-audit) (`s25_phase4b_raw.log`)
- [7. Galaxy S25 Phase 4A Runtime ABI Raw Log (build-audit)](#7-galaxy-s25-phase-4a-runtime-abi-raw-log-build-audit) (`s25_phase4a_raw.log`)
- [8. Galaxy S25 Phase 3 Universal Backend Raw Log (build-audit)](#8-galaxy-s25-phase-3-universal-backend-raw-log-build-audit) (`s25_phase3_raw.log`)
- [9. Galaxy A35 Phase 4B Specialization Raw Log (build-audit)](#9-galaxy-a35-phase-4b-specialization-raw-log-build-audit) (`a35_phase4b_raw.log`)
- [10. Galaxy A35 Phase 4A Runtime ABI Raw Log (build-audit)](#10-galaxy-a35-phase-4a-runtime-abi-raw-log-build-audit) (`a35_phase4a_raw.log`)
- [11. Galaxy A35 Phase 3 Universal Backend Raw Log (build-audit)](#11-galaxy-a35-phase-3-universal-backend-raw-log-build-audit) (`a35_phase3_raw.log`)

---

## 1. Toolchain & Build Fingerprints

> **Source File**: `build_environment.txt`

```text
=== AMUDA 2.x Clean Build Toolchain Fingerprints ===

--- Galaxy S25 ---
clang version 21.1.8
Target: aarch64-unknown-linux-android30
Thread model: posix
InstalledDir: /data/data/com.termux/files/usr/bin
cmake version 4.3.4

CMake suite maintained and supported by Kitware (kitware.com/cmake).
GNU Make 4.4.1
LLD 21.1.8 (compatible with GNU linkers)

--- Galaxy A35 ---
clang version 21.1.8
Target: aarch64-unknown-linux-android24
Thread model: posix
InstalledDir: /data/data/com.termux/files/usr/bin
cmake version 4.4.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
GNU Make 4.4.1
LLD 21.1.8 (compatible with GNU linkers)
```

---

## 2. Source Code Manifest (SHA-256)

> **Source File**: `source-manifest.sha256`

```text
0586fe9302df2fc096b5d07b23fb3528436554badb5e421a98b722831eb45f3e  include/ac_dispatch.h
3934c2d980016a59b4868f003d4d8134f597fe8d83d3712dbf62cc4801844d43  src/backend/vulkan/ac_vk_dispatch.cpp
ec7406fb5ab59f6db45310eae74a04f98edb2e5c0d8860a6fb23b1b3354f0a5a  src/runtime/ac_runtime.cpp
812c1a898b858e9a3e06820e452bf508701eab92ef5eccbca23ce4d38d3a12e0  tests/test_phase4b_specialization.cpp
4e6c46f85af6c92ee9cee392b2442a7bc8069e03caadf11fc574050de9d97b42  src/shaders/vector_add_spec.spv
```

---

## 3. Clean-Build Binary Manifest (SHA-256)

> **Source File**: `binary-manifest.sha256`
> **Binary Hash Independence Note**:
> The binaries were built independently for different Android API targets (S25: `android30`, A35: `android24`) and with different CMake versions (S25: `4.3.4`, A35: `4.4.3`). Binary hash equality across devices is therefore not required. The exact byte-level causes of the difference were not decomposed.
> (두 바이너리는 서로 다른 Android API 타깃과 CMake 버전에서 독립적으로 빌드되었으므로 동일 해시를 요구하지 않는다. 바이트 수준의 차이 원인은 별도로 분해하지 않았다.)

```text
# Galaxy S25 clean-build binaries
70ed04db7b91a8cbb8ae5ceff449af12fdc204aea0bbc90e79ce1ebc92cdd911  build-audit/test_phase4b_specialization
d614ebeec9a394f79d70a8f72a727fead4546f6d50b1d9a5e9bf44682c5302ce  build-audit/test_phase4_runtime_abi
0930796ae7bd14df5c5d4fe495eb4c1e64eb46b6e25fd9eaea528f39521204f7  build-audit/test_phase3_universal_backend

# Galaxy A35 clean-build binaries
c39ed68aa308616c0a904c58c8a6b768a3ece9c463ff0f70f032374dd38567d3  build-audit/test_phase4b_specialization
4bdca957c206f4d48c156dc6d43e66dacf889164c82fef3bc1159f4ac7cbc255  build-audit/test_phase4_runtime_abi
03d681fa44839e0096f31601faa4731d5c19fb11befdea0636738aee0d6f8339  build-audit/test_phase3_universal_backend
```

---

## 4. Galaxy S25 Official Vulkan Device Limits

> **Source File**: `s25_limits_official.txt`

```text
=== OFFICIAL VULKAN DEVICE LIMITS PROBE ===
PhysicalDeviceIndex: 0
deviceName: Adreno (TM) 830
vendorID: 0x5143
deviceID: 0x44050001
driverVersion: 0x80320040
apiVersion: 1.3.284 (0x0040311C)
maxComputeWorkGroupCount: [65535, 65535, 65535]
maxComputeWorkGroupSize: [1024, 1024, 1024]
maxComputeWorkGroupInvocations: 1024
=== PROBE COMPLETE ===
```

---

## 5. Galaxy A35 Official Vulkan Device Limits

> **Source File**: `a35_limits_official.txt`

```text
=== OFFICIAL VULKAN DEVICE LIMITS PROBE ===
PhysicalDeviceIndex: 0
deviceName: Mali-G68
vendorID: 0x13B5
deviceID: 0x92041010
driverVersion: 0x09801000
apiVersion: 1.3.219 (0x004030DB)
maxComputeWorkGroupCount: [4294967295, 4294967295, 4294967295]
maxComputeWorkGroupSize: [512, 512, 512]
maxComputeWorkGroupInvocations: 512
=== PROBE COMPLETE ===
```

---

## 6. Galaxy S25 Phase 4B Specialization Raw Log (build-audit)

> **Source File**: `s25_phase4b_raw.log`

```text
=== Galaxy S25 Phase 4B Specialization ===
Binary: build-audit/test_phase4b_specialization
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase4b_specialization
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA Phase 4B: Specialization Constants (Public C ABI 2.1.0)
======================================================================

[Gate 1/10] ABI Version Validation & SHA-256 FIPS KAT...
  -> ABI 0x00020100 validated, Legacy 0x00020000 rejected, FIPS SHA-256 KAT matched.
[Gate 2/10] Schema Validation Rejection Matrix...
  -> Schema rejection matrix verified (10 sub-conditions rejected with exact codes).
[Gate 3/10] Zero Specialization Execution (Schema Defaults)...
  -> Default execution verified: (64,1,1) grid, OP_MODE=0 addition, 0/1024 mismatches.
[Gate 4/10] Bit-Preserving Representation & Range Verification...
  -> Bit-preserving representation verified across U32, I32, and F32 IEEE-754.
[Gate 5/10] Caller Rejection Matrix...
  -> Caller rejection matrix verified (duplicate, undeclared, mismatch, NaN, zero size).
[Gate 6/10] Device Workgroup Limit & Dispatch Budget Enforcement...
  -> 6.1 Device size limit validation (oversized local size rejected): PASS
  -> 6.2 Device count limit validation (injected Adreno & Mali limit unit tests): PASS
  -> 6.3 Zero-axis validation (x=0, y=0, z=0 rejected): PASS
  -> 6.4 Runtime dispatch-budget validation (uniform 3-axis & total budget): PASS
  -> 6.5 Safe ceil-div overflow validation (division by zero and bounds): PASS
  -> 6.6 Runtime dispatch E2E integration rejection (submission counter unchanged, event not created): PASS
  -> Hardware workgroup limits and runtime safety budgets strictly enforced across heterogeneous targets.
[Gate 7/10] Local Size 32, 64, 128 Numerical Parity...
  -> Local sizes [32, 64, 128] verified with non-power-of-two N=1000 (0 mismatches across all).
[Gate 8/10] Domain-Separated LE Pipeline Identity & Order Invariance...
  -> Identity verified: Default equivalence, Order invariance, and Divergence proven.
     Canonical SHA-256: 162c9ff7671411683a3f360606407c8b0468dbea85defc42fa502ad84d8d1990
[Gate 9/10] Algorithmic Specialization (Fused Scale & Add)...
  -> Algorithmic specialization verified: c = a * 2.5 + b, 0/1000 mismatches (max_diff=0.000000e+00).
[Gate 10/10] 1,000-Cycle Pipeline Lifecycle RSS Stability...
  -> RSS: start=32096 KB, end=32096 KB, delta=0 KB across 1000 pipeline cycles.
  -> No monotonic RSS growth was observed within the defined 128 KB test tolerance.
  -> 1,000-cycle lifecycle memory stability verified.

======================================================================
AMUDA Phase 4B (Public C ABI 2.1.0) Result: 10/10 Gates PASSED
======================================================================

--- STDERR ---
```

---

## 7. Galaxy S25 Phase 4A Runtime ABI Raw Log (build-audit)

> **Source File**: `s25_phase4a_raw.log`

```text
=== Galaxy S25 Phase 4A Runtime ABI ===
Binary: build-audit/test_phase4_runtime_abi
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase4_runtime_abi
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA 2.0 Phase 4A: Public Opaque C Runtime ABI Test Suite
======================================================================

[Gate 1/10] Public ABI Contract & Strict Descriptor Validation...
  -> Verified: Bad ABI version, struct size, flags, and null checks strictly rejected with output NULL.
  -> Gate 1 PASS

[Gate 2/10] Context Lifecycle & Active Children Rejection...
  -> Active child buffer blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)
  -> Clean context destruction after child release: Verified (AC_SUCCESS)
  -> Gate 2 PASS

[Gate 3/10] Active Event Lifespan Tracking & Context Teardown Rejection (UAF Guard)...
  -> Waited but unretired public Event blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)
  -> Clean context destruction after event release: Verified (AC_SUCCESS)
  -> Gate 3 PASS

[Gate 4/10] Buffer Usage Flags Validation & Safe Output Getters...
  -> Buffer Getters: Logical=4096 B, Allocation=4096 B, is_coherent=1
  -> Buffer flags, zero checks, and getters verified.
  -> Gate 4 PASS

[Gate 5/10] Kernel Descriptor & Metadata Defensive Validation...
  -> Kernel metadata null checks, alignment, duplicate binding, and hash rejection verified.
  -> Gate 5 PASS

[Gate 6/10] Cross-Context Ownership Isolation & Launch Defensive Checks...
  -> Cross-context isolation and defensive launch checks verified.
  -> Gate 6 PASS

[Gate 7/10] BitNet GEMV I2_S Compute Dispatch via Public Opaque ABI...
  -> BitNet GEMV I2_S (256/256 weights): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 7 PASS

[Gate 8/10] Event Synchronization & Hardware Timing Query...
  -> Host Synchronous Wall Latency: 332552 ns (332.552 µs, 0.333 ms)
  -> GPU Timestamp Command Interval: 8385 ns (8.385 µs, 0.008 ms)
  -> Event Double-Wait Idempotency: Verified (Returned AC_SUCCESS)
  -> Gate 8 PASS

[Gate 9/10] Synchronous Dispatch Mode Contract (Vector-Add, out_event == NULL)...
  -> Synchronous Vector-Add Parity (N=256, alpha=2.5f): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 9 PASS

[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability via Public ABI...
  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS 35148 KB -> 35148 KB (Delta: 0 KB)
  -> Context Teardown: Completed cleanly via public ABI (Returned AC_SUCCESS)
  -> Gate 10 PASS

======================================================================
AMUDA 2.0 Phase 4A Verification Gates: 10 / 10 Passed
Public Opaque C ABI Runtime Status: VERIFIED PASS
======================================================================

--- STDERR ---
```

---

## 8. Galaxy S25 Phase 3 Universal Backend Raw Log (build-audit)

> **Source File**: `s25_phase3_raw.log`

```text
=== Galaxy S25 Phase 3 Universal Backend ===
Binary: build-audit/test_phase3_universal_backend
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase3_universal_backend
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA 2.0 Phase 3: Universal Backend & Cross-Device Hardware Test
======================================================================

[Gate 1/10] Hardware Diagnostic & Capabilities Inspection...
  -> Physical GPU Selected: Adreno (TM) 830 (Vendor: 0x5143, Device: 0x44050001)
  -> Driver Version: 0x80320040, API Version: 1.3.284
  -> Non-Coherent Atom Size: 1 bytes
  -> Timestamp Compute Support: TRUE, Valid Bits: 48, Period: 52.08 ns/tick
  -> Gate 1 PASS

[Gate 2/10] Universal AcKernelMetadata & Hash Authentication (BitNet GEMV I2_S)...
  -> Kernel 'bitnet_gemv_i2s' created with SHA-256: 6d6a55fe2beb5291333c36280b2326e34d7470f29635b9d5e271c2e17463504a
  -> BitNet GEMV I2_S Decoding (256/256 weights): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 2 PASS

[Gate 3/10] Universal Vector-Add Kernel (vector_add.spv, N=256, alpha=2.5f)...
  -> Vector Add (N=256): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 3 PASS

[Gate 4/10] Residual In-Place Vector Addition (x += alpha * y)...
  -> Residual In-Place (N=512): Mismatches: 0/512, Max Error: 0.00e+00
  -> Gate 4 PASS

[Gate 5/10] Arbitrary Non-Power-of-2 Sizes & Tail Boundaries...
  -> Tail N=37 (Groups=1): Mismatch=0, Canary Corrupted=0
  -> Tail N=255 (Groups=1): Mismatch=0, Canary Corrupted=0
  -> Tail N=1023 (Groups=4): Mismatch=0, Canary Corrupted=0
  -> Tail N=4099 (Groups=17): Mismatch=0, Canary Corrupted=0
  -> Gate 5 PASS

[Gate 6/10] IEEE-754 Special Floating-Point Values Preservation...
  -> [0] 5 + Inf = inf (isinf: 1)
  -> [1] 5 + (-Inf) = -inf (isinf: 1)
  -> [6] 1 + NaN = nan (isnan: 1)
  -> [7] 42 + 8 = 50.000000 (expected: 50.0)
  -> Gate 6 PASS

[Gate 7/10] Memory Management & Physical Non-Coherent Verification...
  -> Default Allocated Buffer Properties:
     Memory Type Index: 6 | Property Flags: 0x0006 | is_coherent: TRUE
     Logical Size: 4096 bytes | Physical Driver Allocation: 4096 bytes
  -> [SKIP] NON_COHERENT_NOT_AVAILABLE: Device does not offer non-coherent host-visible memory type.
  -> Gate 7 PASS

[Gate 8/10] Fine-Grained GPU Timestamp Command Interval vs Host Wall Latency...
  -> Avg Synchronous End-to-End Wall Latency: 342917 ns (342.917 µs, 0.343 ms)
  -> GPU timestamp command interval duration: 6979 ns (6.979 µs, 0.007 ms)
     (Measured between TOP_OF_PIPE and BOTTOM_OF_PIPE pipeline stages)
  -> GPU Command Interval Active Ratio: 2.04% of host wall time
  -> Gate 8 PASS

[Gate 9/10] Metadata Validation Failures (Corrupted Hash & Unsupported Features)...
  -> Corrupted Hash Rejection: Validated (Returned -304)
  -> Unsupported Feature Mask Rejection: Validated (Returned -108)
  -> Gate 9 PASS

[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability & Strict Teardown...
  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS 35184 KB -> 35184 KB (Delta: 0 KB)
  -> Internal Counters Check: active_child_buffers=0, active_child_kernels=0, submissions=0
  -> Context Teardown: Cleanup completed with internal counters 0 (Returned 0)
  -> Gate 10 PASS

======================================================================
AMUDA 2.0 Phase 3 Verification Gates: 10 / 10 Passed
Storage-Buffer Compute Pipeline Status: VERIFIED PASS
======================================================================

--- STDERR ---
```

---

## 9. Galaxy A35 Phase 4B Specialization Raw Log (build-audit)

> **Source File**: `a35_phase4b_raw.log`

```text
=== Galaxy A35 Phase 4B Specialization ===
Binary: build-audit/test_phase4b_specialization
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase4b_specialization
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA Phase 4B: Specialization Constants (Public C ABI 2.1.0)
======================================================================

[Gate 1/10] ABI Version Validation & SHA-256 FIPS KAT...
  -> ABI 0x00020100 validated, Legacy 0x00020000 rejected, FIPS SHA-256 KAT matched.
[Gate 2/10] Schema Validation Rejection Matrix...
  -> Schema rejection matrix verified (10 sub-conditions rejected with exact codes).
[Gate 3/10] Zero Specialization Execution (Schema Defaults)...
  -> Default execution verified: (64,1,1) grid, OP_MODE=0 addition, 0/1024 mismatches.
[Gate 4/10] Bit-Preserving Representation & Range Verification...
  -> Bit-preserving representation verified across U32, I32, and F32 IEEE-754.
[Gate 5/10] Caller Rejection Matrix...
  -> Caller rejection matrix verified (duplicate, undeclared, mismatch, NaN, zero size).
[Gate 6/10] Device Workgroup Limit & Dispatch Budget Enforcement...
  -> 6.1 Device size limit validation (oversized local size rejected): PASS
  -> 6.2 Device count limit validation (injected Adreno & Mali limit unit tests): PASS
  -> 6.3 Zero-axis validation (x=0, y=0, z=0 rejected): PASS
  -> 6.4 Runtime dispatch-budget validation (uniform 3-axis & total budget): PASS
  -> 6.5 Safe ceil-div overflow validation (division by zero and bounds): PASS
  -> 6.6 Runtime dispatch E2E integration rejection (submission counter unchanged, event not created): PASS
  -> Hardware workgroup limits and runtime safety budgets strictly enforced across heterogeneous targets.
[Gate 7/10] Local Size 32, 64, 128 Numerical Parity...
  -> Local sizes [32, 64, 128] verified with non-power-of-two N=1000 (0 mismatches across all).
[Gate 8/10] Domain-Separated LE Pipeline Identity & Order Invariance...
  -> Identity verified: Default equivalence, Order invariance, and Divergence proven.
     Canonical SHA-256: 162c9ff7671411683a3f360606407c8b0468dbea85defc42fa502ad84d8d1990
[Gate 9/10] Algorithmic Specialization (Fused Scale & Add)...
  -> Algorithmic specialization verified: c = a * 2.5 + b, 0/1000 mismatches (max_diff=0.000000e+00).
[Gate 10/10] 1,000-Cycle Pipeline Lifecycle RSS Stability...
  -> RSS: start=33848 KB, end=33840 KB, delta=-8 KB across 1000 pipeline cycles.
  -> No monotonic RSS growth was observed within the defined 128 KB test tolerance.
  -> 1,000-cycle lifecycle memory stability verified.

======================================================================
AMUDA Phase 4B (Public C ABI 2.1.0) Result: 10/10 Gates PASSED
======================================================================

--- STDERR ---
```

---

## 10. Galaxy A35 Phase 4A Runtime ABI Raw Log (build-audit)

> **Source File**: `a35_phase4a_raw.log`

```text
=== Galaxy A35 Phase 4A Runtime ABI ===
Binary: build-audit/test_phase4_runtime_abi
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase4_runtime_abi
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA 2.0 Phase 4A: Public Opaque C Runtime ABI Test Suite
======================================================================

[Gate 1/10] Public ABI Contract & Strict Descriptor Validation...
  -> Verified: Bad ABI version, struct size, flags, and null checks strictly rejected with output NULL.
  -> Gate 1 PASS

[Gate 2/10] Context Lifecycle & Active Children Rejection...
  -> Active child buffer blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)
  -> Clean context destruction after child release: Verified (AC_SUCCESS)
  -> Gate 2 PASS

[Gate 3/10] Active Event Lifespan Tracking & Context Teardown Rejection (UAF Guard)...
  -> Waited but unretired public Event blocks context teardown: Verified (AC_ERR_RESOURCE_IN_USE)
  -> Clean context destruction after event release: Verified (AC_SUCCESS)
  -> Gate 3 PASS

[Gate 4/10] Buffer Usage Flags Validation & Safe Output Getters...
  -> Buffer Getters: Logical=4096 B, Allocation=4096 B, is_coherent=1
  -> Buffer flags, zero checks, and getters verified.
  -> Gate 4 PASS

[Gate 5/10] Kernel Descriptor & Metadata Defensive Validation...
  -> Kernel metadata null checks, alignment, duplicate binding, and hash rejection verified.
  -> Gate 5 PASS

[Gate 6/10] Cross-Context Ownership Isolation & Launch Defensive Checks...
  -> Cross-context isolation and defensive launch checks verified.
  -> Gate 6 PASS

[Gate 7/10] BitNet GEMV I2_S Compute Dispatch via Public Opaque ABI...
  -> BitNet GEMV I2_S (256/256 weights): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 7 PASS

[Gate 8/10] Event Synchronization & Hardware Timing Query...
  -> Host Synchronous Wall Latency: 2516269 ns (2516.269 µs, 2.516 ms)
  -> GPU Timestamp Command Interval: 16884 ns (16.884 µs, 0.017 ms)
  -> Event Double-Wait Idempotency: Verified (Returned AC_SUCCESS)
  -> Gate 8 PASS

[Gate 9/10] Synchronous Dispatch Mode Contract (Vector-Add, out_event == NULL)...
  -> Synchronous Vector-Add Parity (N=256, alpha=2.5f): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 9 PASS

[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability via Public ABI...
  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS 37148 KB -> 37148 KB (Delta: 0 KB)
  -> Context Teardown: Completed cleanly via public ABI (Returned AC_SUCCESS)
  -> Gate 10 PASS

======================================================================
AMUDA 2.0 Phase 4A Verification Gates: 10 / 10 Passed
Public Opaque C ABI Runtime Status: VERIFIED PASS
======================================================================

--- STDERR ---
```

---

## 11. Galaxy A35 Phase 3 Universal Backend Raw Log (build-audit)

> **Source File**: `a35_phase3_raw.log`

```text
=== Galaxy A35 Phase 3 Universal Backend ===
Binary: build-audit/test_phase3_universal_backend
Command: cd ~/ameva-compute && unset LD_LIBRARY_PATH && timeout 120s ./build-audit/test_phase3_universal_backend
Exit Status: 0

--- STDOUT ---
======================================================================
AMUDA 2.0 Phase 3: Universal Backend & Cross-Device Hardware Test
======================================================================

[Gate 1/10] Hardware Diagnostic & Capabilities Inspection...
  -> Physical GPU Selected: Mali-G68 (Vendor: 0x13b5, Device: 0x92041010)
  -> Driver Version: 0x09801000, API Version: 1.3.219
  -> Non-Coherent Atom Size: 64 bytes
  -> Timestamp Compute Support: TRUE, Valid Bits: 64, Period: 38.46 ns/tick
  -> Gate 1 PASS

[Gate 2/10] Universal AcKernelMetadata & Hash Authentication (BitNet GEMV I2_S)...
  -> Kernel 'bitnet_gemv_i2s' created with SHA-256: 6d6a55fe2beb5291333c36280b2326e34d7470f29635b9d5e271c2e17463504a
  -> BitNet GEMV I2_S Decoding (256/256 weights): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 2 PASS

[Gate 3/10] Universal Vector-Add Kernel (vector_add.spv, N=256, alpha=2.5f)...
  -> Vector Add (N=256): Mismatches: 0/256, Max Error: 0.00e+00
  -> Gate 3 PASS

[Gate 4/10] Residual In-Place Vector Addition (x += alpha * y)...
  -> Residual In-Place (N=512): Mismatches: 0/512, Max Error: 0.00e+00
  -> Gate 4 PASS

[Gate 5/10] Arbitrary Non-Power-of-2 Sizes & Tail Boundaries...
  -> Tail N=37 (Groups=1): Mismatch=0, Canary Corrupted=0
  -> Tail N=255 (Groups=1): Mismatch=0, Canary Corrupted=0
  -> Tail N=1023 (Groups=4): Mismatch=0, Canary Corrupted=0
  -> Tail N=4099 (Groups=17): Mismatch=0, Canary Corrupted=0
  -> Gate 5 PASS

[Gate 6/10] IEEE-754 Special Floating-Point Values Preservation...
  -> [0] 5 + Inf = inf (isinf: 1)
  -> [1] 5 + (-Inf) = -inf (isinf: 1)
  -> [6] 1 + NaN = nan (isnan: 1)
  -> [7] 42 + 8 = 50.000000 (expected: 50.0)
  -> Gate 6 PASS

[Gate 7/10] Memory Management & Physical Non-Coherent Verification...
  -> Default Allocated Buffer Properties:
     Memory Type Index: 0 | Property Flags: 0x0006 | is_coherent: TRUE
     Logical Size: 4096 bytes | Physical Driver Allocation: 4096 bytes
  -> Forced Non-Coherent Allocation: Type Index=1, Flags=0x0002, is_coherent=FALSE (Atom=64 B)
  -> Physical Non-Coherent Memory Path: VERIFIED on this device (Flush/Invalidate SUCCESS)
  -> Gate 7 PASS

[Gate 8/10] Fine-Grained GPU Timestamp Command Interval vs Host Wall Latency...
  -> Avg Synchronous End-to-End Wall Latency: 557384 ns (557.384 µs, 0.557 ms)
  -> GPU timestamp command interval duration: 15038 ns (15.038 µs, 0.015 ms)
     (Measured between TOP_OF_PIPE and BOTTOM_OF_PIPE pipeline stages)
  -> GPU Command Interval Active Ratio: 2.70% of host wall time
  -> Gate 8 PASS

[Gate 9/10] Metadata Validation Failures (Corrupted Hash & Unsupported Features)...
  -> Corrupted Hash Rejection: Validated (Returned -304)
  -> Unsupported Feature Mask Rejection: Validated (Returned -108)
  -> Gate 9 PASS

[Gate 10/10] 1,000-Cycle Buffer Lifecycle RSS Stability & Strict Teardown...
  -> 1,000 Repeated Buffer Alloc/Free Cycles: VmRSS 36760 KB -> 36760 KB (Delta: 0 KB)
  -> Internal Counters Check: active_child_buffers=0, active_child_kernels=0, submissions=0
  -> Context Teardown: Cleanup completed with internal counters 0 (Returned 0)
  -> Gate 10 PASS

======================================================================
AMUDA 2.0 Phase 3 Verification Gates: 10 / 10 Passed
Storage-Buffer Compute Pipeline Status: VERIFIED PASS
======================================================================

--- STDERR ---
```

---
