# Evidence Bundle: AMUDA P0-3 Stabilization (GGUF Bounded Reader & Hardening)

## 1. Executive Summary & Verification Disposition
This evidence bundle documents the empirical verification and contract sealing for **P0-3 (GGUF Bounded Reader, Dimension Limits, and Offset Boundary Hardening)** on Samsung Galaxy S22 (`SM-S901N`) / Qualcomm Adreno 730.

* **Target Subsystem**: `src/amuda/amuda_gguf.c`, `include/amuda_gguf.h`
* **Defect Class**: Out-of-Bounds Memory Access, Buffer Overflow on `ne[4]`, Unaligned Word Dereference (SIGBUS), Unbounded Recursion, and Silent Truncation Acceptance.
* **Pre-Patch Baseline**: Terminated with `Bus error (Exit Code 135)` on malformed and truncated GGUF injection.
* **Post-Patch Baseline**: **20 / 20 Gates PASSED** (`VAL_TEST_EXIT_CODE=0`), CTest 3/3 PASSED (100%), Vulkan Phase 4B 10/10 PASSED. Zero compiler warnings.

## 2. Empirical Comparison: Pre-Patch vs Post-Patch

| Verification Metric | Pre-Patch (Empirical) | Post-Patch (Hardened & Audited) | Delta & Disposition |
| :--- | :--- | :--- | :--- |
| **Malformed File Fault Response** | `Exit Code 135 (SIGBUS / Bus Error)` | `AC_ERR_BINARY_CORRUPTED (-304)` / Safe Fail-Closed | **RESOLVED** (Crash eliminated, clean fail-closed error) |
| **`ne[4]` Buffer Boundary (`n_dims > 4`)** | Unbounded write into subsequent struct fields | `AC_ERR_BINARY_CORRUPTED` (Gate 06 PASS) | **RESOLVED** (Strict `1 <= n_dims <= 4` invariant enforced) |
| **Dimension Zero (`ne[d] == 0`)** | Unchecked zero-element shape | `AC_ERR_BINARY_CORRUPTED` (Gate 09 PASS) | **RESOLVED** (Strict positivity validation) |
| **Shape Product Overflow** | Integer multiplication wrap-around | `AC_ERR_BINARY_CORRUPTED` (Gate 10 PASS) | **RESOLVED** (Safe checked multiplication) |
| **Alignment Constraint Enforcement** | Division by zero or integer overflow | Power-of-2 validated, range guarded (Gate 12-14 PASS) | **RESOLVED** (Non-power-of-2 and overflow rejected) |
| **Tensor Data Offset Boundary** | Unchecked pointer arithmetic | Invariant: `toff <= avail && tsz <= avail - toff` | **RESOLVED** (OOB memory access prevented) |
| **Metadata & Tensor Sane Caps** | Unbounded loop and malloc | Capped at 1,000,000 count with overflow guard | **RESOLVED** (OOM and DoS vectors prevented) |
| **Validation Gates Passed** | 0 / 20 Gates (Crashed at Gate 1-2) | **20 / 20 Gates PASSED** | Full coverage across boundaries and injections |
| **CTest Suite Execution** | 2 / 2 Passed (Pre-P0-3) | **3 / 3 Passed (100%)** | `amuda_gguf_validation` added to automated CTest |
| **Vulkan Phase 4B Regression** | 10 / 10 Gates PASS | **10 / 10 Gates PASS** (`EXIT_CODE=0`) | Zero regression across Vulkan runtime |

## 3. Production Fix Architecture: `GgufReader`
1. **Stateful Bounded Reader**:
   - Encapsulates `cur` and `end` pointers.
   - All primitive readers (`reader_read_u32`, `reader_read_u64`, `reader_read_f32`, `reader_read_string`) verify `cur + size <= end` before dereferencing.
   - Little-endian byte assembly avoids unaligned memory faults on ARM64.
2. **Safe Arithmetic Invariants**:
   - `checked_mul_u64`: Verifies `UINT64_MAX / a >= b` before multiplying.
   - `checked_align_up`: Prevents `val + add` integer overflow.
   - Offset invariant: `offset <= available && size <= available - offset` eliminates addition wrap-around vulnerabilities.
3. **Fail-Closed Resource Management**:
   - Parsing errors immediately clean up `tensors`, unmap memory (`munmap`), close file descriptor, and set `*out_model = NULL`.

## 4. Test Matrix & Verification Coverage (20 Gates)
* **Gate 01**: `TRUNCATED_HEADER_REJECTED` -> PASS
* **Gate 02**: `TRUNCATED_METADATA_COUNT_REJECTED` -> PASS
* **Gate 03**: `TRUNCATED_METADATA_KEY_REJECTED` -> PASS
* **Gate 04**: `TRUNCATED_TENSOR_NAME_REJECTED` -> PASS
* **Gate 05**: `NDIMS_ZERO_REJECTED` -> PASS
* **Gate 06**: `NDIMS_FIVE_REJECTED` (`n_dims=5` reproducer) -> PASS
* **Gate 07**: `NDIMS_UINT32_MAX_REJECTED` -> PASS
* **Gate 08**: `TRUNCATED_DIMENSION_ARRAY_REJECTED` -> PASS
* **Gate 09**: `ZERO_DIMENSION_REJECTED` -> PASS
* **Gate 10**: `DIMENSION_PRODUCT_OVERFLOW_REJECTED` -> PASS
* **Gate 11**: `UNKNOWN_TENSOR_TYPE_REJECTED` -> PASS
* **Gate 12**: `ALIGNMENT_ZERO_REJECTED` -> PASS
* **Gate 13**: `ALIGNMENT_NON_POWER_OF_TWO_REJECTED` -> PASS
* **Gate 14**: `ALIGN_UP_OVERFLOW_REJECTED` -> PASS
* **Gate 15**: `TENSOR_OFFSET_OUT_OF_RANGE_REJECTED` -> PASS
* **Gate 16**: `TENSOR_END_OUT_OF_RANGE_REJECTED` -> PASS
* **Gate 17**: `TENSOR_COUNT_ALLOCATION_OVERFLOW_REJECTED` -> PASS
* **Gate 18**: `METADATA_COUNT_LIMIT_REJECTED` -> PASS
* **Gate 19**: `VALID_MINIMAL_GGUF_ACCEPTED` -> PASS
* **Gate 20**: `VALID_REAL_GGUF_ACCEPTED` -> PASS
