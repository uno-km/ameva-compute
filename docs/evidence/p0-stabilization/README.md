# Evidence Bundle: AMUDA P0-2 Stabilization (Synchronous Contract & Input Validation)

## 1. Executive Summary & Verification Disposition
This evidence bundle documents the empirical verification and contract sealing for **P0-2 (P0-2A Synchronous Contract and P0-2B Input Validation & Fail-Closed Semantics)** on Samsung Galaxy S22 (`SM-S901N`) / Qualcomm Adreno 730.

* **Target Subsystem**: OpenCL / AMUDA Host Dispatch & Tensor Operators
* **Unaffected Subsystems**: Vulkan 2.0 Compute Core (Vulkan dispatch, event lifecycle, and timeout mechanisms were already independently verified in Phase 4B and regression-tested with 10/10 PASS).

## 2. Empirical Comparison: Pre-Patch vs Post-Patch

| Verification Metric | Pre-Patch (Empirical) | Post-Patch (Hardened & Audited) | Delta & Disposition |
| :--- | :--- | :--- | :--- |
| **GPU Baseline 1024x1024 Duration** | `64,211 us` | `64,214 us` (Median of 3 runs) | Stable reference benchmark |
| **out_event == NULL Host Return** | `1,340 us` (Asynchronous return defect) | `281,853 us` (Full synchronous blocking) | **RESOLVED** (Host strictly blocks until completion) |
| **Timing Ratio (Host / GPU)** | `0.021 << 0.50` (FAIL) | `4.389 >= 0.80` (PASS) | Host execution duration exceeds GPU work time |
| **Residual Queue Finish Probe** | N/A (GPU workload still executing) | **`20 us`** (Median of 3 runs: 25, 19, 20 us) | **PROVEN**: Queue completely idle upon function return (< 500 us) |
| **Sync Audit Gates Passed** | `1/5 Gates` | **`8/8 Gates`** | Includes immediate enqueue rejection (1 us) & error mapping |
| **Validation Gates Passed** | `0/8 Gates` | **`28/28 Gates`** | Full operator coverage & overflow site isolation |
| **Vulkan Phase 4B Regression** | 10/10 PASS | **10/10 PASS** (`VULKAN_REGRESSION_EXIT_CODE=0`) | Zero regression across Vulkan runtime |
| **Existing AMUDA Regressions** | UMA & Ops PASS | **UMA & Ops PASS** (`EXIT_CODE=0`) | 100% backward compatible |

## 3. Production Fix Commits
1. **`b18a926`** - `fix(amuda): honor synchronous null-event dispatch contract`
   - Internal native `cl_event` guaranteed for `out_event == NULL`.
   - Host blocks via `clWaitForEvents` before release.
   - Non-blocking async path preserves caller event ownership.
   - OpenCL wait errors mapped to explicit `AcResult` codes.
2. **`61c148f`** - `fix(amuda): reject invalid dispatch dimensions and fail closed`
   - Non-zero grid and block constraints enforced (`AC_ERR_INVALID_WORK_ITEM_DIMS`, `AC_ERR_INVALID_WORK_GROUP_SIZE`).
   - Global work dimensions guarded against 32-bit overflow (`UINT32_MAX`, `AC_ERR_NUMERICAL_OVERFLOW`).
   - Dimension non-positivity and buffer capacity validated across all operators (`gemm`, `bitnet`, `rmsnorm`, `swiglu`, `rope`).
   - Fail-closed `*out_event = NULL` initialization on entry.

## 4. Test Matrix & Overflow Site Disambiguation
The 28 validation gates in `tests/test_amuda_dispatch_validation.c` explicitly identify and isolate failure sites:
* **`OVERFLOW_SITE=checked_mul`**: Dimension products exceeding 64-bit/allocation bounds -> `AC_ERR_NUMERICAL_OVERFLOW` (-502)
* **`OVERFLOW_SITE=global_size`**: Grid-block product exceeding 32-bit GPU limits -> `AC_ERR_NUMERICAL_OVERFLOW` (-502)
* **`OVERFLOW_SITE=buffer_required_bytes`**: Undersized buffer allocation without arithmetic overflow -> `AC_ERR_INVALID_BUFFER_SIZE` (-206)
* **`OVERFLOW_SITE=dispatch_budget`**: Block dimension exceeding max workgroup size -> `AC_ERR_INVALID_WORK_GROUP_SIZE` (-401)
* **`OVERFLOW_SITE=non_positive_dim`**: Non-positive dimensions -> `AC_ERR_KERNEL_ARG_INVALID` (-303)

## 5. Remaining Engineering Constraints & Disclosures
1. **Timing-Based Portability**: Host timing ratios depend on CPU/GPU frequency scaling (DVFS). The residual `clFinish()` probe (20 us) serves as the primary invariant proof.
2. **Device Hardware Availability**: AMUDA OpenCL path operates on platforms providing vendor OpenCL ICD (e.g. Qualcomm Adreno). Exynos / Mali devices utilize the Vulkan backend path.
3. **Release & Push Policy**: Remote git push and release tagging remain strictly on HOLD pending completion of all P0 stabilization items.
