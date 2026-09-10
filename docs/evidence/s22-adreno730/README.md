# Evidence Bundle: Galaxy S22 Adreno 730 Validation & Heterogeneous Phase 4B Regression

## 1. Directory Scope & Historical Context
* **Directory Name Origin**: This directory was originally established for the qualification and verification baseline of **Samsung Galaxy S22 (`SM-S901N`) / Qualcomm Adreno 730**.
* **Cross-Device Regression Scope**: In addition to S22 validation records, this bundle contains cross-device regression logs for **Samsung Galaxy S25 (Qualcomm Adreno 830)** and **Samsung Galaxy A35 (ARM Mali-G68 MP5)**. These logs were captured to rigorously verify that the Gate 10 RSS decision contract (`evaluate_rss_growth`, fail-closed enum classification, and 8 safety boundary unit tests) holds across heterogeneous GPU architectures and memory subsystems without regressions.
* **Remote Deployment Model**: Remote Android Termux devices operate as build and execution deployment targets without standalone `.git` repositories. Source provenance is cryptographically bound to repository commit `088c457eee5d880dc5efedcd9415c8fea0d503a9` via exact SHA-256 bit-for-bit hash match of `tests/test_phase4b_specialization.cpp` (`95cb71360fcc64c977c53e332ccecd4c141949cef7915ab39e39b1787814146c`).
* **S22 Build Directory Discrepancy Verification**: S22 was initially deployed under `build-phase4c-audit/`. Both `build-audit/` and `build-phase4c-audit/` were independently compiled from the same source using identical flags, yielding identical SHA-256 binaries (`ff38807c5e9727c4af4656d324ee1c45f21426567dfcf2c80bdfa3284ab2e4c8`).
* **Future Reorganization Plan**: In future milestone reorganizations, multi-device evidence will be split into device-dedicated paths under `docs/evidence/phase4b/{s22-adreno730, s25-adreno830, a35-mali-g68}/`.

## 2. Hardware Fleet Disambiguation Matrix
| Field | Galaxy S22 | Galaxy S25 | Galaxy A35 |
| :--- | :--- | :--- | :--- |
| **Model Number** | `SM-S901N` | `SM-S931N` | `SM-A356N` |
| **SoC** | Qualcomm Snapdragon 8 Gen 1 | Qualcomm Snapdragon 8 Elite | Samsung Exynos 1380 |
| **GPU Model** | Qualcomm Adreno 730 | Qualcomm Adreno 830 | ARM Mali-G68 MP5 |
| **Vulkan vendorID** | `0x5143` | `0x5143` | `0x13B5` |
| **Vulkan deviceID** | `0x07030001` | `0x08300000` | `0x00080001` |
| **Vulkan Driver Version** | `0x80267062` | `0x80302000` | `0x00260000` |
| **Tested Source SHA-256** | `95cb71360fcc64c977c53e332ccecd4c141949cef7915ab39e39b1787814146c` | `95cb71360fcc64c977c53e332ccecd4c141949cef7915ab39e39b1787814146c` | `95cb71360fcc64c977c53e332ccecd4c141949cef7915ab39e39b1787814146c` |
| **Binary SHA-256** | `ff38807c5e9727c4af4656d324ee1c45f21426567dfcf2c80bdfa3284ab2e4c8` | `ff38807c5e9727c4af4656d324ee1c45f21426567dfcf2c80bdfa3284ab2e4c8` | `cd23a57c879e0286b05e2389e008a583a487df67ebd0600fcb9b8e5ed2a68afe` |
| **Target Android API Level** | API 30 (`.note.android.ident`: `0x1e`) | API 30 (`.note.android.ident`: `0x1e`) | API 24 (`.note.android.ident`: `0x18`) |
| **Binary Provenance Note** | Identical to S25 binary | Identical to S22 binary | Diverges due to Android API level 24 toolchain target |
| **Device Execution Exit Code** | `EXIT_CODE=0` | `EXIT_CODE=0` | `EXIT_CODE=0` |
| **Host-Side SSH Exit Code** | `S22_SSH_EXIT_CODE=0` | `S25_SSH_EXIT_CODE=0` | `A35_SSH_EXIT_CODE=0` |

## 3. Bundle File Manifest
Total bundle contains **14 files** (13 cryptographic verification target files + 1 `SHA256SUMS` index file):
1. `device-environment.txt`: Galaxy S22 physical hardware and toolchain inventory.
2. `official_vulkan_probe_full.cpp`: Official Vulkan physical device probe source code.
3. `s22-probe-build.txt`: Native Clang build and dynamic linking metadata on Galaxy S22.
4. `vulkan-limits.txt`: Official Vulkan hardware limits probed directly on Adreno 730.
5. `s22_phase3_raw.log`: Galaxy S22 Phase 3 (Universal Backend) execution raw log (10/10 PASS).
6. `s22_phase4a_raw.log`: Galaxy S22 Phase 4A (Runtime Opaque ABI) execution raw log (10/10 PASS).
7. `s22_phase4b_raw.log`: Galaxy S22 Phase 4B (Specialization Constants) execution raw log with `EXIT_CODE=0` (10/10 PASS).
8. `s25_phase4b_regression.log`: Galaxy S25 Phase 4B cross-device regression log with `EXIT_CODE=0` and provenance metadata (10/10 PASS).
9. `a35_phase4b_regression.log`: Galaxy A35 Phase 4B cross-device regression log with `EXIT_CODE=0` and provenance metadata (10/10 PASS).
10. `source-manifest.sha256`: SHA-256 hashes of all tested source files.
11. `binary-manifest.sha256`: SHA-256 hashes of Galaxy S22 binaries.
12. `s22-provenance.txt`: Provenance details linking sources, binaries, and ELF headers on Galaxy S22, S25, and A35.
13. `README.md`: Evidence bundle scope, provenance documentation, and hardware disambiguation.
14. `SHA256SUMS`: Index of SHA-256 checksums verifying files 1 through 13.

## 4. Tag Scope & Certification Boundary
Any tag issued against this baseline (e.g. `v2.1.2-phase4b`) represents **Phase 4B specialization regression validation baseline only**.
* **Validated Scope**:
  - RSS growth classification test contract (`RssGrowthResult`, fail-closed enum, 8 safety boundary tests).
  - Galaxy S22, S25, and A35 Phase 4B regression execution (`EXIT_CODE=0`, `$LASTEXITCODE=0`).
  - Evidence checksums and recorded exit codes.
* **Non-Certification Scope**:
  - This is NOT a public release readiness certification.
  - Known P0 defects documented in `ameva_compute_consolidated.md` remain unresolved and are tracked for resolution in dedicated stabilization branches.
