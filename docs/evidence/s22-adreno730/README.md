# Evidence Bundle: Galaxy S22 Adreno 730 Validation & Heterogeneous Phase 4B Regression

## 1. Directory Scope & Historical Context
* **Directory Name Origin**: This directory was originally established for the qualification and verification baseline of **Samsung Galaxy S22 (`SM-S901N`) / Qualcomm Adreno 730**.
* **Cross-Device Regression Scope**: In addition to S22 validation records, this bundle contains cross-device regression logs for **Samsung Galaxy S25 (Qualcomm Adreno 830)** and **Samsung Galaxy A35 (ARM Mali-G68 MP5)**. These logs were captured to rigorously verify that the Gate 10 RSS decision contract (`evaluate_rss_growth`, fail-closed enum classification, and 8 safety boundary unit tests) holds across heterogeneous GPU architectures and memory subsystems without regressions.
* **Future Reorganization Plan**: In future milestone reorganizations, multi-device evidence will be split into device-dedicated paths under `docs/evidence/phase4b/{s22-adreno730, s25-adreno830, a35-mali-g68}/`.

## 2. Hardware Fleet Disambiguation Matrix
| Device | SoC | GPU Model | Vulkan vendorID | Vulkan Driver Version | Tested Commit | Binary SHA-256 | Exit Code |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S22 (SM-S901N)** | Snapdragon 8 Gen 1 | Qualcomm Adreno 730 | `0x5143` | `0x80267062` | `088c457` | `ff38807c...` | `EXIT_CODE=0` |
| **Galaxy S25 (SM-S931N)** | Snapdragon 8 Elite | Qualcomm Adreno 830 | `0x5143` | `0x80302000` | `088c457` | `ff38807c...` | `EXIT_CODE=0` |
| **Galaxy A35 (SM-A356N)** | Exynos 1380 | ARM Mali-G68 MP5 | `0x13B5` | `0x00260000` | `088c457` | `cd23a57c...` | `EXIT_CODE=0` |

*Note: S22 and S25 binaries share bit-for-bit identical SHA-256 due to identical Clang 21.1.8 compiler toolchains and optimization flags. A35 binary was compiled locally on the A35 Termux toolchain from identical source (`tests/test_phase4b_specialization.cpp` SHA-256: `95cb71360fcc64c977c53e332ccecd4c141949cef7915ab39e39b1787814146c`).*

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
12. `s22-provenance.txt`: Provenance details linking sources, binaries, and ELF headers on Galaxy S22.
13. `README.md`: Evidence bundle scope, provenance documentation, and hardware disambiguation.
14. `SHA256SUMS`: Index of SHA-256 checksums verifying files 1 through 13.
