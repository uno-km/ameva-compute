# S22 Adreno 730 Hardware Validation & 3-Device Phase 4B Regression Evidence Bundle

This directory contains cryptographic and execution evidence for:
1. **Galaxy S22 (SM-S901N / Qualcomm Adreno 730)** hardware verification across Phase 3, Phase 4A, and Phase 4B (30/30 Gates PASS).
2. **Multi-device Phase 4B regression bundle** verifying Gate 10 RSS fail-closed behavior across:
   - Samsung Galaxy S22 (SM-S901N / Qualcomm Adreno 730)
   - Samsung Galaxy S25 (SM-S931N / Qualcomm Adreno 830)
   - Samsung Galaxy A35 (SM-A356N / ARM Mali-G68 MP5)

### Files in this Bundle
- `device-environment.txt`: Galaxy S22 physical hardware and toolchain inventory.
- `official_vulkan_probe_full.cpp`: Official Vulkan physical device probe source code.
- `s22-probe-build.txt`: Native Clang build and ELF dynamic linking metadata on S22.
- `vulkan-limits.txt`: Official Vulkan limits probed directly on Adreno 730.
- `s22_phase3_raw.log`: Galaxy S22 Phase 3 (Universal Backend) execution log (10/10 PASS).
- `s22_phase4a_raw.log`: Galaxy S22 Phase 4A (Runtime Opaque ABI) execution log (10/10 PASS).
- `s22_phase4b_raw.log`: Galaxy S22 Phase 4B (Specialization Constants) execution log (10/10 PASS).
- `s25_phase4b_regression.log`: Galaxy S25 Phase 4B regression execution log (10/10 PASS).
- `a35_phase4b_regression.log`: Galaxy A35 Phase 4B regression execution log (10/10 PASS).
- `source-manifest.sha256`: SHA-256 hashes of tested source files.
- `binary-manifest.sha256`: SHA-256 hashes of tested S22 binaries.
- `s22-provenance.txt`: Provenance details linking sources, binaries, and ELF headers.
- `SHA256SUMS`: Cryptographic checksums of all files in this evidence bundle.
