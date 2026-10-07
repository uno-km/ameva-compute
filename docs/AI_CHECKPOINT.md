# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T14:12:00+09:00  
**Branch**: `perf/opencl-mali`  
**Current Goal**: Mali OpenCL Prompt Path Mathematical Optimization & Wan 2.1 5D Broadcast  

### Target Devices
- **Primary**: Galaxy S21 (Mali-G78), Galaxy A53 (Mali-G68)
- **Secondary**: Galaxy S25 (Xclipse 950 / Adreno 830), Galaxy A35 (Mali-G68), Galaxy S20 (Adreno 650)

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit: `termux-bitnet` and `termux-vision` public repos stripped of all `.cpp`/`.comp` sources.
- [PASS] Engine Isolation: BitNet & FastCV engines migrated to `dev/ameva/ameva-compute/adapters/`.
- [PASS] Clean-Slate Auto-Provisioning on A53: Stripped `libtermux_bitnet.so` (240 KB) auto-downloaded from GitHub Release v2.0.1 and executed via Vulkan GPU (30/30 layers in VRAM, 2.93 t/s).
- [PASS] 5-Modality Real-Device Inference on A53: LLM (50.9 t/s prompt / 24.2 t/s gen), BitNet (2.95 t/s), Vision CV filters, TTS (Kokoro ONNX 216KB WAV), STT (Whisper Tiny 100% transcript).
- [PASS] Anti-Hijacking Guard: `ac_security_guard` rejects unauthorized 3rd-party APK package contexts with `AMEVA-SEC-001`.

### Current Problem & Focus
- Wan 2.1 5D tensor broadcast and Mali OpenCL prompt path inner loop optimization.
- A35 translated tile kernel slower than native Vulkan (2.1~2.9x).

### Relevant Files
- `src/runtime/ac_runtime.cpp`
- `src/runtime/ac_security_guard.c`
- `adapters/bitnet/`
- `adapters/vision/`
- `scripts/glsl_to_opencl.py`

### Next Actions
1. Verify Mali OpenCL block integer MUL_MAT kernel inner loops.
2. Maintain single-file state in this checkpoint; do not reread historical logs.
