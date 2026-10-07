# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T14:58:00+09:00  
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
- [PASS] Block-integer q4_K OpenCL prototype (`opencl/ameva_cl_q4k.cl`: `quant_a8b`, `gemm_q4k_i8_blk`, `gemv_q4k_i8_blk`), 2026-10-07: every output vs the double reference (`AMEVA_CL_CHECK_ALL=1`), max_rel_err <= 6e-8 on A53 (full shapes) and S21 (small shapes, correctness only).
- Bench fix (`tests/ameva_cl_bench.c`): the activation-row sample step was even and saw rows 0 and 4 of a work-item's 8; now odd. Mutation control on A53: HEAD bench PASS, fixed bench FAIL.
- Prototype limits: the GEMV kernel is valid for M = 1 only, the GEMM kernel needs N % DRW == 0 and M % DRA == 0, neither is in the product path. A53 GEMM time varies 10x with rows per item (2x8: 19.8 ms, 8x8: 195 ms at 352x2048x2048).

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
1. Write the block-integer MUL_MAT as GLSL kernels (shared by Vulkan and OpenCL through `scripts/glsl_to_opencl.py`), with the tail rows and row ranges the prototype lacks; check with `ameva_qmm_check`, sweep rows per item per device.
2. Maintain single-file state in this checkpoint; do not reread historical logs.
