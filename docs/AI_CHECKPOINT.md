# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T16:28:00+09:00  
**Branch**: `perf/opencl-mali` (pushed to origin; code at `29b5362`)  
**Current Goal**: Mali OpenCL prompt path (primary S21 Mali-G78, A53 Mali-G68; secondary S25, A35, S20) & Wan 2.1 5D Broadcast  

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit: `termux-bitnet` and `termux-vision` public repos stripped of all `.cpp`/`.comp` sources.
- [PASS] Engine Isolation: BitNet & FastCV engines migrated to `dev/ameva/ameva-compute/adapters/`.
- [PASS] Clean-Slate Auto-Provisioning on A53: Stripped `libtermux_bitnet.so` (240 KB) auto-downloaded from GitHub Release v2.0.1 and executed via Vulkan GPU (30/30 layers in VRAM, 2.93 t/s).
- [PASS] 5-Modality Real-Device Inference on A53: LLM (50.9 t/s prompt / 24.2 t/s gen), BitNet (2.95 t/s), Vision CV filters, TTS (Kokoro ONNX 216KB WAV), STT (Whisper Tiny 100% transcript).
- [PASS] Anti-Hijacking Guard: `ac_security_guard` rejects unauthorized 3rd-party APK package contexts with `AMEVA-SEC-001`.
- [PASS] Block-integer q4_K OpenCL prototype (`opencl/ameva_cl_q4k.cl`) with every output checked (`AMEVA_CL_CHECK_ALL=1`, bench sample fixed), A53 and S21, 2026-10-07.
- [PASS] Block-integer MUL_MAT as GLSL kernels (`shaders/ameva_mmqb.comp`, `ameva_quant_a8b.comp`; q4_K, q5_K, q6_K; 6 layouts; tail rows, slices, chunks), opt-in `AMEVA_MMQB=1 | <w>x<a> | <type>=<w>x<a>,...`: `ameva_qmm_check` 516/516 on A53 OpenCL, A35 Vulkan, A35 OpenCL; 214 kernels, existing SPIR-V byte-identical (generated on A53, glslang 16.3.0).
- Kernel time 352x2048x2048 (ms, tile `ameva_mmq` -> best `ameva_mmqb`): A53 OpenCL q4_K 28.0 -> 19.3 (2x8), q6_K 33.3 -> 32.0 (1x8), q5_K no gain; A35 OpenCL q4_K 33.5-42.2 -> 15.9 (2x8), q6_K 60-66 -> 25.4 (1x8); A35 Vulkan: the tile is faster in every layout (q4_K 14.8 vs 23-40).
- A53 Llama-3.2-1B Q4_K_M prompt t/s (OpenCL, 3 alternating rounds): CPU 47.2-47.9, tile 45.9-46.1, `AMEVA_MMQB=q4_K=2x8` 57.4-57.9; prompt text equal to the CPU's; batch KLD 3.13e-3 (tile 2.14e-3, gate 0.01). Generation unchanged (10.6-11.1 vs CPU 13.4).
- Cause found by on-device kernel-text bisect: block minima decoded to four ivec4 before the unit loop cost 19.7 -> 26.4 ms (2-row kernel, Mali-G68); decode after the loop.
- Known, not from this change: A53 OpenCL 64-token generation text of Llama-3.2-1B differs from the CPU's from byte 70 (default configuration).

### Current Problem & Focus
- Wan 2.1 5D tensor broadcast and Mali OpenCL prompt path inner loop optimization.
- A35 translated tile kernel slower than native Vulkan (2.1~2.9x); `ameva_mmqb` closes it for q4_K on OpenCL. Two weight rows per invocation double the time for q5_K and q6_K (cause unknown).

### Relevant Files
- `src/runtime/ac_runtime.cpp`
- `src/runtime/ac_security_guard.c`
- `adapters/bitnet/`
- `adapters/vision/`
- `scripts/glsl_to_opencl.py`, `scripts/build_ameva_kernels.py`
- `adapters/llama/ggml-ameva/shaders/ameva_mmqb.comp`, `docs/design/opencl-prompt-path.md` (stage B, order 4)

### Next Actions
1. Measure `ameva_mmqb` on S21 (another session's `sd-cli-vulkan` held the GPU), S25 and S20; then decide per-device defaults (today opt-in only).
2. Isolate the 2-row slowdown of q5_K and q6_K; write the token-generation block kernel (`gemv_q4k_i8_blk`) in GLSL.
3. Maintain single-file state in this checkpoint; do not reread historical logs.
