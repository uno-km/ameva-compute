# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T21:18:00+09:00  
**Branch**: `perf/opencl-mali` (clean working tree synced with origin)  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit: `termux-bitnet` and `termux-vision` public repos stripped of all `.cpp`/`.comp` sources.
- [PASS] Engine Isolation: BitNet & FastCV engines migrated to `dev/ameva/ameva-compute/adapters/`.
- [PASS] Clean-Slate Auto-Provisioning on A53: Stripped `libtermux_bitnet.so` (240 KB) auto-downloaded from GitHub Release v2.0.1 and executed via Vulkan GPU (30/30 layers in VRAM, 2.93 t/s).
- [PASS] 5-Modality Real-Device Inference on A53: LLM (50.9 t/s prompt / 24.2 t/s gen), BitNet (2.95 t/s), Vision CV filters, TTS (Kokoro ONNX 216KB WAV), STT (Whisper Tiny 100% transcript).
- [PASS] Anti-Hijacking Guard: `ac_security_guard` rejects unauthorized 3rd-party APK package contexts with `AMEVA-SEC-001`.
- [PASS] Block-integer q4_K OpenCL prototype (`opencl/ameva_cl_q4k.cl`) with every output checked (`AMEVA_CL_CHECK_ALL=1`), A53 and S21, 2026-10-07.
- [PASS] Block-integer MUL_MAT as GLSL kernels (`shaders/ameva_mmqb.comp`, `ameva_quant_a8b.comp`; q4_K, q5_K, q6_K; 6 layouts), opt-in `AMEVA_MMQB=1 | <w>x<a> | <type>=<w>x<a>,...` (`29b5362`).
- [PASS] Cause of the 2-row slowdown of q5_K and q6_K (A53, on-device kernel-text bisect, one loop written out per variant): the loop over the weight rows that decodes a unit's weights stays a loop in the Mali OpenCL compiler once the body holds the high-bit decode; written out (`UNIT_ROW`, new `shaders/ameva_blk.glsl`) q5_K 2x8 67 -> 24 ms, q6_K 62 -> 28; the activation-row loop must stay a loop (written out too: 33 and 46 ms).
- [PASS] With that fix, headers regenerated on A53 (glslang 16.3.0; 18 of 214 kernel hashes changed, all `ameva_mmqb`): `ameva_qmm_check` 516/516 in 9/9 settings on A53 OpenCL (plugin `cedaf25f6bea`) and 18/18 on A35 (Vulkan 9, OpenCL 9; `7f8501322218`).
- Kernel time 352x2048x2048 after the fix (ms, tile `ameva_mmq` -> `ameva_mmqb` 2x8): A53 OpenCL q4_K 27.7-28.0 -> 18.7-18.9, q5_K 31.2-32.1 -> 23.8-24.0 (before 72-76), q6_K 32.3-32.8 -> 27.8-28.9 (before 59-71). A35 OpenCL 2x8 before -> after: q4_K 19.6-21.5 -> 15.5, q5_K 89 -> 22.9, q6_K 80-90 -> 40.2 (best stays 1x8 24.5). A35 Vulkan 2-row layouts are unchanged and slow (60-120; cause not isolated).
- `ameva_mmqb` loses to the tile on every Vulkan driver and on Adreno OpenCL: A35 Vulkan; S25 (Vulkan 2.44 -> 2.93, OpenCL 3.83 -> 6.12 for q4_K); S20 Vulkan (14.6 -> 52.7; its OpenCL has no int8 dot product). `ameva_qmm_check` Vulkan 516/516 9/9 on S25 and S20. **Keep it off on S25 and S20.**
- A53 Llama-3.2-1B Q4_K_M prompt t/s at `e3a3af8` (OpenCL, 349 tokens, 3 alternating rounds, cores 0-7): CPU 47.1-47.5, tile 45.8-46.7, `AMEVA_MMQB=q4_K=2x8` 57.3-58.2, `AMEVA_MMQB=2x8` (q4_K, q5_K, q6_K) 57.8-58.8; prompt text equal to the CPU's in 9/9 GPU runs. Generation unchanged (10.9-11.1 vs CPU 13.5-13.8). Batch KLD before the fix 3.13e-3 (gate 0.01; same arithmetic, not measured again).
- A35 the same benchmark (ssh cores 0-2, plugin `7f8501322218`): prompt CPU 16.8-17.0, **Vulkan tile 82.2-82.5**, OpenCL tile 39.6-40.4, `q4_K=2x8` 55.1-61.1, `2x8` 59.1-67.3; text equal to the CPU's in 12/12 GPU runs. Generation CPU 4.5-4.6, Vulkan 11.8-12.0, OpenCL 13.9-14.0. **On A35 OpenCL with block kernels stays slower than the Vulkan tile for prompts.**
- Token-generation block kernel written: `shaders/ameva_mmvqb.comp` (GLSL `gemv_q4k_i8_blk`; q4_K, q5_K, q6_K; interface and lane split of `ameva_mmvq`). **Not built into the plugin.** A53, text swapped in for `ameva_mmvq` (`AMEVA_CL_SOURCE_DIR`), kernel ms, chosen layout -> block: q4_K 2048x2048 0.25 -> 0.25, 8192x2048 0.97 -> 0.91, 2048x8192 1.01 -> 1.11, q5_K 8192x2048 1.50 -> 1.80, q6_K 32064x2048 5.10 -> 11.4. No gain: the chosen layouts already read weights at the uncached-memory rate (9.7-10.6 GB/s). Compared with the f32 product only, not with a block-rounded reference.
- Known, not examined: 64-token generation text of Llama-3.2-1B departs from the CPU's at byte 70 on A53 OpenCL and on A35 Vulkan and OpenCL (same place, coherent text, block kernels on or off); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1, `ameva_qmv_q2_K_c1_r4` build -6); S20 OpenCL `ameva_qmm_check` exits 139 and its f32 tile takes 8.6-16.3 s per kernel.

### Current Problem & Focus
- Stage 1 [Diffusion]: VAE decoder GPU optimization & Wan 2.1 5D tensor broadcast (B, C, T, H, W).
- Fleet Defaults: A53 OpenCL MMQB 2x8, A35 Vulkan tile, S25/S20 off. S21 queued unattended overnight (pid 22747).

### Relevant Files
- `dev/termux/termux-diffusion/`, `dev/termux/termux-stt/`, `dev/termux/termux-bitnet/`, `dev/termux/termux-vision/`, `dev/termux/termux-tts/`
- `adapters/llama/ggml-ameva/`, `src/runtime/ac_runtime.cpp`, `include/ac_base.h`

### Next Actions
1. Stage 1 (Diffusion): Inspect LOCAL `dev/termux/termux-diffusion/` (do NOT fetch remote branches), isolate VAE decode bottleneck, and design Wan 2.1 5D tensor broadcast kernel.
2. S21: Unattended watcher (pid 22747) will read `mmqb_s21.txt` when diffusion sweep ends.
3. Follow 5-Modality Master Order: Diffusion -> STT -> BitNet -> Vision -> TTS.
4. Maintain single-file state in this checkpoint (<40 lines); do not reread historical logs.
