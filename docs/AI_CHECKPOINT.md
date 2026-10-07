# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T23:58:00+09:00  
**Branch**: `perf/opencl-mali` (uncommitted: this file, `docs/design/diffusion-vae-conv.md`, new `shaders/ameva_conv.comp`, `tests/ameva_conv_bench.cpp`, `scripts/termux/lab/conv_bench.sh`; changed `shaders/ameva_qdecode.glsl` (`TYPE_F16`), `shaders/ameva_qgemm.comp` (`ABLATE`, tuning only))  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`, `29b5362` + 2-row fix): `ameva_qmm_check` 516/516 on A53 OpenCL 9/9 and A35 18/18. Wins on Mali OpenCL (A53 Llama-3.2-1B prompt 46 -> 58 t/s, text equal to CPU), loses on every Vulkan driver and Adreno OpenCL. `ameva_mmvqb.comp` written, not built in, no gain on A53.
- Fleet defaults: A53 OpenCL MMQB 2x8, A35 Vulkan tile, S25/S20 MMQB off. Numbers: `docs/design/opencl-prompt-path.md`, git log.
- **Stage 1 (Diffusion), 2026-10-07. Design, tables, bench results: `docs/design/diffusion-vae-conv.md` (sections 1-2 A35 `sd-cli`, section 7 the bench).**
- A35 VAE decode of one 256x256 image: default 432.74 s (two convolutions with a 288 MiB IM2COL matrix run on the CPU, unreported); 192x192 23.55 s; `--vae-tile-size 24` 92.37 s; `--vae-conv-direct` 226.89 s. GPU part: MUL_MAT 97.1 % at 12.2-16.8 GFLOPS (1180 ms per 19.33 GFLOP convolution).
- Wan 2.1 VAE (code reading, not run): ggml `[W,H,T,B*C]`, `B == 1`; PERMUTE+CONT around RMS_NORM, 6-node CausalConv3d with a 27x column matrix; 13.5 TFLOP per 480x832 latent frame (calculated).
- **`ameva_conv_bench` (direct `ac_runtime`, kernel `ameva_conv.comp`: tile of `ameva_qgemm`, activations staged from the source tensor, host unit table, f16 or q8_0 weights).** 3 VAE shapes + one 3x3x3 causal + 4 small (stride, dilation, padded channels), A35 / S25 / S20, every output compared with an f32 CPU convolution: A35 bit-equal to the fused-multiply-add reference, S25 and S20 bit-equal to the multiply-then-add reference (they do not fuse `fma()`), 2.9e-6 of rms from the other.
- ms per 19.33 GFLOP convolution, matrix (`ameva_qgemm` on a CPU-built column matrix) / conv: A35 q8_0 238-246 / 273-275, f16 248-257 / 307-315 (277-287 with `unpackHalf2x16`, still bit-equal there); S25 56-57 / 45-46 (both types); S20 q8_0 172-179 / 152-171, f16 196-206 / 170-190. 3x3x3: A35 293 / 348 and 311 / 381, S25 71 / 56, S20 214 / 186.
- Mali-G68 ablation (q8_0): conv 273 = 199 + 45 planar output + 29 source loads; qgemm 236 = 201 + 35 matrix loads. The deficit on Mali is ggml's planar output (64 runs in 64 planes per tile), not the staging.
- Gate of step 1 (A35, 4x ggml-vulkan MUL_MAT, exact f16): 3.74-3.84x with the exact decoder (not met), 4.10-4.26x with the built-in conversion, 4.29-4.32x with q8_0 weights.

### Known, Not Examined
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (`clEnqueueNDRangeKernel` -54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`.
- A35 Vulkan 2-row `ameva_mmqb` layouts slow (60-120 ms). S21 `mmqb_s21.txt` (watcher pid 22747) not read yet.
- Stage 1 unmeasured: ggml-vulkan MUL_MAT time on S25/S20 (no gate there), conv bench on S21/A53, `unpackHalf2x16` on subnormals outside the A35, int8 activations in the conv kernel and their image effect, Wan VAE on a device, tensors above one binding (bench refuses them).

### Current Problem & Focus
- Decisions open for the user: (a) f16 decoder of the product kernel: exact everywhere, or built-in conversion after a load-time subnormal check per device; (b) keep ggml's planar layout (Mali pays 16 %) or a channels-last layout inside ggml-ameva for VAE activations.
- Earlier decisions (2026-10-07) stand: default tile size 24 on 256 MiB devices and fail-fast on CPU split; exact mode first, int8 behind a switch after a PSNR check; full Wan VAE on S25-class only, `taew2_1` elsewhere.

### Relevant Files
- `docs/design/diffusion-vae-conv.md`, `adapters/llama/ggml-ameva/shaders/ameva_conv.comp`, `ameva_qgemm.comp`, `ameva_qdecode.glsl`, `tests/ameva_conv_bench.cpp`, `scripts/termux/lab/conv_bench.sh` (`all|build|run`, `SHAPES`, `TYPES`, `EXTRA_DEFS=F16_UNPACK`, `AMEVA_CONV_BENCH_ABLATE=1`, `_REF=muladd`)
- Devices: `~/abi-lab/src-conv` (sources), `~/abi-lab/out-conv/run_final.log`, `run_ref_*.log` (A35, S25, S20), A35 `~/abi-lab/out-conv-unpack/`, A35 `~/claude_vae_prof/`

### Next Actions
1. K1 2D in ggml-ameva (CONV_2D accepted, weights converted once to the unit order, unit table, dispatch as the bench) + group norm + remaining ops; sd.cpp built with ggml-ameva (ggml `8e800cef`); gates in the design, section 6.
2. Then STT -> BitNet -> Vision -> TTS. Keep this file under 40 lines.
