# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T02:25:00+09:00  
**Branch**: `perf/opencl-mali` (pushed: `109ba54` CONV_2D in ggml-ameva, `ea0a336` conv kernel and bench)  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`, `29b5362` + 2-row fix): wins on Mali OpenCL only. Fleet defaults: A53 OpenCL MMQB 2x8, A35 Vulkan tile, S25/S20 MMQB off. Numbers: `docs/design/opencl-prompt-path.md`.
- **Stage 1 (Diffusion). Everything below with tables: `docs/design/diffusion-vae-conv.md` (sections 1-2 A35 `sd-cli`, 7 the kernel bench, 8 the integration).**
- A35 VAE decode of one 256x256 image with the 2.7.9 bundle: default 432.74 s (two convolutions with a 288 MiB IM2COL matrix run on the CPU, unreported); `--vae-tile-size 24` 92.37 s; `--vae-conv-direct` 226.89 s. GPU part: MUL_MAT 97.1 % (1180 ms per 19.33 GFLOP convolution).
- `ameva_conv` kernel alone (`ameva_conv_bench`, direct `ac_runtime`): bit-equal to an f32 CPU convolution with each driver's arithmetic (A35 fuses `fma()`, S25 and S20 do not). Against the same tile kernel on a column matrix: faster on Adreno (S25 1.22-1.28x, S20 1.00-1.18x), 0.80-0.90x on Mali-G68, where ggml's planar output costs 45 of 273 ms.
- **CONV_2D in ggml-ameva (2026-10-08)**: `enc_conv_2d`; weights copied per node into the kernel's K order in the scratch buffer (`ameva_conv_w`), no side buffer; SPIR-V-only kernel family (`VK_ONLY`, header regenerated on the S25, existing modules unchanged); declined on the OpenCL backend; IM2COL stays declined.
- f16 decoder: built-in `unpackHalf2x16` after a per-process device check of 4096 values (every f16 subnormal), else the exact decoder (`AMEVA_CONV_F16=exact` forces it). Built-in selected on A35, S20, S25. Layout: ggml's planar layout kept (user decisions 2026-10-07).
- `tests/ameva_conv_check` (12 cases x 5 kernel selections, bit comparison with an f32 CPU reference): A35 12/12 x 5, S20 12/12 x 5, S25 11/11 x 5 (sources before the last two edits). `test-backend-ops -o CONV_2D` on AMEVA0: S20 and A35 1568/1568, none declined. `ameva_qmm_check` 516/516 on all three.
- Through `graph_compute`, f16 kernel, ms per 19.33 GFLOP convolution: A35 271.7 / 287.6 / 271.3 (4.10-4.34x ggml-vulkan MUL_MAT, cores 0-2, 949 MHz), S20 158.8 / 163.8 / 182.4.

### Known, Not Examined
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (`clEnqueueNDRangeKernel` -54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`.
- A35 Vulkan 2-row `ameva_mmqb` layouts slow (60-120 ms). S21 `mmqb_s21.txt` (watcher pid 22747) not read yet.
- S25 unreachable since about 01:10 on 2026-10-08 (connection timed out): final sources and `test-backend-ops` not run there. Conv on S21/A53, ggml-vulkan MUL_MAT time on S25/S20, int8 activations in the conv kernel, Wan VAE on a device: not measured.
- A35 throttles an idle ssh session (a compile sat at 1 % CPU until polled); one timing run right after a 12-minute compile read 364 / 289 / 354 ms.

### Current Problem & Focus
- Convolution is one of the decoder's ops. No decoded image exists yet: GROUP_NORM, UPSCALE, PAD, CONCAT, SOFT_MAX, f32 MUL_MAT for the middle attention and the bias ADD after each convolution still decide whether a VAE graph stays on ggml-ameva.
- Earlier decisions stand: default tile size 24 on 256 MiB devices and fail-fast on CPU split; exact mode first, int8 behind a switch after a PSNR check; full Wan VAE on S25-class only, `taew2_1` elsewhere.

### Relevant Files
- `adapters/llama/ggml-ameva/ggml-ameva.cpp` (`enc_conv_2d`, `ameva_conv2d_geometry`, `ameva_conv_f16_check`), `shaders/ameva_conv.comp`, `ameva_conv_w.comp`, `ameva_qdecode.glsl`, `scripts/build_ameva_kernels.py` (`VK_ONLY`)
- `tests/ameva_conv_check.cpp`, `scripts/termux/lab/conv_check.sh` (`kernels`, `SKIP_PLUGIN_BUILD=1`, `TIME_SHAPES`), `tests/ameva_conv_bench.cpp`, `scripts/termux/lab/conv_bench.sh`
- Devices: `~/abi-lab/src-conv`, `~/abi-lab/out-convk/` (plugin, `conv_check_run.log`, `tbo_conv2d.txt`), `~/abi-lab/out-conv/` (bench logs)

### Next Actions
1. VAE op census on ggml-ameva (`tests/ameva_graph_census.cpp` on the SD 1.x decoder graph), then K2 group norm, K3 ops and the bias fused into the convolution.
2. sd.cpp built against ggml-ameva (leejet ggml `8e800cef` headers, ABI probe); gate: decoded image against the f16 Vulkan image, decode time per 256 px tile, no scheduler split.
3. Re-run `conv_check.sh` and `test-backend-ops -o CONV_2D` on the S25 when it answers. Then STT -> BitNet -> Vision -> TTS. Keep this file under 40 lines.
