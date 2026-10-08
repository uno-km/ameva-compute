# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T09:50:00+09:00  
**Branch**: `perf/opencl-mali` (pushed: `10d1ec9`). K2 group norm and the convolution bias are in the working tree, NOT committed.  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`): wins on Mali OpenCL only. Fleet defaults: A53 OpenCL MMQB 2x8, A35 Vulkan tile, S25/S20 MMQB off. Numbers: `docs/design/opencl-prompt-path.md`.
- **Stage 1 (Diffusion). Tables: `docs/design/diffusion-vae-conv.md` (1-2 A35 `sd-cli`, 7 kernel bench, 8 CONV_2D, 9 group norm and bias).**
- A35 VAE decode of one 256x256 image with the 2.7.9 bundle: default 432.74 s (two convolutions with a 288 MiB IM2COL matrix run on the CPU, unreported); `--vae-tile-size 24` 92.37 s; `--vae-conv-direct` 226.89 s. GPU part: MUL_MAT 97.1 % (1180 ms per 19.33 GFLOP convolution).
- CONV_2D in ggml-ameva (`109ba54`, section 8): `enc_conv_2d`, weights copied per node into the kernel's K order in the scratch buffer, SPIR-V-only kernels (`VK_ONLY`), declined on OpenCL, IM2COL declined, f16 decoder picked by a device check, ggml's planar layout kept. Bit-equal to an f32 CPU convolution with each driver's arithmetic. A35 271.7 / 287.6 / 271.3 ms per 19.33 GFLOP convolution (4.10-4.34x ggml-vulkan MUL_MAT), S20 158.8 / 163.8 / 182.4.
- **GROUP_NORM and convolution bias in ggml-ameva (2026-10-08, section 9, working tree)**: `enc_group_norm` = `ameva_cnorm_stat` (mean and 1/sqrt(var + eps) of every group into the scratch buffer, f32 partial sums) + `ameva_cnorm` (one dispatch per batch element). `ameva_find_chain` gives the in-place MUL (weight), ADD (bias), SILU after a norm, and the in-place bias ADD after a CONV_2D, to the first node's dispatch, passing over RESHAPE nodes (sd.cpp's graph form). `AMEVA_FUSE=0` runs every node alone. SPIR-V only, declined on OpenCL. Header regenerated on the S21 (glslang 16.3.0, spirv-val): 420 added lines, nothing else changed, OpenCL header identical.
- S21 (Mali-G78), S20 (Turnip Adreno 650), A53 (Mali-G68): `ameva_conv_check` 23/23 x 6 runs (bias fused, bit-equal to the reference plus bias), `ameva_cnorm_check` 14/14 fused and with `AMEVA_FUSE=0` (fused chain equal in bits to the separate nodes; against the CPU within 0.29 of the test's rounding bound), `test-backend-ops` GROUP_NORM 2/2, GROUP_NORM_MUL_ADD 2/2, CONV_2D 1568/1568, `ameva_qmm_check` 516/516. S21 and S20: fuse / kvq / chunk checks pass. S21 every op 3596/3596.
- Timing: fused norm chain 2.5-11.2 ms at decoder shapes on the S20, 3-6 % of the same-size convolution. S21 convolution 168.9-175.0 / 183.0-185.1 / 110.3-110.5 ms (first on that device).

### Known, Not Examined
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (`clEnqueueNDRangeKernel` -54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`.
- S25 (since 01:10) and A35 (09:00) unreachable on 2026-10-08, connection timed out: nothing of section 9 ran there, and the S25 never ran the final CONV_2D sources.
- Norm timing was not alternated and the four-node form uses separate tensors; the S21's GPU clock was still rising inside those graphs (one 89 ms reading against 6.6). A53 timing for section 9: not run.
- Tensors above one binding (conv, norm), int8 activations in the conv kernel, Wan VAE (CONV_3D, RMS norm) on a device: not built or not measured.

### Current Problem & Focus
- No decoded image exists yet. UPSCALE, PAD, CONCAT, SOFT_MAX and the f32 MUL_MAT of the middle attention (K3) still decide whether a VAE graph stays on ggml-ameva.
- Earlier decisions stand: default tile size 24 on 256 MiB devices and fail-fast on CPU split; exact mode first, int8 behind a switch after a PSNR check; full Wan VAE on S25-class only, `taew2_1` elsewhere.

### Relevant Files
- `adapters/llama/ggml-ameva/ggml-ameva.cpp` (`enc_conv_2d`, `enc_group_norm`, `ameva_find_chain`, `ameva_group_norm_geometry`), `shaders/ameva_conv.comp`, `ameva_conv_w.comp`, `ameva_cnorm_stat.comp`, `ameva_cnorm.comp`, `scripts/build_ameva_kernels.py` (`VK_ONLY`)
- `tests/ameva_conv_check.cpp`, `tests/ameva_cnorm_check.cpp`, `scripts/termux/lab/conv_check.sh` (`kernels`, `SKIP_PLUGIN_BUILD=1`, `TIME_SHAPES`, `CNORM_TIME_SHAPES`), `tests/ameva_conv_bench.cpp`
- Devices (S21, S20, A53): `~/abi-lab/src-cnorm`, `~/abi-lab/out-cnorm/` (plugin, `run.log`, `cnorm_check_*.log`, `tbo_*.txt`); section 8 logs in `~/abi-lab/out-convk/`

### Next Actions
1. Commit the working tree (K2, bias fusion, regenerated header, docs) once the user says so; stage by path.
2. VAE op census (`tests/ameva_graph_census.cpp` on the SD 1.x decoder graph), then K3: UPSCALE (nearest), PAD, CONCAT, SOFT_MAX, f32 MUL_MAT.
3. sd.cpp built against ggml-ameva (leejet ggml `8e800cef` headers, ABI probe); gate: decoded image against the f16 Vulkan image, decode time per 256 px tile, no scheduler split.
4. Re-run `conv_check.sh` and `test-backend-ops` on the S25 and A35 when they answer. Then STT -> BitNet -> Vision -> TTS. Keep this file under 40 lines.
