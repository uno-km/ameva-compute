# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T11:20:00+09:00  
**Branch**: `perf/opencl-mali` (pushed up to `34440b6`; K3 and the VAE census are in the working tree, not committed).  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`): wins on Mali OpenCL only. Numbers: `docs/design/opencl-prompt-path.md`.
- **Stage 1 (Diffusion). Tables: `docs/design/diffusion-vae-conv.md` (1-2 A35 `sd-cli`, 7 kernel bench, 8 CONV_2D, 9 group norm and bias, 10 K3 and census).**
- CONV_2D (`109ba54`) and GROUP_NORM with the fused weight, bias, SILU and convolution bias (`c93f05b`): SPIR-V only, declined on OpenCL, IM2COL declined. S21, S20, A53 results in sections 8-9.
- **K3 (2026-10-08, section 10, working tree)**: UPSCALE nearest (`ameva_upscale`), PAD zeros and circular (`ameva_pad`), CONCAT f32/f16 (two `ameva_cpy` dispatches), SOFT_MAX with scale, mask, in place (`ameva_soft_max`), f32 x f32 MUL_MAT of any shape (`ameva_mm` where `ameva_qmm` does not apply: K not a multiple of 32, several matrices). Op kernels: SPIR-V and OpenCL C. Header regenerated on the S21 (942 added SPIR-V lines, none changed).
- S21 (Mali-G78 Vulkan), A53 (Mali-G68 Vulkan and OpenCL): `ameva_k3_check` 39/39 (UPSCALE, PAD, CONCAT bit-equal to the CPU form); `test-backend-ops` UPSCALE 4/4, PAD 15/15, CONCAT 16/16, SOFT_MAX 65/65, MUL_MAT 371/371, on Vulkan also CONV_2D 1568/1568 and GROUP_NORM 2/2. S21: every op 3885/3885, `conv_check.sh` 23/23 x 6, 14/14 x 2, `ameva_qmm_check` 516/516.
- **VAE census (`tests/ameva_vae_graph_check`, graphs rebuilt from sd.cpp `50d6405`)**: SD 1.x decoder with `--vae-conv-direct` 226 compute nodes, 12 signatures, none declined; Flux-type (z = 16) none of 224; TAESD none of 114. sd.cpp's default form: 80 declined (40 IM2COL, 40 f16 MUL_MAT), so the wrapper must pass `--vae-conv-direct`.
- Whole SD 1.x decoder on ggml-ameva against the CPU backend with the same kernels as f32: RMS difference 3.9e-6 to 4.1e-6 of the reference RMS (S21 16x16 and 32x32 latent, A53 16x16); TAESD 7.2e-7 to 2.1e-6. Against the CPU with f16 kernels 1e-2: ggml's CPU convolution rounds activations to f16 there.
- One `graph_compute` of that decoder (random weights, one run, not alternated): S21 32x32 latent 7.07 s (CPU 4 threads 20.4 s f32 / 11.9 s f16 kernels), S21 16x16 1.18 and 2.71 s, A53 16x16 5.13 s (CPU 70.3 / 29.9 s). S21 `sd-cli` (ggml-vulkan, real weights, `--vae-conv-direct`) 133.96 s for a 256x256 decode, another program.

### Known, Not Examined
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (-54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`. S21 `sd-cli` is static and prints no scheduler splits (no census from it).
- S25 and A35 unreachable on 2026-10-08 (connection timed out); S20 held by another session's benchmark: nothing of section 10 ran on Adreno, nothing of sections 9-10 on the S25 or A35.
- Not rerun after K3: `ameva_fuse_check`, `ameva_kvq_check`, `ameva_chunk_check`; LLM text parity. Timings above are single runs.
- Declined by design so far: bilinear/bicubic UPSCALE, SOFT_MAX with ALiBi bias or sinks, tensors above one binding, CONV_3D, Wan RMS norm.

### Current Problem & Focus
- No decoded image exists yet: sd.cpp has not been built against ggml-ameva (step 3).
- Earlier decisions stand: default tile size 24 on 256 MiB devices and fail-fast on CPU split; exact mode first, int8 behind a switch after a PSNR check; full Wan VAE on S25-class only, `taew2_1` elsewhere.

### Relevant Files
- `adapters/llama/ggml-ameva/ggml-ameva.cpp` (`enc_upscale`, `enc_pad`, `enc_concat`, `enc_soft_max`, `enc_mul_mat_dense`, `mul_mat_dense_reason`), `shaders/ameva_upscale.comp`, `ameva_pad.comp`, `ameva_soft_max.comp`, `ameva_mm.comp`, `scripts/build_ameva_kernels.py` (`OPS`)
- `tests/ameva_k3_check.cpp`, `tests/ameva_vae_graph_check.cpp` (`AMEVA_VAE_CHECK_LATENT`, `AMEVA_VAE_CHECK_ALL`), `scripts/termux/lab/k3_check.sh` (`kernels`, `BACKENDS`, `TBO_OPS`, `LATENT`), `scripts/termux/lab/conv_check.sh`
- Devices (S21, A53): `~/abi-lab/src-k3`, `~/abi-lab/out-k3/` (plugin, `k3_check_*.log`, `vae_graph_check.log`, `tbo_*.txt`), run logs `~/abi-lab/k3_run*.log`, `k3_conv.log`

### Next Actions
1. Commit the working tree (K3 kernels, regenerated headers, tests, script, docs) once the user says so; stage by path.
2. sd.cpp built against ggml-ameva (leejet ggml `8e800cef` headers, ABI probe, `--vae-conv-direct`); gate: decoded image against the f16 Vulkan image, decode time per 256 px tile, no scheduler split.
3. `k3_check.sh` and `conv_check.sh` on the S25, S20 and A35 when they are free. Then STT -> BitNet -> Vision -> TTS. Keep this file under 40 lines.
