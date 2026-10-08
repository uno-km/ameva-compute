# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T14:40:00+09:00  
**Branch**: `perf/opencl-mali` (pushed: `f333149` E2E VAE decode verified, `2abb68a`, `d31bfba`).  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`): wins on Mali OpenCL only. Numbers: `docs/design/opencl-prompt-path.md`.
- **Stage 1 (Diffusion). Tables: `docs/design/diffusion-vae-conv.md` (1-2 A35 `sd-cli`, 7 kernel bench, 8 CONV_2D, 9 group norm and bias, 10 K3 and census, 11 sd.cpp decoded image).**
- CONV_2D, GROUP_NORM (fused weight, bias, SILU, convolution bias): SPIR-V only, declined on OpenCL, IM2COL declined. K3 (UPSCALE nearest, PAD, CONCAT, SOFT_MAX, dense f32 MUL_MAT): SPIR-V and OpenCL C. S21/A53 op tests and `ameva_vae_graph_check` (SD 1.x 226 nodes none declined with `--vae-conv-direct`; f32 RMS difference 4e-6): section 10.
- **Step 3 (2026-10-08, section 11): sd.cpp `50d6405` (ggml `8e800cef`) decodes on ggml-ameva, sd.cpp unchanged.** `sd-cli --backend "cpu,vae=AMEVA0" --vae-conv-direct`, plugin through `GGML_BACKEND_PATH`, built by `scripts/termux/lab/sd_vae_e2e.sh build` on the S25 (ggml shared, `GGML_BACKEND_DL`, `GGML_OPENMP=OFF`); the S21, A35, A53 ran the same binaries.
- Same latent in every run (text encoder and UNet on CPU, one seed; two CPU decodes give the same file). `anime.gguf`, 256x256. AMEVA image against the CPU image: S25 60.47 dB, S21 61.11 dB, A35 63.08 dB, A53 63.08 dB; largest difference 1 level of 255; two AMEVA decodes give the same file on each device. VAE graph 226 nodes, 249 dispatches, no scheduler split on all four.
- Decode time of the 256 px image, CPU / AMEVA (one process each, not alternated, GPU clock not recorded): S25 15.0 / 1.97 s (4 threads), S21 22.5 / 11.3 s (4), A35 73-77 / 10.3-10.6 s (3), A53 103-106 / 18.7-18.9 s (3).
- ABI: sd.cpp compiles ggml with `GGML_MAX_NAME=160` and `ggml_tensor.extra` lies behind `name`. `ameva_abi_check` compares the name size and the VAE op names; a plugin built without `-DGGML_MAX_NAME=160` is refused at load (S25 negative control). `build_ggml_ameva.sh --cxxflags --list-devices-cmd`.

### Known, Not Examined
- **f16 Vulkan reference image (the gate's named reference): not obtained.** `sd-cli --backend "cpu,vae=Vulkan0"` with ggml-vulkan `8e800cef` segfaults (139) after "vae compute buffer size" on the S25 and the S21, with and without `--vae-conv-direct`, with and without the local `ggml-vulkan.cpp` change; text encoder and UNet on `Vulkan0` run. Op not identified (no debugger). Backend and log: `~/abi-lab/out-sd/vk-crash` (S25, S21); put the library back into `sd-build/bin` and `sd_vae_e2e.sh run` adds the `Vulkan0` run. `ggml-vulkan.cpp` at `-O3` did not finish in 67 min on the S25; `-O1` takes 2-3 min.
- No f32-kernel image comparison: `--tensor-type-rules` has no effect on this GGUF in sd-cli `50d6405`. The 40 dB floor of the script is a choice, not a measured limit.
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (-54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`.
- Sections 9-10 op tests (`k3_check.sh`, `conv_check.sh`) not run on Adreno (S25, S20) or the A35; section 11 not on the S20 (held by another session). Not rerun after K3: `ameva_fuse_check`, `ameva_kvq_check`, `ameva_chunk_check`; LLM text parity.
- Declined by design so far: bilinear/bicubic UPSCALE, SOFT_MAX with ALiBi bias or sinks, tensors above one binding, CONV_3D, Wan RMS norm.

### Current Problem & Focus
- User Decision (2026-10-08): Full CUDA-Level Diffusion completion required before moving to STT. Must handle 512px, multi-binding (>256MB), VAE tiling (tile size 24), and UNet graph validation across models/quantizations (SD 1.x, SDXL, Z-Image-Turbo).
- Earlier decisions stand: Fail-fast on CPU splits; CPU image accepted as official reference; exact FP16 mode default.

### Relevant Files
- `adapters/llama/ggml-ameva/ggml-ameva.cpp`, `ggml-ameva-device.cpp`, `shaders/ameva_conv.comp`, `scripts/termux/lab/sd_vae_e2e.sh`
- Devices (S25, S21, A35, A53): `~/abi-lab/sd-build/bin`, `out-sd/`

### Next Actions
1. Multi-binding buffer split & 512x512 VAE tiling: ensure tensors exceeding 256 MiB maxStorageBufferRange never split to CPU; verify 512x512 decode on A35/S25.
2. UNet & MMDiT op census & quantisation verification (Q4_K, Q8_0) across SD 1.x, SDXL, and Turbo models.
3. LoRA merge compatibility & 4-device full E2E 512px image benchmark with PSNR check. Only then proceed to STT.
