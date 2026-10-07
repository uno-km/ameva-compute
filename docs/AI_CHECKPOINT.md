# Current Engineering Checkpoint

**Timestamp**: 2026-10-07T21:58:00+09:00  
**Branch**: `perf/opencl-mali` (uncommitted: this file, `docs/design/diffusion-vae-conv.md`)  
**Current Goal**: 5-Modality Pipeline Optimization (1. Diffusion -> 2. STT -> 3. BitNet -> 4. Vision -> 5. TTS); STRICTLY LOCAL under `dev/` (no remote branch fetch).

### Verified (Ground Truth)
- [PASS] Zero-Source Leakage Audit, Engine Isolation (`adapters/`), A53 clean-slate auto-provisioning, 5-modality real-device inference on A53, `ac_security_guard` (`AMEVA-SEC-001`). Details: git log up to `09e0186`.
- [PASS] Block-integer MUL_MAT (`ameva_mmqb`, opt-in `AMEVA_MMQB`, `29b5362` + 2-row fix): `ameva_qmm_check` 516/516 on A53 OpenCL 9/9 and A35 18/18. Wins on Mali OpenCL (A53 Llama-3.2-1B prompt 46 -> 58 t/s, text equal to CPU), loses on every Vulkan driver and Adreno OpenCL. `ameva_mmvqb.comp` written, not built in, no gain on A53.
- Fleet defaults: A53 OpenCL MMQB 2x8, A35 Vulkan tile, S25/S20 MMQB off. Numbers: `docs/design/opencl-prompt-path.md`, git log.
- **Stage 1 (Diffusion), 2026-10-07, A35 Mali-G68 Vulkan, bundle 2.7.9 `sd-cli`, SD 1.x, ssh cores 0-2. Design and tables: `docs/design/diffusion-vae-conv.md`.**
- VAE decode of one 256x256 image: default path 432.74 s; 192x192 23.55 s; `--vae-tiling --vae-tile-size 24` 92.37 s (4 tiles); `--vae-conv-direct` 226.89 s.
- Cause of the 432 s: two convolutions per default tile (`up.1.upsample.conv`, `up.0.block.0.conv1`) have a 288 MiB IM2COL matrix, above `maxStorageBufferRange` (256 MiB); ggml-vulkan declines them and the ggml scheduler runs them on the CPU (`GGML_SCHED_DEBUG=2`: SPLIT #1 and #3 CPU; 4 threads busy, `gpu_busy` 0 %). Not logged at the default level. The wrapper passes `--vae-tiling` with no tile size, so every default tile has it.
- GPU part: MUL_MAT is 97.1 % of kernel time at 12.2-16.8 GFLOPS; direct CONV_2D runs at 2.75-2.80 GFLOPS.
- Same products (19.33 GFLOP, 3 shapes) through AMEVA (`ameva_mm_bench`, `out-m2`): `ameva_qmm` f16 1263-1558 ms (no gain over ggml-vulkan 1180), `ameva_qgemm` q8_0 x f32 234-247 ms (4.8-5.0x), `ameva_mmq` q8_0 x int8 101-118 ms (10-11.7x, 2 % of rms per product).
- Images vs the default run: direct 59.45 dB, 24-latent tiles 49.92 dB.
- Wan 2.1 VAE (code reading, not run): `(B,C,T,H,W)` is ggml `[W,H,T,B*C]` with `B == 1` asserted; per-channel broadcast already works in that layout; cost is PERMUTE+CONT around RMS_NORM (5 passes) and the 6-node CausalConv3d with a 27x column matrix. 13.5 TFLOP per 480x832 latent frame (calculated).

### Known, Not Examined
- Llama-3.2-1B 64-token text departs from the CPU's at byte 70 (A53 OpenCL, A35 Vulkan and OpenCL); S25 OpenCL `ameva_qmm_check` 510/516 (q2_K M=1); S20 OpenCL `ameva_qmm_check` exits 139.
- S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet (`clEnqueueNDRangeKernel` -54); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()` with `GGML_VK_PERF_LOGGER`.
- A35 Vulkan 2-row `ameva_mmqb` layouts slow (60-120 ms). S21 `mmqb_s21.txt` (watcher pid 22747) not read yet.
- Stage 1 unmeasured: S25/S20/S21/A53 VAE decode, image effect of int8 activations, Wan VAE on a device (no weights on the fleet, only `taew2_1`), CPU split time with all cores.

### Current Problem & Focus
- Stage 1: convolution without column matrix (`ameva_conv`), channel-axis norm with broadcast (`ameva_cnorm`), VAE op census.
- User decisions APPROVED (2026-10-07): (1) Default tile size 24 on 256 MiB devices & fail-fast on CPU split; (2) Exact mode first, int8 behind switch after PSNR check; (3) Full Wan VAE on S25-class only, `taew2_1` for others.

### Relevant Files
- `docs/design/diffusion-vae-conv.md`, `adapters/llama/ggml-ameva/shaders/ameva_qgemm.comp`, `ameva_mmq.comp`, `ameva_binary.comp`, `tests/ameva_mm_bench.cpp`
- `dev/termux/termux-diffusion/thirdparty/stable-diffusion.cpp/src/model/vae/wan_vae.hpp`, `src/core/ggml_extend.hpp`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- A35 `~/claude_vae_prof/` (logs, images), A35 `~/abi-lab/out-m2` (`LD_LIBRARY_PATH=$B/ggml-rt:$B`)

### Next Actions
1. `ameva_conv_bench`: tile kernel staging from the source tensor against a materialised matrix, 3 shapes + one 3x3x3, A35 then S25/S20; every output compared with an f32 reference.
2. K1 2D + group norm + remaining ops in ggml-ameva; sd.cpp built with ggml-ameva (ggml `8e800cef`); gates in the design, section 6.
3. Then STT -> BitNet -> Vision -> TTS. Keep this file under 40 lines.
