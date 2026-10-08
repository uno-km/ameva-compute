# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T17:10:00+09:00  
**Branch**: `perf/opencl-mali`, HEAD `1483ca8`. **This session's work is in the working tree, NOT committed** (files below).  
**Current Goal**: Diffusion to full completion before STT (user decision 2026-10-08); STRICTLY LOCAL under `dev/`.

### Verified (Ground Truth). Tables: `docs/design/diffusion-vae-conv.md` sections 1-12
- Earlier work stands: LLM / OpenCL (`docs/design/opencl-prompt-path.md`), VAE ops and the 256 px decode (sections 8-11).
- **Section 12 (2026-10-08). Tensors above one binding (256 MiB on Mali) run in windows of planes**: CONV_2D (passes over channel groups, `conv_*_acc` kernels, bit-equal to one pass), GROUP_NORM, UPSCALE, binary ops, SOFT_MAX, dense MUL_MAT. New ops: NORM, TIMESTEP_EMBEDDING, REPEAT. `AMEVA_STRICT=1` aborts on a declined op. The graph wait scales with the work (30 s fixed gave -405 on the A53).
- Op checks, **S21 only** (`conv_check.sh`): conv 31/31 (25/25 and 31/31 under `AMEVA_BINDING_LIMIT`), cnorm 15/15 x4, window 18/18, unet_ops 15/15, qmm 516/516.
- **512 px untiled VAE decode on AMEVA, no CPU split** (`sd_e2e.sh`, SD 1.x `anime.gguf`, 4 steps), decode s AMEVA / CPU: S25 7.71 / 82.54, S21 25.68 / 56.89, A35 42.45 / 333.02, A53 58.38 / 152.48. 62.66 dB, max 1 of 255, on all four. 5 window nodes on each Mali.
- **UNet + VAE on AMEVA, no split**, 512 px sampling / decode s (CPU): S25 21.00 / 7.42 (102.32 / 82.54); S21 90.88 / 21.57 (90.26 / 56.89); with `--mmap --diffusion-fa` A53 143.93 / 57.70 (135.46 / 147.03), A35 99.41 / 42.06 (330.01 / 328.05).
- **PSNR against the CPU image does not measure the backend with quantized weights**: ggml CPU rounds activations to 8 bits (CPU q4_0 vs CPU `--type f32`: 30.53 dB). Against the f32-weight CPU run AMEVA is 50.16 dB (4 steps), 52.80 dB (1 step). Raw 512 px PSNR vs the same-weight CPU: S25 40.69, S21 29.70, A53 33.33, A35 33.98 dB (A53 and A35: flash attention in both runs).
- S25 only: SDXL base Q4_0 512 px 21.35 / 9.59 s (CPU 114.84 / 63.09), 51.30 dB; SDXS 1 step 1.91 / 0.76 s (7.13 / 6.83), 36.69 dB; `--type` q8_0 39.98, q4_K 23.23, f16 44.29 dB; tiled decode 15.45 s (CPU 143.43), 63.04 dB.
- LoRA at run time on AMEVA: 192/192 tensors, no split; vs CPU 30.71 dB (q4_0), 45.84 dB (f16); the adapter moves the image by 19.3-19.8 dB on both.

### Known, Not Done
- **LoRA merged at load ("baking") is declined on AMEVA** (in-place ADD into an f16 weight; repacked weights are read-only). On the CPU it works with f16 weights and asserts in ggml with q4_0.
- **MMDiT not validated.** The Z-Image-Turbo census on the S21 was ended by the system during text encoding (5.4 GB of parameters); no op list. SDXL ran on the S25 only.
- A53: the UNet at 512 px without `--diffusion-fa` ended the Termux session (A35 and A53 UNet rows use `--mmap --diffusion-fa`). A35 and A53 dropped off the network for minutes during long runs. S21: UNet on AMEVA no faster than the CPU at 512 px (dense attention MUL_MAT; not profiled).
- `ameva_mmq` (int8 activations; default on S25, S20, A35) costs 8-14 dB against exact at one step; `AMEVA_MMQ=0` for diffusion is undecided. q4_K with f32 activations: 34.17 dB from exact at 4 steps (44.22 at 1), cause not separated from the CPU text encoder.
- Not rerun after these changes: `test-backend-ops`, LLM text parity, op checks on A35 / A53 / S25 / S20. New kernels are SPIR-V only. PAD, CONCAT, NORM, RMS_NORM, FLASH_ATTN_EXT above one binding are declined.
- Unchanged from before: Vulkan0 VAE segfault (no f16 Vulkan reference image), Llama-3.2-1B text departs at byte 70, S20 not covered.

### Relevant Files (uncommitted)
- `adapters/llama/ggml-ameva/`: `ggml-ameva.cpp`, `ggml-ameva-device.cpp`, `ggml-ameva.h`, `ggml_ameva_kernels_spv.h` (generated on the S21, spirv-val), `shaders/ameva_conv.comp`, `ameva_conv_w.comp`, new `ameva_norm.comp`, `ameva_timestep_embd.comp`
- `scripts/build_ameva_kernels.py`, `scripts/termux/lab/conv_check.sh`, new `sd_e2e.sh`, `make_test_lora.py`; tests `ameva_conv_check`, `ameva_cnorm_check`, new `ameva_window_check`, `ameva_unet_ops_check`; `docs/design/diffusion-vae-conv.md`
- Devices: sources `~/abi-lab/src-win5` (S25, S21); plugin `out-sdw5` (S25), `out-sdw4` (S21, A35, A53). Added on the S25: `sdxl-base-1.0-Q4_0.gguf` (3.94 GB), `anime-q4_K.gguf` (1.63 GB), `~/abi-lab/lora-test`.

### Next Actions
1. Commit the working tree (awaiting the user); rerun `test-backend-ops` and LLM parity; op checks on the A35, A53, S25.
2. LoRA merge on AMEVA (f16 binary ADD and a write path for repacked weights).
3. MMDiT census on the S25 (model has to be copied there), attention kernel for the Mali UNet, the `AMEVA_MMQ` default for diffusion.
