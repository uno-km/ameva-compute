# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T21:10:00+09:00  
**Branch**: `perf/opencl-mali` (pushed: `177193d` MMDiT Z-Image-Turbo, binary_f16, GPU LoRA merge, `d10fb5d`).  
**Current Goal**: Diffusion to full completion before STT (user decision 2026-10-08); STRICTLY LOCAL under `dev/`.  
**User order 2026-10-08**: no model downloads on the S25, no new model downloads on the A35 / S21 (data cost).

### Verified (Ground Truth). Tables: `docs/design/diffusion-vae-conv.md` sections 1-13
- Earlier work stands: LLM / OpenCL, VAE ops, tensors above one binding, SD 1.x UNet + VAE at 512 px on four devices, SDXL and SDXS on the S25, LoRA at run time (sections 8-12).
- **Section 13. MMDiT (Z-Image-Turbo Q2_K, Qwen3-4B text encoder on the CPU) runs on AMEVA with nothing declined and no split.** `--mmap --backend te=cpu,diffusion=AMEVA0,vae=AMEVA0 --max-vram AMEVA0=1.5 --stream-layers --params-backend diffusion=cpu` (`scripts/termux/lab/mmdit_census.sh`, memory guard).
  - A35 256 px 1 step: sampling / decode 79.92 / 11.67 s (CPU 372.72 / 119.26), 40.86 dB against the CPU image. S21 256 px 1 step, `AMEVA_STRICT=1`: 55.63 / 8.90 s (CPU 83.47 / 18.78), 39.55 dB. S21 512 px 8 steps, strict: 1994.08 / 36.37 s, image matches the prompt, no CPU reference. Not benchmark conditions.
  - Ops on AMEVA: ADD MUL REPEAT CONCAT NORM RMS_NORM GROUP_NORM MUL_MAT SCALE CONT SOFT_MAX CONV_2D UPSCALE TIMESTEP_EMBEDDING UNARY GLU. A split of this graph crashes in ggml's CPU REPEAT (seen with the stale `out-sdw4` plugin).
  - Lowest MemAvailable: A35 692 MB (DiT), S21 1181 MB (512 px). No session ended.
- **Text-encoder isolation as ordered does not exist.** termux-diffusion 2.0.1 `--isolate-text-encoder` is parsed and never acted on; sd-cli cannot save or load a condition. In process the 1483.75 MB embedding buffer stays until the image is done (zram on the A35); `--params-backend te=disk` frees it but needs 2.77 GB anonymous (guard-killed on the A35 at 14 s).
- **LoRA merged at load runs on AMEVA for f16 weights.** New op kernel `binary_f16` (`ameva_binary.comp -DF16`, ADD only); the f32 kernel's SPIR-V is unchanged. `ameva_unet_ops_check` 24/24 (9 f16 ADD cases bit-equal) on the S21 and A35; `test-backend-ops -o ADD` on the S21 passes 32 f16 cases.
  - `lora_merge_e2e.sh` (SD 1.x `--type f16`, 256 px, 4 steps, strict), S25 and S21: merged AMEVA vs CPU 45.75 dB, vs no adapter 19.72 dB, vs at run time 62.20 dB.
- Op checks with this source (`conv_check.sh kernels`), S21 and A35: conv 31/31 (25/25, 31/31 under limits), cnorm 15/15 x4, window 18/18, qmm 516/516.

### Known, Not Done
- Z-Image-Turbo: no CPU reference at 512 px; not on the S25 (no model there) or A53; Q4_0 file not run; text encoder on AMEVA not tried. 249 s per step at 512 px on the S21: every step re-sends and repacks the weights (25187 MiB in 8 steps).
- LoRA merge: quantized weights not possible (repacked weights are read-only; sd.cpp's own CPU path asserts on q4_0); end to end not run on the A35 (f16 CPU reference thrashed 20 min, stopped) or A53; `binary_f16` on OpenCL translated, not run.
- `test-backend-ops` on the S21 with this source (`OUT=out-win6`): default settings 3969/3969. The `AMEVA_REPACK_ALL=1` pass was started at 21:26 and not read (`~/abi-lab/results/tbo_repack_all.log`, ends with `TBO_DONE`). LLM text parity not rerun. Op checks not rerun on the S25, A53, S20.
- Unchanged: `AMEVA_MMQ` default for diffusion, Vulkan0 VAE segfault, S21 UNet time at 512 px, S20 not covered, Llama-3.2-1B text departs at byte 70.

### Relevant Files (uncommitted)
- `adapters/llama/ggml-ameva/ggml-ameva.cpp`, `shaders/ameva_binary.comp`, `ggml_ameva_kernels_spv.h` / `_cl.h` (regenerated on the S21), `scripts/build_ameva_kernels.py`, `tests/ameva_unet_ops_check.cpp`
- `scripts/termux/lab/mmdit_census.sh`, `lora_merge_e2e.sh` (new), `sd_e2e.sh` (`GUARD_MB`), `docs/design/diffusion-vae-conv.md`
- Devices: source `~/abi-lab/src-win6` (S25, S21, A35); sd plugin `out-sdw6` (`17e6c72e296e`, built on the S25; S21, A35 copies); b8471 plugin and tests `out-win6` (S21, A35). `out-sdw4` is stale.

### Next Actions
1. Commit (awaiting the user). Read the S21 `AMEVA_REPACK_ALL` `test-backend-ops` result; rerun LLM text parity.
2. User decision: is Stage 1 closed with the "Not Done" list, or is the two-process text-encoder isolation (an sd-cli change in termux-diffusion) wanted first?
3. If continued: keep streamed weights repacked between steps (the 512 px cost), CPU reference at 512 px, `AMEVA_MMQ` default for diffusion.
