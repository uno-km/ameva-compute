# Current Engineering Checkpoint

**Timestamp**: 2026-10-10T00:00:00+09:00
**Branch**: `perf/opencl-mali` (origin at a1d4a12; ebccab6, 61896f7 and this checkpoint not pushed)
**Goal**: Stage 4, vision encoder (SigLIP, llama.cpp mtmd / clip.cpp) on ggml-ameva (Stages 1-3 closed).

### Completed
- `ebccab6`: ameva_flash_attn RPI rows per invocation + DUNROLL; host picks the d64 prompt tile (64,32,16,2) when the tile rows fit, (64,16,32,1) for token generation. `AMEVA_FA_VARIANT` / `AMEVA_FA_VARIANTS` for experiments.
- `61896f7`: tests/ameva_vision_parity.cpp (CPU vs AMEVA0 through libmtmd; F32-dequantized reference, gate 1.5x the CPU Q8_0 floor; no-fallback, op, GPU-work and mapped-library checks; caption), lab scripts, docs/design/vision-siglip-ameva.md.
- Census (SmolVLM-500M mmproj, SigLIP-base + Idefics3): 394 nodes, 1 split, 0 unsupported ops on AMEVA0. Patch embedding is IM2COL + MUL_MAT (not CONV_2D); projections are Q8_0 (MMQ), so `ameva_conv` / `mm_f16a` are not reached.

### Verified (test-1.jpeg, 64 tokens, encode ms per image AMEVA vs CPU Q8_0, floor ratio)
- S25 Vulkan: 481-516 vs 1930-2893, 1.17, caption identical; MMQ off 833-898, 0.03.
- S20 Vulkan: 2778-2814 vs 9669-9896, 1.05. S21 Vulkan 2572-2589 vs 1936-1974 (slower), 0.03; S21 OpenCL 1784-1805, 1.10.
- A35 Vulkan 4941-5032 / OpenCL 3363-3367; A53 Vulkan 5311-5349 / OpenCL 4295-4306 (CPU 27-31 s on ssh cores 0-2 only).
- FA ms per SigLIP layer, previous vs new tile: S25 38.3 -> 22.3, S20 135.1 -> 112.9, S21 126.8 -> 126.5, A53 254.0 -> 163.6.
- test-backend-ops FLASH_ATTN_EXT 1048/1048 (S25, S20, S21); forced d64 variants 224/224 each (S25).
- Qwen2.5-0.5B prompt 368 tokens (S25): 1194-1220 t/s vs 1108-1123 with the previous tile, same text.

### Remaining
1. Mali FA 25-40 GFLOP/s (S21 126 ms/layer, half the encode): Mali tile or OpenCL-specific kernel.
2. termux-vision wiring: `GGML_BACKEND_PATH`, `MTMD_BACKEND_DEVICE=AMEVA0`, drop `--no-mmproj-offload`; bundle with this plugin.
3. Per-device FA choice ((64,64,16,2) is 11 % faster on Adreno 830 only); DMAX 128 prompt tile unmeasured.
4. Moondream2 not on the fleet (no downloads); SmolVLM's SigLIP used. Push of the branch awaits approval.

### Commands
- S25 build tree ~/abi-lab/src-vis, out ~/abi-lab/out-vis: `bash vis_rebuild.sh` (kernels + plugin), `bash vis_tool.sh` (tool).
- Run: `bash vision_parity_run.sh TAG --reps 3 --caption 32 --ref-mmproj ~/abi-lab/vis-models/mmproj-smolvlm-500m-f32.gguf` (`AMEVA_BACKEND=opencl` for Mali OpenCL, `AMEVA_PROFILE=2` for per-op GPU time).
- Sweep: `bash vision_fa_sweep.sh P "0 1 2"` with an `AMEVA_FA_VARIANTS=` build. Fleet copies: ~/abi-lab/out-vis, vis-pkg, vis-models.
- Library order: `LD_LIBRARY_PATH=out:<bundle>/lib` (Termux clang puts $PREFIX/lib before -rpath; A35 has an older ggml there).

### Known failures
- None open in the verified scope. A35 ssh resets once during the sweeps; its jobs continued under nohup.
