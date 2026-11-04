## AMUDA Android SDK — unreleased changes (AmevaDiffusion)

Written 2026-11-03 while integrating the SDK into AmfyUI (`dev/app/amfyui`). Not released, not tagged.

### C API (`include/ameva_sd_api.h`)

- `ameva_sd_load_ex(AmevaSdLoadOptions*)`: multi-file models (`diffusion_model_path`, `llm_path`,
  `clip_l_path`, `t5xxl_path`, `vae_path`), TAESD decoder (`taesd_path`), per-component placement
  (`text_encoder_placement`, `vae_placement`), diffusion-weight residency (`AMEVA_SD_PARAMS_DISK` = sd.cpp
  params backend `disk`), `enable_mmap`. `struct_size` is checked; a mismatch returns
  `AMEVA_SD_ERR_ABI_MISMATCH`. `GPU_ONLY` now sums all given files for its budget check. `CPU_ONLY` no
  longer loads the GPU plugin.
- `ameva_sd_generate(AmevaSdGenerateParams*)`: negative prompt, sampler and scheduler by sd.cpp name
  (unknown name: `AMEVA_SD_ERR_INVALID_ARGUMENT`), flow shift, img2img (`init_rgb` + `strength`),
  VAE tiling, per-unit progress callback, duty cycle (`duty_cycle_percent`: idle (100-d)/d of each
  progress unit's measured time, capped at 5 s per unit).
- `ameva_sd_cancel()`: wraps `sd_cancel_generation(SD_CANCEL_ALL)`; the running generate returns
  `AMEVA_SD_ERR_CANCELLED`.
- One generation per process at a time (sd.cpp's progress hook is process-global); the old
  `ameva_sd_txt2img*` also take that lock now.
- Fix: `ggml_backend_load()` was called on every load, appending a duplicate AMEVA0 device to ggml's
  process-global registry each time (no unregister exists). The plugin is now registered once per process;
  a different plugin path afterwards is refused.
- `free_sd_images()` is used instead of per-image `free()`.

### C API, second round (same day)

- `AmevaSdLoadOptions`: `clip_vision_path`, `ip_adapter_path`, `control_net_path`, `motion_module_path`,
  `photo_maker_path`. `ip_adapter_path` without `clip_vision_path` is refused.
- `AmevaSdGenerateParams`: LoRA list (`lora_paths`, `lora_multipliers`, `lora_count`; a missing file is
  refused), `clip_skip`, ControlNet hint (`control_rgb`, `control_strength`, `control_canny` runs sd.cpp's
  Canny with the sd-cli defaults), IP-Adapter reference (`ip_adapter_rgb` of any size,
  `ip_adapter_strength`), hires fix (`hires_scale` (1, 4], `hires_steps`, `hires_denoise`,
  `hires_upscaler`; the output buffer must match the scaled size).
- `ameva_sd_generate_video()`: AnimateDiff (motion module) frames, 1..64.
- `ameva_sd_upscaler_load/_factor/_upscale/_free`: ESRGAN upscaler on the AMEVA plugin or the CPU.
- JNI: `nativeGenerate` now reads the Kotlin `GenerateParams` object's fields directly instead of 16
  positional arguments; `nativeLoadEx` takes a 12-entry path array.

### Kotlin (`com.ameva.amuda.diffusion.AmevaDiffusion`)

- Second round: `LoadOptions.clipVisionPath/ipAdapterPath/controlNetPath/motionModulePath/photoMakerPath`;
  `GenerateParams.loraPaths/loraMultipliers/clipSkip/controlRgb/controlStrength/controlCanny/
  ipAdapterRgb/ipAdapterWidth/ipAdapterHeight/ipAdapterStrength/hiresScale/hiresSteps/hiresDenoise/
  hiresUpscaler`; `generateVideo(params, frames, fps, listener)`; new class `AmevaUpscaler`.

- `load(context, LoadOptions)`, `generate(GenerateParams, ProgressListener?) : ByteArray` (RGB8),
  `cancel()`, `rgbToBitmap()`, `bitmapToRgb()`. `LoadOptions`/`GenerateParams` are mutable classes with
  setters so Java callers can use them without Kotlin default arguments.
- A `ProgressListener` exception cancels the generation and is rethrown from `generate()`.
- `close()` cancels a running generation, waits for it, then frees; it is idempotent. Calls after
  `close()` throw `IllegalStateException` instead of using a freed handle.

### Verified

- `gradle assembleRelease` on Windows (AGP 8.5.2, NDK 26.3): builds; `libameva_sd_jni.so` exports
  `nativeLoadEx`, `nativeGenerate`, `nativeCancel`.
- In AmfyUI on an x86_64 Android 17 emulator with ARM translation (`libndk_translation`): the four
  libraries load and the JNI bindings resolve. GPU behaviour under translation is not representative and
  is not counted as a device result.

### Not verified

- Any Galaxy device (fleet offline during this work). img2img, cancel, duty cycle and the disk residency
  path have not run on hardware.

### Performance fix: Conv2d direct (2026-11-03, measured on the fleet)

- `ameva_sd_load_ex` / `ameva_sd_load` now set sd.cpp `diffusion_conv_direct` (and `vae_conv_direct` when the
  VAE is on the GPU) on the GPU path, and both on the CPU path. Before, every convolution was lowered to
  IM2COL + MUL_MAT; at 512x512 those IM2COL tensors exceed a storage-buffer binding, ggml-ameva declines them
  and the graph is split between GPU and CPU.
- S25 (Adreno 830, Vulkan), SDXS 512x512, 1 step, in-app: generate 42.6–43.6 s → 1.76–1.94 s
  (sampling 26.5 s → 1.1–1.3 s, TAESD decode 16.3 s → 0.57–0.6 s). Output equal to the CPU path within
  PSNR 28.8 dB (same image by inspection).
- conv-direct is enabled only when the device accepts a probe CONV_2D node (`ggml_backend_dev_supports_op`):
  ggml-ameva's OpenCL backend has no convolution kernel, and conv-direct there failed every SD generation on
  A35/A53/S21 (CONV_2D on the CPU, then FAIL-FAST "tensor is not in an AMEVA buffer" on a split view ADD).
  The OpenCL backend also lacks GROUP_NORM and TIMESTEP_EMBEDDING; SD on OpenCL remains unsupported.

### GPU placement for `GPU_THEN_CPU` (2026-11-03)

- sd.cpp auto-fit reserves a fixed 2 GiB of compute memory per DiT/conditioner. With phone budgets of
  1.3–2.0 GiB it put SDXS's 488 MB UNet on the CPU (A35/A53/S21: `auto-fit: --backend diffusion=cpu`), so a
  "GPU" request ran its sampling on the CPU.
- The SDK now decides first (budget = free x fraction - 512 MiB, compute reserve 384 MiB, graph-cut via
  `max_vram` in every tier) and logs `GPU_THEN_CPU: ... -> <decision>`:
  1. all files + reserve fit: explicit all-GPU placement;
  2. the largest phase fits (UNet + ControlNet/IP-Adapter/motion module, text encoder, CLIP-Vision, VAE; single-file
     checkpoints split by their GGUF tensor table): GPU time-share (`params_backend` disk for each module);
  3. otherwise sd.cpp auto-fit.
- Measured, SDXS 512x512 1 step, warm GPU vs CPU in-app: S25 1.87 s / 11.4 s, S21 5.1 s / 32.7 s,
  A35 10.7 s / 30.2 s, A53 15.8 s / 38.2 s, S20 12.3 s (cold) / 42.5 s.
- Explicit placements (tiers 1 and 2) pass graph-cut a positive `max_vram` capped at 1.5 GiB per cut graph; the
  time-share tier uses a 256 MiB reserve. Auto-fit keeps the negative "free minus reserve" form.
- Diffusion flash attention is enabled when the device accepts FLASH_ATTN_EXT for head sizes 40/64/80/128/160
  (f16 K/V). SD1.5 512x512: UNet compute buffer 1057 MB -> 100 MB; S25 sampling 23.0 s -> 15.7 s
  (Anything q8, 4 steps); output PSNR 39.2 dB against the non-flash path. Without it, S20 (Adreno 650) allocated
  the 1.07 GB attention buffer once and failed to allocate it on every later generation.

### New C API: `ameva_sd_placement()` (Kotlin `AmevaDiffusion.placement`, `diffusionOnCpu`)

- Returns stable-diffusion.cpp's load report ("total params memory size = ... diffusion_model ...MB(RAM|VRAM)").
  When `GPU_THEN_CPU` ends in auto-fit and the diffusion model lands in RAM, the sampling runs on the CPU;
  callers must show that instead of reporting a GPU run (AmfyUI shows it on the result).

### `GPU_THEN_CPU` fourth tier: the diffusion model streamed to the GPU (2026-11-03, second round)

- When neither all files nor the largest phase fit, the SDK no longer falls through to sd.cpp auto-fit (which put the
  whole SD1.5 UNet on the CPU of A35/A53: Anything q8 + IP-Adapter phase 1491 MiB against budgets of 1372-1456 MiB).
  New tier: diffusion params backend `CPU` (weights mmap'd in RAM), compute on AMEVA0, sd.cpp `stream_layers`
  (graph-cut segments resident while the budget allows, the others uploaded per evaluation). Text encoder, VAE,
  CLIP-Vision and ControlNet time-share the GPU when each fits by itself, otherwise run on the CPU (CLIP-Vision
  ViT-H, 1206 MiB, on A35/A53-sized budgets). Requires a single-file UNet checkpoint and a budget of at least 1 GiB.
  Logged as `GPU_THEN_CPU: ... -> diffusion streamed to the GPU (weights in RAM), graph-cut` and
  `GPU_THEN_CPU stream: te GPU, vae GPU, clip_vision CPU`.
- `ameva_sd_placement()` prefixes `STREAM diffusion=GPU; ` in that tier (the load report lists the diffusion weights
  as RAM); Kotlin `diffusionOnCpu` is false there and the new `diffusionStreamed` is true.
- Measured on S21 (Mali-G78) with the budget fraction lowered to reproduce 1322 MiB, Anything q8 + IP-Adapter
  512x512, 4 steps: the UNet ran on the GPU; sampling within 7 % of the time-share tier at equal temperature
  (70.3 s cold; 147.1 against 137.9 s warm). Not yet run on A35/A53 themselves (both offline during this work).

### Thermal note for every measurement on the fleet

- S21 halves its GPU throughput from about 39 degC (AP sensor): sampling and VAE decode both double (70 -> 147 s,
  17 -> 36 s). A/B results are only comparable from the same cooled state (AmfyUI `scripts/fleet/cool_ab.sh` waits
  below a set AP temperature before every run).

### Diffusion flash attention on Mali-G78 (measured)

- S21, Anything q8 512x512, 4 steps, each run started below 36.5 degC: sampling 58.8 / 68.8 s with flash attention
  against 83.4 / 83.4 s without (`AMEVA_DECLINE=FLASH_ATTN_EXT`). Kept on. Mali-G68 not measured.

### ggml-ameva: profile report reachable from a host

- `ggml_backend_ameva_profile_report()` (also via `ggml_backend_reg_get_proc_address`). An Android app process
  does not run atexit handlers on exit, so the AMEVA_PROFILE totals never appeared; `ameva_sd_generate` now prints
  them after every generation when `AMEVA_PROFILE` is set. S20 (Adreno 650), Anything q8 512x512 4 steps, one
  generation: CONV_2D 63.6 % of GPU time, FLASH_ATTN_EXT 20.4 %, MUL_MAT about 13 %.

### ggml-ameva: BM64 convolution tile

- `ameva_conv.comp` `BM64`: 64 output positions per workgroup (4 x 8 sums per invocation). 12 new SPIR-V-only
  kernels (f16 exact / f16 built-in / f32 weights, vec4 / scalar staging, with and without ACCUM), appended to the
  enum; the existing conv kernels share one shared-memory array now (their SPIR-V changed, results unchanged).
- `tests/ameva_conv_bench.cpp` (3x3, f16, GFLOPS conv_v -> conv_v64): Adreno 830 413-420 -> 478-494, Mali-G78
  153-162 -> 167-178, Turnip Adreno 650 96-113 -> 15-16 (7x slower).
- Used for convolutions with rows of at least 128 positions (the VAE decoder) on "Adreno (TM) 8xx" and "Mali-G7x"
  only (`AMEVA_CONV_TILE=auto|64|32`). At 64 positions (the SD1.5 UNet) S25 sampling was 4 % slower, so the UNet
  keeps the 32-position tile. In-app S25 Anything 512x512: VAE decode 7.5-7.7 -> 6.8-7.1 s, sampling unchanged,
  image identical bit for bit. Mali-G68 (A35/A53) stays on the 32 tile until measured.
- `tests/ameva_conv_check.cpp`: two cases with rows of 130 and 160 positions (BM64 with a partial tile, in
  channel-group windows and whole). conv_check.sh passed on S25, S21 and S20 (35/35, window runs 27/27, cnorm,
  window, unet_ops, stt_ops, qmm).

### ggml-ameva: flash-attention tile with two rows per invocation on Adreno (2026-11-04)

- `FA_VARIANTS` gains `(128, 16, 32, 2, 1)` (appended; one new SPIR-V module and its OpenCL translation, every other
  kernel byte-identical). `ameva_fa_find` takes it for DMAX-128 tiles on devices named "Adreno" when its 16 rows
  fill; elsewhere, and for token generation, the choice is the one before.
- `tests/ameva_fa_bench.cpp` (one FLASH_ATTN_EXT node, f16 K/V, checked against a double reference on 48 rows per
  head) and `scripts/termux/lab/fa_bench.sh`. Auto choice before -> after: Turnip Adreno 650 D 80 / 1024 tokens
  152 -> 102 ms, D 128 / 512 50.6 -> 33.2 ms; Adreno 830 48.3 -> 36.4 ms and 16.9 -> 12.3 ms; Mali-G78 unchanged
  (the tile is slower there: 191 -> 227 and 62.5 -> 75.1 ms when forced). D 40 (SD1.5 64x64, most of the
  attention time): none of six other tiles beat the current one on S20, S21 or S25.

### `ameva_sd_generate`: img2img with a one-step schedule (2026-11-04)

- sd.cpp starts img2img at step floor(steps x strength); with one step and any strength below 1 that is the full
  step from sigma_max and the source image was ignored (S21, SDXS 512, strength 0.55: unrelated pictures). For that
  case the SDK passes `strength_as_noise_level=true,force_first_sigma=true`, so the single step starts at the
  strength's noise level. Note: SDXS still does not denoise from an intermediate level (grainy results at 0.3 and
  0.55); AmfyUI refuses photo conversion with it.

### Measured on Mali-G68 (A35, A53), 2026-11-05

- Streamed diffusion tier, Anything q8 + IP-Adapter, 512x512, 4 steps (AmfyUI MIDDLE comic cut): the UNet runs on
  the GPU (budgets 1196-1460 MiB). Per cut A35 212-214 s (was 230-241 s on the CPU), A53 232 s (was 342-350 s).
- BM64 convolution tile, VAE decode: A35 41.0 -> 36.7 s, A53 57.0 / 57.2 -> 50.5 / 50.3 s; Mali-G68 now in the auto
  list (in-app A53 with the auto choice: 50.4 s). Sampling unchanged (the UNet keeps the 32 tile).
- Diffusion flash attention: A53 sampling 123.7-127.6 s with it, 183.6 s without. Kept on.
