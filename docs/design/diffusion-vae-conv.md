# Diffusion Stage 1: VAE decode bottleneck and convolution / channel-norm kernels (2D and Wan 2.1)

- Written 2026-10-07, branch `perf/opencl-mali`. Follows `AMUDA_ENGINE_UNIFICATION_PLAN.md` (diffusion phase).
- Sources read: local `dev/termux/termux-diffusion` only (sd.cpp `50d6405`, ggml `8e800cef` plus an uncommitted 60-line
  `ggml-vulkan.cpp` change). No remote fetch. The A35 device binary is the 2.7.9 bundle `sd-cli` (older sd.cpp layout),
  so line numbers of the local tree are approximate for it.
- Status: measurements and design. Step 1 of section 6 (the kernel in isolation, `ameva_conv_bench`) is measured on
  three devices: section 7. CONV_2D is in ggml-ameva and checked on devices: section 8. The other decoder ops and
  sd.cpp on ggml-ameva are not done. Section 5 lists what is predicted and not measured.

## 1. Measured: 2D VAE decode on the Galaxy A35 (Mali-G68, Vulkan)

Conditions: `sd-cli -m anime.gguf` (SD 1.x, VAE 94 MB f16) `--steps 1 --cfg-scale 1`, ssh session (cores 0-2, A55 only),
`maxStorageBufferRange` 268435456, 2026-10-07. Logs: A35 `~/claude_vae_prof/`. One run per row.

| run | image | decode wall | where it ran |
|---|---|---|---|
| default (im2col + MUL_MAT) | 256x256 | 432.74 s | 2 convolutions on the CPU, rest on Vulkan |
| default | 192x192 | 23.55 s | Vulkan only (no scheduler split) |
| default + `--vae-tiling --vae-tile-size 24` | 256x256, 4 tiles | 92.37 s | Vulkan only |
| `--vae-conv-direct` | 256x256 | 226.89 s | Vulkan only |

- **Unreported CPU execution.** At 256x256 (one default tile: 32 latent = 256 px) the scheduler prints
  `SPLIT #1: CPU` and `SPLIT #3: CPU` (`GGML_SCHED_DEBUG=2`): `decoder.up.1.upsample.conv` and
  `decoder.up.0.block.0.conv1`, each IM2COL 288 MiB + MUL_MAT. ggml-vulkan declines any op with a tensor above
  `maxStorageBufferRange` (`tensor_size_supported` in `ggml_backend_vk_device_supports_op`), and ggml moves the node to
  the CPU. Nothing is logged at the default level. During those splits: 4 threads at 52-100 % CPU, `gpu_busy` 0 %.
  The two convolutions are 116 of 620 GFLOP and take about 400 of the 433 s (by subtraction of the GPU part; cores
  0-2, other core sets not measured).
- The wrapper (`termux_diffusion/core.py`) passes `--vae-tiling` with no tile size, so every tile of the product's
  default decode on this device has this split. It never passes `--vae-conv-direct`.
- **GPU time is the matrix product.** `GGML_VK_PERF_LOGGER`, first three resolution levels: MUL_MAT 25.41 of 26.15 s
  (97.1 %) at 12.2-16.8 GFLOPS, IM2COL 0.26 s, all other ops (ADD, MUL, GROUP_NORM, SILU, CONT, UPSCALE) 0.49 s.
- Direct convolution: CONV_2D is 224.5 of 225.7 s at 2.75-2.80 GFLOPS (3x3), about 6 times slower per flop than MUL_MAT.
- Output compared with the default run (PNG, 8-bit): direct 59.45 dB PSNR (max difference 1), 24-latent tiles 49.92 dB
  (max 7).
- Not usable for this measurement: S20 `sd-cli` (OpenCL build `3f8527a`) aborts in the UNet
  (`clEnqueueNDRangeKernel` -54, `ggml-opencl.cpp:1180`); S20 `~/sd-cli-vulkan` asserts `compute_ctx.expired()`
  (`ggml-vulkan.cpp:17160`) with the timing logger on. Neither was examined.

## 2. Measured: matrix-product kernels at convolution shapes (A35, Vulkan)

One 3x3 convolution as a product, 19.33 GFLOP each: positions x output channels x (input channels x 9).
AMEVA rows: `ameva_mm_bench`, build `out-m2` (2026-10-07), wall ms per `graph_compute`, 3-4 iterations.

| kernel | 16384x256x2304 | 4096x512x4608 | 65536x128x1152 | GFLOPS | difference |
|---|---|---|---|---|---|
| ggml-vulkan MUL_MAT f16 (product today) | 1180 | 1179 | (not timed) | 16.4 | reference |
| ggml-vulkan CONV_2D | 7005 | 6987 | 6937 | 2.8 | 59.45 dB image |
| AMEVA `ameva_qmm` f16 weights, f32 activations | 1306 | 1558 | 1263 | 12-15 | - |
| AMEVA `ameva_qgemm` q8_0 weights, f32 activations (`AMEVA_MMQ=0`) | 238 | 247 | 234 | 78-83 | 7e-6..1.5e-5 of rms vs `ameva_qmm` |
| AMEVA `ameva_mmq` q8_0 weights, int8 activations | 103 | 101 | 118 | 164-191 | 1.8e-2..2.0e-2 of rms vs `ameva_qmm` |

- AMEVA's generic dense kernel gives nothing over ggml-vulkan. The tile kernels are 4.8-5.0x (f32 activations) and
  10-11.7x (int8 activations) faster at these shapes. They exist only for repacked quantized weights and read a
  materialised activation matrix.
- The int8 path changes each product by about 2 % of its rms. Its effect on a decoded image is not measured.

## 3. Wan 2.1 VAE as built by sd.cpp (`src/model/vae/wan_vae.hpp`)

- `GGML_MAX_DIMS` is 4. Torch `(B, C, T, H, W)` is held as ggml `ne = [W, H, T, B*C]` and every block asserts `B == 1`.
  Element index: `w + W*(h + H*(t + T*(c + C*b)))`. A per-channel operand `[1,1,1,C]` already broadcasts over this
  layout in ggml and in `ameva_binary.comp` (`i3 mod b_ne3`); the broadcast itself needs no new arithmetic.
- The cost is in how the blocks reach that layout:
  - `RMS_norm` (channel axis): PERMUTE+CONT, RMS_NORM, MUL by gamma, PERMUTE+CONT, then SILU: 5 full-tensor passes,
    two of them strided copies.
  - `CausalConv3d`: CONCAT (cached frames), PAD, IM2COL_3D, MUL_MAT, PERMUTE+CONT, ADD (bias). On Vulkan
    `ggml_ext_conv_3d` always takes IM2COL_3D, since the backend reports it supported; the column matrix is
    27 x C x 2 bytes per position.
  - `Resample`: time convolution, reshape/PERMUTE+CONT (frame interleave), PERMUTE+CONT, UPSCALE, 2D convolution,
    PERMUTE+CONT.
  - A residual block is about 23 compute nodes.
- Decoder: dim 96, multipliers {1,2,4,4}, 2+1 residual blocks per stage, temporal upsample {true,true,false},
  z 16, 33 cached convolutions, one latent frame per decoder pass (4 output frames after the first).
- Calculated from that configuration, not measured: 33.75 Mflop per output-frame pixel per latent frame
  (8.4 Mflop per output pixel; the SD 2D decoder is 9.5). 480x832: 13.5 TFLOP per latent frame. Last-stage tensor
  `[832,480,4,96]` f32 is 613 MiB; its IM2COL_3D matrix would be 8.3 GB (1.36 GB for a 256x256 tile).
  By the rule of section 1 those convolutions would leave the GPU on a 256 MiB device.
- The fleet holds no Wan VAE weights (only `taew2_1.safetensors`), and the Python wrapper has no video mode.

## 4. Design

Target: sd.cpp graphs on the `ggml-ameva` backend, convolution as one node, no column matrix.

### K1. `ameva_conv` (GGML_OP_CONV_2D, GGML_OP_CONV_3D)

- sd.cpp emits CONV_2D with `--vae-conv-direct`, and CONV_3D by itself when the backend declines IM2COL_3D
  (`ggml_ext_conv_3d`). ggml-ameva declines IM2COL and IM2COL_3D and accepts the two direct ops; no sd.cpp change.
- Kernel: the tile structure of `ameva_qgemm` / `ameva_mmq`, with the activation staging step reading the source
  tensor instead of a matrix. K is ordered tap-major, channel-minor, so a 32-wide K unit is 32 consecutive channels
  at one tap `(dx, dy, dt)`. The 32 activation rows of a tile are 32 consecutive `w` of one `(h, t)`, so staging reads
  runs that are contiguous in memory. Out-of-range taps stage zeros (2D and 3D padding; PAD is folded when it feeds
  the convolution directly). Stride and dilation enter the tap offset only.
- Output is written at `w + W*(h + H*(t + T*oc))` with the bias added (the separate ADD and PERMUTE+CONT disappear).
- Weights: converted once per kernel tensor into a side buffer in the repacked tile layout, K reordered. Two modes:
  exact (f16 weights, f32 activations; needs an f16 unit decoder in `ameva_qdecode.glsl`) and integer (q8_0 weights,
  int8 activations). The side buffer costs up to half the f16 size again per VAE.
- Integer mode: the source tensor is quantised once per convolution input in blocks of 32 channels per position,
  written position-major (channels contiguous), with one scale per block. A K unit of the product is then one
  block of the quantised tensor: C bytes per position, in place of the 9C or 27C of a column matrix.
- Causal time axis, second step: the cached frames are a second binding read for `t + dt < 0`, replacing CONCAT.
- Channel counts that are not multiples of 32 (z 16 or 4, RGB 3): K is padded with zero weights.

### K2. `ameva_cnorm` (channel-axis normalisation with per-channel scale: the 5D broadcast kernel)

- Recognised patterns: Wan `CONT(PERMUTE) -> RMS_NORM -> MUL(gamma) -> CONT(PERMUTE) [-> SILU]`, and
  `GROUP_NORM -> MUL -> ADD [-> SILU]` of the 2D decoders (groups of C/32 channels over the whole plane).
- Two dispatches on the unpermuted layout, both streaming along `w`:
  1. `ss[p] += sum over a channel window of x^2` into a plane of `W*H*T` floats (RMS), or per-group sums (group norm);
  2. `y = act(x * gamma[c mod C] * inversesqrt(ss[p]/C + eps) + beta)`.
- A tensor above one binding is processed in windows of whole channel planes; a position's channels never need to be
  bound together. `c mod C` keeps `B > 1` correct in the folded axis. 5 passes become 2 reads and 1 write.

### K3. Remaining ops of the VAE graphs

UPSCALE (nearest), PAD, CONCAT, SOFT_MAX, and f32 x f32 MUL_MAT for the middle attention. Nearest 2x followed by a
3x3 convolution can be folded into K1's tap offset later (`w >> 1`); not in the first step.

### Contract

Each op gets a reason string in `ameva_op_unsupported_reason`. A graph that needs a declined op fails at
`graph_compute` with that reason; there is no CPU path behind the backend. The wrapper selects the AMEVA plan only on
devices where the VAE op census passes (`tests/ameva_graph_census.cpp` extended to sd.cpp graphs).

## 5. Predicted, not measured

- ~~That staging from the source tensor keeps the tile kernels' throughput.~~ Measured, section 7: it does on Adreno
  (faster than the matrix); on Mali-G68 the staging is cheaper than the matrix's but the planar output costs 16 %.
- ~~Exact mode with f16 weights is assumed to match the q8_0 / f32 timing.~~ Measured, section 7: 10-15 % slower on
  Mali-G68 with the exact decoder, 1-5 % with the built-in conversion; equal on Adreno 830, 9-15 % slower on Adreno 650.
- Image effect of int8 activations through about 36 stacked convolutions.
- The other Mali devices (S21, A53), and everything of sections 1 and 2 on Adreno (no ggml-vulkan time there).
- At 164-191 GFLOPS a 480x832 Wan latent frame would take about 75 s on the A35 (13.5 TFLOP), 81 frames about 26 min.

## 6. Order of work and gates

1. `ameva_conv_bench`: K1 staging against a materialised matrix on the three shapes of section 2 and one 3x3x3 shape,
   A35 then S25 and S20. Gate: at least 4x ggml-vulkan MUL_MAT in exact mode, every output compared with an f32
   reference convolution.
2. K1 2D in ggml-ameva + K2 group norm + K3, op census of the SD 1.x and Flux-type decoders, `test-backend-ops`.
3. sd.cpp built with ggml-ameva (leejet ggml `8e800cef` headers, ABI probe). Gate: decoded image against the f16
   Vulkan image (PSNR per mode), decode time per 256 px tile, no scheduler split.
4. K1 3D + K2 RMS + cached-frame binding. Needs Wan VAE weights on a device. Gate: frames against the CPU decode.

Usable today without new code (measured in section 1): 24-latent tiles keep the decode on the GPU on the A35,
92.37 s against 432.74 s for a 256x256 image.

## 7. Measured: `ameva_conv_bench` (step 1 of section 6, 2026-10-07)

`tests/ameva_conv_bench.cpp` through `ac_runtime` directly (Vulkan), kernel `shaders/ameva_conv.comp`, f16 unit
decoder in `ameva_qdecode.glsl` (`TYPE_F16`), script `scripts/termux/lab/conv_bench.sh`. Logs:
`~/abi-lab/out-conv/run_final.log` on each device (A35 also `out-conv-unpack/`).

- Paths per weight type: **matrix** = `ameva_qgemm` on a column matrix built on the CPU (dispatches sliced as
  ggml-ameva does), **conv_s** / **conv_v** = `ameva_conv` with scalar / aligned-vec4 staging from the source tensor,
  **conv_v_direct** = `conv_v` with each invocation writing its own output block.
- Correctness: every destination is filled with NaN bytes, computed, and compared with an f32 CPU convolution that
  uses the same dequantised weights and summation order, on the four shapes below and four small ones (4 -> 3
  channels, 1x1, stride 2 with dilation 2, 33 -> 65 channels 3x3x3 causal). A35: equal bit for bit to the reference
  built with fused multiply-add (largest difference 0 on all 60 checks). S25 and S20: 5.0e-7 to 2.9e-6 of the rms
  from that reference, and equal bit for bit to a reference with a rounded product and a rounded sum per term
  (`AMEVA_CONV_BENCH_REF=muladd`, 36 checks each): these two drivers do not fuse the kernels' `fma()`.
- The kernel differs from section 4 K1 in two points: the order of the K units comes from a host table (tap offset
  and channel group per unit), and the output goes through shared memory.

Wall ms per convolution, mean of 4 rounds after one that is not counted, paths in turn.

A35, Mali-G68, cores 0-2, GPU at 949 MHz in most samples:

| source -> channels, kernel | GFLOP | q8_0 matrix | q8_0 conv_v | f16 matrix | f16 conv_v | f16 conv_v, built-in conversion |
|---|---|---|---|---|---|---|
| 128x128x256 -> 256, 3x3 | 19.33 | 242.2 | 274.9 | 250.4 | 307.0 | 277.1 |
| 64x64x512 -> 512, 3x3 | 19.33 | 246.2 | 273.1 | 257.1 | 315.1 | 287.4 |
| 256x256x128 -> 128, 3x3 | 19.33 | 238.4 | 274.4 | 247.9 | 309.4 | 277.5 |
| 96x96x4x96 -> 96, 3x3x3 causal | 18.35 | 293.3 | 347.5 | 311.2 | 381.3 | 341.8 |

S25, Adreno 830, cores 0-1,4-5, `max_gpuclk` 832 MHz, an idle `llama-server` of another session present; and S20,
Turnip Adreno 650, cores 0-7, `max_gpuclk` 587 MHz. matrix / conv_s:

| source -> channels, kernel | S25 q8_0 | S25 f16 | S20 q8_0 | S20 f16 |
|---|---|---|---|---|
| 128x128x256 -> 256, 3x3 | 57.3 / 45.2 | 56.3 / 45.5 | 176.3 / 153.2 | 200.0 / 170.2 |
| 64x64x512 -> 512, 3x3 | 56.8 / 45.5 | 55.5 / 45.5 | 179.0 / 151.6 | 205.7 / 174.0 |
| 256x256x128 -> 128, 3x3 | 56.5 / 45.0 | 55.6 / 45.6 | 171.7 / 170.9 | 196.3 / 189.7 |
| 96x96x4x96 -> 96, 3x3x3 causal | 71.0 / 55.7 | 70.0 / 56.0 | 213.7 / 186.4 | 234.6 / 203.6 |

- **Staging from the source tensor against the matrix.** Adreno 830: 1.22-1.28x faster (328-430 GFLOPS). Adreno 650:
  1.15-1.18x faster on three shapes, 1.00-1.03x on 256x256x128. Mali-G68: 0.84-0.90x with q8_0 weights, 0.80-0.82x
  with f16. On Adreno the scalar and vec4 staging and the two output forms are within 7 % of each other.
- **Where the Mali-G68 time is** (ablation builds, 128x128x256 -> 256, q8_0, ms): `ameva_qgemm` 236 = 201 without
  its activation loads + 35; `ameva_conv` 273 = 199 (no loads, row-major output) + 45 (planar output) + 29 (source
  loads). The staging itself is cheaper from the source than from the matrix. The deficit is the planar output of
  ggml's layout: a tile writes 64 runs into 64 planes. MUL_MAT does not write that layout (in the product graph the
  CONT after the permute does, and the 1180 ms of section 2 do not contain it).
- Against that cost: one plane run per invocation through shared memory gives q8_0 275 against 278 ms and f16 307
  against 325. Also measured, each within 3 ms: a flat against a 3D grid (270 / 269), tap-major against group-major
  unit order (275 / 272), tap counters in the kernel against the unit table. The padding mask costs 0-3 ms, the
  third aligned load and the shift 0-5 ms.
- **f16 decode** (Mali-G68, matrix path, the three 3x3 shapes): `ameva_h2f` per element 288-302 ms, branch-free
  `h2f4` 248-257, the built-in `unpackHalf2x16` (`F16_UNPACK`) 219-239. With the built-in conversion the A35 output
  is still equal bit for bit to the reference (the random weights contain f16 subnormals). No other driver was
  checked for that.
- **Gate of step 1** (A35, 4x ggml-vulkan MUL_MAT f16 = 1180 ms, exact mode): `conv_v` f16 with the exact decoder
  3.74-3.84x: not met. With the built-in conversion 4.10-4.26x, with q8_0 weights 4.29-4.32x. The convolution node
  also adds the bias and writes the planar layout, which MUL_MAT does not. S25 and S20 have no ggml-vulkan time.
- Open: (1) which f16 decoder the product kernel uses (exact everywhere, or the built-in one after a check of
  subnormals on the device at load); (2) a channels-last activation layout inside ggml-ameva would remove the planar
  output and the shift, and the matrix columns above are what such a kernel reads and writes; (3) int8 activations
  (`ameva_mmq` structure) are not in this bench; (4) a source or destination above one binding is refused by the bench.

## 8. K1 2D in ggml-ameva (step 2 of section 6, convolution part, 2026-10-08)

User decisions of 2026-10-07 applied: the built-in f16 conversion after a check on the device, and ggml's planar
layout kept.

- **Op.** `GGML_OP_CONV_2D` is computed by the graph engine (`enc_conv_2d`, contract in `ameva_conv2d_geometry`,
  `adapters/llama/ggml-ameva/ggml-ameva.cpp`): f16 or f32 kernel tensor, f32 data and result, stride, padding and
  dilation up to 255, a batch, up to 8192 input and 16384 output channels, every tensor inside one binding.
  `IM2COL` stays declined, so a graph has to carry the direct op (sd.cpp: `--vae-conv-direct`).
- **Weights.** Per node one dispatch of `shaders/ameva_conv_w.comp` copies the bit patterns of the kernel tensor into
  the scratch buffer in the K order of `ameva_conv` (channel groups of 32, all taps of a group, channels). No side
  buffer is kept, so no weight memory is added; the copy runs again at every `graph_compute`. The unit tables (per
  kernel size, dilation and group count) and one zero vector for the bias binding are created once. The bias is
  still a node of its own in the graph (not fused; the kernel adds it since section 9).
- **Kernels.** A SPIR-V-only family of `scripts/build_ameva_kernels.py` (`VK_ONLY`: `conv_{f16,f16u,f32}_{s,v}`,
  `conv_w_{f16,f32}`, table `kAmevaVkKernels`). `ggml_ameva_kernels_spv.h` was regenerated on the S25 (glslang
  16.3.0, spirv-val): 4030 added lines, no existing module changed, and the OpenCL header is identical. On the
  OpenCL backend the op is declined with a reason: `scripts/glsl_to_opencl.py` does not translate these kernels.
- **f16 decoder.** Once per process, before the first graph with an f16 convolution is recorded, a 1x1 convolution
  runs through the kernel with every f16 subnormal, both zeros and 2048 normal values as weights against an identity
  source. The built-in conversion is used only when all 4096 results carry the bits of the CPU conversion; otherwise
  the exact decoder, with one line on stderr either way. `AMEVA_CONV_F16=exact` forces the exact decoder. The check
  selected the built-in conversion on the A35, S20 and S25.
- **Staging.** `AMEVA_CONV_STAGE=auto` takes aligned vec4 loads on Mali and scalar loads elsewhere (section 7);
  scalar for a stride above 1 and when the source ends in a partial vec4 of its binding.

Verification, plugin built against ggml b8471 and run with the ggml libraries of the 2.8.1 bundle
(`scripts/termux/lab/conv_check.sh`, logs `~/abi-lab/out-convk/` on each device):

| | A35, Mali-G68 | S20, Turnip Adreno 650 | S25, Adreno 830 |
|---|---|---|---|
| `tests/ameva_conv_check` (12 cases x 5 kernel selections, f32 CPU reference) | 12/12 x 5, bit-equal to the fused reference | 12/12 x 5, bit-equal to the multiply-then-add reference | 11/11 x 5, same reference as S20 (sources before the last two edits) |
| `test-backend-ops -b AMEVA0 -o CONV_2D` | 1568/1568, none declined | 1568/1568, none declined | not run |
| `ameva_qmm_check` (MUL_MAT regression) | 516/516 | 516/516 | 516/516 |
| ms per 19.33 GFLOP convolution through `graph_compute`, f16 kernel (128x128x256, 64x64x512, 256x256x128) | 271.7 / 287.6 / 271.3 | 158.8 / 163.8 / 182.4 | not measured |

- The twelfth case is a kernel larger than the padded source (one output row, every tap outside), which ggml
  defines and `test-backend-ops` contains 111 times; it was declined until the last edit. The S25 stopped answering
  (connection timed out from about 01:10 on 2026-10-08) before the final sources and `test-backend-ops` could run there.
- A35 against ggml-vulkan MUL_MAT (1180 ms): 4.10-4.34x per convolution node, weight copy included. The isolated
  kernel of section 7 took 277 / 287 / 278 ms, so the integration adds nothing measurable. A35 timing conditions:
  cores 0-2, GPU clock sampled (949 MHz while computing), second run 272.2 / 286.6 / 271.0. One earlier run
  directly after the 12-minute plugin compile on the device gave 364 / 289 / 354 ms; its clock was not sampled.
- Device trap met on the A35: no `patchelf`, so the plugin keeps `$PREFIX/lib` before `$ORIGIN` in its RUNPATH, and
  the bundle's lib directory holds an older `libggml-ameva.so`. With that directory first on `LD_LIBRARY_PATH` every
  test failed at load ("cannot locate symbol"); `conv_check.sh` now puts its own output directory first.
- Not done: bias fusion, CONV_3D, tensors above one binding (declined), an OpenCL translation, group norm and the
  other decoder ops (K2, K3), and sd.cpp built against ggml-ameva (step 3). No decoded image exists yet.
  (Bias fusion and group norm: section 9.)

## 9. K2 group norm and the convolution bias in ggml-ameva (step 2 of section 6, 2026-10-08)

- **Op.** `GGML_OP_GROUP_NORM` is computed by the graph engine (`enc_group_norm`, contract in
  `ameva_group_norm_geometry`): f32 `[W, H, C, B]`, contiguous, inside one binding, up to 65535 channels, result in
  place of the source or apart from it. Groups follow ggml (`ceil(C / groups)` channels each, a short last group,
  groups without channels skipped). Declined with a reason on the OpenCL backend, as the convolution is.
- **Kernels.** Two more SPIR-V-only modules. `ameva_cnorm_stat`: one workgroup per group and batch element; the
  mean, then the squared differences from it (ggml's two passes), each in 512 f32 partial sums reduced in shared
  memory; it writes the mean and `1 / sqrt(var + eps)` into the scratch buffer. `ameva_cnorm`: one dispatch per
  batch element, grid `(ceil(W*H / 256), C)`, `y = (x - mean) * scale`. This is not the sketch of K2 in section 4:
  both statistics passes are one dispatch, there is no `ss[p]` plane, and the RMS form of the Wan VAE is not built.
- **Fused chain.** sd.cpp's GroupNorm32 + SiLU is GROUP_NORM, MUL by a `[1, 1, C, 1]` weight, ADD of a `[1, 1, C, 1]`
  bias, SILU, each stored in place of the result before; in graph order the RESHAPE of the weight and of the bias
  sit between them. `ameva_find_chain` passes over metadata-only nodes and gives the nodes that follow in that
  order to the norm's second dispatch (`flags` of `ameva_cnorm`), provided each is stored in place of the norm's
  result: then no node can read a value the fused dispatch leaves out. Four nodes take 2 dispatches instead of 5.
  Every step is rounded on its own (`precise`), as the separate kernels round.
- **Convolution bias.** A CONV_2D followed by the in-place ADD of a `[1, 1, OC, 1]` f32 bias (sd.cpp's
  `ggml_ext_conv_2d`) binds that bias where the zero vector was; the kernel adds it after its sum, the one rounded
  addition of the separate ADD. `AMEVA_FUSE=0` runs every node on its own.
- **Numerics.** The norm is not bit-equal to ggml's CPU code, which accumulates the statistics in double.
  `tests/ameva_cnorm_check` bounds the difference by the worst case of the f32 partial sums,
  `(n / 512 + 32) * eps * (1 + |mean| * scale) * (1 + |y|)` for a group of `n` elements, widened for MUL, ADD and
  SILU (`tolerance()` in the test). Measured: the plain norm at most 0.031 of that bound on each of the three
  devices below; its largest absolute error on the S21 and the S20 1.2e-7 to 4.8e-7 for inputs of order 1 and
  6.8e-6 to 6.9e-6 with a mean 30 times the spread.

Verification with the kernel header regenerated on the S21 (glslang 16.3.0, spirv-val: 420 added lines, no existing
line changed, OpenCL header identical), plugin built against ggml b8471, ggml libraries of the 2.8.1 bundle
(`scripts/termux/lab/conv_check.sh`, logs `~/abi-lab/out-cnorm/` on each device):

| | S21, Mali-G78 | S20, Turnip Adreno 650 | A53, Mali-G68 |
|---|---|---|---|
| `tests/ameva_conv_check`: 11 convolutions without and with the bias, one declined source; 6 runs (5 kernel selections, `AMEVA_FUSE=0`) | 23/23 x 6, bit-equal to the fused reference, 11 biases fused (0 with `AMEVA_FUSE=0`) | 23/23 x 6, bit-equal to the multiply-then-add reference, same counters | 23/23 x 6, bit-equal to the fused reference, same counters |
| `tests/ameva_cnorm_check`: 12 shapes x 4 graphs, one declined source, one decoder block; fused and `AMEVA_FUSE=0` | 14/14 and 14/14; largest error 0.287 of the bound | 14/14 and 14/14; 0.223 of the bound | 14/14 and 14/14; 0.287 of the bound |
| fused MUL + ADD against the separate nodes, fused SILU against the separate node (elements that differ in bits) | 0, 0 in all 12 shapes | 0, 0 in all 12 shapes | 0, 0 in all 12 shapes |
| decoder block (norm, weight, bias, SILU, 3x3 convolution, bias: 5 fused nodes), max error / rms | 3.88e-06 | 4.12e-06 | 3.88e-06 |
| `test-backend-ops -b AMEVA0 -o GROUP_NORM`, `-o GROUP_NORM_MUL_ADD`, `-o CONV_2D` | 2/2, 2/2, 1568/1568 | 2/2, 2/2, 1568/1568 | 2/2, 2/2, 1568/1568 |
| `test-backend-ops -b AMEVA0`, every op | 3596/3596 | not run | not run |
| `ameva_qmm_check`, `ameva_fuse_check` (same hash with fusion on and off, 5 and 0 pairs), `ameva_kvq_check`, `ameva_chunk_check` | 516/516, pass, pass, pass | 516/516, pass, pass, pass | 516/516; the other three not run |

`test-backend-ops` computes one node per graph, so it checks the unfused kernels; the fused chains are checked by
the two tests above, with the counters of fused nodes required to move.

On the OpenCL backend (A53, `AMEVA_BACKEND=opencl`, same plugin) `supports_op` declines GROUP_NORM in all 13 cases
of `ameva_cnorm_check` (group norm counter 0) and CONV_2D in all 11 of `ameva_conv_check`; nothing is computed
another way. `ameva_qmm_check` there: 516/516.

Timing, ms per `graph_compute`, two passes per device, graphs timed in the order norm alone, four nodes, fused (not
alternated). The four-node form has a result tensor per node, the fused form works in place of its input, so this
shows the size of the op, not a controlled comparison of the two forms:

| 32 groups | S20 (cores 0-7, `max_gpuclk` 587 MHz) | S21 (cores 0-2,4-7, GPU clock sampled between 130 and 858 MHz) |
|---|---|---|
| 32x32x512: norm alone / four nodes / fused | 1.5-1.7 / 3.6-4.0 / 2.5-2.7 | 2.2-2.4 / 7.9-9.6 / 2.3-2.4 |
| 64x64x512 | 4.9-5.0 / 11.9-12.1 / 5.1 | 6.6-89.1 / 14.3-19.6 / 3.2-7.4 |
| 128x128x256 | 8.0-9.1 / 18.8-21.7 / 6.7 | 11.3-12.2 / 21.3-31.0 / 4.8-8.0 |
| 256x256x128 | 12.1-15.3 / 29.2-31.7 / 10.6-11.2 | 18.0-21.0 / 31.4-38.6 / 6.7-7.6 |
| 19.33 GFLOP convolution, f16 kernel (128x128x256, 64x64x512, 256x256x128) | 158.3-170.7 / 163.8-164.1 / 182.4-182.9 | 168.9-175.0 / 183.0-185.1 / 110.3-110.5 |

- On the S21 the GPU clock was still rising inside these graphs of a few ms (the 89.1 ms is one pass of one shape;
  the other pass gave 6.6), so its norm figures are a range, not a measurement of the kernel. Its convolution
  figures are the first on that device and repeat within 4 %.
- A fused norm chain is 3-6 % of the 19.33 GFLOP convolution of the same size (S20: 5.1 of 164, 6.7 of 158-171,
  10.6-11.2 of 182-183 ms), so the convolution still decides the decode time.
- Not done: UPSCALE, PAD, CONCAT, SOFT_MAX and the f32 MUL_MAT of the middle attention (K3), CONV_3D and the RMS
  form of the Wan VAE, tensors above one binding (declined), an OpenCL translation of these kernels, and sd.cpp
  built against ggml-ameva (step 3). No decoded image exists yet. S25 and A35 did not answer on 2026-10-08
  (connection timed out), so nothing of this section ran on them.
