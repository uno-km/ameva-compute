# Diffusion Stage 1: VAE decode bottleneck and convolution / channel-norm kernels (2D and Wan 2.1)

- Written 2026-10-07, branch `perf/opencl-mali`. Follows `AMUDA_ENGINE_UNIFICATION_PLAN.md` (diffusion phase).
- Sources read: local `dev/termux/termux-diffusion` only (sd.cpp `50d6405`, ggml `8e800cef` plus an uncommitted 60-line
  `ggml-vulkan.cpp` change). No remote fetch. The A35 device binary is the 2.7.9 bundle `sd-cli` (older sd.cpp layout),
  so line numbers of the local tree are approximate for it.
- Status: measurements and design. Step 1 of section 6 (the kernel in isolation, `ameva_conv_bench`) is measured on
  three devices: section 7. CONV_2D is in ggml-ameva and checked on devices: section 8. Group norm and the other
  decoder ops: sections 9-10. sd.cpp on ggml-ameva with a decoded image compared against the CPU's: section 11.
  Section 5 lists what is predicted and not measured.

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

## 10. K3 ops and the VAE graph census (step 2 of section 6, 2026-10-08)

- **Ops.** UPSCALE (nearest), PAD (zeros and circular), CONCAT (f32, f16), SOFT_MAX (scale, f32 or f16 mask, in
  place) and f32 x f32 MUL_MAT of any shape are computed by the graph engine (`enc_upscale`, `enc_pad`, `enc_concat`,
  `enc_soft_max`, `enc_mul_mat_dense`; contract in `ameva_op_unsupported_reason` and `mul_mat_dense_reason`).
- **Kernels.** Four op kernels, built as SPIR-V and translated to OpenCL C like the other op kernels (unlike the
  convolution and the group norm): `ameva_upscale`, `ameva_pad` (one invocation per result element, words copied
  without a float conversion, any strides), `ameva_soft_max` (one workgroup per row; maximum and sum reduced in
  shared memory; an invocation writes only what it read, so the result may lie in place of the source) and `ameva_mm`
  (one invocation per result element, scalar loads, four partial sums). CONCAT is two dispatches of `ameva_cpy`,
  each source copied into its part of the result.
- **Exactness.** UPSCALE, PAD and CONCAT equal ggml's CPU result in every bit. ggml takes nearest index
  `(int64_t)(i / sf)` with `sf = (float)dst / src`; the kernel computes `i * src / dst` in integers, and a node is
  accepted only when both agree for every index of every axis (always so for whole factors; checked index by index
  for fractional ones, declined when they differ). SOFT_MAX and MUL_MAT differ from the CPU by rounding (f32 tree
  sums against ggml's double sum and SIMD order); the tests bound it.
- **Declined with a reason.** UPSCALE in any mode other than nearest without flags; SOFT_MAX with an ALiBi bias or
  attention sinks; CONCAT of other types than f32 and f16; circular padding wider than the source (ggml's own index
  is negative there); tensors above one binding.
- **f32 MUL_MAT.** The first product of sd.cpp's VAE attention (K = 512 channels) ran on `ameva_qmm` before. The
  second has K = h * w latent positions: `ameva_qmm` takes it when that is a multiple of 32, `ameva_mm` otherwise
  (a 30x30 latent: K = 900) and for tensors of several matrices. ggml repeats each matrix of src0 over
  `ne2(src1) / ne2(src0)` consecutive matrices of src1 (index `i / r`, not `i % n`): the first version of the
  kernel's host code used the modulo and failed 64 cases of `test-backend-ops`; corrected before the runs below.
- **Census** (`tests/ameva_vae_graph_check`, graphs rebuilt node for node from stable-diffusion.cpp `50d6405`, not
  read from a running sd.cpp; compute memory from `ggml_gallocr` as in sd.cpp). The S21's `sd-cli` is linked
  statically and prints no scheduler splits, so its graph could not be recorded.

| decoder graph | nodes (compute) | distinct signatures | declined by ggml-ameva |
|---|---|---|---|
| SD 1.x AutoencoderKL, `ggml_conv_2d_direct` (`--vae-conv-direct`) | 346 (226) | 12 | 0 |
| the same with z = 16 and no `post_quant_conv` (Flux, Z-Image) | 343 (224) | 12 | 0 |
| TAESD `TinyDecoder` | 146 (114) | 7 | 0 |
| SD 1.x with `ggml_conv_2d` (sd.cpp's default) | 586 (306) | 14 | 80: 40 IM2COL, 40 f16 x f16 MUL_MAT |

  SD 1.x signatures: CONV_2D f16 kernel 40, ADD of a `[1, 1, C, 1]` vector 70, GROUP_NORM 30, MUL by a vector 30,
  SILU 29, ADD of two tensors 15, CONT of a permuted tensor 5, UPSCALE 3, MUL_MAT 2, SCALE 1, SOFT_MAX 1. TAESD: CONV_2D
  35, ADD of a vector 32, RELU 31, ADD 10, UPSCALE 3, SCALE 2, TANH 1. PAD and CONCAT are not in these three graphs
  (sd.cpp uses them for circular padding and in the video decoders); they are covered by the op tests only. The
  wrapper has to pass `--vae-conv-direct`: the default form is declined at its IM2COL nodes.
- **Whole graph against the CPU backend.** With an f16 kernel ggml's CPU convolution rounds every activation to f16
  before its product (`ggml_compute_forward_conv_2d_impl`, `GGML_CPU_FP32_TO_FP16`); `ameva_conv` does not. The
  reference therefore holds the same kernel values as f32. Against it the SD 1.x decoder output differs by 3.9e-6
  to 4.1e-6 of its RMS (largest difference 1.5e-5 at values up to 3.3); against the CPU path with f16 kernels by
  9.6e-3 to 1.05e-2, which is that path's own rounding. A comparison of a decoded image with sd.cpp's CPU image
  will show that second figure.

Verification with the kernel header regenerated on the S21 (glslang 16.3.0, spirv-val: SPIR-V header 942 added
lines and none changed; OpenCL header 425 added, 5 changed by the renumbering of four op kernels), plugin built
against ggml b8471, ggml libraries of the 2.8.1 bundle (`scripts/termux/lab/k3_check.sh`, logs `~/abi-lab/out-k3/`):

| | S21, Mali-G78, Vulkan | A53, Mali-G68, Vulkan | A53, Mali-G68, OpenCL |
|---|---|---|---|
| `tests/ameva_k3_check`: UPSCALE 8, PAD 7, CONCAT 7, SOFT_MAX 8, MUL_MAT 9 cases | 39/39 | 39/39 | 39/39 |
| `test-backend-ops -b AMEVA0 -o UPSCALE`, `PAD`, `CONCAT`, `SOFT_MAX`, `MUL_MAT` (passed; declined) | 4/4; 18, 15/15; 0, 16/16; 16, 65/65; 147, 371/371; 883 | the same | the same |
| `-o CONV_2D`, `-o GROUP_NORM` | 1568/1568, 2/2 | 1568/1568, 2/2 | declined (no OpenCL translation) |
| `test-backend-ops -b AMEVA0`, every op | 3885/3885 (3596 before this section) | not run | not run |
| `ameva_vae_graph_check`, 16x16 latent: SD 1.x and TAESD, RMS difference / reference RMS | 4.1e-6, 2.1e-6 | 4.1e-6, 2.1e-6 | not applicable |
| the same, 32x32 latent (256x256 image) | 3.9e-6, 7.2e-7 | not run | not applicable |
| `conv_check.sh` (convolution 6 runs, group norm 2 runs, `ameva_qmm_check`) | 23/23 x 6, 14/14 x 2, 516/516 | not run | not run |

Wall time of one `graph_compute` of the rebuilt SD 1.x decoder (random weights; one run each, not alternated, CPU
with 4 threads): S21 32x32 latent 7.07 s, CPU 20.4 s with f32 kernels and 11.9 s with f16 kernels (cores 0-2,4-7);
S21 16x16 latent 1.18 s and 2.71 s in two runs of the same graph (the GPU clock swing of that device), A53 16x16
latent 5.13 s, CPU 70.3 s and 29.9 s (cores 0-2). For scale only: the S21's `sd-cli` (ggml-vulkan, real SD 1.x
weights, `--vae-conv-direct`) reported 133.96 s for the decode of one 256x256 image on the same day; that is
another program and was not alternated with the figure above.

- Not done: sd.cpp built against ggml-ameva (step 3), so no decoded image exists yet; CONV_3D and the RMS form of
  the Wan VAE; tensors above one binding; attention sinks and the ALiBi bias of SOFT_MAX; bilinear and bicubic
  UPSCALE; an OpenCL translation of the convolution and group norm kernels. Nothing of this section ran on Adreno:
  the S25 and the A35 did not answer (connection timed out) and the S20 was running another session's benchmark.
  `ameva_fuse_check`, `ameva_kvq_check` and `ameva_chunk_check` were not rerun.

## 11. sd.cpp on ggml-ameva: a decoded image (step 3 of section 6, 2026-10-08)

- **Build** (`scripts/termux/lab/sd_vae_e2e.sh build`). stable-diffusion.cpp `50d6405` with its ggml `8e800cef`, ggml
  as shared libraries with dynamic backends, `sd-cli` only; the plugin against that ggml's headers and libraries.
  sd.cpp is not changed: it takes backends by registry name (`--backend "cpu,vae=AMEVA0" --vae-conv-direct`) and the
  plugin comes in through `GGML_BACKEND_PATH`. Built on the S25; the S21, A35 and A53 ran the same binaries.
- **ABI.** sd.cpp compiles ggml with `GGML_MAX_NAME=160` (llama.cpp: 64). `ggml_tensor.extra`, which the buffer code
  reads, lies behind `name`, so a plugin compiled with the other size reads it from inside the name.
  `ameva_abi_check` now also compares the name size (a 511-character name set and read back through libggml-base)
  and the names of CONV_2D, GROUP_NORM, UPSCALE, PAD, CONCAT and SOFT_MAX. Negative control on the S25: the plugin
  built without the definition is refused at load ("struct ggml_tensor (GGML_MAX_NAME) differs") and AMEVA0 is not
  registered. `build_ggml_ameva.sh` takes the definition (`--cxxflags`) and the host's device listing
  (`--list-devices-cmd`) for its registration check.
- **Two failures of the first build on the S25, both before any graph ran.** `sd-cli` bound the ggml of `$PREFIX/lib`
  (the Termux clang driver writes that directory first into the RUNPATH): "cannot locate symbol
  gguf_write_to_file_ptr"; the runs put a directory with this build's ggml libraries alone in front. With OpenMP,
  `libggml-base.so` needs `__kmpc_dispatch_deinit`, which `$PREFIX/lib/libomp.so` of that device does not export; the
  build uses `GGML_OPENMP=OFF`.
- **Method** (`sd_vae_e2e.sh run`, `img_psnr.py`). Text encoder and UNet on the CPU in every run with one seed, so
  every run decodes the same latent; two CPU decodes have to give the same file before anything is compared. SD 1.x
  `anime.gguf` with the VAE as stored (decoder parameters 94.47 MB), 256x256 image (32x32 latent), `--cfg-scale 1`,
  euler. PSNR over the 8-bit RGB samples of the PNG files. A run on AMEVA0 counts only without a scheduler split
  (sd.cpp creates the scheduler, with a CPU backend behind it, only when the VAE backend declines a node) and with
  dispatches in `AMEVA_PROFILE`. The 40 dB floor is the script's choice.

| | S25 (Adreno 830) | S21 (Mali-G78) | A35 (Mali-G68) | A53 (Mali-G68) |
|---|---|---|---|---|
| cores of the session, threads, sampling steps | 0-1,4-5, 4, 8 | 0-2,4-7, 4, 4 | 0-2, 3, 2 | 0-2, 3, 2 |
| CPU decodes give the same file | yes (3 runs) | yes (2 runs) | yes (2 runs) | yes (2 runs) |
| AMEVA decodes give the same file | yes (2 runs) | yes (2 runs) | yes (2 runs) | yes (2 runs) |
| AMEVA image against the CPU image: PSNR | 60.47 dB | 61.11 dB | 63.08 dB | 63.08 dB |
| largest difference, differing samples of 196608 | 1 level, 11482 | 1 level, 9906 | 1 level, 6288 | 1 level, 6289 |
| VAE graph on AMEVA0: nodes, dispatches, scheduler splits | 226, 249, 0 | 226, 249, 0 | 226, 249, 0 | 226, 249, 0 |
| decode time, CPU (`latent 1 decoded`) | 15.09, 15.02, 14.05 s | 22.44, 22.58 s | 72.97, 77.00 s | 106.49, 103.00 s |
| decode time, AMEVA0 (Vulkan) | 1.97, 1.98 s | 11.31, 11.26 s | 10.57, 10.27 s | 18.71, 18.86 s |

Every device ran the binaries built on the S25. The latent differs between the columns (sampling steps), so PSNR is
comparable within a column only. Decode times are one process per figure in the order CPU, CPU, AMEVA, AMEVA (S25:
CPU, CPU, AMEVA, CPU, AMEVA), not alternated within a minute, and the GPU clock was not recorded (the S21's governor
moves its figure 2-3x); the AMEVA figure includes creating the pipelines of that process. The S25 image shows the
prompt's subject (a lighthouse on a coast at sunset) and was looked at; the others are compared only. The S20 was
held by another session and did not run.

- The third CPU run and the second AMEVA run on the S25 carried `--tensor-type-rules "^first_stage_model\.=f32"` to
  obtain f32 kernels. It had no effect on this GGUF model (parameter buffer 94.47 MB and the image file unchanged,
  also with unanchored patterns), so they are repeats of the same decode and not an f32 comparison. The f32
  comparison of the decoder remains `ameva_vae_graph_check` (section 10; not run on Adreno).
- **The f16 Vulkan reference of the gate was not obtained: ggml-vulkan stops in this decoder.** The same `sd-cli` with
  `libggml-vulkan.so` of ggml `8e800cef` and `--backend "cpu,vae=Vulkan0"` ends with a segmentation fault (exit 139)
  after "vae compute buffer size" and before any decode time, on the S25 (Adreno 830) and the S21 (Mali-G78), with
  and without `--vae-conv-direct`, and with the committed `ggml-vulkan.cpp` as with the local 60-line change. Text
  encoder and UNet on `Vulkan0` with the VAE on the CPU run in that build (S25, one step, image written). The op it
  stops in is not identified (no debugger on the devices). The backend and its log are kept in
  `~/abi-lab/out-sd/vk-crash` on both devices; `sd_vae_e2e.sh run` adds the `Vulkan0` run when
  `sd-build/bin/libggml-vulkan.so` is present.
- Build note: `ggml-vulkan.cpp` at the build's `-O3` had not finished on the S25 after 67 min (31 min CPU) and was
  stopped; that one file compiled at `-O1` in 1.6 to 3 min. The arithmetic is in the SPIR-V shaders, so the host
  optimisation level does not enter the image.
- Not done: the Vulkan reference above; an f32-kernel image; 512 px and tiled decodes (binding limit); int8 mode;
  decode times alternated with the GPU clock recorded; the S20; the wrapper's selection of the AMEVA VAE plan.

## 12. Tensors above one binding, the UNet on ggml-ameva, 512 px end to end (2026-10-08)

- **Windows.** A tensor above `maxStorageBufferRange` (256 MiB on the Mali devices; the 512 x 512 x 256 f32 tensors of a
  512 px decode are 256 MiB, 64 bytes above what a binding covers from any offset) is given to the kernels in windows
  of whole planes, each bound from the storage alignment below its first plane (`ameva_bind_span`):
  CONV_2D in passes over windows of channel groups (`AmevaConv2d::win_groups`; also when the K-ordered weights exceed
  a binding) and in slices of output planes; GROUP_NORM per window of whole groups, with no kernel change; UPSCALE,
  the binary ops and SOFT_MAX per window of planes; dense MUL_MAT one matrix at a time (the 512 MiB attention weights
  of a 64 x 64 latent). A convolution pass after the first runs the `conv_*_acc` kernels (`-DACCUM`), which start
  their sums from the stored result. The unit table is group-major, so a window is a run of consecutive units and
  the passes add the terms of an output in the order of the one-pass kernel: the result is the same in every bit.
  `ameva_conv_w` got a channel offset. What the windows cannot express is declined with its reason (UPSCALE that
  scales dimension 2 or 3, an operand repeated along them, a mask per plane, 32 source channels or one plane above a
  binding). PAD, CONCAT, RMS_NORM, NORM and FLASH_ATTN_EXT above one binding are still declined.
- **UNet ops.** A census of the SD 1.x UNet (`--backend "cpu,diffusion=AMEVA0" --diffusion-conv-direct`) found two
  declined ops: NORM (288 queries) and TIMESTEP_EMBEDDING (7). Both are kernels now (`ameva_norm`,
  `ameva_timestep_embd`, SPIR-V only). The embedding takes its frequencies from a host table computed with the
  expression of ggml and computes sine and cosine itself (reduction by pi / 4 in three parts, Cephes polynomials):
  the built-in functions of Vulkan are bounded by 2^-11 inside [-pi, pi] only and the arguments reach 1000. REPEAT
  (the LoRA merge) runs on `ameva_pad` with circular padding after the source.
- **Fail-fast.** `AMEVA_STRICT=1` ends the process when an op is declined; `sd_e2e.sh` sets it for every AMEVA run and
  also requires no `## SPLIT` line, no `declined` line and a dispatch count. Before that the census run had ended by
  itself: the scheduler put an in-place MUL on the AMEVA backend whose view source lay in a CPU buffer, and the
  engine stopped with "tensor is not in an AMEVA buffer".
- **Graph wait.** The final wait was 30 s for any graph. A 512 px decode takes 42 to 58 s on a Mali-G68: the A53
  ended with `AC_ERR_LAUNCH_TIMEOUT` (-405). It is now 30 s plus the recorded work at 0.05e9 multiply-accumulates per
  second, at most two hours (`AMEVA_WAIT_TIMEOUT_S` replaces it).

**Op checks** (`conv_check.sh`, Galaxy S21, Mali-G78 r38p0; not run on the other devices). `AMEVA_BINDING_LIMIT`
makes small tensors take the window paths.

| check | result |
|---|---|
| `ameva_conv_check`, six kernel selections | 31/31 each, every element equal to the fma reference |
| the same with `AMEVA_BINDING_LIMIT=131072`, four selections | 25/25 each (three cases declined under that limit), 12 convolutions in windows (15 nodes with `AMEVA_FUSE=0`), equal to the same reference |
| the same with `AMEVA_BINDING_LIMIT=1048576` | 31/31, 2 in windows |
| `ameva_cnorm_check`, fused and `AMEVA_FUSE=0`, without and with `AMEVA_BINDING_LIMIT=1048576` | 15/15 each; 18 and 30 nodes in windows; largest error 0.287 of the bound in all four |
| `ameva_window_check` (limit 65536: UPSCALE, ADD, SUB, MUL, DIV, MUL_MAT, SOFT_MAX) | 18/18, 16 nodes in windows |
| `ameva_unet_ops_check` | 15/15: NORM at most 0.026 of its bound, TIMESTEP_EMBEDDING at most 5.96e-08 from the libm of the CPU, REPEAT equal |
| `ameva_qmm_check` | 516/516 |

**512 px, SD 1.x** (`sd_e2e.sh`; `anime.gguf`: q4_0 linear weights, f16 convolutions; 4 Euler steps, cfg 1, seed 42;
text encoder on the CPU; one process per run, not alternated, GPU clock not recorded). "window nodes" is the count of
nodes computed in windows (`AMEVA_PROFILE window_nodes`). PSNR is against the CPU image of the same row.

| device (cores, threads) | CPU: sampling / decode | VAE on AMEVA: decode, window nodes, PSNR | UNet + VAE on AMEVA: sampling / decode, window nodes, PSNR |
|---|---|---|---|
| S25, Adreno 830 (0-1,4-5; 4) | 102.32 / 82.54 s | 7.71 s, 0, 62.66 dB (max 1) | 21.00 / 7.42 s, 0, 40.69 dB (max 79) |
| S21, Mali-G78 (0-2,4-7; 4) | 90.26 / 56.89 s | 25.68 s, 5, 62.66 dB (max 1) | 90.88 / 21.57 s, 65, 29.70 dB (max 234) |
| A35, Mali-G68 (0-2; 3) | 423.16 / 333.02 s | 42.45 s, 5, 62.66 dB (max 1) | with `--mmap --diffusion-fa` in both runs: 99.41 / 42.06 s (CPU 330.01 / 328.05 s), 5, 33.98 dB (max 216) |
| A53, Mali-G68 r32p1 (0-7; 4) | 164.20 / 152.48 s | 58.38 s, 5, 62.66 dB (max 1) | with `--mmap --diffusion-fa` in both runs: 143.93 / 57.70 s (CPU 135.46 / 147.03 s), 5, 33.33 dB (max 156) |

- Every AMEVA run above: no scheduler split, nothing declined. VAE graph 226 nodes (529 dispatches with windows, 518
  without); UNet + VAE 5522 nodes in 5 graphs. Compute buffers: VAE 704.06 MB; UNet 559.62 MB on the CPU, 1066.62 MB
  on AMEVA (the attention weights of the 64 x 64 latent are one 512 MiB tensor).
- On the S21 the UNet on AMEVA is no faster than the CPU at 512 px (90.88 against 90.26 s). Not profiled; the dense
  MUL_MAT kernel that computes the attention products has one invocation per element.
- A53: the UNet + VAE run without flash attention ended the Termux session during sampling (2.3 GB of buffers on a
  5.5 GB device), so the UNet cells of the A53 and the A35 use `--mmap --diffusion-fa` in both runs (UNet graphs
  5266 nodes; the VAE decode of those runs is untiled as well: 5 window nodes).
- Tiled decode (S25, `--vae-tiling`, the same GPU-sampled latent): CPU 143.43 s, AMEVA 15.45 s, 63.04 dB (max 2); two
  AMEVA runs give the same file.

**What the image comparison measures** (S25, 256 px; "exact" is a CPU run with `--type f32`: with quantized weights
the CPU code of ggml multiplies with activations rounded to 8 bits, with f16 weights with f16 activations).

| pair | 4 steps | 1 step |
|---|---|---|
| exact against CPU, q4_0 weights | 30.53 dB | 39.55 dB |
| exact against AMEVA, f32 weights | 50.16 dB (max 14) | 52.80 dB (max 4) |
| exact against AMEVA, q4_0 weights, f32 activations (`AMEVA_MMQ=0`) | 42.82 dB | 47.12 dB |
| exact against AMEVA, q4_0 weights, int8 activations (the default where the device has integer dot products) | 28.27 dB | 39.23 dB |
| CPU against CPU with `--diffusion-fa`, q4_0 weights | 34.34 dB | |
| q4_K file: exact against CPU / AMEVA int8 / AMEVA f32 activations / AMEVA f32 weights | 22.28 / 28.54 / 34.17 dB / - | 28.33 / 35.93 / 44.22 / 51.87 dB |

- The CPU image of a quantized model is itself 22 to 40 dB from the exact one, so a PSNR against it does not measure
  the backend. Against the exact reference the backend is at 50 to 53 dB with f32 weights. With quantized weights and
  f32 activations it is at 34 to 47 dB; the text encoder of those runs is the quantized one of the CPU, which that
  of the reference is not, and that part of the difference was not separated.
- `ameva_mmq` (int8 activations) costs 8 to 14 dB against exact at one step. It is the default on the S25, S20 and
  A35 for many activation rows. Whether the diffusion model should run with `AMEVA_MMQ=0` is not decided.

**Other models and weight types** (S25, UNet + VAE on AMEVA, no split, nothing declined in any of them).

| run | CPU: sampling / decode | AMEVA: sampling / decode | PSNR against the CPU image |
|---|---|---|---|
| SD 1.x `--type q8_0`, 256 px, 4 steps | 13.97 / 15.39 s | 6.27 / 1.88 s | 39.98 dB |
| SD 1.x `--type q4_K`, 256 px, 4 steps | 25.82 / 14.96 s | 16.21 / 1.90 s | 23.23 dB (see the table above) |
| SD 1.x `--type f16`, 256 px, 4 steps | 14.08 / 14.95 s | 8.67 / 2.11 s | 44.29 dB |
| SDXL base 1.0 Q4_0 (gpustack GGUF), 512 px, 4 steps | 114.84 / 63.09 s | 21.35 / 9.59 s | 51.30 dB (max 11); 14750 nodes, 37298 dispatches |
| SDXS 512-DS (q8_0), 512 px, 1 step | 7.13 / 6.83 s | 1.91 / 0.76 s | 36.69 dB |

**LoRA** (S25, 256 px, a synthetic rank-4 adapter on the 48 self-attention projections,
`scripts/termux/lab/make_test_lora.py`; it is a test input, not a trained adapter).

- At run time (`--lora-apply-mode at_runtime`): 192 of 192 tensors applied, UNet graphs 6546 nodes instead of 5522, no
  split. q4_0 weights: AMEVA against CPU 30.71 dB; the adapter moves the image by 19.33 dB (CPU) and 19.45 dB (AMEVA)
  from the run without it. f16 weights: 45.84 dB; 19.75 and 19.72 dB.
- Merged at load (`immediately`): **not on AMEVA.** The merge adds into the f16 weight in place; the engine has no f16
  binary kernel and no op may write a repacked weight, so the op is declined and `AMEVA_STRICT` ends the run
  (`--params-backend cpu` does not change that). On the CPU the merge works with f16 weights (44.65 dB from its own
  run-time form) and stops in ggml (`GGML_ASSERT(src1->type == GGML_TYPE_F32)`) with the q4_0 file.

- Not done: an MMDiT model. A census of Z-Image-Turbo Q2_K on the S21 loaded 5398 MB of parameters (text encoder
  2766 MB in RAM, diffusion model 2472 MB on AMEVA) and the process was ended by the system during text encoding,
  before any graph of the diffusion model was built: no op list. Also not done: the op checks on the A35, A53, S25 and
  S20; `test-backend-ops` and LLM text parity after these changes; the S20; an OpenCL translation of the new
  kernels; the cause of the UNet time on the S21; LoRA merged at load on AMEVA.
  (Section 13 has the MMDiT run and the merge at load. Its log shows that the run described here ended after text
  encoding had completed, at the load of the diffusion model, not during text encoding.)

## 13. An MMDiT model on ggml-ameva and the LoRA merge at load (2026-10-08)

Z-Image-Turbo (`z_image_turbo-Q2_K.gguf`, 2472 MB of parameters), text encoder Qwen3-4B Q2_K, VAE
`z_image_ae.safetensors`, sd-cli `50d6405` unchanged. Script: `scripts/termux/lab/mmdit_census.sh` (one run under a
guard that ends sd-cli when MemAvailable falls below 600 MB, with the op census and the memory of each phase).

- **Flags.** `--mmap --backend te=cpu,diffusion=AMEVA0,vae=AMEVA0 --max-vram AMEVA0=1.5 --stream-layers
  --params-backend diffusion=cpu --vae-conv-direct --diffusion-conv-direct`. The key of `--max-vram` is the backend's
  device name. `--stream-layers` is ignored unless the DiT parameters stay on the CPU (`--params-backend
  diffusion=cpu`; with `--mmap` they are the mapped file). sd-cli lowers the budget to the free memory the backend
  reports and cuts the DiT graph to it: 283 MB and 13 segments on the A35, 384 MB and 8 segments on the S21 at 256 px
  (19 at 512 px). Every segment's weights are sent to the backend, and repacked, in every step.
- **Census: nothing declined.** Ops of the DiT and VAE graphs on the backend: ADD, MUL, REPEAT, CONCAT, NORM,
  RMS_NORM, GROUP_NORM, MUL_MAT, SCALE, CONT, SOFT_MAX, CONV_2D, UPSCALE, TIMESTEP_EMBEDDING, UNARY, GLU. No scheduler
  split and no declined op in any run below; the S21 runs had `AMEVA_STRICT=1`. The text encoder ran on the CPU in
  every run (`te=cpu`): it is not part of this census.

| device, run | graphs / nodes / dispatches | text encoder (CPU) | sampling | decode | CPU: sampling / decode | PSNR against the CPU image |
|---|---|---|---|---|---|---|
| A35, 256 px, 1 step (plugin of `bf0ec99`) | 14 / 2309 / 4710 | 42.08 s | 79.92 s | 11.67 s | 372.72 / 119.26 s | 40.86 dB (max 22 of 255) |
| S21, 256 px, 1 step, strict (plugin of this section) | 9 / 2262 / 4397 | 5.45 s | 55.63 s | 8.90 s | 83.47 / 18.78 s | 39.55 dB |
| S21, 512 px, 8 steps, strict | 216 / 17730 / 45825 | 6.95 s | 1994.08 s | 36.37 s | not run | not run |

  One process per run, not alternated, GPU clock not recorded: the times are not a benchmark. A35: ssh core set 0-2,
  4 threads, and sd-cli's times (monotonic clock) summed to 134 s where the wall clock gave 609 s. The PSNR is against
  the CPU run of the same Q2_K weights (see section 12 on what that reference measures); the A35 multiplies with int8
  activations (`ameva_mmq`, 213 MUL_MATs), the S21 with f32 (`ameva_qmv`, 208). The 512 px image shows the prompt (a
  lighthouse on a rocky coast at sunset); 5 of its VAE nodes ran in windows. 8 steps at 512 px repacked 25187 MiB of
  weights: the streaming, not the kernels, sets the 249 s per step.
- **The plugin on the S21 and A35 was older than the source** (`out-sdw4`, built before REPEAT was added). With it
  REPEAT was declined, the scheduler split the graph, and ggml's CPU REPEAT stopped the process
  (`ggml-cpu/ops.cpp:1708: GGML_ASSERT(ggml_can_repeat(src0, dst))`, exit 134): for this graph a declined op is a
  crash, not a slower path.
- **Memory.** Lowest MemAvailable per phase (text encoder / DiT / VAE): A35 1026 / 692 / 1075 MB (2521 MB before the
  run), S21 2125 / 1355 / 1807 MB at 256 px and 2280 / 1181 / 1661 MB at 512 px. The process's anonymous memory peaks
  at 1563-1580 MB: the token embedding, which sd-cli converts to f32 (1483.75 MB); the other 385 text-encoder tensors
  are the mapped file (1246.57 MB).
  - sd-cli releases a module's parameters after use only when that module's residency is disk
    (`ModelManager::release_params_storage_blocks`). With the flags above the 1483.75 MB buffer is released after
    `generate_image completed`: it is allocated through the DiT and the VAE. The A35 moved it to zram (the process's
    VmSwap 1552-1560 MB during the DiT, 8-15 MB anonymous resident); the S21 kept it resident or swapped part of it.
  - `--params-backend te=disk,diffusion=cpu` does release it, and reads all 386 text-encoder tensors into anonymous
    memory instead of mapping them: on the A35 the guard ended the run 14 s into text encoding (MemAvailable 495 MB;
    the process at 798 MB anonymous and 1924 MB swapped).
  - termux-diffusion 2.0.1 `--isolate-text-encoder` / `--sequential-stages` is parsed and stored
    (`GenerationResult.isolated_stages`) and nothing acts on it; `SequentialClusterPipeline.execute_stages` is called
    by a unit test only. sd-cli has no option to write or read a computed condition, so a text-encode-then-exit
    process followed by a DiT process cannot be run today.
- **LoRA merged at load on the backend (f16 weights).** stable-diffusion.cpp merges with
  `ggml_add_inplace(weight, diff)` on the f16 weight. New op kernel `binary_f16` (`shaders/ameva_binary.comp -DF16`):
  f16 + f32 or f16 -> f16, in place or not, values converted as the CPU does, halves of a word stored with atomics.
  Only ADD is accepted for f16. The f32 kernel's SPIR-V is the same file as before (SHA-256 `f46e519d...`); the
  regenerated header only adds the new module (S21, spirv-val).
  - Op check (`tests/ameva_unet_ops_check`, 9 new cases: the size of an attention projection, a 3 x 3 x 4 x 5 kernel
    with rows of odd length, broadcast diffs, diffs 1e-4 and 1e3 times the weights): equal to
    `f32 -> f16(f16 -> f32(a) + b)` in every bit on the S21 and the A35 (24/24 with the earlier cases).
    `test-backend-ops -o ADD` on the S21: 32 f16 cases pass on the backend (12 broadcast shapes declined, as for f32).
  - Regression of the other op checks with this source (`conv_check.sh kernels`), S21 and A35: conv 31/31 (25/25 and
    31/31 under `AMEVA_BINDING_LIMIT`), cnorm 15/15 x4, window 18/18, qmm 516/516. `test-backend-ops` on the S21,
    default settings: 3969/3969 (the pass with `AMEVA_REPACK_ALL=1` was started and not read for this section).
  - End to end (`scripts/termux/lab/lora_merge_e2e.sh`: SD 1.x `anime.gguf --type f16`, 256 px, 4 steps, the synthetic
    adapter of section 12, `AMEVA_STRICT=1`): no declined op, no split, 6226 nodes in 7 graphs.

| pair | S25 | S21 |
|---|---|---|
| merged at load: AMEVA against CPU (floor 40 dB) | 45.75 dB | 45.75 dB |
| AMEVA: merged against no adapter (has to be below 30 dB) | 19.72 dB | 19.72 dB |
| AMEVA: merged against applied at run time | 62.20 dB (max 2 of 255) | 62.20 dB |
| CPU: merged against no adapter | 19.77 dB | 19.77 dB |
| CPU: merged against applied at run time | 44.65 dB | 44.65 dB |
| applied at run time: AMEVA against CPU | 45.84 dB | 45.84 dB |

  - Not available: the merge into quantized weights. stable-diffusion.cpp casts the weight to f32, adds and copies
    back to the quantized type, which reads and writes a repacked weight (declined), and its CPU path stops in ggml
    with a q4_0 file as well; `--lora-apply-mode auto` applies at run time for quantized models (section 12).
- Not done: a CPU reference for the 512 px image; Z-Image-Turbo on the S25 and A53 (the S25 has no copy of the
  model); the Q4_0 file; the text encoder on the backend; a two-process run that frees the text encoder; the LoRA
  merge end to end on the A35 (its f16 CPU reference ran 20 min in swap and was stopped) and the A53; `binary_f16`
  on OpenCL (translated, not run); f16 ADD above one binding (declined); LLM text parity after this change.
