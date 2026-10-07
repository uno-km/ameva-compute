# Diffusion Stage 1: VAE decode bottleneck and convolution / channel-norm kernels (2D and Wan 2.1)

- Written 2026-10-07, branch `perf/opencl-mali`. Follows `AMUDA_ENGINE_UNIFICATION_PLAN.md` (diffusion phase).
- Sources read: local `dev/termux/termux-diffusion` only (sd.cpp `50d6405`, ggml `8e800cef` plus an uncommitted 60-line
  `ggml-vulkan.cpp` change). No remote fetch. The A35 device binary is the 2.7.9 bundle `sd-cli` (older sd.cpp layout),
  so line numbers of the local tree are approximate for it.
- Status: measurements and design. No kernel is written yet. Section 5 lists what is predicted and not measured.

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

- That staging from the source tensor keeps the tile kernels' throughput. Three op-count predictions missed on
  2026-10-05, so this is tested first, in isolation.
- Exact mode with f16 weights is assumed to match the q8_0 / f32 timing (weights are small and cached per tile).
- Image effect of int8 activations through about 36 stacked convolutions.
- Adreno (S25, S20) and the other Mali devices: nothing in this document was measured there.
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
