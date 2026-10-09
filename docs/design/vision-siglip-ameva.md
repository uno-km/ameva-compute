# Vision Stage 4: the SigLIP encoder and projector of llama.cpp mtmd on ggml-ameva

- Written 2026-10-09, branch `perf/opencl-mali`. Follows `AMUDA_ENGINE_UNIFICATION_PLAN.md` (Phase 4, vision / mtmd).
- Model: SmolVLM-500M-Instruct Q8_0 with its mmproj (SigLIP-base vision tower + Idefics3 projector), the files already
  on all five phones (`~/.cache/termux-vision/models/smolvlm-500m-q4`, mmproj sha256 equal on every phone). Moondream2
  is on none of them and models are not downloaded to the fleet; SmolVLM's encoder is the SigLIP family named in the
  task. No model was downloaded; about 9 MB of plugin, tool, gguf-py and scripts went to each phone (plus 2.8 MB of
  sources to the S25, the build phone), and each phone made its own F32 reference (393 MB) from its mmproj.
- Status: the whole encoder graph (patch embedding, 12 transformer layers, projector) runs on AMEVA0 with no operator
  left to the CPU, on all five phones, within the parity gate. Faster than the CPU on the S25, the S20 and, with
  OpenCL, the S21; slower than the CPU on the S21 with Vulkan (section 4). Flash attention was the bottleneck and was
  reworked for long queries (section 3), which also speeds up LLM prompts.

## 1. How termux-vision computes today

`termux_vision/vlm/engine.py` starts `termux-vlm-cli` / `llama-mtmd-cli` as a process with `--mmproj` and, by default,
`--no-mmproj-offload` (`no_mmproj_offload=True`): the vision encoder runs on the CPU even when the text model is on a
GPU. Inside llama.cpp (b8471, `tools/mtmd/clip.cpp`) the encoder is one ggml graph; clip.cpp opens the backend named
by `MTMD_BACKEND_DEVICE`, else the first GPU device, else the CPU. **That choice falls back without an error**: an
unknown name gives a warning and the default GPU (the bundle's Vulkan0), no GPU gives the CPU. Operators the backend
declines go to the CPU through the scheduler with only a warning list.

Routing termux-vision to AMEVA needs `GGML_BACKEND_PATH=<libggml-ameva.so>`, `MTMD_BACKEND_DEVICE=AMEVA0` and the
offload switch on; not wired into termux-vision yet.

## 2. Graph census and the kernels it reaches

SmolVLM-500M mmproj: image 512 x 512, patch 16 (1024 positions), width 768, 12 heads of 64, FFN 3072 (GELU), 12
layers, pixel shuffle by 4 to 64 tokens of 12288, projection to 960. Q8_0 weights for the 72 layer matrices and the
projector, F32 for everything else including the patch-embedding kernel. clip.cpp builds 394 nodes; on AMEVA0:
`graph splits = 1`, no unsupported operator, flash attention enabled.

| Graph part | ggml ops | AMEVA kernel |
| :--- | :--- | :--- |
| patch embedding (`ggml_conv_2d`, stride 16) | IM2COL + MUL_MAT f32 | `im2col` (Stage 2), f32 `ameva_qmm` |
| LayerNorm (x 25) | NORM, MUL, ADD | `ameva_norm`, `ameva_binary` |
| Q, K, V, O, FFN (Q8_0) | MUL_MAT | repacked weights, `ameva_mmq` (int8 dot) where the device has it, else the f32 tile |
| attention | FLASH_ATTN_EXT (no mask, f32 K/V) | `ameva_flash_attn` (section 3) |
| GELU, pixel shuffle, projector | UNARY, CONT/CPY, MUL_MAT | `ameva_unary`, `ameva_cpy`, `ameva_mmq` |

The task named `ameva_conv` (Stage 1) for the patch embedding and `mm_f16a` (Stage 2) for the projections. Neither is
reached by this graph: clip.cpp expresses the patch embedding as IM2COL + MUL_MAT (not CONV_2D), and the projections
have Q8_0 weights (the repacked integer-dot path, not f16). No new operator kernel was needed.

## 3. Flash attention for long queries

GPU time per op kind on the S25 (`AMEVA_PROFILE=2`) before this stage: FLASH_ATTN_EXT 71 % of the encoder, 43.9 ms per
layer for 1024 x 1024 x 64 x 12 heads (about 73 GFLOP/s), against about 1.1 TFLOP/s in the MMQ matmuls. Without flash
attention the per-head f32 MUL_MATs were slower still (2.9 s per image). The tile kernel read Q and K from shared
memory for every multiply (two vec4 loads per dot product) with 16 rows per 128 invocations.

Change (`shaders/ameva_flash_attn.comp`): RPI rows per invocation, so each staged K / V vec4 serves RPI rows; an
optional head-loop unroll (DUNROLL); same sums in the same order (the variants with equal BK give bit-identical
output). Variant list and the host's choice in `scripts/build_ameva_kernels.py` (FA_VARIANTS): among the variants with
the smallest DMAX that fits, the first whose rows per tile (BQ) do not exceed the tile rows, else the fewest rows
(token generation keeps the previous tile). `AMEVA_FA_VARIANT=<i>` forces one, `AMEVA_FA_VARIANTS=` builds an
experiment list (`scripts/termux/lab/vision_fa_sweep.sh`).

FLASH_ATTN_EXT GPU ms per dispatch, SigLIP 1024 x 1024, d 64, Vulkan (DMAX, BQ, BK, RPI):

| Tile | S25 Adreno 830 | S20 Adreno 650 | S21 Mali-G78 | A53 Mali-G68 | A35 Mali-G68 (throttled ssh) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| previous kernel (64, 16, 32, 1) | 43.9 | | | | |
| (64, 16, 32, 1) this source | 37.0-38.3 | 135.1 | 126.8 | 254.0 | 315.1 |
| (64, 32, 16, 2) **chosen** | 21.8-22.3 | 112.9 | 126.5 | 163.6 | 311.3 |
| (64, 64, 16, 2) | 19.7-19.8 | 160.4 | 174.7 | 239.7 | 439.7 |
| (64, 64, 16, 4) | 20.8-20.9 | 211.7 | 197.7 | 236.0 | 485.9 |
| (64, 64, 16, 1) | 21.4 | 391.7 | 148.9 | 261.6 | 372.9 |
| (64, 32, 32, 2) full unroll | 86.8 | | | | |

The fully unrolled head loop with RPI 2 doubled the time on the Adreno 830; smaller key blocks (less shared memory)
helped there. (64, 64, 16, 2) would gain another 11 % on the Adreno 830 only; per-device choice is not built.

Checks: `test-backend-ops -o FLASH_ATTN_EXT -b AMEVA0` 1048/1048 on the S25, S20 and S21 with the chosen list, and
224/224 hsk=64 cases for each of five forced d64 variants on the S25. LLM prompt, Qwen2.5-0.5B Q4_K_M, 368 tokens,
S25 cores 0-1,4-5, alternating: 1194-1220 t/s against 1108-1123 t/s with the previous tile forced, same text.

## 4. Parity and encode time

Reference (`tests/ameva_vision_parity.cpp`): the CPU backend on the mmproj with every weight dequantized to F32
(`scripts/termux/lab/gguf_dequant_f32.py`), so the reference has no quantization. The CPU on the Q8_0 mmproj quantizes
its activations to q8_0 and lands at relative L2 1.92e-2 from it on every phone (min token cosine 0.99873): the floor.
Gate: AMEVA's relative L2 within 1.5 x the floor. A fixed cosine gate of 0.999 would fail the CPU itself. The tool
also checks the backend clip.cpp opened (no fallback), the GPU operations completed, the operators left to the CPU,
and the library files mapped. Caption: 32 greedy tokens by the text model on the CPU from each set of embeddings.

Image `tools/mtmd/test-1.jpeg` (64 output tokens), `--reps 3` after clip.cpp's warm-up, ms per image:

| Phone | Path | AMEVA ms | CPU Q8_0 ms | floor ratio | caption |
| :--- | :--- | ---: | ---: | ---: | :--- |
| S25 (cores 0-1,4-5, 832 MHz) | Vulkan, MMQ | 481-516 | 1930-2893 | 1.17 | identical |
| S25 | Vulkan, `AMEVA_MMQ=0` | 833-898 | 2719-2809 | 0.03 | identical |
| S20 (cores 0-7, 305-587 MHz) | Vulkan, MMQ | 2778-2814 | 9669-9896 | 1.05 | identical |
| S21 (cores 0-7) | Vulkan (f32 tile) | 2572-2589 | 1936-1974 | 0.03 | identical |
| S21 | OpenCL (`AMEVA_BACKEND=opencl`) | 1784-1805 | 2071-2088 | 1.10 | differs at token 4 (lead 0.024) |
| A35 (ssh: cores 0-2) | Vulkan, MMQ | 4941-5032 | 26814-27028 | 1.09 | identical |
| A35 | OpenCL | 3363-3367 (cold 6206) | 26918-29128 | 1.10 | differs at token 4 (lead 0.024) |
| A53 (ssh: cores 0-2) | Vulkan (f32 tile) | 5311-5349 | 30993-31275 | 0.03 | identical |
| A53 | OpenCL | 4295-4306 (cold 7358) | 30784-31072 | 1.10 | differs at token 4 (lead 0.024) |

Reading:

- Every row passes the gate, with the whole encoder on AMEVA0 (`graph splits = 1`, no unsupported operator, 300 GPU
  operations per encode) and the bundle's libraries mapped (`mapped` lines of the log).
- The integer-dot paths (S25, S20, A35 Vulkan, Mali OpenCL) carry the same kind of error as the CPU's own Q8_0 path
  (int8 activations; floor ratio 1.05-1.17); the f32 paths agree with the F32 reference to 5.6e-4 relative L2.
  Greedy captions flip at near-ties (logit lead 0.012-0.024 where they did), so the caption is reported, not gated.
  The three Mali OpenCL runs give the same embeddings to the printed digits.
- S25: 4.4 x the CPU (medians); the encoder is FLASH_ATTN_EXT 22.0 ms x 12 plus the MMQ matmuls. S20: 3.5 x.
- S21: Vulkan is slower than its CPU (0.76 x), OpenCL 1.15 x. On the Mali-G68 phones OpenCL is 1.47 x (A35) and
  1.24 x (A53) Vulkan. Their CPU columns ran on cores 0-2 only (the ssh session's cpuset), so the 5-9 x against the
  CPU there is not a whole-device CPU comparison.
- Mali flash attention stays at 25-40 GFLOP/s: about half of the S21 encode.

## 5. Next, by measured value

1. Mali flash attention (S21: 126 ms per layer, half the encode): a Mali-specific tile or the OpenCL path.
2. termux-vision wiring (section 1) and an ameva-runtime bundle with this plugin.
3. Per-device flash-attention choice (11 % on the Adreno 830), a DMAX 128 prompt tile (not measured).
