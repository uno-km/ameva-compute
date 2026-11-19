# BitNet b1.58 on the AMUDA runtime (Stage 3)

Status 2026-10-09: the BitNet adapter (`adapters/bitnet`, the native core of termux-bitnet) runs its GPU layers on
`ac_runtime` only. The raw-Vulkan `VulkanBitNetEngine` (own instance, device, pipelines, descriptors, command buffers,
private shader set) is removed (`860b80e`).

## Structure

| Part | File |
|---|---|
| Engine (buffers, kernels, one command batch per token) | `adapters/bitnet/src/core/ac_bitnet_engine.{h,cpp}` |
| Kernels | `src/shaders/bitnet_{gemv_i2_s,norm_q8,rope,attn_decode,gemv_f16}.comp` |
| Kernel header (SPIR-V, SHA-256, bindings, local size) | `scripts/build_bitnet_kernels.py` -> `adapters/bitnet/src/core/bitnet_kernels_spv.h` (generate on a device: glslang 16.3.0 + spirv-val; regenerates bit-identically) |
| Core (CPU path, GGUF, tokenizer, engine calls) | `adapters/bitnet/src/cpu/llama_bitnet_core.cpp` |
| Build on Termux | `scripts/termux/build_bitnet_adapter.sh [--kernels]` (S25-built binaries run on S21 and S20) |
| CPU vs GPU check | `ac_bitnet_parity` (`adapters/bitnet/src/cli/ac_bitnet_parity.cpp`) |

Per token and layer: `norm_q8(x, attn_norm)` -> Q, K, V GEMV (V into its cache row) -> RoPE (K into its cache row) ->
attention -> `norm_q8(att, attn_sub_norm)` -> output GEMV added to x -> `norm_q8(x, ffn_norm)` -> gate, up GEMV ->
`norm_q8(relu^2(gate) * up or silu(gate) * up, ffn_sub_norm)` -> down GEMV added to x. 13 dispatches, 8 explicit
barriers. With the logits requested, the final norm and the f16 LM head (slices of at most 128 MiB) join the batch.

## Numerical contract

The GPU computes what the CPU path of the same library computes (W1.58A8):
- activations: RMS norm, then int8 absmax quantization (round half away from zero, clamp +-127), per vector;
- i2_s GEMV: exact integer sum `sum (c - 1) q` (as `sum c q - sum q`), then `float(sum) * (act_dequant * w_scale)`;
  bit-identical to `ggml_vec_dot_i2_i8_s` + the CPU scale products for the same activations. Code 3 counts as +2 on
  both paths (the reference dequantizer maps it to 0; the i2_s quantizer does not emit it). The integer-dot, emulated
  and every work-split variant give identical logits (checked on S25, S21, S20);
- RoPE factors computed on the host by the CPU path's expression (GPU sin / cos are unspecified outside [-pi, pi]);
- roundings that the CPU path keeps separate are marked `precise` (no multiply-add contraction); the CPU reference is
  built with `-ffp-contract=off`.

What still differs: summation order of norms, attention and the LM head, and the GPU's exp / sqrt / division. Through
the int8 rounding these flip single activation values, and the effect grows through 30 layers.

## Verification method and gate

`ac_bitnet_parity --save ref.bin --backend cpu` (prompt + greedy tokens of the CPU) then
`--load ref.bin --backend gpu`: teacher-forced, KL(CPU || GPU) of the softmax per position, arg-max agreement and the
reference's top-2 lead at every disagreement.

The CPU path's own sensitivity (S25, same tokens):

| vs CPU reference (`-ffp-contract=off`) | positions | KLD mean | KLD max | arg-max | lead at the flips |
|---|---|---|---|---|---|
| CPU build `-ffp-contract=fast` | 110 | 1.34e-3 | 5.89e-2 | 108/110 | 0.069, 0.174 |
| CPU scalar build | 30 | 9.44e-4 | 5.82e-3 | 29/30 | 0.063 |

Gate (`ac_bitnet_parity` defaults): KLD mean <= 2.7e-3 (twice the FMA floor) and no decisive disagreement, i.e. no
arg-max difference where the reference leads its second choice by >= 0.5 logit. Positions 0 and 1 give KLD ~1e-12 on
S25 (no flip yet), the direct check that the kernels carry no systematic error.

## Results (bitnet-2b-ggml-model-i2_s.gguf, final build, each device's own CPU reference)

| Device | GPU | GEMV | KLD mean (110 pos.) | arg-max | decisive | GPU ms/token | CPU path ms/token |
|---|---|---|---|---|---|---|---|
| S25 | Adreno 830 | integer dot | 8.76e-4 | 110/110 | 0 | 36.0 | 252.6 (cores 0-1,4-5) |
| S21 | Mali-G78 | emulated | 1.87e-3 | 109/110 | 0 | 66.2 | 127.4-136.4 |
| S20 | Adreno 650 (Turnip) | integer dot | 1.45e-3 | 109/110 | 0 | 103.8 | 160.8-207.8 |

The one S21 / S20 disagreement is position 13 (reference lead 0.069), the same flip the CPU FMA build makes. A35 and
A53 were offline. The CPU path is this library's (single-threaded i2_s GEMV), not bitnet.cpp. ms/token includes the
LM head on every token (teacher-forced).

GPU time per kernel at position 60 (`AC_BITNET_PROFILE=1`):

| Device | total | i2_s GEMV | LM head | norm_q8 | attention | first build total |
|---|---|---|---|---|---|---|
| S25 | 35.1 | 14.2 | 16.9 | 2.9 | 1.0 | 37.0 |
| S21 | 63.5 | 32.6 | 22.9 | 5.2 | 2.6 | 94.8 |
| S20 | 103.4 | 48.7 | 35.9 | 8.3 | 9.1 | 109.3 |

Text: `ac-bitnet-cli -d gpu`, 64 greedy tokens on "Write a short paragraph about the history of Paris, France." gives
coherent English on all three devices (the continuations differ after the first sentence, as the arg-max flips allow).

## Switches

- `AC_BITNET_GEMV=auto|dot|emu`: GEMV through `VK_KHR_shader_integer_dot_product` (auto where enabled) or the
  emulated form (two 32-bit multiplications per four products, as `ameva_mmvq -DEMU_DOT`).
- `AC_BITNET_GEMV_SHAPE=l32w1|l16w2|l8w4` (default l32w1): lanes per row x words per load. l16w2 is equal within
  noise; l8w4 is 10-20 % slower on S25 and S20.
- `AC_BITNET_LMHEAD_SHAPE=w1|w2|w4` (default w1): words per load of the LM head. w4 costs 1.8x on S20 (Turnip) and
  gains 1-3 % on S25 / S21.
- `AC_BITNET_PROFILE=1`: GPU time per kernel for every token (stderr).
- `--chunk-layers N`: submit every N layers (bounds one submission's GPU time).

## Open items

- The tokenizer is a greedy longest match over the byte-level vocabulary, not the BPE merges of the reference
  tokenizer: prompts can be split differently from llama.cpp / bitnet.cpp (both paths share it).
- Mali (S21, A53): the emulated GEMV is ALU-bound (32.6 ms for 498 MB); the OpenCL backend with `arm_dot` is the
  next candidate.
- LM head: 626 MB of f16 read per generated token, the largest cost on S25 and S21 and near each device's achieved
  bandwidth. A smaller LM head would change the model's output precision (a product decision, CPU and GPU together).
- Attention: one workgroup per head; grows with the context (S20 9.1 ms at position 60).
- A35 and A53: not yet run (offline on 2026-10-09).
