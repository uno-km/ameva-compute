# TTS Stage 5: Kokoro-82M on ggml-ameva

- Written 2026-10-10, branch `perf/opencl-mali`. Follows Stage 4 (`vision-siglip-ameva.md`).
- Model: Kokoro-82M v0.19, the sherpa-onnx package `kokoro-int8-en-v0_19` (onnxruntime dynamic int8 quantization of
  hexgrad's export), already on all five phones (`model.int8.onnx` sha256 `e0530c50...` on every phone checked). No
  model was downloaded or copied to a phone: each phone converts its own file to GGUF in about 2 s
  (`adapters/tts/kokoro/convert_onnx_to_gguf.py`, python3 + numpy only).
- Synthesis is headless: every run writes WAV / raw float files to disk; nothing is played on a phone. The files are
  checked by script (`scripts/termux/lab/tts_wav_check.py`: RIFF header, rate, channels, duration, RMS, log-mel
  distance to a reference).

## 1. How termux-tts computes today

termux-tts runs Kokoro through sherpa-onnx on the CPU (int8 weights and int8 dynamic activation quantization, its own
espeak-ng G2P), and VITS / Piper models through sherpa-onnx or sherpa-ncnn. None of it uses ggml or AMUDA.

## 2. The network as ggml graphs (adapters/tts/kokoro/kokoro.cpp)

Two graphs on one backend, no scheduler and no second backend: before computing, every node is checked with
`ggml_backend_supports_op`, and one unsupported node fails the call with its op and name.

1. Text side (one graph): PL-BERT (ALBERT, one shared layer applied 12 times), the duration encoder (3 x biLSTM +
   AdaLayerNorm), the duration LSTM and projection, and the text encoder (3 x conv + LayerNorm + leaky ReLU, biLSTM).
2. The host rounds the durations (round half to even, at least one frame, as the export) and lays out the frame
   index list. Host work besides: voice-row lookup, the SineGen noise (splitmix64 + Box-Muller, the same function as
   the reference, 4 threads), the inverse-STFT window sum (a function of the length only).
3. Synthesis (one graph): frame alignment (GET_ROWS), the shared LSTM, F0 / energy predictors (AdaIN residual blocks
   with transposed-conv up-sampling), the decoder, the harmonic source, its STFT, the iSTFTNet generator (two
   transposed-conv up-samplings, 6 + 2 AdaIN residual blocks with Snake activations, kernels 3 / 7 / 11, dilations
   1 / 3 / 5) and the inverse STFT.

Operations the export has and ggml has not, composed of ggml ops (exact up to float rounding unless stated):

| export op | ggml |
|---|---|
| LSTM (bidirectional, gates i o f c) | input projection of all steps as one MUL_MAT (backward direction on the time-reversed input), then per step one batched MUL_MAT and the gate ops for all directions; independent LSTMs of equal length share the steps |
| ConvTranspose1d, kernel = m x stride | MUL_MAT giving every tap of every output channel, overlap-add of m shifted blocks |
| depthwise ConvTranspose1d (k 3, s 2, p 1, op 1) | even / odd outputs x[i] w1, x[i] w2 + x[i+1] w0 interleaved by CONCAT |
| InstanceNorm + AdaIN affine | GROUP_NORM with one group per channel, MUL and ADD in place (fused by the engine) |
| Conv1d | CONV_2D direct with KH = 1 |
| Linear over a sequence (ALBERT, LSTM input, transposed-conv taps) | 1x1 CONV_2D: the weight [Cout, Cin] is the kernel [1, 1, Cin, Cout] in the same memory; on AMEVA0 this is the tiled convolution kernel instead of the GEMV-oriented f16 MUL_MAT path (14-15 GFLOP/s on the S20) |
| ALBERT attention | FLASH_ATTN_EXT without mask |
| STFT (20-point, hop 5) | 20-tap CONV_2D with a cos / -sin basis whose bin 0 and 10 imaginary rows are exactly zero |
| atan(im / re) + quadrant fix | exact selections (STEP products) and an odd polynomial on |u| <= tan(pi/8), max error 1.9e-8 rad |
| reflection padding | GET_ROWS of a reflected index list, or CONCAT of the mirrored column |
| NewGELU | composed from TANH and products (ggml-cpu's GELU goes through an f16 table) |

## 3. What ggml-ameva needed

- Unary kernel: SIN, COS (ameva_sincos: fused reduction by pi/2, so the 1e5..1e6 radian SineGen phases stay
  accurate), SQR, SQRT, LEAKY_RELU, SGN, STEP, FLOOR.
- CUMSUM (`ameva_cumsum.comp`, one invocation per row, ggml-cpu's sequential order, so the sums are bit-equal).
- UPSCALE bilinear (half-pixel, ggml-cpu's formula) on dimensions 0 and 1.
- SNAKE chain fusion (`ameva_snake.comp`): MUL(x, a) -> SIN -> SQR -> MUL(., 1/a) -> ADD(., x) written in place is
  one dispatch.
- ac_runtime: the Vulkan batch recorder reserved its per-dispatch buffer list one dispatch at a time
  (`reserve(size + count)`), which reallocates and copies the whole list on every dispatch: quadratic in the dispatches
  of one graph. Kokoro's text graph of a 26 s utterance (19k dispatches) spent about 120 us per dispatch there; with
  geometric growth about 25 us. This affects every long graph, not only TTS.

## 4. Verification

References (PC): `adapters/tts/kokoro/reference_onnx.py` rewrites the int8 file into a float graph with the converter's
weights and injected noise (onnxruntime, one thread). Against the shipped int8 model it gives the same durations, F0
within 0.74 % (median 3.5 cents, no voiced / unvoiced flip), log-mel distance 4.3 dB (two int8 runs differing only in
noise: 1.6 dB). Phonemes: `g2p_tokens.py` (espeak-ng through phonemizer, kokoro v0.19's replacements).

Device runs: `tests/ameva_tts_parity.cpp` (`scripts/termux/lab/tts_parity_run.sh`). Runs: CPU with F32 weights (the
reference), CPU with F16 weights (the floor: ggml-cpu rounds activations to f16 for f16 weights), AMEVA0 with F16
weights (the product), AMEVA0 with F32 weights. Same phonemes, voice 0, noise seed 1234, the reference's frame counts.
Gate: AMEVA0's log-spectral distance and F0 relative L2 from the reference within 1.5 x the floor; AMEVA0 ran every node
with GPU work counted by the plugin. Each run's own rounded durations are compared too.

Implementation against the onnxruntime float reference (t1, 117 tokens, 7.55 s, S20, final engine): CPU F32 SNR
36.2 dB, log-mel 0.125 dB, F0 relative 1.0e-6 (max 0.002 Hz); AMEVA0 F32 33.9 dB / 0.155 dB; AMEVA0 F16 21.8 dB /
0.456 dB, log-mel correlation 0.9997. The waveform SNR is phase-limited: the harmonic source integrates F0 over the
utterance, so tiny F0 differences shift later samples; spectra and F0 are the gated measures.

| device (ssh cores, GPU) | engine | AMEVA0 F16 ms, RTF (t1 7.55 s) | t3 26.05 s RTF | log-spectral dB (floor) | F0 rel (floor) | CPU F32 ms |
|---|---|---|---|---|---|---|
| S25 (0-1,4-5, Adreno 830, kgsl max 832 MHz) | final | 1313, 0.174 | 0.176 (t2: 0.177) | 1.49 (3.16) | 1.1e-4 (2.0e-3) | 9290 |
| S21 (0-7, Mali-G78 at 858 MHz) | final | 3194, 0.423 | open (sshd stopped after t1) | 1.48 (3.16) | 1.1e-4 (2.0e-3) | 5978 |
| S20 (0-7, Adreno 650 at 441 MHz) | final | 6017, 0.797 | 0.788 (t2 3.67 s: 0.803) | 1.49 (3.16) | 1.1e-4 (2.0e-3) | 16850 |
| A53 (0-2, Mali-G68) | v3 (before conv linears / FA) | 12596, 1.668 | - | 1.49 (2.99) | invalid (readback bug, fixed) | 80396 |

t3 (446 phonemes, 26.05 s, S20, final engine): AMEVA0 F16 20535 ms, log-spectral 3.14 dB (floor 3.52), F0 relative
1.4e-4 (floor 1.8e-3), durations equal for every token (the CPU F16 run flipped one); AMEVA0 F32 SNR 28.7 dB, 1.00 dB,
F0 1.0e-6. Over 26 s the f16 weights move the waveform phase (CPU and GPU alike); against the onnxruntime reference
the AMEVA0 F16 WAV has log-mel distance 1.14 dB (t2: 0.31 dB, t1: 0.49 dB).

t3 on the S25 (final engine): AMEVA0 F16 4588 ms (RTF 0.176), log-spectral 3.14 dB (floor 3.52), F0 1.4e-4, durations
equal (the CPU F16 run flipped one token again). The S25's early engine (no fusions, linears as MUL_MAT) was 1651 ms
(RTF 0.219) on t1.

The t1 WAVs of the S25, S21 and S20 against the onnxruntime reference WAV (16-bit): log-mel distance 0.492 / 0.482 /
0.487 dB, correlation 0.9997; against the reference's float output 0.63 dB on each (the 16-bit rounding of the device
WAV alone accounts for about 0.2 dB). S25 against S21: SNR 34.8 dB, log-mel 0.16 dB.

All runs: every node on AMEVA0, durations equal to the reference's for every token, RESULT PASS, WAV headers 16-bit
PCM mono 24 kHz with the expected length. Open: the A35 (Tailscale offline) and the A53 (sshd refuses connections)
final-engine runs, and the S21's t2 / t3 (its sshd stopped after the t1 run).

CPU product baseline (sherpa-onnx int8, 4 threads, text to WAV, S20): 10.49 s for 6.50 s of audio, RTF 1.61.

S20 profile (final engine, t1): CONV_2D 82 % of GPU time (the generator's convolutions, about 380 GFLOP per t1), the
rest element-wise ops, GROUP_NORM and the LSTM steps (about 10 us per small dispatch, about 140 ms per synthesis).

## 5. Next, by measured value

1. Final-engine runs on the A35 and A53 (Mali-G68), S21 t2 / t3.
2. Mali: the convolution kernel reaches about 63 GFLOP/s on the Mali-G68 (`ameva_conv.comp` notes), so the A53 / A35
   stay above real time on convolutions alone; candidates are f16 arithmetic in ameva_conv and q8_0 conv weights
   (10 % on G68 per the kernel notes).
3. On-device G2P: espeak-ng is inside sherpa-onnx's Python module on the S25 / S20 / A35 only; the S21 and A53 would
   need the Termux espeak package (a download).
