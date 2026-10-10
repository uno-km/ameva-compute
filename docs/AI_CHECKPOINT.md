# Current Engineering Checkpoint

**Timestamp**: 2026-10-10T11:30:00+09:00
**Branch**: `perf/opencl-mali` (pushed to origin/perf/opencl-mali up to 2aa9315, clean sync)
**Goal**: Stage 5, TTS (Kokoro-82M v0.19) on ggml-ameva (Stages 1-4 closed). Design note `docs/design/tts-kokoro-ameva.md`.

### Completed
- `5bb49d7` adapters/tts/kokoro: two ggml graphs on one backend, every node checked with supports_op (no CPU path).
  Weights from each phone's own sherpa-onnx `kokoro-int8-en-v0_19` (convert_onnx_to_gguf.py, ~2 s, no model moved).
  Large-M linears as 1x1 CONV_2D, ALBERT attention as FLASH_ATTN_EXT, AdaIN / Snake in the AMEVA fusion forms.
- `2f1a7e2` ggml-ameva: SIN COS SQR SQRT LEAKY_RELU SGN STEP FLOOR, CUMSUM, bilinear UPSCALE, AMEVA_FUSE_SNAKE.
- `f74b427` ac_runtime: Vulkan batch buffer list grew one dispatch at a time (O(n^2) encode, 120 -> 25 us/dispatch).

### Verified (t1 = 117 tokens, 7.55 s; WAV only, never played)
- Implementation vs onnxruntime float reference (PC): CPU F32 SNR 36.2 dB, log-mel 0.125 dB, F0 rel 1.0e-6.
- S20 final engine: AMEVA0 F16 6017 ms RTF 0.797 (t2 0.803, t3 26 s 0.788), log-spectral 1.49 dB vs floor 3.16,
  F0 1.1e-4 vs floor 2.0e-3, AMEVA0 F32 F0 1.0e-6; CPU F32 16.9 s; sherpa int8 CPU product RTF 1.61. t1, t3 PASS.
- S25 final engine (cores 0-1,4-5): t1 1313 ms RTF 0.174, t2 0.177, t3 0.176; LSD / F0 within gate; PASS x3.
- S21 final engine (Mali-G78 858 MHz): t1 3194 ms RTF 0.423 (CPU F16 0.631), PASS.
- A53 v3 RTF 1.668 (CPU F32 80 s), PASS on audio. t1 WAVs S25 / S21 / S20 vs ORT WAV: log-mel 0.49 dB each.
- test-backend-ops new ops: S25 41/41 (final plugin), A53 106/106, S20 104/104.
- `4f7ad6d` kernel headers regenerated with glslang 16.3.0; S21 and S25 output byte-identical.

### Remaining
1. A35 / A53 final-engine runs (A35 Tailscale offline, A53 sshd refused); S21 t2 / t3 (sshd stopped after t1).
2. Mali-G68 RTF < 1: ameva_conv ~63 GFLOP/s keeps A53 / A35 above real time; candidates f16 math, q8_0 weights.
3. Branch `perf/opencl-mali` pushed to origin (8 commits, clean sync).
4. On-device G2P (espeak-ng only inside sherpa's Python module on S25 / S20 / A35).

### Commands
- Build (S25 / A53; S20 builds only the tool): `GGML_DIR=<b8471 ggml> bash tts_build.sh SRC OUT all|kernels|plugin|tool`.
- Convert: `python3 convert_onnx_to_gguf.py ~/.cache/termux-tts/models/kokoro-int8-en-v0_19 kokoro-f16.gguf [--wtype f32]`.
- Parity: `bash tts_parity_run.sh TAG OUT MODELS --tokens-file t1.tokens --reps 3 --threads 4`; timing `--gpu-only`.
- Fleet copies: ~/abi-lab/out-tts, tts-models (t1/t2/t3.tokens, GGUFs), src-tts. Reference: PC scratch ref/t1..t3.

### Known failures
- None open in the verified scope. Tool before 5bb49d7 read F0 from a reused buffer (A53 F0 numbers invalid).
