# Current Engineering Checkpoint

**Timestamp**: 2026-10-11T13:00:00+09:00
**Branch**: `perf/opencl-mali` (origin/perf/opencl-mali in clean sync)
**Goal**: Stage 5 Kokoro TTS t3 OOM resolution on Mali & BitNet live sampling verification.

### Completed
- `feat(bitnet)`: `ac_bitnet_sample` added and verified across all 5 phones (S25, S20, S21, A35, A53).
  Confirmed 100% genuine on-device GPU inference (temp 1.0/1.5, prob ~94.4%, distinct sampling texts).
- `docs(tts)`: Stages 1-4 closed. Kokoro-82M Stage 5 t1/t2 PASS across all 5 devices (S25 RTF 0.174, S21 0.423).
  S25 / S20 t3 (26s) PASS (RTF 0.178 / 0.788, peak 1.41 GB). Audio fidelity SNR > 34 dB, log-mel < 0.49 dB.

### Identified Root Cause (Mali t3 OOM Kill)
- Mali GPU driver accumulates ~38 KB per GPU step until graph completion.
- t3 length (29,973 steps) causes ~1.2 GB driver-held memory + 390 MB buffer -> OOM kill on 6GB RAM (A35/A53/S21).
- Unreleased descriptor pools (1 set per step, 256 sets/pool) compound driver footprint.

### Next / Remaining
1. Update `ac_runtime` to release completed GPU dispatches mid-graph and cap descriptor pool allocation.
2. Re-test t3 Kokoro TTS on A35, A53, S21 with < 1.0 GB peak RAM target.
3. Upstream microsoft/BitNet PR #633: address Matheus's build findings via submodule quants.c & #588 ReLU².

### Commands
- BitNet Sample: `ac_bitnet_sample -m <model> -p "The capital of France is" --temp 1.0 -k 3`
- TTS Parity: `bash tts_parity_run.sh TAG OUT MODELS --tokens-file t1.tokens --reps 3 --threads 4`
- Build TTS: `GGML_DIR=<b8471 ggml> bash tts_build.sh SRC OUT all|plugin|tool`
