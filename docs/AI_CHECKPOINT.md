# Current Engineering Checkpoint

**Timestamp**: 2026-10-11T16:30:00+09:00
**Branch**: `perf/opencl-mali` (fix committed and pushed as `084d513`)
**Goal**: Stage 5 Kokoro TTS t3 OOM on Mali (resolved, see below); BitNet upstream PR #633 pending user decision.

### Completed
- `ac_bitnet_sample` on all 5 phones (temp 1.0/1.5, top-1 prob ~94.4 %, distinct sampled texts).
- Kokoro-82M Stage 5 t1/t2 PASS on 5 phones (S25 RTF 0.174, S21 0.423); S25 / S20 t3 PASS (RTF 0.178 / 0.788).
- **Mali t3 OOM resolved (2026-10-11).** Cause, by ablation on the A35 (30000 ameva_qmm dispatches): the Mali
  driver allocates ~100 KB per `vkCmdDispatch` in the command buffer, independent of barriers, flush count and
  descriptor-set reuse; it is held until the command buffer is freed/reset, and `vkFreeCommandBuffers` returns most of
  it to the pool only. Descriptor sets cost ~0.7 KB each (not the cause; the previous checkpoint's suspicion was wrong).
- Fix (`ac_vk_batch_flush`): a fence per flushed command buffer, completed ones re-recorded by later flushes,
  at most 4 pending flushes (then the recording thread waits for the oldest; 60 s -> AC_ERR_LAUNCH_TIMEOUT, batch aborted).
- Verified: probe peak RSS 3159 -> 121 MiB. `test_batch_graph_kat` gains MEMORY (30000 dispatches, flush 128,
  RSS growth < 48 MiB): PASS on all 5 phones (-1.5..1.3 MiB); new test against the OLD runtime on the A35: FAIL (+2932.5 MiB).
- t3 `--gpu-only` peak RSS: A35 1048, A53 1016, S21 1079 MiB (old runtime: killed). t3 4-model parity (f16 AMEVA vs
  f32 CPU, gate 1.5 x CPU f16 floor): PASS on A35 (LSD 3.134 dB, peak 1765 MiB), A53 (3.135 dB, 1709 MiB), S21 (3.13 dB, 1799 MiB).
- S25 t2 timing old vs new runtime, alternated: 651.9-656.4 ms both (no measurable change).

### Open
- "< 1.0 GB peak" is not met as stated: gpu-only t3 peaks 1.02-1.08 GiB with two resident models (an unused CPU copy
  included by the parity tool); the AMEVA-side remainder was not decomposed.
- Without `--ref-model` the parity tool's floor is 0 (CPU f16 vs itself), so its gate cannot pass an f16 GPU run.

### Next / Remaining
1. Rebuild the bundle backend with `084d513` (`build_runtime_backend.sh` runs the KAT); release is the user's call.
2. Upstream microsoft/BitNet PR #633: user decided 2026-10-11 "not now".

### Commands
- Batch KAT: `scripts/termux/build_runtime_backend.sh` (or lab: build runtime + `tests/test_batch_graph_kat.cpp`)
- TTS Parity: `bash tts_parity_run.sh TAG OUT MODELS --tokens-file t3.tokens --reps 1 --threads 4`
- Long runs: Termux wake lock first on S25/A35/A53 (`am startservice ... com.termux.service_wake_lock`)
