# Current Engineering Checkpoint

**Timestamp**: 2026-10-11T19:30:00+09:00
**Branch**: `perf/opencl-mali` (fix committed and pushed as `084d513`)
**Goal**: seal the Mali t3 fix (official backend build, one-model t3 < 1.0 GiB), research note AOSF-TR-2026-MALI-DRIVER-04.

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
- `ameva_tts_parity --gpu-only` no longer loads the CPU model copy (`f56b917`). One-model t3 peak RSS with the
  official 084d513 artifacts: A35 883, A53 862, S21 920 MiB (all < 1.0 GiB).
- Official backend build of 084d513 on the S21 (bundle v2.8.1): libggml-ameva.so 6b953c49..., libac_runtime.so.2
  84d973f2...; QMM check 516/516, batch KATs incl. MEMORY, AMEVA0 registration. Not published.
- Without `--ref-model` the parity tool's floor is 0 (CPU f16 vs itself), so its gate cannot pass an f16 GPU run.
- tools/build_pages.py injects duplicate shared assets into uno-km docs/benchmarks.html (already loaded at lines 8-10);
  that change was not committed.

### Next / Remaining
1. Research note AOSF-TR-2026-MALI-DRIVER-04 published in uno-km (`cbbdf03`, docs/research_papers/).
2. Release of the 084d513 backend is the user's call. BitNet PR #633: frozen by user decision 2026-10-11.
3. Lab cleanup: only this session's scratch removed; rel27x/rel280/rel-2.8.1/iso staging copies (~3.3 GB per phone) await the user.

### Commands
- Batch KAT: `scripts/termux/build_runtime_backend.sh` (or lab: build runtime + `tests/test_batch_graph_kat.cpp`)
- TTS Parity: `bash tts_parity_run.sh TAG OUT MODELS --tokens-file t3.tokens --reps 1 --threads 4`
- Long runs: Termux wake lock first on S25/A35/A53 (`am startservice ... com.termux.service_wake_lock`)
