# Current Engineering Checkpoint

**Timestamp**: 2026-10-08T23:40:00+09:00  
**Branch**: `perf/opencl-mali` (pushed: `e58d508` lazy weight repack + GPU GET_ROWS for q4_0/q5_0, `aed3e39`).  
**Current Goal**: Stage 2, STT (Whisper on ggml-ameva). Stage 1 (Diffusion) closed by user decision 2026-10-08.  
**User order 2026-10-08**: no model downloads on the S25, no new model downloads on the A35 / S21 (data cost).

### Verified (Ground Truth). Tables: `docs/design/stt-whisper-ameva.md`
- **Route**: the shipped STT bundle `whisper-cli` (2.7.9, ggml 0.25.1) loads `libggml-ameva.so` through `GGML_BACKEND_PATH`, built against the ggml headers of local `dev/whisper.cpp` `d87636ba` (`scripts/termux/lab/stt_build.sh`). AMEVA0 is `-dev 1`, or `-dev 0` with `GGML_VK_VISIBLE_DEVICES=""`. No whisper.cpp change.
- **Whole Whisper large-v3-turbo f16 graph on AMEVA (`AMEVA_STRICT=1`) on the S25, S21, A35; transcripts identical to a CPU reference.** New kernels: `im2col` (`ameva_im2col.comp`), `mm_f16` / `mm_f16w` / `mm_f16wu` (`ameva_mm.comp -DF16`). SPIR-V of existing kernels unchanged.
- Op checks on the S21 (`conv_check.sh kernels`): `ameva_stt_ops_check` 24/24 with both f16 decoders, conv 31/31, cnorm 15/15 x4, window 18/18, unet ops 24/24, qmm 516/516.
- S25 (11 s audio, one window, cores 0-1,4-5), encode / total ms: bundle CPU 75741 / 79621; bundle Vulkan 26943 / 33167; AMEVA strict 23226 / 28979. q5_0 (made on the phone with `whisper-quantize`): bundle Vulkan 33987 / 37825; AMEVA 21233 / 24241; AMEVA `AMEVA_REPACK_ALL=1` 9954 / 14241.
- S21 (60 s audio): AMEVA strict encode 168105 and 327281 ms in two runs; another CPU build (`~/.local/bin/whisper-cli`) 39481 ms. A35 (11 s): AMEVA strict encode 197242 ms, lowest MemAvailable 621 MB.
- **The bundle's `libggml-cpu.so` ends with SIGILL on the S21 and A35** (built with i8mm). Any node placed on the CPU ends the process there.
- **whisper.cpp marks its buffer as weights after the upload, so ggml-ameva repacks nothing** and every MUL_MAT runs `ameva_qmm` (f16 and q5_0 both 21 s on the S25). `AMEVA_REPACK_ALL=1` halves the encode and sends the token-embedding GET_ROWS to the CPU.
- Diarization (A35, `stt_diar_check.sh`): sherpa-onnx alone and beside the strict AMEVA run give identical intervals and identical whisper JSON; `SpeakerMapper.align` labels the segment. 9.1 s alone, 35.5 s beside whisper.

### Known, Not Done
- OpenCL with Whisper not run. GET_ROWS from quantized / repacked weights not on the GPU (quantized models end in the SIGILL above on the S21 / A35). `-nfa` declines the attention MUL_MATs.
- Diarization not checked on the S25 (no models) or S21 (no `sherpa_onnx`); `termux-stt diarize` not run end to end; single-speaker audio only.
- `test-backend-ops`, LLM text parity and the STT ops check on the S25 / A35 not run with this source. S21 2x run-to-run swing unexplained (GPU clock not readable).
- Stage 1 leftovers unchanged (Z-Image-Turbo 512 px CPU reference, LoRA merge on quantized weights, `AMEVA_MMQ` default for diffusion).

### Relevant Files (uncommitted)
- `adapters/llama/ggml-ameva/ggml-ameva.cpp`, `ggml-ameva-device.cpp`, `shaders/ameva_im2col.comp` (new), `shaders/ameva_mm.comp`, `ggml_ameva_kernels_spv.h` / `_cl.h` (regenerated on the S21), `scripts/build_ameva_kernels.py`
- `tests/ameva_stt_ops_check.cpp` (new), `scripts/termux/lab/stt_build.sh`, `stt_bench.sh`, `stt_diar_check.sh`, `ggml_list_devices.c` (new), `conv_check.sh`, `docs/design/stt-whisper-ameva.md` (new)
- Devices: source `~/abi-lab/src-stt4` (S25), `src-stt3` (S21), `src-stt2` + scripts (A35); plugin `~/abi-lab/out-stt3` (S25 built there; S21 built there; A35 copy of the S21 build); ggml headers `~/abi-lab/wsp-ggml-d87636ba`; S25 `~/abi-lab/ggml-large-v3-turbo-q5_0.bin` (547 MB, local).

### Next Actions
1. Commit (awaiting the user).
2. Repack weights of a buffer marked after its first write, and GET_ROWS from repacked weights on the GPU (S25 encode 21.2 -> 10.0 s measured with the switch).
3. f16 weights on the tile kernel; the stem on `ameva_conv`; per-op GPU time on Mali with the GPU clock recorded.
4. ameva-runtime: rebuild the STT bundle CPU backend without i8mm; wire the two environment variables into the STT adapter.
