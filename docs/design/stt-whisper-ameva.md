# STT Stage 2: whisper.cpp on ggml-ameva (Whisper large-v3-turbo, speaker diarization beside it)

- Written 2026-10-08, branch `perf/opencl-mali`. Follows `AMUDA_ENGINE_UNIFICATION_PLAN.md` (Phase 3, whisper.cpp).
- Sources read: local `dev/termux/termux-stt` (2.0.5), local `dev/whisper.cpp` (`d87636ba`, ggml 0.25.1, carries a local
  Adreno 6xx flash-attention patch), the installed ameva-runtime 2.7.9 STT bundle on the phones. No remote fetch.
- Status: the whole Whisper graph runs on AMEVA (Vulkan) on the Galaxy S25, S21 and A35 with the shipped bundle
  binary and transcripts equal to a CPU reference. Speed is ahead of the bundle's own paths on the S25 only. OpenCL
  was not run. Sections 5 and 6 list what is measured, what is not, and the next steps in order of measured value.
- Data moved to the phones: about 16 MB of sources, headers and plugin copies. No model was downloaded: the S25 q5_0
  file was made on the phone from its f16 file with the bundle's `whisper-quantize`.

## 1. How termux-stt computes today

| Stage | Where | Compute |
| :--- | :--- | :--- |
| Text | `termux_stt/engine/whisper_engine.py` starts `whisper-cli` (ameva-runtime STT bundle, whisper 1.9.4, ggml 0.25.1) as a process | ggml: CPU backend, or the bundle's own `libggml-vulkan` selected with `-dev 0` |
| Speaker intervals | `termux_stt/diarization/sherpa_diarizer.py`: `sherpa_onnx.OfflineSpeakerDiarization` in process, or the `sherpa-onnx-offline-speaker-diarization` binary | onnxruntime on the CPU (pyannote segmentation 3.0, CAM++ embedding) |
| Join | `termux_stt/diarization/mapper.py` `SpeakerMapper.align` | Python, by segment times |

The two neural stages share nothing but the audio file, so GPU work on the text stage cannot change the diarization
result; what has to hold is that the GPU run delivers the same segments and times (section 4).

Whisper large-v3-turbo per 30 s window: encoder 32 layers of 1500 positions x 1280, about 9.4e11 multiply-accumulates in
MUL_MAT plus about 1.9e11 in attention; the two stem convolutions 8.9e9; the decoder (4 layers) about 1.6e8 per token
and beam. The encoder is the load; the decoder is many small graphs.

## 2. Integration: a plugin of the shipped bundle, no change to whisper.cpp

- `whisper-cli` calls `ggml_backend_load_all()`, which loads the library named by `GGML_BACKEND_PATH`. The plugin is
  `libggml-ameva.so` built against the ggml headers of `d87636ba` (`scripts/termux/lab/stt_build.sh`); the header set
  that was on the S25 (`~/whisper.cpp/ggml`) is older and the plugin's ABI check refused it at load, as designed.
- whisper.cpp picks the N-th GPU device (`-dev N`). With the bundle's Vulkan0 present AMEVA0 is `-dev 1`. With
  `GGML_VK_VISIBLE_DEVICES=""` the bundle registers no Vulkan device and AMEVA0 is `-dev 0`: two environment variables
  route an unmodified termux-stt / ameva-runtime adapter to AMEVA. Not wired into either package yet.
- whisper.cpp returns to the CPU without an error when the requested device does not exist ("no GPU found"). Every
  measurement below checks the backend line of the log (`scripts/termux/lab/stt_bench.sh`).

Graph census (`AMEVA_STRICT=1`, turbo f16): everything was accepted except the encoder stem,
`ggml_conv_1d` = IM2COL (f32 -> f16) + MUL_MAT (f16 columns x f16 kernel). Added in this stage:

| Kernel | Source | Use |
| :--- | :--- | :--- |
| `im2col` | `shaders/ameva_im2col.comp` | IM2COL 1-D and 2-D, f16 or f32 result, CPU element order, f16 rounded to nearest even |
| `mm_f16`, `mm_f16w`, `mm_f16wu` | `shaders/ameva_mm.comp -DF16 [-DF16_WORDS [-DF16_UNPACK]]` | MUL_MAT f16 x f16 -> f32: one value per load with the exact decoder; two per load; two per load through the built-in conversion on a device whose conversion passed the existing check (`ameva_conv_f16_prepare`) |

The SPIR-V of every existing kernel is unchanged (generated header: additions only); the OpenCL text of the f32 `mm`
kernel changed in form (macros), not in meaning. With these the strict run passes on all three phones.

Check `tests/ameva_stt_ops_check.cpp` (in `conv_check.sh`), Galaxy S21: 24/24 with the built-in conversion and 24/24
with `AMEVA_CONV_F16=exact` (IM2COL equal to the reference in every bit, f16 and f32, 1-D and 2-D; `ggml_conv_1d` /
`ggml_conv_2d` within the rounding bound, largest 0.020 of it). Regression there: conv 31/31, cnorm 15/15 x4, window
18/18, unet ops 24/24, qmm 516/516. Not run on the S25 or A35, and `test-backend-ops` not run with this source.

## 3. Measurements

whisper.cpp's own stage times (monotonic clock). 5 beams, `-t 4`, flash attention on, English, ssh sessions. The phones
suspend an ssh job nobody polls, so wall time is not reported. Transcripts: identical to the CPU reference in every
row that has one.

Galaxy S25 (Adreno 830, cores 0-1,4-5, GPU clock 832 MHz under load), `jfk.wav` 11 s = one encoder window:

| Model | Path | encode ms | decode ms (132 batches) | total ms |
| :--- | :--- | ---: | ---: | ---: |
| turbo f16 | bundle CPU | 75741 | 1881 | 79621 |
| turbo f16 | bundle ggml-vulkan | 26943 / 27207 | 2608 / 3339 | 33167 / 34169 |
| turbo f16 | AMEVA, stem on the CPU (before this stage's kernels) | 21195 / 20686 | 648 / 618 | 27484 / 24367 |
| turbo f16 | AMEVA strict, whole graph | 23226 / 23084 | 628 / 655 | 28979 / 27956 |
| turbo q5_0 | bundle CPU | 62224 | 2313 | 65831 |
| turbo q5_0 | bundle ggml-vulkan | 33987 / 34338 | 2474 / 2410 | 37825 / 38394 |
| turbo q5_0 | AMEVA (token-embedding GET_ROWS on the CPU) | 21233 / 21220 | 807 / 777 | 24241 / 23567 |
| turbo q5_0 | AMEVA, `AMEVA_REPACK_ALL=1` | 9954 / 10108 | 2391 / 1858 | 14241 / 13731 |

Galaxy S21 (Mali-G78, cores 0-7, GPU clock not readable), `jfk_1min.wav` 60 s, turbo f16:

| Path | encode ms | decode ms | total ms |
| :--- | ---: | ---: | ---: |
| bundle CPU | SIGILL | | |
| CPU, another build (`~/.local/bin/whisper-cli`: FP16_VA, OpenMP) | 39481 | 4699 | 47934 |
| AMEVA strict, first run | 168105 | 8105 | 181712 |
| AMEVA strict, second run 30 min later (final kernels) | 327281 | 17714 | 351128 |

Galaxy A35 (Mali-G68, cores 0-2), first 11 s of that audio, turbo f16, AMEVA strict: encode 210340 (exact f16 stem) and
197242 ms (final kernels), decode 3552 ms, total 213816 ms, lowest MemAvailable 621 MB. No CPU reference can be made
on the A35 (the bundle CPU ends with SIGILL, no other build there): the reference is the S21's other CPU build on the
same cut and model file (audio and model sha256 equal on both phones), and the two transcripts are identical. That
S21 CPU run of one window: encode 15787 ms, total 18906 ms.

Reading:

- **The bundle's CPU backend does not run on the S21 or the A35.** `libggml-cpu.so` stops with SIGILL at the first
  graph (same address on both; the build reports `MATMUL_INT8 = 1`, these cores have no i8mm). Any node the scheduler
  places on the CPU ends the process there. The S25 CPU rows above are that same build (no FP16_VA, no OpenMP) in a
  4-core session: they are the bundle's CPU path, not what an S25 CPU can do.
- S25: AMEVA is ahead of both bundle paths (f16 encode 1.2x the bundle Vulkan, decode 4x; q5_0 with repacked weights
  total 2.7x the bundle Vulkan and 4.7x the bundle CPU).
- S21 and A35: correct and far too slow. On the S21 a well-built CPU binary is 4 to 8 times faster than AMEVA with
  f16 weights, and two AMEVA runs differ by 2x in encode and decode alike (device state, not the kernel change: the
  decoder path is the same in both).
- **Weight type made no difference until the weights were repacked.** f16 and q5_0 both encode in 21 s on the S25
  because whisper.cpp marks its model buffer as weights after uploading it (`whisper.cpp:1950` then `:1968`), and
  ggml-ameva repacks a quantized weight only when the buffer is marked at the first write. Every MUL_MAT then runs
  the generic `ameva_qmm` kernel: `repacked_tensors=0`, `mmq_mul_mat=0`. With `AMEVA_REPACK_ALL=1` 233 tensors
  (525 MiB) are repacked, 200 MUL_MATs take the integer-dot kernel and the encode time halves; the decode gets slower
  because the token embedding, which whisper also uses as the output projection, is now repacked and its GET_ROWS
  goes to the CPU through the converting buffer interface.
- The stem on the GPU costs time: about 2.2 s more than the CPU stem on the S25, 4.7 s for the conv graph on the S21.
  `ameva_mm` is one invocation per result element with scalar loads. The first version (exact f16 decode per value)
  did 1.5e9 products per second on the Adreno 830 and a 1e9-product dispatch of it did not come back in 10 minutes;
  the word-load builds and a 32x work estimate for the exact decoder are the fix, verified by the strict S25 rows.
- Without flash attention (`-nfa`) the attention MUL_MATs are declined (non-contiguous, K not a multiple of 32,
  batched) and the encode takes 45.9 s: flash attention stays on.

## 4. Speaker diarization beside the GPU run (Galaxy A35)

`scripts/termux/lab/stt_diar_check.sh`, models already on the phone, first 11 s of `jfk_1min.wav`, 2 clusters:

| Run | Diarization | Whisper on AMEVA0 (strict) |
| :--- | :--- | :--- |
| each alone | rc 0, 5 intervals, 9.1 s | rc 0, 1 segment, encode 210 s |
| both at once | rc 0, 5 intervals, 35.5 s | rc 0, 1 segment, encode 209 s |

Speaker intervals identical between the two runs, whisper JSON identical, `SpeakerMapper.align` of the installed
termux-stt 2.0.5 labels the segment. The diarization takes 4 times longer while whisper runs (three A55 cores shared
with whisper's CPU side). Not checked: the S25 (no diarization models there, download not allowed) and the S21 (no
`sherpa_onnx` module, no models); `termux-stt diarize` end to end through its own CLI (it may prompt or download);
audio with two real speakers.

Also found in termux-stt: `scripts/install_sherpa_onnx.sh` creates an empty file as the binary and
`scripts/download_models.sh` writes placeholder text files. The Python installer is what the phones actually used.

## 5. Not done

- OpenCL (`AMEVA_BACKEND=opencl`) with Whisper: not run. The new kernels are translated; nothing executed them.
- GET_ROWS from a quantized or repacked weight on the GPU: with a quantized model one node per decode step runs on the
  CPU, which on the S21 and A35 is the SIGILL above (A35 q5_0 run: encoder done, then SIGILL).
- Repacking for hosts that mark the buffer after the upload: only through `AMEVA_REPACK_ALL=1` so far.
- Korean audio, long audio, timestamps per token, VAD, translation; turbo q5_0 on the S21; any model on the A53 / S20.
- `test-backend-ops` and the LLM text parity with this source; the STT ops check on the S25 and A35.
- The A35 run leaves 621 MB available with the f16 model: not a configuration to ship.

## 6. Next steps, by measured value

1. Repack weights of a buffer that is marked as weights after its first write (at the first graph that multiplies by
   them), and GET_ROWS from repacked weights on the GPU. Measured upside on the S25: encode 21.2 -> 10.0 s; removes the
   CPU node that ends quantized runs on the S21 and A35.
2. f16 weights on the tile kernel (`ameva_qgemm` has no f16 build): the f16 model is what every phone has.
3. The stem as one convolution on the existing `ameva_conv` kernel instead of IM2COL + dense MUL_MAT.
4. Per-op GPU time of the encoder on Mali (attention against MUL_MAT) before any Mali kernel work, with the GPU
   clock recorded; the S21 2x swing has to be explained first.
5. Rebuild the STT bundle's CPU backend for ARMv8.2 without i8mm (ameva-runtime), independent of the GPU work.
6. Wire `GGML_BACKEND_PATH` + `GGML_VK_VISIBLE_DEVICES=""` into the ameva-runtime STT adapter once 1 is in.
