## AMUDA Android SDK v0.3.0 — Speech, Image Generation, Auto Backend-Select (AAR)

Adds three finished-product Kotlin APIs and a model-download helper to the v0.2.0 AAR
(`com.ameva.amuda.llm.AmevaLlm`, unchanged). Same deployment model as v0.2.0: no Termux, no shell
daemons, no localhost server -- everything runs in-process inside a standard Android application.

### New in v0.3.0

- **`com.ameva.amuda.stt.AmevaSpeech`** -- load a whisper.cpp GGML model, transcribe 16 kHz mono
  PCM. Backed by `stable-diffusion.cpp`'s sibling, `whisper.cpp` (pinned commit `610e664b`), statically
  linked with the formal AMEVA Vulkan STT backend. Only two encoder matmul shapes (f16, q5_0
  weights) get GPU dispatch today; every other op runs wherever ggml's scheduler places it
  (normally CPU) -- this is not a full whisper-encoder GPU port.
- **`com.ameva.amuda.diffusion.AmevaDiffusion`** -- load a single-file diffusion checkpoint,
  generate an image with `txt2img()`. Backed by `stable-diffusion.cpp` (pinned commit `50d6405`,
  matching the fleet's already-tested Termux build) + a second, independently-built copy of the
  `ggml-ameva` backend (`libggml-ameva-sd.so`, separate from AmevaLlm's `libggml-ameva.so` --
  stable-diffusion.cpp pins its own ggml fork with a real ABI difference, `GGML_MAX_NAME=160` vs.
  llama.cpp's default 64; the two backends must never be swapped).
- **`AmevaLlm.load(..., backend = "auto")`** (new default) -- benchmarks cpu/vulkan/opencl with a
  short, fixed-size GEMV before loading the real model, and uses whichever is actually fastest on
  this device. Explicit `"cpu"`/`"vulkan"`/`"opencl"` skip the benchmark. `AmevaLlm.reset()` also
  ships in this release: clears the KV cache between conversations without a full reload.
- **`com.ameva.amuda.modelhub.AmevaModelHub.download(url, dest)`** -- one-line background download
  of a model file into app storage (no bundled weights; this SDK ships the inference engine, not
  model files).
- **`com.ameva.amuda.stt.sherpa.AmevaSenseVoice`** -- offline SenseVoice transcription (text,
  language, emotion and sound-event tags) via k2-fsa/sherpa-onnx v1.13.8. This is a different
  engine from `AmevaSpeech`: ONNX Runtime on CPU, **not** the AMEVA Vulkan backend, so nothing on
  this path is GPU-accelerated. sherpa-onnx's classes and its arm64-v8a libraries are unpacked from
  the upstream .aar (downloaded and sha256-verified at build time) and bundled into this AAR (AGP cannot nest a local .aar); do not also
  add the upstream sherpa-onnx .aar to your app.

### Verification scope -- read this before relying on a claim below

- **AmevaSpeech**: verified on Samsung Galaxy A35 and A53 (Mali-G68) -- correct GPU-accelerated
  transcript of the standard whisper.cpp JFK sample on both. S21 was unreachable this cycle
  (sshd down); not tested.
- **AmevaDiffusion**: one model only (SDXS, single-step distilled, 683 MB GGUF), 1 sampling step,
  seed 42, prompt "a cat sitting on a sofa". Verified on A35 and S21 at 256x256 and 512x512. Those
  runs executed the AAR's native layer under ART via `app_process` by calling the JNI methods
  directly; the Kotlin `AmevaDiffusion.load()/txt2img()` wrapper (needs a Context and `Bitmap`) was
  not run. (An earlier A35 run at the default guidance, 512x512, took ~7.5 minutes.)
  **Correction to the earlier release text**: the over-saturated "burnt" colors seen in the first A35
  image were not an SDXS characteristic. `txt2img()` left guidance at stable-diffusion.cpp's default of
  7.0, while SDXS is guidance-distilled and needs about 1.0. Measured on the S21 at 256x256: channel
  values at exactly 0 were 43.6% on CPU and 43.1% on the AMEVA GPU path at guidance 7.0, and 0.1% (CPU)
  and 0.0% (GPU) at guidance 1.0. `txt2img()` now takes `cfgScale` (a value <= 0 keeps the 7.0 default;
  the C API gained `ameva_sd_txt2img_ex`, `ameva_sd_txt2img` is unchanged). At guidance 1.0 the GPU
  image matches the CPU reference at 22.5 dB PSNR (256x256) and shows a clean tabby on a sofa at both
  sizes; two consecutive GPU runs were identical.
  **Speed**: on the S21 the 256x256 image took 17-19 s through the AAR's GPU path versus 5.6 s for the
  CPU `sd-cli` -- the GPU path is about 3x slower for this model on this phone. 512x512 took 152 s on
  the GPU path (no CPU timing taken). On the A35 at guidance 1.0: 256x256 took 22-25 s and 512x512
  116 s (no CPU timing taken there). The A35 image matches the CPU reference at 28.7 dB PSNR; the
  A35 and S21 GPU images match each other at only 21.9 dB (different drivers and kernel paths; the
  outputs are not bit-equal across devices). Two VAE decode tensors exceed a single Vulkan buffer
  binding and are explicitly declined to the CPU (logged, not hidden).
  Not verified: A53 (same Mali-G68 family as the A35, but no copy of the model there and none was
  moved or downloaded), other models, multi-step sampling, img2img, LoRA.
- **Auto backend-select**: verified on A35 -- real per-backend timings for a fixed GEMV (cpu
  6.77 ms, vulkan 3.39 ms, opencl 2.92 ms/op), correctly selected OpenCL as this device's fastest
  path. Not yet verified on other GPU families.
- **AmevaModelHub**: compiles clean, pure Kotlin, no native code -- not exercised against a real
  download in this cycle.
- **AmevaSenseVoice**: the `AmevaSenseVoice` Kotlin class and the sherpa-onnx v1.13.8 libraries as
  packaged in this AAR were run on Samsung Galaxy A35, A53 and S21 (CPU/ONNX Runtime, SenseVoice int8
  `2024-07-17`, the 5 upstream test wavs: en, zh, ja, ko, yue). Method: the AAR's classes dexed with d8
  and run under ART via `app_process` from Termux (no APK installed, no adb). All three produced the
  expected language/emotion/event tags and transcripts (10 of 10 per phone) and passed three guard
  checks. Warm real-time factor, 4 threads: A35 0.34-0.42 and A53 0.31-0.40 (ssh cpuset cores 0-2),
  S21 0.059-0.079 (cores 0-2,4-7). Not verified: running inside an installed APK, the APK-asset branch
  of `load()` (the harness passes a `ContextWrapper(null)`; only absolute filesystem paths were
  exercised), and recognition accuracy (one English output reads "50 pieces of code"; no fp32
  reference was available). An Android x86_64 emulator (API 37, ARM translation) crashed inside
  `libonnxruntime.so`'s static constructor before any AMEVA code ran; that is an emulator limitation
  (the same library loads on all three phones) and is not counted as a pass.
  **Defect fixed in this release**: `load()` used to pass `context.assets` together with an absolute
  model path; sherpa-onnx then logs an error and calls `exit(-1)`, killing the app process. `load()`
  now passes a null AssetManager for filesystem paths, rejects a missing file with
  `AMU_ERR_MODEL_FILE_MISSING` and mixed path sources with `AMU_ERR_MIXED_SOURCES`. Only the offline
  SenseVoice path is wrapped; sherpa-onnx streaming and diarization APIs are present in the library
  but have no wrapper and no verification.

### Bundled Native Libraries (`jni/arm64-v8a`)

- `libac_runtime.so`, `libameva_jni.so`, `libggml-ameva.so`, `libggml-base.so`, `libggml-cpu.so`,
  `libggml.so`, `libllama.so` (unchanged from v0.2.0)
- `libameva_stt_jni.so` (statically links whisper.cpp + its own isolated ggml)
- `libameva_sd_jni.so` (statically links stable-diffusion.cpp + its own isolated ggml)
- `libsherpa-onnx-jni.so`, `libsherpa-onnx-c-api.so`, `libsherpa-onnx-cxx-api.so`, `libonnxruntime.so`
  (unmodified k2-fsa/sherpa-onnx v1.13.8 prebuilts, Apache-2.0; ONNX Runtime is MIT)
- `libggml-ameva-sd.so` (the diffusion-specific build of the AMEVA backend -- do not substitute
  `libggml-ameva.so` here, see above)

### Quick Start

```kotlin
import com.ameva.amuda.stt.AmevaSpeech
import com.ameva.amuda.diffusion.AmevaDiffusion
import com.ameva.amuda.llm.AmevaLlm

// Speech-to-text
val stt = AmevaSpeech.load(context, modelPath = "/sdcard/models/ggml-base.en-q5_0.bin")
val text = stt.transcribe(pcm16, sampleRate = 16000)
stt.close()

// Image generation
val sd = AmevaDiffusion.load(context, modelPath = "/sdcard/models/sdxs.gguf")
// cfgScale = 1f: SDXS (and other guidance-distilled few-step models) need ~1.0; the default 7.0
// produces burnt, over-saturated images.
val bitmap = sd.txt2img(prompt = "a photo of a cat", width = 512, height = 512, steps = 1, cfgScale = 1f)
sd.close()

// LLM, now with automatic fastest-backend selection
val llm = AmevaLlm.load(context, modelPath = "/sdcard/models/qwen2.5-0.5b-q4_k_m.gguf") // backend = "auto" by default
val reply = llm.generate(prompt = "...")
llm.reset() // clear KV cache for the next conversation without a full reload
llm.close()
```

### Known limitations

- `AmevaSpeech`/`AmevaDiffusion` both run synchronous, per-op GPU dispatch (no batched/layer-level
  graph submission) -- this is why a 1-step 512x512 image takes minutes, not seconds, on a phone
  GPU. AmevaLlm's decode path already batches a full graph into one submission; STT and diffusion
  do not yet.
- S21 and the rest of the fleet beyond A35/A53 are unverified for the new modules in this release.
