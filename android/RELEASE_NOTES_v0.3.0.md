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
- **AmevaDiffusion**: verified on A35 only -- one model (SDXS, single-step distilled, 683 MB GGUF),
  512x512, 1 sampling step, 1 run. Produced a correct, recognizable image (the color saturation is
  this specific model's known 1-step-distillation characteristic, not a backend defect). Two VAE
  decode tensors exceed a single Vulkan buffer binding and correctly, explicitly fall back to CPU
  (logged, not hidden) -- this adds real wall-clock time (~7.5 minutes end to end on A35 for this
  run). A53, S21, other models, other resolutions, multi-step sampling, img2img, and LoRA are
  **not** verified.
- **Auto backend-select**: verified on A35 -- real per-backend timings for a fixed GEMV (cpu
  6.77 ms, vulkan 3.39 ms, opencl 2.92 ms/op), correctly selected OpenCL as this device's fastest
  path. Not yet verified on other GPU families.
- **AmevaModelHub**: compiles clean, pure Kotlin, no native code -- not exercised against a real
  download in this cycle.
- **AmevaSenseVoice**: the sherpa-onnx v1.13.8 native libraries as packaged in this release AAR
  (`libsherpa-onnx-c-api.so`, `libsherpa-onnx-jni.so` byte-identical to upstream; `libonnxruntime.so`
  differs from upstream only in ELF section-table metadata, 8 bytes) were run on Samsung Galaxy A35,
  A53 and S21 (Termux, CPU/ONNX Runtime, SenseVoice int8 `2024-07-17`, 5 upstream test wavs: en, zh,
  ja, ko, yue). All three produced identical transcripts with language/emotion/event tags. Warm
  real-time factor (4 threads): A35 0.31-0.37 and A53 0.31-0.33 (ssh cpuset cores 0-2), S21 0.063-0.068
  (cores 0-2,4-7). On the S21 the sherpa Java API through `libsherpa-onnx-jni.so` (OpenJDK 21,
  `AmevaSenseVoice`'s configuration, `AssetManager` = null) gave the same transcripts. Not verified:
  the `AmevaSenseVoice` Kotlin class itself (it needs an Android `Context`; no APK was installed),
  and recognition accuracy (one English output reads "50 pieces of code"; an fp32 reference run was
  not possible). Only the offline SenseVoice path is wrapped; sherpa-onnx streaming and diarization
  APIs are present in the library but have no wrapper and no verification.

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
val bitmap = sd.txt2img(prompt = "a photo of a cat", width = 512, height = 512, steps = 1)
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
