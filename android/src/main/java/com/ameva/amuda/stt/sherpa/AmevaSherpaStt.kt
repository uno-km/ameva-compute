package com.ameva.amuda.stt.sherpa

import android.content.Context
import java.io.File
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig

/**
 * SenseVoice / Moonshine / diarization / streaming-Zipformer STT via k2-fsa/sherpa-onnx
 * (upstream sherpa-onnx-1.13.8.aar, fetched into android/libs/ by android/scripts/fetch_sherpa_onnx.sh
 * -- no Maven Central listing exists for it).
 *
 * This is a DIFFERENT engine from [com.ameva.amuda.stt.AmevaSpeech]: ONNX Runtime on CPU/NNAPI,
 * not whisper.cpp + the ggml-ameva Vulkan backend. Nothing here is AMEVA-GPU-accelerated -- say so
 * plainly rather than letting the package name imply otherwise. Use [AmevaSpeech] when GPU
 * acceleration matters and a whisper.cpp-compatible model is enough; use this when the task needs
 * SenseVoice's emotion/sound-event tags, Moonshine, or a model whisper.cpp cannot run.
 *
 * Only the offline (non-streaming) SenseVoice path is wrapped here. sherpa-onnx's streaming
 * (OnlineRecognizer/Zipformer) and diarization (OfflineSpeakerDiarization) APIs are present in the
 * same upstream .aar and reachable the same way, but have no Kotlin wrapper or device verification
 * in this module yet.
 *
 * ```
 * val sv = AmevaSenseVoice.load(
 *     context,
 *     modelPath = "${context.filesDir}/sense-voice/model.int8.onnx",
 *     tokensPath = "${context.filesDir}/sense-voice/tokens.txt"
 * )
 * val result = sv.transcribe(pcm16, sampleRate = 16000)
 * println("${result.text} (lang=${result.language}, emotion=${result.emotion}, event=${result.event})")
 * sv.close()
 * ```
 */
class AmevaSenseVoice private constructor(private val recognizer: OfflineRecognizer) : AutoCloseable {

    /** [text] has inverse-text-normalization already applied (numbers, punctuation) when the
     *  model config requests it. [emotion]/[event] are SenseVoice-specific: short tags such as
     *  "HAPPY"/"SAD" and "Laughter"/"Applause", empty string when the model found nothing notable. */
    data class Result(
        val text: String,
        val language: String,
        val emotion: String,
        val event: String
    )

    companion object {
        /**
         * @param modelPath Filesystem path to a SenseVoice ONNX model (e.g.
         *        `sense-voice-small.onnx`) -- not bundled with this library.
         * @param tokensPath Filesystem path to the matching `tokens.txt`.
         * @param language BCP-47-ish language hint ("auto" lets the model detect it).
         * @param numThreads CPU threads for ONNX Runtime; this path has no GPU backend.
         */
        fun load(
            context: Context,
            modelPath: String,
            tokensPath: String,
            language: String = "auto",
            useInverseTextNormalization: Boolean = true,
            numThreads: Int = 2
        ): AmevaSenseVoice {
            val config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80, dither = 0.0f),
                modelConfig = OfflineModelConfig(
                    senseVoice = OfflineSenseVoiceModelConfig(
                        model = modelPath,
                        language = language,
                        useInverseTextNormalization = useInverseTextNormalization
                    ),
                    tokens = tokensPath,
                    numThreads = numThreads,
                    modelType = "sense_voice"
                )
            )
            // sherpa-onnx does NOT fall back between the two sources: with a non-null AssetManager
            // every path is opened as an APK asset, and an absolute path there makes the native
            // layer log an error and call SHERPA_ONNX_EXIT(-1) (kills the process; file-utils.cc
            // ReadFile(AAssetManager*) in v1.13.8). So an absolute path -- e.g. from
            // AmevaModelHub.download() -- must go in with a null AssetManager, and is checked here
            // so a missing file is a catchable exception instead of a process exit.
            val fromFilesystem = modelPath.startsWith("/")
            require(fromFilesystem == tokensPath.startsWith("/")) {
                "AMU_ERR_MIXED_SOURCES: modelPath and tokensPath must both be absolute filesystem paths " +
                    "or both be APK asset names (modelPath=$modelPath, tokensPath=$tokensPath)"
            }
            if (fromFilesystem) {
                for (path in listOf(modelPath, tokensPath)) {
                    require(File(path).isFile) { "AMU_ERR_MODEL_FILE_MISSING: $path" }
                }
            }
            return AmevaSenseVoice(OfflineRecognizer(if (fromFilesystem) null else context.assets, config))
        }
    }

    /** [pcm16] must be 16 kHz mono -- SenseVoice's only supported input rate, same convention as
     *  [com.ameva.amuda.stt.AmevaSpeech.transcribe]: validated, never silently resampled. */
    fun transcribe(pcm16: ShortArray, sampleRate: Int): Result {
        require(sampleRate == 16000) {
            "AmevaSenseVoice.transcribe: sampleRate must be 16000, got $sampleRate -- resample before calling"
        }
        val pcmf32 = FloatArray(pcm16.size) { i -> pcm16[i] / 32768.0f }
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(pcmf32, sampleRate)
            recognizer.decode(stream)
            val r = recognizer.getResult(stream)
            return Result(text = r.text, language = r.lang, emotion = r.emotion, event = r.event)
        } finally {
            stream.release()
        }
    }

    override fun close() {
        recognizer.release()
    }
}
