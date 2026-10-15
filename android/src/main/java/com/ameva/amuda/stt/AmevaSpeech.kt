package com.ameva.amuda.stt

import android.content.Context
import java.io.File
import java.security.MessageDigest

/**
 * v0.3 finished-product API: load a whisper.cpp GGML model and transcribe 16 kHz mono PCM,
 * backed by whisper.cpp (third_party/whisper.cpp) + the formal AMEVA Vulkan STT backend
 * (adapters/whisper/ggml-backend-ameva.cpp), statically linked into this module's own
 * libameva_stt_jni.so (separate .so from AmevaLlm's libameva_jni.so -- each pins its own ggml
 * checkout, third_party/llama.cpp vs third_party/whisper.cpp, and the two are not
 * ABI-compatible; Android's per-.so symbol resolution keeps them from colliding in one process).
 *
 * Ground truth: only two encoder matmul shapes (f16, q5_0 weights) currently get AMEVA GPU
 * acceleration -- this extracts their SPIR-V binaries from this module's own assets the first
 * time [load] runs in a given [cacheDir] and skips re-extraction on later calls (sha256-checked).
 * Every other op in the graph runs wherever ggml's scheduler places it (normally CPU) -- this is
 * ordinary multi-backend scheduling, not a defect to silently work around.
 *
 * The caller supplies the model file; it is not bundled with this library.
 *
 * ```
 * val stt = AmevaSpeech.load(context, modelPath = "/sdcard/models/ggml-base.en-q5_0.bin")
 * val text = stt.transcribe(pcm16, sampleRate = 16000)
 * stt.close()
 * ```
 */
class AmevaSpeech private constructor(private val handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_stt_jni")
        }

        private const val ASSET_DIR = "ameva-stt"
        private val ASSET_SHA256 = mapOf(
            "matmul_f16_f32.spv" to "63ab1fecca95ef16bc1387881421d900a134c5769861bb61c548bb6f85539e50",
            "matmul_q5_0_f32.spv" to "343fd4b667135404fbe552965b59c33c1c4a37f6488f304264bccea407485829"
        )

        /**
         * [context] supplies this module's own bundled SPIR-V shader assets, extracted once to
         * [cacheDir] (default [context.filesDir]) and reused on later loads (sha256-verified,
         * re-extracted if the asset was replaced by an SDK update or the cached copy is corrupt).
         */
        fun load(
            context: Context, modelPath: String, language: String = "auto", cacheDir: String? = null
        ): AmevaSpeech {
            val resolvedCacheDir = File(cacheDir ?: context.filesDir.path, ASSET_DIR)
            resolvedCacheDir.mkdirs()
            val f16Path = extractAssetIfNeeded(context, resolvedCacheDir, "matmul_f16_f32.spv")
            val q50Path = extractAssetIfNeeded(context, resolvedCacheDir, "matmul_q5_0_f32.spv")
            return AmevaSpeech(nativeLoad(modelPath, f16Path, q50Path, language))
        }

        private fun extractAssetIfNeeded(context: Context, destDir: File, name: String): String {
            val dest = File(destDir, name)
            val expectedSha256 = ASSET_SHA256.getValue(name)
            if (dest.exists() && sha256(dest) == expectedSha256) {
                return dest.path
            }
            context.assets.open("$ASSET_DIR/$name").use { input ->
                dest.outputStream().use { output -> input.copyTo(output) }
            }
            val actual = sha256(dest)
            check(actual == expectedSha256) {
                "AmevaSpeech: extracted $name sha256 mismatch (expected $expectedSha256, got $actual)"
            }
            return dest.path
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buf = ByteArray(8192)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    digest.update(buf, 0, n)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, spirvF16Path: String, spirvQ50Path: String, language: String
        ): Long
        @JvmStatic private external fun nativeTranscribe(
            handle: Long, pcmf32: FloatArray, translate: Boolean, outBufSize: Int
        ): String
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Throws [com.ameva.amuda.AmudaException] on any failure. [pcm16] must be 16 kHz mono --
     *  whisper.cpp has no other supported input rate/channel layout, so this validates rather
     *  than silently resampling. [outBufSize] bounds the transcript in bytes. */
    fun transcribe(
        pcm16: ShortArray, sampleRate: Int, translate: Boolean = false, outBufSize: Int = 8192
    ): String {
        require(sampleRate == 16000) {
            "AmevaSpeech.transcribe: sampleRate must be 16000 (whisper.cpp's only supported " +
                "rate), got $sampleRate -- resample before calling, this will not do it silently"
        }
        val pcmf32 = FloatArray(pcm16.size) { i -> pcm16[i] / 32768.0f }
        return nativeTranscribe(handle, pcmf32, translate, outBufSize)
    }

    override fun close() {
        nativeFree(handle)
    }
}
