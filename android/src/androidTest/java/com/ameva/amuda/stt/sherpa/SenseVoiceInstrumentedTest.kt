package com.ameva.amuda.stt.sherpa

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real-runtime proof for [AmevaSenseVoice] (the Kotlin wrapper itself, in an ordinary Android app
 * process -- not a Termux JVM harness). SenseVoice is not bundled, so the model directory is supplied
 * out of band and the test FAILS (never skips) when it is absent:
 *
 *   adb push sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 /data/local/tmp/sv
 *   adb shell am instrument -w -e svDir /data/local/tmp/sv \
 *       -e class com.ameva.amuda.stt.sherpa.SenseVoiceInstrumentedTest \
 *       com.ameva.amuda.test/androidx.test.runner.AndroidJUnitRunner
 *
 * The directory must hold `model.int8.onnx`, `tokens.txt` and `test_wavs/{en,zh,ja,ko,yue}.wav`
 * (the upstream sherpa-onnx SenseVoice release layout). Results are logged under tag "AmevaSenseVoice".
 */
class SenseVoiceInstrumentedTest {

    private companion object {
        const val TAG = "AmevaSenseVoice"

        /** wav -> (language tag SenseVoice must report, substring the transcript must contain). */
        val EXPECTED = linkedMapOf(
            "en" to ("<|en|>" to "tribal chieftain"),
            "zh" to ("<|zh|>" to "开饭时间"),
            "ja" to ("<|ja|>" to "弁当"),
            "ko" to ("<|ko|>" to "생각을"),
            "yue" to ("<|yue|>" to "表达唔到"),
        )
    }

    private val args get() = InstrumentationRegistry.getArguments()
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun modelDir(): File {
        val dir = File(args.getString("svDir") ?: "/data/local/tmp/sv")
        for (name in listOf("model.int8.onnx", "tokens.txt", "test_wavs/en.wav")) {
            assertTrue("AMU_ERR_MODEL_FILE_MISSING: ${File(dir, name)} (see class KDoc for the adb push)", File(dir, name).isFile)
        }
        return dir
    }

    private fun load(dir: File, threads: Int = 2) = AmevaSenseVoice.load(
        context,
        modelPath = File(dir, "model.int8.onnx").absolutePath,
        tokensPath = File(dir, "tokens.txt").absolutePath,
        numThreads = threads,
    )

    /** 16 kHz mono PCM16 -> ShortArray; walks RIFF chunks instead of assuming a 44-byte header. */
    private fun readWav(file: File): ShortArray {
        val b = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(b.array(), 0, 4))
        var pos = 12
        while (pos + 8 <= b.limit()) {
            val id = String(ByteArray(4) { b.get(pos + it) })
            val len = b.getInt(pos + 4)
            if (id == "fmt ") {
                assertEquals("PCM format", 1, b.getShort(pos + 8).toInt())
                assertEquals("mono", 1, b.getShort(pos + 10).toInt())
                assertEquals("sample rate", 16000, b.getInt(pos + 12))
                assertEquals("bits per sample", 16, b.getShort(pos + 22).toInt())
            }
            if (id == "data") {
                val out = ShortArray(len / 2)
                b.position(pos + 8)
                b.asShortBuffer().get(out)
                return out
            }
            pos += 8 + len + (len and 1)
        }
        fail("AMU_ERR_NO_DATA_CHUNK: $file")
        error("unreachable")
    }

    @Test
    fun transcribesFiveLanguagesOnTheRealModel() {
        val dir = modelDir()
        load(dir).use { sv ->
            for ((name, expected) in EXPECTED) {
                val pcm = readWav(File(dir, "test_wavs/$name.wav"))
                val t0 = System.nanoTime()
                val r = sv.transcribe(pcm, 16000)
                val ms = (System.nanoTime() - t0) / 1_000_000
                Log.i(TAG, "wav=$name audio_s=${"%.2f".format(pcm.size / 16000.0)} ms=$ms lang=${r.language} " +
                    "emotion=${r.emotion} event=${r.event} text=${r.text}")
                assertEquals("language tag for $name", expected.first, r.language)
                assertTrue("transcript for $name must contain '${expected.second}', got '${r.text}'", r.text.contains(expected.second))
                assertEquals("emotion for $name", "<|NEUTRAL|>", r.emotion)
                assertEquals("event for $name", "<|Speech|>", r.event)
            }
        }
    }

    /** Guards the SHERPA_ONNX_EXIT(-1) defect: a missing file must be a catchable exception. */
    @Test
    fun missingModelFileIsACatchableErrorNotAProcessExit() {
        val dir = modelDir()
        try {
            AmevaSenseVoice.load(context, modelPath = File(dir, "does-not-exist.onnx").absolutePath,
                tokensPath = File(dir, "tokens.txt").absolutePath)
            fail("load() must reject a missing model file")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message, e.message!!.startsWith("AMU_ERR_MODEL_FILE_MISSING"))
        }
    }

    @Test
    fun mixedAssetAndFilesystemPathsAreRejected() {
        val dir = modelDir()
        try {
            AmevaSenseVoice.load(context, modelPath = File(dir, "model.int8.onnx").absolutePath, tokensPath = "tokens.txt")
            fail("load() must reject mixed path sources")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message, e.message!!.startsWith("AMU_ERR_MIXED_SOURCES"))
        }
    }

    @Test
    fun wrongSampleRateIsRejectedNotResampled() {
        val dir = modelDir()
        load(dir).use { sv ->
            try {
                sv.transcribe(ShortArray(8000), 8000)
                fail("transcribe() must reject a non-16 kHz input")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message, e.message!!.contains("16000"))
            }
        }
    }
}
