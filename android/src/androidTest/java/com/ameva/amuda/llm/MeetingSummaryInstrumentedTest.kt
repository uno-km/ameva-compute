package com.ameva.amuda.llm

import androidx.test.platform.app.InstrumentationRegistry
import android.util.Log
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * v0.2 real-device proof: AmevaLlm.generate() produces a real, non-garbled summary of a meeting
 * transcript on real hardware -- not just "load succeeded" (an AmudaException on any load/decode
 * failure already fails this test; see AmudaContext.kt's error-propagation convention).
 *
 * Model file: pushed ahead of time to this app's own external files dir (scoped-storage safe),
 * reusing the qwen2.5-0.5b-instruct-q4_k_m.gguf already present on-device -- no download.
 */
class MeetingSummaryInstrumentedTest {

    companion object {
        private const val TAG = "AmevaAARPoC"
        private const val MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

        // Qwen2.5-Instruct's own ChatML template -- plain completion-style prompting on an
        // Instruct-tuned model just continues/echoes the input instead of following the
        // instruction (confirmed: first attempt without this template did exactly that).
        private val MEETING_TRANSCRIPT = """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            다음 회의록을 한국어로 3줄 이내로 요약해.

            [회의록 - AMUDA Android SDK 주간 점검]
            김은호: v0.1 JNI 레이어는 실기기 5대 중 4대(S25, S21, A35, A53)에서 Vulkan과 OpenCL
            양쪽 다 vector_add 결과가 정확히 일치하는 것을 확인했습니다. 탭 S8+는 다른 팀이
            점유해서 아직 확인 못했습니다.
            이수진: v0.2 쪽은 ggml-ameva 백엔드를 동적 플러그인으로 로드하는 방식으로 붙였고,
            플래시 어텐션은 꺼뒀습니다. 업스트림 Vulkan 백엔드는 아예 로드하지 않습니다.
            박민재: 보안 가드는 사용자 결정으로 서드파티 앱 제한을 풀었습니다. 커밋 완료했습니다.
            김은호: 다음 주까지 안드로이드 스튜디오에서 실기기 검증을 마무리하고, 깃허브 릴리스와
            Labs 공지는 검증이 끝난 뒤에 올리기로 했습니다.<|im_end|>
            <|im_start|>assistant
        """.trimIndent()
    }

    @Test
    fun generateProducesNonGarbledSummary() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // Internal storage (context.filesDir), not external: external app-specific storage goes
        // through a FUSE layer that only recognizes a file as app-readable if the app itself wrote
        // it -- a file placed there by `adb shell cp` (UID shell) fails open() with EACCES even
        // when `ls`/`run-as ls` show it as readable. Internal storage has no such restriction.
        val modelFile = File(File(context.filesDir, "models"), MODEL_FILE)
        assertTrue("model file missing at ${modelFile.path}; push it before running this test",
            modelFile.exists())

        val llm = AmevaLlm.load(context, modelPath = modelFile.path)
        try {
            val summary = llm.generate(
                prompt = MEETING_TRANSCRIPT, maxTokens = 150, temperature = 0.7f, seed = 123L
            )
            Log.i(TAG, "MeetingSummary device=${android.os.Build.MODEL} output=\"$summary\"")

            assertFalse("summary was blank", summary.isBlank())
            assertTrue("summary suspiciously short (${summary.length} chars): \"$summary\"",
                summary.trim().length >= 10)

            val replacementChars = summary.count { it == '�' }
            assertTrue("summary contains $replacementChars U+FFFD replacement character(s) " +
                "(encoding corruption): \"$summary\"", replacementChars == 0)

            val controlChars = summary.count { it.code in 0x00..0x08 || it.code in 0x0E..0x1F }
            assertTrue("summary contains $controlChars raw control character(s) " +
                "(likely garbled output): \"$summary\"", controlChars == 0)
        } finally {
            llm.close()
        }
    }
}
