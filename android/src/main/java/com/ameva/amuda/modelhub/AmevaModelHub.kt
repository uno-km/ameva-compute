package com.ameva.amuda.modelhub

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * "엔진을 파는 거지 기름을 파는 게 아니다" -- this SDK ships the inference engine, not model
 * weights. AmevaLlm/AmevaSpeech always take a caller-supplied model path; nothing is bundled.
 * This is the one-line convenience the policy still owes app developers: download a GGUF/whisper
 * model from HuggingFace or any HTTPS URL straight into the app's own storage, with no silent
 * partial-file success and no hidden retry-forever loop.
 *
 * ```
 * val modelFile = AmevaModelHub.download(
 *     "https://huggingface.co/.../qwen2.5-1.5b-instruct-q4_k_m.gguf",
 *     File(context.filesDir, "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
 * )
 * ```
 */
object AmevaModelHub {

    /**
     * Downloads [url] to [dest] on [Dispatchers.IO]. Streams to a sibling `.part` file and
     * renames it onto [dest] only after every byte is written and the connection closes cleanly
     * -- a crash or network drop mid-transfer leaves no half-written file at [dest] for a caller
     * to mistake for a complete model. Re-running with the same [dest] always re-downloads: this
     * does not check whether a file already there is complete or matches [url].
     *
     * @param onProgress invoked on [Dispatchers.IO] (not the caller's original dispatcher) after
     *        each chunk; [totalBytes] is -1 if the server did not send Content-Length.
     * @throws IOException on any non-2xx HTTP status (with the status in the message) or I/O
     *         failure; [dest] is left untouched (only the `.part` file is created/deleted).
     */
    suspend fun download(
        url: String,
        dest: File,
        onProgress: ((bytesRead: Long, totalBytes: Long) -> Unit)? = null
    ): File = withContext(Dispatchers.IO) {
        dest.parentFile?.mkdirs()
        val partFile = File(dest.parentFile, "${dest.name}.part")

        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.connect()

            val status = connection.responseCode
            if (status !in 200..299) {
                throw IOException("AmevaModelHub.download: HTTP $status for $url")
            }

            val totalBytes = connection.contentLengthLong
            var bytesRead = 0L
            connection.inputStream.use { input ->
                partFile.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        bytesRead += n
                        onProgress?.invoke(bytesRead, totalBytes)
                    }
                }
            }
        } catch (e: Exception) {
            partFile.delete()
            throw e
        } finally {
            connection.disconnect()
        }

        if (!partFile.renameTo(dest)) {
            partFile.delete()
            throw IOException("AmevaModelHub.download: failed to move ${partFile.path} to ${dest.path}")
        }
        dest
    }
}
