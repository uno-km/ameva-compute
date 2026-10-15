package com.ameva.amuda.llm

import android.content.Context
import com.ameva.amuda.AmudaException
import java.io.File

/**
 * v0.2 finished-product API: load a GGUF model and generate text, backed by llama.cpp +
 * the ggml-ameva GPU backend (loaded as a dynamic plugin, same mechanism as the Termux
 * deployment -- no static link, no upstream ggml-vulkan fallback).
 *
 * The caller supplies the model file; it is not bundled with this library.
 *
 * ```
 * val llm = AmevaLlm.load(context, modelPath = "/sdcard/models/qwen2.5-0.5b-q4_k_m.gguf")
 * val text = llm.generate(prompt = "...", maxTokens = 128, temperature = 0.8f, seed = 42)
 * llm.close()
 * ```
 */
class AmevaLlm private constructor(private val handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_jni")
        }

        /**
         * [context] supplies the on-device path to the bundled libggml-ameva.so
         * (Context.applicationInfo.nativeLibraryDir) -- the dynamic backend plugin is never
         * statically linked into this library's own .so -- and, by default, [context.filesDir]
         * as the AMEVA_PIPELINE_CACHE / AMEVA_QMV_TUNE_FILE location. Without a per-app-writable
         * directory, ggml-ameva falls back to $HOME, which does not exist in an app process, and
         * both silently turn off: every load recompiles pipelines from scratch and GEMV
         * auto-tuning measurements are never kept. Measured impact (Tab S8+, Adreno 730, 2026-10):
         * cold first-token 1972ms -> 297ms, 64-token generation 2.4-2.5s -> 1.67-1.77s.
         */
        fun load(context: Context, modelPath: String, nCtx: Int = 2048, cacheDir: String? = null): AmevaLlm {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva.so").path
            val resolvedCacheDir = cacheDir ?: context.filesDir.path
            return AmevaLlm(nativeLoad(modelPath, backendPluginPath, resolvedCacheDir, nCtx))
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, backendPluginPath: String, cacheDir: String, nCtx: Int
        ): Long
        @JvmStatic private external fun nativeGenerate(
            handle: Long, prompt: String, maxTokens: Int, temperature: Float, seed: Long, outBufSize: Int
        ): String
        @JvmStatic private external fun nativeReset(handle: Long)
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Throws [AmudaException] on any failure (model/backend load issues surface at [load] time;
     *  tokenize/decode/output-buffer issues surface here). [outBufSize] bounds the generated text
     *  in bytes; generation that would not fit fails rather than silently truncating. */
    fun generate(
        prompt: String, maxTokens: Int = 128, temperature: Float = 0.8f, seed: Long = 42L,
        outBufSize: Int = 4096
    ): String = nativeGenerate(handle, prompt, maxTokens, temperature, seed, outBufSize)

    /** Clears the KV cache so the next [generate] call starts a fresh conversation/summary,
     *  without the reload cost of [close] + [load] (model weights and GPU pipeline/tuning state
     *  stay resident). */
    fun reset() {
        nativeReset(handle)
    }

    override fun close() {
        nativeFree(handle)
    }
}
