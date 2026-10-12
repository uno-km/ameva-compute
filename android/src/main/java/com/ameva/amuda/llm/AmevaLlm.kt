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
         * statically linked into this library's own .so.
         */
        fun load(context: Context, modelPath: String, nCtx: Int = 2048): AmevaLlm {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva.so").path
            return AmevaLlm(nativeLoad(modelPath, backendPluginPath, nCtx))
        }

        @JvmStatic private external fun nativeLoad(modelPath: String, backendPluginPath: String, nCtx: Int): Long
        @JvmStatic private external fun nativeGenerate(
            handle: Long, prompt: String, maxTokens: Int, temperature: Float, seed: Long, outBufSize: Int
        ): String
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Throws [AmudaException] on any failure (model/backend load issues surface at [load] time;
     *  tokenize/decode/output-buffer issues surface here). [outBufSize] bounds the generated text
     *  in bytes; generation that would not fit fails rather than silently truncating. */
    fun generate(
        prompt: String, maxTokens: Int = 128, temperature: Float = 0.8f, seed: Long = 42L,
        outBufSize: Int = 4096
    ): String = nativeGenerate(handle, prompt, maxTokens, temperature, seed, outBufSize)

    override fun close() {
        nativeFree(handle)
    }
}
