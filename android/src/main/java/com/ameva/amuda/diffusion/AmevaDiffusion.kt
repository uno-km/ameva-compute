package com.ameva.amuda.diffusion

import android.content.Context
import android.graphics.Bitmap
import java.io.File

/**
 * v0.4 finished-product API: load a single-file diffusion checkpoint and generate an image,
 * backed by stable-diffusion.cpp (third_party/stable-diffusion.cpp) + the ggml-ameva GPU backend
 * (adapters/llama/ggml-ameva -- the same backend AmevaLlm loads; CONV_2D/GROUP_NORM/UPSCALE/
 * TIMESTEP_EMBEDDING, the ops a diffusion UNet actually needs, already existed there and are
 * unchanged by this module), loaded as a dynamic plugin exactly like AmevaLlm's.
 *
 * The caller supplies the model file; it is not bundled with this library.
 *
 * ```
 * val sd = AmevaDiffusion.load(context, modelPath = "/sdcard/models/sd15-pruned.safetensors")
 * val bitmap = sd.txt2img(prompt = "a photo of a cat", width = 512, height = 512)
 * sd.close()
 * ```
 */
class AmevaDiffusion private constructor(private val handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_sd_jni")
        }

        /**
         * [nThreads] <= 0 uses stable-diffusion.cpp's own default.
         *
         * Loads libggml-ameva-sd.so, not AmevaLlm's libggml-ameva.so -- same backend source, but
         * a separate build against stable-diffusion.cpp's own pinned ggml (GGML_MAX_NAME=160,
         * an ABI-level difference from llama.cpp's ggml). Mixing them would silently corrupt
         * ggml_tensor layout; see android/jni/CMakeLists.txt's ggml-ameva-sd target comment.
         */
        fun load(context: Context, modelPath: String, nThreads: Int = 0): AmevaDiffusion {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva-sd.so").path
            return AmevaDiffusion(nativeLoad(modelPath, backendPluginPath, nThreads))
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, backendPluginPath: String, nThreads: Int
        ): Long
        @JvmStatic private external fun nativeTxt2Img(
            handle: Long, prompt: String, width: Int, height: Int, steps: Int, seed: Long
        ): ByteArray
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Throws [com.ameva.amuda.AmudaException] on any failure. [seed] -1 lets
     *  stable-diffusion.cpp pick a random seed; a fixed value makes the result reproducible. */
    fun txt2img(
        prompt: String, width: Int = 512, height: Int = 512, steps: Int = 20, seed: Long = -1L
    ): Bitmap {
        val rgb = nativeTxt2Img(handle, prompt, width, height, steps, seed)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val argb = IntArray(width * height)
        for (i in argb.indices) {
            val r = rgb[i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            argb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bitmap.setPixels(argb, 0, width, 0, 0, width, height)
        return bitmap
    }

    override fun close() {
        nativeFree(handle)
    }
}
