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
 * // Guidance-distilled few-step models (SDXS, SD-Turbo, LCM) need cfgScale = 1f, e.g.
 * // sd.txt2img(prompt, width = 512, height = 512, steps = 1, cfgScale = 1f)
 * sd.close()
 * ```
 */
class AmevaDiffusion private constructor(private val handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_sd_jni")
        }

        /**
         * [nThreads] <= 0 uses stable-diffusion.cpp's own default. [cacheDir] (default
         * [context.filesDir]) is where the first load's shader pipeline compiles and per-shape
         * GEMV auto-tuning get persisted -- without it (ggml-ameva's own fallback is $HOME, which
         * does not exist in an app process) every load redoes this work from scratch. This is most
         * of why a first AmevaDiffusion generation takes minutes: a UNet has dozens of distinct
         * weight shapes, each paying its own one-time compile+tune cost with nowhere to cache it.
         *
         * Loads libggml-ameva-sd.so, not AmevaLlm's libggml-ameva.so -- same backend source, but
         * a separate build against stable-diffusion.cpp's own pinned ggml (GGML_MAX_NAME=160,
         * an ABI-level difference from llama.cpp's ggml). Mixing them would silently corrupt
         * ggml_tensor layout; see android/jni/CMakeLists.txt's ggml-ameva-sd target comment.
         */
        fun load(
            context: Context, modelPath: String, nThreads: Int = 0, cacheDir: String? = null
        ): AmevaDiffusion {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva-sd.so").path
            val resolvedCacheDir = cacheDir ?: context.filesDir.path
            return AmevaDiffusion(nativeLoad(modelPath, backendPluginPath, resolvedCacheDir, nThreads))
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, backendPluginPath: String, cacheDir: String, nThreads: Int
        ): Long
        @JvmStatic private external fun nativeTxt2Img(
            handle: Long, prompt: String, width: Int, height: Int, steps: Int, cfgScale: Float, seed: Long
        ): ByteArray
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Throws [com.ameva.amuda.AmudaException] on any failure. [seed] -1 lets
     *  stable-diffusion.cpp pick a random seed; a fixed value makes the result reproducible.
     *
     *  [cfgScale] is the classifier-free-guidance scale; a value <= 0 keeps stable-diffusion.cpp's
     *  default of 7.0. Guidance-distilled one- to four-step models (SDXS, SD-Turbo, LCM, ...) need
     *  about 1.0: at the default they come out burnt and over-saturated (measured on SDXS, 256x256,
     *  1 step: pure-black channel values 43% at 7.0 versus 0.1% at 1.0, on both CPU and the AMEVA
     *  GPU path), which is a sampler setting, not a backend defect. */
    fun txt2img(
        prompt: String, width: Int = 512, height: Int = 512, steps: Int = 20, seed: Long = -1L,
        cfgScale: Float = 0f
    ): Bitmap {
        val rgb = nativeTxt2Img(handle, prompt, width, height, steps, cfgScale, seed)
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
