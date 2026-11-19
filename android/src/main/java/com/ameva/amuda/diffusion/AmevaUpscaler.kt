package com.ameva.amuda.diffusion

import android.content.Context
import java.io.File

/**
 * ESRGAN upscaler (RealESRGAN weights) from stable-diffusion.cpp, on the AMEVA GPU plugin or the CPU.
 * Throws [com.ameva.amuda.AmudaException] on failure. Not thread-safe; one call at a time.
 */
class AmevaUpscaler private constructor(private var handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_sd_jni")
        }

        /** [gpu] false runs on the CPU. [tileSize] <= 0: 128. */
        @JvmStatic
        fun load(context: Context, modelPath: String, gpu: Boolean, nThreads: Int, tileSize: Int): AmevaUpscaler {
            val plugin = if (gpu) File(context.applicationInfo.nativeLibraryDir, "libggml-ameva-sd.so").path else ""
            return AmevaUpscaler(nativeLoad(modelPath, plugin, nThreads, tileSize))
        }

        @JvmStatic private external fun nativeLoad(modelPath: String, plugin: String, nThreads: Int, tile: Int): Long
        @JvmStatic private external fun nativeFactor(handle: Long): Int
        @JvmStatic private external fun nativeUpscale(handle: Long, rgb: ByteArray, width: Int, height: Int): ByteArray
        @JvmStatic private external fun nativeFree(handle: Long)
    }

    /** Scale factor of the loaded model (for example 4). */
    val factor: Int get() {
        check(handle != 0L) { "AmevaUpscaler is closed" }
        return nativeFactor(handle)
    }

    /** Packed RGB8 in; packed RGB8 of (width*factor) x (height*factor) out. */
    @Synchronized
    fun upscale(rgb: ByteArray, width: Int, height: Int): ByteArray {
        check(handle != 0L) { "AmevaUpscaler is closed" }
        return nativeUpscale(handle, rgb, width, height)
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }
}
