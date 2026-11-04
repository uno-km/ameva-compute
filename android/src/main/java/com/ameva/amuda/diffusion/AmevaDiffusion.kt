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

    /**
     * How [load] sizes the GPU ("AMEVA0") placement against currently free device memory. Mirrors
     * [com.ameva.amuda.llm.AmevaLlm.GpuBudgetMode] -- same four modes, same opt-in contract:
     * [DEFAULT] is the only mode that reproduces the previous fixed behavior (the whole model
     * unconditionally placed on "AMEVA0"), which can OOM-kill the whole process if it does not
     * fit. Picking any other mode is a deliberate caller decision, never automatic.
     *
     * Values match [AmevaSdGpuBudgetMode] in ameva_sd_api.h by declaration order -- do not reorder
     * without updating the native enum and ameva_sd_jni.cpp's nativeLoad binding together.
     */
    enum class GpuBudgetMode {
        /** No budget-fit: unconditional "AMEVA0", exactly like before this option existed. */
        DEFAULT,
        /** Fit the whole model in the budget, or fail with [com.ameva.amuda.AmudaException] --
         *  never spills to CPU silently or otherwise. A coarser, file-size-based check than
         *  [GPU_THEN_CPU]'s own component-level estimate. */
        GPU_ONLY,
        /** Hands off to stable-diffusion.cpp's own component-level (DiT/VAE/conditioner)
         *  auto-fit placement. Always logged (logcat tag "ameva_sd") -- never silent. */
        GPU_THEN_CPU,
        /** Runs entirely on CPU; the AMEVA plugin is not placed at all. */
        CPU_ONLY,
    }

    /**
     * @param mode see [GpuBudgetMode]. Default [GpuBudgetMode.DEFAULT] (no budget-fit).
     * @param budgetFraction fraction of the device's currently free memory the budget may use,
     *        e.g. 0.95f for 95%. Ignored when [mode] is DEFAULT or CPU_ONLY.
     */
    data class GpuBudgetOptions(
        val mode: GpuBudgetMode = GpuBudgetMode.DEFAULT,
        val budgetFraction: Float = 0.95f,
    )

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
         *
         * [gpuBudgetOptions] null (default): no budget-fit, same fixed behavior as before this
         * option existed. See [GpuBudgetOptions] to opt into fitting the GPU placement against
         * free device memory instead -- recommended for any model whose size is not known in
         * advance to comfortably fit the target device's RAM.
         */
        fun load(
            context: Context, modelPath: String, nThreads: Int = 0, cacheDir: String? = null,
            gpuBudgetOptions: GpuBudgetOptions? = null
        ): AmevaDiffusion {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva-sd.so").path
            val resolvedCacheDir = cacheDir ?: context.filesDir.path
            val mode = gpuBudgetOptions?.mode ?: GpuBudgetMode.DEFAULT
            val fraction = gpuBudgetOptions?.budgetFraction ?: 0.95f
            return AmevaDiffusion(nativeLoad(modelPath, backendPluginPath, resolvedCacheDir, nThreads,
                mode.ordinal, fraction))
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, backendPluginPath: String, cacheDir: String, nThreads: Int,
            gpuBudgetMode: Int, gpuBudgetFraction: Float
        ): Long
        @JvmStatic private external fun nativeTxt2Img(
            handle: Long, prompt: String, width: Int, height: Int, steps: Int, cfgScale: Float, seed: Long
        ): ByteArray
        @JvmStatic private external fun nativeFree(handle: Long)

        /**
         * Extended load: multi-file models (Z-Image = diffusion + LLM + VAE), TAESD decode,
         * component placement and diffusion-weight residency. See [LoadOptions].
         * Throws [com.ameva.amuda.AmudaException] on any failure (code = AmevaSdStatus).
         */
        @JvmStatic
        fun load(context: Context, options: LoadOptions): AmevaDiffusion {
            val plugin = if (options.gpuBudgetMode == GpuBudgetMode.CPU_ONLY) ""
                         else File(context.applicationInfo.nativeLibraryDir, "libggml-ameva-sd.so").path
            val cache = options.cacheDir ?: context.filesDir.path
            val paths = arrayOf(options.modelPath, options.diffusionModelPath, options.llmPath,
                options.clipLPath, options.t5xxlPath, options.vaePath, options.taesdPath,
                options.clipVisionPath, options.ipAdapterPath, options.controlNetPath,
                options.motionModulePath, options.photoMakerPath)
            return AmevaDiffusion(nativeLoadEx(paths, plugin, cache, options.nThreads,
                options.gpuBudgetMode.ordinal, options.budgetFraction,
                options.textEncoderPlacement.ordinal, options.vaePlacement.ordinal,
                options.diffusionParams.ordinal, options.enableMmap))
        }

        /** Packed RGB8 (as returned by [generate]) to an ARGB_8888 bitmap. */
        @JvmStatic
        fun rgbToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap {
            require(rgb.size == width * height * 3) { "rgb.size must be width*height*3" }
            val argb = IntArray(width * height)
            for (i in argb.indices) {
                val r = rgb[i * 3].toInt() and 0xFF
                val g = rgb[i * 3 + 1].toInt() and 0xFF
                val b = rgb[i * 3 + 2].toInt() and 0xFF
                argb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(argb, 0, width, 0, 0, width, height)
            return bitmap
        }

        /** A bitmap to packed RGB8 (alpha dropped), e.g. for [GenerateParams.initRgb]. */
        @JvmStatic
        fun bitmapToRgb(bitmap: Bitmap): ByteArray {
            val w = bitmap.width
            val h = bitmap.height
            val argb = IntArray(w * h)
            bitmap.getPixels(argb, 0, w, 0, 0, w, h)
            val rgb = ByteArray(w * h * 3)
            for (i in argb.indices) {
                val p = argb[i]
                rgb[i * 3] = ((p shr 16) and 0xFF).toByte()
                rgb[i * 3 + 1] = ((p shr 8) and 0xFF).toByte()
                rgb[i * 3 + 2] = (p and 0xFF).toByte()
            }
            return rgb
        }

        @JvmStatic private external fun nativeLoadEx(
            paths: Array<String?>, backendPluginPath: String, cacheDir: String, nThreads: Int,
            budgetMode: Int, budgetFraction: Float, textEncoderPlacement: Int, vaePlacement: Int,
            diffusionParams: Int, enableMmap: Boolean
        ): Long
        /** videoFrames 0: one image; > 0: AnimateDiff video (needs motionModulePath at load). */
        @JvmStatic private external fun nativeGenerate(
            handle: Long, params: GenerateParams, listener: ProgressListener?, videoFrames: Int, fps: Int
        ): ByteArray
        @JvmStatic private external fun nativeCancel(handle: Long)
        @JvmStatic private external fun nativePlacement(handle: Long): String
    }

    /** Where a model component runs. Ignored under [GpuBudgetMode.GPU_THEN_CPU] (auto-fit decides,
     *  logged) and [GpuBudgetMode.CPU_ONLY]. Ordinals match AmevaSdPlacement. */
    enum class Placement { GPU, CPU }

    /** Diffusion-weight residency. [DISK] re-reads weights from the file per use and frees them
     *  afterwards (lowest steady memory). Ignored under GPU_THEN_CPU. Ordinals match AmevaSdParamsResidency. */
    enum class ParamsResidency { RESIDENT, DISK }

    /** Java-friendly mutable options for [load] (all setters). At least [modelPath] or
     *  [diffusionModelPath] must be set. */
    class LoadOptions {
        var modelPath: String? = null
        var diffusionModelPath: String? = null
        var llmPath: String? = null
        var clipLPath: String? = null
        var t5xxlPath: String? = null
        var vaePath: String? = null
        var taesdPath: String? = null
        /** <= 0: stable-diffusion.cpp default. */
        var nThreads: Int = 0
        /** null: Context.filesDir. */
        var cacheDir: String? = null
        var gpuBudgetMode: GpuBudgetMode = GpuBudgetMode.DEFAULT
        var budgetFraction: Float = 0.95f
        var textEncoderPlacement: Placement = Placement.GPU
        var vaePlacement: Placement = Placement.GPU
        var diffusionParams: ParamsResidency = ParamsResidency.RESIDENT
        var enableMmap: Boolean = true
        /** CLIP-Vision ViT-H/14 encoder; required with [ipAdapterPath]. */
        var clipVisionPath: String? = null
        /** IP-Adapter (SD 1.5 / SDXL, classic or Plus); enables [GenerateParams.ipAdapterRgb]. */
        var ipAdapterPath: String? = null
        /** ControlNet; enables [GenerateParams.controlRgb]. */
        var controlNetPath: String? = null
        /** AnimateDiff motion module (SD 1.5); enables [generateVideo]. */
        var motionModulePath: String? = null
        /** PhotoMaker (SDXL). */
        var photoMakerPath: String? = null
    }

    /** Called on the generating thread once per progress unit (sampling step, VAE tile, or
     *  lazily loaded tensor batch). Throwing cancels the generation; the exception is rethrown
     *  from [generate]. */
    fun interface ProgressListener {
        fun onProgress(step: Int, steps: Int, secondsForStep: Float)
    }

    /** Java-friendly mutable parameters for [generate]. */
    class GenerateParams {
        var prompt: String = ""
        var negativePrompt: String = ""
        /** Multiples of 8. */
        var width: Int = 512
        var height: Int = 512
        var steps: Int = 20
        /** <= 0: model default (7.0). Guidance-distilled few-step models need ~1.0. */
        var cfgScale: Float = 0f
        /** <= 0: model default. Flow models only. */
        var flowShift: Float = 0f
        /** stable-diffusion.cpp sampler name; null: model default. Unknown: AmudaException. */
        var sampler: String? = null
        /** stable-diffusion.cpp scheduler name; null: model default. Unknown: AmudaException. */
        var scheduler: String? = null
        /** -1: random. */
        var seed: Long = -1L
        /** img2img source, packed RGB8 of exactly width*height*3 bytes; null: txt2img. */
        var initRgb: ByteArray? = null
        /** img2img denoising strength in (0, 1]. */
        var strength: Float = 0.75f
        var vaeTiling: Boolean = false
        /** 1..100: GPU busy share of each progress unit; the rest is idle time. 100: no idling. */
        var dutyCyclePercent: Int = 100
        /** LoRA files (absolute paths) and multipliers, same length. Applied for this call only. */
        var loraPaths: Array<String>? = null
        var loraMultipliers: FloatArray? = null
        /** <= 0: model default; 2 skips the last CLIP layer (anime SD 1.5 checkpoints). */
        var clipSkip: Int = 0
        /** ControlNet hint, packed RGB8 width*height*3. */
        var controlRgb: ByteArray? = null
        /** <= 0: 0.9. */
        var controlStrength: Float = 0f
        /** Run Canny edge detection on [controlRgb] first (canny ControlNets). */
        var controlCanny: Boolean = false
        /** IP-Adapter reference, packed RGB8 [ipAdapterWidth]*[ipAdapterHeight]*3 (any size). */
        var ipAdapterRgb: ByteArray? = null
        var ipAdapterWidth: Int = 0
        var ipAdapterHeight: Int = 0
        /** <= 0: 1.0; 0.6-0.8 leaves more room for the text prompt. */
        var ipAdapterStrength: Float = 0f
        /** Hires fix: <= 1 off; otherwise output is (width*scale) x (height*scale), rounded down to 8. */
        var hiresScale: Float = 0f
        /** <= 0: same as [steps]. */
        var hiresSteps: Int = 0
        /** <= 0: 0.5. */
        var hiresDenoise: Float = 0f
        /** null: latent. */
        var hiresUpscaler: String? = null
    }

    private val genLock = Any()
    private val stateLock = Any()
    @Volatile private var closed = false

    /**
     * Generates one image and returns it as packed RGB8 (width*height*3). Blocks the calling
     * thread; call it off the UI thread. Throws [com.ameva.amuda.AmudaException] on failure,
     * including AMEVA_SD_ERR_CANCELLED after [cancel].
     */
    fun generate(params: GenerateParams, listener: ProgressListener?): ByteArray {
        synchronized(genLock) {
            check(!closed) { "AmevaDiffusion is closed" }
            return nativeGenerate(handle, params, listener, 0, 0)
        }
    }

    /**
     * AnimateDiff video: returns [frames] packed RGB8 frames back to back
     * (frames*width*height*3 bytes). Requires [LoadOptions.motionModulePath].
     */
    fun generateVideo(params: GenerateParams, frames: Int, fps: Int, listener: ProgressListener?): ByteArray {
        require(frames in 1..64) { "frames must be 1..64" }
        synchronized(genLock) {
            check(!closed) { "AmevaDiffusion is closed" }
            return nativeGenerate(handle, params, listener, frames, fps)
        }
    }

    /** Where each module's weights live (stable-diffusion.cpp's load report; "(RAM)" = runs on the CPU).
     *  See ameva_sd_placement() in ameva_sd_api.h. */
    val placement: String by lazy {
        synchronized(stateLock) {
            check(!closed) { "AmevaDiffusion is closed" }
            nativePlacement(handle)
        }
    }

    /** True when the diffusion model (the sampling loop) runs on the CPU. A streamed diffusion model
     *  ("STREAM diffusion=GPU; " prefix) keeps its weights in RAM but computes on the GPU. */
    val diffusionOnCpu: Boolean
        get() = !diffusionStreamed &&
            Regex("""(diffusion_model|unet)\s+[0-9.]+MB\(RAM\)""").containsMatchIn(placement)

    /** True when the diffusion weights stay in RAM and are uploaded to the GPU per graph segment. */
    val diffusionStreamed: Boolean
        get() = placement.startsWith("STREAM diffusion=GPU")

    /** Asks a running [generate] to stop; safe from any thread, no effect when idle or closed. */
    fun cancel() {
        synchronized(stateLock) {
            if (!closed) nativeCancel(handle)
        }
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
        val rgb = synchronized(genLock) {
            check(!closed) { "AmevaDiffusion is closed" }
            nativeTxt2Img(handle, prompt, width, height, steps, cfgScale, seed)
        }
        return rgbToBitmap(rgb, width, height)
    }

    /** Cancels a running generation, waits for it to return, then frees the native context.
     *  Idempotent. */
    override fun close() {
        cancel()
        synchronized(genLock) {
            synchronized(stateLock) {
                if (closed) return
                closed = true
                nativeFree(handle)
            }
        }
    }
}
