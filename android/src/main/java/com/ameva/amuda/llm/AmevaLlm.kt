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

    /**
     * How [load] sizes the GPU (ggml-ameva) layer offload against currently free device memory.
     * [DEFAULT] is the only mode that reproduces the previous fixed behavior (every layer
     * offloaded unconditionally) -- this is opt-in, never automatic: picking any other mode is a
     * deliberate caller decision, not something [load] decides for you. On a memory-constrained
     * device, [DEFAULT] with a model too large for free device memory can OOM-kill the whole
     * process (observed: 7 GiB-RAM device, ~5.7 GiB model, GPU backend).
     *
     * Values match [AmevaGpuBudgetMode] in ameva_llm_api.h by declaration order -- do not reorder
     * without updating the native enum and ameva_llm_jni.cpp's nativeLoad binding together.
     */
    enum class GpuBudgetMode {
        /** No budget-fit: offload every layer, exactly like before this option existed. */
        DEFAULT,
        /** Fit every layer within the budget, or fail with [AmudaException] -- never spills
         *  layers to CPU silently or otherwise. */
        GPU_ONLY,
        /** Offload as many layers as the budget allows; the rest run on CPU. Always logged
         *  (logcat tag "ameva_llm") when fewer than every layer fits -- never silent. */
        GPU_THEN_CPU,
        /** 0 layers offloaded, even though the GPU backend plugin stays loaded. */
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
         *
         * [backend] one of "cpu", "vulkan", "opencl", "auto" (default). "auto" benchmarks all
         * three with a short, fixed-size GEMV (ameva_backend_bench.h) and loads whichever is
         * actually fastest on this device -- not a fixed cpu-then-gpu rule. The benchmark adds a
         * few hundred ms to this call; pass an explicit backend to skip it if you already know
         * which one you want.
         *
         * [gpuBudgetOptions] null (default): no budget-fit, same fixed behavior as before this
         * option existed. See [GpuBudgetOptions] to opt into fitting the GPU offload against free
         * device memory instead -- recommended for any model whose size is not known in advance
         * to comfortably fit the target device's RAM.
         */
        fun load(
            context: Context, modelPath: String, nCtx: Int = 2048, cacheDir: String? = null,
            backend: String = "auto", gpuBudgetOptions: GpuBudgetOptions? = null
        ): AmevaLlm {
            val backendPluginPath = File(context.applicationInfo.nativeLibraryDir, "libggml-ameva.so").path
            val resolvedCacheDir = cacheDir ?: context.filesDir.path
            val mode = gpuBudgetOptions?.mode ?: GpuBudgetMode.DEFAULT
            val fraction = gpuBudgetOptions?.budgetFraction ?: 0.95f
            return AmevaLlm(nativeLoad(modelPath, backendPluginPath, resolvedCacheDir, backend, nCtx,
                mode.ordinal, fraction))
        }

        @JvmStatic private external fun nativeLoad(
            modelPath: String, backendPluginPath: String, cacheDir: String, backend: String, nCtx: Int,
            gpuBudgetMode: Int, gpuBudgetFraction: Float
        ): Long
        @JvmStatic private external fun nativeGenerate(
            handle: Long, prompt: String, maxTokens: Int, temperature: Float, seed: Long, outBufSize: Int
        ): String
        @JvmStatic private external fun nativeReset(handle: Long)
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeLoadMmproj(handle: Long, mmprojPath: String)
        @JvmStatic private external fun nativeSupportsVision(handle: Long): Boolean
        @JvmStatic private external fun nativeGenerateWithImage(
            handle: Long, imagePath: String, prompt: String, maxTokens: Int, temperature: Float,
            seed: Long, outBufSize: Int
        ): String
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

    /**
     * Attaches a CLIP/vision projector (mmproj GGUF) to this already-loaded text model, enabling
     * [generateWithImage]. The mmproj file must match the loaded text model's family -- e.g.
     * mmproj-Qwen2-VL-2B-Instruct-f16.gguf for a Qwen2-VL-2B-Instruct text model; a mismatched
     * pair throws [AmudaException] rather than silently producing garbage output.
     *
     * The vision encoder runs on the same backend [load] chose (GPU or CPU) -- there is no
     * separate backend selection for it.
     *
     * Call at most once per instance; to switch mmproj files, [close] and [load] again.
     */
    fun loadVision(mmprojPath: String) {
        nativeLoadMmproj(handle, mmprojPath)
    }

    /** Whether [loadVision] has successfully attached a vision encoder. */
    fun supportsVision(): Boolean = nativeSupportsVision(handle)

    /**
     * Encodes one image and generates text conditioned on it and [prompt]. Requires a prior
     * [loadVision] call -- throws [AmudaException] otherwise, never a silent text-only fallback.
     *
     * [prompt] may contain mtmd's media marker ("<__media__>") to place the image explicitly;
     * otherwise the marker is prepended automatically, so a plain "describe this image" prompt
     * works unmodified. [imagePath] must point to a JPEG/PNG/BMP/GIF/static-WebP file.
     */
    fun generateWithImage(
        imagePath: String, prompt: String, maxTokens: Int = 128, temperature: Float = 0.8f,
        seed: Long = 42L, outBufSize: Int = 4096
    ): String = nativeGenerateWithImage(handle, imagePath, prompt, maxTokens, temperature, seed,
        outBufSize)

    override fun close() {
        nativeFree(handle)
    }
}
