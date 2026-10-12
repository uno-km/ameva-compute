package com.ameva.amuda

/**
 * Kotlin wrapper over ac_runtime.h (AMUDA 2.3 opaque C runtime ABI).
 *
 * v0.1 scope: context/buffer/kernel lifecycle and one blocking dispatch call -- the minimum
 * needed to prove a real GPU kernel runs and produces correct output from an app process.
 * Nothing here talks to a model or does LLM inference; that is a separate, later layer.
 */
class AmudaContext private constructor(private val handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("ameva_jni")
        }

        /** [backendKind]: one of [AmudaBackend]. Throws [AmudaException] if the backend is unavailable
         *  on this device, or (per ac_security_guard.c) this process is not an authorized host. */
        fun create(backendKind: Int): AmudaContext = AmudaContext(nativeContextCreate(backendKind))

        @JvmStatic private external fun nativeContextCreate(backendKind: Int): Long
        @JvmStatic private external fun nativeContextDestroy(handle: Long)
        @JvmStatic private external fun nativeGetDeviceName(handle: Long): String
        @JvmStatic private external fun nativeGetBackendKind(handle: Long): Int

        @JvmStatic external fun nativeBufferCreate(ctxHandle: Long, sizeBytes: Long, usage: Int): Long
        @JvmStatic external fun nativeBufferDestroy(handle: Long)
        @JvmStatic external fun nativeBufferWrite(handle: Long, offset: Long, data: ByteArray)
        @JvmStatic external fun nativeBufferRead(handle: Long, offset: Long, outData: ByteArray)

        @JvmStatic external fun nativeKernelCreateSpirv(
            ctxHandle: Long, spirv: ByteArray, kernelName: String, entryPoint: String,
            bindingTypes: IntArray, pushConstantSize: Long, localX: Int, localY: Int, localZ: Int,
            expectedSha256Hex: String?
        ): Long

        @JvmStatic external fun nativeKernelCreateSource(
            ctxHandle: Long, source: String, kernelName: String, entryPoint: String,
            bindingTypes: IntArray, pushConstantSize: Long, localX: Int, localY: Int, localZ: Int
        ): Long

        @JvmStatic external fun nativeKernelDestroy(handle: Long)

        @JvmStatic external fun nativeDispatchBlocking(
            ctxHandle: Long, kernelHandle: Long, bufferHandles: LongArray, pushConstants: ByteArray?,
            gridX: Int, gridY: Int, gridZ: Int
        )
    }

    val deviceName: String get() = nativeGetDeviceName(handle)
    val backendKind: Int get() = nativeGetBackendKind(handle)

    fun allocBuffer(sizeBytes: Long, usage: Int = AmudaBufferUsage.STORAGE): AmudaBuffer =
        AmudaBuffer(nativeBufferCreate(handle, sizeBytes, usage))

    /** Vulkan contexts take SPIR-V bytecode; see [createKernelFromSource] for OpenCL contexts. */
    fun createKernelFromSpirv(
        spirv: ByteArray, kernelName: String, entryPoint: String, bindingTypes: IntArray,
        pushConstantSize: Long, localSize: Triple<Int, Int, Int>, expectedSha256Hex: String? = null
    ): AmudaKernel = AmudaKernel(
        nativeKernelCreateSpirv(
            handle, spirv, kernelName, entryPoint, bindingTypes, pushConstantSize,
            localSize.first, localSize.second, localSize.third, expectedSha256Hex
        )
    )

    /** OpenCL contexts take OpenCL C source text; see [createKernelFromSpirv] for Vulkan contexts. */
    fun createKernelFromSource(
        source: String, kernelName: String, entryPoint: String, bindingTypes: IntArray,
        pushConstantSize: Long, localSize: Triple<Int, Int, Int>
    ): AmudaKernel = AmudaKernel(
        nativeKernelCreateSource(
            handle, source, kernelName, entryPoint, bindingTypes, pushConstantSize,
            localSize.first, localSize.second, localSize.third
        )
    )

    /** Blocks until the kernel has finished executing (out_event == NULL in ac_runtime_dispatch). */
    fun dispatchBlocking(
        kernel: AmudaKernel, buffers: List<AmudaBuffer>, pushConstants: ByteArray?,
        gridDim: Triple<Int, Int, Int>
    ) {
        nativeDispatchBlocking(
            handle, kernel.handle, buffers.map { it.handle }.toLongArray(), pushConstants,
            gridDim.first, gridDim.second, gridDim.third
        )
    }

    override fun close() {
        nativeContextDestroy(handle)
    }
}

class AmudaBuffer internal constructor(internal val handle: Long) : AutoCloseable {
    fun write(data: ByteArray, offset: Long = 0) =
        AmudaContext.nativeBufferWrite(handle, offset, data)

    fun read(outData: ByteArray, offset: Long = 0) =
        AmudaContext.nativeBufferRead(handle, offset, outData)

    override fun close() = AmudaContext.nativeBufferDestroy(handle)
}

class AmudaKernel internal constructor(internal val handle: Long) : AutoCloseable {
    override fun close() = AmudaContext.nativeKernelDestroy(handle)
}
