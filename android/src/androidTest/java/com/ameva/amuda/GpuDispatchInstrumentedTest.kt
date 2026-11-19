package com.ameva.amuda

import androidx.test.platform.app.InstrumentationRegistry
import android.util.Log
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * v0.1 real-device proof: a real GPU kernel (vector_add, reused from src/shaders/vector_add.spv /
 * its OpenCL-C equivalent) runs from an ordinary instrumented-test app process -- not Termux --
 * and produces the correct result on each backend family.
 *
 * Per CLAUDE.md ground truth rules, a clean process exit is not evidence; every assertion below
 * checks the actual numeric output against a CPU-computed expected value. Results are also logged
 * to Logcat under tag "AmevaAARPoC" so `adb logcat -d | grep AmevaAARPoC` gives a durable record
 * independent of the JUnit XML.
 */
class GpuDispatchInstrumentedTest {

    companion object {
        private const val TAG = "AmevaAARPoC"
        private const val N = 256
        private const val ALPHA = 2.5f

        // Reused from the existing real-hardware fixture src/shaders/vector_add.spv /
        // src/shaders/vector_add_spv.h -- not re-authored for this test.
        private const val VECTOR_ADD_SHA256 =
            "75019395c3855e1ebe412d252ecc429403cf5b2d2238af0fd35e4098bb139f4e"

        private val OPENCL_SOURCE = """
            typedef struct { uint n_elements; float alpha; } PushConsts;
            __kernel void vector_add(__global const float* in_a, __global const float* in_b,
                                      __global float* out_c, PushConsts pc) {
                uint gid = get_global_id(0);
                if (gid < pc.n_elements) {
                    out_c[gid] = in_a[gid] + pc.alpha * in_b[gid];
                }
            }
        """.trimIndent()
    }

    private fun loadVectorAddSpirv(): ByteArray {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return ctx.assets.open("vector_add.spv").use { it.readBytes() }
    }

    private fun pushConstants(nElements: Int, alpha: Float): ByteArray =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(nElements).putFloat(alpha).array()

    private fun floatsToBytes(values: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buf.putFloat(it) }
        return buf.array()
    }

    private fun bytesToFloats(bytes: ByteArray, count: Int): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(count) { buf.getFloat(it * 4) }
    }

    private fun runVectorAdd(ctx: AmudaContext, kernel: AmudaKernel): FloatArray {
        val fa = FloatArray(N) { i -> i * 0.5f }
        val fb = FloatArray(N) { i -> (N - i) * 0.25f }

        val bufA = ctx.allocBuffer((N * 4).toLong())
        val bufB = ctx.allocBuffer((N * 4).toLong())
        val bufC = ctx.allocBuffer((N * 4).toLong())
        try {
            bufA.write(floatsToBytes(fa))
            bufB.write(floatsToBytes(fb))

            ctx.dispatchBlocking(
                kernel, listOf(bufA, bufB, bufC), pushConstants(N, ALPHA),
                Triple((N + 255) / 256, 1, 1)
            )

            val outBytes = ByteArray(N * 4)
            bufC.read(outBytes)
            return bytesToFloats(outBytes, N)
        } finally {
            bufA.close(); bufB.close(); bufC.close()
        }
    }

    private fun assertVectorAddCorrect(result: FloatArray, backendLabel: String) {
        var maxErr = 0f
        for (i in 0 until N) {
            val expected = (i * 0.5f) + ALPHA * ((N - i) * 0.25f)
            maxErr = maxOf(maxErr, abs(result[i] - expected))
        }
        Log.i(TAG, "$backendLabel vector_add max_err=$maxErr")
        if (maxErr > 1e-3f) {
            throw AssertionError("$backendLabel vector_add output incorrect, max_err=$maxErr")
        }
    }

    @Test
    fun vulkanVectorAddProducesCorrectOutput() {
        val ctx = AmudaContext.create(AmudaBackend.VULKAN)
        try {
            Log.i(TAG, "Vulkan context created: device=${ctx.deviceName} backendKind=${ctx.backendKind}")
            val kernel = ctx.createKernelFromSpirv(
                spirv = loadVectorAddSpirv(),
                kernelName = "vector_add",
                entryPoint = "main",
                bindingTypes = intArrayOf(
                    AmudaBindingType.STORAGE_BUFFER,
                    AmudaBindingType.STORAGE_BUFFER,
                    AmudaBindingType.STORAGE_BUFFER
                ),
                pushConstantSize = 8L,
                localSize = Triple(256, 1, 1),
                expectedSha256Hex = VECTOR_ADD_SHA256
            )
            try {
                val result = runVectorAdd(ctx, kernel)
                assertVectorAddCorrect(result, "Vulkan[${ctx.deviceName}]")
            } finally {
                kernel.close()
            }
        } catch (e: AmudaException) {
            Log.e(TAG, "Vulkan path failed: code=${e.code} msg=${e.message}")
            throw e
        } finally {
            ctx.close()
        }
    }

    @Test
    fun openClVectorAddProducesCorrectOutput() {
        val ctx = AmudaContext.create(AmudaBackend.OPENCL)
        try {
            Log.i(TAG, "OpenCL context created: device=${ctx.deviceName} backendKind=${ctx.backendKind}")
            val kernel = ctx.createKernelFromSource(
                source = OPENCL_SOURCE,
                kernelName = "vector_add",
                entryPoint = "vector_add",
                bindingTypes = intArrayOf(
                    AmudaBindingType.STORAGE_BUFFER,
                    AmudaBindingType.STORAGE_BUFFER,
                    AmudaBindingType.STORAGE_BUFFER
                ),
                pushConstantSize = 8L,
                localSize = Triple(256, 1, 1)
            )
            try {
                val result = runVectorAdd(ctx, kernel)
                assertVectorAddCorrect(result, "OpenCL[${ctx.deviceName}]")
            } finally {
                kernel.close()
            }
        } catch (e: AmudaException) {
            // This is the actual open question v0.1 exists to answer (android_load_sphal_library
            // from a normal app process) -- log and rethrow, do not swallow or soften it.
            Log.e(TAG, "OpenCL path failed: code=${e.code} msg=${e.message}")
            throw e
        } finally {
            ctx.close()
        }
    }
}
