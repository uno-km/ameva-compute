package com.ameva.amuda

/** Mirrors AcBackendKind in ac_base.h. */
object AmudaBackend {
    const val VULKAN = 1
    const val OPENCL = 2
}

/** Mirrors AC_BUFFER_USAGE_* in ac_runtime.h. */
object AmudaBufferUsage {
    const val STORAGE = 1 shl 0
    const val TRANSFER_SRC = 1 shl 1
    const val TRANSFER_DST = 1 shl 2
    const val UNIFORM = 1 shl 3
}

/** Mirrors AC_RUNTIME_BINDING_* in ac_runtime.h. */
object AmudaBindingType {
    const val STORAGE_BUFFER = 0
    const val UNIFORM_BUFFER = 1
}
