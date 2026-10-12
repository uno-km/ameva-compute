package com.ameva.amuda

/**
 * Thrown for every ac_runtime.h call that returns an AcResult != AC_SUCCESS.
 * [code] is the raw AcResult value (see ac_base.h) -- never swallowed, never mapped to a default.
 */
class AmudaException(val code: Int, message: String) : RuntimeException(message)
