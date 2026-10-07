/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_security_guard.h
 * @brief AMEVA Compute Runtime Host Environment Verification and Anti-Tamper Guard
 *
 * Verifies that the host execution context is authorized (Termux or official AMEVA application).
 * Prevents unauthorized third-party Android APKs from leeching the proprietary runtime.
 */

#ifndef AC_SECURITY_GUARD_H
#define AC_SECURITY_GUARD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ac_runtime.h"

/**
 * @brief Verifies that the host process environment is authorized to execute AMEVA compute.
 *
 * Checks process credentials, command line identity, and sandbox isolation.
 *
 * @return AC_SUCCESS if authorized; AC_ERR_NOT_INITIALIZED or AC_ERR_BACKEND_UNAVAILABLE if unauthorized.
 */
AcResult ac_security_verify_host_environment(void);

#ifdef __cplusplus
}
#endif

#endif /* AC_SECURITY_GUARD_H */
