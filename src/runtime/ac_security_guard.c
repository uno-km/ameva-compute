/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file ac_security_guard.c
 * @brief Host Environment Verification (open-distribution policy)
 *
 * Previously rejected (AC_ERR_BACKEND_UNAVAILABLE) any host process whose package name was not
 * Termux or com.ameva.* / com.uno_km.*, to keep third-party app embedding out per the AMEVA EULA.
 * User decision (2026-10-12): open the AAR distribution to any app -- this check is retired, not
 * replaced, so every caller gets AC_SUCCESS unconditionally. A real licensing mechanism (if one
 * is wanted later) is a separate decision, not a revival of this allowlist.
 */

#include "ac_security_guard.h"

AcResult ac_security_verify_host_environment(void) {
    return AC_SUCCESS;
}
