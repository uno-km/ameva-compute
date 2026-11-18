/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
 */
/**
 * @file ac_security_guard.c
 * @brief Host Environment Verification (open-distribution policy)
 *
 * Previously rejected (AC_ERR_BACKEND_UNAVAILABLE) any host process whose package name was not
 * Termux or com.ameva.* / com.uno_km.*, to keep third-party app embedding out.
 * User decision (2026-10-12): open the AAR distribution to any app -- this check is retired, not
 * replaced, so every caller gets AC_SUCCESS unconditionally. Licensing is by terms, not by a runtime check:
 * LICENSING.md (PolyForm Noncommercial 1.0.0, commercial use under a separate license).
 */

#include "ac_security_guard.h"

AcResult ac_security_verify_host_environment(void) {
    return AC_SUCCESS;
}
