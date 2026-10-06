/**
 * @file ac_security_guard.c
 * @brief Host Environment Verification and Anti-Hijacking Implementation
 */

#include "ac_security_guard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ANDROID__)
#include <unistd.h>
#include <sys/types.h>

static int is_authorized_package(const char* name) {
    if (!name || name[0] == '\0') {
        return 1; /* Empty cmdline fallback for kernel threads */
    }

    /* Termux core and child binaries */
    if (strstr(name, "com.termux") != NULL) {
        return 1;
    }

    /* Official AMEVA ecosystem applications */
    if (strncmp(name, "com.uno_km.", 11) == 0 ||
        strncmp(name, "com.ameva.", 10) == 0) {
        return 1;
    }

    /* Standard POSIX CLI binaries running within Termux or native shell */
    if (strchr(name, '/') != NULL) {
        /* If absolute or relative path, ensure it does not originate from an unauthorized app sandbox */
        if (strstr(name, "/data/data/") != NULL || strstr(name, "/data/user/") != NULL) {
            if (strstr(name, "com.termux") == NULL &&
                strstr(name, "com.uno_km") == NULL &&
                strstr(name, "com.ameva") == NULL) {
                return 0; /* Unauthorized third-party app data sandbox */
            }
        }
        return 1;
    }

    /* Standalone CLI binary names (e.g. python, python3, ac_cl_kat, llama-cli, sh, bash) */
    if (strcmp(name, "python") == 0 ||
        strcmp(name, "python3") == 0 ||
        strcmp(name, "llama-cli") == 0 ||
        strcmp(name, "llama-server") == 0 ||
        strcmp(name, "sd-cli") == 0 ||
        strcmp(name, "whisper-cli") == 0 ||
        strcmp(name, "sh") == 0 ||
        strcmp(name, "bash") == 0 ||
        strncmp(name, "ac_", 3) == 0 ||
        strncmp(name, "test_", 5) == 0) {
        return 1;
    }

    /*
     * If the process name follows Android Java package notation (e.g. "com.thirdparty.app")
     * without a path slash, it is an Android Application forked by Zygote.
     * Reject all unauthorized third-party application IDs.
     */
    if (strchr(name, '.') != NULL) {
        return 0; /* Unauthorized third-party APK process */
    }

    return 1;
}

AcResult ac_security_verify_host_environment(void) {
    char cmdline[512] = {0};
    FILE* fp = fopen("/proc/self/cmdline", "rb");
    if (!fp) {
        return AC_SUCCESS; /* If /proc is restricted, allow standard POSIX fallback */
    }

    size_t n = fread(cmdline, 1, sizeof(cmdline) - 1, fp);
    fclose(fp);

    if (n > 0) {
        cmdline[n] = '\0';
        if (!is_authorized_package(cmdline)) {
            /* Unauthorized APK package execution detected */
            fprintf(stderr,
                "\n================================================================================\n"
                "[SECURITY ALERT: AMEVA-SEC-001] Unauthorized Host Application Context!\n"
                "Process '%s' is not authorized to bind or execute AMEVA proprietary runtime.\n"
                "Commercial or third-party embedding is strictly prohibited under AMEVA EULA.\n"
                "Execution terminated to protect proprietary hardware compute assets.\n"
                "================================================================================\n",
                cmdline
            );
            return AC_ERR_BACKEND_UNAVAILABLE;
        }
    }

    return AC_SUCCESS;
}

#else

AcResult ac_security_verify_host_environment(void) {
    return AC_SUCCESS;
}

#endif
