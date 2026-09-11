#include "ameva_tts_counters.h"
#include <stdio.h>

namespace ameva {
namespace tts {

Counters g_counters;

void ResetCounters() {
    g_counters.context_create_count = 0;
    g_counters.context_destroy_count = 0;
    g_counters.kernel_create_count = 0;
    g_counters.kernel_destroy_count = 0;
    g_counters.custom_layer_exec_count = 0;
    g_counters.gpu_dispatch_count = 0;
    g_counters.silent_fallback_count = 0;
    g_counters.staging_h2d_bytes = 0;
    g_counters.staging_d2h_bytes = 0;
}

void DumpCounters() {
    fprintf(stderr, "======================================================================\n");
    fprintf(stderr, "[AMEVA-TTS-METRICS] HARDWARE & RUNTIME EXECUTION COUNTERS:\n");
    fprintf(stderr, "  AMEVA_CONTEXT_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.context_create_count.load());
    fprintf(stderr, "  AMEVA_CONTEXT_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.context_destroy_count.load());
    fprintf(stderr, "  AMEVA_KERNEL_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.kernel_create_count.load());
    fprintf(stderr, "  AMEVA_KERNEL_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.kernel_destroy_count.load());
    fprintf(stderr, "  CUSTOM_LAYER_EXECUTION_COUNT=%llu\n", (unsigned long long)g_counters.custom_layer_exec_count.load());
    fprintf(stderr, "  AMEVA_GPU_DISPATCH_COUNT=%llu\n", (unsigned long long)g_counters.gpu_dispatch_count.load());
    fprintf(stderr, "  AMEVA_GPU_OP_COUNT=%llu\n", (unsigned long long)g_counters.gpu_dispatch_count.load());
    fprintf(stderr, "  SILENT_FALLBACK_COUNT=%llu\n", (unsigned long long)g_counters.silent_fallback_count.load());
    fprintf(stderr, "  MEASURED_STAGING_H2D_BYTES=%llu\n", (unsigned long long)g_counters.staging_h2d_bytes.load());
    fprintf(stderr, "  MEASURED_STAGING_D2H_BYTES=%llu\n", (unsigned long long)g_counters.staging_d2h_bytes.load());
    fprintf(stderr, "======================================================================\n");
}

} // namespace tts
} // namespace ameva
