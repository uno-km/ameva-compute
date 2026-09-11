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
    g_counters.buffer_create_count = 0;
    g_counters.buffer_destroy_count = 0;
    g_counters.event_create_count = 0;
    g_counters.event_destroy_count = 0;
    g_counters.staging_alloc_count = 0;
    g_counters.staging_free_count = 0;
    g_counters.custom_layer_create_count = 0;
    g_counters.custom_layer_destroy_count = 0;
    g_counters.double_destroy_count = 0;
    g_counters.use_after_free_count = 0;
    g_counters.teardown_call_depth_max = 0;
    g_counters.teardown_reentrancy_required = false;
    g_counters.custom_layer_exec_count = 0;
    g_counters.gpu_dispatch_count = 0;
    g_counters.gpu_failure_count = 0;
    g_counters.silent_fallback_count = 0;
    g_counters.staging_h2d_bytes = 0;
    g_counters.staging_d2h_bytes = 0;
}

void DumpCounters() {
    bool ctx_balanced = (g_counters.context_create_count == g_counters.context_destroy_count);
    bool kern_balanced = (g_counters.kernel_create_count == g_counters.kernel_destroy_count);
    bool buf_balanced = (g_counters.buffer_create_count == g_counters.buffer_destroy_count);
    bool ev_balanced = (g_counters.event_create_count == g_counters.event_destroy_count);
    bool stg_balanced = (g_counters.staging_alloc_count == g_counters.staging_free_count);
    bool layer_balanced = (g_counters.custom_layer_create_count == g_counters.custom_layer_destroy_count);
    bool full_balanced = ctx_balanced && kern_balanced && buf_balanced && ev_balanced && stg_balanced && layer_balanced;

    fprintf(stderr, "======================================================================\n");
    fprintf(stderr, "[AMEVA-TTS-METRICS] HARDWARE & RUNTIME EXECUTION COUNTERS:\n");
    fprintf(stderr, "  AMEVA_CONTEXT_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.context_create_count.load());
    fprintf(stderr, "  AMEVA_CONTEXT_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.context_destroy_count.load());
    fprintf(stderr, "  AMEVA_KERNEL_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.kernel_create_count.load());
    fprintf(stderr, "  AMEVA_KERNEL_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.kernel_destroy_count.load());
    fprintf(stderr, "  AMEVA_BUFFER_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.buffer_create_count.load());
    fprintf(stderr, "  AMEVA_BUFFER_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.buffer_destroy_count.load());
    fprintf(stderr, "  EVENT_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.event_create_count.load());
    fprintf(stderr, "  EVENT_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.event_destroy_count.load());
    fprintf(stderr, "  STAGING_ALLOC_COUNT=%llu\n", (unsigned long long)g_counters.staging_alloc_count.load());
    fprintf(stderr, "  STAGING_FREE_COUNT=%llu\n", (unsigned long long)g_counters.staging_free_count.load());
    fprintf(stderr, "  CUSTOM_LAYER_CREATE_COUNT=%llu\n", (unsigned long long)g_counters.custom_layer_create_count.load());
    fprintf(stderr, "  CUSTOM_LAYER_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.custom_layer_destroy_count.load());
    fprintf(stderr, "  DOUBLE_DESTROY_COUNT=%llu\n", (unsigned long long)g_counters.double_destroy_count.load());
    fprintf(stderr, "  USE_AFTER_FREE_COUNT=%llu\n", (unsigned long long)g_counters.use_after_free_count.load());
    fprintf(stderr, "  TEARDOWN_REENTRANCY_REQUIRED=%s\n", g_counters.teardown_reentrancy_required.load() ? "true" : "false");
    fprintf(stderr, "  TEARDOWN_CALL_DEPTH_MAX=%llu\n", (unsigned long long)g_counters.teardown_call_depth_max.load());
    fprintf(stderr, "  CUSTOM_LAYER_EXECUTION_COUNT=%llu\n", (unsigned long long)g_counters.custom_layer_exec_count.load());
    fprintf(stderr, "  AMEVA_GPU_DISPATCH_COUNT=%llu\n", (unsigned long long)g_counters.gpu_dispatch_count.load());
    fprintf(stderr, "  AMEVA_GPU_FAILURE_COUNT=%llu\n", (unsigned long long)g_counters.gpu_failure_count.load());
    fprintf(stderr, "  AMEVA_GPU_OP_COUNT=%llu\n", (unsigned long long)g_counters.gpu_dispatch_count.load());
    fprintf(stderr, "  SILENT_FALLBACK_COUNT=%llu\n", (unsigned long long)g_counters.silent_fallback_count.load());
    fprintf(stderr, "  MEASURED_STAGING_H2D_BYTES=%llu\n", (unsigned long long)g_counters.staging_h2d_bytes.load());
    fprintf(stderr, "  MEASURED_STAGING_D2H_BYTES=%llu\n", (unsigned long long)g_counters.staging_d2h_bytes.load());
    fprintf(stderr, "  CONTEXT_LIFECYCLE_BALANCED=%s\n", ctx_balanced ? "true" : "false");
    fprintf(stderr, "  KERNEL_LIFECYCLE_BALANCED=%s\n", kern_balanced ? "true" : "false");
    fprintf(stderr, "  BUFFER_LIFECYCLE_BALANCED=%s\n", buf_balanced ? "true" : "false");
    fprintf(stderr, "  FULL_RESOURCE_LIFECYCLE_BALANCED=%s\n", full_balanced ? "true" : "false");
    fprintf(stderr, "======================================================================\n");
}

} // namespace tts
} // namespace ameva
