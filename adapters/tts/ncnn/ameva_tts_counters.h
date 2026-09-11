#ifndef AMEVA_TTS_COUNTERS_H
#define AMEVA_TTS_COUNTERS_H

#include <stdint.h>
#include <atomic>

namespace ameva {
namespace tts {

struct Counters {
    std::atomic<uint64_t> context_create_count{0};
    std::atomic<uint64_t> context_destroy_count{0};
    std::atomic<uint64_t> kernel_create_count{0};
    std::atomic<uint64_t> kernel_destroy_count{0};
    std::atomic<uint64_t> buffer_create_count{0};
    std::atomic<uint64_t> buffer_destroy_count{0};
    std::atomic<uint64_t> event_create_count{0};
    std::atomic<uint64_t> event_destroy_count{0};
    std::atomic<uint64_t> staging_alloc_count{0};
    std::atomic<uint64_t> staging_free_count{0};
    std::atomic<uint64_t> custom_layer_create_count{0};
    std::atomic<uint64_t> custom_layer_destroy_count{0};
    std::atomic<uint64_t> double_destroy_count{0};
    std::atomic<uint64_t> use_after_free_count{0};
    std::atomic<uint64_t> teardown_call_depth_max{0};
    std::atomic<bool>     teardown_reentrancy_required{false};
    std::atomic<uint64_t> custom_layer_exec_count{0};
    std::atomic<uint64_t> gpu_dispatch_count{0};
    std::atomic<uint64_t> gpu_failure_count{0};
    std::atomic<uint64_t> silent_fallback_count{0};
    std::atomic<uint64_t> staging_h2d_bytes{0};
    std::atomic<uint64_t> staging_d2h_bytes{0};
};

extern Counters g_counters;

void DumpCounters();
void ResetCounters();

} // namespace tts
} // namespace ameva

#endif // AMEVA_TTS_COUNTERS_H
