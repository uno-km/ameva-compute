#include "whisper.h"
#include "ggml-ameva.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <unistd.h>

static long get_rss_kb() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long size = 0, resident = 0;
    if (fscanf(f, "%ld %ld", &size, &resident) == 2) {
        fclose(f);
        return resident * (sysconf(_SC_PAGESIZE) / 1024);
    }
    fclose(f);
    return 0;
}

// Read simple mono 16kHz WAV
static bool read_wav(const char* fname, std::vector<float>& pcmf32) {
    FILE* f = fopen(fname, "rb");
    if (!f) return false;
    fseek(f, 44, SEEK_SET); // Skip standard 44-byte WAV header
    std::vector<int16_t> pcm16(112000);
    size_t n = fread(pcm16.data(), sizeof(int16_t), 112000, f);
    fclose(f);
    pcmf32.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pcmf32[i] = (float)pcm16[i] / 32768.0f;
    }
    return true;
}

int main() {
    setenv("AMEVA_OFFLOAD_ENCODER", "1", 1);

    const char* model_path = "/data/data/com.termux/files/home/ameva-models/stt/ggml-tiny.bin";
    const char* audio_path = "/data/data/com.termux/files/home/ameva-compute-stt/tests/fixtures/stt/stt-ko-16k.wav";

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // CPU context with AMEVA mixed offload
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to load whisper model\n");
        return 1;
    }

    std::vector<float> pcmf32;
    if (!read_wav(audio_path, pcmf32)) {
        fprintf(stderr, "Failed to read audio file\n");
        whisper_free(ctx);
        return 2;
    }

    const int ITERATIONS = 20;
    int output_parity_failures = 0;

    // Warm-up run to settle memory allocations
    {
        whisper_full_params wparams_warmup = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams_warmup.print_realtime = false;
        wparams_warmup.print_progress = false;
        wparams_warmup.print_timestamps = false;
        wparams_warmup.print_special = false;
        wparams_warmup.language = "ko";
        wparams_warmup.n_threads = 4;
        whisper_full(ctx, wparams_warmup, pcmf32.data(), pcmf32.size());
    }

    // Measure baseline RSS
    long rss_before_kb = get_rss_kb();

    AmevaSttStats stats_start = {};
    ameva_stt_vulkan_get_stats(&stats_start);
    uint64_t ops_start = stats_start.intercept_success_count;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime = false;
        wparams.print_progress = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.language = "ko";
        wparams.n_threads = 4;

        int ret = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
        if (ret != 0) {
            fprintf(stderr, "whisper_full failed at iter %d with ret %d\n", iter, ret);
            output_parity_failures++;
            continue;
        }

        const int n_segments = whisper_full_n_segments(ctx);
        std::string full_text = "";
        for (int i = 0; i < n_segments; ++i) {
            full_text += whisper_full_get_segment_text(ctx, i);
        }

        if (full_text.find("배치") == std::string::npos || full_text.find("쿠팡") == std::string::npos) {
            fprintf(stderr, "Parity mismatch at iter %d: %s\n", iter, full_text.c_str());
            output_parity_failures++;
        }
    }

    long rss_after_kb = get_rss_kb();
    long rss_delta_kb = rss_after_kb - rss_before_kb;

    AmevaSttStats stats = {};
    ameva_stt_vulkan_get_stats(&stats);
    uint64_t actual_ops = stats.intercept_success_count - ops_start;

    printf("=== [REPEAT STABILITY TEST RESULTS] ===\n");
    printf("ITERATIONS=%d\n", ITERATIONS);
    printf("EXPECTED_AMEVA_OPS=%d\n", ITERATIONS);
    printf("ACTUAL_AMEVA_OPS=%llu\n", (unsigned long long)actual_ops);
    printf("GRAPH_MUTATION_LEAK_COUNT=%llu\n", (unsigned long long)stats.graph_mutation_leak_count);
    printf("OUTPUT_PARITY_FAILURES=%d\n", output_parity_failures);
    printf("CONTEXT_CREATE_COUNT=%llu\n", (unsigned long long)stats.context_create_count);
    printf("KERNEL_CREATE_COUNT=%llu\n", (unsigned long long)stats.kernel_create_count);
    printf("RSS_BEFORE_KB=%ld\n", rss_before_kb);
    printf("RSS_AFTER_KB=%ld\n", rss_after_kb);
    printf("RSS_DELTA_KB=%ld\n", rss_delta_kb);

    whisper_free(ctx);
    bool pass = (actual_ops == ITERATIONS && output_parity_failures == 0 && stats.graph_mutation_leak_count == 0);
    printf("REPEAT_TEST_RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
}
