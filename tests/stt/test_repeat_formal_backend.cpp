/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * @file test_repeat_formal_backend.cpp
 * @brief Formal AMEVA ggml_backend 20-iteration repeated inference stability test
 */

#include "whisper.h"
#include "ggml-backend-ameva.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
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

static bool read_wav(const char* fname, std::vector<float>& pcmf32) {
    FILE* f = fopen(fname, "rb");
    if (!f) return false;
    fseek(f, 44, SEEK_SET); // Skip standard 44-byte WAV header
    std::vector<int16_t> pcm16(160000);
    size_t n = fread(pcm16.data(), sizeof(int16_t), 160000, f);
    fclose(f);
    pcmf32.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pcmf32[i] = (float)pcm16[i] / 32768.0f;
    }
    return true;
}

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

int main() {
    printf("=== [AMEVA FORMAL BACKEND 20-REPEAT STABILITY TEST] ===\n");

    const char* model_path = "/data/data/com.termux/files/home/ameva-models/stt/ggml-tiny.bin";
    const char* audio_path = "/data/data/com.termux/files/home/ameva-compute-stt/tests/fixtures/stt/stt-ko-16k.wav";

    // 1. Initialize Whisper context with formal backend registered
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // Use GPU / ACCEL backend

    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "FATAL: Failed to load whisper model from %s\n", model_path);
        return 1;
    }

    std::vector<float> pcmf32;
    if (!read_wav(audio_path, pcmf32)) {
        fprintf(stderr, "FATAL: Failed to read audio file %s\n", audio_path);
        whisper_free(ctx);
        return 2;
    }

    // 2. Warmup iterations (2 iterations) to stabilize allocations & caches
    const int WARMUP_ITERATIONS = 2;
    printf("WARMUP_ITERATIONS=%d\n", WARMUP_ITERATIONS);
    for (int w = 0; w < WARMUP_ITERATIONS; ++w) {
        whisper_full_params wparams_warmup = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams_warmup.print_realtime = false;
        wparams_warmup.print_progress = false;
        wparams_warmup.print_timestamps = false;
        wparams_warmup.print_special = false;
        wparams_warmup.language = "ko";
        wparams_warmup.n_threads = 4;
        whisper_full(ctx, wparams_warmup, pcmf32.data(), pcmf32.size());
    }

    // 3. Dynamic Baseline Measurement: Run 1 inference and measure dispatches
    ameva_backend_reset_counters();
    std::string baseline_text = "";
    {
        whisper_full_params wparams_base = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams_base.print_realtime = false;
        wparams_base.print_progress = false;
        wparams_base.print_timestamps = false;
        wparams_base.print_special = false;
        wparams_base.language = "ko";
        wparams_base.n_threads = 4;

        int ret = whisper_full(ctx, wparams_base, pcmf32.data(), pcmf32.size());
        if (ret != 0) {
            fprintf(stderr, "FATAL: Baseline inference failed with ret %d\n", ret);
            whisper_free(ctx);
            return 3;
        }

        const int n_segments = whisper_full_n_segments(ctx);
        for (int s = 0; s < n_segments; ++s) {
            baseline_text += whisper_full_get_segment_text(ctx, s);
        }
        baseline_text = trim(baseline_text);
    }

    AmevaBackendCounters c_base = {};
    ameva_backend_get_counters(&c_base);
    const uint64_t baseline_dispatches_per_inference = c_base.ameva_gpu_op_count;

    printf("BASELINE_DISPATCHES_PER_INFERENCE=%llu\n", (unsigned long long)baseline_dispatches_per_inference);
    printf("BASELINE_TRANSCRIPT=\"%s\"\n", baseline_text.c_str());

    if (baseline_dispatches_per_inference == 0) {
        fprintf(stderr, "FATAL: Baseline dispatches per inference is 0! Backend not executing.\n");
        whisper_free(ctx);
        return 4;
    }

    // 4. Measured 20 Iterations
    const int MEASURED_ITERATIONS = 20;
    const uint64_t expected_gpu_ops = baseline_dispatches_per_inference * MEASURED_ITERATIONS;
    printf("MEASURED_ITERATIONS=%d\n", MEASURED_ITERATIONS);
    printf("EXPECTED_GPU_OPS=%llu\n", (unsigned long long)expected_gpu_ops);

    long rss_before_kb = get_rss_kb();
    printf("RSS_BEFORE_KB=%ld\n", rss_before_kb);

    ameva_backend_reset_counters();

    int text_exact_match_count = 0;
    int output_parity_failures = 0;

    for (int iter = 1; iter <= MEASURED_ITERATIONS; ++iter) {
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
        std::string current_text = "";
        for (int s = 0; s < n_segments; ++s) {
            current_text += whisper_full_get_segment_text(ctx, s);
        }
        current_text = trim(current_text);

        if (current_text == baseline_text) {
            text_exact_match_count++;
        } else {
            fprintf(stderr, "Parity mismatch at iter %d: \"%s\" != \"%s\"\n",
                    iter, current_text.c_str(), baseline_text.c_str());
            output_parity_failures++;
        }
    }

    long rss_after_kb = get_rss_kb();
    long rss_delta_kb = rss_after_kb - rss_before_kb;
    const long rss_limit_kb = 10240; // 10 MB maximum growth threshold over 20 runs
    bool rss_threshold_pass = (rss_delta_kb <= rss_limit_kb);

    AmevaBackendCounters c_meas = {};
    ameva_backend_get_counters(&c_meas);
    uint64_t actual_gpu_ops = c_meas.ameva_gpu_op_count;

    printf("ACTUAL_GPU_OPS=%llu\n", (unsigned long long)actual_gpu_ops);
    printf("OUTPUT_PARITY_FAILURES=%d\n", output_parity_failures);
    printf("TEXT_EXACT_MATCH=%d/%d\n", text_exact_match_count, MEASURED_ITERATIONS);
    printf("RSS_AFTER_KB=%ld\n", rss_after_kb);
    printf("RSS_DELTA_KB=%ld\n", rss_delta_kb);
    printf("RSS_LIMIT_KB=%ld\n", rss_limit_kb);
    printf("RSS_THRESHOLD_PASS=%s\n", rss_threshold_pass ? "true" : "false");
    printf("MEMORY_LEAK_PROVEN_ABSENT=false\n");

    // 5. Teardown and Lifecycle Audit AFTER 20 iterations
    whisper_free(ctx);
    ameva_backend_teardown_hardware();

    AmevaBackendCounters c_final = {};
    ameva_backend_get_counters(&c_final);

    printf("CONTEXT_CREATE_COUNT=%llu\n", (unsigned long long)c_final.context_create_count);
    printf("CONTEXT_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.context_destroy_count);
    printf("KERNEL_CREATE_COUNT=%llu\n", (unsigned long long)c_final.kernel_create_count);
    printf("KERNEL_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.kernel_destroy_count);
    printf("BUFFER_CREATE_COUNT=%llu\n", (unsigned long long)c_final.buffer_create_count);
    printf("BUFFER_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.buffer_destroy_count);
    printf("EVENT_CREATE_COUNT=%llu\n", (unsigned long long)c_final.event_create_count);
    printf("EVENT_DESTROY_COUNT=%llu\n", (unsigned long long)c_final.event_destroy_count);

    bool lifecycle_balanced = (c_final.context_create_count == c_final.context_destroy_count &&
                               c_final.kernel_create_count == c_final.kernel_destroy_count &&
                               c_final.buffer_create_count == c_final.buffer_destroy_count &&
                               c_final.event_create_count == c_final.event_destroy_count);
    printf("LIFECYCLE_BALANCED=%s\n", lifecycle_balanced ? "true" : "false");

    bool all_pass = (actual_gpu_ops == expected_gpu_ops &&
                     text_exact_match_count == MEASURED_ITERATIONS &&
                     output_parity_failures == 0 &&
                     rss_threshold_pass &&
                     lifecycle_balanced);

    printf("FORMAL_REPEAT_TEST_RESULT=%s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 5;
}
