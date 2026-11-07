/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
// hwc_sample SECONDS [PERIOD_MS]: Mali hardware counters (Arm HWCPipe, libGPUCounters, MIT) summed over a window, for
// the decode research (ameva-foundation N-20261107-decode-roofline): does the GPU's external bus read what the roofline
// says, how long do the reads wait, how busy are the load/store units, how many reads are partial.
// Samples every PERIOD_MS (default 50) for SECONDS; cycle and byte counters are deltas per sample and are summed,
// ratio counters are averaged over the samples in which the GPU was active. A counter that this GPU does not have is
// printed as HWC_SKIP, never as zero. Run it next to the workload (start it first, then the workload).
// Build (Termux, from an HWCPipe checkout built with CMake into build/):
//   clang++ -O2 -std=c++17 -I HWCPipe/hwcpipe/include -I HWCPipe/backend/device/include hwc_sample.cpp \
//       HWCPipe/build/hwcpipe/libhwcpipe.a HWCPipe/build/backend/device/libdevice.a -o hwc_sample
#include <hwcpipe/counter_database.hpp>
#include <hwcpipe/gpu.hpp>
#include <hwcpipe/sampler.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

struct Want {
    hwcpipe_counter id;
    const char* name;
};

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? atof(argv[1]) : 10.0;
    const int period_ms = argc > 2 ? atoi(argv[2]) : 50;
    hwcpipe::gpu gpu(0);
    if (!gpu) {
        fprintf(stderr, "HWC_FAIL no Mali GPU at /dev/mali0\n");
        return 1;
    }
    printf("HWC_DEVICE cores=%u bus_width=%u\n", (unsigned)gpu.num_shader_cores(), (unsigned)gpu.bus_width());
    const Want want[] = {
        {MaliGPUActiveCy, "gpu_active_cy"},       {MaliCoreActiveCy, "core_active_cy"},
        {MaliExtBusRdBy, "ext_rd_bytes"},         {MaliExtBusRdBt, "ext_rd_beats"},
        {MaliExtBusRd, "ext_rd_tx"},              {MaliExtBusRdStallCy, "ext_rd_stall_cy"},
        {MaliExtBusRdLat0, "ext_rd_lat0"},        {MaliExtBusRdLat128, "ext_rd_lat128"},
        {MaliExtBusRdLat192, "ext_rd_lat192"},    {MaliExtBusRdLat256, "ext_rd_lat256"},
        {MaliExtBusRdLat320, "ext_rd_lat320"},    {MaliExtBusRdLat384, "ext_rd_lat384"},
        {MaliExtBusRdOTQ1, "ext_rd_ot_q1"},       {MaliExtBusRdOTQ2, "ext_rd_ot_q2"},
        {MaliExtBusRdOTQ3, "ext_rd_ot_q3"},       {MaliExtBusRdOTQ4, "ext_rd_ot_q4"},
        {MaliExtBusWrBy, "ext_wr_bytes"},         {MaliL2CacheRd, "l2_rd"},
        {MaliL2CacheRdLookup, "l2_rd_lookup"},    {MaliL2CacheL1Rd, "l2_l1_rd"},
        {MaliL2CacheRdStallCy, "l2_rd_stall_cy"}, {MaliLSFullRd, "ls_full_rd"},
        {MaliLSPartRd, "ls_part_rd"},             {MaliLSIssueCy, "ls_issue_cy"},
        {MaliLSRdCy, "ls_rd_cy"},                 {MaliEngInstr, "eng_instr"},
    };
    auto config = hwcpipe::sampler_config(gpu);
    std::vector<Want> used;
    for (const Want& w : want) {
        if (config.add_counter(w.id)) printf("HWC_SKIP %s (not on this GPU)\n", w.name);
        else used.push_back(w);
    }
    auto sampler = hwcpipe::sampler<>(config);
    if (sampler.start_sampling()) {
        fprintf(stderr, "HWC_FAIL start_sampling\n");
        return 1;
    }
    std::vector<double> sum(used.size(), 0.0);
    const auto t0 = std::chrono::steady_clock::now();
    int samples = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
        if (sampler.sample_now()) {
            fprintf(stderr, "HWC_FAIL sample_now\n");
            return 1;
        }
        ++samples;
        for (size_t i = 0; i < used.size(); ++i) {
            hwcpipe::counter_sample s;
            if (sampler.get_counter_value(used[i].id, s)) continue;
            sum[i] += s.type == hwcpipe::counter_sample::type::uint64 ? (double)s.value.uint64 : s.value.float64;
        }
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sampler.stop_sampling();
    printf("HWC_WINDOW seconds=%.2f samples=%d\n", el, samples);
    for (size_t i = 0; i < used.size(); ++i) printf("HWC %s=%.0f\n", used[i].name, sum[i]);
    return 0;
}
