# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# gpu_sysfs.sh (sourced): GPU clock cap, current clock and temperature of a fleet phone for the measurement protocol
# (grade A: temperature gate and clock-cap gate). Two families:
#   Exynos Mali  /sys/kernel/gpu: gpu_max_clock (kHz), gpu_clock (MHz), gpu_tmu (degrees C, decimal)
#   Adreno       /sys/class/kgsl/kgsl-3d0: max_gpuclk (Hz), gpuclk (Hz, not readable by an app on the S20), temp
#                (milli-degrees C)
# Every function prints NA when the node is absent or unreadable; callers must treat NA as "gate not available" and
# say so in their output, not as a pass.
#   gpu_family     mali | adreno | unknown
#   gpu_cap_khz    clock cap in kHz
#   gpu_now_mhz    current clock in MHz
#   gpu_temp_c     temperature in whole degrees C
#   gpu_cl_vendor  the vendor OpenCL library an ICD file must name (for ocl-icd builds such as cl_hw_bench)
_gs_rd() { local v; v=$(cat "$1" 2>/dev/null | tr -d ' \n') && [ -n "$v" ] && echo "$v" || echo NA; }
gpu_family() {
    # kgsl first: Samsung's Snapdragon kernels also have /sys/kernel/gpu, with gpu_max_clock in MHz (S20: 587)
    if [ -r /sys/class/kgsl/kgsl-3d0/max_gpuclk ]; then echo adreno
    elif [ -r /sys/kernel/gpu/gpu_max_clock ]; then echo mali
    else echo unknown; fi
}
gpu_cap_khz() {
    case $(gpu_family) in
        mali) _gs_rd /sys/kernel/gpu/gpu_max_clock ;;
        adreno) local v; v=$(_gs_rd /sys/class/kgsl/kgsl-3d0/max_gpuclk); [ "$v" = NA ] && echo NA || echo $((v / 1000)) ;;
        *) echo NA ;;
    esac
}
gpu_now_mhz() {
    case $(gpu_family) in
        mali) _gs_rd /sys/kernel/gpu/gpu_clock ;;
        # gpuclk (Hz) is not readable by an app on the S20; clock_mhz is
        adreno) local v; v=$(_gs_rd /sys/class/kgsl/kgsl-3d0/gpuclk)
                if [ "$v" != NA ]; then echo $((v / 1000000)); else _gs_rd /sys/class/kgsl/kgsl-3d0/clock_mhz; fi ;;
        *) echo NA ;;
    esac
}
gpu_temp_c() {
    local v
    case $(gpu_family) in
        mali) v=$(_gs_rd /sys/kernel/gpu/gpu_tmu); [ "$v" = NA ] && echo NA || echo "${v%%.*}" ;;
        adreno) v=$(_gs_rd /sys/class/kgsl/kgsl-3d0/temp); [ "$v" = NA ] && echo NA || echo $((v / 1000)) ;;
        *) echo NA ;;
    esac
}
gpu_cl_vendor() {
    local f
    for f in /vendor/lib64/egl/libGLES_mali.so /vendor/lib64/libOpenCL.so /system/vendor/lib64/libOpenCL.so; do
        [ -e "$f" ] && { echo "$f"; return; }
    done
    echo NA
}
# gpu_gate_wait TARGET_KHZ TMU_MAX COOL_MAX: wait (5 s steps, at most COOL_MAX s) until the cap is TARGET_KHZ and the
# temperature is at most TMU_MAX; prints the seconds waited. A gate whose node reads NA does not hold the wait.
gpu_gate_wait() {
    local target=$1 tmax=$2 cmax=$3 w=0 t c
    while [ "$w" -lt "$cmax" ]; do
        t=$(gpu_temp_c); c=$(gpu_cap_khz)
        { [ "$t" = NA ] || [ "$t" -le "$tmax" ]; } && { [ "$c" = NA ] || [ "$c" = "$target" ]; } && break
        sleep 5; w=$((w + 5))
    done
    echo "$w"
}
# gpu_clock_watch_start FILE / gpu_clock_watch_stop: sample the current GPU clock (MHz) every 0.5 s into FILE while a
# run is timed (a sampler on the phone, killed by stop; not a wait loop); gpu_clock_watch_summary FILE prints
# "clk_med=<MHz> clk_min=<MHz> clk_n=<samples>" (NA when the node is unreadable)
gpu_clock_watch_start() {
    : > "$1"
    ( while :; do gpu_now_mhz >> "$1"; sleep 0.5; done ) > /dev/null 2>&1 &
    _GS_WATCH=$!
}
gpu_clock_watch_stop() { [ -n "${_GS_WATCH:-}" ] && kill "$_GS_WATCH" 2> /dev/null; wait "$_GS_WATCH" 2> /dev/null; _GS_WATCH=; }
gpu_clock_watch_summary() {
    grep -E "^[0-9]+$" "$1" | sort -n | awk '{a[NR] = $1} END {if (NR) printf "clk_med=%d clk_min=%d clk_n=%d\n", a[int((NR + 1) / 2)], a[1], NR; else print "clk_med=NA clk_min=NA clk_n=0"}'
}
