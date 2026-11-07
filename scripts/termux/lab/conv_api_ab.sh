#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# conv_api_ab.sh: CONV_2D time on Vulkan against OpenCL, and against OpenCL kernel-text variants, under a fixed
# protocol (research program, ameva-foundation 03_core/docs/research/RESEARCH_PROGRAM.md):
#   - VARIANTS (space separated): "vk" (AMEVA_BACKEND=vulkan), "cl" (OpenCL, built-in kernel text) or
#     "<name>:<dir>" (OpenCL with AMEVA_CL_SOURCE_DIR=<dir>, the kernel text under test).
#     "vkspin" (Vulkan, AMEVA_VK_WAIT=spin) and "clsleep" (OpenCL, AMEVA_CL_WAIT=sleep): the other way of waiting.
#   - ROUNDS rounds (default 5); round r runs the variants rotated by r, so that each variant runs in every position
#     of the order and a drift over the session (heat) does not favour one of them.
#   - before every run: wait (at most COOL_MAX seconds, default 120) until the GPU temperature (Exynos
#     /sys/kernel/gpu/gpu_tmu) is at most TMU_MAX (default 40), and record it with the clock cap.
#   - first, once per variant, the 35 exactness cases of tests/ameva_conv_check (a kernel text that computes a wrong
#     result fails there and the variant is dropped); an override that is not announced by the backend fails too.
#   - then every timed run is one process for one shape: AMEVA_CONV_CHECK_TIME_ONLY=1 with ITERS timed iterations
#     after one warm-up (default 20) and AMEVA_PROFILE=2, so that the GPU time of the kind CONV_2D holds that shape
#     alone. Reported: wall mean and fastest per graph_compute, and GPU time per convolution (gpu_ms / (ITERS + 1)).
# Prints one line per run: AB round= pos= var= shape= mean_ms= best_ms= gpu_ms= tmu0= tmu1= cap0= cap1= cool_wait=.
# The tests come from conv_check.sh (OUT, default ~/abi-lab/out-convk).
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-convk}
ROUNDS=${ROUNDS:-5}
ITERS=${ITERS:-20}
TMU_MAX=${TMU_MAX:-40}
COOL_MAX=${COOL_MAX:-120}
SHAPES=${SHAPES:-"32,32,320,320,3 16,16,640,640,3 8,8,1280,1280,3 128,128,256,256,3"}
VARIANTS=${VARIANTS:-"vk cl"}
RT="$OUT:$OUT/ggml-rt"
TMU=/sys/kernel/gpu/gpu_tmu
CAP=/sys/kernel/gpu/gpu_max_clock
[ -x "$OUT/ameva_conv_check" ] || { echo "AB_FAIL no $OUT/ameva_conv_check (run conv_check.sh first)"; exit 1; }
rd() { cat "$1" 2>/dev/null | tr -d ' \n' || echo NA; }
echo "AB_START $(date +%F_%T) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') rounds=$ROUNDS iters=$ITERS" \
     "tmu_max=$TMU_MAX variants=\"$VARIANTS\" shapes=\"$SHAPES\" governor=$(rd /sys/kernel/gpu/gpu_governor)"
read -r -a V <<< "$VARIANTS"
read -r -a S <<< "$SHAPES"
venv() {
    case "$1" in
        vk) envs=(AMEVA_BACKEND=vulkan) ;;
        cl) envs=(AMEVA_BACKEND=opencl) ;;
        vkspin) envs=(AMEVA_BACKEND=vulkan AMEVA_VK_WAIT=spin) ;;
        clsleep) envs=(AMEVA_BACKEND=opencl AMEVA_CL_WAIT=sleep) ;;
        *:*) envs=(AMEVA_BACKEND=opencl "AMEVA_CL_SOURCE_DIR=${1#*:}") ;;
        *) echo "AB_FAIL bad variant $1"; exit 1 ;;
    esac
}
cool() {
    waited=0
    while [ -r "$TMU" ] && [ "$(rd "$TMU" | cut -d. -f1)" -gt "$TMU_MAX" ] && [ "$waited" -lt "$COOL_MAX" ]; do
        sleep 5; waited=$((waited + 5))
    done
}
OK=()
for v in "${V[@]}"; do
    venv "$v"; cool
    log="$OUT/ab_check_${v%%:*}.log"
    env "${envs[@]}" LD_LIBRARY_PATH="$RT" timeout 600 "$OUT/ameva_conv_check" > "$log" 2>&1
    rc=$?
    chk=$(grep -oE "CONV_CHECK [0-9]+/[0-9]+ passed" "$log" | awk '{print $2}')
    ovr=$(grep -c "(AMEVA_CL_SOURCE_DIR)" "$log")
    case "$v" in *:*) [ "$ovr" -ge 1 ] || rc=99 ;; esac
    echo "AB_CHECK var=${v%%:*} rc=$rc check=${chk:-NA} overrides=$ovr"
    [ "$rc" -eq 0 ] && OK+=("$v")
done
n=${#OK[@]}
[ "$n" -ge 1 ] || { echo "AB_FAIL no variant passed its check"; exit 1; }
for r in $(seq 1 "$ROUNDS"); do
    for sh in "${S[@]}"; do
        for p in $(seq 0 $((n - 1))); do
            v=${OK[$(( (p + r - 1) % n ))]}
            venv "$v"; cool
            t0=$(rd "$TMU"); c0=$(rd "$CAP")
            log="$OUT/ab_run.log"
            env "${envs[@]}" AMEVA_CONV_CHECK_TIME_ONLY=1 AMEVA_CONV_CHECK_TIME="$sh" AMEVA_CONV_CHECK_TIME_ITERS="$ITERS" \
                AMEVA_PROFILE=2 LD_LIBRARY_PATH="$RT" timeout 600 "$OUT/ameva_conv_check" > "$log" 2>&1
            rc=$?
            t1=$(rd "$TMU"); c1=$(rd "$CAP")
            line=$(grep "^CONV_TIME" "$log")
            gms=$(grep -E 'AMEVA_PROFILE kind="CONV_2D' "$log" | grep -oE "gpu_ms=[0-9.]+" | cut -d= -f2 | awk '{s+=$1} END {if (NR) print s}')
            if [ "$rc" -ne 0 ] || [ -z "$line" ] || [ -z "$gms" ]; then
                echo "AB round=$r pos=$p var=${v%%:*} shape=$sh rc=$rc FAILED: $(grep -E "FAIL|error|abort" "$log" | head -3 | tr '\n' ' ')"
                continue
            fi
            read -r _ shape _ _ mean _ _ _ _ best _ <<< "$line"
            gpu=$(awk -v g="$gms" -v n="$ITERS" 'BEGIN {printf "%.2f", g / (n + 1)}')
            echo "AB round=$r pos=$p var=${v%%:*} shape=$shape mean_ms=$mean best_ms=$best gpu_ms=$gpu" \
                 "tmu0=$t0 tmu1=$t1 cap0=$c0 cap1=$c1 cool_wait=$waited"
        done
    done
done
echo "AB_END $(date +%F_%T)"
