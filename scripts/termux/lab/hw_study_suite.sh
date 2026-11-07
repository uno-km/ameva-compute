#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# hw_study_suite.sh KIT: the hardware study of the prompt MUL_MAT (ameva-foundation N-20261107-hw-bottleneck) on one
# fleet phone, unattended, from a kit that scripts/fleet/hw_study_fleet.sh put on the phone:
#   KIT/bin   libggml-ameva.so, libac_runtime.so.2, ameva_qmm_check, ameva_mmq_bench, ggml-rt/ (one backend build)
#   KIT/rt    the ggml / llama libraries the backend was built against (llama-completion runs with them)
#   KIT/src   cl_hw_bench.c, include/CL, gpu_sysfs.sh, llm_plugin_ab.sh, prompt_corpus.txt
# Stages (STAGES, default "info hw qmm mm"; "llm" runs when MODEL is set):
#   info  phone, SoC, CPU set, GPU family, clock cap, temperature, OpenCL vendor library
#   hw    cl_hw_bench (unit throughputs per clock and core) at the clock cap
#   qmm   ameva_qmm_check on every API of APIS (default "opencl vulkan"); an API whose check fails is left out of mm
#   mm    ameva_mmq_bench on SHAPES (type:M:N:K) for every API: ameva_mmq (AMEVA_MMQB=0), the f32 tile kernel
#         (AMEVA_MMQ=0) and ameva_mmqb at every layout of LAYOUTS, ROUNDS rounds with the variants rotated; the mmq and
#         mmqb variants only where the probe run says MMQ is on for that API
#   llm   llm_plugin_ab.sh MODEL with the default build, AMEVA_MMQB=0 and Vulkan, LLM_ROUNDS rounds
# Every run waits for the clock-cap and temperature gates (gpu_sysfs.sh; a gate that cannot be read is reported once
# and does not hold the run) and is printed with cap0/cap1 and valid=0/1 (protocol rule 5). Nothing is skipped
# silently: a stage or variant that cannot run prints a SUITE_SKIP or SUITE_FAIL line with the reason.
# Output: OUT (default KIT/results/<date>) holds suite.log, mm.tsv, hw.log, qmm_<api>.log, llm.log; suite.rc is
# written last (0 when every stage ran, 1 when one failed).
set -u
KIT=$(cd "${1:?kit directory}" && pwd)
B=$KIT/bin; SRC=$KIT/src
. "$SRC/gpu_sysfs.sh" || { echo "SUITE_FAIL stage=setup reason=no gpu_sysfs.sh in $SRC"; exit 2; }
STAGES=${STAGES:-info hw qmm mm}
[ -n "${MODEL:-}" ] && case " $STAGES " in *" llm "*) ;; *) STAGES="$STAGES llm" ;; esac
APIS=${APIS:-opencl vulkan}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-7}
SHAPES=${SHAPES:-q4_K:512:9216:2304 q4_K:512:2304:9216 q4_K:512:2048:2304 q6_K:512:2304:9216 q5_K:512:9216:2304}
LAYOUTS=${LAYOUTS:-1x4 1x8 1x16 2x4 2x8 2x16}
TMU_MAX=${TMU_MAX:-42}
COOL_MAX=${COOL_MAX:-180}
COOL_MIN=${COOL_MIN:-8}
CAP_TARGET=${CAP_TARGET:-$(gpu_cap_khz)}
OUT=${OUT:-$KIT/results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
LOG=$OUT/suite.log
say() { echo "$*" | tee -a "$LOG"; }
failed=0
RTP="$B:$B/ggml-rt"
DEV=$(getprop ro.product.model 2>/dev/null | tr ' ' _)

if [[ " $STAGES " == *" info "* ]]; then
    say "SUITE_INFO date=$(date +%F_%T) model=$DEV soc=$(getprop ro.soc.model 2>/dev/null) android=$(getprop ro.build.version.release 2>/dev/null)" \
        "cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') gpu_family=$(gpu_family) cap_khz=$CAP_TARGET" \
        "temp_c=$(gpu_temp_c) now_mhz=$(gpu_now_mhz) cl_vendor=$(gpu_cl_vendor) kit=$(cat "$KIT/KIT_ID" 2>/dev/null)"
    [ "$CAP_TARGET" = NA ] && say "SUITE_NOTE the clock-cap gate cannot be read on this phone: runs are not gated on the cap"
    [ "$(gpu_temp_c)" = NA ] && say "SUITE_NOTE the temperature gate cannot be read on this phone: runs are not gated on it"
fi

if [[ " $STAGES " == *" hw "* ]]; then
    v=$(gpu_cl_vendor)
    if [ ! -e "$PREFIX/lib/libOpenCL.so" ]; then
        say "SUITE_FAIL stage=hw reason=no ICD loader at \$PREFIX/lib/libOpenCL.so (pkg install ocl-icd)"; failed=1
    elif [ "$v" = NA ]; then
        say "SUITE_FAIL stage=hw reason=no vendor OpenCL library found"; failed=1
    elif ! clang -O2 -I"$SRC/include" -o "$OUT/cl_hw_bench" "$SRC/cl_hw_bench.c" -lOpenCL > "$OUT/hw_build.log" 2>&1; then
        say "SUITE_FAIL stage=hw reason=build failed ($(grep -m1 error "$OUT/hw_build.log"))"; failed=1
    else
        # Mali: Termux's ICD loader with an ICD file naming libGLES_mali.so. Adreno: the vendor libOpenCL.so in place of
        # the loader (through the loader clGetPlatformIDs returns -30 on the S20, 2026-11-07).
        mkdir -p "$OUT/icd"; echo "$v" > "$OUT/icd/vendor.icd"
        if [ "$(gpu_family)" = adreno ]; then CLENV=(LD_LIBRARY_PATH=/vendor/lib64:/system/lib64)
        else CLENV=(OCL_ICD_VENDORS=$OUT/icd LD_LIBRARY_PATH=$PREFIX/lib:/system/lib64:/vendor/lib64:/vendor/lib64/egl); fi
        w=$(gpu_gate_wait "$CAP_TARGET" "$TMU_MAX" "$COOL_MAX"); sleep 20
        mhz=$(( ${CAP_TARGET/NA/0} / 1000 ))
        env "${CLENV[@]}" timeout 1200 "$OUT/cl_hw_bench" "$mhz" 7 > "$OUT/hw.log" 2>&1
        rc=$?
        say "SUITE hw rc=$rc cool_wait=$w cap_after=$(gpu_cap_khz) lines=$(grep -c '^HW' "$OUT/hw.log")"
        [ $rc -ne 0 ] && { failed=1; tail -3 "$OUT/hw.log" | sed 's/^/  /' | tee -a "$LOG"; }
    fi
fi

USABLE=""
if [[ " $STAGES " == *" qmm "* ]]; then
    for api in $APIS; do
        gpu_gate_wait "$CAP_TARGET" "$TMU_MAX" "$COOL_MAX" > /dev/null
        AMEVA_BACKEND=$api LD_LIBRARY_PATH=$RTP timeout 2400 "$B/ameva_qmm_check" > "$OUT/qmm_$api.log" 2>&1
        rc=$?
        say "SUITE qmm api=$api rc=$rc mmqb_cases=$(grep -c 'ameva_mmqb' "$OUT/qmm_$api.log") last=\"$(tail -1 "$OUT/qmm_$api.log")\""
        if [ $rc -eq 0 ]; then USABLE="$USABLE $api"; else failed=1; say "SUITE_SKIP stage=mm api=$api reason=ameva_qmm_check rc=$rc"; fi
    done
else
    USABLE=$APIS
fi

if [[ " $STAGES " == *" mm "* ]]; then
    TSV=$OUT/mm.tsv
    printf 'dev\tapi\ttype\tM\tN\tK\tvariant\tround\tpath\texpect\tmedian_ms\tgops\tcheck\tcap0\tcap1\tvalid\trc\n' > "$TSV"
    for api in $USABLE; do
        # probe: is MMQ on for this API? (the default run of the first shape)
        IFS=: read -r t m n k <<< "${SHAPES%% *}"
        AMEVA_BACKEND=$api LD_LIBRARY_PATH=$RTP timeout 600 "$B/ameva_mmq_bench" "$t" "$m" "$n" "$k" 3 > "$OUT/probe_$api.log" 2>&1
        mmq_on=0; grep -q 'MMQ on' "$OUT/probe_$api.log" && mmq_on=1
        say "SUITE mm_probe api=$api mmq_on=$mmq_on $(grep -o 'path=[a-z_]*' "$OUT/probe_$api.log")"
        [ $mmq_on = 0 ] && say "SUITE_SKIP stage=mm api=$api variants=mmq,mmqb reason=MMQ off on this API ($(grep -m1 'MMQ off' "$OUT/probe_$api.log"))"
        for s in $SHAPES; do
            IFS=: read -r t m n k <<< "$s"
            V=("f32|AMEVA_MMQ=0;AMEVA_MMQB=0|other")
            if [ $mmq_on = 1 ]; then
                V+=("mmq|AMEVA_MMQB=0|ameva_mmq")
                for L in $LAYOUTS; do V+=("mmqb_$L|AMEVA_MMQB=$t=$L|ameva_mmqb"); done
            fi
            nv=${#V[@]}
            for r in $(seq 1 "$ROUNDS"); do
                for p in $(seq 0 $((nv - 1))); do
                    IFS='|' read -r name envs expect <<< "${V[$(( (p + r - 1) % nv ))]}"
                    IFS=";" read -r -a E <<< "$envs"
                    w=$(gpu_gate_wait "$CAP_TARGET" "$TMU_MAX" "$COOL_MAX"); sleep "$COOL_MIN"
                    c0=$(gpu_cap_khz)
                    env "${E[@]}" AMEVA_BACKEND=$api LD_LIBRARY_PATH=$RTP timeout 900 "$B/ameva_mmq_bench" "$t" "$m" "$n" "$k" "$REPS" > "$OUT/mm_run.log" 2>&1
                    rc=$?
                    c1=$(gpu_cap_khz)
                    l=$(grep '^MMQ_BENCH type=' "$OUT/mm_run.log")
                    g() { echo "$l" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }
                    valid=1; { [ "$c0" = "$CAP_TARGET" ] && [ "$c1" = "$CAP_TARGET" ]; } || valid=0
                    path=$(g path); path=${path:-NA}
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DEV" "$api" "$t" "$m" "$n" "$k" "$name" "$r" \
                        "$path" "$expect" "$(g median_ms)" "$(g GOP/s_median)" "$(g check)" "$c0" "$c1" "$valid" "$rc" >> "$TSV"
                    if [ $rc -ne 0 ] || [ "$path" != "$expect" ]; then
                        say "SUITE_FAIL stage=mm api=$api shape=$s variant=$name rc=$rc path=$path expect=$expect: $(grep -m1 -E 'FAIL|ABORT|error' "$OUT/mm_run.log")"
                        failed=1
                    fi
                done
            done
            say "SUITE mm api=$api shape=$s runs=$((nv * ROUNDS)) cap=$(gpu_cap_khz) temp=$(gpu_temp_c)"
        done
    done
fi

if [[ " $STAGES " == *" llm "* ]]; then
    if [ -z "${MODEL:-}" ] || [ ! -r "$MODEL" ]; then
        say "SUITE_FAIL stage=llm reason=MODEL not readable (${MODEL:-unset})"; failed=1
    elif ! command -v llama-completion > /dev/null; then
        say "SUITE_FAIL stage=llm reason=no llama-completion on PATH"; failed=1
    else
        LV=""
        for api in $USABLE; do
            case $api in
                opencl) LV="$LV def:$B:opencl mmq:$B:opencl:AMEVA_MMQB=0" ;;
                *) LV="$LV $api:$B:$api" ;;
            esac
        done
        VARIANTS="${LLM_VARIANTS:-$LV}" ROUNDS=${LLM_ROUNDS:-5} OUTD=$OUT/llm GGML_RT=$KIT/rt CORPUS=$SRC/prompt_corpus.txt \
            CAP_TARGET=$CAP_TARGET TMU_MAX=$TMU_MAX bash "$SRC/llm_plugin_ab.sh" "$MODEL" > "$OUT/llm.log" 2>&1
        rc=$?
        say "SUITE llm rc=$rc runs=$(grep -c '^LAB round' "$OUT/llm.log") valid=$(grep -c 'valid=1' "$OUT/llm.log")"
        [ $rc -ne 0 ] && failed=1
    fi
fi

say "SUITE_END date=$(date +%F_%T) failed=$failed out=$OUT"
echo "$failed" > "$OUT/suite.rc"
