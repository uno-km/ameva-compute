#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# sd_e2e.sh
# stable-diffusion.cpp images with modules on ggml-ameva (diffusion model, VAE), compared with the CPU's images.
# Needs the sd-cli and the plugin built by sd_vae_e2e.sh build (SDB, OUT below).
#
# RUNS   entries "tag=backend assignment" (sd-cli --backend), separated by blanks. Default:
#          cpu=cpu  vae=cpu,vae=AMEVA0  gpu=cpu,diffusion=AMEVA0,vae=AMEVA0
#        The text encoder stays on the CPU in the default runs. Every run uses one seed, --vae-conv-direct and
#        --diffusion-conv-direct (ggml-ameva takes CONV_2D and declines IM2COL).
# A run that names AMEVA0 passes only when: sd-cli exits 0 and writes the image; the log has no "## SPLIT" line (sd-cli
# creates the scheduler, with a CPU backend behind it, only when the module's backend declines a node; GGML_SCHED_DEBUG=1
# prints the splits) and no "ggml-ameva: declined" line; AMEVA_PROFILE reports dispatches. AMEVA_STRICT=1 is set for
# those runs: a declined op ends the process, so an image cannot come from a CPU split.
# PAIRS  "ref:test:gate" compared with img_psnr.py; gate: a PSNR floor in dB, "equal", or "-" (reported only).
#        Default: cpu:vae:PSNR_MIN cpu:gpu:-   (the gpu run samples on another backend: its latent differs from the
#        CPU's by rounding accumulated over the steps, so its PSNR is reported, not gated, unless GPU_PSNR_MIN is set).
# MODEL (default ~/.cache/termux-diffusion/models/anime.gguf), SIZE (512), STEPS (8), SEED (42), THREADS (4), CFG (1),
# SAMPLER (euler), PROMPT, EXTRA (further sd-cli arguments for every run, e.g. "--type q4_K", "--vae-tiling",
# "--lora-model-dir DIR"), TAG (suffix of the result directory), PSNR_MIN (40; this script's choice).
# GUARD_MB: when set, sd-cli is ended as soon as MemAvailable falls below it (checked once a second), before the
# system ends the Termux session; the run then fails with a GUARD_KILL line.
# Prints results only; exit status 0 when everything that ran passed.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src}
SDB=${SDB:-$LAB/sd-build}
OUT=${OUT:-$LAB/out-sd}
MODEL=${MODEL:-$HOME/.cache/termux-diffusion/models/anime.gguf}
SIZE=${SIZE:-512}
STEPS=${STEPS:-8}
SEED=${SEED:-42}
THREADS=${THREADS:-4}
CFG=${CFG:-1}
SAMPLER=${SAMPLER:-euler}
PROMPT=${PROMPT:-a lighthouse on a rocky coast at sunset, detailed painting}
PSNR_MIN=${PSNR_MIN:-40}
EXTRA=${EXTRA:-}
RES=${RES:-$OUT/e2e-${TAG:-$(basename "$MODEL" .gguf)-$SIZE}}
RUN_TIMEOUT=${RUN_TIMEOUT:-5400}
SDCLI=$SDB/bin/sd-cli
RT=$OUT/ggml-rt:$OUT
RUNS=${RUNS:-"cpu=cpu vae=cpu,vae=AMEVA0 gpu=cpu,diffusion=AMEVA0,vae=AMEVA0"}
PAIRS=${PAIRS:-"cpu:vae:$PSNR_MIN cpu:gpu:${GPU_PSNR_MIN:--}"}

[ -x "$SDCLI" ] || { echo "NO_SD_CLI $SDCLI (run: sd_vae_e2e.sh build)"; exit 1; }
[ -f "$OUT/libggml-ameva.so" ] || { echo "NO_PLUGIN $OUT/libggml-ameva.so"; exit 1; }
[ -f "$MODEL" ] || { echo "NO_MODEL $MODEL"; exit 1; }
mkdir -p "$RES"

# run_one TAG BACKEND_ASSIGNMENT: one sd-cli process, one line of results; returns 1 when the run does not pass
run_one() {
    local tag=$1 spec=$2 plugin=() r ok=1 splits declined prof dispatches t0 t1 gpid a
    case "$spec" in
        *AMEVA0*) plugin=(GGML_BACKEND_PATH="$OUT/libggml-ameva.so" AMEVA_PROFILE=1 AMEVA_STRICT=1) ;;
    esac
    rm -f "$RES/$tag.png" "$RES/$tag.guard"
    if [ -n "${GUARD_MB:-}" ]; then
        ( while sleep 1; do
              a=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
              if [ "$a" -lt "$GUARD_MB" ]; then
                  echo "GUARD_KILL mem_available_mb=$a (< $GUARD_MB)" >> "$RES/$tag.guard"
                  pkill -9 -x sd-cli
              fi
          done ) &
        gpid=$!
    fi
    t0=$(date +%s)
    # shellcheck disable=SC2086
    env GGML_SCHED_DEBUG=1 LD_LIBRARY_PATH="$RT" "${plugin[@]}" timeout "$RUN_TIMEOUT" "$SDCLI" -m "$MODEL" \
        -p "$PROMPT" --steps "$STEPS" --cfg-scale "$CFG" --sampling-method "$SAMPLER" -W "$SIZE" -H "$SIZE" \
        --seed "$SEED" -t "$THREADS" --backend "$spec" --vae-conv-direct --diffusion-conv-direct $EXTRA -v \
        -o "$RES/$tag.png" > "$RES/$tag.log" 2>&1
    r=$?
    t1=$(date +%s)
    if [ -n "${GUARD_MB:-}" ]; then kill $gpid 2>/dev/null; wait $gpid 2>/dev/null; fi
    [ -s "$RES/$tag.guard" ] && { ok=0; head -1 "$RES/$tag.guard"; }
    splits=$(grep -c "## SPLIT" "$RES/$tag.log")
    declined=$(grep -c "ggml-ameva: declined" "$RES/$tag.log")
    prof=$(grep -o "AMEVA_PROFILE graphs=[0-9]* nodes=[0-9]* dispatches=[0-9]*" "$RES/$tag.log" | tail -1)
    dispatches=${prof##*dispatches=}
    [ $r -eq 0 ] && [ -s "$RES/$tag.png" ] || ok=0
    if [ ${#plugin[@]} -gt 0 ]; then
        [ "$splits" -eq 0 ] && [ "$declined" -eq 0 ] && [ -n "$prof" ] && [ "${dispatches:-0}" -gt 0 ] || ok=0
    fi
    echo "RUN tag=$tag backend=$spec rc=$r $([ $ok -eq 1 ] && echo PASS || echo FAIL)" \
         "sampling_s=$(grep -o "sampling completed, taking [0-9.]*" "$RES/$tag.log" | tail -1 | grep -o "[0-9.]*$")" \
         "decode_s=$(grep -o "latent 1 decoded, taking [0-9.]*" "$RES/$tag.log" | tail -1 | grep -o "[0-9.]*$")" \
         "wall_s=$((t1 - t0)) sched_splits=$splits declined=$declined" \
         "window_nodes=$(grep -o "AMEVA_PROFILE window_nodes=[0-9]*" "$RES/$tag.log" | tail -1 | grep -o "[0-9]*$")" \
         "${prof:+[$prof]}"
    if [ $ok -eq 0 ]; then
        grep -E "## SPLIT|\[ERROR\]|declin|FAIL-FAST|not supported|ABI mismatch|AMU_ERR|Assert|assert|abort|Killed" "$RES/$tag.log" \
            | grep -v "^WARNING: linker" | cut -c1-240 | sort | uniq -c | sort -rn | head -12
        tail -3 "$RES/$tag.log" | cut -c1-240
        return 1
    fi
    return 0
}

rc=0
echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status) threads=$THREADS" \
     "model=$(basename "$MODEL") size=${SIZE}x$SIZE steps=$STEPS cfg=$CFG sampler=$SAMPLER seed=$SEED" \
     "extra=\"$EXTRA\" mem_available_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" \
     "plugin_sha256=$(sha256sum "$OUT/libggml-ameva.so" | cut -c1-12)"
for e in $RUNS; do
    run_one "${e%%=*}" "${e#*=}" || rc=1
done
for e in $PAIRS; do
    a=${e%%:*} rest=${e#*:}
    b=${rest%%:*} gate=${rest#*:}
    if [ ! -s "$RES/$a.png" ] || [ ! -s "$RES/$b.png" ]; then
        echo "PAIR ref=$a test=$b gate=$gate NO_IMAGE"
        [ "$gate" = - ] || rc=1
        continue
    fi
    case "$gate" in
        -) arg= ;;
        equal) arg=--equal ;;
        *) arg="--min-psnr $gate" ;;
    esac
    if python3 "$SRC/scripts/termux/lab/img_psnr.py" "$RES/$a.png" "$RES/$b.png" $arg > "$RES/pair_${a}_$b.txt" 2>&1; then
        echo "PAIR gate=$gate $([ "$gate" = - ] && echo REPORT || echo PASS) $(sed "s|$RES/||g" "$RES/pair_${a}_$b.txt")"
    else
        echo "PAIR gate=$gate FAIL $(sed "s|$RES/||g" "$RES/pair_${a}_$b.txt" | tail -1)"
        rc=1
    fi
done
echo "SD_E2E $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
