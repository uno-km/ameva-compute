#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# mmdit_census.sh
# One stable-diffusion.cpp run of an MMDiT model (Z-Image-Turbo: LLM text encoder, DiT, VAE) with modules on
# ggml-ameva, under a memory guard, with the op census and the memory curve of the run. Needs the sd-cli and the
# plugin built by sd_vae_e2e.sh build (SDB, OUT below).
#
# The three modules are never resident together by design of the run, not by a second process: sd-cli (50d6405) loads
# the parameters of a module on first use and releases the text encoder when the condition is computed
# (conditioner->runner_done()). This script measures that (RSS and MemAvailable per phase) instead of assuming it.
# MAXVRAM   --max-vram value (default AMEVA0=1.5): the DiT graph is cut into segments whose parameters fit the budget.
#           Empty: no graph cut, the whole DiT is loaded on the backend.
# STREAM    1 adds --stream-layers (default 1; no effect without MAXVRAM).
# PARAMS    --params-backend value (default diffusion=cpu: sd-cli ignores --stream-layers unless the DiT parameters
#           stay on the CPU; with --mmap they are the mapped file). Set PARAMS=none for the sd-cli default.
# BACKEND   --backend value (default te=cpu,diffusion=AMEVA0,vae=AMEVA0).
# STRICT    0 (default, census): a declined op runs on the CPU through the scheduler and is listed.
#           1: AMEVA_STRICT=1, a declined op ends the process; the run passes only without splits and declines.
# GUARD_MB  the guard ends sd-cli when MemAvailable falls below this (default 600), before the system ends the
#           Termux session. RUN_TIMEOUT seconds (default 3600).
# DIT, VAE, LLM (files under ~/.cache/termux-diffusion/models), SIZE (256), STEPS (1), SEED (42), THREADS (4),
# PROMPT, EXTRA (further sd-cli arguments), TAG (suffix of the result directory).
# Prints results only; exit status 0 when the run passed.
set -u
LAB=$HOME/abi-lab
SDB=${SDB:-$LAB/sd-build}
OUT=${OUT:-$LAB/out-sdw5}
M=${M:-$HOME/.cache/termux-diffusion/models}
DIT=${DIT:-$M/z_image_turbo-Q2_K.gguf}
VAE=${VAE:-$M/z_image_ae.safetensors}
LLM=${LLM:-$M/Qwen3-4B-Instruct-2507-Q2_K.gguf}
SIZE=${SIZE:-256}
STEPS=${STEPS:-1}
SEED=${SEED:-42}
THREADS=${THREADS:-4}
PROMPT=${PROMPT:-a lighthouse on a rocky coast at sunset, detailed painting}
BACKEND=${BACKEND:-te=cpu,diffusion=AMEVA0,vae=AMEVA0}
MAXVRAM=${MAXVRAM-AMEVA0=1.5}
STREAM=${STREAM:-1}
PARAMS=${PARAMS:-diffusion=cpu}
[ "$PARAMS" = none ] && PARAMS=
STRICT=${STRICT:-0}
GUARD_MB=${GUARD_MB:-600}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
EXTRA=${EXTRA:-}
RES=${RES:-$OUT/mmdit-${TAG:-$(basename "$DIT" .gguf)-$SIZE}}
SDCLI=$SDB/bin/sd-cli
LOG=$RES/run.log
MEM=$RES/mem.tsv

[ -x "$SDCLI" ] || { echo "NO_SD_CLI $SDCLI (run: sd_vae_e2e.sh build)"; exit 1; }
[ -f "$OUT/libggml-ameva.so" ] || { echo "NO_PLUGIN $OUT/libggml-ameva.so"; exit 1; }
for f in "$DIT" "$VAE" "$LLM"; do [ -f "$f" ] || { echo "NO_MODEL $f"; exit 1; }; done
mkdir -p "$RES"
rm -f "$RES/out.png" "$LOG" "$MEM"

args=(--diffusion-model "$DIT" --vae "$VAE" --llm "$LLM" -p "$PROMPT" --cfg-scale 1 --steps "$STEPS"
      -W "$SIZE" -H "$SIZE" --seed "$SEED" -t "$THREADS" --mmap --backend "$BACKEND"
      --vae-conv-direct --diffusion-conv-direct -v -o "$RES/out.png")
[ -n "$MAXVRAM" ] && args+=(--max-vram "$MAXVRAM")
[ -n "$MAXVRAM" ] && [ "$STREAM" = 1 ] && args+=(--stream-layers)
[ -n "$PARAMS" ] && args+=(--params-backend "$PARAMS")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && args+=($EXTRA)
envs=(GGML_SCHED_DEBUG=1 LD_LIBRARY_PATH="$OUT/ggml-rt:$OUT")
# the plugin is loaded only where it is the backend under test: a CPU reference run does not depend on it
ameva=0
case "$BACKEND" in *AMEVA0*) ameva=1; envs+=(GGML_BACKEND_PATH="$OUT/libggml-ameva.so" AMEVA_PROFILE=1) ;; esac
[ $ameva -eq 1 ] && [ "$STRICT" = 1 ] && envs+=(AMEVA_STRICT=1)

avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status) threads=$THREADS" \
     "dit=$(basename "$DIT") llm=$(basename "$LLM") size=${SIZE}x$SIZE steps=$STEPS seed=$SEED backend=$BACKEND" \
     "max_vram=\"$MAXVRAM\" stream=$STREAM params=\"$PARAMS\" strict=$STRICT extra=\"$EXTRA\"" \
     "mem_available_mb=$(avail_mb) guard_mb=$GUARD_MB plugin_sha256=$(sha256sum "$OUT/libggml-ameva.so" | cut -c1-12)"

# a run of many minutes with the screen off: without the wake lock the device dozes and leaves the network
command -v termux-wake-lock > /dev/null && termux-wake-lock
trap 'command -v termux-wake-unlock > /dev/null && termux-wake-unlock' EXIT
env "${envs[@]}" "$SDCLI" "${args[@]}" > "$LOG" 2>&1 &
pid=$!
t0=$(date +%s)
phase=te
guard=
# one sample a second: seconds, phase (te until the condition is computed, dit until sampling ends, then vae),
# MemAvailable, the process's resident memory, its anonymous and file-backed parts (the mapped model files are
# file-backed: the system can drop them) and its swapped memory (MiB)
while kill -0 $pid 2>/dev/null; do
    if [ $phase = te ] && grep -q "get_learned_condition completed" "$LOG" 2>/dev/null; then phase=dit; fi
    if [ $phase = dit ] && grep -q "sampling completed" "$LOG" 2>/dev/null; then phase=vae; fi
    a=$(avail_mb)
    rss=$(awk '/VmRSS/ {r = int($2 / 1024)} /RssAnon/ {an = int($2 / 1024)} /RssFile/ {fl = int($2 / 1024)}
               /VmSwap/ {s = int($2 / 1024)} END {print r + 0, an + 0, fl + 0, s + 0}' /proc/$pid/status 2>/dev/null)
    now=$(($(date +%s) - t0))
    echo "$now $phase $a ${rss:-0 0 0 0}" >> "$MEM"
    if [ "$a" -lt "$GUARD_MB" ]; then
        guard="GUARD_KILL phase=$phase t=${now}s mem_available_mb=$a (< $GUARD_MB)"
        kill -9 $pid
    elif [ "$now" -gt "$RUN_TIMEOUT" ]; then
        guard="TIMEOUT_KILL phase=$phase t=${now}s"
        kill -9 $pid
    fi
    sleep 1
done
wait $pid
r=$?
t1=$(date +%s)

splits=$(grep -c "## SPLIT" "$LOG")
declined=$(grep -c "^AMEVA_PROFILE declined op=" "$LOG")
prof=$(grep -o "AMEVA_PROFILE graphs=[0-9]* nodes=[0-9]* dispatches=[0-9]*" "$LOG" | tail -1)
dispatches=${prof##*dispatches=}
ok=1
[ $r -eq 0 ] && [ -s "$RES/out.png" ] || ok=0
if [ $ameva -eq 1 ]; then [ -n "$prof" ] && [ "${dispatches:-0}" -gt 0 ] || ok=0; fi
if [ $ameva -eq 1 ] && [ "$STRICT" = 1 ]; then [ "$splits" -eq 0 ] && [ "$declined" -eq 0 ] || ok=0; fi
echo "RUN rc=$r $([ $ok -eq 1 ] && echo PASS || echo FAIL) ${guard:+[$guard] }" \
     "te_s=$(grep -o "get_learned_condition completed, taking [0-9.]*" "$LOG" | tail -1 | grep -o "[0-9.]*$")" \
     "sampling_s=$(grep -o "sampling completed, taking [0-9.]*" "$LOG" | tail -1 | grep -o "[0-9.]*$")" \
     "decode_s=$(grep -o "latent 1 decoded, taking [0-9.]*" "$LOG" | tail -1 | grep -o "[0-9.]*$")" \
     "wall_s=$((t1 - t0)) sched_splits=$splits declined_kinds=$declined ${prof:+[$prof]}"
# per phase: samples, the process's peak resident memory, its anonymous memory (peak, first and last sample), its
# file-backed memory at the last sample, the lowest MemAvailable
awk '{n[$2]++; if ($4 > pk[$2]) pk[$2] = $4; if ($5 > ap[$2]) ap[$2] = $5; if (!($2 in f)) f[$2] = $5; l[$2] = $5;
      fl[$2] = $6; if (!($2 in mn) || $3 < mn[$2]) mn[$2] = $3; if ($7 > sw[$2]) sw[$2] = $7}
     END {for (p in n) printf "MEM phase=%s samples=%d rss_peak_mb=%d anon_peak_mb=%d anon_first_mb=%d anon_last_mb=%d file_last_mb=%d swap_peak_mb=%d mem_available_min_mb=%d\n",
                              p, n[p], pk[p], ap[p], f[p], l[p], fl[p], sw[p], mn[p]}' "$MEM" | sort
# where the text encoder's parameter buffer is released: before "generating image" (released for the DiT) or after
# "generate_image completed" (resident, or swapped, through the DiT and the VAE)
grep -n -E "releasing params backend buffer|generating image:|generate_image completed" "$LOG" \
    | sed -E "s/\[[A-Z ]*\] [a-z_.-]*:[0-9]* *- //" | cut -c1-110 | tr "\n" ";"
echo
grep -E "total params memory size|graph cut max_vram|streaming budget|ignoring" "$LOG" | cut -c1-260 | head -6
echo "CENSUS ops on ggml-ameva (nodes per graph):"
grep -o "AMEVA_PROFILE encode_op op=[A-Z_0-9]* nodes_per_graph=[0-9.]*" "$LOG" | sed "s/AMEVA_PROFILE encode_op //" | tr "\n" ";"
echo
echo "CENSUS declined:"
grep "^AMEVA_PROFILE declined" "$LOG" | cut -c1-300
grep "^ggml-ameva: declined" "$LOG" | sed "s/ -- it runs on another backend//; s/([^,]*, /(/" | sort | uniq -c | sort -rn | cut -c1-200
grep -E "^AMEVA_PROFILE (window_nodes|mmq_mul_mat|repack)" "$LOG" | cut -c1-240
if [ $ok -eq 0 ]; then
    grep -E "\[ERROR\]|FAIL-FAST|not supported|ABI mismatch|AMU_ERR|Assert|assert|abort|Killed|unknown" "$LOG" \
        | grep -v "^WARNING: linker" | cut -c1-240 | sort | uniq -c | sort -rn | head -12
    tail -4 "$LOG" | cut -c1-240
fi
echo "MMDIT_CENSUS $([ $ok -eq 1 ] && echo PASS || echo FAIL)"
[ $ok -eq 1 ]
