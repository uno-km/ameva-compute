#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Whisper through the STT bundle's whisper-cli on CPU, on the bundle's own ggml-vulkan, and on ggml-ameva (plugin of
# stt_build.sh), in alternating order. Reports whisper.cpp's own stage times and compares the transcripts.
#
#   bash scripts/termux/lab/stt_bench.sh [--out DIR] [--model FILE] [--wav FILE] [--modes "cpu ameva vulkan"]
#                                        [--reps N] [--threads N] [--lang L] [--res DIR] [--extra "ARGS"]
#
#   --out DIR     plugin directory of stt_build.sh (default ~/abi-lab/out-stt)
#   --modes LIST  cpu (-ng), vulkan (bundle ggml-vulkan, -dev of Vulkan0), ameva (plugin, -dev of AMEVA0),
#                 ameva-strict (AMEVA_STRICT=1: a declined op ends the run and names the op)
#   --extra ARGS  appended to every whisper-cli command (e.g. "-bs 1 -bo 1" or "-nfa")
#
# Environment: GUARD_MB (default 500) ends whisper-cli when MemAvailable falls below it, before Android ends the
# Termux session; AMEVA_* and GGML_SCHED_DEBUG pass through to the ameva runs. CPU_CLI names another whisper-cli for
# the cpu mode: the bundle's libggml-cpu is built with i8mm and stops with SIGILL on cores without it (Exynos 2100,
# Exynos 1380), so the CPU reference there has to come from another build.
#
# whisper.cpp returns to the CPU without an error when the requested GPU device does not exist ("no GPU found"). A
# run therefore counts only when its log names the expected backend; anything else is reported as FAIL.
#
# Result lines (also in RES/summary.txt):
#   RUN <mode> rep=<n> rc=<rc> backend=<name> load=<ms> mel=<ms> encode=<ms>/<runs> decode=<ms>/<runs>
#       batchd=<ms>/<runs> total=<ms> wall=<s> cpu=<user+sys s> mem_min=<MB> gpu_clk=<value> text_sha=<12 hex>
#   TEXT <mode> <transcript of the first repetition>
#   CMP <mode> vs cpu: identical | differs (<n> of <m> words)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$HOME/abi-lab/out-stt" BUNDLE="$HOME/.local/share/ameva/current/stt"
MODEL="$HOME/.cache/termux-stt/models/whisper/ggml-large-v3-turbo.bin"
WAV="$HOME/.cache/termux-stt/samples/jfk_1min.wav"
MODES="cpu ameva" REPS=2 THREADS=4 LANG_ARG="en" RES="" EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --bundle) BUNDLE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --wav) WAV="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --lang) LANG_ARG="$2"; shift 2 ;;
        --res) RES="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "[stt_bench] ERROR: $*" >&2; exit 2; }
CLI="$BUNDLE/bin/whisper-cli"
[ -x "$CLI" ] || die "$CLI not found"
[ -f "$MODEL" ] || die "model $MODEL not found (no download is attempted)"
[ -f "$WAV" ] || die "audio $WAV not found"
RES="${RES:-$HOME/abi-lab/results/stt_$(date +%m%d_%H%M%S)}"
mkdir -p "$RES"
GUARD_MB=${GUARD_MB:-500}

avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
gpu_clk() {
    local f
    for f in /sys/class/kgsl/kgsl-3d0/gpuclk /sys/kernel/gpu/gpu_clock /sys/class/kgsl/kgsl-3d0/clock_mhz; do
        if [ -r "$f" ]; then cat "$f" 2>/dev/null | tr -d '\n'; return; fi
    done
    printf 'na'
}

# GPU index of a device, in the order whisper.cpp counts GPU devices for -dev N.
dev_index() {  # $1 = device name, $2 = "plugin" to load the ameva plugin
    local list
    if [ "${2:-}" = plugin ]; then
        list="$(LD_LIBRARY_PATH="$OUT/ggml-rt:$OUT" GGML_BACKEND_PATH="$OUT/libggml-ameva.so" "$OUT/ggml_list_devices" 2>/dev/null)"
    else
        list="$(LD_LIBRARY_PATH="$OUT/ggml-rt" "$OUT/ggml_list_devices" 2>/dev/null)"
    fi
    sed -n "s/.*(-dev \([0-9]*\)) $1 .*/\1/p" <<< "$list" | sed -n 1p
}

NEED_PLUGIN=0
case " $MODES " in *" ameva "*|*" ameva-strict "*) NEED_PLUGIN=1 ;; esac
if [ "$NEED_PLUGIN" = 1 ] || case " $MODES " in *" vulkan "*) true ;; *) false ;; esac; then
    [ -x "$OUT/ggml_list_devices" ] || die "$OUT/ggml_list_devices not found: run stt_build.sh"
fi
AMEVA_DEV="" VULKAN_DEV=""
if [ "$NEED_PLUGIN" = 1 ]; then
    [ -f "$OUT/libggml-ameva.so" ] || die "$OUT/libggml-ameva.so not found: run stt_build.sh"
    AMEVA_DEV="$(dev_index AMEVA0 plugin)"
    [ -n "$AMEVA_DEV" ] || die "AMEVA0 is not registered by the bundle's ggml with $OUT/libggml-ameva.so"
fi
case " $MODES " in *" vulkan "*)
    VULKAN_DEV="$(dev_index Vulkan0)"
    [ -n "$VULKAN_DEV" ] || die "Vulkan0 is not registered by the bundle's ggml" ;;
esac

{
    echo "conditions: $(date +%F_%T) device=$(getprop ro.product.model) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status) threads=$THREADS"
    echo "conditions: model=$(basename "$MODEL") $(du -m "$MODEL" | cut -f1)MB wav=$(basename "$WAV") lang=$LANG_ARG extra=\"$EXTRA\" modes=\"$MODES\" reps=$REPS"
    echo "conditions: mem_available_mb=$(avail_mb) guard_mb=$GUARD_MB gpu_clk=$(gpu_clk) ameva_dev=${AMEVA_DEV:-na} vulkan_dev=${VULKAN_DEV:-na}"
    echo "conditions: cli_sha256=$(sha256sum "$CLI" | cut -c1-12) plugin_sha256=$( [ -f "$OUT/libggml-ameva.so" ] && sha256sum "$OUT/libggml-ameva.so" | cut -c1-12 || echo na)"
} | tee "$RES/summary.txt"

ms() {  # $1 = log, $2 = timing label; prints "<ms>/<runs>" or "<ms>"
    awk -v k="$2" '$0 ~ "whisper_print_timings:" && $0 ~ k" time" {
        v = $0; sub(/.*= */, "", v); split(v, a, " "); r = "";
        if (match($0, /\( *[0-9]+ runs/)) { r = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", r); r = "/" r }
        printf "%s%s", a[1], r; exit }' "$1"
}

run_one() {  # $1 = mode, $2 = rep
    local mode="$1" rep="$2" tag="$1.$2" expect rc pid gpid t0 t1 backend cli="$CLI"
    local -a env=() dev=()
    case "$mode" in
        cpu) dev=(-ng); expect="no GPU found"; cli="${CPU_CLI:-$CLI}" ;;
        vulkan) dev=(-dev "$VULKAN_DEV"); expect="using Vulkan0 backend" ;;
        ameva) dev=(-dev "$AMEVA_DEV"); env=(GGML_BACKEND_PATH="$OUT/libggml-ameva.so"); expect="using AMEVA0 backend" ;;
        ameva-strict) dev=(-dev "$AMEVA_DEV"); env=(GGML_BACKEND_PATH="$OUT/libggml-ameva.so" AMEVA_STRICT=1)
                      expect="using AMEVA0 backend" ;;
        *) die "unknown mode $mode" ;;
    esac
    # The plugin finds libac_runtime.so.2 beside itself only when its RUNPATH starts with $ORIGIN, which needs
    # patchelf at build time (Galaxy A35 has none, and $PREFIX/lib holds an older copy). OUT holds nothing else.
    if [ "${#env[@]}" -gt 0 ]; then env+=(LD_LIBRARY_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"); fi
    echo "$(avail_mb)" > "$RES/$tag.mem"
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086
    ( TIMEFORMAT='%U %S'; time env "${env[@]}" "$cli" "${dev[@]}" -m "$MODEL" -f "$WAV" -t "$THREADS" -l "$LANG_ARG" -nt $EXTRA \
          > "$RES/$tag.txt" 2> "$RES/$tag.log" ) 2> "$RES/$tag.time" &
    pid=$!
    ( while kill -0 $pid 2>/dev/null; do
          a=$(avail_mb)
          [ "$a" -lt "$(cat "$RES/$tag.mem")" ] && echo "$a" > "$RES/$tag.mem"
          if [ "$a" -lt "$GUARD_MB" ]; then
              echo "GUARD_KILL mem_available_mb=$a (< $GUARD_MB)" > "$RES/$tag.guard"
              # by parent: Termux starts the program through linker64, so its process name is not whisper-cli
              pkill -9 -P $pid
          fi
          sleep 1
      done ) &
    gpid=$!
    wait $pid; rc=$?
    kill $gpid 2>/dev/null; wait $gpid 2>/dev/null
    t1=$(date +%s.%N)
    if grep -q "using AMEVA0 backend" "$RES/$tag.log"; then backend=AMEVA0
    elif grep -q "using Vulkan0 backend" "$RES/$tag.log"; then backend=Vulkan0
    elif grep -q "no GPU found" "$RES/$tag.log"; then backend=CPU
    else backend=unknown; fi
    local verdict=""
    grep -q "$expect" "$RES/$tag.log" || verdict=" FAIL(expected \"$expect\")"
    [ -f "$RES/$tag.guard" ] && verdict="$verdict $(cat "$RES/$tag.guard")"
    [ "$rc" = 0 ] || verdict="$verdict FAIL(rc=$rc: $(grep -vE '^\s*$' "$RES/$tag.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300))"
    if [ -z "$verdict" ] && [ -s "$RES/$tag.txt" ]; then touch "$RES/$tag.ok"; fi
    echo "RUN $mode rep=$rep rc=$rc backend=$backend load=$(ms "$RES/$tag.log" load) mel=$(ms "$RES/$tag.log" mel)" \
         "encode=$(ms "$RES/$tag.log" encode) decode=$(ms "$RES/$tag.log" decode) batchd=$(ms "$RES/$tag.log" batchd)" \
         "total=$(ms "$RES/$tag.log" total) wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f", b - a}')" \
         "cpu=$(tail -n 1 "$RES/$tag.time" 2>/dev/null | awk '{printf "%.1f", $1 + $2}') mem_min=$(cat "$RES/$tag.mem")" \
         "gpu_clk=$(gpu_clk) text_sha=$(tr -s ' \n' ' ' < "$RES/$tag.txt" | sha256sum | cut -c1-12)$verdict" | tee -a "$RES/summary.txt"
}

for rep in $(seq 1 "$REPS"); do
    for mode in $MODES; do run_one "$mode" "$rep"; done
done

for mode in $MODES; do
    echo "TEXT $mode $(tr -s ' \n' ' ' < "$RES/$mode.1.txt" | cut -c1-600)" | tee -a "$RES/summary.txt"
done
# transcripts are compared only between runs that completed on the expected backend with a non-empty result
if [ -f "$RES/cpu.1.ok" ]; then
    for mode in $MODES; do
        [ "$mode" = cpu ] && continue
        if [ ! -f "$RES/$mode.1.ok" ]; then
            echo "CMP $mode vs cpu: not compared (the $mode run failed)" | tee -a "$RES/summary.txt"
        elif cmp -s <(tr -s ' \n' ' ' < "$RES/cpu.1.txt") <(tr -s ' \n' ' ' < "$RES/$mode.1.txt"); then
            echo "CMP $mode vs cpu: identical" | tee -a "$RES/summary.txt"
        else
            m=$(tr -s ' \n' '\n' < "$RES/cpu.1.txt" | grep -c .)
            n=$(diff <(tr -s ' \n' '\n' < "$RES/cpu.1.txt") <(tr -s ' \n' '\n' < "$RES/$mode.1.txt") | grep -c '^[<>]')
            echo "CMP $mode vs cpu: differs ($n changed lines of a $m-word reference)" | tee -a "$RES/summary.txt"
        fi
    done
else
    echo "CMP: no CPU reference (the cpu run is absent or failed)" | tee -a "$RES/summary.txt"
fi
echo "BENCH_DONE $RES" | tee -a "$RES/summary.txt"
