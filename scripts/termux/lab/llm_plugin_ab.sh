#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# llm_plugin_ab.sh MODEL.gguf: prompt processing and token generation of llama-completion on AMEVA0 for several builds
# of the plugin and backends, under the measurement protocol of the research program (grade A, ameva-foundation
# 03_core/docs/research/RESEARCH_PROGRAM.md section 3 item 7):
#   - VARIANTS (space separated) "<name>:<plugin dir>:<vulkan|opencl>"; the plugin is <plugin dir>/libggml-ameva.so.
#     An optional fourth field holds environment settings of that variant, separated by ";" (a comma is part of values such as
#     AMEVA_MMQB=q4_K=2x4,q6_K=1x8), e.g. "mmqb:DIR:opencl:AMEVA_MMQB=q4_K=2x4,q6_K=1x8;AMEVA_FUSE=1").
#   - ROUNDS rounds (default 5), the variants rotated by the round number.
#   - before every run: wait (at most COOL_MAX s, default 120) until the GPU temperature (gpu_sysfs.sh: Exynos Mali or Adreno)
#     is at most TMU_MAX (default 42) and the clock cap (gpu_sysfs.sh) is back at CAP_TARGET (default: the cap when the
#     session starts); temperature and cap recorded before and after. A run whose cap was below CAP_TARGET before
#     or after is printed with valid=0 and must be left out (protocol rule 5: S21 throttled to 585 and 403 MHz
#     with the temperature at the gate, 2026-11-07).
#   - one process per run: the first BYTES (default 1500, about 370 tokens) of wikitext-2 test as the prompt, N
#     (default 64) generated tokens, greedy, -c 2048 -b/-ub 512, FA on, -t THREADS (default 4).
# Prints one line per run: LAB round= pos= var= pp_tps= tg_tps= text= (sha256 of the generated text, which must be
# equal for builds that compute the same arithmetic) tmu0= tmu1= cap0= cap1=.
# PROFILE_RE=<awk regex> runs with AMEVA_PROFILE=2 and adds sel_gpu_ms= (GPU time of the AMEVA_PROFILE kinds that
# match) and all_gpu_ms=; the timestamps cost time, so take t/s from a pass without it. The profile counts the
# warm-up graph too (llama-completion decodes BOS+EOS before the prompt; --no-warmup removes it).
set -u
LAB=$HOME/abi-lab
M=${1:?model}
ROUNDS=${ROUNDS:-5}
N=${N:-64}
BYTES=${BYTES:-1500}
THREADS=${THREADS:-4}
TMU_MAX=${TMU_MAX:-42}
COOL_MAX=${COOL_MAX:-120}
VARIANTS=${VARIANTS:?VARIANTS}
# GPU clock cap and temperature of Exynos Mali and Adreno (gpu_sysfs.sh next to this script); the gates that read NA
# are reported at the start and do not hold a run.
. "$(dirname "$0")/gpu_sysfs.sh" || { echo "LAB FAIL: gpu_sysfs.sh not found next to $0"; exit 2; }
CAP_TARGET=${CAP_TARGET:-$(gpu_cap_khz)}
OUTD=${OUTD:-$LAB/results}
CORPUS=${CORPUS:-$LAB/corpus/wiki.test.raw}
[ -r "$CORPUS" ] || { echo "LAB FAIL: no prompt corpus at $CORPUS (set CORPUS)"; exit 2; }
P=$OUTD/prompt_$BYTES.txt
mkdir -p "$OUTD"
head -c "$BYTES" "$CORPUS" > "$P"
echo "LAB_GATES family=$(gpu_family) cap_target=$CAP_TARGET temp_now=$(gpu_temp_c) (NA: gate not available on this device)"
echo "LAB_START $(date +%F_%T) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') model=$(basename "$M")" \
     "rounds=$ROUNDS n=$N bytes=$BYTES threads=$THREADS variants=\"$VARIANTS\""
read -r -a V <<< "$VARIANTS"
n=${#V[@]}
for r in $(seq 1 "$ROUNDS"); do
    for p in $(seq 0 $((n - 1))); do
        v=${V[$(( (p + r - 1) % n ))]}
        IFS=: read -r name dir api envs <<< "$v"
        IFS=";" read -r -a E <<< "${envs:-}"
        waited=$(gpu_gate_wait "$CAP_TARGET" "$TMU_MAX" "$COOL_MAX")
        RT="$dir"
        [ -d "${GGML_RT:-$LAB/rt098}" ] && RT="${GGML_RT:-$LAB/rt098}:$dir"
        t0=$(gpu_temp_c); c0=$(gpu_cap_khz)
        env "${E[@]}" AMEVA_PROFILE=${PROFILE_RE:+2} AMEVA_BACKEND=$api GGML_BACKEND_PATH="$dir/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 900 llama-completion \
            -m "$M" --device AMEVA0 -ngl 99 --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" \
            -f "$P" -n "$N" -c 2048 -b 512 -ub 512 > "$OUTD/lab_run.out" 2> "$OUTD/lab_run.log" < /dev/null
        rc=$?
        t1=$(gpu_temp_c); c1=$(gpu_cap_khz)
        valid=1; { [ "$c0" = "$CAP_TARGET" ] && [ "$c1" = "$CAP_TARGET" ]; } || valid=0
        pp=$(grep "prompt eval time" "$OUTD/lab_run.log" | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
        tg=$(grep -E "[^t] eval time" "$OUTD/lab_run.log" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
        txt=$(sha256sum < "$OUTD/lab_run.out" | cut -c1-12)
        prof=""
        if [ -n "${PROFILE_RE:-}" ]; then
            prof=$(grep 'AMEVA_PROFILE kind=' "$OUTD/lab_run.log" | awk -v re="$PROFILE_RE" '
                { k = $0; sub(/.*kind="/, "", k); sub(/".*/, "", k); g = $0; sub(/.*gpu_ms=/, "", g); sub(/ .*/, "", g)
                  all += g; if (k ~ re) sel += g }
                END { printf " sel_gpu_ms=%.1f all_gpu_ms=%.1f", sel, all }')
        fi
        echo "LAB round=$r pos=$p var=$name rc=$rc pp_tps=${pp:-NA} tg_tps=${tg:-NA} text=$txt tmu0=$t0 tmu1=$t1 cap0=$c0" \
             "cap1=$c1 valid=$valid cool_wait=$waited$prof"
    done
done
echo "LAB_END $(date +%F_%T)"
