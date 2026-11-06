#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# llm_plugin_ab.sh MODEL.gguf: prompt processing and token generation of llama-completion on AMEVA0 for several builds
# of the plugin and backends, under the measurement protocol of the research program (grade A, ameva-foundation
# 03_core/docs/research/RESEARCH_PROGRAM.md section 3 item 7):
#   - VARIANTS (space separated) "<name>:<plugin dir>:<vulkan|opencl>"; the plugin is <plugin dir>/libggml-ameva.so.
#   - ROUNDS rounds (default 5), the variants rotated by the round number.
#   - before every run: wait (at most COOL_MAX s, default 120) until the GPU temperature (/sys/kernel/gpu/gpu_tmu)
#     is at most TMU_MAX (default 42); temperature and clock cap recorded before and after.
#   - one process per run: the first BYTES (default 1500, about 370 tokens) of wikitext-2 test as the prompt, N
#     (default 64) generated tokens, greedy, -c 2048 -b/-ub 512, FA on, -t THREADS (default 4).
# Prints one line per run: LAB round= pos= var= pp_tps= tg_tps= text= (sha256 of the generated text, which must be
# equal for builds that compute the same arithmetic) tmu0= tmu1= cap0= cap1=.
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
TMU=/sys/kernel/gpu/gpu_tmu
CAP=/sys/kernel/gpu/gpu_max_clock
rd() { cat "$1" 2>/dev/null | tr -d ' \n' || echo NA; }
P=$LAB/results/prompt_$BYTES.txt
mkdir -p "$LAB/results"
head -c "$BYTES" "$LAB/corpus/wiki.test.raw" > "$P"
echo "LAB_START $(date +%F_%T) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') model=$(basename "$M")" \
     "rounds=$ROUNDS n=$N bytes=$BYTES threads=$THREADS variants=\"$VARIANTS\""
read -r -a V <<< "$VARIANTS"
n=${#V[@]}
for r in $(seq 1 "$ROUNDS"); do
    for p in $(seq 0 $((n - 1))); do
        v=${V[$(( (p + r - 1) % n ))]}
        IFS=: read -r name dir api <<< "$v"
        waited=0
        while [ -r "$TMU" ] && [ "$(rd "$TMU" | cut -d. -f1)" -gt "$TMU_MAX" ] && [ "$waited" -lt "$COOL_MAX" ]; do
            sleep 5; waited=$((waited + 5))
        done
        RT="$dir"
        [ -d "$LAB/rt098" ] && RT="$LAB/rt098:$dir"
        t0=$(rd "$TMU"); c0=$(rd "$CAP")
        AMEVA_BACKEND=$api GGML_BACKEND_PATH="$dir/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 900 llama-completion \
            -m "$M" --device AMEVA0 -ngl 99 --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" \
            -f "$P" -n "$N" -c 2048 -b 512 -ub 512 > "$LAB/results/lab_run.out" 2> "$LAB/results/lab_run.log" < /dev/null
        rc=$?
        t1=$(rd "$TMU"); c1=$(rd "$CAP")
        pp=$(grep "prompt eval time" "$LAB/results/lab_run.log" | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
        tg=$(grep -E "[^t] eval time" "$LAB/results/lab_run.log" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
        txt=$(sha256sum < "$LAB/results/lab_run.out" | cut -c1-12)
        echo "LAB round=$r pos=$p var=$name rc=$rc pp_tps=${pp:-NA} tg_tps=${tg:-NA} text=$txt tmu0=$t0 tmu1=$t1 cap0=$c0" \
             "cap1=$c1 cool_wait=$waited"
    done
done
echo "LAB_END $(date +%F_%T)"
