#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# g2_profile.sh MODEL.gguf TAG [N_DECODE] [PROMPT_TOKENS]
# GPU time per op kind (AMEVA_PROFILE=2) for (1) token generation and (2) prompt processing, using llama-completion
# with the graph engine in OUT (default ~/abi-lab/out-g2). Logs in ~/abi-lab/results/prof_TAG_{decode,prompt}.log.
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
M=$1
TAG=$2
ND=${3:-64}
NP=${4:-256}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
mkdir -p "$LAB/results"
PROMPT_FILE=$LAB/results/prof_prompt.txt
# about 3 bytes per token of wikitext: roughly NP tokens, well inside the 2048-token context used below
head -c $((NP * 3)) "$LAB/corpus/wiki.test.raw" > "$PROMPT_FILE"
run() { # name extra-args...
    local name=$1
    shift
    # --fit off: llama.cpp b8471 otherwise shrinks unset arguments to the device's free memory and may keep every layer
    # on the host even with -ngl 99 (seen on A35: AMEVA0 "self 0" in the memory breakdown)
    AMEVA_PROFILE=2 GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 900 llama-completion \
        -m "$M" --device AMEVA0 -ngl 99 --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t 4 "$@" \
        > "$LAB/results/prof_${TAG}_$name.out" 2> "$LAB/results/prof_${TAG}_$name.log" < /dev/null
    echo "=== $name rc=$?"
    grep -v "^WARNING: linker" "$LAB/results/prof_${TAG}_$name.log" |
        grep -E "prompt eval time|       eval time|AMEVA_PROFILE (graphs|per_graph|kind)|ggml-ameva|AMEVA0 \(|error" | head -40
}
run decode -p "The three primary colors are" -n "$ND" -c 4096
run prompt -f "$PROMPT_FILE" -n 1 -c 2048 -b 512 -ub 512
