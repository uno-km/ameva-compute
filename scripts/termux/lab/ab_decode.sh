#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# ab_decode.sh MODEL.gguf ROUNDS BUILD...
# Token generation A/B: decode_bench.sh for every build directory name under ~/abi-lab (e.g. out-h0 out-h1), the
# builds alternating within each round so that thermal drift spreads over all of them. Prints one line per run with
# t/s, the CPU affinity and the Adreno clock cap.
set -u
LAB=$HOME/abi-lab
M=$1
R=$2
shift 2
cd "$LAB" || exit 1
for r in $(seq 1 "$R"); do
    for b in "$@"; do
        line=$(OUT="$LAB/$b" bash decode_bench.sh "$M" "ab_${b}_$r" gpu 2>&1)
        tps=$(printf '%s\n' "$line" | grep "       eval time" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
        cond=$(printf '%s\n' "$line" | grep "^===" | sed -E 's/.*(Cpus_allowed_list:[^g]*)(gpu_max_clk=[0-9a-z\/]+).*/\1\2/')
        echo "round=$r build=$b decode_tps=$tps $cond"
    done
done
