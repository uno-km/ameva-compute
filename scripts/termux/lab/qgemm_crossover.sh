#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# qgemm_crossover.sh [TYPE]
# Repacked-weight MUL_MAT time for small activation-row counts M: ameva_qmv (COLS = 8, default below
# AMEVA_QGEMM_MIN_M) against ameva_qgemm (AMEVA_QGEMM_MIN_M=2), for the two FFN shapes of Qwen2.5-0.5B.
# Needs OUT/ameva_mm_bench (default ~/abi-lab/out-g2).
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
T=${1:-q5_0}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
export LD_LIBRARY_PATH="$RT"
ms() { # extracts the ms= field of the repacked-path line
    awk '/path=qmv/ { for (i = 1; i <= NF; ++i) if ($i ~ /^ms=/) { sub(/^ms=/, "", $i); print $i } }'
}
for M in 2 4 8 12 16 24 32; do
    for shape in "4864 896" "896 4864"; do
        set -- $shape
        q=$(AMEVA_QGEMM_MIN_M=65535 "$OUT/ameva_mm_bench" "$T" "$M" "$1" "$2" 10 2>&1 | ms)
        g=$(AMEVA_QGEMM_MIN_M=2 "$OUT/ameva_mm_bench" "$T" "$M" "$1" "$2" 10 2>&1 | ms)
        echo "type=$T M=$M N=$1 K=$2 qmv_c8_ms=$q qgemm_ms=$g"
    done
done
