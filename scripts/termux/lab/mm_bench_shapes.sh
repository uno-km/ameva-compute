#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# mm_bench_shapes.sh [ITER]
# ameva_mm_bench (OUT/ameva_mm_bench, default ~/abi-lab/out-g2) over the MUL_MAT shapes of Qwen2.5-0.5B Q4_K_M:
# token generation (M=1) and a 185-token prompt (M=185), ggml layout (ameva_qmm) against repacked (ameva_qmv).
# AMEVA_* variables pass through (e.g. AMEVA_QMV_TPR).
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
IT=${1:-30}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
while read -r type m n k; do
    [ -z "$type" ] && continue
    LD_LIBRARY_PATH="$RT" "$OUT/ameva_mm_bench" "$type" "$m" "$n" "$k" "$IT" 2>&1 | grep -v "^WARNING: linker"
done <<'EOF'
q5_0 1 4864 896
q5_0 185 4864 896
q5_0 1 896 896
q5_0 185 896 896
q5_0 1 128 896
q4_K 1 896 4864
q4_K 185 896 4864
q6_K 1 896 4864
q6_K 185 896 4864
q8_0 1 151936 896
EOF
