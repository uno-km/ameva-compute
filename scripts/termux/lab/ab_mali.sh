#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# ab_mali.sh TAG [ROUNDS]
# Same-session comparison of the token-generation GEMV on a Mali device, Qwen2.5-0.5B Q4_K_M:
# ameva_qmv (AMEVA_MALI_QMV=0) against mali_qmv (AMEVA_MALI_QMV=1), the two alternating within each round, with a
# CPU run per round as the reference; then the GPU time per MUL_MAT shape of each kernel (AMEVA_PROFILE=2) and a
# single-MUL_MAT timing per shape (ameva_mm_bench, mean of 50). Build: OUT (default ~/abi-lab/out-a1) from SRC
# (default ~/abi-lab/src-a1). Start with nohup; summary in ~/abi-lab/results/abm_TAG.txt, ending with ABM_DONE.
set -u
LAB=$HOME/abi-lab
TAG=${1:?tag}
R=${2:-3}
OUT=${OUT:-$LAB/out-a1}
SRC=${SRC:-$LAB/src-a1}
M=$HOME/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
DB=$SRC/scripts/termux/lab/decode_bench.sh
S=$LAB/results/abm_$TAG.txt
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
GLIB=/data/data/com.termux/files/usr/lib
[ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
cd "$LAB" || exit 1
: > "$S"
tps() { grep "       eval time" "$1" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/'; }
gms() { grep "AMEVA_PROFILE per_graph_ms" "$1" | sed -E 's/.*gpu=([0-9.]+).*/\1/'; }
echo "device $(getprop ro.product.model) $(grep Cpus_allowed_list /proc/self/status | tr -s '\t' ' ')" >> "$S"
for r in $(seq 1 "$R"); do
    for k in 0 1; do
        AMEVA_MALI_QMV=$k AMEVA_PROFILE=1 OUT="$OUT" bash "$DB" "$M" "abm_${TAG}_k${k}_$r" gpu > /dev/null 2>&1
        f=$LAB/results/bench_abm_${TAG}_k${k}_$r.log
        echo "round=$r mali_qmv=$k decode_tps=$(tps "$f") gpu_ms_per_graph=$(gms "$f")" >> "$S"
    done
    OUT="$OUT" bash "$DB" "$M" "abm_${TAG}_cpu_$r" cpu > /dev/null 2>&1
    echo "round=$r cpu decode_tps=$(tps "$LAB/results/bench_abm_${TAG}_cpu_$r.log")" >> "$S"
done
for k in 0 1; do
    AMEVA_MALI_QMV=$k AMEVA_PROFILE=2 OUT="$OUT" bash "$DB" "$M" "abm_${TAG}_prof$k" gpu > /dev/null 2>&1
    echo "--- GPU time per MUL_MAT shape (token generation), mali_qmv=$k" >> "$S"
    grep 'kind="MUL_MAT .*M=1' "$LAB/results/bench_abm_${TAG}_prof$k.log" | sed -E 's/AMEVA_PROFILE //' >> "$S"
    grep -E 'kind="(FLASH_ATTN_EXT|ADD|RMS_NORM)"' "$LAB/results/bench_abm_${TAG}_prof$k.log" | sed -E 's/AMEVA_PROFILE //' >> "$S"
done
B=$OUT/ameva_mm_bench
if [ ! -x "$B" ]; then
    clang++ -O2 -std=c++17 -I "$LAB/ggml-b8471/include" -I "$SRC/adapters/llama/ggml-ameva" "$SRC/tests/ameva_mm_bench.cpp" \
        "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" -Wl,-rpath,"$OUT" -o "$B" 2>/dev/null
fi
echo "--- single MUL_MAT, one activation row (ameva_mm_bench, mean of 50 graphs incl. submit/wait)" >> "$S"
for shape in "q5_0 4864 896" "q5_0 896 896" "q5_0 128 896" "q4_K 896 4864" "q6_K 896 4864" "q8_0 151936 896"; do
    set -- $shape
    for k in 0 1; do
        line=$(AMEVA_MALI_QMV=$k LD_LIBRARY_PATH="$RT" "$B" "$1" 1 "$2" "$3" 50 2>&1 | grep "path=qmv")
        echo "type=$1 N=$2 K=$3 mali_qmv=$k $(printf '%s' "$line" | grep -oE 'ms=[0-9.]+ weight_GBps=[0-9.]+')" >> "$S"
    done
done
echo ABM_DONE >> "$S"
