#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tbo_paths.sh TAG
# test-backend-ops for AMEVA0 with the build in OUT (default ~/abi-lab/out-g2) under several settings, so that the paths
# for large tensors and long KV caches run with the small shapes of the test suite:
#   default                     every op
#   AMEVA_REPACK_ALL=1          quantized MUL_MAT through the repacked-weight kernels (ameva_mmq for prompt-sized
#                               cases when the device offers integer dot products)
#   AMEVA_FA_SPLIT_KEYS=64      FLASH_ATTN_EXT with key splits and the combine kernel wherever tiles alone are few
#   AMEVA_REPACK_ALL=1 AMEVA_MMQ=0  the same MUL_MAT cases with the f32 tiled kernel instead of ameva_mmq
#   AMEVA_BINDING_LIMIT=1048576 weight row slices, activation row chunks and row-chunked unary ops (MUL_MAT,
#                               TANH, SCALE; test-backend-ops names unary ops after the function, so -o UNARY
#                               selects nothing)
# Start with nohup. Logs: ~/abi-lab/results/tbop_TAG_<setting>.log; summary tbop_TAG.txt, ending with TBOP_DONE.
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
TAG=${1:-run}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
cd "$LAB" || exit 1
S=results/tbop_$TAG.txt
: > "$S"
run() { # name env-assignment op-filter...
    local name=$1 envs=$2
    shift 2
    local log=results/tbop_${TAG}_$name.log
    env $envs GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 \
        tbo/test-backend-ops "$@" -b AMEVA0 > "$log" 2>&1
    local rc=$?
    echo "== $name rc=$rc $(tr -d '\033' < "$log" | grep -E 'tests passed' | tr -s ' ')" >> "$S"
    tr -d '\033' < "$log" | grep FAIL | grep -v "not supported" | head -5 >> "$S"
}
run default "AMEVA_PROFILE="
run repack_all "AMEVA_REPACK_ALL=1"
run fa_split64 "AMEVA_FA_SPLIT_KEYS=64" -o FLASH_ATTN_EXT
run mmq_off "AMEVA_REPACK_ALL=1 AMEVA_MMQ=0" -o MUL_MAT
run bind1m_mm "AMEVA_BINDING_LIMIT=1048576" -o MUL_MAT
run bind1m_tanh "AMEVA_BINDING_LIMIT=1048576" -o TANH
run bind1m_scale "AMEVA_BINDING_LIMIT=1048576" -o SCALE
echo TBOP_DONE >> "$S"
