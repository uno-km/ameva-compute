#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# fa_bench.sh "SHAPE:IDX,IDX ..." [ITER]
# Build the ggml-ameva plugin from SRC (default ~/abi-lab/src-fa) into OUT (default ~/abi-lab/out-fa) and
# tests/ameva_fa_bench, then time FLASH_ATTN_EXT at each shape with each forced variant (AMEVA_FA_VARIANT) and with the
# engine's own choice. SHAPE is D,NQ,NKV,H. A kernel header generated with AMEVA_FA_VARIANTS lists the candidates.
# SKIP_PLUGIN_BUILD=1 reuses OUT/libggml-ameva.so. GGML_HEADERS / GLIB as in conv_check.sh. Prints the bench lines.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src-fa}
OUT=${OUT:-$LAB/out-fa}
GH=${GGML_HEADERS:-$LAB/ggml-b8471}
IT=${2:-5}
if [ -z "${GLIB:-}" ]; then
    for d in "$HOME"/.local/share/ameva/releases/llamacpp/*/lib; do
        [ -e "$d/libggml-base.so.0" ] && GLIB="$d"
    done
    [ -z "${GLIB:-}" ] && [ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
    [ -z "${GLIB:-}" ] && GLIB="${PREFIX:-/data/data/com.termux/files/usr}/lib"
fi
mkdir -p "$OUT"
cd "$SRC" || exit 1
if [ "${SKIP_PLUGIN_BUILD:-0}" != 1 ] || [ ! -f "$OUT/libggml-ameva.so" ]; then
    bash scripts/termux/build_ggml_ameva.sh --ggml-dir "$GH" --ggml-lib "$GLIB" --out "$OUT" > "$OUT/build.log" 2>&1
    [ -f "$OUT/libggml-ameva.so" ] || { echo BUILD_FAIL; tail -20 "$OUT/build.log"; exit 1; }
fi
clang++ -O2 -std=c++17 -I "$GH/include" -I "$SRC/adapters/llama/ggml-ameva" "$SRC/tests/ameva_fa_bench.cpp" \
    "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" -Wl,-rpath,"$OUT" -o "$OUT/ameva_fa_bench" \
    || { echo TEST_BUILD_FAIL; exit 1; }
for item in $1; do
    shape=${item%%:*}; idx=${item#*:}
    set -- $(echo "$shape" | tr ',' ' ')
    "$OUT/ameva_fa_bench" "$1" "$2" "$3" "$4" "$IT" 2>&1 | grep -E "FA_BENCH|abort|Abort|error"
    for i in $(echo "$idx" | tr ',' ' '); do
        AMEVA_FA_VARIANT=$i "$OUT/ameva_fa_bench" "$1" "$2" "$3" "$4" "$IT" 2>&1 | grep -E "FA_BENCH|abort|Abort|error"
    done
done
