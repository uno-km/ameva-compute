#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# Build the plugin from ~/abi-lab/src (QMM engine) against llama.cpp b8471 ggml and run the exactness check.
set -u
LAB=$HOME/abi-lab
SRC=$LAB/src
P=/data/data/com.termux/files/usr
OUT=$LAB/out-qmm
mkdir -p "$OUT"
GLIB="$P/lib"
[ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
bash "$SRC/scripts/termux/build_ggml_ameva.sh" --ggml-dir "$LAB/ggml-b8471" --ggml-lib "$GLIB" --out "$OUT" 2>&1 \
    | grep -E "\[build_ggml_ameva\]|error" || { echo BUILD_FAIL; exit 1; }
clang++ -O2 -std=c++17 -I "$LAB/ggml-b8471/include" -I "$SRC/adapters/llama/ggml-ameva" \
    "$SRC/tests/ameva_qmm_check.cpp" "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" \
    -Wl,-rpath,"$OUT" -o "$OUT/ameva_qmm_check" || { echo TEST_BUILD_FAIL; exit 1; }
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_qmm_check"
echo "rc=$?"
