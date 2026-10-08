#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# k3_check.sh [kernels]
# Build the ggml-ameva plugin from SRC (default ~/abi-lab/src) into OUT (default ~/abi-lab/out-k3) against the ggml of
# llama.cpp b8471 and run
#   tests/ameva_k3_check         UPSCALE, PAD, CONCAT, SOFT_MAX and f32 MUL_MAT against references of ggml's CPU code
#   tests/ameva_vae_graph_check  op census of sd.cpp's 2D VAE decoder graphs and their output against the CPU backend
#   ~/abi-lab/tbo/test-backend-ops -b AMEVA0 for the ops of TBO_OPS (default: the K3 ops and the two of the earlier
#   steps; TBO_OPS=all runs every op, TBO_OPS=none skips it)
# kernels    regenerate ggml_ameva_kernels_spv.h / _cl.h first (scripts/build_ameva_kernels.py with spirv-val)
# SKIP_PLUGIN_BUILD=1 keeps an existing OUT/libggml-ameva.so and rebuilds only the tests.
# BACKENDS="vulkan opencl" runs ameva_k3_check and test-backend-ops once per AMEVA_BACKEND (default: the engine's own
# choice, once); the VAE graphs need the convolution kernels, which exist as SPIR-V only.
# LATENT=WxH is passed to ameva_vae_graph_check as AMEVA_VAE_CHECK_LATENT. GGML_HEADERS and GLIB as in conv_check.sh.
# Prints results only; exit status 0 when everything that ran passed.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src}
OUT=${OUT:-$LAB/out-k3}
GH=${GGML_HEADERS:-$LAB/ggml-b8471}
if [ -z "${GLIB:-}" ]; then
    for d in "$HOME"/.local/share/ameva/releases/llamacpp/*/lib; do
        [ -e "$d/libggml-base.so.0" ] && GLIB="$d"
    done
    [ -z "${GLIB:-}" ] && [ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
    [ -z "${GLIB:-}" ] && GLIB="${PREFIX:-/data/data/com.termux/files/usr}/lib"
fi
mkdir -p "$OUT"
cd "$SRC" || exit 1
echo "ggml headers: $GH"
echo "ggml libraries: $GLIB"
if [ "${SKIP_PLUGIN_BUILD:-0}" = 1 ] && [ -f "$OUT/libggml-ameva.so" ]; then
    echo "plugin build skipped (SKIP_PLUGIN_BUILD=1): $OUT/libggml-ameva.so of $(date -r "$OUT/libggml-ameva.so" +%F_%T)"
else
    if [ "${1:-}" = kernels ]; then
        python3 scripts/build_ameva_kernels.py > "$OUT/kernels.log" 2>&1 || { echo KERNELS_FAIL; tail -20 "$OUT/kernels.log"; exit 1; }
        tail -1 "$OUT/kernels.log"
    fi
    bash scripts/termux/build_ggml_ameva.sh --ggml-dir "$GH" --ggml-lib "$GLIB" --out "$OUT" > "$OUT/build.log" 2>&1
    grep -E "\[build_ggml_ameva\]|error|warning" "$OUT/build.log" | grep -v "^WARNING: linker" | head -30
    [ -f "$OUT/libggml-ameva.so" ] || { echo BUILD_FAIL; exit 1; }
fi
for t in ameva_k3_check ameva_vae_graph_check; do
    clang++ -O2 -std=c++17 -I "$GH/include" -I "$SRC/adapters/llama/ggml-ameva" \
        "$SRC/tests/$t.cpp" "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" \
        -Wl,-rpath,"$OUT" -o "$OUT/$t" || { echo "TEST_BUILD_FAIL $t"; exit 1; }
done
# library search as in conv_check.sh: OUT first, then the ggml libraries alone
if [ ! -d "$OUT/ggml-rt" ]; then
    mkdir -p "$OUT/ggml-rt"
    for n in libggml-base libggml libggml-cpu libggml-vulkan libggml-opencl; do
        if [ -e "$GLIB/$n.so.0" ]; then ln -s "$(readlink -f "$GLIB/$n.so.0")" "$OUT/ggml-rt/$n.so.0"; fi
    done
fi
RT="$OUT:$OUT/ggml-rt"
CPU_LIB=
[ -e "$GLIB/libggml-cpu.so" ] && CPU_LIB="$(readlink -f "$GLIB/libggml-cpu.so")"
TBO_OPS=${TBO_OPS:-"UPSCALE PAD CONCAT SOFT_MAX MUL_MAT CONV_2D GROUP_NORM"}
rc=0
echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
for b in ${BACKENDS:-default}; do
    sel=
    [ "$b" = default ] || sel="AMEVA_BACKEND=$b"
    echo "== ameva_k3_check backend=$b"
    env $sel LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_k3_check" > "$OUT/k3_check_$b.log" 2>&1
    r=$?
    grep -E "FAIL|K3_CHECK [0-9]+/" "$OUT/k3_check_$b.log" | grep -v "^WARNING: linker"
    grep -o "op=[A-Z_]* case=.* PASS" "$OUT/k3_check_$b.log" | awk '{ n[$1]++ } END { for (k in n) printf "%s passed=%d  ", k, n[k]; print "" }'
    echo "rc=$r"
    [ $r -eq 0 ] || rc=1
    if [ "$TBO_OPS" != none ] && [ -x "$LAB/tbo/test-backend-ops" ]; then
        if [ "$TBO_OPS" = all ]; then
            echo "== test-backend-ops backend=$b, every op"
            env $sel GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 \
                "$LAB/tbo/test-backend-ops" -b AMEVA0 > "$OUT/tbo_${b}_all.txt" 2>&1
            r=$?
            grep -E "tests passed|Backend AMEVA0|FAIL" "$OUT/tbo_${b}_all.txt" | tail -4
            echo "not supported: $(grep -c "not supported" "$OUT/tbo_${b}_all.txt") rc=$r"
            [ $r -eq 0 ] || rc=1
        else
            for op in $TBO_OPS; do
                env $sel GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 1800 \
                    "$LAB/tbo/test-backend-ops" -b AMEVA0 -o "$op" > "$OUT/tbo_${b}_$op.txt" 2>&1
                r=$?
                echo "== test-backend-ops backend=$b -o $op: $(grep -E "tests passed" "$OUT/tbo_${b}_$op.txt" | tail -1)" \
                     "not supported: $(grep -c "not supported" "$OUT/tbo_${b}_$op.txt") failed: $(grep -c "FAIL" "$OUT/tbo_${b}_$op.txt") rc=$r"
                [ $r -eq 0 ] || rc=1
            done
        fi
    fi
done
echo "== ameva_vae_graph_check latent=${LATENT:-16x16}"
env ${LATENT:+AMEVA_VAE_CHECK_LATENT=$LATENT} ${CPU_LIB:+AMEVA_CPU_BACKEND_LIB=$CPU_LIB} LD_LIBRARY_PATH="$RT" \
    timeout 3600 "$OUT/ameva_vae_graph_check" > "$OUT/vae_graph_check.log" 2>&1
r=$?
grep -E "^VAE_CENSUS .*(ameva=NO|ameva=PARTLY|declined_nodes)|^VAE_CHECK|FAIL-FAST|declined" "$OUT/vae_graph_check.log" | grep -v "^WARNING: linker"
echo "rc=$r"
[ $r -eq 0 ] || rc=1
exit $rc
