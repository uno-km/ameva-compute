#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# conv_check.sh [kernels]
# Build the ggml-ameva plugin from SRC (default ~/abi-lab/src) into OUT (default ~/abi-lab/out-convk) against the
# ggml of llama.cpp b8471 and run tests/ameva_conv_check (CONV_2D through the backend, compared with an f32 CPU
# convolution) once per kernel selection: staging auto / vec4 / scalar with the f16 decoder the device check picks,
# and vec4 / scalar with the exact decoder, and once with AMEVA_FUSE=0 (the bias of a convolution as a node of its
# own instead of inside the kernel). Then tests/ameva_cnorm_check (GROUP_NORM alone and with the weight, bias and SILU
# nodes fused into it, and one decoder block) with and without fusion, and tests/ameva_qmm_check as a regression of
# the MUL_MAT paths.
#   kernels   regenerate ggml_ameva_kernels_spv.h / _cl.h first (scripts/build_ameva_kernels.py with spirv-val)
# TIME_SHAPES adds a timing run (tests/ameva_conv_check.cpp, AMEVA_CONV_CHECK_TIME), CNORM_TIME_SHAPES one of the
# group norm (tests/ameva_cnorm_check.cpp, AMEVA_CNORM_CHECK_TIME).
# SKIP_PLUGIN_BUILD=1 keeps an existing OUT/libggml-ameva.so and rebuilds only the tests.
# GGML_HEADERS (default ~/abi-lab/ggml-b8471) and GLIB (default: the lib directory of the installed ameva-runtime
# llama bundle, else ~/abi-lab/rt098, else $PREFIX/lib) select the ggml the plugin is built for. Prints results only.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src}
OUT=${OUT:-$LAB/out-convk}
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
for t in ameva_conv_check ameva_cnorm_check ameva_qmm_check; do
    clang++ -O2 -std=c++17 -I "$GH/include" -I "$SRC/adapters/llama/ggml-ameva" \
        "$SRC/tests/$t.cpp" "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" \
        -Wl,-rpath,"$OUT" -o "$OUT/$t" || { echo "TEST_BUILD_FAIL $t"; exit 1; }
done
# Library search of the tests: OUT first, so that the plugin and libac_runtime.so.2 just built are the ones loaded
# (the RUNPATH the Termux linker writes has $PREFIX/lib before $ORIGIN, and both $PREFIX/lib and the bundle's lib
# directory can hold an older libggml-ameva.so: Galaxy A35, 2026-10-08, "cannot locate symbol" in every test). Then
# the ggml libraries alone, in a directory of their own (build_ggml_ameva.sh makes it unless it stops before).
if [ ! -d "$OUT/ggml-rt" ]; then
    mkdir -p "$OUT/ggml-rt"
    for n in libggml-base libggml libggml-cpu libggml-vulkan libggml-opencl; do
        if [ -e "$GLIB/$n.so.0" ]; then ln -s "$(readlink -f "$GLIB/$n.so.0")" "$OUT/ggml-rt/$n.so.0"; fi
    done
fi
RT="$OUT:$OUT/ggml-rt"
rc=0
run() {
    echo "== $*"
    env "$@" LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_conv_check" > "$OUT/conv_check.log" 2>&1
    local r=$?
    grep -E "FAIL|ggml-ameva: f16 convolution|CONV_CHECK [0-9]+/|CONV_TIME" "$OUT/conv_check.log" | grep -v "^WARNING: linker"
    grep -o "equal_to=[a-z]*" "$OUT/conv_check.log" | sort | uniq -c | tr '\n' ' '
    echo "rc=$r"
    [ $r -eq 0 ] || rc=1
}
run AMEVA_CONV_STAGE=auto
run AMEVA_CONV_STAGE=vec4
run AMEVA_CONV_STAGE=scalar
run AMEVA_CONV_F16=exact AMEVA_CONV_STAGE=vec4
run AMEVA_CONV_F16=exact AMEVA_CONV_STAGE=scalar
run AMEVA_FUSE=0
# TIME_SHAPES="W,H,IC,OC,K ...": wall time of graph_compute for those convolutions (f16 kernel), default selection
if [ -n "${TIME_SHAPES:-}" ]; then
    echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    run AMEVA_CONV_CHECK_TIME="$TIME_SHAPES"
fi
# GROUP_NORM: per case the largest error as a fraction of the rounding bound of the test, over its four graphs
run_cnorm() {
    local log="$OUT/cnorm_check_$1.log"
    shift
    echo "== ameva_cnorm_check $*"
    env "$@" LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_cnorm_check" > "$log" 2>&1
    local r=$?
    grep -E "FAIL|CNORM_CHECK [0-9]+/|CNORM_TIME" "$log" | grep -v "^WARNING: linker"
    grep -o -E "(plain|add|silu|fused3)=[0-9.]+ " "$log" | awk -F= '$2 + 0 > m { m = $2 + 0 } END { print "largest err_over_bound=" m + 0 }'
    grep -o -E "fused3_vs_separate_diff=[0-9]+" "$log" | awk -F= '{ n += ($2 > 0) } END { print "cases whose fused SILU differs in bits from the separate node=" n + 0 }'
    echo "rc=$r"
    [ $r -eq 0 ] || rc=1
}
run_cnorm default
run_cnorm fuse0 AMEVA_FUSE=0
if [ -n "${CNORM_TIME_SHAPES:-}" ]; then
    echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    run_cnorm time AMEVA_CNORM_CHECK_TIME="$CNORM_TIME_SHAPES"
fi
echo "== ameva_qmm_check"
LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_qmm_check" 2>&1 | tail -1
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
exit $rc
