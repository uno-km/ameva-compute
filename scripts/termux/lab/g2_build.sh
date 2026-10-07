#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# Build the graph-engine plugin from ~/abi-lab/src into OUT (default ~/abi-lab/out-g2):
# kernels header -> plugin + ac_runtime -> backend checks -> batch KATs. Prints only results.
# Backend checks (each prints one summary line):
#   ameva_qmm_check     MUL_MAT exactness for 15 weight types, every placement and kernel path (also run with
#                       AMEVA_MMQ=0, the f32 tiled kernel, when the default run used ameva_mmq)
#   ameva_kvq_check     quantized KV cache: SET_ROWS / CPY to and from q8_0 / q4_0, FLASH_ATTN_EXT with q8_0 / q4_0 K/V
#   ameva_chunk_check   element-wise ops larger than one binding (row chunks), with AMEVA_BINDING_LIMIT=1048576
#   ameva_fuse_check    graph-engine fusion, run with AMEVA_FUSE=1 and =0: both must pass and print the same hash
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src} # another source tree (e.g. ~/abi-lab/src-<tag>) keeps parallel sessions apart
P=/data/data/com.termux/files/usr
OUT=${1:-$LAB/out-g2}
mkdir -p "$OUT"
GLIB="$P/lib"
[ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
cd "$SRC" || exit 1
python3 scripts/build_ameva_kernels.py > "$OUT/kernels.log" 2>&1 || { echo KERNELS_FAIL; tail -20 "$OUT/kernels.log"; exit 1; }
tail -1 "$OUT/kernels.log"
bash scripts/termux/build_ggml_ameva.sh --ggml-dir "$LAB/ggml-b8471" --ggml-lib "$GLIB" --out "$OUT" > "$OUT/build.log" 2>&1
grep -E "\[build_ggml_ameva\]|error|warning" "$OUT/build.log" | grep -v "^WARNING: linker" | head -30
[ -f "$OUT/libggml-ameva.so" ] || { echo BUILD_FAIL; exit 1; }
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
for t in ameva_qmm_check ameva_kvq_check ameva_chunk_check ameva_fuse_check; do
    clang++ -O2 -std=c++17 -I "$LAB/ggml-b8471/include" -I "$SRC/adapters/llama/ggml-ameva" \
        "$SRC/tests/$t.cpp" "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" \
        -Wl,-rpath,"$OUT" -o "$OUT/$t" || { echo "TEST_BUILD_FAIL $t"; exit 1; }
done
qmm=$(LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_qmm_check" 2>&1 | tail -1)
echo "$qmm"
case "$qmm" in
    *", 0 through ameva_mmq"*) ;;
    *) echo "AMEVA_MMQ=0: $(AMEVA_MMQ=0 LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_qmm_check" 2>&1 | tail -1)" ;;
esac
LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_kvq_check" 2>&1 | tail -1
AMEVA_BINDING_LIMIT=1048576 LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_chunk_check" 2>&1 | tail -1
f1=$(AMEVA_FUSE=1 LD_LIBRARY_PATH="$RT" timeout 300 "$OUT/ameva_fuse_check" 2>&1)
f0=$(AMEVA_FUSE=0 LD_LIBRARY_PATH="$RT" timeout 300 "$OUT/ameva_fuse_check" 2>&1)
h1=$(printf '%s\n' "$f1" | grep -o 'FUSE_HASH=[0-9a-f]*')
h0=$(printf '%s\n' "$f0" | grep -o 'FUSE_HASH=[0-9a-f]*')
echo "fuse on:  $(printf '%s\n' "$f1" | tail -1) $h1"
echo "fuse off: $(printf '%s\n' "$f0" | tail -1) $h0"
[ -n "$h1" ] && [ "$h1" = "$h0" ] && echo "AMEVA_FUSE_BITEXACT=PASS" || echo "AMEVA_FUSE_BITEXACT=FAIL"
for t in test_batch_api_kat test_batch_graph_kat; do
    clang++ -O2 -std=c++17 -I "$SRC/include" "$SRC/tests/$t.cpp" "$OUT/libac_runtime.so.2" -lvulkan \
        -Wl,-rpath,"$OUT" -o "$OUT/$t" || { echo "BUILD_FAIL $t"; exit 1; }
done
LD_LIBRARY_PATH="$OUT" timeout 300 "$OUT/test_batch_api_kat" | grep -cE "PASS|REJECTED" | sed 's/^/batch_api_kat lines passed: /'
LD_LIBRARY_PATH="$OUT" timeout 300 "$OUT/test_batch_graph_kat" | tail -1
