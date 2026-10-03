#!/data/data/com.termux/files/usr/bin/bash
# Build the graph-engine plugin from ~/abi-lab/src into OUT (default ~/abi-lab/out-g2):
# kernels header -> plugin + ac_runtime -> QMM exactness check -> batch KATs. Prints only results.
set -u
LAB=$HOME/abi-lab
SRC=$LAB/src
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
clang++ -O2 -std=c++17 -I "$LAB/ggml-b8471/include" -I "$SRC/adapters/llama/ggml-ameva" \
    "$SRC/tests/ameva_qmm_check.cpp" "$OUT/libggml-ameva.so" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" \
    -Wl,-rpath,"$OUT" -o "$OUT/ameva_qmm_check" || { echo QMM_TEST_BUILD_FAIL; exit 1; }
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
LD_LIBRARY_PATH="$RT" timeout 900 "$OUT/ameva_qmm_check" | tail -1
for t in test_batch_api_kat test_batch_graph_kat; do
    clang++ -O2 -std=c++17 -I "$SRC/include" "$SRC/tests/$t.cpp" "$OUT/libac_runtime.so.2" -lvulkan \
        -Wl,-rpath,"$OUT" -o "$OUT/$t" || { echo "BUILD_FAIL $t"; exit 1; }
done
LD_LIBRARY_PATH="$OUT" timeout 300 "$OUT/test_batch_api_kat" | grep -cE "PASS|REJECTED" | sed 's/^/batch_api_kat lines passed: /'
LD_LIBRARY_PATH="$OUT" timeout 300 "$OUT/test_batch_graph_kat" | tail -1
