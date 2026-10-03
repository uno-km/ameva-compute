#!/data/data/com.termux/files/usr/bin/bash
# llm_parity_run.sh MODEL.gguf TAG [ameva_llm_parity args...]
# Runs the CPU-vs-AMEVA parity on one model; full log in ~/abi-lab/results/parity_TAG.log, summary on stdout.
# OUT selects the build directory holding libggml-ameva.so, libac_runtime.so.2 and ameva_llm_parity
# (default ~/abi-lab/out-g2, the graph engine; ~/abi-lab/out-qmm holds the earlier MUL_MAT-only engine).
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
M=$1
TAG=$2
shift 2
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
mkdir -p "$LAB/results"
LOG=$LAB/results/parity_$TAG.log
t0=$(date +%s)
AMEVA_PROFILE=1 LD_LIBRARY_PATH="$RT" timeout 5400 "$OUT/ameva_llm_parity" --model "$M" --plugin "$OUT/libggml-ameva.so" \
    --text "$LAB/corpus/wiki.test.raw" "$@" > "$LOG" 2>&1 < /dev/null
rc=$?
echo "=== $TAG rc=$rc secs=$(( $(date +%s) - t0 ))"
grep -v "^WARNING: linker" "$LOG" | grep -E "^AMEVA_LLM_PARITY|^tensor_types|^run=|^gen |AMEVA_PROFILE|ggml-ameva|graph splits|FAIL|error|Error" | cut -c1-400
