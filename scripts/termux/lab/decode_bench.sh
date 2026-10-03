#!/data/data/com.termux/files/usr/bin/bash
# decode_bench.sh MODEL.gguf TAG MODE [N_DECODE]
# Token generation speed with llama-completion (prompt "The three primary colors are", greedy, FA on).
#   MODE gpu  AMEVA0 with the graph engine in OUT (default ~/abi-lab/out-g2); AMEVA_* variables pass through
#        cpu  CPU only (--device none)
# AMEVA_PROFILE is left as given (unset: no profiling overhead). THREADS (default 4) and POLL (default: llama.cpp's)
# set -t and --poll, CTX (default 4096) -c. Log: ~/abi-lab/results/bench_TAG.log
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
M=$1
TAG=$2
MODE=$3
ND=${4:-64}
THREADS=${THREADS:-4}
EXTRA=()
[ -n "${POLL:-}" ] && EXTRA=(--poll "$POLL")
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
mkdir -p "$LAB/results"
if [ "$MODE" = cpu ]; then
    DEV=(--device none -ngl 0)
else
    DEV=(--device AMEVA0 -ngl 99)
fi
GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 900 llama-completion \
    -m "$M" "${DEV[@]}" --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" "${EXTRA[@]}" \
    -p "The three primary colors are" -n "$ND" -c "${CTX:-4096}" \
    > "$LAB/results/bench_$TAG.out" 2> "$LAB/results/bench_$TAG.log" < /dev/null
rc=$?
# Adreno exposes its current thermal cap (max_gpuclk); a throttled run is not comparable with an unthrottled one
CLK=$(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null || echo n/a)
echo "=== $TAG ($MODE) rc=$rc $(grep Cpus_allowed_list /proc/self/status | tr -s '[:space:]' ' ')gpu_max_clk=$CLK"
grep -E "prompt eval time|       eval time|AMEVA_PROFILE (graphs|per_graph|repack)" "$LAB/results/bench_$TAG.log"
