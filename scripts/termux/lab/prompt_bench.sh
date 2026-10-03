#!/data/data/com.termux/files/usr/bin/bash
# prompt_bench.sh MODEL.gguf TAG gpu|cpu [BYTES]
# Prompt processing speed with llama-completion: the first BYTES (default 1500, about 370 tokens) of wikitext-2 test
# as one prompt, one generated token, -b/-ub 512, FA on.
#   gpu: AMEVA0 with the graph engine in OUT (default ~/abi-lab/out-g2); AMEVA_* variables pass through
#   cpu: CPU only. THREADS (default 4) sets -t.
# Log: ~/abi-lab/results/pbench_TAG.log
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
M=$1
TAG=$2
MODE=$3
BYTES=${4:-1500}
THREADS=${THREADS:-4}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
mkdir -p "$LAB/results"
P=$LAB/results/prompt_$BYTES.txt
head -c "$BYTES" "$LAB/corpus/wiki.test.raw" > "$P"
if [ "$MODE" = cpu ]; then
    DEV=(--device none -ngl 0)
else
    DEV=(--device AMEVA0 -ngl 99)
fi
GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 1800 llama-completion \
    -m "$M" "${DEV[@]}" --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" \
    -f "$P" -n 1 -c 2048 -b 512 -ub 512 > "$LAB/results/pbench_$TAG.out" 2> "$LAB/results/pbench_$TAG.log" < /dev/null
rc=$?
# Adreno exposes its current thermal cap (max_gpuclk); a throttled run is not comparable with an unthrottled one
CLK=$(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null || echo n/a)
echo "=== $TAG ($MODE) rc=$rc $(grep Cpus_allowed_list /proc/self/status | tr -s '[:space:]' ' ')gpu_max_clk=$CLK"
grep -E "prompt eval time|AMEVA_PROFILE (graphs|per_graph|repack)" "$LAB/results/pbench_$TAG.log"
