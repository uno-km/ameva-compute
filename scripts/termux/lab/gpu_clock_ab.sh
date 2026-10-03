#!/data/data/com.termux/files/usr/bin/bash
# gpu_clock_ab.sh TAG [ROUNDS]
# Token-generation speed of the GEMV layouts together with the GPU clock the governor chose during each run, on
# Samsung Exynos devices that expose /sys/kernel/gpu (gpu_clock in kHz, gpu_busy, gpu_tmu). Per round the three modes
# run in rotating order: measured layouts (tune file of tune_verify.sh TAG), default ameva_qmv, mali_qmv.
# A sampler reads the clock every 50 ms while llama-completion runs; reported per run: decode speed, GPU time per
# graph, mean clock and share of samples at the maximum clock over the samples with a busy GPU, temperature.
# Build ~/abi-lab/out-TAG from ~/abi-lab/src-TAG. Summary: ~/abi-lab/results/clk_TAG.txt, ending with CLK_DONE.
set -u
TAG=${1:?tag}
R=${2:-8}
LAB=$HOME/abi-lab
SRC=$LAB/src-$TAG
OUT=$LAB/out-$TAG
M=$HOME/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
S=$LAB/results/clk_$TAG.txt
G=/sys/kernel/gpu
export AMEVA_QMV_TUNE_FILE=$LAB/results/qmv_tune_$TAG.txt
cd "$LAB" || exit 1
: > "$S"
[ -r "$G/gpu_clock" ] || { echo "no readable $G/gpu_clock" >> "$S"; echo CLK_DONE >> "$S"; exit 1; }
MAXC=$(cat "$G/gpu_max_clock")
echo "device $(getprop ro.product.model) governor=$(cat $G/gpu_governor) max_clock=$MAXC table=$(cat $G/gpu_freq_table)" >> "$S"
run() {
    mode=$1
    r=$2
    tag=clk_${TAG}_${mode}_$r
    smp=$LAB/results/$tag.clk
    : > "$smp"
    (while :; do echo "$(cat $G/gpu_clock) $(cat $G/gpu_busy | tr -d ' %')"; sleep 0.05; done) >> "$smp" 2> /dev/null &
    sp=$!
    t0=$(cat $G/gpu_tmu)
    case $mode in
        tuned) AMEVA_PROFILE=1 OUT="$OUT" bash "$SRC/scripts/termux/lab/decode_bench.sh" "$M" "$tag" gpu > /dev/null 2>&1 ;;
        qmv) AMEVA_QMV_KERNEL=qmv AMEVA_PROFILE=1 OUT="$OUT" bash "$SRC/scripts/termux/lab/decode_bench.sh" "$M" "$tag" gpu > /dev/null 2>&1 ;;
        mali) AMEVA_QMV_KERNEL=mali AMEVA_PROFILE=1 OUT="$OUT" bash "$SRC/scripts/termux/lab/decode_bench.sh" "$M" "$tag" gpu > /dev/null 2>&1 ;;
    esac
    kill $sp 2> /dev/null
    wait $sp 2> /dev/null
    f=$LAB/results/bench_$tag.log
    stats=$(awk -v m="$MAXC" '$2 > 0 { n++; s += $1; if ($1 >= m) x++ } END { if (n) printf "busy_samples=%d mean_clock=%d at_max=%.0f%%", n, s / n, 100 * x / n; else printf "busy_samples=0" }' "$smp")
    echo "round=$r mode=$mode decode_tps=$(grep '       eval time' "$f" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')" \
         "gpu_ms_per_graph=$(grep 'AMEVA_PROFILE per_graph_ms' "$f" | sed -E 's/.*gpu=([0-9.]+).*/\1/') $stats tmu=$t0->$(cat $G/gpu_tmu)" >> "$S"
}
for r in $(seq 1 "$R"); do
    case $((r % 3)) in
        1) order="tuned qmv mali" ;;
        2) order="qmv mali tuned" ;;
        0) order="mali tuned qmv" ;;
    esac
    for mode in $order; do run "$mode" "$r"; done
done
echo CLK_DONE >> "$S"
