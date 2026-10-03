#!/data/data/com.termux/files/usr/bin/bash
# tune_verify.sh TAG [ROUNDS]
# Verification of the token-generation GEMV layout selection (ggml-ameva.cpp) from a source tree synced to
# ~/abi-lab/src-TAG, built into ~/abi-lab/out-TAG. Start with nohup; summary in ~/abi-lab/results/tunev_TAG.txt,
# ending with TUNEV_DONE. Measurements go to files under ~/abi-lab/results (AMEVA_QMV_TUNE_FILE), not ~/.cache.
#   1. g2_build.sh: kernel header (spirv-val), plugin, ameva_qmm/kvq/chunk/fuse checks (layouts measured), batch KATs
#   2. ameva_qmm_check with every layout fixed (ameva_qmv rows x tpr, mali_qmv); test-backend-ops MUL_MAT with
#      AMEVA_REPACK_ALL=1 and measured layouts (every one-row shape is measured, nothing kept)
#   3. Qwen2.5-0.5B Q4_K_M: first run (measures, writes the tune file), second run (reads it)
#   4. ROUNDS rounds (default 3), each: measured layouts, default ameva_qmv (rows 4, tpr 16), mali_qmv, the earlier
#      build ~/abi-lab/out-a1 with ameva_qmv when present (macro-sized kernel: no specialization constant), CPU
#   5. AMEVA_PROFILE=2: GPU time per MUL_MAT shape with the measured layouts
#   6. the greedy text of every mode of round 1
set -u
TAG=${1:?tag}
ROUNDS=${2:-3}
LAB=$HOME/abi-lab
export SRC=$LAB/src-$TAG
OUT=$LAB/out-$TAG
S=$LAB/results/tunev_$TAG.txt
M=$HOME/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
DB=$SRC/scripts/termux/lab/decode_bench.sh
TF=$LAB/results/qmv_tune_$TAG.txt
mkdir -p "$LAB/results"
cd "$LAB" || exit 1
: > "$S"
say() { echo "$*" >> "$S"; }
tps() { grep "       eval time" "$1" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/'; }
gms() { grep "AMEVA_PROFILE per_graph_ms" "$1" | sed -E 's/.*gpu=([0-9.]+).*/\1/'; }
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
say "device $(getprop ro.product.model) android $(getprop ro.build.version.release)" \
    "$(grep Cpus_allowed_list /proc/self/status | tr -s '\t' ' ')" \
    "other llama/whisper/test processes at start: $(ps -A -o comm 2>/dev/null | grep -cE 'llama|whisper|test-backend|ameva_')"

AMEVA_QMV_TUNE_FILE=$LAB/results/qmv_tune_${TAG}_checks.txt bash "$SRC/scripts/termux/lab/g2_build.sh" "$OUT" >> "$S" 2>&1
if [ ! -f "$OUT/libggml-ameva.so" ] || [ ! -x "$OUT/ameva_qmm_check" ]; then
    say "build failed"
    say TUNEV_DONE
    exit 1
fi

say "--- ameva_qmm_check, every token-generation layout fixed"
for rows in 4 2 1; do
    for tpr in 8 16 32 64; do
        line=$(AMEVA_QMV_KERNEL=qmv AMEVA_QMV_ROWS=$rows AMEVA_QMV_TPR=$tpr LD_LIBRARY_PATH="$RT" timeout 1800 \
            "$OUT/ameva_qmm_check" 2>&1 | tail -1)
        say "qmv rows=$rows tpr=$tpr: $line"
    done
done
say "mali_qmv: $(AMEVA_QMV_KERNEL=mali LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_qmm_check" 2>&1 | tail -1)"
if [ -x "$LAB/tbo/test-backend-ops" ]; then
    log=$LAB/results/tunev_${TAG}_tbo_mul_mat.log
    AMEVA_QMV_TUNE_FILE=0 AMEVA_REPACK_ALL=1 GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" \
        timeout 3600 "$LAB/tbo/test-backend-ops" -o MUL_MAT -b AMEVA0 > "$log" 2>&1
    rc=$?
    say "test-backend-ops MUL_MAT (measured layouts, AMEVA_REPACK_ALL=1) rc=$rc" \
        "$(tr -d '\033' < "$log" | grep -E 'tests passed' | tr -s ' ')" \
        "shapes measured: $(grep -c 'GEMV layout for' "$log"), layouts left out: $(grep -c 'left out' "$log")"
    tr -d '\033' < "$log" | grep -E "FAIL|WARNING" | grep -v "not supported" | head -5 >> "$S"
else
    say "test-backend-ops: not on this device"
fi

[ -f "$M" ] || { say "no model $M"; say TUNEV_DONE; exit 1; }
export AMEVA_QMV_TUNE_FILE=$TF
rm -f "$TF"
for run in first second; do
    AMEVA_PROFILE=1 OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_$run" gpu > /dev/null 2>&1
    f=$LAB/results/bench_tunev_${TAG}_$run.log
    say "--- $run run: decode_tps=$(tps "$f") shapes measured: $(grep -c 'GEMV layout for' "$f")" \
        "$(grep -E 'GEMV layouts of' "$f")"
    grep -E "GEMV layout for|WARNING|FAIL" "$f" >> "$S"
    grep "AMEVA_PROFILE qmv_tune" "$f" | sed 's/AMEVA_PROFILE //' >> "$S"
done
say "--- tune file"
cat "$TF" >> "$S" 2>&1

for r in $(seq 1 "$ROUNDS"); do
    for mode in tuned qmv mali a1 cpu; do
        f=$LAB/results/bench_tunev_${TAG}_${mode}_$r.log
        case $mode in
            tuned) AMEVA_PROFILE=1 OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_${mode}_$r" gpu > /dev/null 2>&1 ;;
            qmv) AMEVA_QMV_KERNEL=qmv AMEVA_PROFILE=1 OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_${mode}_$r" gpu \
                     > /dev/null 2>&1 ;;
            mali) AMEVA_QMV_KERNEL=mali AMEVA_PROFILE=1 OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_${mode}_$r" gpu \
                      > /dev/null 2>&1 ;;
            a1) [ -f "$LAB/out-a1/libggml-ameva.so" ] || continue
                AMEVA_MALI_QMV=0 AMEVA_PROFILE=1 OUT="$LAB/out-a1" bash "$DB" "$M" "tunev_${TAG}_${mode}_$r" gpu \
                    > /dev/null 2>&1 ;;
            cpu) OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_${mode}_$r" cpu > /dev/null 2>&1 ;;
        esac
        say "round=$r mode=$mode decode_tps=$(tps "$f") gpu_ms_per_graph=$(gms "$f")" \
            "gpu_max_clk=$(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null || echo n/a)"
    done
done

AMEVA_PROFILE=2 OUT="$OUT" bash "$DB" "$M" "tunev_${TAG}_prof" gpu > /dev/null 2>&1
f=$LAB/results/bench_tunev_${TAG}_prof.log
say "--- GPU time per MUL_MAT shape (token generation), measured layouts"
grep 'kind="MUL_MAT .*M=1' "$f" | sed 's/AMEVA_PROFILE //' >> "$S"
grep "AMEVA_PROFILE qmv_layout" "$f" | sed 's/AMEVA_PROFILE //' >> "$S"

say "--- greedy text, round 1"
for mode in tuned qmv mali a1 cpu; do
    o=$LAB/results/bench_tunev_${TAG}_${mode}_1.out
    [ -f "$o" ] && say "$mode md5=$(md5sum < "$o" | cut -c1-12) $(tr '\n' ' ' < "$o" | cut -c1-200)"
done
say TUNEV_DONE
