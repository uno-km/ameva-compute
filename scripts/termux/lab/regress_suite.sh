#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
# regress_suite.sh <kit dir>: the regression checks of one backend build on this phone (scripts/fleet/regress_fleet.sh
# uploads the kit and starts this with nohup). Results in $OUT (default <kit>/results/<time>): one line per check in
# summary.txt ("CHECK <name> PASS|FAIL|SKIP <detail>"), logs beside it, suite.rc last (0 = every check passed).
#
#   abi    ac_abi_check: accepted caller versions, struct layout and exported functions of libac_runtime.so.2
#          against docs/abi in the kit
#   kat    test_batch_graph_kat (batches, barriers, events, the bounded-memory test of 30000 dispatches)
#   qmm    ameva_qmm_check on the automatic backend (the API this GPU gets by default): 516 MUL_MAT cases
#   llm    llama-completion of the installed bundle with the kit's plugin (GGML_BACKEND_PATH, automatic backend):
#          greedy text of the CPU and of AMEVA0 must be equal; prompt and generation t/s of both are recorded with
#          the core set and the GPU clock; one sampled answer (seed and temperature printed) is kept for reading
#
# MODEL=<gguf> (default: the first of Qwen2.5-0.5B, Llama-3.2-1B, Gemma-2-2B found in ~/.cache/termux-llamacpp/models)
# CHECKS="abi kat qmm llm" selects checks. Nothing is installed or downloaded.
set -u
K=$(cd "${1:?kit dir}" && pwd)
OUT=${OUT:-$K/results/$(date +%Y%m%d-%H%M%S)}
CHECKS=${CHECKS:-abi kat qmm llm}
mkdir -p "$OUT"
S=$OUT/summary.txt
: > "$S"
fails=0
say() { echo "$*" | tee -a "$S"; }
result() { # name status detail
    say "CHECK $1 $2 $3"
    [ "$2" = FAIL ] && fails=$((fails + 1))
    return 0
}
clock() {
    if [ -r /sys/class/kgsl/kgsl-3d0/max_gpuclk ]; then echo "gpu_cap=$(( $(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk) / 1000000 ))MHz"
    elif [ -r /sys/kernel/gpu/gpu_clock ]; then echo "gpu_clk=$(cat /sys/kernel/gpu/gpu_clock)"
    else echo "gpu_clk=n/a"; fi
}
cpus() { awk '/Cpus_allowed_list/ {print $2}' /proc/self/status; }
B=$K/bin
BL=$(readlink -f "$HOME/.local/share/ameva/current/llamacpp" 2> /dev/null)/lib
say "device $(getprop ro.product.model) android $(getprop ro.build.version.release) kit $(head -1 "$K/KIT_ID") cpus=$(cpus) $(clock)"
say "plugin $(sha256sum "$B/libggml-ameva.so" | cut -c1-16) runtime $(sha256sum "$B/libac_runtime.so.2" | cut -c1-16) bundle $(basename "$(dirname "$BL")")"
cd /

for c in $CHECKS; do
    case $c in
    abi)
        LD_LIBRARY_PATH=$B timeout 300 "$B/ac_abi_check" > "$OUT/abi.log" 2>&1
        rc=$?
        grep '^ABI_LAYOUT ' "$OUT/abi.log" > "$OUT/abi_layout.txt"
        readelf --dyn-syms -W "$B/libac_runtime.so.2" | awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $8}' \
            | sed 's/@.*//' | grep '^ac_runtime_' | sort -u > "$OUT/abi_symbols.txt"
        lay=same; diff -q "$K/abi/ac_runtime_layout.txt" "$OUT/abi_layout.txt" > /dev/null || lay=DIFFERENT
        sym=same; diff -q "$K/abi/ac_runtime_symbols.txt" "$OUT/abi_symbols.txt" > /dev/null || sym=DIFFERENT
        line=$(grep -E '^AC_ABI_CHECK=' "$OUT/abi.log")
        if [ $rc -eq 0 ] && [ $lay = same ] && [ $sym = same ]; then result abi PASS "$line layout=$lay symbols=$sym"
        else result abi FAIL "rc=$rc ${line:-no result} layout=$lay symbols=$sym"; fi
        ;;
    kat)
        LD_LIBRARY_PATH=$B timeout 900 "$B/test_batch_graph_kat" > "$OUT/kat.log" 2>&1
        rc=$?
        line=$(grep -E '^AMUDA_BATCH_GRAPH_KAT=' "$OUT/kat.log")
        case "$rc:$line" in 0:AMUDA_BATCH_GRAPH_KAT=PASS*) result kat PASS "$line" ;; *) result kat FAIL "rc=$rc ${line:-no result}" ;; esac
        ;;
    qmm)
        LD_LIBRARY_PATH=$B/ggml-rt:$B AMEVA_QMV_TUNE_FILE=$OUT/qmv-tune.txt AMEVA_PIPELINE_CACHE=$OUT/pc.bin timeout 2400 \
            "$B/ameva_qmm_check" > "$OUT/qmm.out" 2> "$OUT/qmm.err"
        rc=$?
        line=$(tail -1 "$OUT/qmm.out")
        api=vulkan; grep -q "kernels run as OpenCL C" "$OUT/qmm.err" && api=opencl
        case "$rc:$line" in 0:AMEVA_QMM_CHECK=PASS*) result qmm PASS "$line [$api]" ;; *) result qmm FAIL "rc=$rc ${line:-no result} [$api]" ;; esac
        ;;
    llm)
        M=${MODEL:-}
        if [ -z "$M" ]; then
            for f in qwen2.5-0.5b-instruct-q4_k_m.gguf Llama-3.2-1B-Instruct-Q4_K_M.gguf gemma-2-2b-it-Q4_K_M.gguf; do
                [ -f "$HOME/.cache/termux-llamacpp/models/$f" ] && { M=$HOME/.cache/termux-llamacpp/models/$f; break; }
            done
        fi
        LC=$(readlink -f "$PREFIX/bin/llama-completion" 2> /dev/null)
        if [ -z "$M" ] || [ ! -f "$M" ] || [ ! -x "$LC" ]; then result llm SKIP "no model or no installed llama-completion"; continue; fi
        head -c 1500 "$K/src/prompt_corpus.txt" > "$OUT/prompt.txt"
        run() { # label device kind [extra...]
            local label=$1 dev=$2 kind=$3; shift 3
            local d io
            if [ "$dev" = cpu ]; then d=(--device none -ngl 0); else d=(--device AMEVA0 -ngl 99 -fa on); fi
            if [ "$kind" = decode ]; then io=(-p "The three primary colors are" -n 64 -c 2048)
            else io=(-f "$OUT/prompt.txt" -n 1 -c 2048 -b 512 -ub 512); fi
            GGML_BACKEND_PATH=$B/libggml-ameva.so LD_LIBRARY_PATH=$B AMEVA_QMV_TUNE_FILE=$OUT/qmv-tune.txt \
                AMEVA_PIPELINE_CACHE=$OUT/pc.bin timeout 900 "$LC" -m "$M" "${d[@]}" --fit off -no-cnv --seed 42 -t 4 \
                "${io[@]}" "$@" > "$OUT/$label.out" 2> "$OUT/$label.log" < /dev/null
            local rc=$?
            local pe ev
            pe=$(grep -m1 "prompt eval time" "$OUT/$label.log" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
            ev=$(grep -E "_print: +eval time" "$OUT/$label.log" | sed -E 's/.*, +([0-9.]+|inf) tokens per second.*/\1/')
            local api=cpu
            if [ "$dev" != cpu ]; then api=vulkan; grep -q "kernels run as OpenCL C" "$OUT/$label.log" && api=opencl; fi
            say "  run $label rc=$rc api=$api prompt_tps=${pe:-?} gen_tps=${ev:-?} cpus=$(cpus) $(clock)"
            return $rc
        }
        run warm ameva decode --temp 0 --top-k 1 > /dev/null
        run dec_cpu cpu decode --temp 0 --top-k 1; r1=$?
        run dec_gpu ameva decode --temp 0 --top-k 1; r2=$?
        run pp_cpu cpu prompt --temp 0 --top-k 1; r3=$?
        run pp_gpu ameva prompt --temp 0 --top-k 1; r4=$?
        run sampled ameva decode --temp 1.0 --seed 1234 > /dev/null
        say "  sampled (AMEVA0, temp 1.0, seed 1234): $(tail -c 300 "$OUT/sampled.out" | tr '\n' ' ')"
        same=no; cmp -s "$OUT/dec_cpu.out" "$OUT/dec_gpu.out" && same=yes
        if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ] && [ $r4 -eq 0 ] && [ $same = yes ] \
            && ! grep -q -E 'GGML_ASSERT|abort' "$OUT/dec_gpu.log" "$OUT/pp_gpu.log"; then
            result llm PASS "$(basename "$M") greedy text equal to the CPU's"
        else
            result llm FAIL "$(basename "$M") rc=$r1/$r2/$r3/$r4 greedy_text_equal=$same"
        fi
        ;;
    *) result "$c" FAIL "unknown check" ;;
    esac
done
say "REGRESS_DONE failures=$fails"
echo "$fails" > "$OUT/suite.rc"
[ "$fails" -eq 0 ]
