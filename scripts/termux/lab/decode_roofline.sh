#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# decode_roofline.sh KIT API OUT MODEL...: is token generation on this phone bound by the DRAM bandwidth each processor
# pulls (the explanation of the literature: Xiao et al. arXiv 2410.03613, HeteroLLM arXiv 2501.14794, PowerInfer-2
# arXiv 2406.06282), and how close do the CPU and the AMEVA GPU path come to their own ceilings?
#   1. cpu_bw_bench: DRAM read bandwidth of 1, 2, 4 and all allowed cores (big cores first)
#   2. cl_hw_bench HW_STREAM_SECONDS: the GPU's sustained read bandwidth (OpenCL; on Adreno through the vendor library)
#   3. both at once (all allowed cores + the GPU stream): what the SoC gives two processors together
#   4. per MODEL: token generation (128 tokens after a short prompt) with the CPU backend (--device none, -t THREADS)
#      and with AMEVA0 on API, 3 rotated rounds; the weight bytes from the load log ("model buffer size"), so
#      achieved GB/s = t/s x bytes; then one AMEVA_PROFILE=2 run: GPU busy ms per token against wall ms per token
# The clock-cap and temperature gates of gpu_sysfs.sh precede every run. KIT is a hw_study_fleet.sh kit (bin, rt, src).
# Prints RL_* lines; the cpuset of the process (Cpus_allowed_list) is printed, since ssh sessions on some phones get
# only part of the cores (S25: 0-1,4-5).
set -u
KIT=$(cd "${1:?kit}" && pwd); API=${2:?api}; OUT=${3:?out}; shift 3
B=$KIT/bin; SRC=$KIT/src
mkdir -p "$OUT"
. "$SRC/gpu_sysfs.sh"
THREADS=${THREADS:-4}
CAP=$(gpu_cap_khz)
gate() { gpu_gate_wait "$CAP" "${TMU_MAX:-42}" "${COOL_MAX:-180}" > /dev/null; sleep "${1:-10}"; }
echo "RL_INFO model=$(getprop ro.product.model) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') family=$(gpu_family) cap_khz=$CAP api=$API threads=$THREADS kit=$(head -1 "$KIT/KIT_ID")"
clang -O2 -o "$OUT/cpu_bw_bench" "$SRC/cpu_bw_bench.c" -lpthread > "$OUT/build.log" 2>&1 || { echo "RL_FAIL build cpu_bw_bench"; exit 1; }
clang -O2 -I"$SRC/include" -o "$OUT/cl_hw_bench" "$SRC/cl_hw_bench.c" -lOpenCL >> "$OUT/build.log" 2>&1 || { echo "RL_FAIL build cl_hw_bench"; exit 1; }
mkdir -p "$OUT/icd"; gpu_cl_vendor > "$OUT/icd/vendor.icd"
# Adreno: the vendor libOpenCL.so, /system/lib64 first (the S25's system liblog needs the system libc++)
if [ "$(gpu_family)" = adreno ]; then CLENV=(LD_LIBRARY_PATH=/system/lib64:/vendor/lib64)
else CLENV=(OCL_ICD_VENDORS=$OUT/icd LD_LIBRARY_PATH=$PREFIX/lib:/system/lib64:/vendor/lib64:/vendor/lib64/egl); fi
sleep 30 # the compile's heat
# 1-3: bandwidth
gate 10; "$OUT/cpu_bw_bench" 512 3 | sed 's/^/RL_/'
gate 10; env "${CLENV[@]}" HW_STREAM_SECONDS=6 "$OUT/cl_hw_bench" 0 1 2>&1 | grep -E "HW_STREAM|FAIL" | sed 's/^/RL_GPU_ALONE /'
gate 10
NALL=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}' | tr ',' '\n' | awk -F- '{n += (NF == 2 ? $2 - $1 + 1 : 1)} END {print n}')
STREAM=1 "$OUT/cpu_bw_bench" 512 6 "$NALL" > "$OUT/both_cpu.log" 2>&1 &
cpid=$!
env "${CLENV[@]}" HW_STREAM_SECONDS=6 "$OUT/cl_hw_bench" 0 1 > "$OUT/both_gpu.log" 2>&1
wait $cpid
echo "RL_BOTH cpu[$(grep -o 'threads=.*GB/s=[0-9.]*' "$OUT/both_cpu.log")] gpu[$(grep -o 'GB/s=[0-9.]*' "$OUT/both_gpu.log")]"
# 4: models
P="The history of the city begins with"
for M in "$@"; do
    mn=$(basename "$M" .gguf)
    for r in 1 2 3; do
        for v in $( [ $r = 2 ] && echo "gpu cpu" || echo "cpu gpu" ); do
            gate 8
            if [ $v = gpu ]; then
                AMEVA_BACKEND=$API GGML_BACKEND_PATH=$B/libggml-ameva.so LD_LIBRARY_PATH=$KIT/rt:$B timeout 900 llama-completion -m "$M" \
                    --device AMEVA0 -ngl 99 --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" -p "$P" -n 128 -c 2048 \
                    > "$OUT/o.txt" 2> "$OUT/l.txt" < /dev/null
            else
                LD_LIBRARY_PATH=$KIT/rt timeout 900 llama-completion -m "$M" --device none -ngl 0 --fit off -fa on -no-cnv --temp 0 \
                    --top-k 1 --seed 42 -t "$THREADS" -p "$P" -n 128 -c 2048 > "$OUT/o.txt" 2> "$OUT/l.txt" < /dev/null
            fi
            rc=$?
            tg=$(grep -E "[^t] eval time" "$OUT/l.txt" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
            mib=$(grep -E "model buffer size" "$OUT/l.txt" | grep -oE "[0-9.]+ MiB" | awk '{s += $1} END {printf "%.0f", s}')
            echo "RL_DECODE model=$mn round=$r var=$v rc=$rc tg_tps=${tg:-NA} weights_mib=$mib text=$(sha256sum < "$OUT/o.txt" | cut -c1-12) cap=$(gpu_cap_khz) temp=$(gpu_temp_c)"
        done
    done
    gate 8
    AMEVA_PROFILE=2 AMEVA_BACKEND=$API GGML_BACKEND_PATH=$B/libggml-ameva.so LD_LIBRARY_PATH=$KIT/rt:$B timeout 900 llama-completion -m "$M" \
        --device AMEVA0 -ngl 99 --fit off -fa on -no-cnv --temp 0 --top-k 1 --seed 42 -t "$THREADS" -p "$P" -n 64 -c 2048 --no-warmup \
        > "$OUT/o.txt" 2> "$OUT/prof_$mn.txt" < /dev/null
    awk -v m="$mn" '
        /AMEVA_PROFILE kind=/ { g = $0; sub(/.*gpu_ms=/, "", g); sub(/ .*/, "", g); gpu += g
                                d = $0; sub(/.*dispatches=/, "", d); sub(/ .*/, "", d); disp += d }
        /prompt eval time/ { pw = $0; sub(/.*= */, "", pw); sub(/ ms.*/, "", pw); pn = $0; sub(/.*\/ */, "", pn); sub(/ tokens.*/, "", pn) }
        /[^t] eval time/ && !/prompt/ { ew = $0; sub(/.*= */, "", ew); sub(/ ms.*/, "", ew); en = $0; sub(/.*\/ */, "", en); sub(/ runs.*/, "", en) }
        END { graphs = en + 1; printf "RL_PROFILE model=%s gpu_ms_total=%.1f dispatches=%d wall_prompt_ms=%s prompt_tokens=%s wall_eval_ms=%s eval_tokens=%s gpu_ms_per_graph=%.2f wall_ms_per_token=%.2f dispatches_per_graph=%.0f\n",
                    m, gpu, disp, pw, pn, ew, en, gpu / graphs, ew / en, disp / graphs }' "$OUT/prof_$mn.txt"
done
echo "RL_END $(date +%T)"
