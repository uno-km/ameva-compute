#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# ladder_run.sh API: the LLM size ladder on one phone - every model of MODELS that is on the phone, GPU (API) vs CPU,
# ROUNDS rounds (default 5) rotated, through llm_plugin_ab.sh (variants "gpu" and "cpu") from the hardware-study kit
# (~/abi-lab/hwkit/current, scripts/fleet/hw_study_fleet.sh). Analysis on the PC: scripts/fleet/ladder_fit.py.
#   - The clock-cap gate target is read ONCE here and passed to every model (CAP_TARGET): read per model, a cap the
#     system had already lowered (S21 858 -> 403 MHz after about 2 h) became the target and throttled runs passed
#     as valid (2026-11-08). COOL_MAX (default 600 s) gives the cap time to come back; runs below it print valid=0.
#   - A model that is not on the phone prints LADDER_SKIP; a model whose runs fail prints LADDER_FAIL.
# Output: ~/abi-lab/hwkit/ladder-<date>/ (info, <model>.log, done.rc); ~/abi-lab/hwkit/ladder_last names it.
K=$HOME/abi-lab/hwkit/current; API=${1:?api (vulkan or opencl)}
O=$HOME/abi-lab/hwkit/ladder-$(date +%Y%m%d-%H%M%S); mkdir -p "$O"; echo "$O" > "$HOME/abi-lab/hwkit/ladder_last"
D=$HOME/.cache/termux-llamacpp/models
. "$K/src/gpu_sysfs.sh" || { echo "LADDER_FAIL reason=no gpu_sysfs.sh" > "$O/info"; echo 2 > "$O/done.rc"; exit 2; }
export CAP_TARGET=${CAP_TARGET:-$(gpu_cap_khz)} COOL_MAX=${COOL_MAX:-600}
echo "kit=$(readlink "$K") api=$API cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') cap_target=$CAP_TARGET cool_max=$COOL_MAX" > "$O/info"
rc=0
for m in ${MODELS:-qwen2.5-0.5b-instruct-q4_k_m Llama-3.2-1B-Instruct-Q4_K_M qwen2.5-1.5b-instruct-q4_k_m gemma-2-2b-it-Q4_K_M qwen2.5-3b-instruct-q4_k_m Llama-3.2-3B-Instruct-Q4_K_M}; do
    f=$D/$m.gguf
    [ -s "$f" ] || f=$(find "$HOME" -name "$m.gguf" -size +100M 2>/dev/null | head -1)
    [ -s "$f" ] || { echo "LADDER_SKIP model=$m reason=not_on_phone" >> "$O/info"; continue; }
    VARIANTS="${LADDER_VARIANTS:-gpu:$K/bin:$API cpu:$K/bin:cpu}" ROUNDS=${ROUNDS:-5} OUTD=$O/$m GGML_RT=$K/rt CORPUS=$K/src/prompt_corpus.txt \
        bash "$K/src/llm_plugin_ab2.sh" "$f" > "$O/$m.log" 2>&1 || { rc=1; echo "LADDER_FAIL model=$m" >> "$O/info"; }
done
echo $rc > "$O/done.rc"
