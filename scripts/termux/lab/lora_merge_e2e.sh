#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# lora_merge_e2e.sh
# A LoRA merged into the weights at load (sd-cli --lora-apply-mode immediately) with the diffusion model on ggml-ameva,
# against the same merge on the CPU, the adapter applied at run time, and no adapter. Runs sd_e2e.sh three times
# (SD 1.x model with --type f16: stable-diffusion.cpp merges into f16 or f32 weights only) and compares the images.
# The adapter is the synthetic one of make_test_lora.py, written to LORA_DIR when it is not there.
#
# What has to hold for the merge on the backend:
#   every AMEVA run passes sd_e2e.sh (AMEVA_STRICT=1: no declined op, no scheduler split, dispatches reported);
#   merged on AMEVA against merged on the CPU: at least PSNR_MIN dB (default 40, this script's choice);
#   merged on AMEVA against no adapter on AMEVA: below EFFECT_MAX dB (default 30): the merge changed the weights.
# The other pairs are reported: merged against applied at run time on each backend, the adapter's effect on the CPU.
# SRC, OUT, SDB, MODEL, SIZE (256), STEPS (4), GUARD_MB as in sd_e2e.sh. Exit status 0 when all of that holds.
set -u
LAB=$HOME/abi-lab
export SRC=${SRC:-$LAB/src}
export OUT=${OUT:-$LAB/out-sd}
export SIZE=${SIZE:-256}
export STEPS=${STEPS:-4}
LORA_DIR=${LORA_DIR:-$LAB/lora-test}
PSNR_MIN=${PSNR_MIN:-40}
EFFECT_MAX=${EFFECT_MAX:-30}
P=${PROMPT:-a lighthouse on a rocky coast at sunset, detailed painting}
E2E=$SRC/scripts/termux/lab/sd_e2e.sh
PSNR=$SRC/scripts/termux/lab/img_psnr.py

mkdir -p "$LORA_DIR"
if [ ! -s "$LORA_DIR/testlora.safetensors" ]; then
    python3 "$SRC/scripts/termux/lab/make_test_lora.py" "$LORA_DIR/testlora.safetensors" || { echo LORA_FILE_FAIL; exit 1; }
fi
echo "lora: $(sha256sum "$LORA_DIR/testlora.safetensors" | cut -c1-12) $LORA_DIR/testlora.safetensors"

rc=0
runs="cpu=cpu gpu=cpu,diffusion=AMEVA0,vae=AMEVA0"
TAG=mrg-nolora EXTRA="--type f16" PROMPT="$P" RUNS="$runs" PAIRS="cpu:gpu:-" bash "$E2E" || rc=1
for mode in at_runtime immediately; do
    TAG=mrg-$mode EXTRA="--type f16 --lora-model-dir $LORA_DIR --lora-apply-mode $mode" PROMPT="$P <lora:testlora:1>" \
        RUNS="$runs" PAIRS="cpu:gpu:-" bash "$E2E" || rc=1
    grep -h -o "lora_apply_mode: [a-z_]*\|wtype: [a-z0-9_]*" "$OUT/e2e-mrg-$mode/gpu.log" | sort -u | tr "\n" " "
    echo "applied: $(grep -h -c -i "lora.*applied\|applied.*lora" "$OUT/e2e-mrg-$mode/gpu.log") log lines"
done

# pair NAME REF TEST GATE: GATE "min N" (at least N dB), "max N" (below N dB) or "-" (reported)
pair() {
    local name=$1 a=$OUT/$2 b=$OUT/$3 gate=$4 line db verdict=REPORT
    if [ ! -s "$a" ] || [ ! -s "$b" ]; then
        echo "MERGE_PAIR $name NO_IMAGE"
        [ "$gate" = - ] || rc=1
        return
    fi
    line=$(python3 "$PSNR" "$a" "$b" 2>&1 | tail -1)
    db=$(echo "$line" | grep -o "psnr_db=[0-9.a-z]*" | cut -d= -f2)
    case "$gate" in
        "min "*) verdict=FAIL; awk -v d="$db" -v g="${gate#min }" 'BEGIN {exit !(d == "inf" || d + 0 >= g + 0)}' && verdict=PASS ;;
        "max "*) verdict=FAIL; awk -v d="$db" -v g="${gate#max }" 'BEGIN {exit !(d != "inf" && d != "" && d + 0 < g + 0)}' && verdict=PASS ;;
    esac
    [ $verdict = FAIL ] && rc=1
    echo "MERGE_PAIR $name gate=\"$gate\" $verdict $(echo "$line" | grep -o "psnr_db=.*")"
}
pair "merged: AMEVA against CPU" e2e-mrg-immediately/cpu.png e2e-mrg-immediately/gpu.png "min $PSNR_MIN"
pair "AMEVA: merged against no adapter" e2e-mrg-nolora/gpu.png e2e-mrg-immediately/gpu.png "max $EFFECT_MAX"
pair "AMEVA: merged against at run time" e2e-mrg-at_runtime/gpu.png e2e-mrg-immediately/gpu.png -
pair "CPU: merged against no adapter" e2e-mrg-nolora/cpu.png e2e-mrg-immediately/cpu.png -
pair "CPU: merged against at run time" e2e-mrg-at_runtime/cpu.png e2e-mrg-immediately/cpu.png -
pair "at run time: AMEVA against CPU" e2e-mrg-at_runtime/cpu.png e2e-mrg-at_runtime/gpu.png -
echo "LORA_MERGE_E2E $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
