#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
# tbo_kit.sh [kit dir]: ggml test-backend-ops (~/abi-lab/tbo, llama.cpp b8471) for AMEVA0 with the plugin of a
# regression kit (scripts/fleet/regress_fleet.sh; default ~/abi-lab/regkit/current), on the automatic backend, twice:
# default settings and AMEVA_REPACK_ALL=1. Results: <kit>/results/tbo-<time>/summary.txt ending with TBO_DONE and
# tbo.rc (0 = both modes passed).
# A mode counts only when the log lists the device AMEVA0, says "Backend AMEVA: OK" and passed a nonzero number of
# tests: test-backend-ops exits 0 when the backend was refused at load time (zero tests), which is reported as
# TBO_NOT_TESTED, never as a pass.
set -u
K=$(cd "${1:-$HOME/abi-lab/regkit/current}" && pwd)
B=$K/bin
T=$HOME/abi-lab/tbo/test-backend-ops
O=$K/results/tbo-$(date +%Y%m%d-%H%M%S)
mkdir -p "$O"
S=$O/summary.txt
[ -x "$T" ] || { echo "TBO_FAIL no $T" | tee "$S"; echo 2 > "$O/tbo.rc"; exit 2; }
echo "device $(getprop ro.product.model) plugin $(sha256sum "$B/libggml-ameva.so" | cut -c1-16) tbo $(sha256sum "$T" | cut -c1-16) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)" > "$S"
fails=0
cd "$O" || exit 2
for mode in default repack_all; do
    e=(AMEVA_QMV_TUNE_FILE="$O/qmv-tune.txt" AMEVA_PIPELINE_CACHE="$O/pc.bin")
    [ $mode = repack_all ] && e+=(AMEVA_REPACK_ALL=1)
    env "${e[@]}" GGML_BACKEND_PATH="$B/libggml-ameva.so" LD_LIBRARY_PATH="$B/ggml-rt:$B" timeout 7200 "$T" -b AMEVA0 \
        > "$O/$mode.log" 2>&1
    rc=$?
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$O/$mode.log")
    passed=$(echo "$clean" | grep -E '^ +[0-9]+/[0-9]+ tests passed' | tr -s ' ' | head -1)
    listed=$(echo "$clean" | grep -cE "Backend [0-9]+/[0-9]+: AMEVA0")
    ok=$(echo "$clean" | grep -cE "Backend AMEVA: .*OK")
    api=vulkan; grep -q "kernels run as OpenCL C" "$O/$mode.log" && api=opencl
    n=$(echo "$passed" | sed -E 's/^ *([0-9]+)\/.*/\1/')
    if [ "$rc" -eq 0 ] && [ "$listed" -gt 0 ] && [ "$ok" -gt 0 ] && [ "${n:-0}" -gt 0 ]; then
        echo "TBO $mode PASS api=$api$passed" >> "$S"
    elif [ "$listed" -eq 0 ] || [ "${n:-0}" -eq 0 ]; then
        echo "TBO $mode TBO_NOT_TESTED rc=$rc api=$api listed=$listed tests=${n:-0}" >> "$S"; fails=$((fails + 1))
    else
        echo "TBO $mode FAIL rc=$rc api=$api$passed" >> "$S"; fails=$((fails + 1))
        echo "$clean" | grep -E "FAIL" | grep -v "not supported" | head -5 | cut -c1-200 | sed 's/^/  | /' >> "$S"
    fi
done
echo TBO_DONE >> "$S"
echo $fails > "$O/tbo.rc"
