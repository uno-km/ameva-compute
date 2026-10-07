#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tbo_run.sh: ggml test-backend-ops (~/abi-lab/tbo, b8471) for AMEVA0, twice: default settings, then
# AMEVA_REPACK_ALL=1 (quantized MUL_MAT through the repacked-weight kernels). Meant to be started with nohup:
#   (nohup bash ~/abi-lab/tbo_run.sh > /dev/null 2>&1 &)
# Logs: ~/abi-lab/results/tbo_default.log, tbo_repack_all.log; the second ends with TBO_DONE.
# The backend filter is the device name AMEVA0 (with "AMEVA" every backend is skipped).
set -u
LAB=$HOME/abi-lab
OUT=${OUT:-$LAB/out-g2}
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
cd "$LAB" || exit 1
mkdir -p results
rm -f results/tbo_default.log results/tbo_repack_all.log
GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 tbo/test-backend-ops -b AMEVA0 \
    > results/tbo_default.log 2>&1
echo "rc=$?" >> results/tbo_default.log
AMEVA_REPACK_ALL=1 GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 \
    tbo/test-backend-ops -b AMEVA0 > results/tbo_repack_all.log 2>&1
echo "rc=$?" >> results/tbo_repack_all.log
echo TBO_DONE >> results/tbo_repack_all.log
