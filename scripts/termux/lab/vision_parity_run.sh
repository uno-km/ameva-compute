#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# vision_parity_run.sh TAG [ameva_vision_parity args...]: SmolVLM-500M mmproj parity / encode time, log in ~/abi-lab/vis-TAG.log
# OUT holds libggml-ameva.so and libac_runtime.so.2 (scripts/termux/build_ggml_ameva.sh --ggml-dir <llama.cpp b8471>/ggml
# --ggml-lib $B/lib) and ameva_vision_parity, built against the bundle's libraries with the b8471 headers:
#   clang++ -O2 -std=c++17 -I <b8471>/include -I <b8471>/ggml/include -I <b8471>/tools/mtmd tests/ameva_vision_parity.cpp
#       $B/lib/libmtmd.so.0 $B/lib/libllama.so.0 $B/lib/libggml-base.so.0 $B/lib/libggml.so.0 -ldl -Wl,-rpath,$B/lib
# The F32 reference for --ref-mmproj: python3 gguf_dequant_f32.py <b8471>/gguf-py MMPROJ.gguf MMPROJ-f32.gguf.
# Copy this script and vision_fa_sweep.sh to ~/abi-lab.
set -u
TAG=$1; shift
OUT=$HOME/abi-lab/out-vis
# the bundle libraries first: Termux clang puts $PREFIX/lib ahead of -rpath, where an older ggml / llama may lie
B=$HOME/.local/share/ameva/releases/llamacpp/v2.8.1-1d086faf6227
M=$HOME/.cache/termux-vision/models/smolvlm-500m-q4
IMG=$HOME/abi-lab/vis-pkg/test-1.jpeg
[ -f "$IMG" ] || IMG=$HOME/llama.cpp/tools/mtmd/test-1.jpeg
L=$HOME/abi-lab/vis-$TAG.log
clk() {
    if [ -r /sys/class/kgsl/kgsl-3d0/gpuclk ]; then echo "kgsl max $(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk) cur $(cat /sys/class/kgsl/kgsl-3d0/gpuclk)";
    elif [ -r /sys/kernel/gpu/gpu_clock ]; then echo "exynos $(cat /sys/kernel/gpu/gpu_clock) max $(cat /sys/kernel/gpu/gpu_max_clock 2>/dev/null)";
    else echo "gpu clock not readable"; fi
}
{
echo "device $(getprop ro.product.model) cpus $(grep Cpus_allowed_list /proc/self/status | cut -f2) $(clk) AMEVA_BACKEND=${AMEVA_BACKEND:-vulkan}"
LD_LIBRARY_PATH=$OUT:$B/lib${VIS_EXTRA_LD:+:$VIS_EXTRA_LD} timeout 1800 $OUT/ameva_vision_parity --model $M/SmolVLM-500M-Instruct-Q8_0.gguf \
    --mmproj $M/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf --image $IMG --plugin $OUT/libggml-ameva.so "$@"
echo "rc=$?"
echo "after: $(clk)"
} > $L 2>&1
echo done > $HOME/abi-lab/vis-$TAG.done
