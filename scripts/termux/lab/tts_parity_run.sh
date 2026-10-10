#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tts_parity_run.sh TAG OUT MODELS [ameva_tts_parity args...]: Kokoro CPU / AMEVA0 parity and RTF, log in
# ~/abi-lab/tts-TAG.log, WAV files in ~/abi-lab/tts-TAG/ (written to disk only; nothing is played).
# MODELS holds kokoro-f16.gguf and kokoro-f32.gguf (adapters/tts/kokoro/convert_onnx_to_gguf.py on the device's own
# kokoro-int8-en-v0_19) and t*.tokens (phoneme ids from g2p_tokens.py). OUT holds libggml-ameva.so and
# ameva_tts_parity (tts_build.sh). AMEVA_BACKEND=opencl selects the Mali OpenCL path.
set -u
TAG=$1
OUT=$2
M=$3
shift 3
B=${B:-$HOME/.local/share/ameva/releases/llamacpp/v2.8.1-1d086faf6227}
L=$HOME/abi-lab/tts-$TAG.log
D=$HOME/abi-lab/tts-$TAG
mkdir -p "$D"
clk() {
    if [ -r /sys/class/kgsl/kgsl-3d0/max_gpuclk ]; then echo "kgsl max $(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk) cur $(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null || echo unreadable)";
    elif [ -r /sys/kernel/gpu/gpu_clock ]; then echo "exynos $(cat /sys/kernel/gpu/gpu_clock) max $(cat /sys/kernel/gpu/gpu_max_clock 2>/dev/null)";
    else echo "gpu clock not readable"; fi
}
{
echo "device $(getprop ro.product.model) cpus $(grep Cpus_allowed_list /proc/self/status | cut -f2) $(clk) AMEVA_BACKEND=${AMEVA_BACKEND:-vulkan}"
# the output directory first: Termux clang puts $PREFIX/lib ahead of -rpath, and the bundle lib directory carries
# its own libggml-ameva.so
LD_LIBRARY_PATH=$OUT:$B/lib timeout 1800 "$OUT/ameva_tts_parity" --model "$M/kokoro-f16.gguf" --ref-model "$M/kokoro-f32.gguf" \
    --plugin "$OUT/libggml-ameva.so" --out-dir "$D" "$@"
echo "rc=$?"
echo "after: $(clk)"
} > "$L" 2>&1
echo done > "$HOME/abi-lab/tts-$TAG.done"
