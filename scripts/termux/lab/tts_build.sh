#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tts_build.sh SRC OUT [kernels|plugin|tool|all]: Stage 5 (Kokoro TTS) build on a fleet device, log in OUT/*.log.
#   kernels  scripts/build_ameva_kernels.py (glslangValidator + spirv-val on the device)
#   plugin   libggml-ameva.so + libac_runtime.so.2 against the runtime bundle's ggml (b8471 headers in GGML_DIR)
#   tool     ameva_tts_parity (adapters/tts/kokoro + tests/ameva_tts_parity.cpp) against the bundle's libggml*
# GGML_DIR (default ~/llama.cpp/ggml) must be the ggml of the bundle (llama.cpp b8471); B is the bundle root.
set -u
SRC=$1
OUT=$2
WHAT=${3:-all}
B=${B:-$HOME/.local/share/ameva/releases/llamacpp/v2.8.1-1d086faf6227}
GGML_DIR=${GGML_DIR:-$HOME/llama.cpp/ggml}
mkdir -p "$OUT"
cd "$SRC" || exit 1
if [ "$WHAT" = kernels ] || [ "$WHAT" = all ]; then
    python3 scripts/build_ameva_kernels.py > "$OUT/kernels.log" 2>&1 || { echo KERNELS_FAIL; tail -20 "$OUT/kernels.log"; exit 1; }
    tail -2 "$OUT/kernels.log"
fi
if [ "$WHAT" = plugin ] || [ "$WHAT" = all ]; then
    bash scripts/termux/build_ggml_ameva.sh --ggml-dir "$GGML_DIR" --ggml-lib "$B/lib" --out "$OUT" \
        --list-devices-cmd "$B/bin/llama-completion --list-devices" > "$OUT/plugin.log" 2>&1 ||
        { echo PLUGIN_FAIL; tail -20 "$OUT/plugin.log"; exit 1; }
    tail -3 "$OUT/plugin.log"
fi
if [ "$WHAT" = tool ] || [ "$WHAT" = all ]; then
    clang++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -I "$GGML_DIR/include" adapters/tts/kokoro/kokoro.cpp \
        tests/ameva_tts_parity.cpp "$B/lib/libggml-base.so.0" "$B/lib/libggml.so.0" "$B/lib/libggml-cpu.so.0" -ldl \
        -Wl,-rpath,"$B/lib" -o "$OUT/ameva_tts_parity" > "$OUT/tool.log" 2>&1 || { echo TOOL_FAIL; tail -20 "$OUT/tool.log"; exit 1; }
    grep -E "warning|error" "$OUT/tool.log" | head -5
fi
echo BUILD_OK
