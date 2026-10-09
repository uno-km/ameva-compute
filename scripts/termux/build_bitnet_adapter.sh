#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the BitNet adapter (adapters/bitnet) with its AMUDA GPU engine (ac_runtime, Vulkan) on a Termux device.
#
#   bash scripts/termux/build_bitnet_adapter.sh [--out DIR] [--kernels]
#
#   --out DIR    output directory (default build-bitnet)
#   --kernels    regenerate adapters/bitnet/src/core/bitnet_kernels_spv.h first (glslangValidator and spirv-val,
#                pkg install glslang spirv-tools)
#
# Output: DIR/libac_runtime.so.2, DIR/libac_bitnet.so, DIR/ac-bitnet-cli, DIR/ac_bitnet_parity. Run them with
# LD_LIBRARY_PATH=DIR first: the Termux clang driver puts $PREFIX/lib in front of -rpath, so a libac_runtime.so.2 left
# there by another installation would otherwise be bound.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build-bitnet" KERNELS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --kernels) KERNELS=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "[build_bitnet_adapter] ERROR: $*" >&2; exit 2; }
A="$ROOT/adapters/bitnet"
HEADER="$A/src/core/bitnet_kernels_spv.h"
if [ "$KERNELS" = 1 ]; then
    python3 "$ROOT/scripts/build_bitnet_kernels.py" --out "$HEADER" || die "kernel build failed"
fi
[ -e "$HEADER" ] || die "$HEADER is missing (run with --kernels)"
mkdir -p "$OUT"

# libac_runtime.so.2, as scripts/termux/build_ggml_ameva.sh builds it
clang -O2 -fPIC -c -I "$ROOT/include" -I "$ROOT/src/runtime" "$ROOT"/src/runtime/ac_security_guard.c \
    -o "$OUT/ac_security_guard.o"
clang++ -O2 -fPIC -shared -std=c++17 -I "$ROOT/include" -I "$ROOT/src" \
    "$ROOT"/src/backend/vulkan/ac_vk_buffer.cpp "$ROOT"/src/backend/vulkan/ac_vk_context.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_dispatch.cpp "$ROOT"/src/backend/vulkan/ac_vk_event.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_loader.cpp "$ROOT"/src/backend/vulkan/ac_vk_pipeline.cpp \
    "$ROOT"/src/backend/opencl/ac_cl_backend.cpp \
    "$ROOT"/src/runtime/ac_runtime.cpp "$ROOT"/src/runtime/ac_error.cpp "$OUT/ac_security_guard.o" \
    -lvulkan -ldl -lm -Wl,-soname,libac_runtime.so.2 -Wl,--no-undefined -o "$OUT/libac_runtime.so.2"
rm -f "$OUT/ac_security_guard.o"
echo "[build_bitnet_adapter] libac_runtime.so.2"

# the adapter library: CPU path and AMUDA GPU engine; -ffp-contract=off keeps the CPU reference free of
# multiply-add contraction, as the GPU kernels mark their roundings precise
CXXFLAGS="-O3 -std=c++17 -fPIC -ffp-contract=off -march=armv8.2-a+dotprod+fp16 -DAC_BITNET_GPU -Wall -Wextra \
-Wno-unused-parameter -Wno-missing-field-initializers -I $ROOT/include -I $A/include -I $A/src"
# shellcheck disable=SC2086
clang++ $CXXFLAGS -shared -fvisibility=hidden -o "$OUT/libac_bitnet.so" -Wl,-soname,libac_bitnet.so \
    "$A/src/cpu/ggml_bitnet_mad.cpp" "$A/src/cpu/llama_bitnet_core.cpp" "$A/src/cpu/c_api.cpp" \
    "$A/src/core/ac_bitnet_engine.cpp" "$OUT/libac_runtime.so.2" -lm -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined
echo "[build_bitnet_adapter] libac_bitnet.so"
# shellcheck disable=SC2086
clang++ $CXXFLAGS -o "$OUT/ac-bitnet-cli" "$A/src/cpu/main.cpp" "$OUT/libac_bitnet.so" -Wl,-rpath,'$ORIGIN'
# shellcheck disable=SC2086
clang++ $CXXFLAGS -o "$OUT/ac_bitnet_parity" "$A/src/cli/ac_bitnet_parity.cpp" "$OUT/libac_bitnet.so" -Wl,-rpath,'$ORIGIN'
echo "[build_bitnet_adapter] ac-bitnet-cli ac_bitnet_parity -> $OUT"
