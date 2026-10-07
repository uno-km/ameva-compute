#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# conv_bench.sh [all|build|run] [ITER]
# Build ameva_conv_bench (tests/ameva_conv_bench.cpp) from SRC (default ~/abi-lab/src) into OUT (default
# ~/abi-lab/out-conv) and run it: staging from the source tensor (ameva_conv) against a materialised column matrix
# (ameva_qgemm), q8_0 and f16 weights, every output compared with an f32 CPU convolution.
# Shapes: SHAPES (space separated, W,H,T,C,OC,KW,KH,KT[,PW,PH,PT0,PT1[,SW,SH,ST[,DW,DH,DT]]]) or the default list:
# four small shapes that exercise padding of channels, strides, dilation and a causal time axis, the three
# convolutions of docs/design/diffusion-vae-conv.md section 2 and one 3x3x3. TYPES passes through (default q8_0,f16).
# EXTRA_DEFS (kernel tuning) adds defines to every kernel of the build, e.g. F16_UNPACK.
# Needs glslangValidator, spirv-val, python3, clang. Prints the build identity, the run conditions and the bench lines.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src}
OUT=${OUT:-$LAB/out-conv}
MODE=${1:-all}
IT=${2:-4}
G="$OUT/gen"
S="$SRC/adapters/llama/ggml-ameva/shaders"

build() {
    mkdir -p "$G" || return 1
    # ac_runtime exactly as scripts/termux/build_ggml_ameva.sh builds it
    clang -O2 -fPIC -c -I "$SRC/include" -I "$SRC/src/runtime" "$SRC/src/runtime/ac_security_guard.c" \
        -o "$OUT/ac_security_guard.o" || return 1
    clang++ -O2 -fPIC -shared -std=c++17 -I "$SRC/include" -I "$SRC/src" \
        "$SRC"/src/backend/vulkan/ac_vk_buffer.cpp "$SRC"/src/backend/vulkan/ac_vk_context.cpp \
        "$SRC"/src/backend/vulkan/ac_vk_dispatch.cpp "$SRC"/src/backend/vulkan/ac_vk_event.cpp \
        "$SRC"/src/backend/vulkan/ac_vk_loader.cpp "$SRC"/src/backend/vulkan/ac_vk_pipeline.cpp \
        "$SRC"/src/backend/opencl/ac_cl_backend.cpp \
        "$SRC"/src/runtime/ac_runtime.cpp "$SRC"/src/runtime/ac_error.cpp "$OUT/ac_security_guard.o" \
        -lvulkan -ldl -lm -Wl,-soname,libac_runtime.so.2 -Wl,--no-undefined -o "$OUT/libac_runtime.so.2" || return 1
    rm -f "$OUT/ac_security_guard.o"
    local name var src defs d x
    while read -r name var src defs; do
        [ -z "$name" ] && continue
        d=""
        for x in $defs ${EXTRA_DEFS:-}; do d="$d -D$x"; done
        # shellcheck disable=SC2086
        glslangValidator -V -q --target-env vulkan1.0 $d -o "$G/$name.spv" "$S/$src" > "$G/$name.log" 2>&1 \
            || { echo "SHADER_FAIL $name"; cat "$G/$name.log"; return 1; }
        spirv-val --target-env vulkan1.0 "$G/$name.spv" || { echo "SPIRV_VAL_FAIL $name"; return 1; }
        python3 "$SRC/scripts/embed_spirv.py" --input "$G/$name.spv" --output "$G/${name}_spv.h" --var "$var" \
            > /dev/null || { echo "EMBED_FAIL $name"; return 1; }
    done <<'EOF'
qgemm_q8_0 QgemmQ80 ameva_qgemm.comp TYPE_Q8_0
qgemm_f16 QgemmF16 ameva_qgemm.comp TYPE_F16
conv_q8_0_s ConvQ80S ameva_conv.comp TYPE_Q8_0
conv_q8_0_v ConvQ80V ameva_conv.comp TYPE_Q8_0 STAGE_VEC4
conv_f16_s ConvF16S ameva_conv.comp TYPE_F16
conv_f16_v ConvF16V ameva_conv.comp TYPE_F16 STAGE_VEC4
conv_q8_0_va1 ConvQ80Va1 ameva_conv.comp TYPE_Q8_0 STAGE_VEC4 ABLATE=1
conv_q8_0_va2 ConvQ80Va2 ameva_conv.comp TYPE_Q8_0 STAGE_VEC4 ABLATE=2
conv_q8_0_va3 ConvQ80Va3 ameva_conv.comp TYPE_Q8_0 STAGE_VEC4 ABLATE=3
conv_f16_va1 ConvF16Va1 ameva_conv.comp TYPE_F16 STAGE_VEC4 ABLATE=1
conv_f16_va2 ConvF16Va2 ameva_conv.comp TYPE_F16 STAGE_VEC4 ABLATE=2
conv_f16_va3 ConvF16Va3 ameva_conv.comp TYPE_F16 STAGE_VEC4 ABLATE=3
conv_q8_0_va4 ConvQ80Va4 ameva_conv.comp TYPE_Q8_0 STAGE_VEC4 ABLATE=4
conv_q8_0_vd ConvQ80Vd ameva_conv.comp TYPE_Q8_0 STAGE_VEC4 OUT_DIRECT
conv_f16_va4 ConvF16Va4 ameva_conv.comp TYPE_F16 STAGE_VEC4 ABLATE=4
conv_f16_vd ConvF16Vd ameva_conv.comp TYPE_F16 STAGE_VEC4 OUT_DIRECT
qgemm_q8_0_a1 QgemmQ80A1 ameva_qgemm.comp TYPE_Q8_0 ABLATE=1
qgemm_q8_0_a2 QgemmQ80A2 ameva_qgemm.comp TYPE_Q8_0 ABLATE=2
qgemm_f16_a1 QgemmF16A1 ameva_qgemm.comp TYPE_F16 ABLATE=1
qgemm_f16_a2 QgemmF16A2 ameva_qgemm.comp TYPE_F16 ABLATE=2
EOF
    [ -f "$G/conv_f16_v_spv.h" ] || return 1
    clang++ -O2 -std=c++17 -Wall -Wextra -I "$SRC/include" -I "$G" "$SRC/tests/ameva_conv_bench.cpp" \
        "$OUT/libac_runtime.so.2" -lm -pthread -Wl,-rpath,"$OUT" -o "$OUT/ameva_conv_bench" || return 1
    echo "BUILD_OK extra_defs=\"${EXTRA_DEFS:-}\" glslang=\"$(glslangValidator --version | head -1)\" conv=$(sha256sum "$S/ameva_conv.comp" | cut -c1-12)" \
        "qgemm=$(sha256sum "$S/ameva_qgemm.comp" | cut -c1-12) qdecode=$(sha256sum "$S/ameva_qdecode.glsl" | cut -c1-12)" \
        "bench=$(sha256sum "$SRC/tests/ameva_conv_bench.cpp" | cut -c1-12)"
}

run() {
    local shapes shape rc=0
    shapes=${SHAPES:-"37,29,1,4,3,3,3,1 40,24,1,64,96,1,1,1 37,29,3,20,70,3,3,3,1,1,2,0,2,2,1,2,1,1 \
70,33,2,33,65,3,3,3,1,1,2,0 128,128,1,256,256,3,3,1 64,64,1,512,512,3,3,1 256,256,1,128,128,3,3,1 \
96,96,4,96,96,3,3,3,1,1,2,0"}
    local clk=/sys/kernel/gpu/gpu_clock sp
    [ -r "$clk" ] || clk=/sys/class/kgsl/kgsl-3d0/gpuclk
    for shape in $shapes; do
        echo "CONDITIONS $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)" \
            "gpu_max_clock=$(cat /sys/kernel/gpu/gpu_max_clock 2>/dev/null || cat /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null || echo unreadable)" \
            "governor=$(cat /sys/kernel/gpu/gpu_governor 2>/dev/null || cat /sys/class/kgsl/kgsl-3d0/devfreq/governor 2>/dev/null || echo unreadable)"
        # the GPU clock is only meaningful while the GPU works: sampled every 0.25 s during the run
        : > "$OUT/clk.log"
        ( while :; do cat "$clk" 2>/dev/null; sleep 0.25; done ) >> "$OUT/clk.log" &
        sp=$!
        # shellcheck disable=SC2086
        LD_LIBRARY_PATH="$OUT" "$OUT/ameva_conv_bench" "$shape" "$IT" ${TYPES:-} 2>&1 | grep -v "^WARNING: linker"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { rc=1; echo "BENCH_FAIL shape=$shape"; }
        kill "$sp" 2>/dev/null
        wait "$sp" 2>/dev/null
        if [ -s "$OUT/clk.log" ]; then
            echo "GPU_CLOCK value:samples $(sort -n "$OUT/clk.log" | uniq -c | awk '{printf "%s:%s ", $2, $1}')"
        else
            echo "GPU_CLOCK not readable on this device during the run (the maximum is in CONDITIONS)"
        fi
    done
    return $rc
}

case "$MODE" in
    build) build || { echo BUILD_FAIL; exit 1; } ;;
    run) run || exit 1 ;;
    all) build || { echo BUILD_FAIL; exit 1; }; run || exit 1 ;;
    *) echo "usage: $0 [all|build|run] [ITER]" >&2; exit 2 ;;
esac
