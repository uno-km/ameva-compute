#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# sd_vae_e2e.sh [build|run|all]      (default: all)
# stable-diffusion.cpp with ggml-ameva as its VAE backend: build, decode, compare the images
# (docs/design/diffusion-vae-conv.md section 6, step 3).
#
# build  SD_SRC (stable-diffusion.cpp with its own ggml, default ~/abi-lab/sd-src) into SDB (default
#        ~/abi-lab/sd-build): sd-cli, ggml as shared libraries with dynamic backends. SD_VULKAN=1 adds ggml-vulkan (the
#        f16 Vulkan reference; Galaxy S25, 2026-10-08: ggml-vulkan.cpp at -O3 did not finish in 67 min, and the VAE
#        decode on Vulkan0 of ggml 8e800cef ended with a segmentation fault there and on the Galaxy S21),
#        SD_CMAKE_ARGS adds options, JOBS sets the compile jobs (default 3). Then the plugin from
#        SRC (default ~/abi-lab/src) into OUT (default ~/abi-lab/out-sd) against that ggml: its headers, its libraries
#        and the GGML_MAX_NAME stable-diffusion.cpp compiles ggml with (ggml_tensor.extra lies behind name).
#        build_ggml_ameva.sh stops unless sd-cli --list-devices shows AMEVA0.
# run    One image per entry "tag:vae-backend" of RUNS, VAE weights as stored in MODEL. Text encoder and UNet run on
#        the CPU in every run with one seed, so every run decodes the same latent; the first pair of PAIRS checks that
#        (two CPU decodes have to give the same file content). --vae-conv-direct is given in every run: ggml-ameva
#        takes CONV_2D and declines IM2COL.
#        No run with f32 VAE weights: sd-cli 50d6405 leaves a GGUF model's VAE as stored whatever --tensor-type-rules
#        says (Galaxy S25, 2026-10-08: same parameter buffer size, same image file). The f32 comparison of the decoder
#        is tests/ameva_vae_graph_check.
#        A run whose backend is AMEVA0 passes only if sd-cli made no scheduler split (it creates the scheduler with a
#        CPU backend behind it only when the VAE backend declines a node; GGML_SCHED_DEBUG=1 prints "## SPLIT") and
#        AMEVA_PROFILE reports dispatches.
#        PAIRS "ref:test:gate" are compared with img_psnr.py: gate "equal", a PSNR floor in dB, or "-" (reported only).
#        PSNR_MIN (default 40) is the floor of the default pairs; it is this script's choice, not a measured limit.
# MODEL (default ~/.cache/termux-diffusion/models/anime.gguf), SIZE (256), STEPS (8), SEED (42), THREADS (4), PROMPT.
# Prints results only; exit status 0 when everything that ran passed.
set -u
LAB=$HOME/abi-lab
SRC=${SRC:-$LAB/src}
SD_SRC=${SD_SRC:-$LAB/sd-src}
SDB=${SDB:-$LAB/sd-build}
OUT=${OUT:-$LAB/out-sd}
RES=${RES:-$OUT/e2e}
MODEL=${MODEL:-$HOME/.cache/termux-diffusion/models/anime.gguf}
SIZE=${SIZE:-256}
STEPS=${STEPS:-8}
SEED=${SEED:-42}
THREADS=${THREADS:-4}
PROMPT=${PROMPT:-a lighthouse on a rocky coast at sunset, detailed painting}
PSNR_MIN=${PSNR_MIN:-40}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
SDCLI=$SDB/bin/sd-cli
# The Termux clang driver writes $PREFIX/lib in front of the build directory into sd-cli's RUNPATH, and $PREFIX/lib can
# hold another ggml (Galaxy S25: "cannot locate symbol gguf_write_to_file_ptr"). OUT/ggml-rt (made by
# build_ggml_ameva.sh) holds links to this build's ggml libraries alone and goes first, then the plugin's directory.
RT=$OUT/ggml-rt:$OUT
mode=${1:-all}

build() {
    [ -f "$SD_SRC/CMakeLists.txt" ] && [ -f "$SD_SRC/ggml/src/ggml-backend-impl.h" ] || { echo "SD_SRC_MISSING $SD_SRC"; exit 1; }
    max_name=$(grep -o "GGML_MAX_NAME=[0-9]*" "$SD_SRC/CMakeLists.txt" | head -1)
    [ -n "$max_name" ] || { echo "SD_SRC_NO_GGML_MAX_NAME: $SD_SRC/CMakeLists.txt does not set it; give the plugin the host's value"; exit 1; }
    vk=OFF targets="sd-cli ggml-cpu"
    if [ "${SD_VULKAN:-0}" = 1 ]; then vk=ON targets="$targets ggml-vulkan"; fi
    mkdir -p "$SDB" "$OUT"
    echo "stable-diffusion.cpp: $SD_SRC -> $SDB (vulkan=$vk, $max_name)"
    # GGML_NATIVE=OFF with a fixed -march: the binaries are also run on other devices of the fleet.
    # GGML_OPENMP=OFF: with it libggml-base.so needs __kmpc_dispatch_deinit, which the libomp.so found at run time on
    # the Galaxy S25 does not have ("cannot locate symbol", sd-cli does not start); ggml's own thread pool is used.
    cmake -S "$SD_SRC" -B "$SDB" -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON \
        -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF \
        -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_VULKAN=$vk ${SD_CMAKE_ARGS:-} > "$OUT/sd_configure.log" 2>&1 \
        || { echo SD_CONFIGURE_FAIL; grep -E "CMake Error|error" -A4 "$OUT/sd_configure.log" | head -30; exit 1; }
    cmake --build "$SDB" --target $targets -j "${JOBS:-3}" > "$OUT/sd_build.log" 2>&1 \
        || { echo SD_BUILD_FAIL; grep -E "error|Error|Killed" -B2 -A6 "$OUT/sd_build.log" | head -60; exit 1; }
    [ -x "$SDCLI" ] && [ -e "$SDB/bin/libggml-base.so" ] && [ -e "$SDB/bin/libggml-cpu.so" ] \
        || { echo "SD_BUILD_INCOMPLETE: $(ls "$SDB/bin" | tr '\n' ' ')"; exit 1; }
    bash "$SRC/scripts/termux/build_ggml_ameva.sh" --ggml-dir "$SD_SRC/ggml" --ggml-lib "$SDB/bin" --out "$OUT" \
        --cxxflags "-D$max_name" --list-devices-cmd "$SDCLI --list-devices" > "$OUT/build.log" 2>&1
    r=$?
    grep -E "\[build_ggml_ameva\]|error|warning" "$OUT/build.log" | grep -v "^WARNING: linker" | head -30
    [ $r -eq 0 ] && [ -f "$OUT/libggml-ameva.so" ] || { echo "PLUGIN_BUILD_FAIL rc=$r"; exit 1; }
    echo "devices: $(GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" "$SDCLI" --list-devices 2>&1 \
        | grep -v "^WARNING: linker" | cut -f1 | tr '\n' ' ')"
}

# run_one TAG VAE_BACKEND: one sd-cli process, one line of results; returns 1 when the run does not pass
run_one() {
    local tag=$1 vae=$2 plugin=() r ok=1 splits prof dispatches
    # the plugin is loaded only where it is the backend under test: the reference runs do not depend on it
    [ "$vae" = AMEVA0 ] && plugin=(GGML_BACKEND_PATH="$OUT/libggml-ameva.so" AMEVA_PROFILE=1)
    rm -f "$RES/$tag.png"
    env GGML_SCHED_DEBUG=1 LD_LIBRARY_PATH="$RT" "${plugin[@]}" timeout "$RUN_TIMEOUT" "$SDCLI" -m "$MODEL" \
        -p "$PROMPT" --steps "$STEPS" --cfg-scale 1 --sampling-method euler -W "$SIZE" -H "$SIZE" --seed "$SEED" \
        -t "$THREADS" --backend "cpu,vae=$vae" --vae-conv-direct -v -o "$RES/$tag.png" \
        > "$RES/$tag.log" 2>&1
    r=$?
    splits=$(grep -c "## SPLIT" "$RES/$tag.log")
    prof=$(grep -o "AMEVA_PROFILE graphs=[0-9]* nodes=[0-9]* dispatches=[0-9]*" "$RES/$tag.log" | tail -1)
    dispatches=${prof##*dispatches=}
    [ $r -eq 0 ] && [ -s "$RES/$tag.png" ] || ok=0
    if [ "$vae" = AMEVA0 ]; then
        [ "$splits" -eq 0 ] && [ -n "$prof" ] && [ "${dispatches:-0}" -gt 0 ] || ok=0
    fi
    echo "RUN tag=$tag vae=$vae rc=$r $([ $ok -eq 1 ] && echo PASS || echo FAIL)" \
         "sampling_s=$(grep -o "sampling completed, taking [0-9.]*" "$RES/$tag.log" | tail -1 | grep -o "[0-9.]*$")" \
         "decode_s=$(grep -o "latent 1 decoded, taking [0-9.]*" "$RES/$tag.log" | tail -1 | grep -o "[0-9.]*$")" \
         "sched_splits=$splits ${prof:+[$prof]}"
    if [ $ok -eq 0 ]; then
        grep -E "## SPLIT|\[ERROR\]|declin|not supported|ABI mismatch|AMU_ERR|Assert|assert|abort|Killed" "$RES/$tag.log" \
            | grep -v "^WARNING: linker" | cut -c1-220 | sort | uniq -c | sort -rn | head -12
        tail -3 "$RES/$tag.log" | cut -c1-220
        return 1
    fi
    return 0
}

run() {
    [ -x "$SDCLI" ] || { echo "NO_SD_CLI $SDCLI (run: sd_vae_e2e.sh build)"; exit 1; }
    [ -f "$OUT/libggml-ameva.so" ] || { echo "NO_PLUGIN $OUT/libggml-ameva.so"; exit 1; }
    [ -f "$MODEL" ] || { echo "NO_MODEL $MODEL"; exit 1; }
    mkdir -p "$RES"
    local have_vk=0 rc=0 e rest a b gate arg
    [ -e "$SDB/bin/libggml-vulkan.so" ] && have_vk=1
    if [ -z "${RUNS:-}" ]; then
        RUNS="cpu_a:cpu cpu_b:cpu ameva:AMEVA0 ameva_b:AMEVA0"
        [ $have_vk -eq 1 ] && RUNS="$RUNS vulkan:Vulkan0"
    fi
    if [ -z "${PAIRS:-}" ]; then
        PAIRS="cpu_a:cpu_b:equal cpu_a:ameva:$PSNR_MIN ameva:ameva_b:-"
        [ $have_vk -eq 1 ] && PAIRS="$PAIRS vulkan:ameva:$PSNR_MIN cpu_a:vulkan:-"
    fi
    echo "conditions: $(date +%F_%T) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status) threads=$THREADS" \
         "model=$(basename "$MODEL") size=${SIZE}x$SIZE steps=$STEPS seed=$SEED psnr_floor_db=$PSNR_MIN"
    for e in $RUNS; do
        run_one "${e%%:*}" "${e#*:}" || rc=1
    done
    for e in $PAIRS; do
        a=${e%%:*} rest=${e#*:}
        b=${rest%%:*} gate=${rest#*:}
        if [ ! -s "$RES/$a.png" ] || [ ! -s "$RES/$b.png" ]; then
            echo "PAIR ref=$a test=$b gate=$gate NO_IMAGE"
            [ "$gate" = - ] || rc=1
            continue
        fi
        case "$gate" in
            -) arg= ;;
            equal) arg=--equal ;;
            *) arg="--min-psnr $gate" ;;
        esac
        if python3 "$SRC/scripts/termux/lab/img_psnr.py" "$RES/$a.png" "$RES/$b.png" $arg > "$RES/pair_${a}_$b.txt" 2>&1; then
            echo "PAIR gate=$gate $([ "$gate" = - ] && echo REPORT || echo PASS) $(sed "s|$RES/||g" "$RES/pair_${a}_$b.txt")"
        else
            echo "PAIR gate=$gate FAIL $(sed "s|$RES/||g" "$RES/pair_${a}_$b.txt" | tail -1)"
            rc=1
        fi
    done
    echo "SD_VAE_E2E $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
    return $rc
}

case "$mode" in
    build) build ;;
    run) run ;;
    all) build && run ;;
    *) echo "usage: sd_vae_e2e.sh [build|run|all]"; exit 2 ;;
esac
