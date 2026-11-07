#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# sd_plugin_ab.sh: one image of sd-cli on AMEVA0 for several builds of the plugin and backends, under the measurement
# protocol of the research program (grade A, ameva-foundation 03_core/docs/research/RESEARCH_PROGRAM.md section 3
# item 7):
#   - VARIANTS (space separated) "<name>:<plugin dir>:<vulkan|opencl>"; the plugin is <plugin dir>/libggml-ameva.so,
#     its ggml in <plugin dir>/ggml-rt (as sd_e2e.sh sets it up).
#   - ROUNDS rounds (default 5), the variants rotated by the round number; before every run wait (at most COOL_MAX s,
#     default 120) until the GPU temperature (/sys/kernel/gpu/gpu_tmu) is at most TMU_MAX (default 42).
#   - the generation of the product adapter: text encoder on the CPU, UNet and VAE on AMEVA0, --diffusion-fa, direct
#     convolutions, --mmap; MODEL (default ~/.cache/termux-diffusion/models/anime.gguf), SIZE (256), STEPS (4),
#     euler, seed 42, -t THREADS (4); AMEVA_STRICT=1 (a graph split or a declined op fails the run).
#   - PROFILE=1 runs with AMEVA_PROFILE=2 and adds the GPU time of CONV_2D and FLASH_ATTN_EXT and the per-graph
#     encode / wait / GPU times; the timestamps cost time, so take the wall times from a pass without it.
# Prints one line per run: SD round= pos= var= rc= sampling_s= decode_s= img= (sha256 of the PNG) tmu0= tmu1= cap0=
# cap1= cool_wait= [conv_ms= fa_ms= per_graph=...]. Images and logs stay in OUT_DIR (default ~/abi-lab/results/sd_ab).
set -u
ROUNDS=${ROUNDS:-5}
TMU_MAX=${TMU_MAX:-42}
COOL_MAX=${COOL_MAX:-120}
MODEL=${MODEL:-$HOME/.cache/termux-diffusion/models/anime.gguf}
SIZE=${SIZE:-256}
STEPS=${STEPS:-4}
THREADS=${THREADS:-4}
SDB=${SDB:-$HOME/abi-lab/sd-build}
OUT_DIR=${OUT_DIR:-$HOME/abi-lab/results/sd_ab}
VARIANTS=${VARIANTS:?VARIANTS}
TMU=/sys/kernel/gpu/gpu_tmu
CAP=/sys/kernel/gpu/gpu_max_clock
rd() { cat "$1" 2>/dev/null | tr -d ' \n' || echo NA; }
mkdir -p "$OUT_DIR"
echo "SD_START $(date +%F_%T) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') model=$(basename "$MODEL")" \
     "size=$SIZE steps=$STEPS rounds=$ROUNDS profile=${PROFILE:-0} variants=\"$VARIANTS\""
read -r -a V <<< "$VARIANTS"
n=${#V[@]}
for r in $(seq 1 "$ROUNDS"); do
    for p in $(seq 0 $((n - 1))); do
        v=${V[$(( (p + r - 1) % n ))]}
        IFS=: read -r name dir api <<< "$v"
        waited=0
        while [ -r "$TMU" ] && [ "$(rd "$TMU" | cut -d. -f1)" -gt "$TMU_MAX" ] && [ "$waited" -lt "$COOL_MAX" ]; do
            sleep 5; waited=$((waited + 5))
        done
        t="$name-r$r"
        t0=$(rd "$TMU"); c0=$(rd "$CAP")
        env AMEVA_BACKEND="$api" AMEVA_STRICT=1 ${PROFILE:+AMEVA_PROFILE=2} GGML_BACKEND_PATH="$dir/libggml-ameva.so" \
            LD_LIBRARY_PATH="$dir/ggml-rt:$dir" timeout 900 "$SDB/bin/sd-cli" -m "$MODEL" \
            -p "a lighthouse on a rocky coast at sunset, detailed painting" --steps "$STEPS" --cfg-scale 1 \
            --sampling-method euler -W "$SIZE" -H "$SIZE" --seed 42 -t "$THREADS" --backend cpu,diffusion=AMEVA0,vae=AMEVA0 \
            --vae-conv-direct --diffusion-conv-direct --diffusion-fa --mmap -v -o "$OUT_DIR/$t.png" > "$OUT_DIR/$t.log" 2>&1 \
            < /dev/null
        rc=$?
        t1=$(rd "$TMU"); c1=$(rd "$CAP")
        smp=$(grep -o 'sampling completed, taking [0-9.]*' "$OUT_DIR/$t.log" | grep -o '[0-9.]*$')
        dec=$(grep -o 'decoded, taking [0-9.]*' "$OUT_DIR/$t.log" | tail -1 | grep -o '[0-9.]*$')
        img=$( [ -f "$OUT_DIR/$t.png" ] && sha256sum < "$OUT_DIR/$t.png" | cut -c1-12 || echo NONE)
        prof=""
        if [ -n "${PROFILE:-}" ]; then
            conv=$(grep 'kind="CONV_2D"' "$OUT_DIR/$t.log" | grep -o 'gpu_ms=[0-9.]*' | cut -d= -f2 | awk '{s += $1} END {print s + 0}')
            fa=$(grep 'kind="FLASH_ATTN_EXT"' "$OUT_DIR/$t.log" | grep -o 'gpu_ms=[0-9.]*' | cut -d= -f2 | awk '{s += $1} END {print s + 0}')
            pg=$(grep -o 'per_graph_ms encode=[0-9.]* wait=[0-9.]* gpu=[0-9.]*' "$OUT_DIR/$t.log" | tr ' ' '_' | tr '\n' ',')
            prof=" conv_ms=$conv fa_ms=$fa per_graph=$pg"
        fi
        echo "SD round=$r pos=$p var=$name rc=$rc sampling_s=${smp:-NA} decode_s=${dec:-NA} img=$img tmu0=$t0 tmu1=$t1" \
             "cap0=$c0 cap1=$c1 cool_wait=$waited$prof"
    done
done
echo "SD_END $(date +%F_%T)"
