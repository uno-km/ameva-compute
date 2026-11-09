#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# matrix_run.sh: the experiment matrix of the research handover (2026-11-09, section 4.1) on one phone: our backend, the
# upstream ggml Vulkan backend of the installed 2.8.1 bundle, the CPU, and layer split, rotated over ROUNDS rounds.
#   MATRIX="name,kind,backend,threads,ngl,fa[,VAR=val;VAR=val] ..."
#     kind ameva  : plugin of the hardware-study kit, --device AMEVA0, AMEVA_BACKEND=<backend> (auto, vulkan, opencl)
#     kind ggmlvk : llama-completion and libggml-vulkan of the bundle, --device Vulkan0
#     kind cpu    : --device none -ngl 0
#   MODELS, ROUNDS (default 3), N (default 64), BYTES (default 1500), CTX (default 2048), CAP_TARGET, TMU_MAX, COOL_MAX
# One line per run: MX model= round= var= rc= pp_tps= tg_tps= gen= (sha256 of the generated text) nonascii= head= (start
# of the generated text) tmu0= tmu1= cap0= cap1= valid= cpus=. A run that aborts keeps its rc and the last line of its
# log in err=, so a route that does not work is reported as such. valid=0: the clock cap moved during the run.
set -u
K=$HOME/abi-lab/hwkit/current
D=$HOME/.cache/termux-llamacpp/models
. "$K/src/gpu_sysfs.sh" || { echo "MX FAIL no gpu_sysfs.sh"; exit 2; }
V=$(ls -d "$HOME"/.local/share/ameva/releases/llamacpp/v2.8.1-* 2>/dev/null | tail -1)
O=${O:-$HOME/abi-lab/gap/matrix-$(date +%Y%m%d-%H%M%S)}; mkdir -p "$O"
ROUNDS=${ROUNDS:-3}; N=${N:-64}; BYTES=${BYTES:-1500}; CTX=${CTX:-2048}
TMU_MAX=${TMU_MAX:-42}; COOL_MAX=${COOL_MAX:-300}
CAP_TARGET=${CAP_TARGET:-$(gpu_cap_khz)}
P=$O/prompt_$BYTES.txt; head -c "$BYTES" "$K/src/prompt_corpus.txt" > "$P"
read -r -a VARS <<< "${MATRIX:?MATRIX}"
echo "MX_START $(date +%F_%T) cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') cap_target=$CAP_TARGET bundle=$(basename "${V:-none}") rounds=$ROUNDS n=$N bytes=$BYTES ctx=$CTX kit=$(readlink "$K")"
run_one() { # model-file var-string round
    local mf=$1 vs=$2 r=$3 name kind be th ngl fa envs
    IFS=, read -r name kind be th ngl fa envs <<< "$vs"
    local E=() BIN=llama-completion LIB="$K/rt:$K/bin" BP="$K/bin/libggml-ameva.so" DEV
    [ -n "${envs:-}" ] && IFS=";" read -r -a E <<< "$envs"
    case $kind in
        ameva)  DEV=(--device AMEVA0 -ngl "$ngl"); E+=("AMEVA_BACKEND=$be") ;;
        ggmlvk) [ -n "$V" ] || { echo "MX var=$name rc=SKIP err=no_bundle"; return; }
                BIN=$V/bin/llama-completion; LIB=$V/lib; BP=$V/lib/libggml-vulkan.so; DEV=(--device Vulkan0 -ngl "$ngl") ;;
        cpu)    DEV=(--device none -ngl 0) ;;
        *)      echo "MX var=$name rc=SKIP err=kind_$kind"; return ;;
    esac
    gpu_gate_wait "$CAP_TARGET" "$TMU_MAX" "$COOL_MAX" > /dev/null
    local t0 c0 t1 c1 rc pp tg
    t0=$(gpu_temp_c); c0=$(gpu_cap_khz)
    env "${E[@]}" GGML_BACKEND_PATH="$BP" LD_LIBRARY_PATH="$LIB" timeout "${RUN_MAX:-900}" "$BIN" -m "$mf" "${DEV[@]}" \
        --fit off -fa "$fa" -no-cnv --temp 0 --top-k 1 --seed 42 -t "$th" -f "$P" -n "$N" -c "$CTX" -b 512 -ub 512 \
        > "$O/run.out" 2> "$O/run.log" < /dev/null
    rc=$?
    t1=$(gpu_temp_c); c1=$(gpu_cap_khz)
    local valid=0; [ "$c0" = "$CAP_TARGET" ] && [ "$c1" = "$CAP_TARGET" ] && valid=1
    pp=$(grep "prompt eval time" "$O/run.log" | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
    tg=$(grep -E "[^t] eval time" "$O/run.log" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
    local gen; gen=$(tail -c +"$(( $(wc -c < "$P") + 1 ))" "$O/run.out")
    local h na hd err=""
    h=$(printf %s "$gen" | sha256sum | cut -c1-12)
    na=$(printf %s "$gen" | LC_ALL=C tr -dc '\200-\377' | wc -c)
    hd=$(printf %s "$gen" | tr -c 'A-Za-z0-9 .,' '_' | cut -c1-48)
    [ "$rc" -ne 0 ] && err=$(grep -vE '^\s*$' "$O/run.log" | tail -1 | tr ' =' '__' | cut -c1-100)
    echo "MX model=$(basename "$mf" .gguf) round=$r var=$name rc=$rc pp_tps=${pp:-NA} tg_tps=${tg:-NA} gen=$h nonascii=$na head=$hd tmu0=$t0 tmu1=$t1 cap0=$c0 cap1=$c1 valid=$valid cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') err=$err"
}
n=${#VARS[@]}
for m in ${MODELS:?MODELS}; do
    f=$D/$m.gguf
    [ -s "$f" ] || { echo "MX model=$m rc=SKIP err=not_on_phone"; continue; }
    for r in $(seq 1 "$ROUNDS"); do
        for p in $(seq 0 $((n - 1))); do
            run_one "$f" "${VARS[$(( (p + r - 1) % n ))]}" "$r"
        done
    done
done
echo "MX_END $(date +%F_%T)"
