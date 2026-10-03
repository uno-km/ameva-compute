#!/data/data/com.termux/files/usr/bin/bash
# fix_verify.sh TAG [full]
# Verification of a source tree synced to ~/abi-lab/src-TAG, built into ~/abi-lab/out-TAG (kept apart from other
# sessions' src / out directories). Start with nohup; summary in ~/abi-lab/results/fixv_TAG.txt, ending with FIXV_DONE.
#   1. g2_build.sh: kernel header (spirv-val), plugin, ameva_qmm/kvq/chunk/fuse checks, batch KATs
#   2. test-backend-ops with AMEVA_REPACK_ALL=1: SET_ROWS, CPY, FLASH_ATTN_EXT, MUL_MAT ("full": every op)
#   3. pipeline cache: two runs of decode_bench.sh from an empty cache file; kernel creation time of each run and the
#      cache file after each
set -u
TAG=${1:?tag}
MODE=${2:-subset}
LAB=$HOME/abi-lab
export SRC=$LAB/src-$TAG
OUT=$LAB/out-$TAG
S=$LAB/results/fixv_$TAG.txt
mkdir -p "$LAB/results"
cd "$LAB" || exit 1
: > "$S"
bash "$SRC/scripts/termux/lab/g2_build.sh" "$OUT" >> "$S" 2>&1
RT="$OUT"
[ -d "$LAB/rt098" ] && RT="$LAB/rt098:$OUT"
if [ -x "$LAB/tbo/test-backend-ops" ]; then
    if [ "$MODE" = full ]; then
        OPS="ALL"
    else
        OPS="SET_ROWS CPY FLASH_ATTN_EXT MUL_MAT"
    fi
    for o in $OPS; do
        log=$LAB/results/fixv_${TAG}_ra_$o.log
        if [ "$o" = ALL ]; then
            AMEVA_REPACK_ALL=1 GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 \
                "$LAB/tbo/test-backend-ops" -b AMEVA0 > "$log" 2>&1
        else
            AMEVA_REPACK_ALL=1 GGML_BACKEND_PATH="$OUT/libggml-ameva.so" LD_LIBRARY_PATH="$RT" timeout 3600 \
                "$LAB/tbo/test-backend-ops" -o "$o" -b AMEVA0 > "$log" 2>&1
        fi
        rc=$?
        echo "REPACK_ALL $o rc=$rc $(tr -d '\033' < "$log" | grep -E 'tests passed' | tr -s ' ')" >> "$S"
        tr -d '\033' < "$log" | grep FAIL | grep -v "not supported" | head -3 >> "$S"
    done
else
    echo "REPACK_ALL: no ~/abi-lab/tbo/test-backend-ops on this device" >> "$S"
fi
M=$HOME/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
if [ -f "$M" ] && [ -f "$SRC/scripts/termux/lab/decode_bench.sh" ]; then
    C=$HOME/.cache/ameva/ggml-ameva-pipelines.bin
    rm -f "$C"
    for i in 1 2; do
        AMEVA_PROFILE=1 OUT="$OUT" bash "$SRC/scripts/termux/lab/decode_bench.sh" "$M" "fixv_${TAG}_pc$i" gpu > /dev/null 2>&1
        k=$(grep -o "kernel creation: [0-9]* kernels, [0-9.]* ms in total" "$LAB/results/bench_fixv_${TAG}_pc$i.log")
        f=$(ls -la "$C" 2>/dev/null | awk '{print $5}')
        e=$(grep -E "pipeline cache" "$LAB/results/bench_fixv_${TAG}_pc$i.log" | head -2 | tr '\n' ' ')
        echo "PIPELINE_CACHE run=$i ${k:-no kernel line} cache_file_bytes=${f:-none} $e" >> "$S"
    done
fi
echo FIXV_DONE >> "$S"
