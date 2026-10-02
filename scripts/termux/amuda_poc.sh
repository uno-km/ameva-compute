#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# AMUDA CUDA-like POC: build everything from source on the device and run the 5 samples.
#
#   bash scripts/termux/amuda_poc.sh [--backends vulkan,opencl] [--out DIR] [--install-deps]
#                                    [--cl-ldpath PATHS] [--no-cl-ldpath] [--wake-lock] [--no-trace]
#
#   OpenCL samples run with LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/vendor/lib64/egl (sample process only).
#   On Android 14+ the linker namespace of a Termux process refuses /vendor/lib64/libOpenCL.so unless its
#   directory is on LD_LIBRARY_PATH. /system/lib64 must come first: /vendor/lib64 also holds vendor builds of
#   libraries the system already loaded (libbase.so), and a vendor-first path made /system/lib64/libziparchive.so
#   bind to the vendor libbase on Galaxy S25 / Android 16 ("cannot locate symbol
#   _ZN7android4base10MappedFile6CreateEilmi", fleet run 20261001-223548).
#   --cl-ldpath P   use P instead of the default
#   --no-cl-ldpath  do not set LD_LIBRARY_PATH for OpenCL samples
#   --wake-lock     hold termux-wake-lock while samples run (A/B test for per-launch latency outliers)
#   --no-trace      skip the AMUDA_TRACE=1 phase-timing run of vector_add
#
# Output lines that matter (machine readable):
#   AMU_DEVICE ...                one per backend, the GPU that actually ran the kernels
#   AMU_RESULT sample=... status=PASS|FAIL|ERROR ...
#   AMU_SUMMARY backend=... pass=N fail=N error=N
# Exit code: 0 all PASS, 1 at least one wrong result (FAIL), 2 build error or a backend unusable.
# Nothing here falls back silently: a missing tool or backend is printed and counted.
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build-poc"
BACKENDS="vulkan,opencl"
INSTALL_DEPS=0
CL_LDPATH="/system/lib64:/vendor/lib64:/vendor/lib64/egl"
WAKE_LOCK=0
TRACE_RUN=1
while [ $# -gt 0 ]; do
    case "$1" in
        --backends) BACKENDS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        --cl-ldpath) CL_LDPATH="$2"; shift 2 ;;
        --no-cl-ldpath) CL_LDPATH=""; shift ;;
        --wake-lock) WAKE_LOCK=1; shift ;;
        --no-trace) TRACE_RUN=0; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
SAMPLES="vector_add saxpy matmul_tiled reduce_sum histogram256"

say() { printf '[amuda-poc] %s\n' "$*"; }
die() { printf '[amuda-poc] ERROR: %s\n' "$*" >&2; echo "AMU_SUMMARY build=ERROR reason=\"$*\""; exit 2; }

say "root=$ROOT out=$OUT backends=$BACKENDS"
say "model=$(getprop ro.product.model 2>/dev/null || echo ?) soc=$(getprop ro.soc.model 2>/dev/null || getprop ro.board.platform 2>/dev/null || echo ?) android=$(getprop ro.build.version.release 2>/dev/null || echo ?)"

# ---------------------------------------------------------------- dependencies
missing=""
command -v clang   >/dev/null 2>&1 || missing="$missing clang"
command -v python3 >/dev/null 2>&1 || missing="$missing python"
command -v glslangValidator >/dev/null 2>&1 || command -v glslc >/dev/null 2>&1 || missing="$missing glslang"
[ -f "$PREFIX/include/vulkan/vulkan.h" ] || missing="$missing vulkan-headers"
[ -e "$PREFIX/lib/libvulkan.so" ] || missing="$missing vulkan-loader-android"
if [ -n "$missing" ]; then
    if [ "$INSTALL_DEPS" = 1 ]; then
        say "installing:$missing"
        pkg install -y $missing || die "pkg install failed for:$missing"
    else
        die "missing packages:$missing  (run: pkg install$missing  or pass --install-deps)"
    fi
fi
CC=clang
CXX=clang++
say "clang=$($CC --version | head -1)"
say "spirv-tool=$(command -v glslangValidator || command -v glslc)"

mkdir -p "$OUT/lib" "$OUT/bin" "$OUT/gen" "$OUT/logs" || die "cannot create $OUT"
CFLAGS="-O2 -fPIC -Wall -Wextra"

# ---------------------------------------------------------------- 1. AMUDA C ABI (Vulkan) runtime
say "building libac_runtime.so"
$CXX $CFLAGS -shared -std=c++17 -I "$ROOT/include" -I "$ROOT/src" \
    "$ROOT"/src/backend/vulkan/ac_vk_buffer.cpp "$ROOT"/src/backend/vulkan/ac_vk_context.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_dispatch.cpp "$ROOT"/src/backend/vulkan/ac_vk_event.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_loader.cpp "$ROOT"/src/backend/vulkan/ac_vk_pipeline.cpp \
    "$ROOT"/src/runtime/ac_runtime.cpp "$ROOT"/src/runtime/ac_error.cpp \
    -lvulkan -ldl -lm -o "$OUT/lib/libac_runtime.so" > "$OUT/logs/build_ac_runtime.log" 2>&1 \
    || { sed 's/^/BUILDLOG| /' "$OUT/logs/build_ac_runtime.log"; die "libac_runtime.so build failed"; }

# ---------------------------------------------------------------- 2. legacy OpenCL engine
WITH_CL=0
case ",$BACKENDS," in *,opencl,*) WITH_CL=1 ;; esac
if [ "$WITH_CL" = 1 ]; then
    say "building libamuda.so (OpenCL engine)"
    $CC $CFLAGS -shared -std=c11 -I "$ROOT/include" \
        "$ROOT"/src/amuda/amuda_context.c "$ROOT"/src/amuda/amuda_memory.c "$ROOT"/src/amuda/amuda_dispatch.c \
        "$ROOT"/src/amuda/amuda_cache.c "$ROOT"/src/amuda/amuda_ops.c "$ROOT"/src/amuda/amuda_gguf.c \
        "$ROOT"/src/amuda/amuda_layer.c \
        -ldl -lm -o "$OUT/lib/libamuda.so" > "$OUT/logs/build_amuda.log" 2>&1 \
        || { sed 's/^/BUILDLOG| /' "$OUT/logs/build_amuda.log"; die "libamuda.so build failed"; }
fi

# ---------------------------------------------------------------- 3. CUDA-like runtime
say "building libamuda_rt.so"
RT_SRC="$ROOT/src/amuda_rt/amuda_rt.c $ROOT/src/amuda_rt/amuda_rt_vulkan.c"
RT_DEF="-DAMU_WITH_VULKAN=1"
RT_LIB="-L$OUT/lib -lac_runtime"
if [ "$WITH_CL" = 1 ]; then
    RT_SRC="$RT_SRC $ROOT/src/amuda_rt/amuda_rt_opencl.c"
    RT_DEF="$RT_DEF -DAMU_WITH_OPENCL=1"
    RT_LIB="$RT_LIB -lamuda"
fi
$CC $CFLAGS -shared -std=c11 -D_GNU_SOURCE $RT_DEF -I "$ROOT/include" -I "$ROOT/src/amuda_rt" $RT_SRC $RT_LIB \
    -Wl,-rpath,'$ORIGIN' -o "$OUT/lib/libamuda_rt.so" > "$OUT/logs/build_amuda_rt.log" 2>&1 \
    || { sed 's/^/BUILDLOG| /' "$OUT/logs/build_amuda_rt.log"; die "libamuda_rt.so build failed"; }

# ---------------------------------------------------------------- 4. kernels (.acu -> header) and samples
# A failing sample is recorded as BUILD_ERROR (with the compiler log) and the others still build and run.
TARGETS="vulkan"
[ "$WITH_CL" = 1 ] && TARGETS="vulkan,opencl"
BUILT=""
build_err=0
for s in $SAMPLES; do
    if ! python3 "$ROOT/tools/amudacc/amudacc.py" --targets "$TARGETS" --emit-dir "$OUT/gen/$s" \
            -o "$OUT/gen/$s.acu.h" "$ROOT/samples/cuda_like/$s.acu" > "$OUT/logs/amudacc_$s.log" 2>&1; then
        echo "AMU_RESULT sample=$s backend=all status=BUILD_ERROR stage=amudacc"
        sed 's/^/AMUDACC| /' "$OUT/logs/amudacc_$s.log"
        echo "AMUDACC| generated GLSL kept in $OUT/gen/$s"
        build_err=$((build_err+1))
        continue
    fi
    if ! $CC -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I "$ROOT/include" -I "$OUT/gen" -I "$ROOT/samples/cuda_like" \
            "$ROOT/samples/cuda_like/${s}_main.c" -L"$OUT/lib" -lamuda_rt -lm -Wl,-rpath,'$ORIGIN/../lib' \
            -o "$OUT/bin/$s" > "$OUT/logs/build_$s.log" 2>&1; then
        echo "AMU_RESULT sample=$s backend=all status=BUILD_ERROR stage=host_cc"
        sed 's/^/CC| /' "$OUT/logs/build_$s.log"
        build_err=$((build_err+1))
        continue
    fi
    BUILT="$BUILT $s"
done
say "built:$BUILT  build_errors=$build_err"

# ---------------------------------------------------------------- 5. run
# run_sample BACKEND BINARY [ARGS...]: environment is applied to the sample process only (not to timeout/env)
run_sample() {
    local b="$1"; shift
    if [ "$b" = opencl ] && [ -n "$CL_LDPATH" ]; then
        timeout 300 env AMUDA_BACKEND="$b" LD_LIBRARY_PATH="$CL_LDPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
    else
        timeout 300 env AMUDA_BACKEND="$b" "$@"
    fi
}

if [ "$WAKE_LOCK" = 1 ]; then
    if command -v termux-wake-lock >/dev/null 2>&1; then
        termux-wake-lock && say "wake_lock=on"
        trap 'termux-wake-unlock >/dev/null 2>&1 || true' EXIT
    else
        say "wake_lock=unavailable (termux-wake-lock not found); running without it"
    fi
else
    say "wake_lock=off"
fi
[ "$WITH_CL" = 1 ] && say "opencl LD_LIBRARY_PATH=${CL_LDPATH:-<unchanged, --no-cl-ldpath>}"

worst=0
IFS=',' read -r -a BL <<< "$BACKENDS"
for b in "${BL[@]}"; do
    pass=0; fail=0; err=0
    for s in $BUILT; do
        log="$OUT/logs/run_${b}_${s}.log"
        t_start="$(date '+%m-%d %H:%M:%S.000')"
        run_sample "$b" "$OUT/bin/$s" > "$log" 2>&1
        rc=$?
        dev_line="$(grep -m1 '^\[' "$log" || true)"
        res_line="$(grep '^AMU_RESULT' "$log" | tail -n 1 || true)"
        [ -n "$dev_line" ] && [ "$s" = vector_add ] && echo "AMU_DEVICE backend=$b $dev_line"
        # The exit code decides, never the printed text: PASS needs rc=0 AND a PASS line, FAIL (wrong numbers)
        # needs rc=1 AND a FAIL line. Anything else (crash, linker error, teardown error) is ERROR.
        if [ $rc -eq 0 ] && [[ "$res_line" == *" status=PASS "* ]]; then
            echo "$res_line"; pass=$((pass+1))
        elif [ $rc -eq 1 ] && [[ "$res_line" == *" status=FAIL "* ]]; then
            echo "$res_line"; fail=$((fail+1))
            sed -n '1,40p' "$log" | sed "s/^/RUNLOG|$b|$s| /"
        else
            sig=""; [ $rc -gt 128 ] && sig=" signal=$((rc-128))"
            echo "AMU_RESULT sample=$s backend=$b status=ERROR rc=$rc$sig"
            err=$((err+1))
            sed -n '1,40p' "$log" | sed "s/^/RUNLOG|$b|$s| /"
            if [ $rc -gt 128 ]; then
                # native crash: the tombstone summary (fault + backtrace) from the crash log buffer
                if crash="$(logcat -d -b crash -T "$t_start" 2>&1)"; then
                    printf '%s\n' "$crash" | grep -E 'Fatal signal|signal [0-9]+ \(|#[0-9]+ pc ' | head -n 24 \
                        | sed "s/^/CRASH|$b|$s| /"
                else
                    echo "CRASH|$b|$s| logcat -b crash unavailable: $(printf '%s' "$crash" | head -n 1)"
                fi
            fi
        fi
    done
    echo "AMU_SUMMARY backend=$b pass=$pass fail=$fail error=$err build_error=$build_err"
    [ $fail -gt 0 ] && worst=1
    [ $err -gt 0 ] && [ $worst -eq 0 ] && worst=2
    # phase timing of every launch (warm-up + 10): which host/driver step owns the wall time
    if [ "$TRACE_RUN" = 1 ] && [ $pass -gt 0 ] && case " $BUILT " in *" vector_add "*) true ;; *) false ;; esac; then
        tlog="$OUT/logs/trace_${b}_vector_add.log"
        AMUDA_TRACE=1 run_sample "$b" "$OUT/bin/vector_add" > "$tlog" 2>&1
        grep '^AMU_TRACE' "$tlog" | sed "s/^/TRACE|$b|vector_add| /"
    fi
done
[ $build_err -gt 0 ] && [ $worst -eq 0 ] && worst=2
exit $worst
