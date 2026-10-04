#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the ggml-ameva backend for the llama bundle of ameva-runtime, test it, and describe the result.
#
#   bash scripts/termux/build_runtime_backend.sh --ggml-dir DIR --bundle-lib LIBDIR --commit ID [--out DIR]
#
#   --ggml-dir DIR     ggml tree of the llama.cpp version of the bundle (include/*.h, src/ggml-backend-impl.h)
#   --bundle-lib DIR   lib/ directory of the installed llama bundle (libggml-base.so.0, libggml.so.0):
#                      ~/.local/share/ameva/current/llamacpp/lib
#   --commit ID        ameva-compute commit the sources were taken from (the device has a source archive, no git)
#   --out DIR          output directory (default build-runtime-backend)
#
# Output: DIR/libggml-ameva.so, DIR/libac_runtime.so.2, DIR/backend-manifest.json.
# The script stops with a non-zero status when the build, the RUNPATH requirement or a test fails; a manifest is
# written only when everything passed.
#
# Requirements on the two libraries, because they are loaded from inside the bundle (llama.cpp loads the plugin named
# by GGML_BACKEND_PATH; the bundle's own libraries are already in the process):
#   - libggml-ameva.so finds libac_runtime.so.2 beside itself: its RUNPATH starts with $ORIGIN;
#   - neither needs a library outside Android, the Termux libc++ package and the bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGML_DIR="" BUNDLE_LIB="" COMMIT="" OUT="$ROOT/build-runtime-backend"
while [ $# -gt 0 ]; do
    case "$1" in
        --ggml-dir) GGML_DIR="$2"; shift 2 ;;
        --bundle-lib) BUNDLE_LIB="$2"; shift 2 ;;
        --commit) COMMIT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "[build_runtime_backend] ERROR: $*" >&2; exit 2; }
[ -n "$GGML_DIR" ] && [ -n "$BUNDLE_LIB" ] && [ -n "$COMMIT" ] || die "--ggml-dir, --bundle-lib and --commit are required"
[ -e "$BUNDLE_LIB/libggml-base.so.0" ] && [ -e "$BUNDLE_LIB/libggml.so.0" ] || die "$BUNDLE_LIB does not hold libggml-base.so.0 and libggml.so.0"
command -v patchelf >/dev/null 2>&1 || die "patchelf is required (the RUNPATH of the plugin has to start with \$ORIGIN): pkg install patchelf"
rm -rf "$OUT" && mkdir -p "$OUT"

bash "$ROOT/scripts/termux/build_ggml_ameva.sh" --ggml-dir "$GGML_DIR" --ggml-lib "$BUNDLE_LIB" --out "$OUT" > "$OUT/build.log" 2>&1 \
    || { grep -v '^WARNING: linker' "$OUT/build.log" | tail -20 >&2; die "build failed (log: $OUT/build.log)"; }
grep -E '^\[build_ggml_ameva\]' "$OUT/build.log"
warnings="$(grep -c 'warning:' "$OUT/build.log" || true)"

dyn() { readelf -d "$1" | sed -n "s/.*($2).*\[\(.*\)\]/\1/p" | paste -sd' ' -; }
runpath="$(dyn "$OUT/libggml-ameva.so" 'RUNPATH')"
case "$runpath" in '$ORIGIN'*) ;; *) die "RUNPATH of libggml-ameva.so is '$runpath'; it has to start with \$ORIGIN" ;; esac
for f in libggml-ameva.so libac_runtime.so.2; do
    for need in $(dyn "$OUT/$f" 'NEEDED'); do
        case "$need" in
            libac_runtime.so.2|libggml-base.so.0|libggml.so.0|libvulkan.so|libdl.so|libm.so|libc.so|libc++_shared.so|liblog.so) ;;
            *) die "$f needs $need, which neither Android, the Termux libc++ package nor the bundle provides" ;;
        esac
    done
done

# Tests. RT exposes the bundle's ggml libraries (build_ggml_ameva.sh created the links) and the new backend.
RT="$OUT/ggml-rt:$OUT"
GGML_BASE="$BUNDLE_LIB/libggml-base.so.0" GGML_REG="$BUNDLE_LIB/libggml.so.0"
A="$ROOT/adapters/llama/ggml-ameva"
clang++ -O2 -std=c++17 -I "$GGML_DIR/include" -I "$A" "$ROOT/tests/ameva_qmm_check.cpp" "$OUT/libggml-ameva.so" \
    "$GGML_BASE" "$GGML_REG" -Wl,-rpath,"$OUT" -o "$OUT/ameva_qmm_check" 2>> "$OUT/build.log" || die "ameva_qmm_check did not build"
qmm="$(LD_LIBRARY_PATH="$RT" timeout 1800 "$OUT/ameva_qmm_check" 2> "$OUT/qmm.err" | tail -1 || true)"
echo "[build_runtime_backend] $qmm"
case "$qmm" in AMEVA_QMM_CHECK=PASS*) ;; *) die "ameva_qmm_check: ${qmm:-no result line} (stderr: $OUT/qmm.err)" ;; esac

kat=""
for t in test_batch_api_kat test_batch_graph_kat; do
    clang++ -O2 -std=c++17 -I "$ROOT/include" "$ROOT/tests/$t.cpp" "$OUT/libac_runtime.so.2" -lvulkan -Wl,-rpath,"$OUT" \
        -o "$OUT/$t" 2>> "$OUT/build.log" || die "$t did not build"
    LD_LIBRARY_PATH="$OUT" timeout 600 "$OUT/$t" > "$OUT/$t.log" 2>&1 || die "$t exited with a failure (log: $OUT/$t.log)"
    if grep -q 'FAIL' "$OUT/$t.log"; then die "$t reports a failure (log: $OUT/$t.log)"; fi
done
api="$(grep -c -E '^BATCH_.*=(PASS|REJECTED)' "$OUT/test_batch_api_kat.log" || true)"
graph="$(grep -E '^AMUDA_BATCH_GRAPH_KAT=' "$OUT/test_batch_graph_kat.log" || true)"
[ "$api" -ge 5 ] || die "test_batch_api_kat: $api result lines (5 expected)"
case "$graph" in AMUDA_BATCH_GRAPH_KAT=PASS*) ;; *) die "test_batch_graph_kat: ${graph:-no result line}" ;; esac
echo "[build_runtime_backend] batch API KAT: $api result lines passed; $graph"

# Registration through the llama.cpp of the bundle, the way the runtime starts it: no LD_LIBRARY_PATH.
reg="not checked (llama-completion not on PATH)"
if command -v llama-completion >/dev/null 2>&1; then
    devs="$(cd / && env -u LD_LIBRARY_PATH GGML_BACKEND_PATH="$OUT/libggml-ameva.so" llama-completion --list-devices 2>&1 || true)"
    grep -q "AMEVA0" <<< "$devs" || { grep -E "load_backend|ggml-ameva" <<< "$devs" >&2 || true; die "llama-completion did not register AMEVA0 without LD_LIBRARY_PATH"; }
    reg="AMEVA0 registered by $(readlink -f "$(command -v llama-completion)") without LD_LIBRARY_PATH"
    echo "[build_runtime_backend] $reg"
fi

sha() { sha256sum "$1" | cut -d' ' -f1; }
{
    printf '{\n'
    printf '  "ameva_compute_commit": "%s",\n' "$COMMIT"
    printf '  "built_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "built_on": "%s, Android %s",\n' "$(getprop ro.product.model 2>/dev/null || echo unknown)" "$(getprop ro.build.version.release 2>/dev/null || echo unknown)"
    printf '  "compiler": "%s",\n' "$(clang++ --version | head -1)"
    printf '  "patchelf": "%s",\n' "$(patchelf --version 2>&1 | head -1)"
    printf '  "ggml_linked": {"libggml-base.so.0": "%s", "libggml.so.0": "%s"},\n' "$(sha "$(readlink -f "$GGML_BASE")")" "$(sha "$(readlink -f "$GGML_REG")")"
    printf '  "ggml_backend_impl_h_sha256": "%s",\n' "$(sha "$GGML_DIR/src/ggml-backend-impl.h")"
    printf '  "kernel_header_sha256": "%s",\n' "$(sha "$A/ggml_ameva_kernels_spv.h")"
    printf '  "compiler_warnings": %s,\n' "$warnings"
    printf '  "files": {\n'
    printf '    "libggml-ameva.so": {"sha256": "%s", "size": %s, "runpath": "%s", "needed": "%s"},\n' "$(sha "$OUT/libggml-ameva.so")" "$(stat -c %s "$OUT/libggml-ameva.so")" "$runpath" "$(dyn "$OUT/libggml-ameva.so" 'NEEDED')"
    printf '    "libac_runtime.so.2": {"sha256": "%s", "size": %s, "runpath": "%s", "needed": "%s"}\n' "$(sha "$OUT/libac_runtime.so.2")" "$(stat -c %s "$OUT/libac_runtime.so.2")" "$(dyn "$OUT/libac_runtime.so.2" 'RUNPATH')" "$(dyn "$OUT/libac_runtime.so.2" 'NEEDED')"
    printf '  },\n'
    printf '  "tests": {\n'
    printf '    "ameva_qmm_check": "%s",\n' "$qmm"
    printf '    "test_batch_api_kat": "%s result lines passed",\n' "$api"
    printf '    "test_batch_graph_kat": "%s",\n' "$graph"
    printf '    "llama_cpp_registration": "%s"\n' "$reg"
    printf '  }\n'
    printf '}\n'
} > "$OUT/backend-manifest.json"
echo "[build_runtime_backend] manifest: $OUT/backend-manifest.json"
