#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build libggml-ameva.so (llama.cpp AMEVA backend) against the ggml of the llama.cpp build that will load it.
#
#   bash scripts/termux/build_ggml_ameva.sh --ggml-dir DIR --ggml-lib LIBDIR [--out DIR]
#
#   --ggml-dir DIR   ggml tree of exactly that llama.cpp version: DIR/include/*.h and DIR/src/ggml-backend-impl.h,
#                    ggml-impl.h (e.g. <llama.cpp checkout>/ggml, or the headers of tag bNNNN)
#   --ggml-lib LIB   directory holding that version's libggml-base.so* and libggml.so* to link against
#   --out DIR        output directory (default build-ggml-ameva); gets libggml-ameva.so and libac_runtime.so.2
#
# ggml-backend-impl.h is not ABI-versioned (GGML_BACKEND_API_VERSION stayed 2 while ggml_backend_i and
# ggml_backend_buffer_i gained members). Headers from another ggml shift every field behind those vtables;
# Galaxy S25 stopped in GGML_ASSERT(device) that way. The plugin checks the layout at load and refuses a
# mismatch; this script additionally loads the result through llama.cpp and requires AMEVA0 to register.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGML_DIR="" GGML_LIB="" OUT="$ROOT/build-ggml-ameva"
while [ $# -gt 0 ]; do
    case "$1" in
        --ggml-dir) GGML_DIR="$2"; shift 2 ;;
        --ggml-lib) GGML_LIB="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "[build_ggml_ameva] ERROR: $*" >&2; exit 2; }
[ -n "$GGML_DIR" ] && [ -n "$GGML_LIB" ] || die "--ggml-dir and --ggml-lib are required (no default: a wrong ggml builds fine and fails at run time)"
[ -f "$GGML_DIR/include/ggml-backend.h" ] || die "$GGML_DIR/include/ggml-backend.h not found"
[ -f "$GGML_DIR/src/ggml-backend-impl.h" ] || die "$GGML_DIR/src/ggml-backend-impl.h not found"
# first match of each glob (no "ls | head": under pipefail, head closing the pipe early fails the pipeline)
first_match() { local f; for f in "$@"; do [ -e "$f" ] && { printf '%s\n' "$f"; return 0; }; done; return 0; }
BASE_LIB="$(first_match "$GGML_LIB"/libggml-base.so*)"
REG_LIB="$(first_match "$GGML_LIB"/libggml.so*)"
[ -n "$BASE_LIB" ] && [ -n "$REG_LIB" ] || die "libggml-base.so / libggml.so not found in $GGML_LIB"
mkdir -p "$OUT"

echo "[build_ggml_ameva] ggml headers: $GGML_DIR"
echo "[build_ggml_ameva] link: $BASE_LIB $REG_LIB"

clang++ -O2 -fPIC -shared -std=c++17 -I "$ROOT/include" -I "$ROOT/src" \
    "$ROOT"/src/backend/vulkan/ac_vk_buffer.cpp "$ROOT"/src/backend/vulkan/ac_vk_context.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_dispatch.cpp "$ROOT"/src/backend/vulkan/ac_vk_event.cpp \
    "$ROOT"/src/backend/vulkan/ac_vk_loader.cpp "$ROOT"/src/backend/vulkan/ac_vk_pipeline.cpp \
    "$ROOT"/src/backend/opencl/ac_cl_backend.cpp \
    "$ROOT"/src/runtime/ac_runtime.cpp "$ROOT"/src/runtime/ac_error.cpp \
    -lvulkan -ldl -lm -Wl,-soname,libac_runtime.so.2 -o "$OUT/libac_runtime.so.2"

A="$ROOT/adapters/llama/ggml-ameva"
clang++ -O2 -fPIC -shared -std=c++20 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
    -Wno-unused-function -I "$GGML_DIR/include" -I "$GGML_DIR/src" -I "$ROOT/include" -I "$A" \
    "$A/ggml-ameva.cpp" "$A/ggml-ameva-device.cpp" "$A/ggml-ameva-buffer.cpp" "$A/ggml-ameva-repack.cpp" \
    "$OUT/libac_runtime.so.2" "$BASE_LIB" "$REG_LIB" -lvulkan -ldl -lm \
    -Wl,-soname,libggml-ameva.so -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined -o "$OUT/libggml-ameva.so"
echo "[build_ggml_ameva] built $OUT/libggml-ameva.so"

# The Termux clang driver puts $PREFIX/lib in front of the -rpath given above. The plugin then looks for
# libac_runtime.so.2 in $PREFIX/lib before its own directory and binds a copy that an earlier installation left there
# (Galaxy S25, 2026-10-04: "dlopen failed: cannot locate symbol", AMEVA0 not registered when llama.cpp loaded the
# plugin without LD_LIBRARY_PATH). $ORIGIN has to come first.
TERMUX_LIB="${PREFIX:-/data/data/com.termux/files/usr}/lib"
runpath="$(readelf -d "$OUT/libggml-ameva.so" | sed -n 's/.*(R\(UN\)\{0,1\}PATH).*\[\(.*\)\]/\2/p')"
case "$runpath" in
    '$ORIGIN'*) ;;
    *)
        if command -v patchelf >/dev/null 2>&1; then
            patchelf --set-rpath "\$ORIGIN:$TERMUX_LIB" "$OUT/libggml-ameva.so"
            echo "[build_ggml_ameva] RUNPATH: \$ORIGIN:$TERMUX_LIB (the linker had written: $runpath)"
        elif [ -e "$TERMUX_LIB/libac_runtime.so.2" ]; then
            die "RUNPATH is '$runpath' ($TERMUX_LIB before \$ORIGIN), patchelf is not installed, and $TERMUX_LIB/libac_runtime.so.2 exists: the plugin would bind that copy instead of $OUT/libac_runtime.so.2. Install patchelf (pkg install patchelf) or remove the copy."
        else
            echo "[build_ggml_ameva] WARNING: RUNPATH is '$runpath' and patchelf is not installed; a libac_runtime.so.2 placed in $TERMUX_LIB later would be bound before the one beside the plugin" >&2
        fi
        ;;
esac

# Load it the way llama.cpp will: the ABI check runs in ggml_backend_init, so a mismatch removes AMEVA0.
if command -v llama-completion >/dev/null 2>&1; then
    # The check must run against the given ggml, not whichever libggml-base.so.0 is active. Only the ggml
    # libraries are exposed (symlinked into one directory): putting a whole lib dir such as $PREFIX/lib on
    # LD_LIBRARY_PATH lets it shadow libraries of /system (Galaxy S25: libunwindstack.so bound Termux liblzma,
    # "cannot locate symbol Xzs_Construct", llama-completion did not start).
    RT="$OUT/ggml-rt"
    rm -rf "$RT" && mkdir -p "$RT"
    for n in libggml-base libggml libggml-cpu libggml-vulkan libggml-opencl; do
        if [ -e "$GGML_LIB/$n.so.0" ]; then ln -s "$(readlink -f "$GGML_LIB/$n.so.0")" "$RT/$n.so.0"; fi
    done
    devs="$(LD_LIBRARY_PATH="$RT:$OUT" GGML_BACKEND_PATH="$OUT/libggml-ameva.so" \
            llama-completion --list-devices 2>&1 || true)"
    # here-strings, not "printf | grep -q": grep -q exits at the first match and printf would fail with SIGPIPE
    if grep -q "AMEVA0" <<< "$devs"; then
        echo "[build_ggml_ameva] llama.cpp registered AMEVA0 from the new library"
    else
        grep -E "ggml-ameva|load_backend" <<< "$devs" >&2 || true
        die "the installed llama.cpp did not register AMEVA0 from $OUT/libggml-ameva.so (ggml version mismatch?)"
    fi
else
    echo "[build_ggml_ameva] llama-completion not on PATH: registration not checked"
fi
