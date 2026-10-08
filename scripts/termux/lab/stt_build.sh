#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the ggml-ameva plugin for the whisper.cpp of the installed ameva-runtime STT bundle and check that AMEVA0
# registers in that ggml. The bundle's whisper-cli calls ggml_backend_load_all(), which loads GGML_BACKEND_PATH, so the
# plugin needs no change to whisper.cpp; whisper-cli selects it with -dev N (N from the list this script prints).
#
#   bash scripts/termux/lab/stt_build.sh --ggml-dir DIR [--bundle DIR] [--out DIR]
#
#   --ggml-dir DIR  ggml tree of the whisper.cpp version of the bundle (include/*.h, src/ggml-backend-impl.h,
#                   src/ggml-impl.h). A tree of another ggml builds and is refused at load by the plugin's ABI check.
#   --bundle DIR    installed STT bundle (default ~/.local/share/ameva/current/stt): bin/whisper-cli, lib/libggml*.so.0
#   --out DIR       output directory (default ~/abi-lab/out-stt)
#
# Output: OUT/libggml-ameva.so, OUT/libac_runtime.so.2, OUT/ggml_list_devices, OUT/ggml-rt/ (links to the bundle ggml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
GGML_DIR="" BUNDLE="$HOME/.local/share/ameva/current/stt" OUT="$HOME/abi-lab/out-stt"
while [ $# -gt 0 ]; do
    case "$1" in
        --ggml-dir) GGML_DIR="$2"; shift 2 ;;
        --bundle) BUNDLE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "[stt_build] ERROR: $*" >&2; exit 2; }
[ -n "$GGML_DIR" ] || die "--ggml-dir is required"
[ -x "$BUNDLE/bin/whisper-cli" ] || die "$BUNDLE/bin/whisper-cli not found"
[ -e "$BUNDLE/lib/libggml.so.0" ] && [ -e "$BUNDLE/lib/libggml-base.so.0" ] || die "bundle ggml libraries not found"
mkdir -p "$OUT"

clang -O2 -I "$GGML_DIR/include" "$ROOT/scripts/termux/lab/ggml_list_devices.c" \
    "$BUNDLE/lib/libggml.so.0" "$BUNDLE/lib/libggml-base.so.0" -o "$OUT/ggml_list_devices"

bash "$ROOT/scripts/termux/build_ggml_ameva.sh" --ggml-dir "$GGML_DIR" --ggml-lib "$BUNDLE/lib" --out "$OUT" \
    --list-devices-cmd "$OUT/ggml_list_devices"

echo "[stt_build] devices (GPU index = whisper-cli -dev N):"
LD_LIBRARY_PATH="$OUT/ggml-rt:$OUT" GGML_BACKEND_PATH="$OUT/libggml-ameva.so" "$OUT/ggml_list_devices"
echo "[stt_build] OK $OUT"
