#!/usr/bin/env bash
# ==============================================================================
# AMUDA (AMEVA-COMPUTE) Automated Native Build & Hardware Verification Runner
# Targeted for: Termux Android Linux (ARM64)
# Supported Fleet: Galaxy S25, S21, S20, A35, A53, S22
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "=========================================================="
echo " [AMUDA] Starting GPGPU Native Build & Verification..."
echo " Root Directory: ${ROOT_DIR}"
echo "=========================================================="

# 1. Ensure Dependencies
command -v clang >/dev/null 2>&1 || { echo "[ERROR] clang is not installed. Run: pkg install clang"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake is not installed. Run: pkg install cmake"; exit 1; }

# 2. Configure & Build
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[BUILD] Configuring CMake with ARM64 NEON optimization..."
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "[BUILD] Compiling AMUDA static & shared libraries..."
cmake --build . --config Release -j"$(nproc)"

echo "[BUILD] Compilation successful!"
echo " - Static Lib:  ${BUILD_DIR}/libamuda.a"
echo " - Shared Lib:  ${BUILD_DIR}/libamuda.so"
echo " - Test Binary: ${BUILD_DIR}/test_amuda_uma"

# 3. Setup Hardware Driver Search Paths (Android Vendor Namespace Interop)
export LD_LIBRARY_PATH="/vendor/lib64:/vendor/lib64/egl:${LD_LIBRARY_PATH:-}"

# 4. Execute Hardware Verification
echo ""
echo "=========================================================="
echo " [AMUDA PHASE 1] Launching Hardware UMA & SIMT Verification..."
echo "=========================================================="

"${BUILD_DIR}/test_amuda_uma"

echo ""
echo "=========================================================="
echo " [AMUDA PHASE 2] Launching Tensor Ops & Cache Verification..."
echo "=========================================================="

"${BUILD_DIR}/test_amuda_ops"

BITNET_MODEL="$HOME/.cache/termux-bitnet/models/bitnet-2b-ggml-model-i2_s.gguf"
if [ -f "${BITNET_MODEL}" ]; then
    echo ""
    echo "=========================================================="
    echo " [AMUDA PHASE 3] Launching Real GGUF Model SIMT Verification..."
    echo "=========================================================="
    "${BUILD_DIR}/test_amuda_gguf" "${BITNET_MODEL}"
fi

echo ""
echo "=========================================================="
echo " [AMUDA] All Verification Suites Completed with ZERO ERRORS!"
echo "=========================================================="

