#!/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
set -u

BUILD_DIR="/data/data/com.termux/files/home/ameva-compute/build-audit"
ORIG_FIXTURES_DIR="/data/data/com.termux/files/home/ameva-compute/tests/fixtures"

echo "======================================================================"
echo "AMEVA Vision - Isolated Failure Injection & Recovery Verification Suite"
echo "======================================================================"

# Create isolated temporary fixture directory in user space (Zero risk to original fixtures)
TMP_PARENT="${PREFIX:-/data/data/com.termux/files/usr}/tmp"
if [ ! -d "$TMP_PARENT" ]; then
    TMP_PARENT="/data/data/com.termux/files/home"
fi
TMP_FIXTURE_DIR=$(mktemp -d -p "$TMP_PARENT" ameva_fixtures_XXXXXX)
cp -r "$ORIG_FIXTURES_DIR"/* "$TMP_FIXTURE_DIR/"

cleanup() {
    rm -rf "$TMP_FIXTURE_DIR"
    unset AMEVA_VISION_FIXTURE_DIR
    unset AMEVA_VISION_CAT_IMAGE_PATH
    unset AMEVA_VISION_FAIL_STAGE
}
trap cleanup EXIT INT TERM HUP

export AMEVA_VISION_FIXTURE_DIR="$TMP_FIXTURE_DIR"

ALL_PASSED=true

# Test 1: Normal execution
echo "--- [TEST 1: NORMAL EXECUTION BASELINE] ---"
"$BUILD_DIR/test_vision_pointwise_conv" > /dev/null 2>&1
CODE=$?
echo "NORMAL_CONV_EXIT_CODE=$CODE"
if [ $CODE -ne 0 ]; then
    echo "FAIL: Normal execution did not exit 0"
    ALL_PASSED=false
fi

CAT_OUT=$("$BUILD_DIR/test_vision_cat_fire2_integration" 2>&1)
CODE=$?
echo "NORMAL_CAT_INTEGRATION_EXIT_CODE=$CODE"
if [ $CODE -ne 0 ]; then
    echo "FAIL: Normal cat integration did not exit 0"
    ALL_PASSED=false
else
    CTX_REC=$(echo "$CAT_OUT" | grep "NORMAL_CONTEXT_CREATE_DESTROY=" | head -n 1)
    BUF_REC=$(echo "$CAT_OUT" | grep "NORMAL_BUFFER_CREATE_DESTROY=" | head -n 1)
    echo "  $CTX_REC"
    echo "  $BUF_REC"
fi

# Test 2: Truncated Bias Fixture (64 bytes -> 16 bytes in isolated tmp dir)
echo ""
echo "--- [TEST 2: TRUNCATED BIAS FIXTURE FAIL-CLOSED] ---"
BIAS_FILE="$TMP_FIXTURE_DIR/squeezenet_fire2_squeeze1x1_bias.bin"
head -c 16 "$ORIG_FIXTURES_DIR/squeezenet_fire2_squeeze1x1_bias.bin" > "$BIAS_FILE"

"$BUILD_DIR/test_vision_pointwise_conv" > /dev/null 2>&1
CODE=$?
# Restore isolated file
cp "$ORIG_FIXTURES_DIR/squeezenet_fire2_squeeze1x1_bias.bin" "$BIAS_FILE"

echo "TRUNCATED_BIAS_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Truncated bias did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    echo "PASS: Truncated bias correctly failed closed (exit code $CODE)"
fi

# Test 3: Truncated Weight Fixture (4096 bytes -> 1024 bytes in isolated tmp dir)
echo ""
echo "--- [TEST 3: TRUNCATED WEIGHT FIXTURE FAIL-CLOSED] ---"
WEIGHT_FILE="$TMP_FIXTURE_DIR/squeezenet_fire2_squeeze1x1_weight.bin"
head -c 1024 "$ORIG_FIXTURES_DIR/squeezenet_fire2_squeeze1x1_weight.bin" > "$WEIGHT_FILE"

"$BUILD_DIR/test_vision_pointwise_conv" > /dev/null 2>&1
CODE=$?
cp "$ORIG_FIXTURES_DIR/squeezenet_fire2_squeeze1x1_weight.bin" "$WEIGHT_FILE"

echo "TRUNCATED_WEIGHT_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Truncated weight did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    echo "PASS: Truncated weight correctly failed closed (exit code $CODE)"
fi

# Test 4: Truncated Input Activation Fixture (802816 bytes -> 1000 bytes in isolated tmp dir)
echo ""
echo "--- [TEST 4: TRUNCATED INPUT ACTIVATION FAIL-CLOSED] ---"
ACT_FILE="$TMP_FIXTURE_DIR/squeezenet_fire2_input_activation_nhwc.bin"
head -c 1000 "$ORIG_FIXTURES_DIR/squeezenet_fire2_input_activation_nhwc.bin" > "$ACT_FILE"

"$BUILD_DIR/test_vision_pointwise_conv" > /dev/null 2>&1
CODE=$?
cp "$ORIG_FIXTURES_DIR/squeezenet_fire2_input_activation_nhwc.bin" "$ACT_FILE"

echo "TRUNCATED_ACTIVATION_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Truncated activation did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    echo "PASS: Truncated activation correctly failed closed (exit code $CODE)"
fi

# Test 5: Missing Cat Image File (using env redirect to non-existent path)
echo ""
echo "--- [TEST 5: MISSING CAT IMAGE FAIL-CLOSED] ---"
export AMEVA_VISION_CAT_IMAGE_PATH="/nonexistent/path/missing_cat_image.png"

"$BUILD_DIR/test_vision_cat_fire2_integration" > /dev/null 2>&1
CODE=$?
unset AMEVA_VISION_CAT_IMAGE_PATH

echo "MISSING_CAT_IMAGE_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Missing cat image did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    echo "PASS: Missing cat image correctly failed closed (exit code $CODE)"
fi

# Test 6: Corrupted Cat Input Activation Fixture (in isolated tmp dir)
echo ""
echo "--- [TEST 6: CORRUPTED CAT INPUT ACTIVATION FIXTURE] ---"
CAT_ACT="$TMP_FIXTURE_DIR/cat_copilot_fire2_input_activation_nhwc.bin"
head -c 500 "$ORIG_FIXTURES_DIR/cat_copilot_fire2_input_activation_nhwc.bin" > "$CAT_ACT"

"$BUILD_DIR/test_vision_pointwise_conv" > /dev/null 2>&1
CODE=$?
cp "$ORIG_FIXTURES_DIR/cat_copilot_fire2_input_activation_nhwc.bin" "$CAT_ACT"

echo "CORRUPTED_CAT_ACT_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Corrupted cat activation did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    echo "PASS: Corrupted cat activation correctly failed closed (exit code $CODE)"
fi

# Test 7: GPU Dispatch Failure & Resource Recovery Verification (P0-3)
echo ""
echo "--- [TEST 7: GPU DISPATCH FAILURE & LIFECYCLE RECOVERY] ---"
export AMEVA_VISION_FAIL_STAGE="dispatch"
FAIL_OUT=$("$BUILD_DIR/test_vision_cat_fire2_integration" 2>&1)
CODE=$?
unset AMEVA_VISION_FAIL_STAGE

echo "DISPATCH_FAILURE_EXIT_CODE=$CODE"
if [ $CODE -eq 0 ]; then
    echo "FAIL: Dispatch failure did not fail-closed (exited 0)"
    ALL_PASSED=false
else
    RECOVERY_LINE=$(echo "$FAIL_OUT" | grep "DISPATCH_FAILURE_RECOVERY=" | head -n 1)
    echo "  $RECOVERY_LINE"
    if echo "$FAIL_OUT" | grep -q "DISPATCH_FAILURE_RECOVERY=PASS"; then
        echo "PASS: GPU Dispatch failure cleanly recovered all resources"
    else
        echo "FAIL: GPU Dispatch failure leaked resources"
        ALL_PASSED=false
    fi
fi

echo ""
echo "======================================================================"
if [ "$ALL_PASSED" = true ]; then
    echo "FAILURE INJECTION SUITE RESULT: ALL PASS (100% FAIL-CLOSED & SAFE LIFECYCLE VERIFIED)"
    exit 0
else
    echo "FAILURE INJECTION SUITE RESULT: FAIL"
    exit 1
fi
