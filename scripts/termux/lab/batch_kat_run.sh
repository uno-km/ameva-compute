#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# Build libac_runtime from ~/abi-lab/src and run both batch KATs against it.
set -u
LAB=$HOME/abi-lab
S=$LAB/src
O=$LAB/out-rt
mkdir -p "$O"
clang++ -O2 -fPIC -shared -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
    -I "$S/include" -I "$S/src" "$S"/src/backend/vulkan/ac_vk_buffer.cpp "$S"/src/backend/vulkan/ac_vk_context.cpp \
    "$S"/src/backend/vulkan/ac_vk_dispatch.cpp "$S"/src/backend/vulkan/ac_vk_event.cpp "$S"/src/backend/vulkan/ac_vk_loader.cpp \
    "$S"/src/backend/vulkan/ac_vk_pipeline.cpp "$S"/src/runtime/ac_runtime.cpp "$S"/src/runtime/ac_error.cpp \
    -lvulkan -ldl -lm -Wl,-soname,libac_runtime.so.2 -Wl,--no-undefined -o "$O/libac_runtime.so.2" || { echo RT_BUILD_FAIL; exit 1; }
for t in test_batch_api_kat test_batch_graph_kat; do
    clang++ -O2 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers -I "$S/include" "$S/tests/$t.cpp" \
        "$O/libac_runtime.so.2" -lvulkan -Wl,-rpath,"$O" -o "$O/$t" || { echo "BUILD_FAIL $t"; exit 1; }
done
echo "--- test_batch_api_kat"
LD_LIBRARY_PATH="$O" timeout 300 "$O/test_batch_api_kat" | grep -E "dispatches=|FAIL|PASS|REJECTED|Error|error"
echo "rc=${PIPESTATUS[0]}"
echo "--- test_batch_graph_kat"
LD_LIBRARY_PATH="$O" timeout 300 "$O/test_batch_graph_kat"
echo "rc=$?"
