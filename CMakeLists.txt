cmake_minimum_required(VERSION 3.18)
project(ameva_compute VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and Warning Flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Werror=return-type -O3 -fPIC)
    # Enable ARM NEON if target is ARM64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_compile_options(-march=armv8-a+simd)
    endif()
endif()

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core Library Sources
set(AMUDA_SOURCES
    src/amuda/amuda_context.c
    src/amuda/amuda_memory.c
    src/amuda/amuda_dispatch.c
    src/amuda/amuda_cache.c
    src/amuda/amuda_ops.c
    src/amuda/amuda_gguf.c
    src/amuda/amuda_layer.c
)

# Build Shared and Static Libraries
add_library(amuda_static STATIC ${AMUDA_SOURCES})
set_target_properties(amuda_static PROPERTIES OUTPUT_NAME amuda)

add_library(amuda_shared SHARED ${AMUDA_SOURCES})
set_target_properties(amuda_shared PROPERTIES OUTPUT_NAME amuda)

# Link dynamic linker library on POSIX/Android
if(UNIX AND NOT APPLE)
    target_link_libraries(amuda_shared dl m)
    target_link_libraries(amuda_static dl m)
endif()

# Test Executables
add_executable(test_amuda_uma tests/test_amuda_uma.c)
target_link_libraries(test_amuda_uma amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_uma dl m)
endif()

add_executable(test_amuda_ops tests/test_amuda_ops.c)
target_link_libraries(test_amuda_ops amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_ops dl m)
endif()

add_executable(test_amuda_gguf tests/test_amuda_gguf.c)
target_link_libraries(test_amuda_gguf amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_gguf dl m)
endif()

add_executable(test_amuda_bitnet_exact tests/test_amuda_bitnet_exact.c)
target_link_libraries(test_amuda_bitnet_exact amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_bitnet_exact dl m)
endif()

add_executable(test_amuda_layer_exact tests/test_amuda_layer_exact.c)
target_link_libraries(test_amuda_layer_exact amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_layer_exact dl m)
endif()

add_executable(test_amuda_dispatch_sync tests/test_amuda_dispatch_sync.c)
target_link_libraries(test_amuda_dispatch_sync amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_dispatch_sync dl m)
endif()

include(CTest)
enable_testing()
add_test(NAME amuda_dispatch_sync COMMAND test_amuda_dispatch_sync)
set_tests_properties(amuda_dispatch_sync PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/opt/vendor/lib:$ENV{LD_LIBRARY_PATH}")

add_executable(test_amuda_dispatch_validation tests/test_amuda_dispatch_validation.c)
target_link_libraries(test_amuda_dispatch_validation amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_dispatch_validation dl m)
endif()
add_test(NAME amuda_dispatch_validation COMMAND test_amuda_dispatch_validation)
set_tests_properties(amuda_dispatch_validation PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/opt/vendor/lib:$ENV{LD_LIBRARY_PATH}")

add_executable(test_amuda_gguf_validation tests/test_amuda_gguf_validation.c)
target_link_libraries(test_amuda_gguf_validation amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_gguf_validation dl m)
endif()
add_test(NAME amuda_gguf_validation COMMAND test_amuda_gguf_validation)
set_tests_properties(amuda_gguf_validation PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/opt/vendor/lib:$ENV{LD_LIBRARY_PATH}")

add_executable(test_amuda_gguf_weighted tests/test_amuda_gguf_weighted.c)
target_link_libraries(test_amuda_gguf_weighted amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_gguf_weighted dl m)
endif()
add_test(NAME amuda_gguf_weighted COMMAND test_amuda_gguf_weighted)
set_tests_properties(amuda_gguf_weighted PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/opt/vendor/lib:$ENV{LD_LIBRARY_PATH}")

add_executable(amuda_cli tools/amuda_cli.c)
target_link_libraries(amuda_cli amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(amuda_cli dl m)
endif()

# =============================================================================
# Vulkan Compute Core (AMUDA 2.0 Native Vulkan Backend)
# =============================================================================
set(AC_VK_SOURCES
    src/backend/vulkan/ac_vk_context.cpp
    src/backend/vulkan/ac_vk_buffer.cpp
    src/backend/vulkan/ac_vk_pipeline.cpp
    src/backend/vulkan/ac_vk_dispatch.cpp
    src/backend/vulkan/ac_vk_event.cpp
    src/backend/vulkan/ac_vk_loader.cpp
    src/runtime/ac_runtime.cpp
)

add_library(ac_vulkan_core STATIC ${AC_VK_SOURCES})
target_include_directories(ac_vulkan_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(ac_vulkan_core vulkan dl m)
endif()

# Vulkan Verification Executables
add_executable(test_phase1_vulkan_i2s tests/test_phase1_vulkan_i2s.cpp)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase1_vulkan_i2s vulkan dl m)
endif()

add_executable(test_phase2_vulkan_core tests/test_phase2_vulkan_core.cpp)
target_link_libraries(test_phase2_vulkan_core ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase2_vulkan_core vulkan dl m)
endif()

add_executable(test_phase2_1_failure_injection tests/test_phase2_1_failure_injection.cpp)
target_link_libraries(test_phase2_1_failure_injection ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase2_1_failure_injection vulkan dl m)
endif()

add_executable(test_phase3_universal_backend tests/test_phase3_universal_backend.cpp)
target_link_libraries(test_phase3_universal_backend ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase3_universal_backend vulkan dl m)
endif()

add_executable(test_phase4_runtime_abi tests/test_phase4_runtime_abi.cpp)
target_link_libraries(test_phase4_runtime_abi ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase4_runtime_abi vulkan dl m)
endif()

add_executable(test_phase4b_specialization tests/test_phase4b_specialization.cpp)
target_link_libraries(test_phase4b_specialization ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_phase4b_specialization vulkan dl m)
endif()

add_executable(test_amuda_vulkan_sgemm tests/test_amuda_vulkan_sgemm.cpp)
target_link_libraries(test_amuda_vulkan_sgemm ac_vulkan_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_vulkan_sgemm vulkan dl m)
endif()
add_test(NAME amuda_vulkan_sgemm COMMAND test_amuda_vulkan_sgemm)

add_executable(test_amuda_qwen_parity tests/test_amuda_qwen_parity.cpp)
target_link_libraries(test_amuda_qwen_parity ac_vulkan_core amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_qwen_parity vulkan dl m)
endif()
add_test(NAME amuda_qwen_parity COMMAND test_amuda_qwen_parity)

# =============================================================================
# Native AMEVA GGML Backend & Verification Suite
# =============================================================================
set(LLAMA_CPP_DIR "/data/data/com.termux/files/home/BitNet_ms/3rdparty/llama.cpp" CACHE PATH "Path to pinned llama.cpp repository")

set(AMEVA_GGML_SOURCES
    adapters/llama/ggml-ameva/ggml-ameva.cpp
    adapters/llama/ggml-ameva/ggml-ameva-buffer.cpp
    adapters/llama/ggml-ameva/ggml-ameva-device.cpp
)

add_executable(test_amuda_ggml_backend
    tests/test_amuda_ggml_backend.cpp
    ${AMEVA_GGML_SOURCES}
)

target_include_directories(test_amuda_ggml_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/llama/ggml-ameva
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src
)

target_link_directories(test_amuda_ggml_backend PRIVATE
    ${LLAMA_CPP_DIR}/build/bin
)

target_link_libraries(test_amuda_ggml_backend
    ac_vulkan_core
    amuda_static
    ggml
    ggml-base
    ggml-cpu
)

if(UNIX AND NOT APPLE)
    target_link_libraries(test_amuda_ggml_backend vulkan dl m pthread)
endif()
add_library(ggml-ameva SHARED
    ${AMEVA_GGML_SOURCES}
)
target_include_directories(ggml-ameva PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/llama/ggml-ameva
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src
)
target_link_directories(ggml-ameva PRIVATE
    ${LLAMA_CPP_DIR}/build/bin
)
target_link_libraries(ggml-ameva
    ac_vulkan_core
    ggml
    ggml-base
    ggml-cpu
)
if(UNIX AND NOT APPLE)
    target_link_libraries(ggml-ameva vulkan dl m pthread)
endif()
add_test(NAME amuda_ggml_backend COMMAND test_amuda_ggml_backend)
