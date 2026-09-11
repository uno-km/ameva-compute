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

if(DEFINED GGML_AMEVA_BUILD_COMMIT)
    add_compile_definitions(GGML_AMEVA_BUILD_COMMIT="${GGML_AMEVA_BUILD_COMMIT}")
endif()


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

# =============================================================================
# AMUDA Vision Ops Subsystem (Track V1 Pointwise Conv)
# =============================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc glslangValidator)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

macro(add_vision_shader SHADER_NAME VAR_NAME)
    set(SRC_COMP "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_NAME}.comp")
    set(GEN_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.spv")
    set(GEN_H "${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}_spv.h")
    set(DST_SPV "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_NAME}.spv")
    set(DST_H "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_NAME}_spv.h")

    if(GLSLC_EXECUTABLE)
        add_custom_command(
            OUTPUT ${GEN_SPV} ${GEN_H}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
            COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute ${SRC_COMP} -o ${GEN_SPV}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py
                    --input ${GEN_SPV} --output ${GEN_H} --var ${VAR_NAME}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GEN_SPV} ${DST_SPV}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GEN_H} ${DST_H}
            DEPENDS ${SRC_COMP} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py
            COMMENT "Compiling SPIR-V shader and generating header: ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND VISION_SHADER_HEADERS ${GEN_H} ${GEN_SPV})
    endif()
endmacro()

add_vision_shader(conv3x3_relu Conv3x3Relu)
add_vision_shader(conv3x3_relu_mutated Conv3x3ReluMutated)
add_vision_shader(concat2_channel Concat2Channel)
add_vision_shader(concat2_channel_mutated Concat2ChannelMutated)
add_vision_shader(pointwise_conv PointwiseConv)
add_vision_shader(pointwise_conv_mutated PointwiseConvMutated)
add_vision_shader(pointwise_conv_relu PointwiseConvRelu)
add_vision_shader(pointwise_conv_relu_mutated PointwiseConvReluMutated)
add_vision_shader(relu Relu)

if(VISION_SHADER_HEADERS)
    add_custom_target(vision_shaders ALL DEPENDS ${VISION_SHADER_HEADERS})
endif()

set(AC_VISION_SOURCES
    src/vision/ac_vision_pointwise_conv.cpp
)

add_library(ac_vision_core STATIC ${AC_VISION_SOURCES})
target_include_directories(ac_vision_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(ac_vision_core
    PUBLIC ac_vulkan_core
)
if(UNIX AND NOT APPLE)
    target_link_libraries(ac_vision_core
        PUBLIC vulkan dl m
    )
endif()
if(TARGET vision_shaders)
    add_dependencies(ac_vision_core vision_shaders)
endif()

add_executable(test_vision_pointwise_conv tests/vision/test_vision_pointwise_conv.cpp)
target_include_directories(test_vision_pointwise_conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_vision_pointwise_conv PRIVATE ac_vision_core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_vision_pointwise_conv PRIVATE pthread)
endif()

add_test(NAME vision_pointwise_conv COMMAND test_vision_pointwise_conv)

# =============================================================================
# AMUDA Vision Diagnostics & Integration Tests (NCNN Integration)
# =============================================================================
option(AMEVA_BUILD_VISION_NCNN_TESTS "Build NCNN-based Vision tests" ON)

if(AMEVA_BUILD_VISION_NCNN_TESTS)
    set(NCNN_DIR "/data/data/com.termux/files/home/ncnn" CACHE PATH "Path to NCNN repository")

    if(NOT EXISTS "${NCNN_DIR}/src/net.h")
        message(FATAL_ERROR "NCNN headers not found at ${NCNN_DIR}/src/net.h")
    endif()
    if(NOT EXISTS "${NCNN_DIR}/build-cpu/src/libncnn.a")
        message(FATAL_ERROR "NCNN static library not found at ${NCNN_DIR}/build-cpu/src/libncnn.a")
    endif()

    find_package(OpenMP QUIET)

    add_executable(audit_ncnn_fire2 scripts/audit_ncnn_fire2.cpp)
    target_include_directories(audit_ncnn_fire2 PRIVATE
        ${NCNN_DIR}/src
        ${NCNN_DIR}/build-cpu/src
    )
    target_link_libraries(audit_ncnn_fire2
        ${NCNN_DIR}/build-cpu/src/libncnn.a
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(audit_ncnn_fire2 OpenMP::OpenMP_CXX)
    else()
        target_compile_options(audit_ncnn_fire2 PRIVATE -fopenmp)
        target_link_libraries(audit_ncnn_fire2 -fopenmp)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(audit_ncnn_fire2 dl m pthread android log)
    endif()

    add_executable(generate_strict_fixtures scripts/generate_strict_fixtures.cpp)
    target_include_directories(generate_strict_fixtures PRIVATE
        ${NCNN_DIR}/src
        ${NCNN_DIR}/build-cpu/src
    )
    target_link_libraries(generate_strict_fixtures
        ${NCNN_DIR}/build-cpu/src/libncnn.a
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(generate_strict_fixtures OpenMP::OpenMP_CXX)
    else()
        target_compile_options(generate_strict_fixtures PRIVATE -fopenmp)
        target_link_libraries(generate_strict_fixtures -fopenmp)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(generate_strict_fixtures dl m pthread android log)
    endif()

    add_executable(test_vision_cat_fire2_integration tests/vision/test_vision_cat_fire2_integration.cpp)
    target_include_directories(test_vision_cat_fire2_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NCNN_DIR}/src
        ${NCNN_DIR}/build-cpu/src
    )
    target_link_libraries(test_vision_cat_fire2_integration
        PRIVATE
        ac_vision_core
        ${NCNN_DIR}/build-cpu/src/libncnn.a
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_vision_cat_fire2_integration PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(test_vision_cat_fire2_integration PRIVATE -fopenmp)
        target_link_libraries(test_vision_cat_fire2_integration PRIVATE -fopenmp)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_vision_cat_fire2_integration PRIVATE dl m pthread android log)
    endif()

    add_test(NAME vision_cat_fire2_integration COMMAND test_vision_cat_fire2_integration)

    add_executable(test_vision_pointwise_conv_relu tests/vision/test_vision_pointwise_conv_relu.cpp)
    target_include_directories(test_vision_pointwise_conv_relu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NCNN_DIR}/src
        ${NCNN_DIR}/build-cpu/src
    )
    target_link_libraries(test_vision_pointwise_conv_relu
        PRIVATE
        ac_vision_core
        ${NCNN_DIR}/build-cpu/src/libncnn.a
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_vision_pointwise_conv_relu PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(test_vision_pointwise_conv_relu PRIVATE -fopenmp)
        target_link_libraries(test_vision_pointwise_conv_relu PRIVATE -fopenmp)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_vision_pointwise_conv_relu PRIVATE dl m pthread android log)
    endif()

    add_test(NAME vision_pointwise_conv_relu COMMAND test_vision_pointwise_conv_relu)

    add_executable(test_vision_fire2_device_resident tests/vision/test_vision_fire2_device_resident.cpp)
    target_include_directories(test_vision_fire2_device_resident PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NCNN_DIR}/src
        ${NCNN_DIR}/build-cpu/src
    )
    target_link_libraries(test_vision_fire2_device_resident
        PRIVATE
        ac_vision_core
        ${NCNN_DIR}/build-cpu/src/libncnn.a
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_vision_fire2_device_resident PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(test_vision_fire2_device_resident PRIVATE -fopenmp)
        target_link_libraries(test_vision_fire2_device_resident PRIVATE -fopenmp)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_vision_fire2_device_resident PRIVATE dl m pthread android log)
    endif()

    add_test(NAME vision_fire2_device_resident COMMAND test_vision_fire2_device_resident)
endif()

