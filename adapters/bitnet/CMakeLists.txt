cmake_minimum_required(VERSION 3.18)
project(ac_adapter_bitnet LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization Flags with strict symbol hidden visibility
if(NOT MSVC)
    add_compile_options(-O3 -fvisibility=hidden -fvisibility-inlines-hidden)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_compile_options(-march=armv8.2-a+dotprod+fp16)
        add_definitions(-D__ARM_NEON -D__ARM_FEATURE_DOTPROD=1)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# CPU path and the AMUDA GPU engine (src/core/ac_bitnet_engine.cpp on ac_runtime; kernels from src/shaders through
# scripts/build_bitnet_kernels.py into src/core/bitnet_kernels_spv.h). On a Termux device:
# scripts/termux/build_bitnet_adapter.sh.
set(BITNET_SOURCES
    src/cpu/ggml_bitnet_mad.cpp
    src/cpu/llama_bitnet_core.cpp
    src/cpu/c_api.cpp
    src/core/ac_bitnet_engine.cpp
)

add_library(ac_adapter_bitnet SHARED ${BITNET_SOURCES})
set_target_properties(ac_adapter_bitnet PROPERTIES
    OUTPUT_NAME "ac_bitnet"
    SOVERSION 1
    VERSION 1.0.0
)
target_compile_definitions(ac_adapter_bitnet PRIVATE AC_BITNET_GPU)
if(NOT MSVC)
    target_compile_options(ac_adapter_bitnet PRIVATE -ffp-contract=off)
endif()
target_link_libraries(ac_adapter_bitnet PRIVATE ac_runtime)

add_executable(ac_bitnet_cli src/cpu/main.cpp)
target_compile_definitions(ac_bitnet_cli PRIVATE AC_BITNET_GPU)
target_link_libraries(ac_bitnet_cli PRIVATE ac_adapter_bitnet)

add_executable(ac_bitnet_parity src/cli/ac_bitnet_parity.cpp)
target_link_libraries(ac_bitnet_parity PRIVATE ac_adapter_bitnet)
