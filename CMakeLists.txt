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

add_executable(amuda_cli tools/amuda_cli.c)
target_link_libraries(amuda_cli amuda_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(amuda_cli dl m)
endif()




