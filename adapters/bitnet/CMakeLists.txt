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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quirks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set(BITNET_GPU_SOURCES
    src/core/vulkan_loader.cpp
    src/core/vulkan_bitnet_engine.cpp
    src/quirks/gpu_quirks.cpp
    src/cpu/ggml_bitnet_mad.cpp
    src/cpu/llama_bitnet_core.cpp
    src/cpu/c_api.cpp
)

# Detect Vulkan library
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR EXISTS "/system/lib64/libvulkan.so")
    find_library(VULKAN_LIB vulkan PATHS /data/data/com.termux/files/usr/lib /system/lib64 /vendor/lib64)
    if(NOT VULKAN_LIB)
        set(VULKAN_LIB "vulkan")
    endif()
    set(BITNET_EXTRA_LIBS ${VULKAN_LIB} dl m pthread)
elseif(UNIX)
    find_library(VULKAN_LIB vulkan)
    set(BITNET_EXTRA_LIBS ${VULKAN_LIB} dl m pthread)
else()
    set(BITNET_EXTRA_LIBS)
endif()

# Build Private Shared Library (libac_bitnet.so)
add_library(ac_adapter_bitnet SHARED ${BITNET_GPU_SOURCES})
set_target_properties(ac_adapter_bitnet PROPERTIES 
    OUTPUT_NAME "ac_bitnet"
    SOVERSION 1
    VERSION 1.0.0
)
if(BITNET_EXTRA_LIBS)
    target_link_libraries(ac_adapter_bitnet PRIVATE ${BITNET_EXTRA_LIBS})
endif()

# Build Standalone CLI Hardware Probe
add_executable(ac_probe_vulkan_gemv
    src/cli/probe_vulkan_gemv.cpp
    src/core/vulkan_loader.cpp
    src/core/vulkan_bitnet_engine.cpp
    src/quirks/gpu_quirks.cpp
)
if(BITNET_EXTRA_LIBS)
    target_link_libraries(ac_probe_vulkan_gemv PRIVATE ${BITNET_EXTRA_LIBS})
endif()
