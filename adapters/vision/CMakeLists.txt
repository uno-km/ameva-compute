cmake_minimum_required(VERSION 3.18)
project(ac_adapter_vision LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    add_compile_options(-O3 -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set(VISION_SOURCES
    src/fast_cv.c
    src/fast_cv_engine.cpp
    src/vulkan_cv.cpp
)

# Detect Vulkan library
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR EXISTS "/system/lib64/libvulkan.so")
    find_library(VULKAN_LIB vulkan PATHS /data/data/com.termux/files/usr/lib /system/lib64 /vendor/lib64)
    if(NOT VULKAN_LIB)
        set(VULKAN_LIB "vulkan")
    endif()
    set(VISION_EXTRA_LIBS ${VULKAN_LIB} dl m pthread)
elseif(UNIX)
    find_library(VULKAN_LIB vulkan)
    set(VISION_EXTRA_LIBS ${VULKAN_LIB} dl m pthread)
else()
    set(VISION_EXTRA_LIBS)
endif()

add_library(ac_adapter_vision SHARED ${VISION_SOURCES})
set_target_properties(ac_adapter_vision PROPERTIES 
    OUTPUT_NAME "ac_vision"
    SOVERSION 1
    VERSION 1.0.0
)
if(VISION_EXTRA_LIBS)
    target_link_libraries(ac_adapter_vision PRIVATE ${VISION_EXTRA_LIBS})
endif()
