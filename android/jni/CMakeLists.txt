cmake_minimum_required(VERSION 3.22)
project(ameva_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AMEVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Mirrors AC_VK_SOURCES in ${AMEVA_ROOT}/CMakeLists.txt (ac_runtime shared target). Kept as an
# independent file list -- not an add_subdirectory(${AMEVA_ROOT}) -- so this module does not pull
# in ggml-ameva/adapters/test targets that assume a Termux-side LLAMA_CPP_DIR. Keep in sync by hand.
set(AC_RUNTIME_SOURCES
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_context.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_buffer.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_pipeline.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_dispatch.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_event.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_loader.cpp
    ${AMEVA_ROOT}/src/backend/opencl/ac_cl_backend.cpp
    ${AMEVA_ROOT}/src/runtime/ac_runtime.cpp
    ${AMEVA_ROOT}/src/runtime/ac_error.cpp
    ${AMEVA_ROOT}/src/runtime/ac_security_guard.c
)

# NO_CMAKE_FIND_ROOT_PATH: the NDK toolchain sets CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY, which
# would otherwise make find_path ignore PATHS and search only the NDK sysroot.
find_path(AC_OPENCL_INCLUDE_DIR CL/cl.h PATHS ${AMEVA_ROOT}/include NO_CMAKE_FIND_ROOT_PATH)
if(NOT AC_OPENCL_INCLUDE_DIR)
    message(FATAL_ERROR "ameva_jni: OpenCL headers (CL/cl.h) not found under ${AMEVA_ROOT}/include")
endif()

add_library(ameva_jni SHARED
    ameva_jni.cpp
    ${AC_RUNTIME_SOURCES}
)

target_include_directories(ameva_jni PRIVATE
    ${AMEVA_ROOT}/include
    ${AC_OPENCL_INCLUDE_DIR}
)

find_library(log-lib log)
# NDK provides a libvulkan.so stub in the sysroot since API 24; -lvulkan resolves through it.
target_link_libraries(ameva_jni PRIVATE ${log-lib} vulkan dl m)
