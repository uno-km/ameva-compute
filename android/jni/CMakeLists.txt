cmake_minimum_required(VERSION 3.22)
project(ameva_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AMEVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Mirrors AC_VK_SOURCES in ${AMEVA_ROOT}/CMakeLists.txt (ac_runtime shared target). Kept as an
# independent file list -- not an add_subdirectory(${AMEVA_ROOT}) -- so this module does not pull
# in adapters/test targets that assume a Termux-side LLAMA_CPP_DIR. Keep in sync by hand.
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

find_library(log-lib log)

# ac_runtime as its own SHARED lib (not compiled directly into ameva_jni) so ggml-ameva.so, loaded
# dynamically at runtime via ggml_backend_load(), resolves the same ac_runtime_* symbols/global
# state (e.g. ac_security_guard's error-generation counters) as ameva_jni -- matching the real
# Termux layout, where libac_runtime.so.2 is a separate file ggml-ameva.so's $ORIGIN RUNPATH finds
# (see tools/packaging/add_ggml_ameva.py in ameva-runtime).
add_library(ac_runtime SHARED ${AC_RUNTIME_SOURCES})
target_include_directories(ac_runtime PUBLIC ${AMEVA_ROOT}/include PRIVATE ${AC_OPENCL_INCLUDE_DIR})
set_target_properties(ac_runtime PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
# NDK provides a libvulkan.so stub in the sysroot since API 24; -lvulkan resolves through it.
target_link_libraries(ac_runtime PRIVATE ${log-lib} vulkan dl m)

# ---------------------------------------------------------------------------------------------
# v0.2: llama.cpp (third_party/llama.cpp, pinned tag b8471) + the ggml-ameva backend plugin.
# ---------------------------------------------------------------------------------------------
set(LLAMA_CPP_SUBMODULE_DIR ${AMEVA_ROOT}/third_party/llama.cpp)
if(NOT EXISTS ${LLAMA_CPP_SUBMODULE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "ameva_jni: third_party/llama.cpp submodule not checked out "
                        "(run: git submodule update --init third_party/llama.cpp)")
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE) # ggml-ameva is the only GPU backend loaded in this process
set(GGML_OPENCL OFF CACHE BOOL "" FORCE) # (see ameva_llm_api.cpp: no upstream-Vulkan FA fallback)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_SUBMODULE_DIR} llama_cpp_build)

# llama.cpp's own CMakeLists.txt (lines 20-21) unconditionally redirects its shared libs into
# <build>/bin/, a directory Gradle/AGP's packaging step never scans (it only harvests the single
# CMAKE_LIBRARY_OUTPUT_DIRECTORY it passed on the cmake command line) -- without this, libllama.so/
# libggml*.so compile fine but silently never reach the APK's jniLibs, surfacing only as an
# UnsatisfiedLinkError at runtime. Force them back to the directory everything else in this file
# actually lands in.
foreach(_ameva_llama_target ggml ggml-base ggml-cpu llama)
    if(TARGET ${_ameva_llama_target})
        set_target_properties(${_ameva_llama_target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )
    endif()
endforeach()

# ggml-ameva: the GPU backend plugin, built as its own shared lib and loaded dynamically at
# runtime via ggml_backend_load() -- never statically linked into ameva_jni or llama.
add_library(ggml-ameva SHARED
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-buffer.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-device.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-repack.cpp
)
target_include_directories(ggml-ameva PRIVATE
    ${AMEVA_ROOT}/include
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva
    ${LLAMA_CPP_SUBMODULE_DIR}/ggml/include
    ${LLAMA_CPP_SUBMODULE_DIR}/ggml/src
)
target_link_libraries(ggml-ameva PRIVATE ac_runtime ggml ggml-base ${log-lib})
set_target_properties(ggml-ameva PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")

add_library(ameva_jni SHARED
    ameva_jni.cpp
    ameva_llm_jni.cpp
    ${AMEVA_ROOT}/adapters/llama/ameva_llm_api.cpp
)

target_include_directories(ameva_jni PRIVATE
    ${AMEVA_ROOT}/include
    ${AC_OPENCL_INCLUDE_DIR}
    ${LLAMA_CPP_SUBMODULE_DIR}/include
    ${LLAMA_CPP_SUBMODULE_DIR}/ggml/include
)

target_link_libraries(ameva_jni PRIVATE ac_runtime llama ggml ggml-base ${log-lib} vulkan dl m)
set_target_properties(ameva_jni PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")

# ---------------------------------------------------------------------------------------------
# v0.3: whisper.cpp (third_party/whisper.cpp, pinned tag v1.8.6 + 2 upstream fixes, commit
# 610e664b) + the formal AMEVA Vulkan STT backend (adapters/whisper/ggml-backend-ameva.cpp),
# statically linked -- ameva_backend_register() self-registers from inside whisper.cpp's own
# whisper_backend_init(), there is no plugin .so to load as there is for the llama path.
#
# Built from raw source files against the SAME ggml/ggml-base/ggml-cpu targets llama.cpp already
# built above -- NOT add_subdirectory(third_party/whisper.cpp), which would pull in whisper's own
# vendored ggml/ copy (colliding by target name with llama.cpp's) and its own embedded
# `ac_runtime` STATIC target (colliding with this file's `ac_runtime` SHARED target above).
# Risk accepted here, not hidden: whisper.cpp's source is pinned a few months before llama.cpp's
# current ggml revision; this is a compile-time-verifiable bet, not a runtime guess -- if the ggml
# scheduler/backend API whisper.cpp calls has moved, this target fails to compile, loudly.
# ---------------------------------------------------------------------------------------------
set(WHISPER_CPP_SUBMODULE_DIR ${AMEVA_ROOT}/third_party/whisper.cpp)
if(NOT EXISTS ${WHISPER_CPP_SUBMODULE_DIR}/include/whisper.h)
    message(FATAL_ERROR "ameva_stt_jni: third_party/whisper.cpp submodule not checked out "
                        "(run: git submodule update --init third_party/whisper.cpp)")
endif()

add_library(ameva_stt_jni SHARED
    ameva_stt_jni.cpp
    ${AMEVA_ROOT}/adapters/whisper/ameva_stt_api.cpp
    ${AMEVA_ROOT}/adapters/whisper/ggml-backend-ameva.cpp
    ${AMEVA_ROOT}/src/runtime/ac_error.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_context.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_buffer.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_pipeline.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_dispatch.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_event.cpp
    ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_loader.cpp
    ${WHISPER_CPP_SUBMODULE_DIR}/src/whisper.cpp
)

target_include_directories(ameva_stt_jni PRIVATE
    ${AMEVA_ROOT}/include
    ${AMEVA_ROOT}/adapters/whisper
    ${AMEVA_ROOT}/src/backend/vulkan # ac_vk_sha256.h
    ${WHISPER_CPP_SUBMODULE_DIR}/include
    ${WHISPER_CPP_SUBMODULE_DIR}/src
    ${LLAMA_CPP_SUBMODULE_DIR}/ggml/include
)

target_link_libraries(ameva_stt_jni PRIVATE ggml ggml-base ggml-cpu vulkan ${log-lib} dl m)
set_target_properties(ameva_stt_jni PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
