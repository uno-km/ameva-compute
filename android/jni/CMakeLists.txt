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
    ${AMEVA_ROOT}/adapters/llama/ameva_backend_bench.cpp
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
# ggml-backend-ameva.cpp #includes third_party/whisper.cpp/ggml/src/ggml-backend-impl.h by a
# hardcoded relative path (not a -I search), so its struct ggml_backend/ggml_backend_i layout
# assumptions come from WHISPER's pinned ggml, not llama.cpp's. An earlier version of this file
# linked this target against llama.cpp's ggml/ggml-base/ggml-cpu (built above) to dodge a
# same-project CMake target-name collision -- that silently builds a backend whose struct offsets
# don't match the ggml-backend.cpp actually running (two different ggml-backend-impl.h layouts in
# one process). Confirmed on-device: GGML_ASSERT(device) in whisper_init_state, backend->device
# NULL for both AMEVA backend instances while CPU's was fine -- i.e. exactly a struct-layout split,
# not a crash in AMEVA's own code. Fix: build whisper's OWN ggml in a fully separate CMake
# configure (ExternalProject_Add), so this target's struct layout and its linked ggml-backend.cpp
# agree -- never reuse llama.cpp's ggml here again.
# ---------------------------------------------------------------------------------------------
set(WHISPER_CPP_SUBMODULE_DIR ${AMEVA_ROOT}/third_party/whisper.cpp)
if(NOT EXISTS ${WHISPER_CPP_SUBMODULE_DIR}/include/whisper.h)
    message(FATAL_ERROR "ameva_stt_jni: third_party/whisper.cpp submodule not checked out "
                        "(run: git submodule update --init third_party/whisper.cpp)")
endif()
# A bare `git submodule update --init` only checks out the pinned commit (610e664b) -- it does NOT
# carry the AMEVA instrumentation (ameva_backend_register() etc.), which lives only as an
# uncommitted working-tree diff inside that submodule (committing it there would pin a commit SHA
# that doesn't exist on the real upstream remote, breaking every other clone's submodule fetch).
# The real, current patch is patches/0004-whisper-ameva-full.patch; apply it before configuring.
file(STRINGS ${WHISPER_CPP_SUBMODULE_DIR}/src/whisper.cpp _ameva_patch_probe REGEX "ameva_backend_register" LIMIT_COUNT 1)
if(NOT _ameva_patch_probe)
    message(FATAL_ERROR "ameva_stt_jni: third_party/whisper.cpp is checked out but the AMEVA patch "
                        "is not applied (run from the repo root: git apply patches/0004-whisper-ameva-full.patch "
                        "--directory=third_party/whisper.cpp)")
endif()

include(ExternalProject)
set(WHISPER_GGML_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/whisper_ggml)
set(WHISPER_GGML_LIB_DIR ${WHISPER_GGML_PREFIX}/src/whisper_ggml_ext-build/ggml/src)
ExternalProject_Add(whisper_ggml_ext
    SOURCE_DIR ${WHISPER_CPP_SUBMODULE_DIR}
    PREFIX ${WHISPER_GGML_PREFIX}
    CMAKE_ARGS
        -G${CMAKE_GENERATOR}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_PLATFORM=${ANDROID_PLATFORM}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_CPU=ON
        -DWHISPER_BUILD_TESTS=OFF
        -DWHISPER_BUILD_EXAMPLES=OFF
        -DWHISPER_BUILD_SERVER=OFF
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ggml ggml-base ggml-cpu
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
        ${WHISPER_GGML_LIB_DIR}/libggml.a
        ${WHISPER_GGML_LIB_DIR}/libggml-base.a
        ${WHISPER_GGML_LIB_DIR}/libggml-cpu.a
)

foreach(_wg_lib ggml ggml-base ggml-cpu)
    add_library(whisper_${_wg_lib} STATIC IMPORTED)
    set_target_properties(whisper_${_wg_lib} PROPERTIES
        IMPORTED_LOCATION ${WHISPER_GGML_LIB_DIR}/lib${_wg_lib}.a)
    add_dependencies(whisper_${_wg_lib} whisper_ggml_ext)
endforeach()

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
    ${WHISPER_CPP_SUBMODULE_DIR}/ggml/include
)

target_compile_definitions(ameva_stt_jni PRIVATE WHISPER_VERSION="v1.8.6-ameva")
# -fopenmp=libomp: ggml-cpu (whisper's own build) needs the OpenMP runtime at final link time.
target_link_libraries(ameva_stt_jni PRIVATE
    whisper_ggml-cpu whisper_ggml-base whisper_ggml vulkan ${log-lib} dl m -fopenmp=libomp)
set_target_properties(ameva_stt_jni PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")

# ---------------------------------------------------------------------------------------------
# v0.4: stable-diffusion.cpp (third_party/stable-diffusion.cpp, pinned commit 50d6405) + the
# ggml-ameva GPU backend (adapters/llama/ggml-ameva -- the SAME backend AmevaLlm loads;
# GGML_OP_CONV_2D/GROUP_NORM/UPSCALE/TIMESTEP_EMBEDDING, the ops a diffusion UNet needs, already
# existed there before this target was added). Loaded as a dynamic plugin exactly like the llama
# path -- new_sd_ctx() resolves its backend ctx-param by NAME against the ggml backend registry
# (src/core/ggml_extend_backend.cpp), so loading AMEVA ourselves first and passing backend=
# "AMEVA0" is enough; no sd.cpp source patch needed for backend selection.
#
# stable-diffusion.cpp pins its OWN ggml fork (leejet/ggml, commit 8e800cef) with a real ABI
# difference from llama.cpp's/whisper.cpp's ggml: GGML_MAX_NAME must be >=160 here (its own code
# static_asserts this; upstream ggml-org/ggml's default is 64) -- confirmed by trying to compile
# sd.cpp's source against llama.cpp's ggml headers first and hitting that assert. So, same lesson
# as whisper.cpp: this needs its OWN isolated ggml build, never llama's or whisper's.
#
# zip (vendored minizip, third_party/stable-diffusion.cpp/thirdparty) is a required dependency of
# the stable-diffusion target itself (target_sources($<TARGET_OBJECTS:zip>)), not something this
# file can skip by hand-listing src/*.cpp -- so the whole upstream CMakeLists.txt is used via
# ExternalProject_Add (not just its ggml/ subdirectory, unlike whisper.cpp, where only the ggml
# half needed isolating).
# ---------------------------------------------------------------------------------------------
set(SDCPP_SUBMODULE_DIR ${AMEVA_ROOT}/third_party/stable-diffusion.cpp)
if(NOT EXISTS ${SDCPP_SUBMODULE_DIR}/include/stable-diffusion.h)
    message(FATAL_ERROR "ameva_sd_jni: third_party/stable-diffusion.cpp submodule not checked out "
                        "(run: git submodule update --init third_party/stable-diffusion.cpp && "
                        "cd third_party/stable-diffusion.cpp && git checkout 50d6405 && "
                        "git submodule update --init ggml)")
endif()

set(SDCPP_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/sdcpp)
set(SDCPP_BUILD_DIR ${SDCPP_PREFIX}/src/sdcpp_ext-build)
ExternalProject_Add(sdcpp_ext
    SOURCE_DIR ${SDCPP_SUBMODULE_DIR}
    PREFIX ${SDCPP_PREFIX}
    CMAKE_ARGS
        -G${CMAKE_GENERATOR}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_PLATFORM=${ANDROID_PLATFORM}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DSD_BUILD_SHARED_LIBS=OFF
        -DSD_BUILD_EXAMPLES=OFF
        -DSD_WEBP=OFF
        -DSD_WEBM=OFF
        -DGGML_CPU=ON
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} stable-diffusion
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
        ${SDCPP_BUILD_DIR}/libstable-diffusion.a
        ${SDCPP_BUILD_DIR}/ggml/src/libggml.a
        ${SDCPP_BUILD_DIR}/ggml/src/libggml-base.a
        ${SDCPP_BUILD_DIR}/ggml/src/libggml-cpu.a
)

foreach(_sd_lib stable-diffusion)
    add_library(sdcpp_${_sd_lib} STATIC IMPORTED)
    set_target_properties(sdcpp_${_sd_lib} PROPERTIES IMPORTED_LOCATION ${SDCPP_BUILD_DIR}/lib${_sd_lib}.a)
    add_dependencies(sdcpp_${_sd_lib} sdcpp_ext)
endforeach()
foreach(_sd_lib ggml ggml-base ggml-cpu)
    add_library(sdcpp_${_sd_lib} STATIC IMPORTED)
    set_target_properties(sdcpp_${_sd_lib} PROPERTIES IMPORTED_LOCATION ${SDCPP_BUILD_DIR}/ggml/src/lib${_sd_lib}.a)
    add_dependencies(sdcpp_${_sd_lib} sdcpp_ext)
endforeach()

# ggml-ameva-sd: a SEPARATE build of the SAME backend source AmevaLlm's ggml-ameva.so uses --
# this one compiled against sd.cpp's isolated ggml (GGML_MAX_NAME=160) instead of llama.cpp's.
# Reusing the already-built llama-path ggml-ameva.so here would be exactly the whisper.cpp bug
# again: a ggml_tensor (whose size depends on GGML_MAX_NAME) built one way, read by code built the
# other way. ac_runtime itself never touches ggml_tensor (it's the Vulkan/OpenCL dispatch layer
# underneath), so the existing SHARED ac_runtime target is safe to reuse as-is.
add_library(ggml-ameva-sd SHARED
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-buffer.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-device.cpp
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva/ggml-ameva-repack.cpp
)
target_include_directories(ggml-ameva-sd PRIVATE
    ${AMEVA_ROOT}/include
    ${AMEVA_ROOT}/adapters/llama/ggml-ameva
    ${SDCPP_SUBMODULE_DIR}/ggml/include
    ${SDCPP_SUBMODULE_DIR}/ggml/src
)
target_compile_definitions(ggml-ameva-sd PRIVATE GGML_MAX_NAME=160)
target_link_libraries(ggml-ameva-sd PRIVATE
    ac_runtime sdcpp_ggml-cpu sdcpp_ggml-base sdcpp_ggml ${log-lib} -fopenmp=libomp)
set_target_properties(ggml-ameva-sd PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")

# ameva_sd_jni itself never links ggml-ameva-sd -- ameva_sd_api.cpp calls ggml_backend_load() on
# it at runtime, a dynamic plugin exactly like the llama path, not a link-time dependency.
add_library(ameva_sd_jni SHARED
    ameva_sd_jni.cpp
    ${AMEVA_ROOT}/adapters/llama/ameva_sd_api.cpp
)

target_include_directories(ameva_sd_jni PRIVATE
    ${AMEVA_ROOT}/include
    ${SDCPP_SUBMODULE_DIR}/include
    ${SDCPP_SUBMODULE_DIR}/ggml/include
)

target_compile_definitions(ameva_sd_jni PRIVATE GGML_MAX_NAME=160)
target_link_libraries(ameva_sd_jni PRIVATE
    sdcpp_stable-diffusion sdcpp_ggml-cpu sdcpp_ggml-base sdcpp_ggml
    vulkan ${log-lib} dl m -fopenmp=libomp)
set_target_properties(ameva_sd_jni PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
