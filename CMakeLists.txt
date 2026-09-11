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

include(CTest)
enable_testing()

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


link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/build-whisper-cpu/src
    ${CMAKE_CURRENT_SOURCE_DIR}/build-whisper-cpu/ggml/src
)

# =============================================================================
# STT / Whisper GGML Formal Backend Verification Targets
# =============================================================================
set(STT_BACKEND_ADAPTER adapters/whisper/ggml-backend-ameva.cpp)

add_executable(test_ggml_ameva_interface_smoke
    tests/stt/test_ggml_ameva_interface_smoke.cpp
    ${STT_BACKEND_ADAPTER}
)
target_include_directories(test_ggml_ameva_interface_smoke PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/whisper
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/src
)
target_link_libraries(test_ggml_ameva_interface_smoke ac_vulkan_core whisper ggml ggml-cpu ggml-base)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_ggml_ameva_interface_smoke vulkan dl m)
endif()

add_executable(test_whisper_fixture_backend
    tests/stt/test_whisper_fixture_backend.cpp
    ${STT_BACKEND_ADAPTER}
)
target_include_directories(test_whisper_fixture_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/whisper
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/src
)
target_link_libraries(test_whisper_fixture_backend ac_vulkan_core whisper ggml ggml-cpu ggml-base)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_whisper_fixture_backend vulkan dl m)
endif()

add_executable(test_repeat_formal_backend
    tests/stt/test_repeat_formal_backend.cpp
    ${STT_BACKEND_ADAPTER}
)
target_include_directories(test_repeat_formal_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/whisper
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/src
)
target_link_libraries(test_repeat_formal_backend ac_vulkan_core whisper ggml ggml-cpu ggml-base)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_repeat_formal_backend vulkan dl m)
endif()

add_executable(test_turbo_t1_matmul
    tests/stt/test_turbo_t1_matmul.cpp
)
target_include_directories(test_turbo_t1_matmul PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/src
)
target_link_libraries(test_turbo_t1_matmul ac_vulkan_core ggml ggml-cpu ggml-base)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_turbo_t1_matmul vulkan dl m)
endif()

add_test(NAME test_ggml_ameva_interface_smoke COMMAND test_ggml_ameva_interface_smoke)
add_test(NAME test_whisper_fixture_backend COMMAND test_whisper_fixture_backend)
add_test(NAME test_repeat_formal_backend COMMAND test_repeat_formal_backend)
add_test(NAME test_turbo_t1_matmul COMMAND test_turbo_t1_matmul)

