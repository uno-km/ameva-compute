# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


if (WHISPER_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (WHISPER_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (WHISPER_COREML_ALLOW_FALLBACK)
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (WHISPER_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

#
# libraries
#

# whisper.coreml

if (WHISPER_COREML)
    set(TARGET whisper.coreml)

    add_library(${TARGET}
        coreml/whisper-compat.m
        coreml/whisper-encoder.h
        coreml/whisper-encoder.mm
        coreml/whisper-encoder-impl.h
        coreml/whisper-encoder-impl.m
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC
        .
        )

    target_link_libraries(${TARGET} PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

    set_target_properties(${TARGET} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
    install(TARGETS ${TARGET} LIBRARY)
endif()

if (WHISPER_OPENVINO)
    set(TARGET whisper.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_OPENVINO)

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# whisper

add_library(whisper
            ../include/whisper.h
            whisper-arch.h
            whisper.cpp
            )

# Set the version numbers
set_target_properties(whisper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(whisper PUBLIC . ../include)
target_compile_features   (whisper PUBLIC cxx_std_11) # don't bump

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_BIG_ENDIAN)
endif()

if (WHISPER_EXTRA_FLAGS)
    target_compile_options(whisper PRIVATE ${WHISPER_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(whisper PUBLIC ggml Threads::Threads)

# AMEVA Vulkan STT Hardware Backend & Runtime Library Target
set(AMEVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
if (EXISTS ${AMEVA_ROOT}/adapters/whisper/ggml-backend-ameva.cpp)
    message(STATUS "Integrating formal AMEVA Vulkan STT Backend and ac_runtime into whisper")
    add_library(ac_runtime STATIC
        ${AMEVA_ROOT}/src/runtime/ac_error.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_context.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_buffer.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_pipeline.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_dispatch.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_event.cpp
        ${AMEVA_ROOT}/src/backend/vulkan/ac_vk_loader.cpp
    )
    set_target_properties(ac_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ac_runtime PUBLIC
        ${AMEVA_ROOT}/include
    )
    target_link_libraries(ac_runtime PUBLIC vulkan Threads::Threads)

    target_sources(whisper PRIVATE
        ${AMEVA_ROOT}/adapters/whisper/ggml-backend-ameva.cpp
    )
    target_include_directories(whisper PRIVATE
        ${AMEVA_ROOT}/include
        ${AMEVA_ROOT}/src/backend/vulkan  # ac_vk_sha256.h
        ${AMEVA_ROOT}/adapters/whisper
    )
    target_link_libraries(whisper PRIVATE ac_runtime)
endif()

if (WHISPER_COREML)
    target_link_libraries(whisper PRIVATE whisper.coreml)
endif()

if (WHISPER_OPENVINO)
    target_link_libraries(whisper PRIVATE whisper.openvino)
endif()

if (WHISPER_MKL)
    target_link_libraries(whisper PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD)
endif()
