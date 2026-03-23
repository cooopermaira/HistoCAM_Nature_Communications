cmake_minimum_required(VERSION 3.22)
project(pathCam)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# =========================
# OPTIONS
# =========================
option(BUILD_PATHCAM_GUI "" ON)
option(BUILD_SPINNAKER "" OFF)
option(OVERRRIDE_SPINNAKER "" OFF)

# TensorRT path (user supplied)
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation")

# =========================
# REQUIRED DEPENDENCIES
# =========================

add_subdirectory(poco)
add_subdirectory(JUCE)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# =========================
# OPTIONAL: CUDA
# =========================

find_package(CUDAToolkit QUIET)

set(PATHCAM_HAS_CUDA OFF)
set(PATHCAM_HAS_TENSORRT OFF)
set(PATHCAM_OPENCV_CUDA OFF)

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    set(PATHCAM_HAS_CUDA ON)
    add_compile_definitions(PATHCAM_HAS_CUDA)

    message(STATUS "CUDA found: enabling GPU support")

    # Detect OpenCV CUDA modules
    foreach(lib ${OpenCV_LIBS})
        if (lib MATCHES "opencv_cuda")
            set(PATHCAM_OPENCV_CUDA ON)
            add_compile_definitions(PATHCAM_OPENCV_CUDA)
            message(STATUS "OpenCV CUDA modules detected")
            break()
        endif()
    endforeach()

    # =========================
    # OPTIONAL: TensorRT
    # =========================
    if (TENSORRT_ROOT)
        message(STATUS "Looking for TensorRT in ${TENSORRT_ROOT}")

        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
                PATHS ${TENSORRT_ROOT}/include)

        find_library(TENSORRT_LIB nvinfer
                PATHS ${TENSORRT_ROOT}/lib)

        if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
            set(PATHCAM_HAS_TENSORRT ON)
            add_compile_definitions(PATHCAM_HAS_TENSORRT)

            message(STATUS "TensorRT found")

            include_directories(${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_LIB})
        else()
            message(WARNING "TensorRT not found at provided path")
        endif()
    endif()

else()
    message(STATUS "CUDA not found: building CPU-only version")
endif()

# =========================
# OPTIONAL: SPINNAKER
# =========================

if (NOT OVERRRIDE_SPINNAKER)
    if (EXISTS "/opt/spinnaker")
        set(BUILD_SPINNAKER ON)
        add_compile_definitions(WITH_SPINNAKER)

        include_directories("/opt/spinnaker/include")
        set(SPINNAKER_LIBRARIES
                "/opt/spinnaker/lib/libSpinnaker.so"
                "/opt/spinnaker/lib/libSpinUpdate.so"
                "/opt/spinnaker/lib/libSpinVideo.so")
    endif()
endif()

# =========================
# CLIPPER
# =========================

set(CLIPPER2_UTILS OFF CACHE BOOL "" FORCE)
set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(Clipper2/CPP)

# =========================
# SOURCES
# =========================

file(GLOB PATHCAM_CPP pathcam/*.cpp pathcam/*.h)

set(PATHCAM_FILES ${PATHCAM_CPP})

# Only include CUDA sources if CUDA exists
if (PATHCAM_HAS_CUDA)
    file(GLOB PATHCAM_CU pathcam/*.cu)
    list(APPEND PATHCAM_FILES ${PATHCAM_CU})

    add_subdirectory(CudaSift)
    include_directories(CudaSift)
endif()

# Remove Spinnaker files if disabled
if (NOT BUILD_SPINNAKER)
    list(FILTER PATHCAM_FILES EXCLUDE REGEX SpinPath)
endif()

# =========================
# LIBRARY
# =========================

add_library(pathCamLib STATIC ${PATHCAM_FILES})

include_directories(pathcam)

# Base libraries (always required)
set(PATHCAM_LIBS
        ${OpenCV_LIBS}
        Poco::Foundation
        Poco::XML
        Poco::Util
        Clipper2
)

# Optional CUDA libs
if (PATHCAM_HAS_CUDA)
    list(APPEND PATHCAM_LIBS CUDA::cudart cudasift)
endif()

# Optional TensorRT
if (PATHCAM_HAS_TENSORRT)
    list(APPEND PATHCAM_LIBS ${TENSORRT_LIBRARIES})
endif()

# Optional Spinnaker
if (BUILD_SPINNAKER)
    list(APPEND PATHCAM_LIBS ${SPINNAKER_LIBRARIES})
endif()

target_link_libraries(pathCamLib ${PATHCAM_LIBS})

# =========================
# EXECUTABLES
# =========================

add_executable(test programs/test.cpp)
target_link_libraries(test PRIVATE pathCamLib)

add_executable(process_list_xml programs/process_list_xml.cpp)
target_link_libraries(process_list_xml pathCamLib)

add_executable(test_phase_corr programs/test_phase_corr.cpp)
target_link_libraries(test_phase_corr pathCamLib)

# =========================
# GUI
# =========================

if (BUILD_PATHCAM_GUI)
    include_directories("${PROJECT_SOURCE_DIR}/JUCE/modules" "${PROJECT_SOURCE_DIR}/programs/pathCamApp")

    juce_add_gui_app(pathCamApp VERSION "0.1"
            PRODUCT_NAME "PathCam"
            ICON_BIG "${PROJECT_SOURCE_DIR}/resources/pathCam.png"
    )

    file(GLOB APP_FILES programs/pathCamApp/*.cpp programs/pathCamApp/*.h)

    target_sources(pathCamApp PRIVATE ${APP_FILES})

    target_link_libraries(pathCamApp PRIVATE
            pathCamLib
            juce::juce_core
            juce::juce_gui_extra
            juce::juce_data_structures
            juce::juce_events
            juce::juce_graphics
            juce::juce_opengl
            juce::juce_audio_basics
            juce::juce_audio_devices
            juce::juce_audio_formats
    )

    target_compile_definitions(pathCamApp PRIVATE
            JUCE_DISPLAY_SPLASH_SCREEN=0
            JUCE_WEB_BROWSER=0
            JUCE_USE_CURL=1
    )
endif()