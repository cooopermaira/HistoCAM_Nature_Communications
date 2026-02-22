cmake_minimum_required(VERSION 3.22)
project(pathCam)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

OPTION(BUILD_IO_TEST "" OFF)
OPTION(BUILD_FLATFIELD_TEST "" OFF)
OPTION(BUILD_DEBAYER "" ON)
OPTION(BUILD_POCO_TEST "" OFF)
OPTION(BUILD_SFML_TEST "" OFF)
OPTION(BUILD_TEST "" OFF)
OPTION(BUILD_NANOGUI_EXAMPLE "" OFF)
OPTION(BUILD_THREAD_TEST "" OFF)
OPTION(BUILD_PATHCAM_GUI "" ON)
OPTION(OVERRRIDE_SPINNAKER "" OFF)
OPTION(BUILD_SPINNAKER "" OFF)


set(CMAKE_SKIP_INSTALL_RULES NO)


OPTION(ENABLE_UTIL "" ON)
OPTION(ENABLE_XML "" ON)
OPTION(ENABLE_FOUNDATION "" ON)

OPTION(ENABLE_ACTIVERECORD "" OFF)
OPTION(ENABLE_ACTIVERECORD_COMPILER "" OFF)
OPTION(ENABLE_CRYPTO "" OFF)
OPTION(ENABLE_DATA "" OFF)
OPTION(ENABLE_DATA_SQLITE "" OFF)
OPTION(ENABLE_JSON "" OFF)
OPTION(ENABLE_JWT "" OFF)
OPTION(ENABLE_MONGODB "" OFF)
OPTION(ENABLE_NET "" OFF)
OPTION(ENABLE_NETSSL "" OFF)
OPTION(ENABLE_NETSSL_WIN "" OFF)
OPTION(ENABLE_PAGECOMPILER "" OFF)
OPTION(ENABLE_PAGECOMPILER_FILE2PAGE "" OFF)
OPTION(ENABLE_PROMETHEUS "" OFF)
OPTION(ENABLE_REDIS "" OFF)
OPTION(ENABLE_ZIP "" OFF)
OPTION(ENABLE_ENCODINGS "" OFF)

add_subdirectory(poco)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 121)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

INCLUDE_DIRECTORIES(${SFML_INCLUDE_DIR})

find_package(CURL REQUIRED)

#list(PREPEND CMAKE_PREFIX_PATH "/usr/local")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CLIPPER2_UTILS OFF CACHE BOOL "" FORCE)
set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)   # ensure static

set(CLIPPER2_HI_PRECISION OFF CACHE BOOL "" FORCE)
add_subdirectory(Clipper2/CPP)

link_directories(${PROJECT_SOURCE_DIR}/TensorRT-10.14.1.48.Linux.aarch64-gnu.cuda-13.0/TensorRT-10.14.1.48/lib)
include_directories(${PROJECT_SOURCE_DIR}/TensorRT-10.14.1.48.Linux.aarch64-gnu.cuda-13.0/TensorRT-10.14.1.48/include)

add_subdirectory(CudaSift)
include_directories(CudaSift)

add_subdirectory(JUCE)

IF (NOT OVERRRIDE_SPINNAKER)
    IF (EXISTS "/opt/spinnaker")
        SET(BUILD_SPINNAKER "" ON)
        add_compile_definitions(WITH_SPINNAKER)
        include_directories("/opt/spinnaker/include")
        SET(SPINNAKER_LIBRARIES "/opt/spinnaker/lib/libSpinnaker.so" "/opt/spinnaker/lib/libSpinUpdate.so" "/opt/spinnaker/lib/libSpinVideo.so")
    endif ()
ENDIF ()

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)



FILE(GLOB PATHCAM_FILES pathcam/*.h pathcam/*.cpp pathcam/*.cu)

IF (NOT BUILD_SPINNAKER)
    list(FILTER PATHCAM_FILES EXCLUDE REGEX SpinPath)
ENDIF ()

add_library(pathCamLib STATIC ${PATHCAM_FILES})

include_directories(pathcam)

target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation Poco::XML Poco::Util CUDA::cudart cudasift nvinfer Clipper2)

IF (BUILD_SPINNAKER)
    target_link_libraries(pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF ()

IF (BUILD_TEST)
    add_executable(test programs/test.cpp)
    target_link_libraries(test pathCamLib)
ENDIF ()

add_executable(old_process_list_xml programs/old_process_list_xml.cpp)
target_link_libraries(old_process_list_xml pathCamLib)

add_executable(process_list_xml programs/process_list_xml.cpp)
target_link_libraries(process_list_xml pathCamLib)

add_executable(test_phase_corr programs/test_phase_corr.cpp)
target_link_libraries(test_phase_corr pathCamLib)

#add_executable(bitwise_semi_norm_dot_product programs/bitwise_semi_norm_dot_product.cpp programs/util.cpp programs/BitwiseImage.cu)
#target_link_libraries(bitwise_semi_norm_dot_product pathCamLib)


IF (BUILD_DEBAYER)
    add_executable(debayer programs/debayer.cpp)
    target_link_libraries(debayer pathCamLib)
ENDIF ()

IF (BUILD_POCO_TEST)
    add_executable(poco_test programs/poco_test.cpp)
    target_link_libraries(poco_test pathCamLib)
ENDIF ()

IF (BUILD_SFML_TEST)
    add_executable(sfml_test programs/sfml_test.cpp)
    target_link_libraries(sfml_test sfml-audio sfml-system pathCamLib)
ENDIF ()

IF (BUILD_THREAD_TEST)
    add_executable(thread_test programs/thread_test.cpp)
    target_link_libraries(thread_test pathCamLib)
ENDIF ()

IF (BUILD_IO_TEST)
    add_executable(image_io_test programs/image_io_test.cpp)
    target_link_libraries(image_io_test pathCamLib)
ENDIF ()

IF (BUILD_FLATFIELD_TEST)
    add_executable(flatfield_test programs/flatfield_test.cpp)
    target_link_libraries(flatfield_test pathCamLib)
ENDIF ()

IF (BUILD_SPINNAKER)
    add_executable(spinnaker_acquisition programs/spinnaker_acquisition.cpp)
    target_link_libraries(spinnaker_acquisition pathCamLib ${SPINNAKER_LIBRARIES})
    add_executable(pathcam_spin_test programs/pathcam_spin_test.cpp)
    target_link_libraries(pathcam_spin_test pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF ()

IF (BUILD_XML_TEST)
    add_executable(simple_xml programs/simple_xml.cpp)
    target_link_libraries(simple_xml pathCamLib)
ENDIF ()

IF (BUILD_PATHCAM_GUI)
    include_directories("${PROJECT_SOURCE_DIR}/JUCE/modules" "${PROJECT_SOURCE_DIR}/programs/pathCamApp")
    juce_add_gui_app(pathCamApp VERSION "0.1"
            PRODUCT_NAME "PathCam"
            ICON_BIG "${PROJECT_SOURCE_DIR}/resources/pathCam.png"
    )

    FILE(GLOB APP_FILES programs/pathCamApp/*.h programs/pathCamApp/*.cpp)
    message(STATUS ${APP_FILES})
    target_sources(pathCamApp
            PRIVATE
            ${APP_FILES}
    )

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
            CURL::libcurl
    )


    target_compile_definitions(pathCamApp PRIVATE JUCE_DISPLAY_SPLASH_SCREEN=0 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=1)

ENDIF ()



