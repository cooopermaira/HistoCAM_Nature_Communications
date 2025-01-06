cmake_minimum_required(VERSION 3.2)
project(pathCam)

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




IF (APPLE)
    SET(JUCE_INSTALL_DESTINATION "lib/cmake/JUCE-7.0.10")
    SET(JUCE_TOOL_INSTALL_DIR "bin/JUCE-7.0.10")
    SET(Torch_DIR "/Users/coopermaira/libtorch")
    SET(OpenCV_DIR "/opt/homebrew/libexec/opencv4/cmake")
    IF (NOT OpenCV_DIR)
        SET(OpenCV_DIR "/opt/local/libexec/opencv4/cmake")
    ENDIF ()
ELSEIF (WIN32)
    SET(Torch_DIR "D:/libtorch-win-shared-with-deps-2.0.1+cu118/libtorch")
    SET(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    SET(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    SET(CUDA_SDK_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    SET(OpenCV_DIR "C:/Users/Max Cooper/documents/opencv/install")
ELSEIF (LINUX)
    SET(OpenCV_DIR "home/opencv/cmake/templates")
ENDIF ()

INCLUDE_DIRECTORIES(${SFML_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#list(APPEND CMAKE_PREFIX_PATH "${CUDA_DIR}")
#message(status "${CMAKE_PREFIX_PATH}")
#find_package(CUDA REQUIRED "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" NO_DEFAULT_PATH)



list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")
find_package(Torch REQUIRED)

add_subdirectory(JUCE)

IF (NOT OVERRRIDE_SPINNAKER)

    IF (APPLE)
        IF (EXISTS "/Applications/Spinnaker")
            #MESSAGE(STATUS "Spinnaker directory found")
            #SET(BUILD_SPINNAKER "" OFF)
            #add_compile_definitions(WITH_SPINNAKER)
            #include_directories( "/Applications/Spinnaker/include" )
            #SET(SPINNAKER_LIBRARIES
            #"/Applications/Spinnaker/lib/libSpinVideo.dylib"
            #"/Applications/Spinnaker/lib/libSpinnaker_C.dylib"
            #"/Applications/Spinnaker/lib/libSpinnaker.dylib"
            #"/Applications/Spinnaker/lib/libSpinUpdate.dylib"
            #"/Applications/Spinnaker/lib/libSpinVideo_C.dylib")
        ENDIF ()
    ELSEIF (WIN32)
        IF (EXISTS "C:/Program Files/FLIR Systems/Spinnaker")
            MESSAGE(STATUS "Spinnaker directory found")
            SET(BUILD_SPINNAKER "" ON)
            add_compile_definitions(WITH_SPINNAKER)
            include_directories("C:/Program Files/FLIR Systems/Spinnaker/include")
            SET(SPINNAKER_LIBRARIES "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/Spinnaker_v140.lib"
                    "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/SpinUpdate_v140.lib"
                    "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/SpinVideo_v140.lib")
        ENDIF ()
    ENDIF ()
ENDIF ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB PATHCAM_FILES pathcam/*.h pathcam/*.cpp)

IF (NOT BUILD_SPINNAKER)
  list(FILTER PATHCAM_FILES EXCLUDE REGEX SpinPath)
ENDIF()

add_library(pathCamLib STATIC ${PATHCAM_FILES})

include_directories(pathcam)


target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation Poco::XML Poco::Util ${TORCH_LIBRARIES})

IF (BUILD_SPINNAKER)
    target_link_libraries(pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF ()

IF (BUILD_TEST)
    add_executable(test programs/test.cpp)
    target_link_libraries(test pathCamLib)
ENDIF ()

IF (BUILD_OLD_LIST)
    add_executable(old_process_list_xml programs/old_process_list_xml.cpp)
    target_link_libraries(old_process_list_xml pathCamLib)
ENDIF ()

add_executable(process_list_xml programs/process_list_xml.cpp)
target_link_libraries(process_list_xml pathCamLib)


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
    )

    target_compile_definitions(pathCamApp PRIVATE JUCE_DISPLAY_SPLASH_SCREEN=0 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

    if (MSVC)
        file(GLOB DLL_FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/*.dll" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/*.pdb")

        add_custom_command(TARGET pathCamApp
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy ${DLL_FILES}
                $<TARGET_FILE_DIR:pathCamApp>)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pathCamApp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pathCamApp>)
  add_custom_command(TARGET debayer
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:debayer>)
    endif ()


ENDIF ()



