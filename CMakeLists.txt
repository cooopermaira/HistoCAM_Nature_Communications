cmake_minimum_required(VERSION 3.2)
project( pathCam )

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/poco/CMakeLists.txt")
    message(STATUS "Poco was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


set(CMAKE_SKIP_INSTALL_RULES YES)

    
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


OPTION(NANOGUI_BUILD_EXAMPLES "" OFF)
OPTION(NANOGUI_BUILD_EXAMPLES "" OFF)
OPTION(NANOGUI_BUILD_PYTHON "" OFF)
OPTION(NANOGUI_INSTALL "" OFF)

add_subdirectory(nanogui)


IF(APPLE)
  SET(OpenCV_DIR "/opt/local/libexec/opencv4/cmake")
ELSEIF(WINDOWS)
    SET(OpenCV_DIR "C:\Users\CooperMaxS\Documents\opencv\opencv-4.6.0\opencv-4.6.0")
ELSEIF(LINUX)
    SET(OpenCV_DIR "/usr/local/cmake/opencv4")
ENDIF()


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

OPTION(BUILD_SPINNAKER "" OFF)

IF(APPLE)
  IF(EXISTS "/Applications/Spinnaker")
    OPTION(BUILD_SPINNAKER "" ON)
    include_directories( "/Applications/Spinnaker/include" )
    SET(SPINNAKER_LIBRARIES "/Applications/Spinnaker/lib/libSpinnaker.dylib"
                    "/Applications/Spinnaker/lib/libSpinUpdate.dylib"
                    "/Applications/Spinnaker/lib/libSpinVideo.dylib")
  ENDIF()
ENDIF()

IF(BUILD_SPINNAKER)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
ENDIF()

add_library(pathCamLib pathcam/pathCam.h
                pathcam/Image.cpp
                pathcam/Image.h
                pathcam/MotionEstimator.cpp
                pathcam/MotionEstimator.h
                pathcam/Match.cpp
                pathcam/Match.h
                pathcam/OverlapMatrix.cpp
                pathcam/OverlapMatrix.h
                pathcam/ImageList.cpp
                pathcam/ImageList.h
                pathcam/DescriptorMatcher.h
                pathcam/DescriptorMatcher.cpp
                pathcam/FeatureDetector.h
                pathcam/FeatureDetector.cpp 
                pathcam/BatchCam.h
                pathcam/BatchCam.cpp
                pathcam/util.h)

include_directories(pathcam)

add_executable( test programs/test.cpp)
#add_executable( process_list_truth programs/process_list_truth.cpp)
#add_executable( process_list_orb programs/process_list_orb.cpp)
add_executable( process_list_xml programs/process_list_xml.cpp)
add_executable( debayer programs/debayer.cpp)
add_executable( poco_test programs/poco_test.cpp)
add_executable( image_io_test programs/image_io_test.cpp)
IF(BUILD_SPINNAKER)
	add_executable( spinnaker_acquisition programs/spinnaker_acquisition.cpp)
ENDIF()

target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation Poco::XML Poco::Util)
target_link_libraries(test pathCamLib)
#target_link_libraries(process_list_truth pathCamLib)
#target_link_libraries(process_list_orb pathCamLib)
target_link_libraries(process_list_xml pathCamLib)
target_link_libraries(debayer pathCamLib)
target_link_libraries(poco_test pathCamLib)
target_link_libraries(image_io_test pathCamLib)
IF(BUILD_SPINNAKER)
	target_link_libraries(spinnaker_acquisition pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF()
