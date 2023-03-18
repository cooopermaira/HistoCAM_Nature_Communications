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


IF(APPLE)
	SET(OpenCV_DIR "/opt/local/libexec/opencv4/cmake")
ELSEIF(WINDOWS)
    SET(OpenCV_DIR "C:\Users\CooperMaxS\Documents\opencv\opencv-4.6.0\opencv-4.6.0")
ENDIF()


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )



add_library(pathCamLib source/pathCam.h
                source/Image.cpp
                source/Image.h
                source/MotionEstimator.cpp
                source/MotionEstimator.h
                source/Match.cpp
                source/Match.h
                source/ImageList.cpp
                source/ImageList.h
                source/DescriptorMatcher.h
                source/DescriptorMatcher.cpp
                source/FeatureDetector.h
                source/FeatureDetector.cpp 
                source/BatchCam.h
                source/BatchCam.cpp
                source/util.h)
    
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

add_executable( test source/test.cpp)
add_executable( process_list_truth source/process_list_truth.cpp)
add_executable( process_list_orb source/process_list_orb.cpp)
add_executable( debayer source/debayer.cpp)
add_executable( poco_test source/poco_test.cpp)


target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation Poco::XML Poco::Util)
target_link_libraries(test pathCamLib)
target_link_libraries(process_list_truth pathCamLib)
target_link_libraries(process_list_orb pathCamLib)
target_link_libraries(debayer pathCamLib)
target_link_libraries(poco_test pathCamLib)

