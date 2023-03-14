cmake_minimum_required(VERSION 2.8)
project( pathCam )

set(CMAKE_SKIP_INSTALL_RULES YES)

IF(APPLE)
	SET(OpenCV_DIR "/opt/local/libexec/opencv4/cmake")
ENDIF(APPLE)


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
                source/FeatureDetector.cpp )
    
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
OPTION(ENABLE_UTIL "" OFF)
OPTION(ENABLE_ZIP "" OFF)
OPTION(ENABLE_ENCODINGS "" OFF)

add_subdirectory(poco)

add_executable( test source/test.cpp)
add_executable( process_list_truth source/process_list_truth.cpp)
add_executable( process_list_orb source/process_list_orb.cpp)
add_executable( debayer source/debayer.cpp)
add_executable( poco_test source/poco_test.cpp)


target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation)
target_link_libraries( test pathCamLib)
target_link_libraries( process_list_truth pathCamLib)
target_link_libraries( process_list_orb pathCamLib)
target_link_libraries( debayer pathCamLib)
target_link_libraries( poco_test pathCamLib)

