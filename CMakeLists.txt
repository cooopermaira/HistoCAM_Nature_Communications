cmake_minimum_required(VERSION 2.8)
project( pathCam )

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
                source/FeatureDetector.cpp)

add_executable( test source/test.cpp)
add_executable( process_list_truth source/process_list_truth.cpp)


target_link_libraries(pathCamLib ${OpenCV_LIBS})
target_link_libraries( test ${OpenCV_LIBS} pathCamLib)
target_link_libraries( process_list_truth ${OpenCV_LIBS} pathCamLib)


