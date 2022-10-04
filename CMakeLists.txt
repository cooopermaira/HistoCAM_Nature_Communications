cmake_minimum_required(VERSION 2.8)
project( pathCam )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Poco REQUIRED Foundation)

add_executable( pathCam source/test.cpp 
								source/Image.cpp 
								source/Image.h)



target_link_libraries( pathCam ${OpenCV_LIBS} Poco::Foundation)


