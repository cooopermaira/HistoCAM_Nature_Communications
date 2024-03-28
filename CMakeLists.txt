cmake_minimum_required(VERSION 3.2)
project( pathCam )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "" )

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

OPTION(BUILD_IO_TEST "" OFF)
OPTION(BUILD_FLATFIELD_TEST "" OFF)
OPTION(BUILD_DEBAYER "" OFF)
OPTION(BUILD_POCO_TEST "" OFF)
OPTION(BUILD_SFML_TEST "" ON)
OPTION(BUILD_TEST "" OFF)
OPTION(BUILD_NANOGUI_EXAMPLE "" OFF)
OPTION(BUILD_THREAD_TEST "" OFF)
OPTION(BUILD_PATHCAM_GUI "" OFF)
OPTION(OVERRRIDE_SPINNAKER "" OFF)

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
    message(STATUS "Poco and NanoGUI were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
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


#OPTION(NANOGUI_BUILD_EXAMPLES "" OFF)
#OPTION(NANOGUI_BUILD_EXAMPLES "" OFF)
#OPTION(NANOGUI_BUILD_PYTHON "" OFF)
#OPTION(NANOGUI_INSTALL "" OFF)

#add_subdirectory(nanogui)

add_subdirectory(JUCE)

#IF(BUILD_PATHCAM_GUI)
# OPTION(SFML_BUILD_AUDIO "" ON)
#  OPTION(SFML_BUILD_EXAMPLES "" OFF)
#  OPTION(SFML_BUILD_GRAPHICS "" OFF)
#  OPTION(SFML_BUILD_NETWORK "" OFF)
#  OPTION(SFML_BUILD_WINDOW "" OFF)
#  add_subdirectory(SFML)
  
#  message(STATUS ${SFML_INCLUDE_DIRS})
#  message(STATUS ${SFML_LIBRARIES})
# ENDIF()

IF(APPLE)
    SET(OpenCV_DIR "/opt/local/libexec/opencv4/cmake")
 ELSEIF(WIN32)
    SET(OpenCV_DIR "C:/Users/Max Cooper/Documents/opencv/install")
ELSEIF(LINUX)
    SET(OpenCV_DIR "/usr/local/cmake/opencv4")
ENDIF()

INCLUDE_DIRECTORIES(${SFML_INCLUDE_DIR})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

IF(NOT OVERRRIDE_SPINNAKER)

  IF(APPLE)
    IF(EXISTS "/Applications/Spinnaker")
      MESSAGE(STATUS "Spinnaker directory found")
      SET(BUILD_SPINNAKER "" ON)
      add_compile_definitions(WITH_SPINNAKER)
      include_directories( "/Applications/Spinnaker/include" )
      SET(SPINNAKER_LIBRARIES "/Applications/Spinnaker/lib/libSpinnaker.dylib"
                      "/Applications/Spinnaker/lib/libSpinUpdate.dylib"
                      "/Applications/Spinnaker/lib/libSpinVideo.dylib")
  ENDIF()
  ELSEIF(WIN32)
    IF(EXISTS "C:/Program Files/FLIR Systems/Spinnaker")
      MESSAGE(STATUS "Spinnaker directory found")
      SET(BUILD_SPINNAKER "" ON)
      add_compile_definitions(WITH_SPINNAKER)
      include_directories( "C:/Program Files/FLIR Systems/Spinnaker/include" )
      SET(SPINNAKER_LIBRARIES "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/Spinnaker_v140.lib"
                             "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/SpinUpdate_v140.lib"
                             "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015/SpinVideo_v140.lib")
    ENDIF()
  ENDIF()
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(PATCAM_FILES pathcam/pathCam.h
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
                pathcam/StreamCam.h
		pathcam/StreamCam.cpp
		pathcam/ConsecQ.h
		pathcam/ConsecQ.cpp
		pathcam/Runnables.h
		pathcam/MatchRunnable.cpp
		pathcam/DiskStreamer.cpp
		pathcam/JobQueue.h
		pathcam/JobQueue.cpp
		pathcam/QManager.cpp
		pathcam/RegManager.cpp
		pathcam/Loader.cpp
    pathcam/Composite.h
    pathcam/Composite.cpp
    pathcam/CompositeManager.cpp
                pathcam/util.h)
                
IF(BUILD_SPINNAKER)
  set(PATCAM_FILES ${PATCAM_FILES} pathcam/SpinPath.h pathcam/SpinPath.cpp)
ENDIF()

add_library(pathCamLib STATIC ${PATCAM_FILES})

include_directories(pathcam)


target_link_libraries(pathCamLib ${OpenCV_LIBS} Poco::Foundation Poco::XML Poco::Util)

IF(BUILD_SPINNAKER)
  target_link_libraries(pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF()

IF(BUILD_TEST)
	add_executable( test programs/test.cpp)
	target_link_libraries(test pathCamLib)
ENDIF()

add_executable( process_list_xml programs/process_list_xml.cpp)
target_link_libraries(process_list_xml pathCamLib)

add_executable( new_process_list_xml programs/new_process_list_xml.cpp)
target_link_libraries(new_process_list_xml pathCamLib)


IF(BUILD_DEBAYER)
	add_executable( debayer programs/debayer.cpp)
	target_link_libraries(debayer pathCamLib)
ENDIF()

IF(BUILD_POCO_TEST)
	add_executable( poco_test programs/poco_test.cpp)
	target_link_libraries(poco_test pathCamLib)
ENDIF()

IF(BUILD_SFML_TEST)
  add_executable( sfml_test programs/sfml_test.cpp)
  target_link_libraries(sfml_test sfml-audio sfml-system pathCamLib)
ENDIF()

IF(BUILD_THREAD_TEST)
	add_executable( thread_test programs/thread_test.cpp)
	target_link_libraries(thread_test pathCamLib)
ENDIF()

IF(BUILD_IO_TEST)
  add_executable( image_io_test programs/image_io_test.cpp)
  target_link_libraries(image_io_test pathCamLib)
ENDIF()

IF(BUILD_FLATFIELD_TEST)
  add_executable( flatfield_test programs/flatfield_test.cpp)
  target_link_libraries(flatfield_test pathCamLib)
ENDIF()

IF(BUILD_SPINNAKER)
	add_executable( spinnaker_acquisition programs/spinnaker_acquisition.cpp)
	target_link_libraries(spinnaker_acquisition pathCamLib ${SPINNAKER_LIBRARIES})
  add_executable( pathcam_spin_test programs/pathcam_spin_test.cpp)
  target_link_libraries(pathcam_spin_test pathCamLib ${SPINNAKER_LIBRARIES})
ENDIF()

IF(BUILD_XML_TEST)
  add_executable( simple_xml programs/simple_xml.cpp)
  target_link_libraries(simple_xml pathCamLib)
ENDIF()

IF(BUILD_NANOGUI_EXAMPLE)
  add_executable( nanogui_example nanogui/src/example1.cpp)
  target_link_libraries(nanogui_example nanogui pathCamLib)
ENDIF()

IF(BUILD_PATHCAM_GUI)
	include_directories("${PROJECT_SOURCE_DIR}/JUCE/modules")
  juce_add_gui_app(pathCam VERSION "0.1")
  target_sources(${PROJECT_NAME}
    PRIVATE
        programs/JUCE/Main.cpp
        programs/JUCE/MainComponent.h
        programs/JUCE/MainComponent.cpp
        programs/JUCE/JuceHeader.h
  )
  
  target_link_libraries(pathCam PRIVATE
      pathCamLib
      juce::juce_core
      juce::juce_gui_extra
      juce::juce_data_structures
      juce::juce_events
      juce::juce_graphics
      juce::juce_opengl
    )
    
  if(MSVC)
    file(GLOB DLL_FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/*.dll" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/*.pdb")
    
    add_custom_command(TARGET pathCam
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_FILES}
        "${CMAKE_BINARY_DIR}/pathCam_artefacts/RelWithDebInfo/")
  endif()

ENDIF()



