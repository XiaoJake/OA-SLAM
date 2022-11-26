cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "-std=c++17 -faligned-new -Wall -Wno-deprecated-declarations -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Protobuf REQUIRED)

# JSON
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(Thirdparty/json)
message(WARNING "Warning !!!! Thirdparty/json forces to use C++11. Please change cxx_std_11 to cxx_std_17 in its Thirdparty/json/CMakeLists.txt:90.")


# Dlib
find_package(dlib REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PROTOBUF_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Osmap.cpp
src/osmap.pb.cc
src/ImageDetections.cc
src/Ellipse.cc
src/Ellipsoid.cc
src/ObjectTrack.cc
src/Distance.cc
src/MapObject.cc
src/OptimizerObject.cc
src/LocalObjectMapping.cc
src/Utils.cc
src/ColorManager.cc
src/Reconstruction.cc
src/Camera.cc
src/Scene.cc
src/p3p.cc
src/Localization.cc
src/ARViewer.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROTOBUF_LIBRARIES}
nlohmann_json::nlohmann_json
stdc++fs
dlib::dlib
)



# Build Examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(oa-slam
Examples/oa-slam.cc)
target_link_libraries(oa-slam ${PROJECT_NAME})

add_executable(oa-slam_localization
Examples/oa-slam_localization.cc)
target_link_libraries(oa-slam_localization ${PROJECT_NAME})

