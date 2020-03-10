cmake_minimum_required(VERSION 3.12)

project(Grab_Cut)
# 添加OpenCV库
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES src/main.cpp src/GrabCut.cpp include/GrabCut.h include/cvui.h include/GCApplication.h include/BorderMatting.h src/BorderMatting.cpp include/GUI.h src/GUI.cpp include/graph.h include/block.h src/maxflow.cpp src/graph.cpp include/KmeansSample.h src/KmeansSample.cpp include/GMM.h src/GMM.cpp src/GCApplication.cpp)

add_executable(Grab_Cut ${SOURCE_FILES})
target_link_libraries(Grab_Cut ${OpenCV_LIBS}) # 链接库