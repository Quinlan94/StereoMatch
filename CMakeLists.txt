cmake_minimum_required(VERSION 3.9)
project(StereoMatch)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(CMAKE_BUILD_TYPE "Realease")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)

add_executable(StereoMatch main.cpp  StereoMatching.h)
target_link_libraries( StereoMatch ${OpenCV_LIBS} )