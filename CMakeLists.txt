cmake_minimum_required(VERSION 3.16)
project(osh_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

add_subdirectory(comm)
option(OSH_BUILD_CAMERA "Build GStreamer camera applications" ON)
option(OSH_BUILD_INFERENCE "Build Hailo inference applications" ON)
if(OSH_BUILD_CAMERA)
    add_subdirectory(camera)
endif()
if(OSH_BUILD_INFERENCE)
    add_subdirectory(inference)
endif()
add_subdirectory(image-stream)
