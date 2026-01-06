cmake_minimum_required(VERSION 3.16)
project(osh_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(comm)
add_subdirectory(camera)
add_subdirectory(inference)
