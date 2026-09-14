cmake_minimum_required(VERSION 3.16)
project(inference_app CXX)

find_package(Threads REQUIRED)
add_executable(npu_probe src/npu_probe.cpp)
target_link_libraries(npu_probe PRIVATE hailort Threads::Threads)

add_executable(inference_driver
    src/inference_driver.cpp
    src/util.cpp
)

target_link_libraries(inference_driver PRIVATE
    comm
    hailort
    Threads::Threads
)

# If hailort headers are not in a standard include dir:
target_include_directories(inference_driver PRIVATE
    /usr/include
    /usr/local/include
)

target_link_options(inference_driver PRIVATE
  -static-libstdc++
  -static-libgcc
)
