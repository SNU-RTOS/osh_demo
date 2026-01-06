cmake_minimum_required(VERSION 3.16)
project(camera_app CXX)

add_executable(camera_overlay
    src/camera_overlay.cpp
)

target_link_libraries(camera_overlay PRIVATE
    comm
)

# GStreamer via pkg-config (recommended)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

target_include_directories(camera_overlay PRIVATE ${GST_INCLUDE_DIRS})
target_link_directories(camera_overlay PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(camera_overlay PRIVATE ${GST_LIBRARIES})
target_compile_options(camera_overlay PRIVATE ${GST_CFLAGS_OTHER})

target_link_options(camera_overlay PRIVATE
  -static-libstdc++
  -static-libgcc
)