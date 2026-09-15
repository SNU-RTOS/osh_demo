cmake_minimum_required(VERSION 3.16)
project(camera_app CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

add_executable(gpu_compositor_probe
    src/gpu_compositor_probe.cpp
)
target_include_directories(gpu_compositor_probe PRIVATE ${GST_INCLUDE_DIRS})
target_link_directories(gpu_compositor_probe PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(gpu_compositor_probe PRIVATE ${GST_LIBRARIES})
target_compile_options(gpu_compositor_probe PRIVATE ${GST_CFLAGS_OTHER})

add_executable(camera_overlay
    src/camera_overlay.cpp
)

target_link_libraries(camera_overlay PRIVATE
    comm
)

# GStreamer via pkg-config (recommended)
target_include_directories(camera_overlay PRIVATE ${GST_INCLUDE_DIRS})
target_link_directories(camera_overlay PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(camera_overlay PRIVATE ${GST_LIBRARIES})
target_compile_options(camera_overlay PRIVATE ${GST_CFLAGS_OTHER})

target_link_options(camera_overlay PRIVATE
  -static-libstdc++
  -static-libgcc
)
