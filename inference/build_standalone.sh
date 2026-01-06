#!/bin/bash
set -e

cd src

docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  osh-compile \
  bash -lc 'g++ inference_driver_standalone.cpp -o inference_driver_standalone -O2 -std=c++17 -pthread \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0) -L/usr/lib -l:libhailort.so \
  -static-libstdc++ -static-libgcc'

cd ..