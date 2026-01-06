#!/bin/bash
# docker run --rm -it -v "$PWD":/workspace -w /workspace ghcr.io/snu-rtos/osh-compile /bin/bash
set -e

pushd ../build/inference
./inference_driver ../../yolov10s.hef
popd