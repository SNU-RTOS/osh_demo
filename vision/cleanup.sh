#!/bin/sh
set -eu
# Only the three demo tasks; retain measurements, assets and existing dispatchers.
kubectl -n osh-vision delete nputask detection segmentation pose --ignore-not-found
kubectl -n osh-vision wait --for=delete pod -l npu.snu-rtos.io/visual-session=demo --timeout=60s
