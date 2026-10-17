#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
python3 vision/prepare_assets.py
docker build -f vision/Dockerfile --target runtime -t osh-vision-runtime:v2 .
mkdir -p vision/build
if [ -e vision/build/runtime ]; then
    mv vision/build/runtime "vision/build/runtime.previous.$(date +%s)"
fi
mkdir -p vision/build/runtime
VIEWER_CONTAINER=$(docker create osh-vision-runtime:v2)
trap 'docker rm "$VIEWER_CONTAINER" >/dev/null 2>&1 || true' EXIT INT TERM
docker export "$VIEWER_CONTAINER" | tar -xf - -C vision/build/runtime
docker rm "$VIEWER_CONTAINER" >/dev/null
trap - EXIT INT TERM
BUNDLE="$ROOT/vision/build/runtime"
"$BUNDLE/lib/ld-linux-aarch64.so.1" --library-path "$BUNDLE/lib:$BUNDLE/usr/lib:$BUNDLE/usr/lib/aarch64-linux-gnu:$BUNDLE/lib/aarch64-linux-gnu" "$BUNDLE/app/vision_make_video" vision/assets/people.jpg vision/assets/replay.mp4
for kind in detection segmentation pose; do
    docker build -f vision/Dockerfile --target "$kind" -t "osh-vision-$kind:v2" .
    docker save "osh-vision-$kind:v2" | k3s ctr images import -
done
docker build -t npu-task-controller:vision-v1 npu-task
docker save npu-task-controller:vision-v1 | k3s ctr images import -
