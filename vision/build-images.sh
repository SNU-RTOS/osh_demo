#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RELEASE=${OSH_VISION_RELEASE:-v3}
cd "$ROOT"
python3 vision/prepare_assets.py
docker build -f vision/Dockerfile --target runtime -t "osh-vision-runtime:$RELEASE" .
mkdir -p vision/build
if [ -e vision/build/runtime ]; then
    mv vision/build/runtime "vision/build/runtime.previous.$(date +%s)"
fi
mkdir -p vision/build/runtime
VIEWER_CONTAINER=$(docker create "osh-vision-runtime:$RELEASE")
trap 'docker rm "$VIEWER_CONTAINER" >/dev/null 2>&1 || true' EXIT INT TERM
docker export "$VIEWER_CONTAINER" | tar -xf - -C vision/build/runtime
docker rm "$VIEWER_CONTAINER" >/dev/null
trap - EXIT INT TERM
BUNDLE="$ROOT/vision/build/runtime"
"$BUNDLE/lib/ld-linux-aarch64.so.1" --library-path "$BUNDLE/lib:$BUNDLE/usr/lib:$BUNDLE/usr/lib/aarch64-linux-gnu:$BUNDLE/lib/aarch64-linux-gnu" "$BUNDLE/app/vision_make_video" vision/assets/people.jpg vision/assets/replay.mp4
for kind in detection segmentation pose; do
    docker build -f vision/Dockerfile --target "$kind" -t "osh-vision-$kind:$RELEASE" .
    docker save "osh-vision-$kind:$RELEASE" | k3s ctr images import -
done
docker build -t npu-task-controller:vision-v1 npu-task
docker save npu-task-controller:vision-v1 | k3s ctr images import -
