#!/bin/sh
# Fallback when Docker RUN cannot start on a full root disk. Requires root,
# Docker, chroot and mount. All installed packages stay inside the temporary SDK.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SDK=${OSH_VISION_SDK_ROOT:-/tmp/osh-vision-sdk-build}
case "$SDK" in /tmp/osh-vision-*) ;; *) echo 'SDK must be under /tmp/osh-vision-*' >&2; exit 1;; esac
[ ! -e "$SDK" ] || { echo "Refusing to overwrite $SDK" >&2; exit 1; }
cd "$ROOT"
python3 vision/prepare_assets.py
mkdir -p "$SDK"
CONTAINER=$(docker create ghcr.io/snu-rtos/osh-compile@sha256:00fc30abffc759106c0bfd0eb8a2a0d2ea8eb3e2a958934e59189dade47be991)
cleanup() {
    umount "$SDK/src" 2>/dev/null || true
    umount "$SDK/dev" 2>/dev/null || true
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
docker export "$CONTAINER" | tar -xf - -C "$SDK"
cp /etc/resolv.conf "$SDK/etc/resolv.conf"
mkdir -p "$SDK/src" "$SDK/build"
mount --bind /dev "$SDK/dev"
mount --bind "$ROOT" "$SDK/src"
chroot "$SDK" /bin/sh -c 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libopencv-dev=4.5.4+dfsg-9ubuntu4 libsdl2-dev=2.0.20+dfsg-2ubuntu1.22.04.1 libglvnd-dev=1.4.0-1 nlohmann-json3-dev=3.10.5-2 python3=3.10.6-1~22.04.1 && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DOSH_BUILD_CAMERA=OFF -DOSH_BUILD_VISION=ON && cmake --build /build -j4 && ctest --test-dir /build --output-on-failure && /build/vision/vision_make_video /src/vision/assets/people.jpg /src/vision/assets/replay.mp4'
python3 vision/package_sdk.py --sdk "$SDK" --output vision/build/runtime
mkdir -p vision/build/models
cp yolov10s.hef vision/build/models/yolov10s.hef
for kind in detection segmentation pose; do
    docker build -f vision/Dockerfile.bundle --target "$kind" -t "osh-vision-$kind:v2" vision
    docker save "osh-vision-$kind:v2" | k3s ctr images import -
done
printf 'Viewer bundle ready: %s/vision/build/runtime\nSDK retained at %s\n' "$ROOT" "$SDK"
