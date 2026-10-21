#!/bin/sh
# Reuse a prepared SDK: no apt, Docker RUN, or controller replacement.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SDK=
RELEASE=${OSH_VISION_RELEASE:-v3}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --sdk) SDK=${2:?--sdk requires a path}; shift 2 ;;
    *) echo 'usage: sudo sh vision/rebuild-from-sdk.sh --sdk /tmp/osh-vision-sdk' >&2; exit 2 ;;
  esac
done
[ "$(id -u)" = 0 ] || { echo 'Run this SDK/chroot build with sudo.' >&2; exit 1; }
[ -n "$SDK" ] || { echo '--sdk is required' >&2; exit 2; }
SDK=$(realpath "$SDK")
case "$SDK" in /tmp/osh-vision-*|/data/osh-vision-*) ;; *) echo 'SDK must be an explicit prepared /tmp/osh-vision-* or /data/osh-vision-* directory' >&2; exit 1;; esac
test -x "$SDK/usr/bin/cmake"
test -f "$SDK/usr/include/hailo/hailort.hpp"
test -d "$SDK/src"
if mountpoint -q "$SDK/src"; then
  echo 'SDK/src is already mounted; finish that build before reusing it.' >&2
  exit 1
fi
BOUND=false
cleanup() {
  if [ "$BOUND" = true ]; then umount "$SDK/src"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mount --bind "$ROOT" "$SDK/src"
BOUND=true
chroot "$SDK" /bin/sh -ec 'cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DOSH_BUILD_CAMERA=OFF -DOSH_BUILD_VISION=ON && cmake --build /build -j3 && ctest --test-dir /build --output-on-failure'
umount "$SDK/src"
BOUND=false
cd "$ROOT"
if [ -e vision/build/runtime ]; then
  BACKUP=$(mktemp -d "$ROOT/vision/build/runtime.previous.XXXXXX")
  mv vision/build/runtime "$BACKUP/runtime"
  printf 'Previous native bundle retained: %s/runtime\n' "$BACKUP"
fi
python3 vision/package_sdk.py --sdk "$SDK" --output vision/build/runtime
mkdir -p vision/build/models
cp yolov10s.hef vision/build/models/yolov10s.hef
for KIND in detection segmentation pose; do
  docker build -f vision/Dockerfile.bundle --target "$KIND" -t "osh-vision-$KIND:$RELEASE" vision
  docker save "osh-vision-$KIND:$RELEASE" | k3s ctr -n k8s.io images import -
done
printf 'Built and imported %s images; native viewer refreshed. No tasks deployed.\n' "$RELEASE"
printf 'Next: sudo python3 vision/osh.py doctor\n'
