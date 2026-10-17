#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUNDLE=${OSH_VISION_BUNDLE:-$ROOT/vision/build/runtime}
export DISPLAY=${DISPLAY:-:0}
export __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-nvidia}
exec "$BUNDLE/lib/ld-linux-aarch64.so.1" --library-path "$BUNDLE/lib:$BUNDLE/usr/lib:$BUNDLE/usr/lib/aarch64-linux-gnu:$BUNDLE/lib/aarch64-linux-gnu:/usr/lib:/usr/lib/aarch64-linux-gnu" "$BUNDLE/app/vision_hub" "$@"
