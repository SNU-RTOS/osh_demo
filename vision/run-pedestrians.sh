#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python3 "$ROOT/vision/prepare_video.py" --offline
OUTPUT=${OSH_VISION_OUTPUT:-$ROOT/vision/results/pedestrians-$(date -u +%Y%m%dT%H%M%SZ)}
EXPECT_OUTPUT=false
for ARG do
  if [ "$EXPECT_OUTPUT" = true ]; then
    OUTPUT=$ARG
    EXPECT_OUTPUT=false
  elif [ "$ARG" = --output ]; then
    EXPECT_OUTPUT=true
  fi
done
printf 'Demo results: %s\n' "$OUTPUT"
exec sh "$ROOT/vision/run-viewer.sh" \
  --source "$ROOT/vision/assets/pedestrians.avi" \
  --socket /data/osh-vision/demo/hub.sock \
  --fps 10 --output "$OUTPUT" --screenshot "$OUTPUT/display.png" "$@"
