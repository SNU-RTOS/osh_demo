#!/bin/sh
set -eu
# Only the three demo tasks; retain measurements, assets and existing dispatchers.
for TASK in detection segmentation pose; do
  POD=$(kubectl -n osh-vision get nputask "$TASK" --ignore-not-found -o jsonpath='{.status.podName}')
  kubectl -n osh-vision delete nputask "$TASK" --ignore-not-found
  if [ -n "$POD" ] && [ -n "$(kubectl -n osh-vision get pod "$POD" --ignore-not-found -o name)" ]; then
    kubectl -n osh-vision wait --for=delete "pod/$POD" --timeout=60s
  fi
done
