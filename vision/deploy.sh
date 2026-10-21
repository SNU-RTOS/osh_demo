#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
kubectl apply -f "$ROOT/npu-task/deploy/crd.yaml"
kubectl -n npu-task-system get deployment/npu-task-controller >/dev/null
# Preserve the installed controller image and its shared-admission policy.
exec python3 "$ROOT/vision/osh.py" demo --deploy-only "$@"
