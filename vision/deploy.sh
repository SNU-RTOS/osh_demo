#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
kubectl apply -f "$ROOT/npu-task/deploy/crd.yaml"
kubectl -n npu-task-system set image deployment/npu-task-controller controller=npu-task-controller:vision-v1
kubectl -n npu-task-system rollout status deployment/npu-task-controller --timeout=90s
kubectl apply -f "$ROOT/vision/tasks.yaml"
kubectl -n osh-vision get nputasks
