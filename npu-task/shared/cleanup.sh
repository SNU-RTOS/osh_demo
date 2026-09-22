#!/usr/bin/env bash
set -euo pipefail

namespace="${NAMESPACE:-default}"
kubectl -n "${namespace}" delete nputask \
    exclusive-a exclusive-b exclusive-c exclusive-d \
    share-a share-b share-c share-d share-overflow \
    --ignore-not-found --wait=true
kubectl -n npu-task-system delete deployment \
    npu-share-dispatcher-0 npu-share-dispatcher-1 --ignore-not-found --wait=true
kubectl -n npu-task-system delete daemonset hailo-monitor-exporter --ignore-not-found --wait=true
kubectl -n npu-task-system delete service hailo-monitor-exporter --ignore-not-found

echo "Shared NPU E2E resources removed. The upgraded CRD and controller remain installed."
