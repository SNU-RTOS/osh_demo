# Multi-NPU fractional sharing: clone-to-monitor runbook

This runbook starts with a clean clone and ends with real Hailo-8 inference,
First Fit placement, weighted per-NPU execution, pending admission, monitoring,
an exclusive/shared comparison, and cleanup. `npuShare` is a logical admission
quantity and execution weight. It is not an exact throughput guarantee.

The validated setup is an ARM64, single-node K3s cluster with Hailo driver
4.21.0 and at least two Hailo-8 devices. The scripts build HailoRT 4.21.0 from
the official `v4.21.0` commit so userspace and the kernel driver match.

## 1. Clone the repository

```bash
git clone https://github.com/SNU-RTOS/osh_demo.git
cd osh_demo
git checkout npu-task-reproduction-runbook
```

Run every command below from the repository root.

## 2. Check the host and cluster

Required commands are `git`, `docker`, `k3s`, `kubectl`, and `python3`. Docker
needs network access during the first build. The Hailo device plugin must
advertise `hailo.ai/npu` and include the isolation fix documented in
[`../plugin/README.md`](../plugin/README.md).

```bash
command -v git docker k3s kubectl python3
hailortcli fw-control identify
hailortcli scan
kubectl get nodes -o wide
kubectl get nodes -o custom-columns=NAME:.metadata.name,HAILO:.status.allocatable.hailo\\.ai/npu
kubectl get daemonset -n kube-system hailo-device-plugin
```

The HAILO column must be at least `2`. Stop unrelated NPU workloads before the
validation because the script deploys two dispatchers and also measures a
four-device exclusive baseline.

The repository contains `yolov10s.hef`. Supply a HailoRT 4.21-compatible Tiny
YOLOv4 HEF explicitly:

```bash
export TINY_YOLOV4_HEF=/absolute/path/to/tiny_yolov4.hef
test -r "$TINY_YOLOV4_HEF"
```

The first build creates `npu-share-builder:ubuntu22.04`, clones the official
HailoRT tag into the ignored `npu-task/shared/.cache` directory, verifies commit
`0df636dcb6be9b3943a458591ad5213674a9845d`, and builds the service, client
library, and CLI.

## 3. Run the complete validation

```bash
bash npu-task/shared/reproduce.sh
```

The script builds and imports all images, updates the CRD/controller, starts two
dispatchers, and runs four exclusive baseline workloads. It then submits shares
`600`, `600`, `400`, and `400`, verifies First Fit as `600+400` on each NPU,
checks an overflow request reaches `Pending/InsufficientNPUShare`, validates
HailoRT utilization and WRR grants, releases capacity, and admits the overflow.

Success ends with `Multi-NPU fractional-sharing E2E passed`. The generated
[`../results/multi-npu-share.json`](../results/multi-npu-share.json) contains
per-workload placement, throughput, average latency, P95 latency, and exclusive
versus shared summaries.

## 4. Monitor allocation and actual activity

`npu_monitor.py` keeps requested/allocated share and measured utilization
separate. It combines:

- `NPUTask.spec`: requested NPU count or logical share;
- `NPUTask.status`: selected NPU, allocated share, phase, and pending reason;
- kubelet PodResources: physical IDs assigned to exclusive tasks/dispatchers;
- dispatcher metrics: active registration, queue/accounting, and grant count;
- HailoRT metrics: physical-device utilization, model utilization, and FPS.

Build the optional node-local PodResources reader to display physical IDs:

```bash
mkdir -p npu-task/bin
docker run --rm -v "$PWD/npu-task:/src" -w /src golang:1.24 \
  go build -o /src/bin/podresources ./cmd/podresources
```

Run a snapshot or a live two-second view:

```bash
python3 npu-task/tools/npu_monitor.py
python3 npu-task/tools/npu_monitor.py --watch 2
```

Start the live view in a second terminal before `reproduce.sh` to see the
overflow request become Pending and later run. Namespace and JSON modes are:

```bash
python3 npu-task/tools/npu_monitor.py -n default
python3 npu-task/tools/npu_monitor.py --json > /tmp/npu-state.json
```

The display includes logical capacity, allocated/free share, active workload
count, dispatcher Pod/node/physical ID, every task's request and assignment,
phase, active/done state, cumulative execution grants, and condition reason.
Pending tasks are repeated in a dedicated section with the full admission
message. Measured physical-device and model runtime metrics have their own
sections because HailoRT cannot always attribute a percentage to one Pod
unambiguously.

If the kubelet socket is unreadable, logical placement, requests, grants, and
pending reasons still work; physical IDs are marked unavailable. Run the tool
on each NPU node with suitable socket permissions for physical assignment.

Raw Prometheus-format metrics are also available:

```bash
MONITOR_IP=$(kubectl -n npu-task-system get service hailo-monitor-exporter \
  -o jsonpath='{.spec.clusterIP}')
curl -fsS "http://${MONITOR_IP}:9788/metrics"
kubectl -n npu-task-system exec deployment/npu-share-dispatcher-0 \
  -c share-scheduler -- curl -fsS http://127.0.0.1:9790/metrics
```

Dispatcher metrics include capacity, allocated/free share, queue depth, active
workloads, per-workload allocation, and cumulative grants.

## 5. Diagnose and clean up

```bash
kubectl get nputasks -A
kubectl get pods -n npu-task-system -o wide
kubectl logs -n npu-task-system deployment/npu-task-controller --tail=200
kubectl logs -n npu-task-system daemonset/hailo-monitor-exporter --tail=200
```

A HailoRT `INVALID_DRIVER_VERSION` means host driver and userspace do not match;
this runbook pins userspace to 4.21.0. A Pending capacity shortage is expected
during the overflow test.

```bash
bash npu-task/shared/cleanup.sh
kubectl get nputasks -A
kubectl get pods -n npu-task-system
```

Cleanup removes validation tasks, dispatchers, and the exporter while leaving
the upgraded CRD and controller installed.
