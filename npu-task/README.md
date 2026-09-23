# Managed Hailo NPU tasks

For a complete copy-and-run procedure, see [REPRODUCE.md](REPRODUCE.md). It is
the authoritative runbook for building, deploying, testing, observing, and
cleaning this milestone.

One `NPUTask` describes one inference process in one pod. Users choose the image,
command, compiled model, and an exclusive allocation of 1–8 NPUs. Start, stop,
resume, and resize are ordinary Kubernetes object updates. The controller replaces
pods when execution settings change and never overlaps old and new generations.

For the utilization-aware allocation milestone, see
[`UTILIZATION.md`](UTILIZATION.md). HailoRT can report device/model utilization
and FPS, but these measurements are first used for observability and admission.
Safe fractional sharing requires a dispatcher that owns physical devices and
schedules inference requests.

The multi-NPU fractional-sharing architecture and completed milestones are
documented in [`MULTI_NPU_MILESTONES.md`](MULTI_NPU_MILESTONES.md). The logical
accounting and First Fit core live in `resource/`; the weighted runtime broker
lives in `dispatcher/`. The existing exclusive `npuCount` API remains available.

Shared tasks use `spec.npuShare` instead of `spec.npuCount`. The controller
discovers ready dispatcher Pods, reconstructs their active allocations, applies
First Fit, and injects `HAILORT_SERVICE_ADDRESS`.
[`shared/README.md`](shared/README.md) is the clone-to-monitor runbook for the
two-dispatcher, four-client hardware validation. It includes image construction,
workload execution, live allocation/pending monitoring, evaluation, cleanup,
and a realistic mixed-arrival/dispatcher-recovery scenario. The formal campaign
protocol is in [`MEASUREMENT_GUIDE.md`](MEASUREMENT_GUIDE.md). Prioritized work
after this milestone is tracked in [`NEXT_STEPS.md`](NEXT_STEPS.md).

The exporter DaemonSet in `deploy/monitor-exporter.yaml` reads HailoRT monitor
files and serves metrics on port 9788. Its image contains the pinned HailoRT
4.21 CLI/library. `tools/npu_monitor.py` combines these observations with task,
dispatcher, PodResources, allocation, grant, and Pending-condition state.

## Install

The tested cluster uses K3s v1.33.6+k3s1 and containerd 2.1.5 on ARM64. Install the
[plugin fix](plugin/README.md) before concurrent workloads; the original shared
sysfs backing directory fails multi-pod HailoRT discovery. Keep the existing
`hailo.ai/npu` resource name. No custom scheduler or modified HailoRT is required.

With Go 1.24 or its container image:

```sh
cd npu-task
go test -race ./...
docker build -t npu-task-controller:m1 .
docker save npu-task-controller:m1 | k3s ctr images import -
kubectl apply -f deploy/crd.yaml -f deploy/controller.yaml
kubectl rollout status deployment/npu-task-controller -n npu-task-system
```

For multiple nodes, publish an architecture-compatible image to your registry
and update the deployment image. The scheduler places each task on one node; an
allocation cannot span nodes. The controller does not hardcode the node name or
cluster-wide total. The v1 API limits each task to eight devices.

## Try a real inference task

From the repository root, build the supplied camera-independent probe:

```sh
docker build -f npu-task/tests/Dockerfile.probe -t npu-task-probe:local .
docker save npu-task-probe:local | k3s ctr images import -
kubectl apply -f npu-task/examples/local-probe.yaml
kubectl get nputasks -w
kubectl logs "$(kubectl get nputask local-probe -o jsonpath='{.status.podName}')"
```

The probe performs real inference with zero-filled inputs and reports each PCI
device used. It supports a single HEF network group with any input/output stream
names, but performs no application-specific decoding or accuracy evaluation.
Its optional frame count is per device; `0` means continuous inference.

## Supply your own workload

Use [examples/batch.yaml](examples/batch.yaml) as the image/model contract.
Replace its example image names and provision the named PVC before applying.

| Field | Contract |
|---|---|
| `image`, `command`, `args` | Compatible HailoRT image and foreground inference binary; shell wrappers must `exec` the binary or forward signals. |
| `model.path` | Absolute compiled HEF path inside the image or model mount. |
| `model.pvc`, `model.mountPath` | Optional same-namespace PVC mounted read-only; the model must be beneath its mount path. |
| `npuCount` | Integer 1–8; mapped to matching device requests and limits. |
| `resources` | Optional CPU/memory requests and limits. |
| `env` | Additional environment variables; `NPU_COUNT` and `MODEL_PATH` are reserved. |
| `mode` | `Batch` (default) retains success/failure and does not retry. `Service` restarts failed containers, and replaces terminal pods. |
| `suspend` | Terminates the owned pod and releases the allocation. |

Kubernetes expands `$(MODEL_PATH)` and `$(NPU_COUNT)` in command/arguments.
Images must include matching preprocessing/postprocessing and compatible HailoRT
libraries. Arbitrary HEFs are not interchangeable in a model-specific driver.
No model compilation, upload service, image builder, or model registry is included.

Tasks should discover only their assigned devices and explicitly select them.
The reference binaries reject count mismatches instead of silently taking other
devices. Handle SIGTERM, bound in-flight I/O, destroy HailoRT resources, and exit
within 30 seconds. A container's Running state is process availability, not proof
of application readiness or correct inference results.

## Manage allocations

```sh
# Release the allocation; wait before requesting a resume.
kubectl patch nputask local-probe --type=merge -p '{"spec":{"suspend":true}}'
kubectl wait nputask/local-probe --for=jsonpath='{.status.phase}'=Suspended --timeout=60s

# Resume with four devices and continuous inference.
kubectl patch nputask local-probe --type=merge -p '{"spec":{"suspend":false,"npuCount":4,"args":["$(MODEL_PATH)","0"]}}'

# Resize; the old pod stops before its replacement is created.
kubectl patch nputask local-probe --type=merge -p '{"spec":{"npuCount":2}}'
```

Insufficient capacity leaves a task Pending with the scheduler's reason in
`status.conditions`. A resize can therefore introduce downtime and then wait
indefinitely for capacity. The controller does not preempt tasks or promise FIFO
ordering. Suspend/resume a finished batch to run it again, or change execution
settings. Deleting its completed pod alone does not rerun it after the controller
has recorded its terminal result. Deleting the task garbage-collects its pod.

Do not force-delete pods to accelerate release: on an unreachable node, removal
of an API object does not prove the process stopped. Normal kubelet termination
and device management remain responsible for actual release. Stuck termination
is surfaced as Stopping; the controller does not reset hardware or edit kubelet
checkpoints. External deletion of an unrecorded batch pod may cause recreation;
this framework does not provide exactly-once execution.

## Observe and test

```sh
kubectl get nputasks -A
kubectl describe nputask local-probe
go build -o bin/podresources ./cmd/podresources
# On each NPU node with access to its kubelet Unix socket:
sudo bin/podresources
python3 tests/e2e.py --cycles 100 --output results/acceptance.json
```

`podresources` reports kubelet-assigned device IDs per container. Node
`allocatable` is total usable inventory, not the number currently free. Controller
runtime metrics are available on container port 8080, health checks on 8081.
The controller exports `nputask_pending_duration_seconds`,
`nputask_allocation_age_seconds`, `nputask_container_restart_count`, requested
and allocated `nputask_share`, and `nputask_phase_info`. Shared task status keeps
the allocation timestamp and current dispatcher broker epoch.

```sh
CONTROLLER_POD=$(kubectl -n npu-task-system get pod -l app=npu-task-controller -o jsonpath='{.items[0].metadata.name}')
kubectl get --raw "/api/v1/namespaces/npu-task-system/pods/${CONTROLLER_POD}:8080/proxy/metrics"
```

The end-to-end test creates and deletes its own namespace. It verifies 3+5
concurrent inference, a waiting 2-device task, release while another task keeps
inferring, resize 4→2→4, bad HEF cleanup, 100 cycles over 1/2/4/8 devices, and denial
without an allocation. Use `--second-image` and `--second-model` for another HEF.
Use `--keep` only when retaining failed pods for diagnosis is intentional.
Zero-input inference checks device lifecycle, not model accuracy or camera I/O.
For the focused multi-model milestone, run `tests/multi_model.py` with the
default YOLOv10s probe and a second image containing Tiny YOLOv4. It verifies
different model processes, different NPU counts, disjoint physical IDs, Pending
behavior, release, and continued inference. The complete command sequence is
in [REPRODUCE.md](REPRODUCE.md).
`python3 tests/recovery.py --restart-components` additionally tests killed inference,
service container restart, cascading task deletion, and controller/plugin rollouts
while another task continues inference. It restarts those two components; run it
after the main suite.

For a maintenance-window K3s restart test, first retain a running service task and
record its device IDs. Restart K3s, wait for node and plugin readiness, verify
the task resumes successful inference, then suspend it and prove another task
can use its devices. This disrupts other workloads and is separate from the
default acceptance script. Do not reboot to work around test failures.

## Camera reference

The existing `inference_driver.yaml` now runs inference directly as PID 1 and
sets `NPU_COUNT=8`. Rebuild both reference applications and use the same
`OSH_TASK_ID=demo` for the host camera process; this namespaces shared memory and
socket names. Other camera/inference pairs need distinct task IDs. The camera
demo still needs its existing host IPC and socket mounts, and is not the generic
NPUTask sample: user images should provide their own input/output integration.

Both camera and inference binaries must be rebuilt together: the ring header now
contains all 24 sequence slots (protocol version 2). Each producer creates its ring
before waiting up to five seconds for its peer, so launch the pair together.

The reference inference driver distributes the eight inputs over `NPU_COUNT`
workers and selects one explicit device for each worker. Its YOLO decoder remains
model-specific. Use the standalone probe when no camera input is available.

GPU composition is available as a separate milestone. `camera_overlay` accepts
`OSH_DISPLAY_BACKEND=opengl` to use GStreamer's `glvideomixer`; the default CPU
compositor remains available for boards without a usable EGL/DRM/Wayland/X11
graphics context. Run `gpu_compositor_probe` before connecting it to camera input.

## Remove

Suspend or delete all NPUTasks and wait for their pods to terminate before
removing the controller or CRD. Removing the controller alone leaves task pods
running. Keep the plugin installed for any other Hailo workloads. The separate
dispatcher project under `/data/hailort-k8s-npu-sharing` is not modified.
