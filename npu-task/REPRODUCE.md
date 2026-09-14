# NPU framework reproduction runbook

This is the single runbook for rebuilding, installing, exercising, and cleaning
the managed Hailo NPU framework. It is written for the tested single-node K3s
setup: K3s v1.33.6+k3s1, containerd 2.1.5, eight Hailo-8 devices, and an ARM64
node. Replace local image tags with registry references when the node cannot
load images through `k3s ctr`.

The commands below are intended to be run from the repository root:

```sh
cd /data/osh_demo
```

## Camera-free testing

You do not need a camera, display, or input image for the framework tests. The
`npu_probe` image runs a compiled HEF with zero-filled tensors and reports real
Hailo device assignment and inference. This exercises the Kubernetes allocation
path, CDI isolation, concurrent model execution, release, resize, crash recovery,
and reuse. The camera overlay and `inference_driver` are only needed later for
camera/IPC integration.

The repository already has suitable compiled models. The default probe uses
`yolov10s.hef`; the runbook optionally builds a second probe image with
`tiny_yolov4.hef`. These files stay outside generated test output. Do not commit
downloaded model or image binaries: place any future assets under
`npu-task/assets/`, which is ignored by Git.

## 1. Check prerequisites

Do this before changing the cluster. The test namespace is isolated, but the
plugin and controller rollout commands affect cluster components.

```sh
set -eu
command -v docker kubectl k3s python3
k3s --version
kubectl version --short 2>/dev/null || kubectl version --client
kubectl get nodes -o wide
kubectl get node ava -o jsonpath='{.status.allocatable.hailo\.ai/npu}{" NPUs allocatable\n"}'
kubectl get daemonset hailo-device-plugin -n kube-system
kubectl get pods -n npu-task-system 2>/dev/null || true
```

The last resource count must be `8 NPUs allocatable`. Do not continue while
other workloads are using the devices unless you intend to test contention.

## 2. Run source-level tests

The controller uses Go 1.24. A cached Go image avoids requiring Go on the host.
The C++ build uses the repository's Hailo runtime/compiler image.

```sh
docker run --rm \
  -v "$PWD/npu-task:/src" -w /src \
  -e GOCACHE=/src/.cache/build -e GOMODCACHE=/src/.cache/mod \
  golang:1.24 sh -c \
  'gofmt -w api controller cmd && go test -race ./... && go vet ./...'

docker run --rm \
  -v "$PWD:/src" -w /src \
  ghcr.io/snu-rtos/osh-compile:latest sh -c \
  'cmake -S . -B /tmp/osh-build && cmake --build /tmp/osh-build -j4 && ctest --test-dir /tmp/osh-build --output-on-failure'
```

The expected result is a passing controller test suite and the `shm_ring` CTest.
The C++ build produces `camera_overlay`, `inference_driver`, and `npu_probe`.

## 3. Build and load the controller

```sh
docker build -t npu-task-controller:m1 npu-task
docker save npu-task-controller:m1 | k3s ctr images import -
```

Install or update the API and controller:

```sh
kubectl apply -f npu-task/deploy/crd.yaml
kubectl apply -f npu-task/deploy/controller.yaml
kubectl set image deployment/npu-task-controller -n npu-task-system \
  controller=npu-task-controller:m1
kubectl rollout status deployment/npu-task-controller \
  -n npu-task-system --timeout=120s
```

Confirm that the controller is ready:

```sh
kubectl get deployment npu-task-controller -n npu-task-system
kubectl get crd nputasks.npu.snu-rtos.io
```

## 4. Build and load the real-inference probe

The probe uses `yolov10s.hef`, performs real Hailo inference on zero-filled
inputs, and prints the physical devices it opened.

```sh
docker build -f npu-task/tests/Dockerfile.probe \
  -t npu-task-probe:local .
docker save npu-task-probe:local | k3s ctr images import -
```

To test a second model, build another image with a second HEF. For example:

```sh
mkdir -p /tmp/osh-second-probe
cp /home/root/workspace/models/hef/tiny_yolov4.hef /tmp/osh-second-probe/
cat >/tmp/osh-second-probe/Dockerfile <<'EOF'
FROM npu-task-probe:local
COPY tiny_yolov4.hef /models/tiny_yolov4.hef
EOF
docker build -t npu-task-probe-second:local /tmp/osh-second-probe
docker save npu-task-probe-second:local | k3s ctr images import -
```

## 5. Install the device-plugin isolation fix

The plugin patch is maintained separately from this repository. It fixes the
reproduced failure where two containers shared `/sys/class/hailo_chardev` and
one container's cleanup removed entries from another container.

Apply the patch in a clean checkout of the upstream plugin at the pinned commit:

```sh
git clone https://github.com/SNU-RTOS/hailo-device-plugin.git /tmp/hailo-device-plugin
cd /tmp/hailo-device-plugin
git checkout 7770bf450b3c06db9016b210b1aa0fb904d87762
git apply /data/osh_demo/npu-task/plugin/milestone-1.patch
go test -race ./...
go build -o hailo-device-plugin .
mkdir -p /tmp/hailo-device-plugin-image
cp hailo-device-plugin /tmp/hailo-device-plugin-image/
cp Dockerfile.local /tmp/hailo-device-plugin-image/ 2>/dev/null || \
  printf '%s\n' 'FROM ubuntu:22.04' 'COPY hailo-device-plugin /hailo-device-plugin' \
    'ENTRYPOINT ["/hailo-device-plugin"]' >/tmp/hailo-device-plugin-image/Dockerfile
docker build -t hailo-device-plugin:nputask-m1 /tmp/hailo-device-plugin-image
docker save hailo-device-plugin:nputask-m1 | k3s ctr images import -
cd /data/osh_demo
```

Before replacing the DaemonSet, make sure no test task is running. Then roll out
the local image:

```sh
kubectl patch daemonset hailo-device-plugin -n kube-system --type=strategic \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"hailo-device-plugin","image":"hailo-device-plugin:nputask-m1","imagePullPolicy":"IfNotPresent"}]}}}}'
kubectl rollout status daemonset/hailo-device-plugin -n kube-system --timeout=120s
kubectl get node ava -o jsonpath='{.status.allocatable.hailo\.ai/npu}{" NPUs allocatable\n"}'
```

The expected count is eight. If the rollout fails, inspect the DaemonSet logs
before changing anything else:

```sh
kubectl logs -n kube-system daemonset/hailo-device-plugin --tail=100
kubectl describe daemonset hailo-device-plugin -n kube-system
```

## 6. Run the one-task smoke test

```sh
kubectl apply -f npu-task/examples/local-probe.yaml
kubectl get nputask local-probe -w
kubectl logs "$(kubectl get nputask local-probe -o jsonpath='{.status.podName}')"
kubectl delete nputask local-probe --wait=true
```

The task should reach `Succeeded`, logs should contain `ASSIGNED_DEVICE` and
`INFERENCE_OK`, and the pod should be gone after deletion.

## 7. Run the full lifecycle acceptance test

This creates a random namespace and deletes it in `finally`. It does not modify
existing tasks. The test verifies concurrent 3+5 allocations, Pending behavior,
release, resize 4→2→4, invalid HEF cleanup, denial without an allocation, and
100 cycles over 1/2/4/8 devices.

```sh
python3 npu-task/tests/e2e.py \
  --cycles 100 \
  --second-image npu-task-probe-second:local \
  --second-model /models/tiny_yolov4.hef \
  --output npu-task/results/acceptance.json
```

Expected final output includes `no-allocation-denied`. Check the machine-readable
result:

```sh
python3 - <<'PY'
import json
r = json.load(open('npu-task/results/acceptance.json'))
assert r['passed'] is True, r
print('passed:', r['passed'], 'checks:', len(r['checks']))
PY
```

The observed baseline is `passed: True checks: 105`. Runtime varies with model
loading and device contention. A failure preserves diagnostic logs and events in
the JSON result; use `--keep` to retain the test namespace for interactive
inspection.

## 8. Run recovery checks

This checks killed inference, device reuse, Service restart behavior, task
deletion cascading, and optional controller/plugin rollouts. It creates and
deletes its own namespace.

```sh
python3 npu-task/tests/recovery.py \
  --restart-components \
  --output npu-task/results/recovery.json
```

Do not use `--restart-components` during an unrelated production workload: it
rolls the controller and Hailo plugin. The script intentionally does not restart
K3s. A K3s restart is a separate maintenance-window test.

## 9. Inspect actual kubelet assignments

The node's `allocatable` value is total inventory, not currently free capacity.
Build and run the PodResources diagnostic on the NPU node:

```sh
docker run --rm \
  -v "$PWD/npu-task:/src" -v "$PWD/npu-task/bin:/out" -w /src golang:1.24 \
  sh -c 'go build -o /out/podresources ./cmd/podresources'
sudo npu-task/bin/podresources
cd /data/osh_demo
```

The output lists each pod/container and the exact `hailo.ai/npu` device IDs
assigned by kubelet. If Go is already installed, `cd npu-task && go build -o
bin/podresources ./cmd/podresources` is equivalent.

## 10. Optional camera demo

Rebuild both binaries because the shared-memory protocol is version 2 and now
supports all 24 camera slots:

```sh
docker run --rm -it -v "$PWD":/workspace -w /workspace \
  ghcr.io/snu-rtos/osh-compile:latest sh -c \
  'cmake -S . -B build && cmake --build build -j8'
kubectl apply -f inference_driver.yaml
kubectl logs -f inference-driver
```

Run the host camera process with the same task ID:

```sh
OSH_TASK_ID=demo ./build/camera/camera_overlay
```

The camera demo is model-specific and is separate from the generic `NPUTask`
contract. It requires the existing IPC, `/tmp`, host camera, and display setup.

## 11. Cleanup and rollback

Remove test resources before uninstalling the controller:

```sh
kubectl get nputasks -A
# Delete only the test tasks you created, or name a specific namespace.
kubectl delete nputask local-probe --wait=true 2>/dev/null || true
kubectl delete -f npu-task/deploy/controller.yaml
kubectl delete -f npu-task/deploy/crd.yaml
```

The last command removes the API type; it does not remove the Hailo plugin.
Keep the plugin installed for other Hailo workloads. To roll back the plugin,
restore the prior image and wait for its DaemonSet rollout. The previously
deployed upstream digest was:

```text
ghcr.io/snu-rtos/hailo-device-plugin@sha256:3befb24fb3432a4ec966f3271d7e3078e71ff0afbf9f72df3848dc65f2d61757
```

Use normal pod termination; do not force-delete an NPU pod to claim that the
hardware has been released. A stuck pod needs node/kubelet investigation.

## What to do next

After this runbook passes repeatedly, the next milestone is a model catalog and
profiling layer. Record model image, HEF, NPU count, throughput, latency, memory,
startup time, and device assignment for each workload. Use those measurements to
define admission and priority policy before attempting automatic eviction or
live reassignment. The current framework deliberately uses explicit
user-controlled suspend and pod replacement.
