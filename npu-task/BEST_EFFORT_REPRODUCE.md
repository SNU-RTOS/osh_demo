# Reproduce best-effort sharing and add containers

This guide builds the current implementation, measures a real model, enables
`utilization-best-effort`, and deploys several containers sharing one Hailo-8.
No cameras are required: the reference runner performs real inference using
zero-filled model inputs. It reports inference progress and performance, not
decoded detections or model accuracy.

Use Bash, run commands from the repository root, and keep the same shell for
the helper function below. You need an ARM64 K3s test node, at least two Hailo-8
devices for the supplied dispatcher manifests, matching driver/HailoRT 4.21.0,
Docker, `kubectl`, `k3s`, and Python 3. Install the
[device-plugin isolation fix](plugin/README.md) first. Registry distribution
and multi-node validation are outside this single-node procedure.

This is a **test-cluster installation procedure**: it installs the bundled static
configuration before switching policies. If shared workloads already exist,
save their specs/configuration, suspend them, and wait for their assigned pods
to disappear using the [drain procedure](ADMISSION.md#install-switch-and-roll-back).
Leave them suspended throughout this demonstration. Do not run the static
`shared/reproduce.sh` campaign while this best-effort demo is active.

## 1. Build and install the current images

The shared-image build currently packages both supplied model types, so provide
a compatible Tiny YOLOv4 HEF even though this demonstration initially uses YOLOv10s.

```bash
export TINY_YOLOV4_HEF=/absolute/path/to/tiny_yolov4.hef
test -r "$TINY_YOLOV4_HEF"
bash npu-task/shared/build-images.sh
docker build -t npu-task-controller:best-effort npu-task
```

Build a profiling image from that exact client image. Python is needed only by
the calibration tool; regular shared clients do not need it.

```bash
docker build -t npu-share-profiler:local -f - npu-task/shared <<'DOCKERFILE'
FROM npu-share-client:local
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY profile_model.py /opt/profile_model.py
DOCKERFILE

docker save npu-task-controller:best-effort npu-share-client:local \
  npu-share-dispatcher:local npu-share-profiler:local | k3s ctr -n k8s.io images import -
kubectl apply -f npu-task/deploy/crd.yaml -f npu-task/deploy/controller.yaml
kubectl -n npu-task-system set image deployment/npu-task-controller \
  controller=npu-task-controller:best-effort
kubectl -n npu-task-system rollout restart deployment/npu-task-controller
kubectl -n npu-task-system rollout status deployment/npu-task-controller --timeout=180s
kubectl create namespace npu-best-effort-demo --dry-run=client -o yaml | kubectl apply -f -
```

At this point the policy is still `static-share`; exclusive profiling works in
either mode. Rebuilt local tags must be imported before restarting their pods.

## 2. Measure the model on one exclusive NPU

Keep the profiling device idle apart from this task. At least one whole NPU
must be available. If existing dispatchers occupy all devices, after draining
their clients scale those dispatchers to zero and wait for their pods to disappear
before profiling; step 3 restores the demo dispatcher.

```bash
kubectl -n npu-best-effort-demo apply -f - <<'YAML'
apiVersion: npu.snu-rtos.io/v1alpha1
kind: NPUTask
metadata:
  name: profile-yolov10s
spec:
  image: npu-share-profiler:local
  command: [/bin/sh, -ec]
  args:
    - |
      python3 /opt/profile_model.py --runner /usr/local/bin/shared_runner \
        --hef "$MODEL_PATH" --key yolov10s --architecture HAILO8 \
        --runtime-version 4.21.0 --output /tmp/admission.json
      printf 'NPU_ADMISSION_CONFIG_BEGIN\n'
      cat /tmp/admission.json
  model:
    path: /opt/models/yolov10s.hef
  npuCount: 1
  mode: Batch
YAML

kubectl -n npu-best-effort-demo wait nputask/profile-yolov10s \
  --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
python3 - <<'PY'
import json
import pathlib
import subprocess

ns = 'npu-best-effort-demo'
task = json.loads(subprocess.check_output(
    ['kubectl', '-n', ns, 'get', 'nputask', 'profile-yolov10s', '-o', 'json']))
log = subprocess.check_output(
    ['kubectl', '-n', ns, 'logs', task['status']['podName']], text=True)
config = json.loads(log.split('NPU_ADMISSION_CONFIG_BEGIN\n', 1)[1])
path = pathlib.Path('/tmp/npu-best-effort.json')
path.write_text(json.dumps(config, indent=2) + '\n')
print(path.read_text())
PY
```

Expected output is a best-effort configuration with a nonempty `profiles.yolov10s`,
a 64-character `modelSHA256`, 500 measured samples, and positive mean/P95 costs.
The tool first performs 50 warm-up executions. Verify `HAILO8` and `4.21.0`
against your actual hardware/runtime; these arguments are declared compatibility
metadata. Save the profile and environment details for repeatable experiments.

If the task fails or stays Pending, inspect it before proceeding:

```bash
kubectl -n npu-best-effort-demo describe nputask profile-yolov10s
kubectl -n npu-best-effort-demo get pods -o wide
```

A completed Batch is not rerun by applying the same manifest. To repeat
calibration, suspend the task, wait for `Suspended`, then resume it, or use a new
task name. Repeat the extraction above after the new run completes.

## 3. Start one dispatcher and enable best-effort admission

The supplied file defines two dispatchers. Keep only dispatcher 0 active for
this demonstration so that sharing is visible on a single physical device.
On an upgrade, restarting dispatcher 0 loads the rebuilt broker image; its
clients must already have been drained.

```bash
kubectl apply -f npu-task/shared/dispatchers.yaml
kubectl -n npu-task-system scale deployment/npu-share-dispatcher-1 --replicas=0
kubectl -n npu-task-system rollout status deployment/npu-share-dispatcher-1 --timeout=180s
kubectl -n npu-task-system rollout restart deployment/npu-share-dispatcher-0
kubectl -n npu-task-system rollout status deployment/npu-share-dispatcher-0 --timeout=180s

kubectl -n npu-task-system create configmap npu-admission-config \
  --from-file=admission.json=/tmp/npu-best-effort.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n npu-task-system rollout restart deployment/npu-task-controller
kubectl -n npu-task-system rollout status deployment/npu-task-controller --timeout=180s
```

The controller reads configuration only at startup. The broker needs a complete
ten-second observation window before new best-effort tasks can be admitted.
Hardware monitor export is optional for this policy: admission uses broker
occupancy, not Hailo's hardware-utilization percentage.

## 4. Deploy the first container

Define this helper once. It inserts the measured model checksum automatically;
there is no physical NPU ID in the request. Its arguments are
`NAME FPS [PROFILE_KEY] [MODEL_PATH] [IMAGE]`.

```bash
deploy_best_effort() {
  python3 - "$@" <<'PY' | kubectl -n npu-best-effort-demo apply -f -
import json
import sys

name, fps = sys.argv[1], float(sys.argv[2])
key = sys.argv[3] if len(sys.argv) > 3 else 'yolov10s'
path = sys.argv[4] if len(sys.argv) > 4 else '/opt/models/yolov10s.hef'
image = sys.argv[5] if len(sys.argv) > 5 else 'npu-share-client:local'
with open('/tmp/npu-best-effort.json') as f:
    profile = json.load(f)['profiles'][key]
print(json.dumps({
    'apiVersion': 'npu.snu-rtos.io/v1alpha1', 'kind': 'NPUTask',
    'metadata': {'name': name},
    'spec': {
        'image': image, 'command': ['/usr/local/bin/shared_runner'],
        'args': ['--hef', '$(MODEL_PATH)', '--runs', '1000000'],
        'model': {'path': path, 'key': key, 'sha256': profile['modelSHA256']},
        'bestEffort': {'targetFPS': fps, 'weight': 100}, 'mode': 'Service'
    }
}))
PY
}

# Pick a rate that leaves profiled demand headroom for four demo clients.
DEMO_FPS=$(python3 - <<'PY'
import json
with open('/tmp/npu-best-effort.json') as f:
    c = json.load(f)
cost = c['profiles']['yolov10s']['p95Seconds']
print(min(20.0, c['budget'] / (4 * 1000 * cost * c['profileMargin'])))
PY
)
deploy_best_effort infer-a "$DEMO_FPS"
kubectl -n npu-best-effort-demo wait nputask/infer-a \
  --for=jsonpath='{.status.phase}'=Running --timeout=180s
```

The rate calculation is an experiment setup, not a throughput guarantee. Live
broker occupancy may temporarily delay admission even when profiled demand fits.
`Running` confirms that the process started; use grant progress and completion
logs to verify actual inference. This runner prints its `NPU_E2E_OK` summary after
the configured number of executions, not once per frame. Service mode restarts
the process if that finite run completes.

## 5. Add containers without restarting services

Submit a new task name for each additional container:

```bash
deploy_best_effort infer-b "$DEMO_FPS"
deploy_best_effort infer-c "$DEMO_FPS"
kubectl -n npu-best-effort-demo wait nputask/infer-b nputask/infer-c \
  --for=jsonpath='{.status.phase}'=Running --timeout=180s
kubectl -n npu-best-effort-demo get nputasks \
  -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,NPU:.status.npuID,FPS:.spec.bestEffort.targetFPS,DEMAND:.status.reservedDemand,WEIGHT:.status.executionWeight
python3 npu-task/tools/npu_monitor.py -n npu-best-effort-demo --watch 2
```

Expected: `infer-a`, `infer-b`, and `infer-c` use `npu-0`, have separate pod names,
positive reserved demand, and weight 100 each. The monitor should show increasing
grant counts for each workload. Press Ctrl+C to exit the monitor.

Adding a container using an **already configured profile** needs no controller
restart or dispatcher restart. Reusing a task name updates that task rather
than creating another container; changing execution settings replaces its pod.
If demand or live occupancy exceeds the budget, the new task stays Pending
with `InsufficientBestEffortCapacity`; existing tasks continue running.

Release one reservation by suspension or deletion:

```bash
kubectl -n npu-best-effort-demo patch nputask infer-b --type=merge \
  -p '{"spec":{"suspend":true}}'
kubectl -n npu-best-effort-demo wait nputask/infer-b \
  --for=jsonpath='{.status.phase}'=Suspended --timeout=180s
```

A waiting request is retried automatically after capacity and telemetry permit.
Resume `infer-b` later by patching `suspend` to `false`.

To use another model already in the startup configuration:

```bash
deploy_best_effort classify-a 10 tiny-yolov4 /opt/models/tiny_yolov4.hef
```

That command requires a measured `profiles.tiny-yolov4` entry, matching runtime
metadata, and a matching HEF in the selected image. For a new model, first drain
shared workloads, profile it using step 2 with a new task/model/key, merge the
new `profiles` entry into the startup JSON, and repeat the configuration restart
in step 3. Preserve existing profile entries. Adding or changing profiles
changes the configuration fingerprint and requires a drain; adding another
container using an existing profile does not.

For a custom image, keep the same runner path or adapt `command`/`args`, and
implement the [client contract](ADMISSION.md#profile-a-model). An arbitrary
Hailo application that opens devices directly cannot become a shared client
just by setting `bestEffort`.

## 6. Troubleshoot, clean up, or return to static sharing

```bash
kubectl -n npu-best-effort-demo describe nputask infer-a
kubectl -n npu-task-system logs deployment/npu-task-controller
```

| Pending reason | Check |
|---|---|
| `AdmissionPolicyMismatch` | Restart with the matching policy or change the task request format. |
| `MissingModelProfile` / `IncompatibleModelProfile` | Configured key and HEF checksum. |
| `IncompatibleDispatcher` | Dispatcher readiness and architecture/runtime annotations. |
| `StaleBrokerObservation` | New broker warm-up, broker image version, and controller access to port 9790. |
| `InsufficientBestEffortCapacity` / `DemandExceedsBudget` | Lower requested FPS, release tasks, or add dispatcher capacity. |
| `AdmissionConfigurationMismatch` | Drain all old shared assignments before using the new configuration. |

Remove only this guide's tasks, then wait for their owned pods to disappear:

```bash
kubectl -n npu-best-effort-demo delete nputask infer-a infer-b infer-c profile-yolov10s --ignore-not-found
kubectl -n npu-best-effort-demo wait pod --all --for=delete --timeout=180s
```

Delete additional tasks you created, such as `classify-a`, as well. Keep the
namespace and shared dispatchers if other experiments need them. To return to
`static-share`, drain **all** shared workloads, restore the saved configuration
or set `sharedAdmissionPolicy` to `static-share`, restart the controller, and
deploy tasks using `npuShare` instead of `bestEffort`. See the complete
[switch/rollback procedure](ADMISSION.md#install-switch-and-roll-back).
