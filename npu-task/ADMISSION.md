# Restart-configured NPU admission

For the full build → profile → enable → deploy/add containers workflow, follow
[BEST_EFFORT_REPRODUCE.md](BEST_EFFORT_REPRODUCE.md). This page describes the
policy contract and switching behavior.

The exclusive `npuCount: 1..8` path is always available. Shared tasks use one
dispatcher/NPU per task and choose a request format matching the controller's
cluster-wide startup policy:

| Policy | Task request | Admission | Runtime |
|---|---|---|---|
| `static-share` (default) | `npuShare: 1..1000` | Existing First Fit with 1000 units/device | Smooth weighted round robin using share |
| `utilization-best-effort` | `bestEffort: {targetFPS: 20, weight: 100}` | Profiled demand plus observed broker occupancy | Same scheduler, with weight independent of demand |

Exactly one of `npuCount`, `npuShare`, or `bestEffort` is allowed. Best-effort
tasks additionally specify `model.key` and the HEF's lowercase `model.sha256`.
The reference client verifies that checksum before registering or loading the
model. Custom images must honor the checksum, target FPS, and grant protocol.
Best effort does not promise achieved FPS or latency.

## Configuration and admission rule

`deploy/controller.yaml` installs `npu-admission-config`, mounted at
`/etc/npu-admission/admission.json`. The controller reads it once using
`--admission-config`. Without that flag, the default remains static sharing.

```json
{
  "sharedAdmissionPolicy": "utilization-best-effort",
  "budget": 800,
  "profileMargin": 1.2,
  "profiles": {}
}
```

Populate `profiles` with real measurements using the tool below. Empty profiles
are valid configuration but cannot admit a best-effort task. Dispatchers must
have verified `npu.snu-rtos.io/device-architecture` and
`npu.snu-rtos.io/runtime-version` annotations matching the profile. The supplied
dispatcher manifests describe HAILO8 with HailoRT 4.21.0; verify these values if
you change the runtime or hardware.

```text
demand = ceil(1000 × targetFPS × profile.p95Seconds × profileMargin)
load = max(sum of reserved demand, broker occupancy in 0..1000 units)
admit if load + demand <= min(budget, dispatcher capacity)
```

The eligible device with lowest load wins; ties use dispatcher ID. For a 10 ms
P95 cost and 20 FPS, demand is approximately 240 units including the 20% margin.
Three such clients can fit within an 800-unit budget on one NPU. Execution
weights may all be 100; they are not the 240-unit reservations.

Occupancy measures time holding an execution grant, including in-flight work,
over a rolling ten-second window. The controller polls every two seconds. A
complete window and observation no older than ten seconds are required;
missing, malformed, future-dated, or stale samples block new admissions.
Broker restart therefore introduces a telemetry warm-up interval. Occupancy
includes client/runtime overhead and is **not hardware utilization**. Existing
Hailo device/model utilization metrics remain separate observations.

Reservations are reconstructed through authoritative Kubernetes pod reads,
including pending and terminating pods. A terminating pod retains its reservation
until disappearance. Completed/failed pods release it once not terminating.
The single controller worker serializes allocation; do not increase reconcile
concurrency or disable leader election. Overload blocks new admission without
evicting, migrating, or automatically resizing existing tasks.

## Profile a model

Rebuild the shared runner and broker images using the existing
`shared/build-images.sh` procedure. The runner now also supports `--warmup` and
`--profile-json` for isolated exclusive execution. Run this tool in a compatible
Hailo runtime environment with Python 3, the updated runner, matching HEF, and an
otherwise idle NPU; do not profile through a shared HailoRT service.

```sh
python3 npu-task/shared/profile_model.py \
  --runner /absolute/path/to/shared_runner \
  --hef /absolute/path/to/yolov10s.hef \
  --key yolov10s --architecture HAILO8 --runtime-version 4.21.0 \
  --output /tmp/npu-best-effort.json
```

The tool performs 50 warm-up and 500 measured executions, computes the model
checksum, and emits a complete best-effort startup configuration. Architecture
and runtime arguments are operator-verified metadata, not automatic discovery.
Keep CPU, thermal, model, and runtime conditions fixed. Measurement excludes
model configuration/warm-up; residency/switching costs still require co-running
validation and conservative headroom. Merge additional measured profiles into
the output's `profiles` map before installation. No synthetic profile is shipped
as a real hardware measurement.

The supplied runner paces each frame before acquiring its grant and completes
the grant immediately after inference. It never sleeps while holding a token
or issues bursts to catch up after a delayed execution. Existing unpaced static
clients retain their `NEXT` behavior. New best-effort registrations use
`REGISTER workload weight reservedDemand utilization-best-effort`; the legacy
three-field `REGISTER workload share` remains supported. Registration identity
and declared demand remain trusted local-client inputs in this milestone.

## Install, switch, and roll back

Build the updated controller and shared images before upgrading. Apply the CRD
and controller manifests as in the main runbook. The initial configuration is
static, preserving existing task request formats.

For every policy **or profile/configuration** change:

1. Save the current ConfigMap and task specs. Suspend every shared task across
   namespaces (`spec.suspend: true`) and wait until all assigned workload pods
   disappear. Leave exclusive tasks running. Do not force-delete pods to drain.
2. For best effort, set task specs to `bestEffort`, removing `npuShare`, and add
   their measured model keys/checksums. For static sharing, remove `bestEffort`
   and provide `npuShare`. Keep them suspended during conversion.
3. Install the reviewed startup JSON and restart the controller:

   ```sh
   kubectl -n npu-task-system create configmap npu-admission-config \
     --from-file=admission.json=/tmp/npu-best-effort.json \
     --dry-run=client -o yaml | kubectl apply -f -
   kubectl -n npu-task-system rollout restart deployment/npu-task-controller
   kubectl -n npu-task-system rollout status deployment/npu-task-controller
   ```

4. Resume the converted tasks and inspect status/metrics. A fresh dispatcher
   needs ten seconds of observations before best-effort admission.

Rollback follows the same drain procedure, restores the saved static config
and static request specs, restarts the controller, and resumes tasks. Updating
a ConfigMap without restarting does not change the active policy. Reapplying
the full controller manifest restores its bundled static configuration on disk;
install your selected configuration afterward and restart deliberately.

The configuration fingerprint is stored in every new shared allocation. New
shared admissions are blocked by `AdmissionConfigurationMismatch` while active
allocations belong to another fingerprint, including when their dispatcher is
missing. Existing workloads can still run, suspend, or delete. Legacy allocations
without fingerprints are accepted only in static mode for upgrade compatibility.
Shared tasks using the inactive format remain Pending with
`AdmissionPolicyMismatch`; exclusive tasks are unaffected by this guard.

## Observe and verify

`status` separates `admissionPolicy`, `admissionFingerprint`, `profileRevision`,
`reservedDemand`, and `executionWeight`; `allocatedShare` retains its static
meaning. Profile revision hashes the measured profile, while configuration
fingerprint includes policy, limits, and all profiles.

Prometheus adds `npu_admission_configuration_info`, `npu_admission_load_units`,
`nputask_reserved_demand_units`, and `nputask_execution_weight`. Existing
`nputask_phase_info` exposes Pending reasons. Broker metrics include occupancy,
observation timestamp/window, completed grants, queue depths, and grant waits.
The monitor CLI displays policy, reservation, weight, and broker occupancy
alongside independent Hailo observations.

```sh
cd npu-task
go test -race ./...
python3 -m unittest discover -s shared -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
c++ -std=c++17 shared/src/pacing_test.cpp -o /tmp/osh-pacing-test
/tmp/osh-pacing-test
```

The hardware-free tests cover multi-client admission on one logical device,
exhaustion/release, authoritative reconstruction, restarts, policy mismatch,
configuration draining, rolling occupancy, protocol compatibility, and pacing.
For hardware acceptance, profile both supplied HEFs, run several paced clients
per dispatcher, confirm positive grants and completed inference, exercise stale
telemetry and capacity exhaustion, and execute the complete drain/switch/rollback
sequence. Preserve raw results and environment metadata. Unit tests do not
substitute for this hardware campaign.
