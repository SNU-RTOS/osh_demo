# Multi-NPU fractional sharing milestones

The target separates logical allocation from runtime observation:

```text
NPUTask share request
        -> global resource manager (placement and accounting)
        -> one dispatcher bound to one physical NPU
        -> per-NPU weighted scheduler
        -> HailoRT
```

`share` is an admission quantity and execution weight. It is not a throughput
or deadline guarantee. Every physical NPU starts with 1000 logical capacity
units. Measured utilization is telemetry and never changes allocated capacity.

## Current implementation

The original `npuCount` path remains available as the exclusive regression
baseline. The new `npuShare` path allocates logical capacity through the
controller, binds the client to a per-NPU dispatcher, and never exposes the
physical device to the client Pod. The implementation was validated on
2026-09-22 with two Hailo-8 dispatchers and four concurrent client workloads.

## Revised milestones

### M0 — dual-device identity

- Verify two Hailo-8 device IDs on the host and through the device plugin.
- Record stable physical ID, health, and node identity.
- Acceptance: both IDs are independently observable after plugin restart.
- Status: passed; the host and device plugin expose independent physical
  devices (the validation node currently reports eight healthy Hailo-8s).

### M1 — dispatcher ownership

- Run one dispatcher per physical NPU; only dispatchers request `hailo.ai/npu`.
- Bind each dispatcher endpoint to the device ID assigned by CDI.
- Acceptance: two dispatchers execute independent inference concurrently.
- Status: passed with two dispatcher Deployments, each owning one
  `hailo.ai/npu` device and one HailoRT service endpoint.

### M2 — fractional accounting

- Implement `NPUState`, `NPURequest`, `WorkloadAllocation`, and `Allocation`.
- Implement register, unregister, allocate, release, health, and observation.
- Keep `Allocated` and `MeasuredUtilization` independent.
- Status: core ledger implemented in `resource/`; Kubernetes allocation state
  is reconstructed from active workload Pod annotations after restarts.

### M3 — shared dispatcher request path

- Define client-to-dispatcher transport and model lifecycle.
- Route multiple workload queues to one dispatcher without direct device access.
- Release allocations on workload deletion and stale-session expiry.
- Acceptance: two models share one dispatcher and both complete inference.
- Status: passed; `yolov10s` and `tiny_yolov4` clients share each dispatcher
  without receiving a physical device resource.

### M4 — multi-NPU placement

- Connect the control plane to the dispatcher registry and allocation ledger.
- Start with `FirstFitPolicy`; persist allocation state or reconstruct it safely.
- Expose the chosen NPU/dispatcher and allocated/available capacity.
- Status: First Fit is connected to `NPUTask.spec.npuShare`. Shared client Pods
  receive no physical device resource; the controller injects the selected
  dispatcher endpoint and pins the Pod to the dispatcher's node.

### M5 — weighted runtime multiplexing

- Connect each dispatcher's ready queues to weighted round robin.
- Treat share as a relative opportunity weight and publish queue depth.
- Status: queue-aware scheduling primitive implemented in `dispatcher/`; HailoRT
  execution is gated by the broker's smooth weighted round-robin grants. The
  validation run recorded 2,000 grants for each assigned workload.

### M6 — monitoring and evaluation

- Export allocated/available share, measured utilization, queue depth, active
  workloads, throughput, average latency, and P95 latency per NPU.
- Compare exclusive and shared runs with two NPUs and four workloads.
- Status: passed. HailoRT monitor metrics expose per-device utilization and
  per-model utilization/FPS independently from allocated share. The reproducible
  comparison is stored in `results/multi-npu-share.json`.

## Validation result

Strict First Fit placed shares `600+400` on `npu-0` and `600+400` on `npu-1`.
An additional share request remained pending with `InsufficientNPUShare`, then
ran after capacity was released. All four shared workloads completed inference.
The measured throughput and latency values are observations, not share
guarantees.
