# Hailo utilization and NPU sharing

## What HailoRT exposes

The Hailo-8 HailoRT branch contains `hailortcli monitor`. An inference
process must be started with `HAILO_MONITOR=1`; the monitor reports device and
model utilization percentage and FPS. This is useful telemetry for capacity
decisions and for verifying that a task is using its assigned device.

The monitor is not a Kubernetes allocator. It reports an observed interval
and does not reserve a fraction of a device. A device at 5% during one sample
may still have a model loaded, queued work, or a latency requirement that
makes sharing unsafe.

## Milestone: utilization-aware allocation

1. Collect monitor output as node-level metrics (Prometheus or a small
   `NPUDeviceMetrics` CRD).
2. Keep exclusive `hailo.ai/npu` allocation. Use recent utilization and task
   SLOs to choose an idle device, reject a request, or recommend a smaller
   `npuCount` on the next execution. Do not overcommit a physical device yet.
3. Release devices by terminating the task pod. Resize or suspend must wait
   for pod termination before a device is considered free; the device plugin
   has no separate user-facing `Deallocate` operation.

## Milestone: true sharing

Fractional allocation requires a dispatcher. The dispatcher owns each
physical Hailo device through the device plugin, while application pods send
model/inference requests to it. It can use measured utilization, queue depth,
model load cost, deadlines, and fairness to schedule requests. This is the
point at which two tasks can share one physical NPU safely. A normal device
plugin cannot provide this because kube-scheduler sees integer whole-device
resources.

Live preemption still requires cooperation from the inference client (drain,
checkpoint, or cancellation). A low utilization sample must never take a
device away from a running HailoRT client.

## Board-side monitor check

```sh
export HAILO_MONITOR=1
<start an NPUTask/inference process>
hailortcli monitor
```

The output should contain device and model tables with `Utilization (%)` and
the model table should include `FPS`. If no files are reported, check that
the application and monitor use the same HailoRT monitor directory and that
the HailoRT service configuration is consistent with the process.

