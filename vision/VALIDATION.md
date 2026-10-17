# Validation on ava — 17 October 2026

The three release-v2 vision tasks are deployed and Running in `osh-vision`, each
with one distinct physical Hailo-8. Existing dispatcher pods were retained.
The viewer completed hardware OpenGL rendering on **NVIDIA RTX A4500 Embedded
GPU/PCIe**, vendor **NVIDIA Corporation**. No host graphics settings were changed.

## Functional evidence

- Four CTest checks pass: legacy shared-memory ring, legacy image replay,
  new model decoding, and new replay/backpressure/reconnect/malformed-packet handling.
- `go test -race ./...` passes for the controller, dispatcher, monitor, policy,
  resource manager and registry. The controller suite was rerun after the final
  visual-session label change and passes.
- Full-resolution photo replay produced nonempty detection boxes, instance masks,
  and human keypoints simultaneously; the saved screenshot was visually inspected.
- Final video acceptance passes, including distinct models/devices, real
  predictions, GPU completion acknowledgments in all three container logs,
  pose suspend/resume while detection progresses, and invalid HEF failure/release.
- All three final task pods report Running with zero container restarts.

Final artifacts (local, excluded from source control):

- `vision/results/hardware-final/acceptance.json`
- `vision/results/hardware-final/display.png`
- `vision/results/hardware-final/frames.jsonl`
- `vision/results/hardware-final/summary.json`
- `vision/results/hardware-final/{detection,segmentation,pose}.log`

## Observed timing

Input: 960×640 generated pan/zoom video, 10 FPS source-rate limit. Each worker has
a two-CPU limit. The run intentionally suspended/resumed pose; throughput includes
that gap. These are observations from one functional run, not performance bounds.
Five warmup frames per worker connection are excluded from the timing summary.

| Task | Completed frames (including warmup) | Inference-call p95 | Acquisition-to-GPU-completion p95 |
|---|---:|---:|---:|
| Detection, YOLOv10s | 197 | 21.22 ms | 150.41 ms |
| Instance segmentation, YOLOv8n-seg | 132 | 30.35 ms | 339.68 ms |
| Pose, YOLOv8s-pose | 168 | 8.67 ms | 87.11 ms |

Segmentation drops pending frames independently; detection continues near the
source rate. Full-resolution JPEG replay is slower because every acquisition
includes decoding the original 4707×3285 image. No labeled-dataset accuracy
campaign or physical display scanout measurement was performed.

## Environment and reproduction

K3s v1.33.6+k3s1, ARM64, HailoRT 4.21, eight advertised NPUs. The runtime bundles
and images were built using the pinned SDK in `vision/Dockerfile`. Because the
host root disk was full, compilation used an extracted SDK under `/tmp`, and
worker images were packaged through `vision/Dockerfile.bundle`. CNI startup was
unblocked by preserving and relocating the VS Code C++ cache to `/data`, as
recorded in `SYSTEM_MODEL.md`. The root disk still has little operational headroom.

Run `sudo python3 vision/tests/hardware.py --exercise-lifecycle` to reproduce the
hardware checks. See `vision/README.md` for build, replay and cleanup commands.
