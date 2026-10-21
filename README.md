
---

# Execution Guide

**To reproduce the current video demo:** follow
[vision/REPRODUCE.md](vision/REPRODUCE.md), including cleanup, v3 image import,
the prepared-SDK build option, and the `doctor` preflight before deployment.

For catalogue-based model deployment and live per-container FPS/latency in the
video display, start with [the vision framework guide](vision/FRAMEWORK.md).

For a reproducible alternate-video version of the original/detection/
segmentation/pose demo, follow [the pedestrian-video demo](vision/PEDESTRIAN_DEMO.md).

To enable utilization-aware NPU sharing and deploy additional inference
containers without choosing device IDs, follow the
[best-effort sharing guide](npu-task/BEST_EFFORT_REPRODUCE.md).

For the multi-container detection, instance segmentation, and pose framework
with prerecorded input, per-stage latency logs, and native GPU display, see
[vision/README.md](vision/README.md).

For a camera-free, side-by-side image replay viewer (including a no-NPU mock
mode), see [image-stream/README.md](image-stream/README.md). The current
architecture, resource model, lifecycle, and validation boundaries are described
in [SYSTEM_MODEL.md](SYSTEM_MODEL.md).

For managed start/suspend/resume/resize and multi-model NPU tasks, see the
[NPU task framework](npu-task/README.md). To reproduce the complete build,
deployment, lifecycle, recovery, and cleanup procedure, use the
[NPU reproduction runbook](npu-task/REPRODUCE.md). The guide below is the camera demo.

This document describes the step-by-step procedure to build and run the OSH demo with K3s, Hailo NPU, and camera overlay.

---

## 0. Change the work directory of k3s
If there is no ephemeral storage left for k3s to use, you can move the working directory of k3s with the following commands.

```bash
sudo systemctl stop k3s
sudo mkdir -p /data/k3s # Let's assume that we are going to use the /data directory
sudo mkdir -p /etc/systemd/system/k3s.service.d
# Add a sysetmd override so k3s uses /data/k3s
sudo tee /etc/systemd/system/k3s.service.d/override.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/k3s server --data-dir /data/k3s
EOF
# Reload system daemon and restart k3s
sudo systemctl daemon-reload
sudo systemctl start k3s
```

---

## 1. Upgrade K3s Version

First, check the installed K3s version.

```bash
k3s --version
```

If your K3s version is **lower than**:

* Version: `v1.33.6+k3s1`
* Go version: `go1.24.9`

upgrade K3s as follows.

```bash
systemctl stop k3s
curl -sfL https://get.k3s.io | sh -
```

---

## 2. Remove Existing CNI Configuration

Remove the default containerd CNI configuration to avoid conflicts.

```bash
rm /etc/cni/net.d/cni-containerd-net.conf
```

---

## 3. Clone the Git Repository

Clone the OSH demo repository.

```bash
git clone https://github.com/SNU-RTOS/osh_demo
```

---

## 4. Build Binaries

Build the required binaries using CMake.

```bash
cd osh-demo
docker run --rm -it -v "$PWD":/workspace -w /workspace ghcr.io/snu-rtos/osh-compile /bin/bash
mkdir build && cd build
cmake ..
make -j8
exit
```

---

## 5. Deploy Hailo Device Plugin

Deploy the Hailo device plugin to expose the NPU resource to Kubernetes.

```bash
kubectl create -f https://raw.githubusercontent.com/SNU-RTOS/hailo-device-plugin/main/deploy/hailo-device-plugin.yaml
```

For more details, refer to:
[https://github.com/SNU-RTOS/hailo-device-plugin](https://github.com/SNU-RTOS/hailo-device-plugin)

---

## 6. Run the Inference Driver Pod

Start the camera and inference processes together (within five seconds). The pod
runs inference directly; exiting the binary terminates the pod and releases NPUs.
Rebuild both binaries: the shared-memory protocol is now version 2 and supports
all 24 frame slots.

```bash
kubectl apply -f inference_driver.yaml
kubectl logs -f inference-driver
```

---

## 7. Run Camera Overlay Binary

Run the camera overlay binary on a system with a display connected to the board.

**Note**

* Use a separate terminal from the inference driver pod.
* Make sure a display is connected to the target board.

```bash
cd build/camera
OSH_TASK_ID=demo ./camera_overlay # It will use total 8 cameras from /dev/video100 to /dev/video107
```

---

## 8. Run Inference Driver without Hailo Device Plugin

Verify that our Hailo Device Plugin enables access of Hailo NPUs from containers.

```bash
kubectl apply -f inference_driver_without_npu.yaml
kubectl exec -it inference-driver-wo-npu -- /bin/bash
cd build/inference
./inference_driver ../../yolov10s.hef
```

---

## Error Handling
Both applications must use the same `OSH_TASK_ID`, be rebuilt together, and start
within five seconds of each other. A stale version-1 shared-memory ring is rejected;
the rebuilt producer initializes a version-2 ring.

---

## 9. Run or reproduce the multi-container vision framework (no cameras)

This framework runs **object detection, instance segmentation, and human pose**
in three separate containers, each with its own model and one exclusive Hailo-8.
A shared native OpenGL viewer displays the source and each task's result. Every
container logs preprocessing, inference, postprocessing, and frame-correlated
GPU completion timings. NPU contention-policy changes are deferred.

### Run the framework already installed on this device

Run these commands from the repository root on `ava`, with its display connected.
The three tasks should be `Running` in namespace `osh-vision`:

```sh
cd /data/osh_demo
sudo kubectl -n osh-vision get nputasks
sudo kubectl -n osh-vision get pods

# Loop the downloaded image as a stream.
sudo vision/run-viewer.sh \
  --source vision/assets/people.jpg \
  --socket /data/osh-vision/demo/hub.sock \
  --output vision/results/image
```

For video replay, close the image viewer first and run:

```sh
sudo vision/run-viewer.sh \
  --source vision/assets/replay.mp4 \
  --fps 10 \
  --socket /data/osh-vision/demo/hub.sock \
  --output vision/results/video
```

You can substitute a local image or video with `--source /absolute/path/to/file`.
The generated video is a ten-second pan/zoom animation of the licensed sample
photo. Assets are local during playback; no camera or internet connection is
needed. Press Escape or close the window to stop the viewer. The containers
retain their models and reconnect when the viewer starts again. Only one viewer
can own a given session socket.

The viewer prints its actual OpenGL vendor and renderer at startup. Hardware
rendering was verified with the **NVIDIA RTX A4500 Embedded GPU**. Software
renderers are rejected by default. `DISPLAY` defaults to `:0`; set it explicitly
if using another display session. `--headless` permits inference/logging without
display and explicitly omits GPU completion measurements.

### Reproduce the build and deployment

Prerequisites are an ARM64 Linux edge node, Docker, Python 3, K3s/kubectl,
matching HailoRT 4.21 driver/device access, and at least three free Hailo-8 NPUs.
Install the repository's [concurrent-workload device-plugin fix](npu-task/plugin/README.md)
if setting up a fresh node. The display also needs a working NVIDIA driver and
an accessible X11 session. Internet access is required for the initial images,
build dependencies, and checksum-verified assets.

From a checkout of this repository:

```sh
# Verify/download the segmentation and pose HEFs and licensed sample photo.
# The existing yolov10s.hef is also checksum-verified.
python3 vision/prepare_assets.py

# Build/import three task images and the controller, export the native viewer
# bundle, and generate vision/assets/replay.mp4.
sudo vision/build-images.sh
```

For a fresh controller installation, first create its CRD, namespace, and RBAC:

```sh
sudo kubectl apply -f npu-task/deploy/crd.yaml -f npu-task/deploy/controller.yaml
```

Then deploy the current controller and three visual tasks:

```sh
sudo vision/deploy.sh
sudo kubectl -n osh-vision wait nputask/detection nputask/segmentation nputask/pose \
  --for=jsonpath='{.status.phase}'=Running --timeout=120s
sudo kubectl -n osh-vision get pods -o wide
```

`vision/tasks.yaml` targets node `ava`, visual session `demo`, and the local
release-v2 task images. Change `visual.nodeName` for a different device. If you
change `visual.session`, also change the viewer socket to
`/data/osh-vision/<session>/hub.sock`. Each task must discover only its assigned
NPU. Existing shared-NPU dispatchers are retained, so their allocations count
against the devices available to these exclusive tasks. Insufficient capacity
leaves a task Pending.

If a full host root filesystem prevents Docker RUN, the temporary-SDK fallback
used for this implementation builds the worker images and viewer under `/tmp`:

```sh
sudo vision/build-sdk.sh

# With a local Go 1.24 toolchain, package the controller without Docker RUN.
CGO_ENABLED=0 go -C npu-task build -o ../vision/build/controller ./cmd/controller
sudo docker build -f vision/Dockerfile.controller-binary \
  -t npu-task-controller:vision-v1 vision
sudo docker save npu-task-controller:vision-v1 | sudo k3s ctr images import -

# Install controller resources first if this is a fresh node, as above.
sudo vision/deploy.sh
```

The fallback installs dependencies only inside its temporary SDK and removes
its bind mounts on exit. It requires root access, sufficient `/tmp` space, and
an unused `/tmp/osh-vision-sdk-build` directory; set `OSH_VISION_SDK_ROOT` to a
new `/tmp/osh-vision-*` path for another run. Kubernetes still needs root-disk
space for CNI state and container logs; the fallback does not remove that need.

### Inspect latency logs and control tasks

The viewer appends joined records to its output directory. Each task processes
frames independently, replacing old pending frames when it falls behind, and
its tile shows the exact source frame associated with the displayed result.

```sh
# Summarize p50/p95/p99 latency, throughput, and dropped frames.
python3 vision/summarize.py vision/results/video/frames.jsonl

# Read a container's pipeline timings and GPU-completion acknowledgments.
POD=$(sudo kubectl -n osh-vision get nputask pose -o jsonpath='{.status.podName}')
sudo kubectl -n osh-vision logs "$POD" --tail=20

# Suspend pose, release its NPU, and let the other tasks continue.
sudo kubectl -n osh-vision patch nputask pose --type=merge \
  -p '{"spec":{"suspend":true}}'
sudo kubectl -n osh-vision wait nputask/pose \
  --for=jsonpath='{.status.phase}'=Suspended --timeout=60s

# Resume pose; it reconnects to the running viewer automatically.
sudo kubectl -n osh-vision patch nputask pose --type=merge \
  -p '{"spec":{"suspend":false}}'
```

Acquisition timing starts before decoding. Inference timing covers the HailoRT
call, including operations embedded in the HEF; host postprocessing is measured
separately. End-to-end display timing ends when submitted OpenGL work completes,
before buffer swap, and does not measure physical screen scanout. The first five
frames of each worker connection are marked warmup. The source FPS is an upper
bound; the full-resolution photo is more expensive to decode than the video.

### Validate and clean up

Close any existing viewer before starting the hardware test:

```sh
# Real inference, distinct NPUs/models, NVIDIA rendering, pose suspend/resume,
# and failure/cleanup of an intentionally invalid model task.
sudo python3 vision/tests/hardware.py --exercise-lifecycle

# Stop the three demo containers and release their NPUs.
sudo vision/cleanup.sh
```

The test writes its report, screenshot, timings, and container-log excerpts to
`vision/results/hardware-video/`. Cleanup preserves assets, measurements, the
controller, and existing dispatcher workloads. Reapply `vision/tasks.yaml` to
create the demo tasks again.

See [vision/README.md](vision/README.md) for transport/model-adapter details,
[vision/VALIDATION.md](vision/VALIDATION.md) for the measured implementation
results, and [asset attribution](vision/assets/ATTRIBUTION.md) for licenses.
Downloaded HEFs, media, native bundles, and generated results are excluded from
Git; the scripts recreate them.
