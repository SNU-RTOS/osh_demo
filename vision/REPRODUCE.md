# Reproduce the current video demo with live FPS and latency

**Use this guide for the current v3 release.** It covers cleanup, building the
matching worker images and viewer, and running the pedestrian video with
original/detection/segmentation/pose tiles. All commands run from the repository
root on the device. It needs the installed NPUTask controller/device plugin,
three free Hailo-8 NPUs, compatible HailoRT/driver 4.21, and the device's graphics
session. The supplied examples target node `ava`.

Each visual container requests one exclusive NPU and batch size one. The shared
admission policy can remain installed; this video demo does not change it or use
fractional sharing. For architecture, metric definitions, and catalogue/custom
model usage, see [FRAMEWORK.md](FRAMEWORK.md).

## 1. Update and remove old demo tasks

Use the `npu-task-reproduction-runbook` branch containing the v3 runner and CLI:

```sh
git fetch origin
git switch npu-task-reproduction-runbook
git pull --ff-only
sudo sh vision/cleanup.sh
kubectl -n osh-vision get nputasks,pods
```

Cleanup removes only the named `detection`, `segmentation`, and `pose` NPUTasks
and waits for their owned pods. It is safe to rerun after those objects are gone.
Additional tasks you created remain; remove those explicitly, for example:

```sh
python3 vision/osh.py delete detector-2
```

Delete the **NPUTask**, not just its pod: the controller recreates a Service
task's pod. Cleanup retains the controller, device plugin, healthy shared
dispatchers, downloaded assets, native build artifacts, and measurement files.
It does not prune Docker containers/images or unrelated Kubernetes workloads.

## 2. Build and import images before deploying

The failure found on this device was `ImagePullBackOff`: tasks referenced
`osh-vision-*:v3`, while the K3s image cache contained only `:v2`. These are local
demo images, not published Docker Hub images. Reapplying YAML or restarting the
pods cannot fix a missing image.

The host root filesystem was also full. For this device, **reuse the prepared
SDK**, which avoids host package installation and Docker RUN:

```sh
test -x /tmp/osh-vision-sdk/usr/bin/cmake
sudo sh vision/rebuild-from-sdk.sh --sdk /tmp/osh-vision-sdk
```

This rebuilds/tests current sources inside the SDK, temporarily mounts and then
unmounts the source tree, saves the previous native viewer bundle under
`vision/build/runtime.previous.*/runtime`, refreshes `vision/build/runtime`,
builds the three v3 images with COPY-only Dockerfiles, and imports them into
K3s's `k8s.io` namespace. It does **not** deploy tasks or change the controller.
Use only a trusted, prepared SDK. Sufficient space is needed on the repository,
SDK, and Docker storage filesystems; the script does not reclaim disk space.

If that prepared SDK does not exist, the existing fallback creates a new one:

```sh
sudo sh vision/build-sdk.sh
```

That path downloads/checks assets, prepares `/tmp/osh-vision-sdk-build`, installs
build dependencies inside it, and builds/imports v3 worker images and the viewer.
It refuses to overwrite an existing SDK directory. You can later reuse its
reported SDK path with `rebuild-from-sdk.sh`. Initial SDK preparation needs network
access. Neither path installs packages on the host.

On a machine where Docker RUN works and root storage is sufficient, the normal
alternative is:

```sh
python3 vision/prepare_assets.py
sudo sh vision/build-images.sh
```

Choose one build path and wait for it to succeed. Building an image in Docker
alone does not make it available to K3s. The provided scripts perform both steps.
For manually built images, import explicitly:

```sh
sudo docker save osh-vision-detection:v3 osh-vision-segmentation:v3 osh-vision-pose:v3 \
  | sudo k3s ctr -n k8s.io images import -
```

On a fresh cluster, install the [current controller and device
plugin](../npu-task/README.md#install) before continuing. On an existing cluster,
preserve its controller image and configuration; do not switch back to an older
controller tag just to run the video demo.

## 3. Verify prerequisites

```sh
sudo python3 vision/osh.py doctor
python3 vision/prepare_video.py
kubectl get nodes -o wide
```

`doctor` is read-only and checks the **local K3s cache**, the native viewer, and
the installed controller. Proceed when `missing_local_images` is `[]`,
`native_viewer_present` is `true`, and `controller_ready_replicas` is positive.
It also reports root/repository free space. This local preflight assumes the
single-node setup; other nodes need their own matching images. It does not
guarantee free devices or a working graphics session.

The video-preparation command downloads and checksum-verifies the pinned
pedestrian clip once; playback thereafter works offline. The same source feeds
all workers. It is different from the original generated pan/zoom video.

## 4. Start the complete demo

From the device's display session:

```sh
sudo python3 vision/osh.py demo --node ava
```

This creates/updates the three model tasks, waits for Running, then starts the
viewer. On another device, replace `ava` with the display node's hostname.
No physical NPU IDs, HEF paths, container commands, or hand-written task YAML are
needed. The catalogue selects the standard runner and compatible model image.

Each result tile shows RUN FPS, inference average/P95 milliseconds, OUT FPS,
GPU end-to-end average/P95, and cumulative drops. The first five frames are
warm-up, and FPS needs at least one measured second before appearing. These
are observed rates/latencies, not throughput or deadline guarantees.

Escape or closing the window stops playback; model containers stay running.
Do not start two viewers on the same session. A later playback-only run is:

```sh
sudo python3 vision/osh.py view
```

For a finite run with an explicit results directory:

```sh
sudo python3 vision/osh.py view --frames 200 --output vision/results/reproduce-v3
python3 vision/summarize.py vision/results/reproduce-v3/frames.jsonl
python3 vision/osh.py metrics detection
```

Use a fresh output directory for independent measurements; logs append to an
existing file. The final `display.png` is labeled SNAPSHOT and retains each
tile's last completed statistics. Container logs contain the runner identity,
stage durations, rolling worker metrics, and output acknowledgments.

## 5. Add a container or change the video

```sh
python3 vision/osh.py deploy detector-2 --model detection --node ava
python3 vision/osh.py status
python3 vision/osh.py metrics detector-2
```

A new task name creates an additional container and viewer tile; it requires one
more free NPU. An existing task name updates that task. The viewer supports up
to eight active workers per session. Insufficient capacity leaves tasks Pending.

To replay your own supported video without rebuilding the models:

```sh
sudo python3 vision/osh.py view --source /absolute/path/to/another-video.mp4
```

## Troubleshooting and stopping

| Symptom | Action |
|---|---|
| `ImagePullBackOff` | Run `doctor`; build/import the exact v3 images into K3s before deploying. Docker Hub credentials will not supply these local images. |
| Task Pending with insufficient `hailo.ai/npu` | Check node capacity and existing allocations. Healthy dispatcher pods reserve NPUs even with no active clients. |
| Docker build fails with no space / Docker RUN cannot start | Use the prepared-SDK build path above; do not prune unrelated data. |
| Tasks Running but no results | Start the viewer on the same node/session and inspect `osh.py logs detection --follow`. |
| FPS `--` / WARMUP | Allow five warm-up frames and at least one second of measured frames. |
| No graphics session or hardware GL | Use the correct device display session. `--headless` tests inference/logging but does not validate visible output or GPU completion. |

When finished:

```sh
python3 vision/osh.py delete detector-2
sudo sh vision/cleanup.sh
```

The cleanup performed for this update removed the three image-pull-backoff
vision tasks/pods and one stopped, failed Docker controller-build container
whose source module was verified as this repository. No images, source files,
results, or unrelated workloads were deleted. Recreate the tasks with step 4.
