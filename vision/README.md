# Multiple vision containers on the edge device

**Start here:** [current v3 reproduction steps](REPRODUCE.md). This is the
build/import/preflight/run guide, including recovery from missing images and
the full-root-filesystem case on the development device.

Use [the standard framework CLI and live-metrics guide](FRAMEWORK.md) to choose
a model, deploy containers, and see rolling FPS/latency on every result tile.
The current runner release is v3; rebuild worker images and the native viewer
before using the new measurement fields.

For the same four-tile pipeline with a different, checksum-pinned motion video,
see [the pedestrian-video demo](PEDESTRIAN_DEMO.md). It includes preparation,
deployment with the current controller, playback, and saved-result validation.

One prerecorded source feeds independent detection, instance segmentation, and
human pose containers. Each container owns one Hailo-8, runs preprocessing →
inference → model-specific postprocessing, and publishes typed results to a
shared native OpenGL viewer. The viewer renders the original plus one tile per
task, then returns GPU completion acknowledgments for each container's log.
No camera, browser, or network connection is needed during playback.

## Run the installed demo

The `osh-vision` namespace contains the three Service tasks. From the repository
root, on the device's display session:

```sh
# Photo replay, with the original and three result tiles.
sudo vision/run-viewer.sh \
  --source vision/assets/people.jpg \
  --socket /data/osh-vision/demo/hub.sock \
  --output vision/results/live

# Or loop the generated ten-second pan/zoom video.
sudo vision/run-viewer.sh \
  --source vision/assets/replay.mp4 \
  --socket /data/osh-vision/demo/hub.sock \
  --output vision/results/live-video
```

Escape or closing the window stops the viewer. Tasks keep their models loaded
and reconnect when it starts again. `--fps 10` is the source-rate upper bound;
`--frames 200` ends after 200 acquisitions plus a five-second drain. Add
`--screenshot PATH.png` to save the final displayed tiles on a finite run.
`--headless` runs inference and logging without claiming GPU/display completion.

The native bundle supplies SDL/OpenCV and GLVND. The installed NVIDIA driver
stays on the host. The viewer defaults to `DISPLAY=:0` and NVIDIA GLX; override
those environment variables for another desktop/GPU. Startup prints the actual
renderer and rejects known software renderers. `--allow-software` is an explicit
debug option and does **not** satisfy GPU acceptance. Hardware rendering was
verified on this device's NVIDIA RTX A4500 Embedded GPU.

## Build and deploy

Download/check the assets before building:

```sh
python3 vision/prepare_assets.py
sudo vision/build-images.sh
sudo vision/deploy.sh
```

The standard Dockerfile uses the existing SDK by digest, HailoRT 4.21, and pinned
SDK development packages. It also exports a native viewer bundle and generates
the replay video. Each final task image contains exactly its own HEF.
`tasks.yaml` targets node `ava`, session `demo`, and image release `v3`; change
these together for another deployment. Existing dispatcher workloads are retained.
Three free exclusive NPUs are required; insufficient capacity leaves tasks Pending.
Build scripts import a controller image but do not install or switch the active
controller. Install the current controller separately on a fresh cluster using
the framework runbook; `vision/deploy.sh` preserves an existing installation.

For a host whose full root disk prevents Docker RUN, use:

```sh
sudo vision/build-sdk.sh
# Build the Go controller through npu-task/Dockerfile, import it as
# npu-task-controller:vision-v1, then run vision/deploy.sh.
```

The fallback builds/tests inside a temporary SDK and makes COPY-only worker
images, plus `vision/build/runtime` for native display. It installs no host
packages and unmounts its temporary bind mounts on exit. The temporary SDK is
retained for inspection. `package_sdk.py` can also package an existing SDK with
compiled binaries under `/build/vision`. Rebuild the controller normally with:

```sh
sudo docker build -t npu-task-controller:vision-v1 npu-task
sudo docker save npu-task-controller:vision-v1 | sudo k3s ctr images import -
```

If Docker RUN is unavailable, a local Go 1.24 toolchain can instead build
`npu-task/cmd/controller` with `CGO_ENABLED=0` into `vision/build/controller`;
`vision/Dockerfile.controller-binary` packages it without RUN instructions.

Native build prerequisites are C++17, CMake, OpenCV, SDL2, GLVND/OpenGL,
nlohmann-json, and HailoRT 4.21. Enable `OSH_BUILD_VISION=ON`; disable
`OSH_BUILD_INFERENCE` to build the viewer and mock workers without HailoRT.
The existing camera applications remain an independent build option.

## Task lifecycle and measurement

```sh
kubectl -n osh-vision get nputasks
kubectl -n osh-vision patch nputask pose --type=merge -p '{"spec":{"suspend":true}}'
kubectl -n osh-vision patch nputask pose --type=merge -p '{"spec":{"suspend":false}}'
# Per-container acquisition, pipeline and GPU completion records:
kubectl -n osh-vision logs "$(kubectl -n osh-vision get nputask pose -o jsonpath='{.status.podName}')"
python3 vision/summarize.py vision/results/live/frames.jsonl
sudo vision/cleanup.sh
```

Tasks drop old pending frames independently. Each has one frame in flight and
one pending latest frame. A tile retains its exact source frame while other
tasks advance. Starting/stale/disconnected states are labeled; disconnected
tiles expire after five seconds. Restarted tasks register automatically.
The source is decoded once per acquisition and distributed to all workers;
frames larger than 1920×1080 are scaled to fit before transport. Original source
aspect ratio is retained, then each adapter letterboxes to its 640×640 input.

The viewer appends joined records to `frames.jsonl`, including typed predictions,
frame/session/task/model/device identity, raw monotonic nanosecond timestamps,
warmup flags, and stage durations in milliseconds. Worker stdout has compact
inference timing records and matching output acknowledgments. Explicit drop
records identify replaced pending sequences. The summary excludes the first
five frames of each worker connection and reports p50/p95/p99 and throughput.

Acquisition starts before image/video decoding. `inference_ms` measures the
host-observed Hailo call, including transfers and any embedded processing;
`postprocess_ms` measures only remaining host decoding. Rendering uses GPU
textures, alpha blending, and geometry for masks/boxes/skeletons. `glFinish`
establishes completion of submitted GL work before buffer swap. The end-to-end
measurement ends there; it does not measure physical display scanout.
Headless records set `displayed=false`, `gpu_complete_ns=null`, and omit GPU
latency durations. Source media timestamps remain separate from monotonic time.

The full-resolution photo is costly to decode repeatedly. Use the 960×640 video
for a lighter stream. No FPS or deadline guarantee is claimed.

## Interfaces and model adapters

`NPUTask.spec.visual` optionally supplies `{session, nodeName}`. Visual tasks
require one exclusive NPU. The controller selects the node and mounts only
`/data/osh-vision/<session>` at `/run/osh-vision`; it injects `OSH_VISION_SOCKET`,
`OSH_VISION_TASK`, and `OSH_VISION_UID`. These environment names are reserved.
Visual changes participate in normal stop-before-replacement lifecycle handling.
Model PVCs continue to work alongside the visual-session mount.

The new transport is independent of version-2 camera IPC. Unix stream packets
have two network-order uint32 lengths followed by UTF-8 JSON and optional RGB
bytes. Protocol version 1 exchanges `hello`, `frame`, `result`, and `ack` messages.
Messages are bounded to 8 MiB metadata and 16 MiB pixels; incomplete/malformed
messages close only that peer. Read/write timeouts bound disconnected sessions.
Session + sequence identify a source frame; task name and Kubernetes UID identify
its consumer. These node-local sockets assume trusted local workloads.

The pinned adapters support:

- **Detection:** YOLOv10s's 80-class Hailo NMS BY CLASS output. Its per-class
  records are compact variable-length lists; NMS is already embedded.
- **Instance segmentation:** YOLOv8n-seg raw DFL box/class/coefficient heads and
  prototypes. Host processing performs DFL decoding, class-aware NMS, mask
  reconstruction/cropping, and original-frame run-length encoding.
- **Pose:** YOLOv8s-pose raw DFL box/class/keypoint heads. Host processing performs
  DFL decoding, NMS, and COCO-17 keypoint/visibility decoding.

HailoRT transforms native FCR and quantized outputs into NHWC float buffers.
Adapters reject unsupported tensor layouts. Arbitrary HEFs require their own
adapter; changing only `model.path` is insufficient. New contention control and
visual shared-NPU inference are deferred. The existing broker is unchanged.

## Validation

```sh
ctest --test-dir YOUR_BUILD --output-on-failure
# Three real models + hardware GL + task suspend/resume + invalid model cleanup:
sudo python3 vision/tests/hardware.py --exercise-lifecycle
```

The hardware test saves acceptance JSON, a screenshot, joined timings and compact
container-log excerpts. It requires meaningful nonempty predictions, distinct
model/device identities, and real NVIDIA completion records. This is a functional
smoke test, not a labeled-dataset accuracy evaluation.

Assets, model checksums and licenses are tracked by `assets.lock.json` and
[the attribution file](assets/ATTRIBUTION.md). The generated video animates one
photo; it is not real footage of motion.

The device-specific measured results are recorded in [VALIDATION.md](VALIDATION.md).
