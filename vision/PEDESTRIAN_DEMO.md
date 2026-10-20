# Alternate video demo: pedestrians in motion

This reproduces the existing four-tile video case with a different input:
OpenCV's `vtest.avi` pedestrian footage, rather than the generated pan/zoom of
`people.jpg`. You see the original video beside independent detection,
instance-segmentation, and pose results. The source loops until you close the
viewer. Video frames are shared by all three containers; overlays retain their
matching source frame even if one worker runs more slowly.

This demo uses the **current framework's exclusive allocation path**: each vision
task requests `npuCount: 1` without choosing a device ID. It works when the shared
admission setting is either `static-share` or `utilization-best-effort`, but these
visual workers do not use `bestEffort` or broker grants. Three free NPUs on the
chosen display node are required. Shared visual inference remains a separate
integration; the [best-effort demo](../npu-task/BEST_EFFORT_REPRODUCE.md) currently
uses generated model inputs rather than video frames.

All commands below run from the repository root on the ARM64 Hailo-8 device.

## 1. Prepare the video

```sh
python3 vision/prepare_video.py
```

The initial download needs network access. The downloader verifies the exact
8,131,690-byte asset and SHA-256 from `video.lock.json` before installing it as
`vision/assets/pedestrians.avi`. Later runs work offline. The clip is pinned to
an OpenCV commit and is not committed as a binary to this repository; see
[asset attribution](assets/ATTRIBUTION.md).

## 2. Check or deploy the three inference containers

If you already ran the original video demo, reuse its containers and native
viewer bundle:

```sh
kubectl -n osh-vision get nputasks
kubectl -n osh-vision get pods -o wide
test -x vision/build/runtime/app/vision_hub
```

The `detection`, `segmentation`, and `pose` tasks should be Running. If you had
suspended one for the lifecycle test, resume it explicitly:

```sh
kubectl -n osh-vision patch nputask pose --type=merge -p '{"spec":{"suspend":false}}'
```

For a fresh installation, follow the [build instructions](README.md#build-and-deploy)
to prepare models, build/import worker images and export the viewer bundle:

```sh
python3 vision/prepare_assets.py
sudo sh vision/build-images.sh
```

Install the current NPUTask controller using the [framework installation
instructions](../npu-task/README.md#install) if it is not already running. When
the current controller is installed, keep it and its selected admission policy;
apply only the CRD and vision tasks:

```sh
kubectl apply -f npu-task/deploy/crd.yaml
kubectl apply -f vision/tasks.yaml
kubectl -n osh-vision wait nputask/detection nputask/segmentation nputask/pose \
  --for=jsonpath='{.status.phase}'=Running --timeout=180s
```

`tasks.yaml` targets node `ava` and visual session `demo`. For another device,
change each task's `spec.visual.nodeName` to the actual node before applying;
run the viewer on that same node. Insufficient free capacity leaves tasks
Pending. Existing dispatcher pods continue reserving their devices; this demo
does not stop them or change the shared admission policy.

Do not use the older `vision/deploy.sh` shortcut on an upgraded deployment:
it selects the `npu-task-controller:vision-v1` image tag. The commands above
preserve the currently running controller version.

## 3. Run the alternate video

Stop any viewer currently using `/data/osh-vision/demo/hub.sock` first. From the
device's display session:

```sh
sudo sh vision/run-pedestrians.sh
```

Expected display: the original pedestrian video plus detection boxes,
segmentation masks, and pose skeletons in the same native OpenGL layout as the
original demo. Predictions depend on person size, occlusion, and model
thresholds; not every visible person will necessarily have every overlay.

The source-rate limit is 10 FPS. Escape or closing the window stops playback;
inference containers remain running and reconnect when you start again. The
script prints a new timestamped results directory under `vision/results/`.
It saves joined `frames.jsonl` records and a final `display.png` screenshot.

For a finite, repeatable run with a known output path:

```sh
sudo sh vision/run-pedestrians.sh --frames 200 \
  --output vision/results/pedestrians-200 \
  --screenshot vision/results/pedestrians-200/display.png
python3 vision/summarize.py vision/results/pedestrians-200/frames.jsonl
```

This acquires 200 frames and drains pending results for five seconds. Use a
fresh output directory for each independent measurement: frame logs append to
existing files. Change `--fps` to request another source rate. Viewer startup
prints the actual GPU renderer and rejects known software renderers by default.
Keep the existing display/GL environment from the original demo; override
`DISPLAY` or the GL vendor only if your display session requires it.

On a node without a display, use `--headless --frames 200`. That exercises video
decoding and inference, but it does not validate visible output or GPU completion.

## 4. Verify inference or try another local video

Use the existing hardware acceptance test with the new source to collect real
predictions, model/device identities, screenshots, and GPU-completion evidence:

```sh
sudo python3 vision/tests/hardware.py \
  --source vision/assets/pedestrians.avi \
  --output vision/results/pedestrians-acceptance
```

This checks all three models and requires nonempty predictions. It is a
functional smoke test, not a labeled accuracy benchmark. Do not run it at the
same time as another viewer. Add `--exercise-lifecycle` only when you also want
the existing pose suspend/resume and invalid-model tests.

For this addition, checksum verification, offline reuse, and actual 768×576
video decoding/replay through the native hub to three mock workers were tested.
Live Hailo predictions and visible GPU rendering for this new clip still require
the hardware acceptance command above; the earlier photo/video measurements in
`VALIDATION.md` are not measurements of this clip.

To replace the input again, keep the same deployed tasks and point the existing
viewer at a local video file; no container rebuild is needed:

```sh
sudo sh vision/run-viewer.sh --source /absolute/path/to/another-video.mp4 \
  --socket /data/osh-vision/demo/hub.sock --fps 10 \
  --output vision/results/my-video
```

The runtime's OpenCV decoder must support that file's codec. If it cannot open
the clip, check the source path and codec rather than changing task allocations.
If no result tiles appear, check task status, the viewer socket/session, and
container logs. Use `sudo sh vision/cleanup.sh` only when you want to remove the
three demo tasks and release their exclusive NPUs; recordings and dispatchers
are retained.
