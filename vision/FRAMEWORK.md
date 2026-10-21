# Deploy a model, replay a source, and see live measurements

For the exact cleanup → build/import → preflight → run sequence, use
[REPRODUCE.md](REPRODUCE.md). In particular, import v3 images before applying
v3 tasks; otherwise Kubernetes will report `ImagePullBackOff`.

The standard runner manages device discovery, model loading, preprocessing,
inference, decoding, and measurement inside each inference container. Users
select a catalogue entry instead of writing that code or passing physical
device IDs. Kubernetes/controller allocation remains centralized; the viewer
centralizes source distribution, display, and GPU-completion acknowledgments.

| Component | Owns |
|---|---|
| `NPUTask` controller + device plugin | Pod lifecycle and exclusive NPU assignment; other tasks can still use the configured shared-admission policy |
| `vision/models.json` + `vision/osh.py` | User-facing model selection and standard task/image/adapter configuration |
| Standard `vision_worker` in each container | Open the assigned device, load HEF, run preprocessing/inference/postprocessing, measure completed frames |
| Native `vision_hub` / viewer | Decode input once, distribute frames, join timings, show overlays/metrics, acknowledge GPU completion |

The current visual adapters support **one exclusive NPU and batch size one per
container**. The worker uses the count-one inference API; the CLI rejects other
counts/batch sizes rather than silently ignoring them. Generic nonvisual
NPUTasks still support exclusive counts 1–8 and the existing shared paths.
This release does not add broker-based sharing to video workers or centralize
all inference in a single process.

## Build or upgrade once

The standard catalogue and task examples now select **v3** worker images, which
include rolling performance measurements. Older v2 containers continue to run
until you replace them; rebuilding only the viewer cannot add container-local
measurements to an old worker.

From the repository root, on the supported ARM64/HailoRT 4.21 device:

```sh
python3 vision/prepare_assets.py
sudo sh vision/build-images.sh
```

This builds/imports the three v3 model images and refreshes the native viewer
bundle. If Docker RUN is unavailable, use the existing isolated-SDK build
procedure in [README.md](README.md#build-and-deploy). Both build paths default
to v3; an `OSH_VISION_RELEASE` override also requires matching catalogue image
tags. Keep compatible model HEFs and runtime/driver versions together.

Install the current [NPUTask controller](../npu-task/README.md#install) if it is
not already present. On an existing installation, preserve its image and shared
admission configuration; only refresh the CRD if needed:

```sh
kubectl apply -f npu-task/deploy/crd.yaml
python3 vision/osh.py models
```

## Run the complete demo

Run this on the chosen device's display session:

```sh
sudo python3 vision/osh.py demo --node ava
```

This applies the namespace and three standard model tasks, waits for Running,
prepares the checksum-verified pedestrian clip if necessary, then starts the
original/detection/segmentation/pose display. It uses three exclusive NPUs.
Change `ava` to your display node's hostname. With one eligible ready NPU node,
you may omit `--node`; multiple nodes require an explicit choice. The viewer
must run on the same node because the transport is node-local.

Applying the same task names updates those tasks and performs the existing
stop-before-replacement lifecycle when settings/images change. It does not
create duplicate containers. Existing dispatcher pods and their reservations
are preserved. If insufficient NPUs are free, tasks remain Pending and the wait
times out; inspect their conditions rather than stopping other workloads blindly.

Inspect exactly what would be deployed without contacting the cluster:

```sh
python3 vision/osh.py demo --node ava --dry-run
```

For deployment without starting a display:

```sh
python3 vision/osh.py demo --node ava --deploy-only
```

The `vision/deploy.sh` wrapper now uses that operation and preserves the running
controller image. It no longer switches back to `npu-task-controller:vision-v1`.

## Add a container or use a different source

```sh
python3 vision/osh.py deploy detector-2 --model detection --node ava
python3 vision/osh.py deploy pose-2 --model pose --node ava
python3 vision/osh.py status
```

Each new task name creates another independently managed container using the
catalogue's image, HEF path, and adapter. Each also requires one more free NPU.
The running viewer automatically adds a tile when the new task connects.
There is no per-container code to write and no physical device ID to choose.
`--namespace` and `--session` select another deployment; tasks intended for one
viewer must use the same visual session on the same node.
The current viewer accepts up to eight active worker connections per session.

Playback is independent of task deployment. To reuse already-running tasks:

```sh
sudo python3 vision/osh.py view --source /absolute/path/to/video.mp4
# Or a finite run using the default pedestrian video:
sudo python3 vision/osh.py view --frames 200 --output vision/results/framework-200
```

The defaults are session `demo`, 10 FPS source limit, and a timestamped output
directory. Do not run two viewers on the same session/socket. `--headless`
keeps the inference/logging path but makes no GPU/display-completion claim.
Changing the input video requires no container rebuild. Arbitrary model HEFs
are different: they must match one of the supported decoding adapters.

## Live measurements and their meaning

Each task tile displays:

- **RUN FPS:** actual completed inference-pipeline frames per wall-clock second,
  measured inside that container. It includes source starvation/backpressure;
  it is not the inverse of inference latency or a maximum-capacity estimate.
- **infer avg/P95:** host-observed Hailo inference-call latency in milliseconds,
  including runtime transfers/embedded processing, over the rolling window.
- **OUT FPS:** completed displayed results per wall-clock second, measured by
  the hub. Refreshing the same tile does not count as another completed frame.
- **GPU E2E avg/P95:** source acquisition to completed GL work for the exact
  submitted frame. This includes decode, waiting, transport, model processing,
  and rendering; it does not measure physical monitor scanout.
- **drops:** cumulative frames replaced in that client's pending queue.

The first five frames of each worker connection are warm-up and excluded from
rolling summaries. Throughput shows `--` until at least one second of measured
wall time is available. The window is five seconds; stalled tasks decay to zero
FPS and expired latency samples disappear. STARTING, WARMUP, STALE, DISCONNECTED,
and MOCK states remain visible. Reconnection resets that connection's window.
GPU E2E summaries show the most recently completed frames, since the frame
currently being rendered has not completed yet.
Final screenshots are labeled SNAPSHOT and retain statistics from each tile's
last completed frame, rather than reporting the idle drain interval as a live rate.

Container stdout now explicitly includes `runner_config` (model/device/count/
batch identity), `container_durations` for preprocessing/inference/postprocessing/
pipeline time, `worker_metrics` for rolling container statistics, and output
acknowledgments with hub-computed `output_metrics`. The native viewer stores
the joined information with typed predictions in `frames.jsonl`.

```sh
python3 vision/osh.py metrics detection
python3 vision/osh.py logs detection --follow
python3 vision/summarize.py vision/results/framework-200/frames.jsonl
```

`metrics` reads the latest measured frame from container logs, so its timestamp
may be old when no source is playing. The live viewer ages its own display;
the offline summary is grouped by session/task and is not the same as the
five-second rolling window. In headless records the GPU fields remain null or
absent; no inference timing is relabeled as GPU timing.

## Lifecycle and custom models

```sh
python3 vision/osh.py suspend detector-2
python3 vision/osh.py resume detector-2
python3 vision/osh.py delete detector-2
```

Suspension releases the NPU after pod termination; resumption may wait for
capacity. Removing the viewer alone does not stop containers or release NPUs.

For a compatible custom model, copy `vision/models.json`, add a named entry
with an image containing that HEF, its absolute `path`, supported `adapter`,
`npus: 1`, and `batchSize: 1`, then select your catalogue:

```sh
python3 vision/osh.py --catalogue /path/to/my-models.json models
python3 vision/osh.py --catalogue /path/to/my-models.json deploy my-task --model my-model --node ava --dry-run
```

The CLI uses the existing public `NPUTask` API; hand-written tasks/custom images
remain supported. New tensor layouts, batched visual execution, and shared-NPU
visual scheduling require runtime/adapter work, not just a catalogue edit.

Hardware-free checks include rolling-window math, CLI rendering/lifecycle
commands, and fast/slow mock-worker replay with real IPC. Build-time CTest runs
them. Live hardware acceptance still uses `vision/tests/hardware.py`; mock
inference timings are test delays and must not be treated as Hailo benchmarks.
