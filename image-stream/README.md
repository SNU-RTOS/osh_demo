# Camera-free image streaming test

This test replays browser-selected images through the existing version-2 camera
IPC protocol and displays the submitted RGB image beside the corresponding
detection overlay. Linux, a C++17 compiler, CMake, Python 3, and a browser are
enough for mock mode. No Python packages, cameras, GStreamer, or NPU are needed.

From the repository root:

```sh
cmake -S . -B build-image-stream -DOSH_BUILD_CAMERA=OFF -DOSH_BUILD_INFERENCE=OFF
cmake --build build-image-stream -j2
python3 image-stream/serve.py
```

Open **http://127.0.0.1:8090**, select one or more images, and click **Start
stream**. Without a selection it uses `sample.png`. Images repeat in filename
order, resized to 640×640 RGB. Choose 1, 2, 4, or 8 logical channels. Each image
is submitted to every selected channel before advancing to the next image.
The display shows the latest channel; it is a paired viewer, not an eight-tile
dashboard. Stop pauses replay; Ctrl+C in the terminal closes the session.

**Mock mode draws a moving test marker, not a model prediction.** It exercises
the real `comm` shared-memory rings, sequence publication, Unix datagrams, and
overlay path. It does not validate Hailo execution, model accuracy, Kubernetes
allocation, or fractional scheduling.

Target FPS is an upper bound for the entire viewer, divided across selected
channels. One request is in flight at a time: the input is retained until its
matching result arrives. This makes input/output comparison reliable, but does
not reproduce the production camera's concurrent latest-frame/drop behavior.

## Existing Hailo inference driver, without cameras

Build the existing `inference_driver` using the normal Hailo build environment.
Launch these in two terminals **within five seconds of each other**, with the
same task ID:

```sh
# Terminal 1
python3 image-stream/serve.py --mode external --task-id image-test
```

```sh
# Terminal 2: example for a host exposing exactly eight NPUs
OSH_TASK_ID=image-test NPU_COUNT=8 ./build/inference/inference_driver ./yolov10s.hef
```

`NPU_COUNT` must equal the number of accessible devices, as required by the
existing driver. The browser can still submit just one channel. External mode
requires the real NPU and compatible HailoRT/model; it removes the camera
requirement. If the driver is in a pod, expose the same host `/dev/shm` and `/tmp`
endpoints using the existing demo mounts, and provide the matching task ID.
Stop the external driver separately after the viewer exits.

Use a unique task ID for each pair. Do not run two pairs with the same ID.
The bridge removes its RGB ring on normal exit and its mock detection ring in
mock mode. The external driver's detection ring remains owned by that driver.
Restart both processes if startup or inference times out.

## Headless verification

```sh
python3 image-stream/serve.py --smoke-test
ctest --test-dir build-image-stream --output-on-failure
```

The smoke test checks 48 input/result pairs across all eight channels, including
triple-buffer slot reuse and detection payload identity. `--bridge PATH` selects
a separately built bridge. `--port PORT` changes the localhost viewer port.

The fractional-sharing `shared_runner` currently benchmarks generated inputs
without returning decoded detections. Connecting image replay to that runner
is a separate integration step; this viewer validates the camera-facing IPC
path today. See [the current system model](../SYSTEM_MODEL.md).
