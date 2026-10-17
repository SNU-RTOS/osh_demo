# Existing pinned HailoRT 4.21 SDK/toolchain; the application build does not change host packages.
FROM ghcr.io/snu-rtos/osh-compile@sha256:00fc30abffc759106c0bfd0eb8a2a0d2ea8eb3e2a958934e59189dade47be991 AS build
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev=4.5.4+dfsg-9ubuntu4 libsdl2-dev=2.0.20+dfsg-2ubuntu1.22.04.1 libglvnd-dev=1.4.0-1 nlohmann-json3-dev=3.10.5-2 python3=3.10.6-1~22.04.1 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY comm comm
COPY camera camera
COPY inference inference
COPY image-stream image-stream
COPY vision vision
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DOSH_BUILD_CAMERA=OFF -DOSH_BUILD_VISION=ON && cmake --build /build -j4
FROM ubuntu:22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d libopencv-videoio4.5d libsdl2-2.0-0 libglvnd0 libgl1 libegl1 libglx0 ca-certificates python3 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/lib/libhailort.so* /usr/lib/
COPY --from=build /usr/include/hailo /usr/include/hailo
COPY --from=build /build/vision/vision_* /app/
COPY vision/assets /assets
ENTRYPOINT ["/app/vision_worker"]
FROM runtime AS detection
COPY yolov10s.hef /models/yolov10s.hef
FROM runtime AS segmentation
COPY vision/models/yolov8n_seg.hef /models/yolov8n_seg.hef
FROM runtime AS pose
COPY vision/models/yolov8s_pose.hef /models/yolov8s_pose.hef
