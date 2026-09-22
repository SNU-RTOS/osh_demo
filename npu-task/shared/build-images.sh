#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hailort_root="${HAILORT_421_ROOT:-${repo_dir}/npu-task/shared/.cache/hailort-4.21.0}"
image_dir="${repo_dir}/npu-task/shared/image"
builder_image="${HAILORT_BUILDER_IMAGE:-npu-share-builder:ubuntu22.04}"

"${repo_dir}/npu-task/shared/prepare-hailort.sh"
test -x "${hailort_root}/build-service/hailort/hailort_service/hailort_service"
install -D -m 0755 "${hailort_root}/build-service/hailort/hailort_service/hailort_service" "${image_dir}/hailort_service"
install -D -m 0755 "${hailort_root}/build-service/hailort/hailortcli/hailortcli" "${image_dir}/hailortcli"
install -D -m 0755 "${hailort_root}/build-service/hailort/libhailort/src/libhailort.so.4.21.0" "${image_dir}/libhailort.so.4.21.0"
install -D -m 0644 "${repo_dir}/yolov10s.hef" "${image_dir}/yolov10s.hef"

tiny_model="${TINY_YOLOV4_HEF:-}"
if [[ -z "${tiny_model}" ]]; then
    tiny_model="$(find /data/k3s/agent/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots -path '*/fs/models/tiny_yolov4.hef' -print -quit 2>/dev/null || true)"
fi
test -n "${tiny_model}"
install -D -m 0644 "${tiny_model}" "${image_dir}/tiny_yolov4.hef"

docker run --rm -v "${repo_dir}/npu-task:/src" -w /src golang:1.24 \
    sh -c 'CGO_ENABLED=0 go build -trimpath -o /src/shared/image/npu-share-broker ./cmd/npu-share-broker'
docker run --rm -v "${repo_dir}:/workspace" -w /workspace \
    "${builder_image}" bash -lc \
    'g++ -std=c++17 -O2 -pthread -Inpu-task/shared/.cache/hailort-4.21.0/hailort/libhailort/include npu-task/shared/src/shared_runner.cpp -Lnpu-task/shared/.cache/hailort-4.21.0/build-service/hailort/libhailort/src -lhailort -o npu-task/shared/image/shared_runner'

docker build -t npu-share-dispatcher:local -f "${repo_dir}/npu-task/shared/Dockerfile.dispatcher" "${repo_dir}/npu-task/shared"
docker build -t npu-share-client:local -f "${repo_dir}/npu-task/shared/Dockerfile.client" "${repo_dir}/npu-task/shared"
