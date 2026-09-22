#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${HAILORT_421_ROOT:-${repo_dir}/npu-task/shared/.cache/hailort-4.21.0}"
expected_commit="0df636dcb6be9b3943a458591ad5213674a9845d"
builder_image="${HAILORT_BUILDER_IMAGE:-npu-share-builder:ubuntu22.04}"

if ! docker image inspect "${builder_image}" >/dev/null 2>&1; then
    docker build -t "${builder_image}" -f "${repo_dir}/npu-task/shared/Dockerfile.builder" "${repo_dir}/npu-task/shared"
fi

if [[ ! -d "${source_dir}/.git" ]]; then
    mkdir -p "$(dirname "${source_dir}")"
    git clone --depth 1 --branch v4.21.0 https://github.com/hailo-ai/hailort.git "${source_dir}"
fi
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${expected_commit}"

docker run --rm \
    -e CCACHE_DIR=/ccache \
    -v "${repo_dir}:/workspace" \
    -v "${repo_dir}/npu-task/shared/.cache/ccache:/ccache" \
    -w "/workspace/npu-task/shared/.cache/hailort-4.21.0" \
    "${builder_image}" bash -lc \
    'cmake -S . -B build-service -DCMAKE_BUILD_TYPE=Release -DHAILO_BUILD_SERVICE=ON && cmake --build build-service -j"${HAILORT_BUILD_JOBS:-8}"'
