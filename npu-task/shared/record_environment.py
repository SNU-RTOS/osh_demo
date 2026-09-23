#!/usr/bin/env python3
"""Record the software, hardware, model, and image identity for a campaign."""

import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def command(*args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": list(args), "exit_code": result.returncode, "output": result.stdout.strip()}


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    output = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "npu-task/results/environment.json")
    # Hash the exact files copied into the client image by build-images.sh.
    model_paths = [ROOT / "npu-task/shared/image/yolov10s.hef",
                   ROOT / "npu-task/shared/image/tiny_yolov4.hef"]
    models = [{"path": str(path), "sha256": digest(path), "size": path.stat().st_size}
              for path in model_paths if path.is_file()]
    report = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine(), "kernel": platform.release()},
        "git": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--porcelain"),
        "kubernetes": command("kubectl", "version", "--output=json"),
        "nodes": command("kubectl", "get", "nodes", "-o", "wide"),
        "node_capacity": command("kubectl", "get", "nodes", "-o", "custom-columns=NAME:.metadata.name,HAILO:.status.allocatable.hailo\\.ai/npu"),
        "hailo_scan": command("hailortcli", "scan"),
        "hailo_identity": command("hailortcli", "fw-control", "identify"),
        "models": models,
        "images": command("docker", "image", "inspect", "npu-task-controller:share",
                          "npu-share-dispatcher:local", "npu-share-client:local", "--format", "{{.RepoTags}} {{.Id}}"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
