#!/usr/bin/env python3
"""Milestone: concurrent processes using different HEFs and NPU counts.

The caller supplies two probe images, each containing a different HEF. The test
uses one isolated namespace and removes it when finished.
"""
import argparse
import json
import pathlib
import subprocess
import time
import uuid


def kubectl(*args, obj=None, check=True):
    result = subprocess.run(["kubectl", *args], input=json.dumps(obj) if obj else None,
                            text=True, capture_output=True, timeout=45)
    if check and result.returncode:
        raise RuntimeError(result.stderr)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-a", default="npu-task-probe:local")
    p.add_argument("--model-a", default="/models/yolov10s.hef")
    p.add_argument("--image-b", default="npu-task-probe-second:local")
    p.add_argument("--model-b", default="/models/tiny_yolov4.hef")
    p.add_argument("--count-a", type=int, default=1)
    p.add_argument("--count-b", type=int, default=2)
    p.add_argument("--count-pending", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output", default="npu-task/results/multi-model.json")
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()
    namespace = "nputask-multi-model-" + uuid.uuid4().hex[:8]
    result = {"namespace": namespace, "tasks": {}, "passed": False}

    def get(kind, name):
        return json.loads(kubectl("get", kind, name, "-n", namespace, "-o", "json").stdout)

    def submit(name, image, model, count):
        kubectl("apply", "-f", "-", obj={"apiVersion": "npu.snu-rtos.io/v1alpha1", "kind": "NPUTask",
            "metadata": {"name": name, "namespace": namespace}, "spec": {
                "image": image, "command": ["/app/npu_probe"],
                "args": ["$(MODEL_PATH)", "0"], "model": {"path": model},
                "npuCount": count, "mode": "Service"}})

    def wait_phase(name, phase):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            task = get("nputask", name)
            status = task.get("status", {})
            if status.get("phase") == phase:
                return task
            if status.get("phase") == "Failed" and phase != "Failed":
                raise RuntimeError(f"{name}: {status}")
            time.sleep(.5)
        raise TimeoutError(f"{name} did not reach {phase}")

    def infer(name, model, count):
        task = wait_phase(name, "Running")
        pod = task["status"]["podName"]
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            logs = kubectl("logs", pod, "-n", namespace).stdout
            ids = {line.split("device=", 1)[1].split()[0] for line in logs.splitlines()
                   if line.startswith("INFERENCE_OK device=")}
            if len(ids) >= count and f"model={pathlib.Path(model).name}" in logs:
                return task, pod, ids, logs
            time.sleep(.5)
        raise TimeoutError(f"{name}: expected model={model} and {count} devices; logs:\n{logs}")

    kubectl("create", "namespace", namespace)
    try:
        submit("model-a", args.image_a, args.model_a, args.count_a)
        submit("model-b", args.image_b, args.model_b, args.count_b)
        _, pod_a, ids_a, logs_a = infer("model-a", args.model_a, args.count_a)
        _, pod_b, ids_b, logs_b = infer("model-b", args.model_b, args.count_b)
        if not ids_a.isdisjoint(ids_b):
            raise AssertionError(f"device overlap: A={ids_a}, B={ids_b}")
        result["tasks"]["model-a"] = {"pod": pod_a, "model": args.model_a,
                                         "npuCount": args.count_a, "devices": sorted(ids_a)}
        result["tasks"]["model-b"] = {"pod": pod_b, "model": args.model_b,
                                         "npuCount": args.count_b, "devices": sorted(ids_b)}

        submit("waiting", args.image_a, args.model_a, args.count_pending)
        wait_phase("waiting", "Pending")
        kubectl("patch", "nputask", "model-a", "-n", namespace, "--type=merge",
                 "-p", '{"spec":{"suspend":true}}')
        wait_phase("model-a", "Suspended")
        _, pod_wait, ids_wait, logs_wait = infer("waiting", args.model_a, args.count_pending)
        if not ids_wait.isdisjoint(ids_b):
            raise AssertionError(f"released devices overlap model B: {ids_wait} vs {ids_b}")
        after_b = kubectl("logs", pod_b, "-n", namespace).stdout
        if len(after_b) <= len(logs_b):
            raise AssertionError("model B stopped while model A released devices")
        result["tasks"]["waiting"] = {"pod": pod_wait, "model": args.model_a,
                                         "npuCount": args.count_pending, "devices": sorted(ids_wait)}
        result["passed"] = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
        if not args.keep:
            kubectl("delete", "namespace", namespace, "--wait=false")


if __name__ == "__main__":
    main()
