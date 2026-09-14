#!/usr/bin/env python3
"""Real-device lifecycle acceptance test. Uses a newly created, isolated namespace.

Requires an installed NPUTask controller and a preloaded image containing npu_probe.
No changes to existing tasks, device-plugin deployments, or K3s services.
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
        raise RuntimeError(f"kubectl {' '.join(args)}: {result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="npu-task-probe:local")
    parser.add_argument("--second-image", default=None)
    parser.add_argument("--model", default="/models/yolov10s.hef")
    parser.add_argument("--second-model", default=None)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="results/e2e.json")
    parser.add_argument("--keep", action="store_true", help="retain namespace for debugging")
    args = parser.parse_args()
    namespace = "nputask-e2e-" + uuid.uuid4().hex[:8]
    results = {"namespace": namespace, "image": args.image, "model": args.model,
               "secondImage": args.second_image, "secondModel": args.second_model, "checks": []}

    def record(name, **details):
        results["checks"].append({"test": name, **details})
        print(json.dumps(results["checks"][-1]), flush=True)

    def get(kind, name):
        return json.loads(kubectl("get", kind, name, "-n", namespace, "-o", "json").stdout)

    def task(name, count, frames="0", model=None, image=None):
        obj = {"apiVersion": "npu.snu-rtos.io/v1alpha1", "kind": "NPUTask",
               "metadata": {"name": name, "namespace": namespace}, "spec": {
                   "image": image or args.image, "command": ["/app/npu_probe"],
                   "args": ["$(MODEL_PATH)", frames], "model": {"path": model or args.model},
                   "npuCount": count, "mode": "Batch"}}
        kubectl("apply", "-f", "-", obj=obj)

    def patch(name, spec):
        kubectl("patch", "nputask", name, "-n", namespace, "--type=merge", "-p", json.dumps({"spec": spec}))

    def wait(name, phase):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            obj = get("nputask", name)
            status = obj.get("status", {})
            if status.get("phase") == phase and status.get("observedGeneration") == obj["metadata"]["generation"]:
                return obj
            if status.get("phase") == "Failed" and phase != "Failed":
                raise RuntimeError(f"{name}: {status}")
            time.sleep(.5)
        raise TimeoutError(f"{name} did not reach {phase}: {obj}")

    def inference(name, count):
        obj = wait(name, "Running")
        pod = obj["status"]["podName"]
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            logs = kubectl("logs", pod, "-n", namespace).stdout
            ids = {line.split("device=", 1)[1].split()[0] for line in logs.splitlines() if line.startswith("INFERENCE_OK device=")}
            if len(ids) == count:
                return ids
            state = get("pod", pod)
            if state.get("status", {}).get("phase") == "Failed":
                raise RuntimeError(logs)
            time.sleep(.5)
        raise TimeoutError(f"{name}: missing real inference on {count} devices: {logs}")

    kubectl("create", "namespace", namespace)
    try:
        results["nodes"] = json.loads(kubectl("get", "nodes", "-o", "json").stdout)
        task("a", 3)
        a = inference("a", 3)
        task("b", 5, model=args.second_model, image=args.second_image)
        b = inference("b", 5)
        assert a.isdisjoint(b), (a, b)
        task("c", 2)
        wait("c", "Pending")
        time.sleep(3)
        c = get("nputask", "c")
        assert c["status"]["phase"] == "Pending", c
        record("exclusive-3-plus-5", a=sorted(a), b=sorted(b))
        patch("a", {"suspend": True})
        wait("a", "Suspended")
        released = time.monotonic()
        c_ids = inference("c", 2)
        assert c_ids.isdisjoint(b)
        # Confirm the B pod is unchanged and still producing successful inference.
        b_pod = get("nputask", "b")["status"]["podName"]
        before = kubectl("logs", b_pod, "-n", namespace).stdout
        time.sleep(2)
        after = kubectl("logs", b_pod, "-n", namespace).stdout
        assert len(after) > len(before), "B stopped inference during A release"
        record("release-unblocks-pending", seconds=time.monotonic()-released, c=sorted(c_ids))
        for name in ("b", "c"):
            patch(name, {"suspend": True}); wait(name, "Suspended")
        patch("a", {"suspend": False, "npuCount": 4})
        inference("a", 4)
        for count in (2, 4):
            old = get("pod", get("nputask", "a")["status"]["podName"])["metadata"]["uid"]
            patch("a", {"npuCount": count})
            inference("a", count)
            new = get("pod", get("nputask", "a")["status"]["podName"])["metadata"]["uid"]
            assert old != new, "resize did not replace pod"
        patch("a", {"suspend": True}); wait("a", "Suspended")
        record("resize-4-2-4")
        task("invalid", 1, "1", model="/models/missing.hef")
        wait("invalid", "Failed")
        record("invalid-hef-releases")
        for cycle in range(args.cycles):
            count = (1, 2, 4, 8)[cycle % 4]
            name = f"batch-{cycle}"
            start = time.monotonic()
            task(name, count, "2")
            finished = wait(name, "Succeeded")
            logs = kubectl("logs", finished["status"]["podName"], "-n", namespace).stdout
            assert logs.count("INFERENCE_OK device=") >= count, logs
            record("batch-cycle", cycle=cycle, npus=count, seconds=time.monotonic()-start)
            kubectl("delete", "nputask", name, "-n", namespace, "--wait=true")
        # A container without an NPU request must fail real device acquisition.
        kubectl("apply", "-f", "-", obj={"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "no-allocation", "namespace": namespace},
            "spec": {"restartPolicy": "Never", "containers": [{"name": "probe", "image": args.image, "imagePullPolicy": "IfNotPresent",
                "command": ["/app/npu_probe"], "args": [args.model, "1"]}]}})
        deadline = time.monotonic()+args.timeout
        while time.monotonic() < deadline:
            p = get("pod", "no-allocation")
            if p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
                break
            time.sleep(.5)
        assert p["status"]["phase"] == "Failed", p
        record("no-allocation-denied")
        results["passed"] = True
    except Exception as exc:
        results["passed"] = False
        results["error"] = str(exc)
        results["pods"] = kubectl("get", "pods", "-n", namespace, "-o", "json", check=False).stdout
        results["logs"] = {}
        for pod in json.loads(results["pods"]).get("items", []):
            name = pod["metadata"]["name"]
            results["logs"][name] = kubectl("logs", name, "-n", namespace, check=False).stdout
        results["events"] = kubectl("get", "events", "-n", namespace, "-o", "json", check=False).stdout
        raise
    finally:
        output = pathlib.Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2))
        if not args.keep:
            kubectl("delete", "namespace", namespace, "--wait=false")


if __name__ == "__main__":
    main()
