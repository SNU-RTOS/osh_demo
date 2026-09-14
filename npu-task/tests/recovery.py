#!/usr/bin/env python3
"""Recovery checks; --restart-components also rolls the task controller and plugin.

Does not restart K3s. Run after e2e.py, with sufficient unallocated devices.
"""
import argparse
import json
import pathlib
import time
import uuid
from e2e import kubectl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="npu-task-probe:local")
    parser.add_argument("--restart-components", action="store_true")
    parser.add_argument("--output", default="results/recovery.json")
    args = parser.parse_args()
    ns = "nputask-recovery-" + uuid.uuid4().hex[:8]
    report = {"namespace": ns, "checks": []}

    def get(kind, name):
        return json.loads(kubectl("get", kind, name, "-n", ns, "-o", "json").stdout)

    def poll(predicate):
        until = time.monotonic() + 120
        while time.monotonic() < until:
            result = predicate()
            if result:
                return result
            time.sleep(.5)
        raise TimeoutError("recovery condition not reached")

    def submit(name, command, arguments, mode="Batch"):
        kubectl("apply", "-f", "-", obj={"apiVersion": "npu.snu-rtos.io/v1alpha1", "kind": "NPUTask",
            "metadata": {"name": name, "namespace": ns}, "spec": {"image": args.image,
            "command": command, "args": arguments, "model": {"path": "/models/yolov10s.hef"},
            "npuCount": 2, "mode": mode}})

    def phase(name, desired):
        obj = get("nputask", name)
        return obj if obj.get("status", {}).get("phase") == desired else None

    def infer(name):
        task = poll(lambda: phase(name, "Running"))
        pod = task["status"]["podName"]
        poll(lambda: "INFERENCE_OK" in kubectl("logs", pod, "-n", ns, check=False).stdout)
        return get("pod", pod)

    def record(name):
        report["checks"].append(name); print(name, flush=True)

    kubectl("create", "namespace", ns)
    try:
        submit("service", ["/app/npu_probe"], ["$(MODEL_PATH)", "0"], "Service")
        pod = infer("service")
        if args.restart_components:
            for kind, name, namespace in [("deployment", "npu-task-controller", "npu-task-system"),
                                           ("daemonset", "hailo-device-plugin", "kube-system")]:
                kubectl("rollout", "restart", f"{kind}/{name}", "-n", namespace)
                # Use short polls so each kubectl invocation is bounded.
                poll(lambda: kubectl("rollout", "status", f"{kind}/{name}", "-n", namespace,
                                     "--timeout=5s", check=False).returncode == 0)
                same = infer("service")
                assert same["metadata"]["uid"] == pod["metadata"]["uid"], "running task was duplicated/replaced"
                before = kubectl("logs", pod["metadata"]["name"], "-n", ns).stdout
                time.sleep(2)
                after = kubectl("logs", pod["metadata"]["name"], "-n", ns).stdout
                assert len(after) > len(before), "inference stopped after component restart"
                record(name + "-restart-preserves-inference")
        # Kill a process that holds Hailo handles; timeout exits nonzero afterward.
        submit("crash", ["/usr/bin/timeout"], ["--signal=KILL", "3", "/app/npu_probe", "$(MODEL_PATH)", "0"])
        crashed = poll(lambda: phase("crash", "Failed"))
        logs = kubectl("logs", crashed["status"]["podName"], "-n", ns).stdout
        assert "INFERENCE_OK" in logs, "crash fixture never acquired a device"
        submit("reuse", ["/app/npu_probe"], ["$(MODEL_PATH)", "2"])
        poll(lambda: phase("reuse", "Succeeded"))
        record("killed-inference-releases-devices")
        # The same service pod restarts its container, using Kubernetes backoff.
        submit("restarting", ["/usr/bin/timeout"], ["--signal=KILL", "3", "/app/npu_probe", "$(MODEL_PATH)", "0"], "Service")
        restarting = infer("restarting")
        def restarted():
            current = get("pod", restarting["metadata"]["name"])
            return any(c.get("restartCount", 0) > 0 for c in current.get("status", {}).get("containerStatuses", []))
        poll(restarted)
        record("service-container-restarts")
        # A real API delete must cascade to the owned pod.
        kubectl("delete", "nputask", "restarting", "-n", ns, "--wait=true")
        poll(lambda: kubectl("get", "pod", restarting["metadata"]["name"], "-n", ns,
                             "--ignore-not-found", check=False).stdout == "")
        record("task-deletion-cascades")
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        raise
    finally:
        report["tasks"] = kubectl("get", "nputasks", "-n", ns, "-o", "json", check=False).stdout
        p = pathlib.Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        kubectl("delete", "namespace", ns, "--wait=false")


if __name__ == "__main__":
    main()
