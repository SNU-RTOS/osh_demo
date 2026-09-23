#!/usr/bin/env python3
"""Mixed long-running/burst workload and dispatcher recovery scenario."""

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "npu-task/shared/realistic-workloads.yaml"
MONITOR = ROOT / "npu-task/tools/npu_monitor.py"
NAMESPACE = "npu-realistic"


def run(*args, check=True):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def kubectl(*args):
    return run("kubectl", *args)


def task(name):
    return json.loads(kubectl("get", "nputask", name, "-n", NAMESPACE, "-o", "json"))


def wait_task(name, phases, timeout=180, reason=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = task(name)
        phase = current.get("status", {}).get("phase", "")
        conditions = current.get("status", {}).get("conditions", [])
        ready = next((value for value in conditions if value.get("type") == "Ready"), None)
        actual_reason = ready.get("reason", "") if ready else ""
        if phase in phases and (reason is None or actual_reason == reason):
            return current
        if phase == "Failed" and "Failed" not in phases:
            raise RuntimeError(f"{name} failed: {actual_reason} {conditions}")
        time.sleep(1)
    raise RuntimeError(f"timeout waiting for {name}: phases={phases}, reason={reason}")


def apply_one(name):
    # Applying each object independently makes the intended arrival order explicit.
    documents = MANIFEST.read_text().split("\n---\n")
    document = next(value for value in documents if f"name: {name}" in value)
    result = subprocess.run(["kubectl", "apply", "-f", "-"], input=document,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"apply {name}: {result.stderr.strip()}")


def snapshot(label, result):
    raw = run("python3", str(MONITOR), "-n", NAMESPACE, "--json")
    data = json.loads(raw)
    result["stages"].append({"label": label, "snapshot": data})
    print(f"\n===== {label} =====", flush=True)
    print(run("python3", str(MONITOR), "-n", NAMESPACE), end="")
    return data


def dispatcher_metrics(npu_id):
    deployment = f"npu-share-dispatcher-{npu_id.rsplit('-', 1)[-1]}"
    text = kubectl("exec", f"deployment/{deployment}", "-n", "npu-task-system",
                   "-c", "share-scheduler", "--", "curl", "-fsS",
                   "http://127.0.0.1:9790/metrics")
    return deployment, text


def controller_metrics():
    pods = json.loads(kubectl("get", "pods", "-n", "npu-task-system",
                             "-l", "app=npu-task-controller", "-o", "json"))["items"]
    ready = next(pod for pod in pods if pod.get("status", {}).get("phase") == "Running")
    name = ready["metadata"]["name"]
    return kubectl("get", "--raw", f"/api/v1/namespaces/npu-task-system/pods/{name}:8080/proxy/metrics")


def grants(metrics, workload_id):
    prefix = f'npu_share_grants_total{{workload="{workload_id}"}} '
    for line in metrics.splitlines():
        if line.startswith(prefix):
            return int(float(line[len(prefix):]))
    return 0


def metric(metrics, name, workload_id):
    prefix = f'{name}{{workload="{workload_id}"}} '
    for line in metrics.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):])
    return None


def restart_count(pod_name):
    pod = json.loads(kubectl("get", "pod", pod_name, "-n", NAMESPACE, "-o", "json"))
    statuses = pod.get("status", {}).get("containerStatuses", [])
    return statuses[0].get("restartCount", 0) if statuses else 0


def prepare_clean_dispatchers(result):
    all_tasks = json.loads(kubectl("get", "nputasks", "-A", "-o", "json")).get("items", [])
    blockers = []
    for current in all_tasks:
        spec, status = current.get("spec", {}), current.get("status", {})
        if spec.get("npuShare", 0) and status.get("phase") in ("Pending", "Running", "Stopping"):
            blockers.append(f"{current['metadata']['namespace']}/{current['metadata']['name']}:{status.get('phase')}")
    if blockers:
        raise RuntimeError("active shared workloads outside scenario: " + ", ".join(blockers))
    for index in (0, 1):
        deployment = f"npu-share-dispatcher-{index}"
        kubectl("rollout", "restart", f"deployment/{deployment}", "-n", "npu-task-system")
        kubectl("rollout", "status", f"deployment/{deployment}", "-n", "npu-task-system", "--timeout=180s")
    result["checks"].append("clean-dispatcher-epochs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "npu-task/results/realistic-scenario.json"))
    parser.add_argument("--keep", action="store_true", help="retain namespace after success")
    args = parser.parse_args()
    result = {"schema_version": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "passed": False, "stages": [], "checks": [], "measurements": {}}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kubectl("delete", "namespace", NAMESPACE, "--ignore-not-found", "--wait=true")
    prepare_clean_dispatchers(result)
    kubectl("create", "namespace", NAMESPACE)
    try:
        expected = [
            ("camera-service", "npu-0"),
            ("classification-service", "npu-0"),
            ("depth-service", "npu-1"),
            ("batch-burst", "npu-1"),
        ]
        for name, npu_id in expected:
            apply_one(name)
            current = wait_task(name, {"Running"}, timeout=180)
            actual = current["status"].get("npuID")
            if actual != npu_id:
                raise RuntimeError(f"{name}: expected {npu_id}, got {actual}")
            result["checks"].append(f"arrival-placement:{name}:{actual}")

        pending_started = time.monotonic()
        apply_one("urgent-detection")
        wait_task("urgent-detection", {"Pending"}, timeout=60, reason="InsufficientNPUShare")
        result["measurements"]["pending_detection_seconds"] = time.monotonic() - pending_started
        before = snapshot("capacity-full-and-urgent-pending", result)
        if len(before["pending"]) != 1 or before["pending"][0]["name"] != "urgent-detection":
            raise RuntimeError("monitor did not expose the urgent pending request")
        result["checks"].append("pending-visible-with-reason")
        epoch_before = next(npu["broker_epoch"] for npu in before["npus"] if npu["id"] == "npu-0")
        if epoch_before == "-":
            raise RuntimeError("npu-0 broker epoch is not observable")
        active_ages = [work["session_age_seconds"] for work in before["workloads"] if work["runtime_active"]]
        if not active_ages or any(age is None or age > 15 for age in active_ages):
            raise RuntimeError(f"heartbeat session ages are stale: {active_ages}")
        result["checks"].append("broker-epoch-and-fresh-heartbeats-visible")
        for name, _ in expected:
            current = task(name)
            if not current["status"].get("allocationTime") or not current["status"].get("brokerEpoch"):
                raise RuntimeError(f"{name}: allocation time or broker epoch missing from status")
        result["checks"].append("allocation-time-and-broker-epoch-persisted")
        control_metrics = controller_metrics()
        for name in ("nputask_pending_duration_seconds", "nputask_allocation_age_seconds",
                     "nputask_container_restart_count", "nputask_share", "nputask_phase_info"):
            if name not in control_metrics:
                raise RuntimeError(f"controller metric {name} is missing")
        if 'task="urgent-detection"' not in control_metrics or 'reason="InsufficientNPUShare"' not in control_metrics:
            raise RuntimeError("pending task labels are missing from controller metrics")
        result["checks"].append("controller-operational-metrics-visible")

        admission_started = time.monotonic()
        kubectl("patch", "nputask", "classification-service", "-n", NAMESPACE,
                "--type=merge", "-p", '{"spec":{"suspend":true}}')
        wait_task("classification-service", {"Suspended"}, timeout=120)
        urgent = wait_task("urgent-detection", {"Running", "Succeeded"}, timeout=180)
        if urgent["status"].get("npuID") != "npu-0":
            raise RuntimeError("urgent request did not use the released npu-0 capacity")
        result["checks"].append("release-admits-urgent-on-npu-0")
        result["measurements"]["pending_admission_after_release_seconds"] = time.monotonic() - admission_started
        snapshot("released-capacity-and-urgent-admitted", result)

        wait_task("urgent-detection", {"Succeeded"}, timeout=180)
        result["checks"].append("urgent-batch-completed-before-fault")

        camera = task("camera-service")
        camera_uid, camera_pod = camera["metadata"]["uid"], camera["status"]["podName"]
        allocation_time = camera["status"].get("allocationTime")
        before_restarts = restart_count(camera_pod)
        deployment, _ = dispatcher_metrics("npu-0")
        recovery_started = time.monotonic()
        kubectl("rollout", "restart", f"deployment/{deployment}", "-n", "npu-task-system")
        kubectl("rollout", "status", f"deployment/{deployment}", "-n", "npu-task-system", "--timeout=180s")

        deadline = time.time() + 180
        recovered_grants = 0
        while time.time() < deadline:
            current = task("camera-service")
            if current.get("status", {}).get("phase") == "Running":
                _, metrics = dispatcher_metrics("npu-0")
                recovered_grants = grants(metrics, camera_uid)
                epoch_status = current.get("status", {}).get("brokerEpoch")
                if restart_count(camera_pod) > before_restarts and recovered_grants >= 20 and epoch_status != epoch_before:
                    break
            time.sleep(2)
        else:
            raise RuntimeError("camera service did not restart, re-register, and resume grants")
        result["checks"].append(f"dispatcher-recovery:restart-count>{before_restarts}:grants={recovered_grants}")
        result["measurements"]["dispatcher_recovery_seconds"] = time.monotonic() - recovery_started
        result["measurements"]["recovered_grants"] = recovered_grants
        final = snapshot("dispatcher-restarted-and-service-recovered", result)
        epoch_after = next(npu["broker_epoch"] for npu in final["npus"] if npu["id"] == "npu-0")
        if epoch_after == epoch_before:
            raise RuntimeError("dispatcher restart did not change broker epoch")
        camera_state = next(work for work in final["workloads"] if work["name"] == "camera-service")
        if camera_state["session_age_seconds"] is None or camera_state["session_age_seconds"] > 15:
            raise RuntimeError(f"recovered camera heartbeat is stale: {camera_state['session_age_seconds']}")
        result["checks"].append(f"broker-epoch-changed:{epoch_before}->{epoch_after}")
        camera = task("camera-service")
        if camera["status"].get("brokerEpoch") != epoch_after:
            raise RuntimeError("controller status does not contain the current broker epoch")
        if camera["status"].get("allocationTime") != allocation_time:
            raise RuntimeError("allocation time changed during dispatcher recovery")
        _, metrics = dispatcher_metrics("npu-0")
        if metric(metrics, "npu_share_grant_wait_seconds_count", camera_uid) is None:
            raise RuntimeError("grant wait metrics are missing")
        result["checks"].append("status-epoch-recovered-and-grant-wait-visible")
        failed = [work["name"] for work in final["workloads"] if work["phase"] == "Failed"]
        if failed:
            raise RuntimeError(f"unexpected failed workloads after recovery: {failed}")
        result["checks"].append("no-unexpected-failed-workloads")
        result["passed"] = True
        print("Realistic multi-NPU scenario passed")
        return 0
    except Exception as error:
        result["error"] = str(error)
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"result: {output}")
        if result["passed"] and not args.keep:
            kubectl("delete", "namespace", NAMESPACE, "--wait=true")


if __name__ == "__main__":
    sys.exit(main())
