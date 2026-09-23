#!/usr/bin/env python3
"""One-screen allocation and runtime view for NPUTask workloads."""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


def command(*args, check=True):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def kubectl_json(*args):
    return json.loads(command("kubectl", *args, "-o", "json"))


def prometheus(text):
    samples = []
    pattern = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$')
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"])*)"', match.group(2) or ""))
        labels = {key: value.replace(r'\"', '"').replace(r'\\', '\\') for key, value in labels.items()}
        samples.append((match.group(1), labels, float(match.group(3))))
    return samples


def metric(samples, name, label=None, value=None, default=0):
    for sample_name, labels, sample_value in samples:
        if sample_name == name and (label is None or labels.get(label) == value):
            return sample_value
    return default


def pod_devices(data):
    result = {}
    for pod in data.get("pod_resources", data.get("podResources", [])):
        ids = []
        for container in pod.get("containers", []):
            for device in container.get("devices", []):
                resource = device.get("resource_name", device.get("resourceName", ""))
                if resource == "hailo.ai/npu":
                    ids.extend(device.get("device_ids", device.get("deviceIds", [])))
        if ids:
            result[(pod.get("namespace", ""), pod.get("name", ""))] = sorted(ids)
    return result


def get_podresources(path):
    candidates = [path] if path else ["npu-task/bin/podresources", "bin/podresources"]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            proc = subprocess.run([candidate], text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
            if proc.returncode == 0:
                return pod_devices(json.loads(proc.stdout))
    return {}


def fetch_metrics(namespace, service):
    try:
        svc = kubectl_json("get", "service", service, "-n", namespace)
        address = svc["spec"]["clusterIP"]
        port = svc["spec"]["ports"][0]["port"]
        with urllib.request.urlopen(f"http://{address}:{port}/metrics", timeout=2) as response:
            return prometheus(response.read().decode())
    except Exception:
        return []


def dispatcher_metrics(namespace, name):
    try:
        text = command("kubectl", "-n", namespace, "exec", name,
                       "-c", "share-scheduler", "--", "curl", "-fsS",
                       "http://127.0.0.1:9790/metrics")
        return prometheus(text)
    except Exception:
        return []


def condition(task):
    conditions = task.get("status", {}).get("conditions", [])
    if not conditions:
        return "-", "-"
    current = next((item for item in conditions if item.get("type") == "Ready"), conditions[-1])
    return current.get("reason", "-"), current.get("message", "-")


def age_seconds(timestamp):
    if not timestamp:
        return None
    try:
        value = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return max(0, (datetime.datetime.now(datetime.timezone.utc) - value).total_seconds())
    except ValueError:
        return None


def snapshot(namespace, podresources_path):
    selector = ["-A"] if namespace == "" else ["-n", namespace]
    tasks = kubectl_json("get", "nputasks", *selector).get("items", [])
    pods = kubectl_json("get", "pods", "-A").get("items", [])
    devices = get_podresources(podresources_path)
    monitor = fetch_metrics("npu-task-system", "hailo-monitor-exporter")
    utilization = {labels.get("device", "?"): value for name, labels, value in monitor
                   if name == "hailo_device_utilization_percent"}
    models = {labels.get("model", "?"): {
        "utilization": value,
        "fps": metric(monitor, "hailo_model_fps", "model", labels.get("model", "?")),
    } for name, labels, value in monitor if name == "hailo_model_utilization_percent"}

    dispatchers = {}
    for pod in pods:
        metadata, status = pod["metadata"], pod.get("status", {})
        if metadata.get("labels", {}).get("npu.snu-rtos.io/component") != "dispatcher":
            continue
        annotations = metadata.get("annotations", {})
        npu_id = annotations.get("npu.snu-rtos.io/dispatcher-id", "?")
        capacity = int(annotations.get("npu.snu-rtos.io/capacity", "1000"))
        physical = devices.get((metadata["namespace"], metadata["name"]), [])
        runtime = dispatcher_metrics(metadata["namespace"], metadata["name"])
        epoch = next((labels.get("epoch", "-") for name, labels, _ in runtime
                      if name == "npu_share_broker_info"), "-")
        dispatchers[npu_id] = {
            "id": npu_id, "pod": metadata["name"], "node": pod.get("spec", {}).get("nodeName", "-"),
            "physical_devices": physical, "capacity": capacity,
            "allocated": int(metric(runtime, "npu_share_allocated_total")),
            "available": int(metric(runtime, "npu_share_available", default=capacity)),
            "active_workloads": int(metric(runtime, "npu_share_active_workloads")),
            "broker_epoch": epoch,
            "utilization": {device: utilization.get(device) for device in physical},
            "runtime_metrics": runtime,
        }

    workloads = []
    pods_by_name = {(pod["metadata"]["namespace"], pod["metadata"]["name"]): pod for pod in pods}
    for task in tasks:
        metadata, spec, status = task["metadata"], task.get("spec", {}), task.get("status", {})
        reason, message = condition(task)
        shared = bool(spec.get("npuShare"))
        npu_id, pod_name = status.get("npuID", "-"), status.get("podName", "-")
        physical = devices.get((metadata["namespace"], pod_name), [])
        if shared and not physical:
            physical = dispatchers.get(npu_id, {}).get("physical_devices", [])
        runtime = dispatchers.get(npu_id, {}).get("runtime_metrics", [])
        workload_id = metadata.get("uid", "")
        grants = int(metric(runtime, "npu_share_grants_total", "workload", workload_id))
        wait_count = metric(runtime, "npu_share_grant_wait_seconds_count", "workload", workload_id)
        wait_sum = metric(runtime, "npu_share_grant_wait_seconds_sum", "workload", workload_id)
        wait_max = metric(runtime, "npu_share_grant_wait_seconds_max", "workload", workload_id)
        session_age = metric(runtime, "npu_share_session_age_seconds", "workload", workload_id, default=-1)
        active = any(name == "npu_share_allocated" and labels.get("workload") == workload_id
                     for name, labels, _ in runtime)
        model_key = spec.get("model", {}).get("key") or os.path.basename(spec.get("model", {}).get("path", "-"))
        model_metric = models.get(model_key.replace("-", "_"), models.get(model_key, {}))
        pod = pods_by_name.get((metadata["namespace"], pod_name), {})
        restart_count = sum(item.get("restartCount", 0) for item in pod.get("status", {}).get("containerStatuses", []))
        ready_condition = next((item for item in status.get("conditions", []) if item.get("type") == "Ready"), {})
        workloads.append({
            "namespace": metadata["namespace"], "name": metadata["name"], "pod": pod_name,
            "mode": "shared" if shared else "exclusive",
            "requested": f"{spec.get('npuShare')} share" if shared else f"{spec.get('npuCount', 0)} NPU",
            "assigned_npu": npu_id, "physical_devices": physical,
            "allocated_share": status.get("allocatedShare", 0), "phase": status.get("phase", "Pending"),
            "runtime_active": active, "grants": grants, "model": model_key,
            "session_age_seconds": session_age if session_age >= 0 else None,
            "grant_wait_average_seconds": wait_sum / wait_count if wait_count else None,
            "grant_wait_max_seconds": wait_max if wait_count else None,
            "restart_count": restart_count,
            "broker_epoch": status.get("brokerEpoch"),
            "allocation_age_seconds": age_seconds(status.get("allocationTime")),
            "pending_duration_seconds": age_seconds(ready_condition.get("lastTransitionTime")) if status.get("phase") == "Pending" else None,
            "model_utilization": model_metric.get("utilization"), "model_fps": model_metric.get("fps"),
            "reason": reason, "message": message,
        })
    for npu in dispatchers.values():
        reserved = sum(work["allocated_share"] for work in workloads
                       if work["assigned_npu"] == npu["id"] and
                       work["phase"] not in ("Succeeded", "Failed", "Suspended"))
        npu["allocated"] = reserved
        npu["available"] = max(0, npu["capacity"] - reserved)
    return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "podresources_available": bool(devices), "npus": list(dispatchers.values()),
            "device_utilization": utilization, "model_metrics": models,
            "workloads": workloads, "pending": [w for w in workloads if w["phase"] == "Pending"]}


def short(value, width):
    value = str(value)
    return value if len(value) <= width else value[:width - 1] + "…"


def render(data):
    print(f"NPU snapshot {data['timestamp']}  physical IDs: " +
          ("available" if data["podresources_available"] else "unavailable (build/run podresources as root)"))
    print("\nNPU POOLS")
    print(f"{'NPU':<8} {'PHYSICAL':<18} {'NODE':<14} {'EPOCH':<10} {'CAP':>5} {'ALLOC':>5} {'FREE':>5} {'ACTIVE':>6} {'UTIL':>8}")
    for npu in data["npus"]:
        physical = ",".join(npu["physical_devices"]) or "-"
        values = [value for value in npu["utilization"].values() if value is not None]
        util = f"{sum(values)/len(values):.1f}%" if values else "-"
        print(f"{npu['id']:<8} {short(physical,18):<18} {short(npu['node'],14):<14} {short(npu['broker_epoch'],10):<10} {npu['capacity']:>5} {npu['allocated']:>5} {npu['available']:>5} {npu['active_workloads']:>6} {util:>8}")
    if data["device_utilization"]:
        print("\nMEASURED PHYSICAL DEVICE UTILIZATION")
        for device, value in sorted(data["device_utilization"].items()):
            print(f"- {device}: {value:.1f}%")
    if data["model_metrics"]:
        print("\nMEASURED MODEL RUNTIME")
        for model, values in sorted(data["model_metrics"].items()):
            print(f"- {model}: utilization={values['utilization']:.1f}% fps={values['fps']:.1f}")
    print("\nWORKLOADS")
    print(f"{'NAMESPACE/NAME':<30} {'POD':<22} {'MODE':<9} {'REQUEST':>10} {'NPU':<8} {'PHYSICAL':<16} {'PHASE':<10} {'USE':<12} {'WAIT':>8} {'RST':>3} {'REASON'}")
    for work in data["workloads"]:
        use = f"active/{work['grants']}" if work["runtime_active"] else (f"done/{work['grants']}" if work["grants"] else "-")
        wait = f"{work['grant_wait_average_seconds']*1000:.1f}ms" if work["grant_wait_average_seconds"] is not None else "-"
        physical = ",".join(work["physical_devices"]) or "-"
        print(f"{short(work['namespace']+'/'+work['name'],30):<30} {short(work['pod'],22):<22} {work['mode']:<9} {work['requested']:>10} {work['assigned_npu']:<8} {short(physical,16):<16} {work['phase']:<10} {use:<12} {wait:>8} {work['restart_count']:>3} {work['reason']}")
    if data["pending"]:
        print("\nPENDING REQUESTS")
        for work in data["pending"]:
            duration = f" for {work['pending_duration_seconds']:.1f}s" if work["pending_duration_seconds"] is not None else ""
            print(f"- {work['namespace']}/{work['name']}: {work['requested']}{duration} — {work['reason']}: {work['message']}")
    print("\nMeasured utilization is observation; requested/allocated share is admission weight, not a throughput guarantee.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--namespace", default="", help="NPUTask namespace; empty means all")
    parser.add_argument("--podresources", help="path to node-local podresources binary")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="refresh continuously")
    args = parser.parse_args()
    while True:
        try:
            data = snapshot(args.namespace, args.podresources)
            if args.watch and not args.json:
                print("\033[2J\033[H", end="")
            print(json.dumps(data, indent=2) if args.json else "", end="\n" if args.json else "")
            if not args.json:
                render(data)
        except (RuntimeError, KeyError, json.JSONDecodeError) as error:
            print(f"npu-monitor: {error}", file=sys.stderr)
            return 1
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
