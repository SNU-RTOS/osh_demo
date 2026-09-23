#!/usr/bin/env python3
import argparse
import datetime
import json
import pathlib
import re
import subprocess

RESULT = re.compile(
    r"NPU_E2E_OK mode=(?P<mode>\w+) workload=(?P<workload>\S+) share=(?P<share>\d+) "
    r"runs=(?P<runs>\d+) elapsed_us=(?P<elapsed_us>\d+) throughput_fps=(?P<throughput_fps>[0-9.]+) "
    r"average_latency_us=(?P<average_latency_us>\d+) p95_latency_us=(?P<p95_latency_us>\d+) "
    r"max_latency_us=(?P<max_latency_us>\d+)"
)

def kubectl(*args):
    return subprocess.run(["kubectl", *args], check=True, text=True, capture_output=True).stdout

def task_result(namespace, name):
    task = json.loads(kubectl("-n", namespace, "get", "nputask", name, "-o", "json"))
    pod = task["status"]["podName"]
    log = kubectl("-n", namespace, "logs", pod)
    match = RESULT.search(log)
    if not match:
        raise RuntimeError(f"{name}: result marker missing")
    values = match.groupdict()
    for key in ("share", "runs", "elapsed_us", "average_latency_us", "p95_latency_us", "max_latency_us"):
        values[key] = int(values[key])
    values["throughput_fps"] = float(values["throughput_fps"])
    values["name"] = name
    values["npu_id"] = task.get("status", {}).get("npuID", "")
    return values

def summary(items):
    return {
        "workloads": len(items),
        "throughput_fps": sum(x["throughput_fps"] for x in items),
        "average_latency_us": sum(x["average_latency_us"] for x in items) / len(items),
        "max_p95_latency_us": max(x["p95_latency_us"] for x in items),
    }

def exclusive_summary(items):
    result = summary(items)
    waves = [items[:2], items[2:]]
    result["execution_waves"] = [[item["name"] for item in wave] for wave in waves]
    result["throughput_fps"] = sum(sum(item["throughput_fps"] for item in wave) for wave in waves) / len(waves)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exclusive = [task_result(args.namespace, f"exclusive-{suffix}") for suffix in "abcd"]
    shared = [task_result(args.namespace, f"share-{suffix}") for suffix in "abcd"]
    result = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exclusive": {"tasks": exclusive, "summary": exclusive_summary(exclusive), "physical_npus": 2},
        "shared": {"tasks": shared, "summary": summary(shared), "physical_npus": len({x["npu_id"] for x in shared})},
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
