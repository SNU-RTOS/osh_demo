#!/usr/bin/env python3
"""Repeatable single-NPU WRR fairness measurement for real Hailo models."""

import argparse
import datetime
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAMESPACE = "npu-fairness"
PROFILES = {
    "same-model-20-80": (("yolov10s", "/opt/models/yolov10s.hef", 200),
                          ("yolov10s", "/opt/models/yolov10s.hef", 800)),
    "mixed-model-30-70": (("yolov10s", "/opt/models/yolov10s.hef", 300),
                           ("tiny-yolov4", "/opt/models/tiny_yolov4.hef", 700)),
}


def run(*args, check=True, input_text=None):
    result = subprocess.run(args, input=input_text, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def kubectl(*args, **kwargs):
    return run("kubectl", *args, **kwargs)


def task(name):
    return json.loads(kubectl("get", "nputask", name, "-n", NAMESPACE, "-o", "json"))


def wait_running(names, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = [task(name) for name in names]
        if any(item.get("status", {}).get("phase") == "Failed" for item in current):
            raise RuntimeError(f"fairness workload failed: {current}")
        if all(item.get("status", {}).get("phase") == "Running" for item in current):
            return current
        time.sleep(1)
    raise RuntimeError("timeout waiting for fairness workloads")


def dispatcher_metrics(npu_id):
    index = npu_id.rsplit("-", 1)[-1]
    return kubectl("exec", f"deployment/npu-share-dispatcher-{index}",
                   "-n", "npu-task-system", "-c", "share-scheduler", "--",
                   "curl", "-fsS", "http://127.0.0.1:9790/metrics")


def samples(text, metric):
    prefix = metric + "{workload=\""
    values = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        workload, raw = line[len(prefix):].split('\"} ', 1)
        values[workload] = float(raw)
    return values


def manifest(profile, trial):
    documents = []
    for suffix, (model, model_path, share) in zip(("a", "b"), profile):
        name = f"fair-{trial}-{suffix}"
        documents.append(f"""apiVersion: npu.snu-rtos.io/v1alpha1
kind: NPUTask
metadata:
  name: {name}
  namespace: {NAMESPACE}
spec:
  image: npu-share-client:local
  command: [/usr/local/bin/shared_runner]
  args: [--hef, $(MODEL_PATH), --runs, \"1000000000\"]
  model: {{path: {model_path}, key: {model}}}
  npuShare: {share}
  mode: Service
""")
    return "---\n".join(documents)


def wait_registered(npu_id, workload_ids, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        metrics = dispatcher_metrics(npu_id)
        allocated = samples(metrics, "npu_share_allocated")
        if all(uid in allocated for uid in workload_ids):
            return metrics
        time.sleep(1)
    raise RuntimeError("workloads did not register with the broker")


def execute_trial(profile_name, profile, trial, warmup, duration):
    names = [f"fair-{trial}-a", f"fair-{trial}-b"]
    kubectl("apply", "-f", "-", input_text=manifest(profile, trial))
    current = wait_running(names)
    npu_ids = {item["status"].get("npuID") for item in current}
    if len(npu_ids) != 1 or "" in npu_ids:
        raise RuntimeError(f"workloads were not colocated: {npu_ids}")
    npu_id = npu_ids.pop()
    uids = [item["metadata"]["uid"] for item in current]
    wait_registered(npu_id, uids)
    time.sleep(warmup)
    before_text = dispatcher_metrics(npu_id)
    started = time.monotonic()
    time.sleep(duration)
    elapsed = time.monotonic() - started
    after_text = dispatcher_metrics(npu_id)
    before_grants = samples(before_text, "npu_share_grants_total")
    after_grants = samples(after_text, "npu_share_grants_total")
    before_wait = samples(before_text, "npu_share_grant_wait_seconds_sum")
    after_wait = samples(after_text, "npu_share_grant_wait_seconds_sum")
    grants = [int(after_grants.get(uid, 0) - before_grants.get(uid, 0)) for uid in uids]
    waits = [after_wait.get(uid, 0) - before_wait.get(uid, 0) for uid in uids]
    if min(grants) <= 0:
        raise RuntimeError(f"no grant progress during trial: {grants}")
    shares = [item[2] for item in profile]
    total_grants = sum(grants)
    observed = [value / total_grants for value in grants]
    expected = [value / sum(shares) for value in shares]
    normalized = [grants[i] / shares[i] for i in range(2)]
    jain = sum(normalized) ** 2 / (len(normalized) * sum(value * value for value in normalized))
    result = {
        "profile": profile_name, "trial": trial, "npu_id": npu_id,
        "duration_seconds": elapsed, "workloads": [], "normalized_jain_index": jain,
        "maximum_share_fraction_error": max(abs(observed[i] - expected[i]) for i in range(2)),
    }
    for index, item in enumerate(current):
        model, _, share = profile[index]
        result["workloads"].append({
            "name": names[index], "uid": uids[index], "model": model, "share": share,
            "grants": grants[index], "grants_per_second": grants[index] / elapsed,
            "observed_grant_fraction": observed[index], "expected_share_fraction": expected[index],
            "average_grant_wait_seconds": waits[index] / grants[index],
        })
    kubectl("delete", "nputask", *names, "-n", NAMESPACE, "--wait=true")
    deadline = time.time() + 60
    while time.time() < deadline:
        active = sum(samples(dispatcher_metrics(npu_id), "npu_share_allocated").values())
        if active == 0:
            return result
        time.sleep(1)
    raise RuntimeError("broker allocation did not drain after trial")


def mean_ci95(values, lower=None, upper=None):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_low": None, "ci95_high": None}
    # Two-sided 95% Student-t critical values; 1.96 is the asymptotic fallback.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(len(values)-1, 1.96)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    low, high = mean - margin, mean + margin
    if lower is not None:
        low = max(lower, low)
    if upper is not None:
        high = min(upper, high)
    return {"mean": mean, "ci95_low": low, "ci95_high": high}


def summarize(trials):
    grouped = {}
    for trial in trials:
        grouped.setdefault(trial["profile"], []).append(trial)
    result = {}
    for profile, values in grouped.items():
        result[profile] = {
            "trials": len(values),
            "normalized_jain_index": mean_ci95([item["normalized_jain_index"] for item in values], 0, 1),
            "maximum_share_fraction_error": mean_ci95([item["maximum_share_fraction_error"] for item in values], 0, 1),
            "workloads": [],
        }
        for index in range(2):
            result[profile]["workloads"].append({
                "model": values[0]["workloads"][index]["model"],
                "share": values[0]["workloads"][index]["share"],
                "grants_per_second": mean_ci95([item["workloads"][index]["grants_per_second"] for item in values]),
                "observed_grant_fraction": mean_ci95([item["workloads"][index]["observed_grant_fraction"] for item in values], 0, 1),
                "average_grant_wait_seconds": mean_ci95([item["workloads"][index]["average_grant_wait_seconds"] for item in values]),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--max-fraction-error", type=float, default=0.03)
    parser.add_argument("--min-jain", type=float, default=0.99)
    parser.add_argument("--output", default=str(ROOT / "npu-task/results/fairness.json"))
    args = parser.parse_args()
    if args.trials < 2 or args.warmup < 0 or args.duration <= 0:
        parser.error("trials must be >=2, warmup >=0, and duration >0")
    existing = json.loads(kubectl("get", "nputasks", "-A", "-o", "json")).get("items", [])
    active = [f"{item['metadata']['namespace']}/{item['metadata']['name']}" for item in existing
              if item.get("spec", {}).get("npuShare") and item.get("status", {}).get("phase") in ("Pending", "Running", "Stopping")]
    if active:
        raise RuntimeError("active shared workloads must be stopped: " + ", ".join(active))
    kubectl("delete", "namespace", NAMESPACE, "--ignore-not-found", "--wait=true")
    kubectl("create", "namespace", NAMESPACE)
    trials = []
    passed = False
    try:
        for profile_name, profile in PROFILES.items():
            for trial in range(1, args.trials + 1):
                print(f"running {profile_name} trial {trial}/{args.trials}", flush=True)
                trials.append(execute_trial(profile_name, profile, trial, args.warmup, args.duration))
        summary = summarize(trials)
        failures = []
        for profile, values in summary.items():
            if values["maximum_share_fraction_error"]["ci95_high"] > args.max_fraction_error:
                failures.append(f"{profile}: share fraction error exceeds {args.max_fraction_error}")
            if values["normalized_jain_index"]["ci95_low"] < args.min_jain:
                failures.append(f"{profile}: normalized Jain index below {args.min_jain}")
        passed = not failures
        result = {"schema_version": 1, "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "passed": passed, "parameters": vars(args),
                  "profiles": PROFILES, "trials": trials, "summary": summary, "failures": failures}
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if failures:
            raise RuntimeError("; ".join(failures))
        return 0
    finally:
        if passed:
            kubectl("delete", "namespace", NAMESPACE, "--wait=true")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        sys.exit(1)
