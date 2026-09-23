#!/usr/bin/env python3
"""Validate completed measurement artifacts against the MVP gates."""

import argparse
import glob
import json
import pathlib
import sys


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def validate(comparison, recovery_runs, fairness, environment=None):
    checks = []
    failures = []

    def check(name, condition, detail):
        (checks if condition else failures).append({"name": name, "detail": detail})

    exclusive = comparison.get("exclusive", {})
    shared = comparison.get("shared", {})
    check("exclusive-four-workloads", len(exclusive.get("tasks", [])) == 4,
          f"tasks={len(exclusive.get('tasks', []))}")
    check("shared-four-workloads", len(shared.get("tasks", [])) == 4,
          f"tasks={len(shared.get('tasks', []))}")
    check("two-npu-baseline", exclusive.get("physical_npus") == 2 and shared.get("physical_npus") == 2,
          f"exclusive={exclusive.get('physical_npus')} shared={shared.get('physical_npus')}")
    check("all-inference-completed", all(item.get("runs", 0) > 0 for item in exclusive.get("tasks", []) + shared.get("tasks", [])),
          "every task has a parsed NPU_E2E_OK result")
    check("latency-and-throughput-positive", all(item.get("throughput_fps", 0) > 0 and item.get("p95_latency_us", 0) > 0
                                                    for item in exclusive.get("tasks", []) + shared.get("tasks", [])),
          "throughput and P95 latency must be positive")
    expected = {"share-a": "npu-0", "share-b": "npu-1", "share-c": "npu-0", "share-d": "npu-1"}
    actual = {item.get("name"): item.get("npu_id") for item in shared.get("tasks", [])}
    check("first-fit-placement", actual == expected, f"actual={actual}")

    check("recovery-repetitions", len(recovery_runs) >= 3, f"runs={len(recovery_runs)}")
    for index, result in enumerate(recovery_runs, 1):
        names = set(result.get("checks", []))
        check(f"recovery-{index}-passed", result.get("passed") is True, f"checks={len(names)}")
        check(f"recovery-{index}-epoch", any(value.startswith("broker-epoch-changed:") for value in names), "dispatcher epoch changed")
        check(f"recovery-{index}-status", "status-epoch-recovered-and-grant-wait-visible" in names, "status and runtime recovered")
        check(f"recovery-{index}-metrics", "controller-operational-metrics-visible" in names, "controller metrics visible")
        recovery_seconds = result.get("measurements", {}).get("dispatcher_recovery_seconds")
        check(f"recovery-{index}-bounded", recovery_seconds is not None and recovery_seconds <= 180,
              f"dispatcher_recovery_seconds={recovery_seconds}")

    summaries = fairness.get("summary", {})
    check("fairness-passed", fairness.get("passed") is True, f"failures={fairness.get('failures', [])}")
    check("two-fairness-profiles", {"same-model-20-80", "mixed-model-30-70"}.issubset(summaries), f"profiles={list(summaries)}")
    for profile, values in summaries.items():
        check(f"{profile}-repetitions", values.get("trials", 0) >= 5, f"trials={values.get('trials')}")
        check(f"{profile}-ci", values.get("normalized_jain_index", {}).get("ci95_low") is not None and
              values.get("maximum_share_fraction_error", {}).get("ci95_high") is not None, "95% CI bounds present")

    if environment is not None:
        check("environment-git", environment.get("git", {}).get("exit_code") == 0,
              environment.get("git", {}).get("output", ""))
        check("environment-hailo", environment.get("hailo_scan", {}).get("exit_code") == 0,
              environment.get("hailo_scan", {}).get("output", ""))
        check("environment-model-checksums", len(environment.get("models", [])) >= 2,
              f"models={len(environment.get('models', []))}")

    return {"schema_version": 1, "passed": not failures, "checks": checks, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="npu-task/results/multi-npu-share.json")
    parser.add_argument("--recovery-glob", default="npu-task/results/recovery-*.json")
    parser.add_argument("--fairness", default="npu-task/results/fairness.json")
    parser.add_argument("--environment", default="npu-task/results/environment.json")
    parser.add_argument("--output", default="npu-task/results/validation.json")
    args = parser.parse_args()
    recovery_paths = sorted(glob.glob(args.recovery_glob))
    report = validate(load(args.comparison), [load(path) for path in recovery_paths], load(args.fairness), load(args.environment))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
