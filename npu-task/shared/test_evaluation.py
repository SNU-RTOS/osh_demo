import importlib.util
import pathlib
import unittest


def module(name):
    path = pathlib.Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fairness = module("fairness_benchmark")
validation = module("validate_results")
evaluation = module("evaluate")


class EvaluationTest(unittest.TestCase):
    def test_exclusive_summary_averages_two_npu_waves(self):
        items = [{"name": name, "throughput_fps": value, "average_latency_us": 1,
                  "p95_latency_us": 2} for name, value in zip("abcd", (10, 20, 30, 40))]
        result = evaluation.exclusive_summary(items)
        self.assertEqual(50, result["throughput_fps"])
        self.assertEqual([["a", "b"], ["c", "d"]], result["execution_waves"])

    def test_confidence_interval_and_summary(self):
        trials = []
        for index, fraction in enumerate((0.2, 0.21, 0.19, 0.2, 0.2), 1):
            trials.append({"profile": "same-model-20-80", "trial": index,
                           "normalized_jain_index": 0.999,
                           "maximum_share_fraction_error": abs(fraction - 0.2),
                           "workloads": [
                               {"model": "yolo", "share": 200, "grants_per_second": 20,
                                "observed_grant_fraction": fraction, "average_grant_wait_seconds": .01},
                               {"model": "yolo", "share": 800, "grants_per_second": 80,
                                "observed_grant_fraction": 1-fraction, "average_grant_wait_seconds": .01}]})
        result = fairness.summarize(trials)["same-model-20-80"]
        self.assertEqual(5, result["trials"])
        self.assertIsNotNone(result["normalized_jain_index"]["ci95_low"])

    def test_validation_rejects_missing_repetitions(self):
        comparison = {"exclusive": {"tasks": [], "physical_npus": 2},
                      "shared": {"tasks": [], "physical_npus": 2}}
        result = validation.validate(comparison, [], {"passed": False, "summary": {}})
        self.assertFalse(result["passed"])
        self.assertTrue(result["failures"])

    def test_validation_accepts_complete_campaign(self):
        def tasks(shared=False):
            names = ["share-a", "share-b", "share-c", "share-d"] if shared else ["exclusive-a", "exclusive-b", "exclusive-c", "exclusive-d"]
            npus = ["npu-0", "npu-1", "npu-0", "npu-1"]
            return [{"name": name, "npu_id": npus[index] if shared else "", "runs": 10,
                     "throughput_fps": 1, "p95_latency_us": 1} for index, name in enumerate(names)]
        comparison = {"exclusive": {"tasks": tasks(), "physical_npus": 2},
                      "shared": {"tasks": tasks(True), "physical_npus": 2}}
        checks = ["broker-epoch-changed:a->b", "status-epoch-recovered-and-grant-wait-visible",
                  "controller-operational-metrics-visible"]
        recovery = [{"passed": True, "checks": checks,
                     "measurements": {"dispatcher_recovery_seconds": 10}} for _ in range(3)]
        summary = {name: {"trials": 5, "normalized_jain_index": {"ci95_low": .99},
                          "maximum_share_fraction_error": {"ci95_high": .01}}
                   for name in ("same-model-20-80", "mixed-model-30-70")}
        environment = {"git": {"exit_code": 0}, "hailo_scan": {"exit_code": 0},
                       "models": [{}, {}]}
        result = validation.validate(comparison, recovery, {"passed": True, "summary": summary}, environment)
        self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()
