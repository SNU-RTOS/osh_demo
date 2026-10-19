import importlib.util
import pathlib
import unittest
from unittest.mock import patch

path = pathlib.Path(__file__).with_name("npu_monitor.py")
spec = importlib.util.spec_from_file_location("npu_monitor", path)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTest(unittest.TestCase):
    def test_best_effort_reservations_and_weights(self):
        task = {"metadata": {"namespace": "apps", "name": "client", "uid": "uid"},
                "spec": {"bestEffort": {"targetFPS": 20}, "model": {"key": "model"}},
                "status": {"npuID": "npu-0", "phase": "Running", "reservedDemand": 240,
                           "executionWeight": 100, "admissionPolicy": "utilization-best-effort"}}
        dispatcher = {"metadata": {"namespace": "system", "name": "dispatcher",
                      "labels": {"npu.snu-rtos.io/component": "dispatcher"},
                      "annotations": {"npu.snu-rtos.io/dispatcher-id": "npu-0"}}}
        # A terminating reservation in a different namespace must still be counted.
        pod = {"metadata": {"namespace": "other", "name": "terminating", "deletionTimestamp": "now",
               "annotations": {"npu.snu-rtos.io/assigned-npu": "npu-0", "npu.snu-rtos.io/allocated-share": "240"}},
               "status": {"phase": "Succeeded"}}
        runtime = monitor.prometheus('npu_broker_occupancy_units 50\n')
        with patch.object(monitor, 'kubectl_json', side_effect=[{"items": [task]}, {"items": [dispatcher, pod]}]), \
             patch.object(monitor, 'get_podresources', return_value={}), \
             patch.object(monitor, 'fetch_metrics', return_value=[]), \
             patch.object(monitor, 'dispatcher_metrics', return_value=runtime):
            result = monitor.snapshot('apps', None)
        self.assertEqual(240, result['npus'][0]['allocated'])
        self.assertEqual(240, result['npus'][0]['admission_load_units'])
        self.assertEqual('20 FPS', result['workloads'][0]['requested'])
        self.assertEqual(100, result['workloads'][0]['execution_weight'])

    def test_prometheus(self):
        samples = monitor.prometheus('npu_share_available 700\nnpu_share_allocated{workload="abc"} 300\n')
        self.assertEqual(700, monitor.metric(samples, "npu_share_available"))
        self.assertEqual(300, monitor.metric(samples, "npu_share_allocated", "workload", "abc"))

    def test_pod_devices(self):
        data = {"pod_resources": [{"namespace": "system", "name": "dispatcher", "containers": [
            {"devices": [{"resource_name": "hailo.ai/npu", "device_ids": ["0000:06:00.0"]}]}
        ]}]}
        self.assertEqual({("system", "dispatcher"): ["0000:06:00.0"]}, monitor.pod_devices(data))


if __name__ == "__main__":
    unittest.main()
