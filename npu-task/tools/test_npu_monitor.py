import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("npu_monitor.py")
spec = importlib.util.spec_from_file_location("npu_monitor", path)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTest(unittest.TestCase):
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
