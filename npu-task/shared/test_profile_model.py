import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('profile_model', Path(__file__).with_name('profile_model.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileTest(unittest.TestCase):
    def test_calibration_contract_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            hef = Path(temp) / 'test.hef'
            hef.write_bytes(b'test model')

            def runner(command, env, check):
                self.assertTrue(check)
                self.assertEqual(['--warmup', '50', '--runs', '500'], command[3:7])
                self.assertNotIn('NPU_TARGET_FPS', env)
                self.assertNotIn('HAILORT_SERVICE_ADDRESS', env)
                Path(command[-1]).write_text(json.dumps({
                    'samples': 500, 'warmup': 50, 'meanSeconds': .009, 'p95Seconds': .01}))

            with patch.dict(os.environ, {'NPU_TARGET_FPS': '20', 'HAILORT_SERVICE_ADDRESS': 'old'}), patch.object(module.subprocess, 'run', side_effect=runner):
                result = module.profile(Path('/runner'), hef, 'test', 'HAILO8', '4.21.0')
            profile = result['profiles']['test']
            self.assertEqual(64, len(profile['modelSHA256']))
            self.assertEqual('utilization-best-effort', result['sharedAdmissionPolicy'])
            self.assertEqual('HAILO8', profile['deviceArchitecture'])

    def test_incomplete_measurement_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            hef = Path(temp) / 'test.hef'
            hef.write_bytes(b'test')
            def runner(command, **unused):
                Path(command[-1]).write_text('{"samples":3,"warmup":50}')
            with patch.object(module.subprocess, 'run', side_effect=runner):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    module.profile(Path('/runner'), hef, 'test', 'HAILO8', '4.21.0')


if __name__ == '__main__':
    unittest.main()
