import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).parents[1] / 'osh.py'
spec = importlib.util.spec_from_file_location('osh', path)
osh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osh)


class FrameworkTest(unittest.TestCase):
    def test_demo_dry_run_is_offline_and_has_no_device_ids(self):
        output = io.StringIO()
        with patch.object(osh.subprocess, 'run', side_effect=AssertionError('external call')), contextlib.redirect_stdout(output):
            osh.main(['demo', '--node', 'node-a', '--dry-run'])
        manifest = json.loads(output.getvalue())
        tasks = manifest['items'][1:]
        self.assertEqual(3, len(tasks))
        for task in tasks:
            self.assertEqual(1, task['spec']['npuCount'])
            self.assertEqual('node-a', task['spec']['visual']['nodeName'])
            self.assertTrue(task['spec']['image'].endswith(':v3'))
            self.assertNotIn('device', json.dumps(task))

    def test_add_task_and_reject_unsupported_batch(self):
        models = osh.catalogue(path.with_name('models.json'))
        task = osh.task('second-detector', models['detection'], 'node-a', 'demo', 'demo')
        self.assertEqual('second-detector', task['metadata']['name'])
        with self.assertRaisesRegex(ValueError, 'batch size one'):
            osh.task('bad', models['detection'], 'node-a', 'demo', 'demo', batch_size=4)
        with self.assertRaisesRegex(ValueError, 'Invalid Kubernetes'):
            osh.task('../bad', models['detection'], 'node-a', 'demo', 'demo')

    def test_apply_uses_existing_nputask_contract(self):
        with patch.object(osh, 'kube', return_value='created\n') as kube, contextlib.redirect_stdout(io.StringIO()):
            osh.main(['deploy', 'extra-pose', '--model', 'pose', '--node', 'node-a', '--no-wait'])
        manifest = kube.call_args.kwargs['data']
        self.assertEqual('List', manifest['kind'])
        self.assertEqual('/app/vision_worker', manifest['items'][1]['spec']['command'][0])

    def test_automatic_node_ambiguity_requires_choice(self):
        with patch.object(osh, 'kube', return_value='{"items":[]}'):
            with self.assertRaisesRegex(ValueError, '--node'):
                osh.choose_node(None, False)

    def test_lifecycle_and_metrics(self):
        with patch.object(osh, 'kube', return_value='patched\n') as kube, contextlib.redirect_stdout(io.StringIO()):
            osh.main(['suspend', 'pose-2'])
        self.assertEqual({'spec': {'suspend': True}}, json.loads(kube.call_args.args[-1]))
        frame = {'event': 'frame', 'task': 'pose-2', 'seq': 10, 'batch_size': 1,
                 'worker_metrics': {'throughput_fps': 9.5}, 'output_metrics': {'throughput_fps': 9.4}}
        output = io.StringIO()
        with patch.object(osh, 'kube', side_effect=[json.dumps({'status': {'podName': 'pod-uid'}}), 'non-json SDK message\n' + json.dumps(frame)]), contextlib.redirect_stdout(output):
            osh.main(['metrics', 'pose-2'])
        self.assertEqual(9.5, json.loads(output.getvalue())['worker_metrics']['throughput_fps'])

    def test_doctor_rejects_missing_local_images(self):
        models = osh.catalogue(path.with_name('models.json'))
        result = type('Result', (), {'returncode': 0, 'stdout': 'docker.io/library/osh-vision-detection:v2\n'})()
        with patch.object(osh.subprocess, 'run', return_value=result), \
             patch.object(osh, 'kube', return_value='{"status":{"readyReplicas":1}}'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Preflight failed'):
                osh.doctor(models)


if __name__ == '__main__':
    unittest.main()
