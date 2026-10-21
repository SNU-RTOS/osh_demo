#!/usr/bin/env python3
"""OSH vision framework: catalogue-driven deployment, playback and measurements."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
NPU_RESOURCE = 'hailo.ai/npu'


def kube(*args, data=None):
    command = ['kubectl', '--request-timeout=30s', *args]
    result = subprocess.run(command, input=json.dumps(data) if data is not None else None,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def catalogue(path):
    data = json.loads(path.read_text())
    if data.get('version') != 1 or not isinstance(data.get('models'), dict):
        raise ValueError('Expected catalogue version 1 with a models object')
    for name, model in data['models'].items():
        if (model.get('adapter') not in ('detection', 'segmentation', 'pose') or
                not model.get('image') or not str(model.get('path', '')).startswith('/') or
                model.get('npus') != 1 or model.get('batchSize') != 1):
            raise ValueError(f'Invalid model {name}: current visual runners require a supported adapter, image, absolute model path, one NPU and batch size one')
    return data['models']


def safe_name(value, max_length=63):
    if len(value) > max_length or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?', value):
        raise ValueError(f'Invalid Kubernetes name: {value}')
    return value


def choose_node(explicit, dry_run):
    if explicit:
        return explicit
    if dry_run:
        raise ValueError('--dry-run requires --node; automatic node discovery needs cluster access')
    nodes = json.loads(kube('get', 'nodes', '-o', 'json'))['items']
    candidates = [n for n in nodes
                  if int(n.get('status', {}).get('allocatable', {}).get(NPU_RESOURCE, 0)) > 0
                  and not n.get('spec', {}).get('unschedulable', False)
                  and any(c['type'] == 'Ready' and c['status'] == 'True' for c in n['status'].get('conditions', []))]
    if len(candidates) != 1:
        raise ValueError('Use --node HOSTNAME when zero or multiple ready NPU nodes are available')
    return candidates[0]['metadata']['labels']['kubernetes.io/hostname']


def task(name, model, node, namespace, session, npus=1, batch_size=1):
    safe_name(name)
    safe_name(namespace)
    safe_name(session, 40)
    if npus != model['npus'] or batch_size != model['batchSize']:
        raise ValueError('Current visual adapters support one exclusive NPU and batch size one; multi-NPU and batched visual execution are not implemented')
    return {'apiVersion': 'npu.snu-rtos.io/v1alpha1', 'kind': 'NPUTask',
            'metadata': {'name': name, 'namespace': namespace},
            'spec': {'image': model['image'], 'command': ['/app/vision_worker'],
                     'args': ['--kind', model['adapter']], 'model': {'path': model['path']},
                     'npuCount': npus, 'mode': 'Service',
                     'visual': {'session': session, 'nodeName': node},
                     'resources': {'requests': {'cpu': '1', 'memory': '256Mi'},
                                   'limits': {'cpu': '2', 'memory': '512Mi'}}}}


def apply(items, args):
    manifest = {'apiVersion': 'v1', 'kind': 'List', 'items': [
        {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': safe_name(args.namespace)}}, *items]}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    print(kube('apply', '-f', '-', data=manifest), end='')
    if not args.no_wait:
        # API request timeout is separate from the overall wait operation.
        command = ['kubectl', '-n', args.namespace, 'wait',
                   *['nputask/' + item['metadata']['name'] for item in items],
                   '--for=jsonpath={.status.phase}=Running', '--timeout=180s']
        subprocess.run(command, check=True)


def view(args):
    source = Path(args.source).resolve()
    if source == ROOT / 'assets/pedestrians.avi':
        subprocess.run([sys.executable, str(ROOT / 'prepare_video.py')], check=True)
    if not source.is_file():
        raise ValueError(f'Source does not exist: {source}')
    safe_name(args.session, 40)
    output = Path(args.output or ROOT / 'results' / ('framework-' + time.strftime('%Y%m%dT%H%M%S'))).resolve()
    command = ['sh', str(ROOT / 'run-viewer.sh'), '--source', str(source),
               '--socket', '/data/osh-vision/' + args.session + '/hub.sock',
               '--output', str(output), '--fps', str(args.fps), '--frames', str(args.frames)]
    if args.headless:
        command += ['--headless']
    else:
        command += ['--screenshot', str(output / 'display.png')]
    print('Measurements:', output, flush=True)
    subprocess.run(command, check=True)


def doctor(models):
    """Local single-node preflight; never creates or modifies cluster objects."""
    result = subprocess.run(['k3s', 'ctr', '-n', 'k8s.io', 'images', 'list', '-q'],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Cannot inspect K3s images; run doctor with sudo. ' + result.stderr.strip())
    cached = set(result.stdout.splitlines())
    missing = [m['image'] for m in models.values()
               if m['image'] not in cached and 'docker.io/library/' + m['image'] not in cached]
    bundle = Path(os.environ.get('OSH_VISION_BUNDLE', str(ROOT / 'build/runtime')))
    controller = json.loads(kube('-n', 'npu-task-system', 'get', 'deployment', 'npu-task-controller', '-o', 'json'))
    report = {'missing_local_images': missing,
              'native_viewer_present': (bundle / 'app/vision_hub').is_file(),
              'controller_ready_replicas': controller.get('status', {}).get('readyReplicas', 0),
              'root_available_bytes': shutil.disk_usage('/').free,
              'repository_available_bytes': shutil.disk_usage(ROOT).free}
    print(json.dumps(report, indent=2))
    if missing or not report['native_viewer_present'] or not report['controller_ready_replicas']:
        raise ValueError('Preflight failed: build/import matching worker images and prepare the viewer/controller before deploying')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', type=Path, default=ROOT / 'models.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('models', help='List supported standard model profiles')
    sub.add_parser('doctor', help='Read-only local K3s image and viewer/controller preflight; use sudo')
    for cmd in ('deploy', 'demo'):
        p = sub.add_parser(cmd)
        if cmd == 'deploy':
            p.add_argument('name')
            p.add_argument('--model', required=True)
            p.add_argument('--npus', type=int, default=1)
            p.add_argument('--batch-size', type=int, default=1)
        else:
            p.add_argument('--deploy-only', action='store_true')
        p.add_argument('--node', help='Display/NPU node hostname; auto-discover on a single eligible node')
        p.add_argument('--namespace', default='osh-vision')
        p.add_argument('--session', default='demo')
        p.add_argument('--dry-run', action='store_true', help='Print manifests without cluster/network access')
        p.add_argument('--no-wait', action='store_true')
    playback = sub.add_parser('view')
    for p in (playback, sub.choices['demo']):
        p.add_argument('--source', default=str(ROOT / 'assets/pedestrians.avi'))
        p.add_argument('--fps', type=float, default=10)
        p.add_argument('--frames', type=int, default=0)
        p.add_argument('--headless', action='store_true')
        p.add_argument('--output')
        if p is playback:
            p.add_argument('--session', default='demo')
    for cmd in ('status', 'logs', 'metrics', 'suspend', 'resume', 'delete'):
        p = sub.add_parser(cmd)
        p.add_argument('--namespace', default='osh-vision')
        if cmd != 'status':
            p.add_argument('name')
        if cmd == 'logs':
            p.add_argument('--follow', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(args, 'fps') and (not math.isfinite(args.fps) or not 0 < args.fps <= 120 or args.frames < 0):
        raise ValueError('FPS must be between 0 and 120 and frames must be nonnegative')
    if args.command == 'models':
        print(json.dumps(catalogue(args.catalogue), indent=2))
    elif args.command == 'doctor':
        doctor(catalogue(args.catalogue))
    elif args.command in ('deploy', 'demo'):
        models = catalogue(args.catalogue)
        node = choose_node(args.node, args.dry_run)
        if args.command == 'deploy':
            if args.model not in models:
                raise ValueError('Unknown model; run "models" to list catalogue entries')
            items = [task(args.name, models[args.model], node, args.namespace, args.session, args.npus, args.batch_size)]
        else:
            items = [task(name, models[name], node, args.namespace, args.session) for name in ('detection', 'segmentation', 'pose')]
        apply(items, args)
        if args.command == 'demo' and not args.dry_run and not args.deploy_only:
            view(args)
    elif args.command == 'view':
        view(args)
    elif args.command == 'status':
        print(kube('-n', args.namespace, 'get', 'nputasks'), end='')
    else:
        safe_name(args.name)
        safe_name(args.namespace)
        if args.command in ('suspend', 'resume'):
            patch = json.dumps({'spec': {'suspend': args.command == 'suspend'}})
            print(kube('-n', args.namespace, 'patch', 'nputask', args.name, '--type=merge', '-p', patch), end='')
        elif args.command == 'delete':
            print(kube('-n', args.namespace, 'delete', 'nputask', args.name, '--ignore-not-found'), end='')
        else:
            obj = json.loads(kube('-n', args.namespace, 'get', 'nputask', args.name, '-o', 'json'))
            pod = obj.get('status', {}).get('podName')
            if not pod:
                raise ValueError('Task has no pod yet; inspect status and Pending conditions')
            if args.command == 'logs':
                command = ['kubectl', '-n', args.namespace, 'logs', pod]
                if args.follow:
                    command += ['-f']
                subprocess.run(command, check=True)
            else:
                lines = kube('-n', args.namespace, 'logs', pod, '--tail=100').splitlines()
                for line in reversed(lines):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    timing = record.get('timing', record)
                    if 'worker_metrics' in timing:
                        print(json.dumps({k: timing[k] for k in ('task', 'model', 'device', 'session', 'seq', 'postprocess_end_ns', 'output_applied_ns', 'batch_size', 'container_durations', 'worker_metrics', 'output_metrics') if k in timing}, indent=2))
                        break
                else:
                    raise ValueError('No measured frames yet; start the viewer and use updated v3 worker images')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('osh:', error, file=sys.stderr)
        sys.exit(1)
