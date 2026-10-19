#!/usr/bin/env python3
"""Profile an isolated model with the exclusive reference runner; emit startup config."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def profile(runner, hef, key, architecture, runtime):
    checksum = hashlib.sha256(Path(hef).read_bytes()).hexdigest()
    env = {k: v for k, v in os.environ.items()
           if k not in ('NPU_SCHEDULER_ADDRESS', 'NPU_WORKLOAD_ID', 'NPU_SHARE',
                        'NPU_TARGET_FPS', 'NPU_RESERVED_DEMAND', 'NPU_ADMISSION_POLICY',
                        'HAILORT_SERVICE_ADDRESS')}
    env['NPU_MODEL_SHA256'] = checksum
    # TemporaryDirectory owns only this run's raw output.
    with tempfile.TemporaryDirectory(prefix='osh-profile-') as temp:
        raw = Path(temp) / 'profile.json'
        subprocess.run([str(runner), '--hef', str(hef), '--warmup', '50', '--runs', '500',
                        '--profile-json', str(raw)], env=env, check=True)
        result = json.loads(raw.read_text())
    if result.get('samples') != 500 or result.get('warmup') != 50:
        raise ValueError('Incomplete profile measurement')
    for field in ('meanSeconds', 'p95Seconds'):
        if not math.isfinite(result[field]) or result[field] <= 0:
            raise ValueError('Invalid measured execution cost')
    result.pop('warmup')
    result.update(modelSHA256=checksum, deviceArchitecture=architecture, runtimeVersion=runtime)
    return {'sharedAdmissionPolicy': 'utilization-best-effort', 'budget': 800,
            'profileMargin': 1.2, 'profiles': {key: result}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--hef', type=Path, required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--architecture', required=True, help='Verified device architecture, e.g. HAILO8')
    parser.add_argument('--runtime-version', required=True, help='Verified runtime, e.g. 4.21.0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = profile(args.runner.resolve(), args.hef.resolve(), args.key,
                     args.architecture, args.runtime_version)
    args.output.write_text(json.dumps(config, indent=2) + '\n')
    print('Profile saved; review compatibility metadata before adding it to cluster configuration.')


if __name__ == '__main__':
    main()
