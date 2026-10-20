#!/usr/bin/env python3
"""Fetch the pinned pedestrian video once; verify it for offline replay."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def prepare(root, entry, offline=False):
    dest = root / entry['path']
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == entry['sha256']:
        return dest
    if offline:
        raise RuntimeError('Video missing or checksum mismatch; run python3 vision/prepare_video.py')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'OSH-Vision/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(entry['bytes'] + 1)
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise RuntimeError('Video checksum/size mismatch; existing asset was retained')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Verify without network access')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    entry = json.loads((root / 'video.lock.json').read_text())
    print('verified', prepare(root, entry, args.offline))


if __name__ == '__main__':
    main()
