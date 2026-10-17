#!/usr/bin/env python3
"""Download verified assets atomically; normal playback is fully offline."""
import hashlib,json,pathlib,urllib.request
root=pathlib.Path(__file__).resolve().parent
for entry in json.loads((root/'assets.lock.json').read_text()):
    dest=root/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==entry['sha256']:
        print('verified',dest.name);continue
    req=urllib.request.Request(entry['url'],headers={'User-Agent':'OSH-Vision/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise RuntimeError('checksum mismatch: '+entry['path'])
    temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(data);temp.replace(dest);print('downloaded',dest.name)
model=root.parent/'yolov10s.hef'
if hashlib.sha256(model.read_bytes()).hexdigest()!='ceab636a9131343ab10cdea413a9081b6a4cd64035a02b26ea45e0195248f0fc':
    raise RuntimeError('existing YOLOv10s HEF differs from the validated model')
