#!/usr/bin/env python3
"""Package compiled apps and their runtime dependency closure from an SDK root.

Useful when Docker RUN cannot start because the host root disk is full. The
result is also a relocatable native viewer bundle; NVIDIA driver libraries stay
on the host and are deliberately excluded.
"""
import argparse,pathlib,re,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--sdk',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
sdk=a.sdk.resolve();out=a.output.resolve();(out/'app').mkdir(parents=True,exist_ok=True);(out/'lib').mkdir(exist_ok=True)
programs=list((sdk/'build/vision').glob('vision_*'))
for program in programs:
    if not program.is_file() or not program.stat().st_mode&0o111:continue
    shutil.copy2(program,out/'app'/program.name)
    listing=subprocess.check_output(['chroot',str(sdk),'ldd','/'+str(program.relative_to(sdk))],text=True)
    if 'not found' in listing:raise RuntimeError(listing)
    for lib in re.findall(r'(/[^\s()]+)',listing):
        real=subprocess.check_output(['chroot',str(sdk),'readlink','-f',lib],text=True).strip()
        src=sdk/real.lstrip('/')
        if src.is_file():shutil.copy2(src,out/'lib'/pathlib.Path(lib).name)
# SDL loads GL dynamically, so its vendor-neutral dispatch libraries must be explicit.
for name in ['libGL.so.1','libGLX.so.0','libGLdispatch.so.0','libOpenGL.so.0','libEGL.so.1']:
    src=sdk/'usr/lib/aarch64-linux-gnu'/name
    if src.exists():shutil.copy2(src.resolve(),out/'lib'/name)
(out/'etc').mkdir(exist_ok=True);(out/'tmp').mkdir(exist_ok=True);(out/'tmp').chmod(0o1777)
# HailoRT needs PCI topology; Kubernetes's device plugin supplies the per-pod sysfs view.
print('runtime bundle:',out)
