#!/usr/bin/env python3
"""Run three deployed real models, NVIDIA display, optional lifecycle/failure checks.

Requires vision/tasks.yaml deployed. Only creates the invalid-model test task;
never touches the existing dispatcher workloads. Restores pose after suspension.
"""
import argparse,json,pathlib,subprocess,time,uuid
ROOT=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--source',default=str(ROOT/'vision/assets/replay.mp4'));p.add_argument('--output',type=pathlib.Path,default=ROOT/'vision/results/hardware-video');p.add_argument('--exercise-lifecycle',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def kube(*args,obj=None,check=True):
    return subprocess.run(['kubectl','--request-timeout=10s',*args],input=json.dumps(obj) if obj else None,text=True,capture_output=True,check=check,timeout=20)
def wait_for(test,seconds=60):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=test()
        if value:return value
        if viewer.poll() is not None:raise RuntimeError('viewer exited: '+(a.output/'viewer.log').read_text())
        time.sleep(.2)
    raise TimeoutError('hardware acceptance timed out')
def task(name):return json.loads(kube('-n','osh-vision','get','nputask',name,'-o','json').stdout)
def patch(name,suspend):kube('-n','osh-vision','patch','nputask',name,'--type=merge','-p',json.dumps({'spec':{'suspend':suspend}}))
log=a.output/'frames.jsonl';before=log.stat().st_size if log.exists() else 0
handle=open(a.output/'viewer.log','w')
viewer=subprocess.Popen([str(ROOT/'vision/run-viewer.sh'),'--source',a.source,'--fps','10','--frames','200','--socket','/data/osh-vision/demo/hub.sock','--output',str(a.output),'--screenshot',str(a.output/'display.png')],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
def records():
    if not log.exists():return []
    with log.open() as f:f.seek(before);lines=f.readlines()
    rows=[]
    for line in lines:
        try:r=json.loads(line)
        except json.JSONDecodeError:continue # Last record may be in flight.
        if r.get('event')=='frame':rows.append(r)
    return rows
def count(kind):return sum(r['result']['kind']==kind for r in records())
bad='invalid-model-'+uuid.uuid4().hex[:6];suspended=False;report={'passed':False,'source':a.source}
try:
    wait_for(lambda:all(count(k)>=15 for k in ['detection','segmentation','pose']))
    rows=records();devices={};models={}
    for kind in ['detection','segmentation','pose']:
        sample=[r for r in rows if r['result']['kind']==kind]
        devices[kind]=sample[-1]['device'];models[kind]=sample[-1]['model']
        assert any(r['result']['objects'] for r in sample),kind+' has no predictions'
        if kind=='segmentation':assert any(o.get('mask',{}).get('runs') for r in sample for o in r['result']['objects'])
        if kind=='pose':assert any(len(o.get('keypoints',[]))==17 and any(k[2]>.5 for k in o['keypoints']) for r in sample for o in r['result']['objects'])
    assert len(set(devices.values()))==3,devices
    assert len(set(models.values()))==3,models
    assert all(r['displayed'] and r['gpu_complete_ns'] and 'NVIDIA' in r['renderer'] and not r['mock'] for r in rows)
    report.update(devices=devices,models=models)
    if a.exercise_lifecycle:
        patch('pose',True);suspended=True
        wait_for(lambda:task('pose')['status']['phase']=='Suspended')
        n=count('detection');wait_for(lambda:count('detection')>=n+5)
        n=count('pose');patch('pose',False);suspended=False;wait_for(lambda:count('pose')>=n+5)
        report['suspend_resume']=True
        kube('apply','-f','-',obj={'apiVersion':'npu.snu-rtos.io/v1alpha1','kind':'NPUTask','metadata':{'name':bad,'namespace':'osh-vision'},'spec':{'image':'osh-vision-detection:v2','command':['/app/vision_worker'],'args':['--kind','detection'],'model':{'path':'/models/missing.hef'},'npuCount':1,'mode':'Batch'}})
        wait_for(lambda:task(bad).get('status',{}).get('phase')=='Failed')
        pod=task(bad)['status']['podName'];error=kube('-n','osh-vision','logs',pod).stdout
        assert 'WORKER_FAILED' in error,error
        (a.output/'invalid-model.log').write_text(error)
        report['invalid_model_failed']=True
        kube('-n','osh-vision','delete','nputask',bad,'--wait=true')
        wait_for(lambda:kube('-n','osh-vision','get','pod',pod,check=False).returncode!=0)
        report['invalid_model_released']=True
    # Capture per-container acknowledgments while frames are still arriving;
    # kubelet may rotate its current log immediately after a long run.
    for name in ['detection','segmentation','pose']:
        pod=task(name)['status']['podName'];text=kube('-n','osh-vision','logs',pod,'--tail=30').stdout
        assert 'gpu_complete_ns' in text,name+' missing output completion log'
        (a.output/(name+'.log')).write_text(text)
    wait_for(lambda:viewer.poll() is not None,seconds=90)
    assert viewer.returncode==0,(a.output/'viewer.log').read_text()
    rows=records();report['frames']={k:sum(r['result']['kind']==k for r in rows) for k in devices}
    assert (a.output/'display.png').exists()
    report['passed']=True
finally:
    if suspended:patch('pose',False)
    kube('-n','osh-vision','delete','nputask',bad,'--ignore-not-found',check=False)
    if viewer.poll() is None:viewer.terminate()
    try:viewer.wait(timeout=8)
    except subprocess.TimeoutExpired:viewer.kill();viewer.wait()
    handle.close();(a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
