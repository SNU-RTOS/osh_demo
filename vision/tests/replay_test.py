#!/usr/bin/env python3
"""Exercise independent backpressure, reconnect, identity and latency over real UDS."""
import argparse,json,pathlib,subprocess,tempfile,time,socket,struct
p=argparse.ArgumentParser();p.add_argument('--bin',type=pathlib.Path,required=True);args=p.parse_args()
with tempfile.TemporaryDirectory(prefix='osh-vision-') as tmp:
    root=pathlib.Path(tmp); image=root/'source.ppm'
    image.write_bytes(b'P6\n320 240\n255\n'+bytes([200,100,50])*(320*240))
    sock=str(root/'hub.sock'); processes=[];handles=[]
    def start(binary,*a,name):
        h=open(root/(name+'.log'),'w');handles.append(h)
        proc=subprocess.Popen([str(args.bin/binary),*a],stdout=h,stderr=subprocess.STDOUT);processes.append(proc);return proc
    def wait_for(test,seconds=10):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if test(): return
            time.sleep(.05)
        raise AssertionError('timed out: '+ '\n'.join(f.read_text() for f in root.glob('*.log')))
    try:
        hub=start('vision_hub','--headless','--source',str(image),'--socket',sock,'--output',str(root/'results'),'--fps','30',name='hub')
        wait_for(lambda:pathlib.Path(sock).exists())
        # Reject an oversized packet before allocation, then keep serving peers.
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as malformed:
            malformed.connect(sock);malformed.sendall(struct.pack('!II',0xffffffff,0))
        fast=start('vision_mock_worker','--mock','--task','fast','--socket',sock,name='fast')
        slow=start('vision_mock_worker','--mock','--task','slow','--kind','pose','--delay-ms','200','--socket',sock,name='slow')
        frames=root/'results/frames.jsonl'
        def records():
            if not frames.exists():return []
            return [d for line in frames.read_text().splitlines() if (d:=json.loads(line)).get('event')=='frame']
        wait_for(lambda:len([r for r in records() if r['task']=='slow'])>=6)
        fast.terminate();fast.wait(timeout=7)
        start('vision_mock_worker','--mock','--task','fast','--socket',sock,name='fast-reconnected')
        count=len([r for r in records() if r['task']=='fast'])
        wait_for(lambda:len([r for r in records() if r['task']=='fast'])>=count+8)
        rows=records(); f=[r for r in rows if r['task']=='fast'];s=[r for r in rows if r['task']=='slow']
        assert len(f)>len(s)*2,(len(f),len(s))
        assert s[-1]['drops']>0
        assert all(not r['displayed'] and r['gpu_complete_ns'] is None for r in rows)
        assert all(r['durations']['acquisition_to_result_ms']>=r['durations']['inference_ms'] for r in rows)
        assert all(r['width']==320 and r['height']==240 for r in rows)
        assert len({(r['session'],r['task'],r['seq']) for r in rows})==len(rows)
        slow.terminate();slow.wait(timeout=7);hub.terminate();hub.wait(timeout=7)
        assert hub.returncode==0,(root/'hub.log').read_text()
        print('VISION_REPLAY_OK',len(f),len(s),'slow_drops',s[-1]['drops'])
    finally:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=7)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for h in handles:h.close()
