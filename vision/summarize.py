#!/usr/bin/env python3
"""Summarize joined frame latency, excluding warmup, grouped by session and task."""
import argparse,collections,json,math,pathlib
p=argparse.ArgumentParser();p.add_argument('log',type=pathlib.Path);a=p.parse_args();groups=collections.defaultdict(list);drops=collections.Counter()
for line in a.log.read_text().splitlines():
    r=json.loads(line)
    if r.get('event')=='drop':drops[(r['session'],r['task'])]+=1
    if r.get('event')=='frame':groups[(r['session'],r['task'])].append(r)
result=[]
for (session,task),allrows in groups.items():
    rows=[r for r in allrows if not r['warmup']];d={}
    for key in sorted({k for r in rows for k in r['durations']}):
        v=sorted(r['durations'][key] for r in rows if key in r['durations'])
        d[key]={f'p{n}':v[max(0,math.ceil(len(v)*n/100)-1)] for n in (50,95,99)}
    elapsed=(max(r['output_applied_ns'] for r in rows)-min(r['acquired_ns'] for r in rows))/1e9 if rows else 0
    result.append({'session':session,'task':task,'frames':len(rows),'warmup_frames':len(allrows)-len(rows),'dropped':drops[(session,task)],'throughput_fps':len(rows)/elapsed if elapsed else 0,'latency_ms':d})
print(json.dumps(result,indent=2))
