#!/usr/bin/env python3
"""Run twelve fixed closed-loop tasks; failed answers stay in the denominator."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import statistics
import time
import gauntlet as g


def run_suite(args):
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be 1..4')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    tasks=g.build_tasks()
    expected={t.id for t in tasks}
    if len(tasks)!=12 or len(expected)!=12:
        raise ValueError('task corpus does not have twelve unique IDs')
    oracle=Path(g.__file__).with_name('oracle.py')
    guard=g.SandboxRunner(args.sandbox_image,oracle).run_tests('__isolation__',out)
    (out/'sandbox-isolation.json').write_text(json.dumps(guard,indent=2)+'\n')
    if not guard['ok']:
        raise RuntimeError('live CPU sandbox isolation gate failed')

    def one(task):
        taskdir=out/'tasks'/task.id
        g.prepare_task_dir(task,taskdir)
        client=g.OpenAIClient(args.base_url,timeout_s=180)
        client.cache_salt=args.cache_salt
        sandbox=g.SandboxRunner(args.sandbox_image,oracle,timeout_s=20)
        runner=g.GauntletRunner(client,sandbox,model=args.model,reasoning_effort='low',
                               max_tokens=1500,max_turns=12,task_timeout_s=180)
        return runner.run_task(task,taskdir,out/'raw')

    started=time.monotonic()
    results=[]
    with (out/'task_results.jsonl').open('x') as log, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            task=futures[f]
            try:
                result=f.result()
            except Exception as exc:
                result={'task_id':task.id,'kind':task.kind,'capture_error':str(exc),'success':False}
            results.append(result)
            log.write(json.dumps(result,sort_keys=True)+'\n');log.flush()
            print(json.dumps({'tag':args.tag,'task':task.id,'success':result['success'],
                              'done':len(results),'expected':len(tasks)}),flush=True)
    wall=time.monotonic()-started
    results.sort(key=lambda r:r['task_id'])
    if len(results)!=12 or {r['task_id'] for r in results}!=expected:
        raise RuntimeError('incomplete/duplicate task capture')
    good=[r['wall_s'] for r in results if r.get('success')]
    summary={
        'tag':args.tag,'model':args.model,'workers':args.workers,
        'task_count':len(results),'expected_task_count':len(tasks),
        'success_count':sum(r['success'] for r in results),
        'capture_error_count':sum('capture_error' in r for r in results),
        'protocol_valid_count':sum(r.get('validity',{}).get('transport_protocol_valid',False) for r in results),
        'wall_s':wall,
        'successful_task_wall_s':{'n':len(good),'median':statistics.median(good) if good else None,
                                  'range':[min(good),max(good)] if good else None},
        'request_settings':{'temperature':0,'seed':42,'reasoning_effort':'low','max_tokens':1500,
                            'max_turns':12,'task_timeout_s':180,'cache_salt':args.cache_salt},
        'statistics_note':'descriptive; repetitions are clustered by the twelve tasks, not independent samples',
        'promotion_authorized':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),Path(g.__file__),oracle)},
        'results':results,
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return 1 if summary['capture_error_count'] else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-url','model','output','tag','cache-salt','sandbox-image'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=4)
    return run_suite(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
