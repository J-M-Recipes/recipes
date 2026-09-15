"""Read-only private evidence audit; stdout contains allowlisted aggregates only."""
import base64, collections, hashlib, json, math, pathlib, statistics, sys, datetime, subprocess, socket
ROOT = pathlib.Path(sys.argv[1])
REL = pathlib.Path(sys.argv[2])
def unique(pairs):
    d = {}
    for k,v in pairs:
        assert k not in d, 'duplicate JSON key'
        d[k] = v
    return d
def parse(s):
    return json.loads(s, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def load(p): return parse(p.read_text())
def lines(p): return [parse(l) for l in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def quant(v,p): return sorted(v)[math.ceil(len(v)*p)-1]
def stats(v): return {'mean':statistics.fmean(v),'min':min(v),'max':max(v)}
manifest=load(REL/'manifest.json')
for p,h in manifest['files'].items(): assert sha(REL/p)==h, p
sys.dont_write_bytecode=True
sys.path.insert(0,str(REL/'gauntlet'))
import gauntlet
sys.path.insert(0,str(REL/'replay'))
import analyze_runs
contract=load(REL/'container-contract.json'); campaign=load(ROOT/'summary.json'); events=lines(ROOT/'receipts.jsonl')
assert campaign['boot_count']==6 and campaign['order']==contract['order']
assert [(e['pair'],e['profile']) for e in events if e['event']=='boot_start']==[tuple(x) for x in contract['order']]
assert events[-1]['event']=='campaign_complete'
assert all(events[i]['t']>=events[i-1]['t'] for i in range(1,len(events)))
tasks={t.id:t for t in gauntlet.build_tasks()}
assert len(tasks)==12
report={'checked_at':datetime.datetime.now().astimezone().isoformat(),'release_manifest_sha256':sha(REL/'manifest.json'),'source_manifest_sha256':sha(REL/'source-manifest.json'),'release_files_verified':len(manifest['files']),'replays':[],'gauntlets':[],'pairs':[], 'limitations':['Three paired boots; descriptive means/ranges, not inferential significance.','Twelve unique gauntlet tasks repeated; not seventy-two independent tasks.','Recorded-history replay is not production capacity.']}
request_maps={}; response_maps={}; private_hashes={}; rows_count=0; task_rows=0
for pair,profile in contract['order']:
    b=ROOT/'boots'/f'pair{pair}-{profile}'
    identity=load(b/'identity.json')['identity']
    for k,v in contract['profiles'][profile].items(): assert identity[k]==v, ('identity',pair,profile,k)
    proof=load(b/'stop-proof.json')['proof'];assert all(p['running'] is False and p['port_30006_dark'] is True for p in proof)
    iso=load(b/'gauntlet-scored/sandbox-isolation.json');assert iso['ok'] and all(iso['evaluator']['checks'].values())
    for phase in ['replay-initial','replay-repeat']:
        p=b/phase; s=load(p/'summary.json'); rr=lines(p/'turn_records.jsonl');assert len(rr)==300
        assert s['fixture_sha256']==contract['fixture_sha256']
        assert s['model']==contract['model'] and s['workers']==4 and s['max_tokens']==400 and s['reasoning_effort']=='low'
        req={}; resp={}; groups=collections.defaultdict(list)
        for r in rr:
            assert r['status']=='ok';key=(r['session_id'],r['turn_index']);assert key not in req
            req[key]=r['request_body_sha256'];groups[r['session_id']].append(r)
            for k in ['t_start','t_first','t_end','duration_s','ttft_s']: assert isinstance(r[k],(int,float)) and math.isfinite(r[k]) and r[k]>0
            assert r['t_start']<=r['t_first']<=r['t_end']
            assert math.isclose(r['t_end']-r['t_start'],r['duration_s'],abs_tol=0.001)
            raw=base64.b64decode(r['response']['raw_sse_base64'],validate=True)
            assert raw.strip().endswith(b'data: [DONE]')
            usage=None; finish=None
            for line in raw.splitlines():
                if not line.startswith(b'data: '): continue
                data=line[6:].strip()
                if data==b'[DONE]':continue
                chunk=parse(data)
                if chunk.get('usage'):usage=chunk['usage']
                for c in chunk.get('choices',[]):
                    if c.get('finish_reason'):finish=c['finish_reason']
            assert usage is not None and finish==r['finish_reason']
            for k in ['prompt_tokens','completion_tokens','total_tokens']:assert usage[k]==r['usage'][k]
            resp[key]=json.dumps({k:r['response'][k] for k in ['content','reasoning','tool_calls']},sort_keys=True)
        assert len(groups)==20
        for rs in groups.values():
            rs.sort(key=lambda r:r['turn_index']);assert [r['turn_index'] for r in rs]==list(range(15))
            assert all(rs[i]['t_start']>=rs[i-1]['t_end'] for i in range(1,15))
        for field,metric in [('prompt_tokens','prompt_tokens_metric'),('completion_tokens','generation_tokens_metric')]:assert sum(r['usage'][field] for r in rr)==s[field]==s['metrics_delta'][metric]
        assert all(math.isfinite(v) and v>=0 for v in s['metrics_delta'].values())
        assert math.isclose(s['completion_tokens']/s['wall_s'],s['aggregate_completion_tok_s'],rel_tol=1e-12)
        for name,q in [('p50',.5),('p90',.9),('p95',.95)]:assert quant([r['ttft_s'] for r in rr],q)==s['ttft_s'][name]
        report['replays'].append({'pair':pair,'profile':profile,'phase':phase,**{k:s[k] for k in ['turn_count','wall_s','completion_tokens','aggregate_completion_tok_s','ttft_s','prefix_cache','acceptance']},'finish_reasons':dict(collections.Counter(r['finish_reason'] for r in rr))})
        request_maps[pair,profile,phase]=req;response_maps[pair,profile,phase]=resp;rows_count+=len(rr)
        private_hashes[str((p/'turn_records.jsonl').relative_to(ROOT))]=sha(p/'turn_records.jsonl')
    g=load(b/'gauntlet-scored/summary.json');gr=lines(b/'gauntlet-scored/task_results.jsonl')
    assert len(gr)==12 and {r['task_id'] for r in gr}==set(tasks)
    assert {r['task_id']:r for r in gr}=={r['task_id']:r for r in g['results']}
    details=[]
    for r in gr:
        t=tasks[r['task_id']];regrade=gauntlet.evaluate_task_result(t,r['final_content'],r['trace'],r['oracle_result'])
        assert regrade['correct']==r['correct']
        for k in ['schema_valid','final_answer_valid']:assert regrade['validity'][k]==r['validity'][k]
        assert r['score_count']==1 and math.isfinite(r['wall_s']) and 0<r['wall_s']<=180
        raw=lines(b/'gauntlet-scored/raw'/f'{t.id}.jsonl');requests=[x for x in raw if x['event']=='request'];responses=[x for x in raw if x['event']=='response']
        assert len(requests)==len(responses) and 1<=len(requests)<=12
        assert len([x for x in raw if x['event']=='score'])==1
        if t.kind=='code_fix' and r['correct']:
            oracle=r['oracle_result'];assert oracle['ok'] and oracle['exit_code']==0 and oracle['evaluator']['completed_cases']==oracle['evaluator']['expected_cases']
            cmd=oracle['command'];assert '--network' in cmd and cmd[cmd.index('--network')+1]=='none' and '--runtime=runc' in cmd and '--read-only' in cmd
        details.append({k:r[k] for k in ['task_id','kind','success','correct','finish_reason','wall_s','ttft_s','tool_call_count','validity']})
    assert sum(r['success'] for r in gr)==g['success_count']
    task_rows+=len(gr)
    report['gauntlets'].append({'pair':pair,'profile':profile,**{k:g[k] for k in ['wall_s','task_count','success_count','protocol_valid_count','capture_error_count']},'tasks':details})
for pair in [1,2,3]:
    base=request_maps[pair,'v14','replay-initial']
    assert all(request_maps[pair,profile,phase]==base for profile in ['v14','v13'] for phase in ['replay-initial','replay-repeat'])
    for phase in ['replay-initial','replay-repeat']:
        a=next(r for r in report['replays'] if (r['pair'],r['profile'],r['phase'])==(pair,'v14',phase));b=next(r for r in report['replays'] if (r['pair'],r['profile'],r['phase'])==(pair,'v13',phase))
        ma=response_maps[pair,'v14',phase];mb=response_maps[pair,'v13',phase]
        report['pairs'].append({'pair':pair,'phase':phase,'v14_throughput_gain_pct':100*(a['aggregate_completion_tok_s']/b['aggregate_completion_tok_s']-1),'v14_wall_reduction_pct':100*(1-a['wall_s']/b['wall_s']),'different_response_records_including_tool_ids':sum(ma[k]!=mb[k] for k in ma)})
report['native_analysis']={}
for phase in ['replay-initial','replay-repeat']:
    analysis=analyze_runs.compare_runs([(v,ROOT/'boots'/f'pair{p}-{v}'/phase/'summary.json') for v in ['v14','v13'] for p in [1,2,3]])
    report['native_analysis'][phase]=analysis['by_label']
report['counts']={'boots':6,'replay_runs':len(report['replays']),'replay_records':rows_count,'gauntlet_runs':len(report['gauntlets']),'task_attempts':task_rows,'unique_tasks':len(tasks)}
assert rows_count==3600 and task_rows==72
report['gauntlet_summary']={v:{'successes':sum(g['success_count'] for g in report['gauntlets'] if g['profile']==v),'suite_wall_s':stats([g['wall_s'] for g in report['gauntlets'] if g['profile']==v])} for v in ['v14','v13']}
report['capture_sha256']=private_hashes
report['finished_at']=datetime.datetime.fromtimestamp(events[-1]['t']).astimezone().isoformat()
print(json.dumps(report,sort_keys=True))
