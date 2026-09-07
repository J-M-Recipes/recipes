"""Finite-corpus verdict; requires all three complete lane receipts."""
import json,math,pathlib,statistics
W=pathlib.Path(__file__).resolve().parent

def upper(k,n,alpha=.05):
 if k==n:return 1.
 lo,hi=0.,1.
 for _ in range(80):
  p=(lo+hi)/2
  cdf=sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
  if cdf>=alpha:lo=p
  else:hi=p
 return (lo+hi)/2

def read(name):return [json.loads(x) for x in (W/'results'/name).read_text().splitlines()]
def audit():
 fixture=json.loads((W/'primary_100_seed20260906.json').read_text());tasks={x['id']:x for x in fixture['tasks']};assert len(tasks)==100
 expected={(tid,r) for tid in tasks for r in range(2)};lanes={};secondary={};speed={};invalid=[]
 for lane in ['v1','sc13g','sc13g-mtp']:
  rows=read(lane+'-primary.jsonl');keys=[(r['task_id'],r['repeat']) for r in rows];assert len(keys)==len(set(keys))==200 and set(keys)==expected,(lane,'primary-count')
  for r in rows:
   assert r['fixture_set_hash']==fixture['metadata']['fixture_set_hash'] and r['fixture_hash']==tasks[r['task_id']]['fixture_hash'] and r['lane']==lane
   if r.get('invalid') is not False or r.get('protocol_error') or type(r.get('pass')) is not bool:invalid.append([lane,r['task_id'],r['repeat']])
  counts={tid:sum(int(r['pass']) for r in rows if r['task_id']==tid) for tid in tasks}
  acc={cat:sum(int(r['pass']) for r in rows if tasks[r['task_id']]['type']==cat)/sum(1 for r in rows if tasks[r['task_id']]['type']==cat) for cat in ['humaneval','gsm8k']}
  acc['overall']=sum(counts.values())/200;lanes[lane]={'counts':counts,'accuracy':acc,'competent':acc['overall']>=.85 and min(acc['humaneval'],acc['gsm8k'])>=.8}
  sec=read(lane+'-secondary.jsonl');sec_expected={(f'{prefix}-{i}',r) for prefix in ['grounded','structure'] for i in range(10) for r in range(2)}|{(f'tool-chain-{i}',0) for i in range(4)}
  sec_keys=[(r['fixture'],r.get('repeat',0)) for r in sec];assert len(sec)==len(set(sec_keys))==44 and set(sec_keys)==sec_expected,(lane,'secondary-count')
  secondary[lane]={'passed':all(r.get('passed') is True for r in sec),'pass_count':sum(r.get('passed') is True for r in sec),'total':len(sec)}
  speeds=read(lane+'-speed.jsonl');sp_keys=[(r['class'],r['repeat']) for r in speeds];assert len(speeds)==len(set(sp_keys))==16 and set(sp_keys)=={(c,r) for c in ['short','long'] for r in range(8)}
  speed[lane]={}
  for cls in ['short','long']:
   sr=[r for r in speeds if r['class']==cls and r['repeat']>0];assert all(r['usage']['completion_tokens']==512 and r['wall_s']>0 for r in sr)
   speed[lane][cls]={'median_effective_tps':statistics.median(r['usage']['completion_tokens']/r['wall_s'] for r in sr),'median_wall_s':statistics.median(r['wall_s'] for r in sr)}
 c=lanes['sc13g-mtp'];b=lanes['sc13g'];loss=sum(c['counts'][t]<b['counts'][t] for t in tasks);win=sum(c['counts'][t]>b['counts'][t] for t in tasks);ub=upper(loss,100)
 deficits={lane:{cat:lanes[lane]['accuracy'][cat]-c['accuracy'][cat] for cat in ['overall','humaneval','gsm8k']} for lane in ['v1','sc13g']}
 gain=speed['sc13g-mtp']['short']['median_effective_tps']/speed['sc13g']['short']['median_effective_tps']-1
 longratio=speed['sc13g-mtp']['long']['median_wall_s']/speed['sc13g']['long']['median_wall_s']
 fail=any(v>.05+1e-12 for d in deficits.values() for v in d.values()) or not secondary['sc13g-mtp']['passed']
 complete=not invalid and all(l['competent'] for l in lanes.values()) and all(s['passed'] for s in secondary.values())
 passed=not fail and complete and ub<.05 and gain>=.1 and longratio<=1.1
 verdict='PASS' if passed else 'INCONCLUSIVE' if invalid or not complete or (not fail and ub>=.05) else 'FAIL_PROMOTION'
 for lane in lanes:lanes[lane].pop('counts')
 return {'verdict':verdict,'claim':'Bounded low-reasoning task-loss gate, not universal quality or mean-score noninferiority','unique_primary_tasks':100,'repeats_per_task':2,'lanes':lanes,'matched_comparison':{'loss':loss,'win':win,'tie':100-loss-win,'one_sided_95pct_gross_loss_upper':ub},'descriptive_accuracy_deficits':deficits,'invalid_rows':invalid,'secondary':secondary,'speed':speed,'short_median_tps_gain_fraction':gain,'long_median_wall_ratio':longratio}
if __name__=='__main__':
 try:result=audit()
 except Exception as e:result={'verdict':'INCONCLUSIVE','invalid_artifacts':str(e),'error_type':type(e).__name__}
 print(json.dumps(result,indent=2))
 raise SystemExit(0 if result['verdict']=='PASS' else 1)
