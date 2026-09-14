"""Run one already-started lane; never stops/starts serving containers."""
import argparse,fcntl,hashlib,json,pathlib,subprocess,sys,time,urllib.request
W=pathlib.Path(__file__).resolve().parent
IM='sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14'
CONTAINERS={'k1-256k':'glm53-big-sc13g-mtp-ctx256k-DAILY-20260913','k2-256k':'glm53-big-sc13g-mtp2-ctx256k-K2-selfrepeat-20260913'}
def main():
 p=argparse.ArgumentParser();p.add_argument('lane',choices=CONTAINERS);a=p.parse_args();lane=a.lane
 with (W/'campaign.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert (W/'CONTROL').read_text().strip()=='RUN'
  expected=json.loads((W/'frozen-files.json').read_text())
  for name,digest in expected.items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==digest,('hash-mismatch',name)
  name=CONTAINERS[lane];info=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert info['State']['Running'] and info['Image']==IM
  key=pathlib.Path('~/.glm_api_key').read_text().strip();assert key
  req=urllib.request.Request('http://127.0.0.1:30001/v1/models',headers={'Authorization':'Bearer '+key})
  with urllib.request.urlopen(req,timeout=15) as response:models=json.load(response)
  assert any(x['id']=='glm-5.3-big' for x in models['data'])
  cmd=info['Config']['Cmd'][:]
  if '--api-key' in cmd:cmd[cmd.index('--api-key')+1]='[REDACTED]'
  out=W/'results';out.mkdir(exist_ok=True)
  assert not (out/(lane+'-primary.jsonl')).exists(),'Fresh outputs required'
  (out/(lane+'-config.json')).write_text(json.dumps({'container':name,'image_id':info['Image'],'cmd':cmd,'start_time':info['State']['StartedAt'],'measured_at_unix':time.time()},indent=2))
  commands=[('primary',[sys.executable,'-u','harness.py','--fixtures','primary_100_seed20260906.json','--sandbox-image',IM,'--key-file','~/.glm_api_key','--base-url','http://127.0.0.1:30001','--model','glm-5.3-big','--lane',lane,'--out',str(out/(lane+'-primary.jsonl')),'--repeats','2','--max-tokens','4096','--reasoning-effort','low']),('secondary',[sys.executable,'-u','secondary_gates.py','--base-url','http://127.0.0.1:30001/v1','--key-file','~/.glm_api_key','--lane',lane,'--out',str(out/(lane+'-secondary.jsonl'))]),('speed',[sys.executable,'-u','speed_probe.py','--lane',lane,'--out',str(out/(lane+'-speed.jsonl'))])]
  for phase,args in commands:
   assert (W/'CONTROL').read_text().strip()=='RUN'
   (W/'status.json').write_text(json.dumps({'lane':lane,'phase':phase,'status':'RUNNING','time':time.time()}))
   print('BEGIN',lane,phase,time.time(),flush=True)
   r=subprocess.run(args,cwd=W)
   if r.returncode:
    (W/'status.json').write_text(json.dumps({'lane':lane,'phase':phase,'status':'BLOCKED','returncode':r.returncode,'time':time.time()}));raise SystemExit(r.returncode)
   print('END',lane,phase,time.time(),flush=True)
  (W/'status.json').write_text(json.dumps({'lane':lane,'status':'COMPLETE','time':time.time()}));print('LANE_COMPLETE',lane,flush=True)
if __name__=='__main__':main()
