"""Drive the production CLI to an actual fake-executable boundary. No GPU calls."""
import json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import controller as c

class NativeCliGateTests(unittest.TestCase):
    def test_real_cli_refuses_foreign_occupancy_before_start_or_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); release=root/'release'; release.mkdir(); fakebin=root/'bin';fakebin.mkdir()
            core=['controller/controller.py','replay/replay_matched.py','replay/analyze_runs.py',
                  'gauntlet/gauntlet.py','gauntlet/oracle.py','gauntlet/run_suite.py']
            source=Path(c.__file__).resolve().parents[1]
            for name in core:
                target=release/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source/name,target)
            (release/'CONTROL').write_text('RUN\n');(release/'RELEASE').write_text('native-cli-probe\n')
            fixture=root/'synthetic-fixture.json';fixture.write_text('[]')
            image='sha256:'+'0'*64
            contract=json.loads((source/'container-contract.json').read_text())
            contract.update({'fixture_sha256':c.sha256_hex(fixture.read_bytes()),'model':'SYNTHETIC',
                             'context':1048576,'sandbox_image':image,'profiles':{}})
            for label in ('v14','v13'):
                contract['profiles'][label]={'id':'id-'+label,'name':'/name-'+label,
                    'image':image,'runtime_sha256':'0'*64}
            (release/'contract.json').write_text(json.dumps(contract))
            files={p.relative_to(release).as_posix():c.sha256_hex(p.read_bytes()) for p in release.rglob('*') if p.is_file()}
            (release/'manifest.json').write_text(json.dumps({'files':files}))
            script='#!'+sys.executable+'\n'+'''import json,os,sys
name=os.path.basename(sys.argv[0]);args=sys.argv[1:]
with open(os.environ['FAKE_CALL_LOG'],'a') as f:f.write(json.dumps([name]+args)+'\\n')
if name=='systemctl':
 if args[0]=='is-active':print('active')
 else:print('{ path=/usr/bin/docker ; argv[]=/usr/bin/docker stop --time 30 id-v14 id-v13 ; ignore_errors=no ; }')
elif name=='docker' and args[0]=='ps':print('{"ID":"foreign","Names":"not-owned"}')
else:sys.exit(9)
'''
            for name in ('docker','systemctl'):
                p=fakebin/name;p.write_text(script);p.chmod(0o755)
            env={**os.environ,'PATH':str(fakebin)+os.pathsep+os.environ.get('PATH',''),
                 'FAKE_CALL_LOG':str(root/'calls.jsonl')}
            env.pop('ROUND7_SANDBOX_IMAGE',None)
            args=[sys.executable,str(release/'controller/controller.py'),'--release',str(release),
                  '--contract',str(release/'contract.json'),'--fixture',str(fixture),'--output',str(root/'out'),
                  '--run-id','native-cli-probe','--guard-unit','synthetic-hardstop']
            result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(1,result.returncode,result.stdout+result.stderr)
            self.assertIn('running',result.stdout+result.stderr)
            commands=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertTrue(any(x[:2]==['docker','ps'] for x in commands),commands)
            self.assertFalse(any(x[:2] in (['docker','start'],['docker','stop']) for x in commands),commands)

if __name__=='__main__':unittest.main()
