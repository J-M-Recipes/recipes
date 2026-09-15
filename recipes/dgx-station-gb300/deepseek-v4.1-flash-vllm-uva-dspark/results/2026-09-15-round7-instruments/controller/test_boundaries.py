import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller as c

class WarmupBoundaryTests(unittest.TestCase):
    def test_loads_actual_replay_module(self):
        module=c._load_replay_module(Path(c.__file__).parent.parent)
        self.assertTrue(callable(module.stream_chat_completion))

class ConcurrentWarmupTests(unittest.TestCase):
    def test_c_labels_represent_real_concurrent_requests(self):
        import threading,time,types
        # Derive the consumer fixture from the production SSE decoder.
        import io
        wire=c._load_replay_module(Path(__file__).resolve().parents[1])
        packet=b'data: {"choices":[{"delta":{"content":"synthetic"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\ndata: [DONE]\n\n'
        response=io.BytesIO(packet);response.status=200
        with patch('urllib.request.OpenerDirector.open',return_value=response):
            native=wire.stream_chat_completion('http://127.0.0.1:9',b'{}',timeout_s=5)
        active=0; peak=0; lock=threading.Lock(); calls=[]
        def transport(url,body,timeout_s):
            nonlocal active,peak
            payload=json.loads(body)
            with lock: active+=1; peak=max(peak,active); calls.append(payload)
            time.sleep(.02)
            with lock: active-=1
            return dict(native)
        mod=types.SimpleNamespace(TOOLS=[],stream_chat_completion=transport,canonical_json_bytes=c.canonical_json_bytes)
        with tempfile.TemporaryDirectory() as d,patch.object(c,'_load_replay_module',return_value=mod):
            out=c.run_prompt_warmups(Path(d),'synthetic','http://127.0.0.1:1/v1','synthetic-salt',Path(d))
        self.assertEqual(4,peak)
        self.assertEqual(sum(range(1,5))*3,len(calls))
        self.assertEqual(len(calls),out['count'])
        self.assertTrue(all(x['wall_s']>0 for x in out['records']))

class CliBoundaryTests(unittest.TestCase):
    def test_real_program_exposes_required_cli(self):
        result=subprocess.run([sys.executable,c.__file__,'--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(0,result.returncode,result.stderr)
        for flag in ('--release','--contract','--fixture','--output','--run-id','--guard-unit','--validate-only'):
            self.assertIn(flag,result.stdout)

class PartialStartTests(unittest.TestCase):
    def test_ambiguous_start_is_stopped(self):
        from test_controller import FakeRuntime,_docker_doc,_profile_from_doc
        class Runtime(FakeRuntime):
            def run(self,argv,timeout_s=None):
                result=super().run(argv,timeout_s)
                if argv[:2]==['docker','start']:
                    raise TimeoutError('synthetic lost start acknowledgement')
                return result
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);release=root/'release';release.mkdir()
            (release/'CONTROL').write_text('RUN');(release/'RELEASE').write_text('test-run')
            doc=_docker_doc();runtime=Runtime([doc])
            contract={'profiles':{'v14':_profile_from_doc(doc)}}
            campaign=c.CampaignController(contract,release,root/'fixture',root/'out','test-run','guard',runtime,None,lambda:True)
            with self.assertRaises(TimeoutError):campaign.run_one_boot(1,'v14')
            self.assertFalse(runtime.started,'ambiguous docker start escaped cleanup')
            self.assertIn(['docker','stop','--time','30',doc['Id']],runtime.commands)

class LockBoundaryTests(unittest.TestCase):
    def test_two_releases_share_one_campaign_lock(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'release-a';b=Path(d)/'release-b';a.mkdir();b.mkdir()
            with c.ReleaseLock(a):
                with self.assertRaises(c.ControllerError):
                    with c.ReleaseLock(b):pass

class ManifestBindingTests(unittest.TestCase):
    def test_unhashed_contract_cannot_authorize_release(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rel=root/'release';rel.mkdir();fixture=root/'fixture';fixture.write_bytes(b'[]')
            files={}
            for name in ('controller/controller.py','replay/replay_matched.py','replay/analyze_runs.py','gauntlet/gauntlet.py','gauntlet/oracle.py','gauntlet/run_suite.py'):
                p=rel/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'pass\n');files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            for name,value in [('CONTROL','RUN'),('RELEASE','test-run')]:
                (rel/name).write_text(value);files[name]=hashlib.sha256(value.encode()).hexdigest()
            contract=rel/'container-contract.json';contract.write_text(json.dumps({'fixture_sha256':hashlib.sha256(b'[]').hexdigest()}))
            manifest=rel/'manifest.json';manifest.write_text(json.dumps({'files':files}))
            with self.assertRaisesRegex(c.ControllerError,'contract.*manifest'):
                c.verify_release_inputs(rel,contract,fixture,root/'out','test-run')
            files[contract.name]=hashlib.sha256(contract.read_bytes()).hexdigest();manifest.write_text(json.dumps({'files':files}))
            c.verify_release_inputs(rel,contract,fixture,root/'out','test-run')
            contract.write_text(contract.read_text()+' ')
            with self.assertRaisesRegex(c.ControllerError,'manifest hash mismatch'):
                c.verify_release_inputs(rel,contract,fixture,root/'out','test-run')

class SandboxBindingTests(unittest.TestCase):
    def test_override_cannot_replace_reviewed_sandbox_image(self):
        fixed='sha256:'+'0'*64;other='sha256:'+'1'*64
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(c.ControllerError):
                c.resolve_sandbox_image({'sandbox_image':fixed},Path(d),other)
            self.assertEqual(fixed,c.resolve_sandbox_image({'sandbox_image':fixed},Path(d)))

class ExactReleaseTests(unittest.TestCase):
    def test_opaque_release_token_is_not_trimmed_into_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'CONTROL').write_text('RUN\n');(root/'RELEASE').write_text(' test-run \n')
            with self.assertRaises(c.ControllerError):c.read_control(root,'test-run')
            (root/'RELEASE').write_text('test-run\n');c.read_control(root,'test-run')

if __name__=='__main__': unittest.main()
