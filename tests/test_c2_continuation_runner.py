import json
import os
import stat
import subprocess
import sys
import time
import hashlib
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
RUNNER = RECIPE / "scripts/window_c2_continuation.py"
LAUNCHER = RECIPE / "scripts/launch-slotcache-portable.sh"
PATCH_GENERATOR = RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py"
PINNED_SOURCE = REPO_ROOT / "tests/fixtures/k2-v3-runtime-source/vllm/v1/worker/gpu_model_runner.py"
C2_EVIDENCE_FIXTURES = REPO_ROOT / "tests/fixtures/c2-prior-evidence"
C1_FIXTURE_ROOT = C2_EVIDENCE_FIXTURES / "c1"
K1_FIXTURE_ROOT = C2_EVIDENCE_FIXTURES / "k1"
INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"
C2 = "glm53-big-c2-continuation-k2"
IMAGE_ID = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
IMAGE_TAG = "vllm-glm53-uva:v0.28.0-2cf0a691"
INCUMBENT_ID = "c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea"
C1_RECEIPT_SHA = "38a650666c747d36bd40f91d77a8d73a00bb3f4330a0b43d00fe4fa0a34bb0b8"
C1_GATE_SHA = "0bf616eb2a7962ce54135065233d899c25fa6844fda8839aeb1ded52a1ca0994"
C1_ACCEPTANCE_SHA = "64271ced8af2e96d3ffcb5392794fdb6157f291233bd522ce403679d01c6b7ae"
K1_EVIDENCE_SHA = "43945bb91fa3cd0e1176d0f7e4399d4b8891e901ccebc80bdb2f4c52a0912be7"
K1_SNAPSHOT_SHA = "26419c77a278e86e6e5aafeb2289209573be45872965bf13222dab62da80f5bf"
GREEDY_SHA = "c8172e6f286d6394aca925ca969dc656fb38f724addd42425f6871036c0116d5"


def make_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def read_events(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_manifest_sha256(recipe: Path = RECIPE) -> str:
    ignored_dirs = {".git", "__pycache__", "capture", "results"}
    rows = []
    for path in sorted(recipe.rglob("*")):
        rel_path = path.relative_to(recipe)
        if any(part in ignored_dirs for part in rel_path.parts):
            continue
        if path.is_symlink():
            raise AssertionError(f"recipe manifest input must not be a symlink: {rel_path}")
        if not path.is_file():
            continue
        rel = rel_path.as_posix()
        if rel.endswith(".pyc"):
            continue
        data = path.read_bytes()
        rows.append(f"{rel}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}")
    manifest = "\n".join(rows).encode() + b"\n"
    return hashlib.sha256(manifest).hexdigest()


def generate_patched_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "gpu_model_runner.patched.py"
    result = subprocess.run([sys.executable, str(PATCH_GENERATOR), "--source", str(PINNED_SOURCE), "--output", str(runner)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return runner


def fake_python(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-python", """#!/usr/bin/env python3
import json, os, pathlib, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record(kind, argv):
    with log.open('a') as f: f.write(json.dumps({'kind': kind, 'argv': argv}) + '\\n')
args = sys.argv[1:]
if args and args[0].endswith('window_gate.py'):
    record('gate', args); print('WINDOW_GATE_OK ' + args[-1])
elif args and args[0].endswith('dflash2_acceptance_probe.py'):
    record('probe', args)
    out = pathlib.Path(args[args.index('--out') + 1]); out.parent.mkdir(parents=True, exist_ok=True)
    speed = float(os.environ.get('FAKE_C2_SPEED', '49.0'))
    rows=[]; total_tokens=0; total_steps=0; speeds=[speed-1, speed, speed, speed+1]
    for i, kind in enumerate(('prose','prose','code','code')):
        n=512; d=n/speeds[i]
        rows.append({'index':i,'kind':kind,'completion_tokens':n,'decode_seconds':d,'decode_tok_s':speeds[i],'metric_delta':{'drafts':900,'draft_tokens':1800,'accepted_tokens':1148}})
        total_tokens += n; total_steps += 900
    out.write_text(json.dumps({'schema':'glm53-dflash2-uva-acceptance-v1','model':'glm-5.3-big','max_tokens':512,'rows':rows,'summary':{'requests':4,'completion_tokens':total_tokens,'verification_steps':total_steps,'acceptance_length_weighted':total_tokens/total_steps,'decode_tok_s_median':(speeds[1]+speeds[2])/2}})+'\\n')
elif args and args[0].endswith('greedy_equiv.py') and len(args) >= 4 and args[1] == '--compare':
    record('greedy-compare', args); print('GREEDY_EQUIV identical=20/20')
elif args and args[0].endswith('greedy_equiv.py'):
    record('greedy-capture', args)
    out = pathlib.Path(args[1]); out.parent.mkdir(parents=True, exist_ok=True)
    ref = pathlib.Path(os.environ['BOUND_INCUMBENT_GREEDY'])
    out.write_text(ref.read_text())
elif args and args[0].endswith('nsys_capture_control.py'):
    record('nsys-control', args); print(f'ACK {args[2]} {args[3]}')
else:
    record('unknown-python', args); sys.exit(2)
""")


def fake_docker(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-docker", f"""#!/usr/bin/env python3
import json, os, pathlib, sys
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE']); log = pathlib.Path(os.environ['FAKE_LOG'])
def load(): return json.loads(state.read_text()) if state.exists() else {{'{INCUMBENT}': True}}
def save(s): state.write_text(json.dumps(s))
with log.open('a') as f: f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}})+'\\n')
args=sys.argv[1:]; s=load()
if args[:2] == ['inspect','-f']:
    name=args[-1]
    if name not in s: print('Error: No such object: '+name, file=sys.stderr); sys.exit(1)
    print('true' if s[name] else 'false')
elif args[:3] == ['image','inspect','--format']:
    print('{IMAGE_ID}')
elif args and args[0] == 'inspect':
    name=args[-1]; running=bool(s.get(name, False)); k=2 if name == '{C2}' else 1
    cfg='{{"method":"mtp","num_speculative_tokens":%d}}' % k
    config_image='{IMAGE_TAG}' + ('@{IMAGE_ID}' if os.environ.get('FAKE_TAG_DIGEST') == '1' and name == '{C2}' else '')
    image_field='{IMAGE_ID}'
    if name == '{C2}':
        mode=os.environ.get('FAKE_C2_CONFIG_IMAGE')
        if mode == 'digest': config_image='{IMAGE_ID}'
        elif mode == 'tag_digest': config_image='{IMAGE_TAG}@{IMAGE_ID}'
        elif mode == 'wrong_digest': config_image='sha256:0000000000000000000000000000000000000000000000000000000000000000'
        elif mode == 'wrong_tag': config_image='vllm-glm53-uva:wrong'
        elif mode == 'ambiguous': config_image='{IMAGE_TAG}@sha256:0000000000000000000000000000000000000000000000000000000000000000'
        elif mode == 'non_string': config_image={{'image':'{IMAGE_ID}'}}
        if os.environ.get('FAKE_C2_IMAGE_FIELD_WRONG') == '1': image_field='sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
    cid='{INCUMBENT_ID}' if name == '{INCUMBENT}' else 'container-'+name
    cmd=['serve','/model','--served-model-name','glm-5.3-big','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]
    if name == '{C2}' and os.environ.get('FAKE_C2_OMIT_MODEL_ARG') == '1':
        cmd=['serve','/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]
    print(json.dumps([{{'Id':cid,'Image':image_field,'Name':'/'+name,'Config':{{'Image':config_image,'Cmd':cmd}},'State':{{'Running':running}},'Args':cmd}}]))
elif args and args[0] == 'stop':
    name=args[-1]
    if name in s:
        s[name]=False; save(s)
        if name == '{INCUMBENT}' and os.environ.get('FAKE_STOP_FAIL_AFTER_MUTATION') == '1': sys.exit(8)
elif args and args[0] == 'start':
    s[args[-1]]=True; save(s)
elif args and args[0] == 'ps': print('{INCUMBENT}' if s.get('{INCUMBENT}') else '')
sys.exit(0)
""")


def fake_bash(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-bash", f"""#!/usr/bin/env python3
import json, os, pathlib, sys
cap=pathlib.Path(os.environ['CAPTURE_DIR'])
snapdir=cap / 'snapshots' / 'c2-continuation'
snapshot_host_exists_before_launch=snapdir.is_dir()
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'bash','argv':sys.argv[1:],'snapshot_host_exists_before_launch':snapshot_host_exists_before_launch,'env':{{k:os.environ.get(k) for k in ('CONTAINER_NAME','STATS_SEC','NSYS','NSYS_OUTPUT','NSYS_CONTROL','SLOT_CACHE_QUIESCENT_SNAPSHOTS','SLOT_CACHE_EXPECTED_LAYERS','SLOT_CACHE_WINDOW_STEPS','SLOT_CACHE_SNAPSHOT_DIR','SLOT_CACHE_K_MODE','SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS','MAX_MODEL_LEN','MAX_NUM_SEQS','IMAGE','SLOT_CACHE_IMAGE_SHA','SLOT_CACHE_PATCHED_RUNNER','SLOT_CACHE_PATCHED_RUNNER_SHA256','SLOT_CACHE_SOURCE_RUNNER','SLOT_CACHE_SOURCE_SHA','SLOT_CACHE_RECIPE_SHA','SLOT_CACHE_ENGINE_GENERATION','SLOT_CACHE_COUNTER_SCOPE')}}}})+'\\n')
if not snapshot_host_exists_before_launch:
    print('SLOT_CACHE_SNAPSHOT_DIR host path must already exist before launch', file=sys.stderr)
    sys.exit(2)
state=pathlib.Path(os.environ['FAKE_DOCKER_STATE']); s=json.loads(state.read_text()) if state.exists() else {{'{INCUMBENT}': False}}
s[os.environ['CONTAINER_NAME']]=True; state.write_text(json.dumps(s))
cap.mkdir(parents=True, exist_ok=True)
(cap / (os.environ['CONTAINER_NAME'] + '.nsys-rep')).write_bytes(b'REAL_NSYS_REPORT_BYTES')
snapdir.mkdir(parents=True, exist_ok=True)
rows=[]
import importlib.util
helper_path=pathlib.Path(r'{RECIPE / "patches/slot_cache_window_instrumentation.py"}')
spec=importlib.util.spec_from_file_location('slot_cache_window_instrumentation_fake_launch', helper_path)
helper=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=helper
spec.loader.exec_module(helper)
phase_flags={{'canonical_phase':'spec_verify','has_spec_verify':True}}
for endpoint, step, seq in [('start',100,2),('end',164,4)]:
    trace_id=helper.make_slot_cache_trace_id(run_id='c2-continuation', engine_generation=1, seq=seq, engine_step=step, boundary='step_complete', phase_flags=phase_flags, graph_mode='unknown', k_mode='K2', has_drafter_config=True, drafter_runs_model_forward=False)
    rows.append({{'schema':'slot-cache-quiescent-snapshot-v1','metadata':{{'valid_for_campaign':False,'window_endpoint':endpoint,'engine_step':step,'boundary':'step_complete','phase_flags':phase_flags,'graph_mode':'unknown','k_mode':'K2','trace_id':trace_id}},'provenance':{{'seq':seq,'engine_generation':1,'run_id':'c2-continuation','phase':'spec_verify','k_mode':'K2'}},'layers':{{str(i):{{'misses':i,'routes':i+10,'steps':step}} for i in range(3,78)}}}})
(snapdir / 'slot-cache-snapshots-fake.jsonl').write_text('\\n'.join(json.dumps(r) for r in rows)+'\\n')
print('launched fake c2')
""")


def fake_health(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-health", "#!/usr/bin/env python3\nprint('health ok')\n")


def fake_api_probe(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-api-probe", """#!/usr/bin/env python3
import json, os, pathlib, sys
if os.environ.get('FAKE_API_PROBE_FAIL') == '1': sys.exit(9)
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
out.joinpath('models.json').write_text(json.dumps({'data':[{'id':'glm-5.3-big'}]})+'\\n')
out.joinpath('completion.json').write_text(json.dumps({'choices':[{'message':{'content':'WINDOW_RESTORE_OK'}}]})+'\\n')
""")


def fake_nsys(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-nsys", """#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f: f.write(json.dumps({'kind':'nsys','argv':sys.argv[1:]})+'\\n')
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
for suffix in ('cuda_gpu_kern_sum','cuda_kern_exec_sum','cuda_gpu_trace','cuda_api_trace'):
    (out.parent / f'{out.name}_{suffix}.csv').write_text('Name,Total Time (ms),Instances\\nfused_bookkeeping,1,1\\n')
""")


def fake_systemd_run(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-systemd-run", """#!/usr/bin/env python3
import datetime, json, os, pathlib, sys, time
reg=pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY']); log=pathlib.Path(os.environ['FAKE_LOG'])
with log.open('a') as f: f.write(json.dumps({'kind':'systemd-run','argv':sys.argv[1:]})+'\\n')
args=sys.argv[1:]; unit=args[args.index('--unit')+1]; cal=args[args.index('--on-calendar')+1]; cmd=args[args.index('--')+1:]
next_us=int(datetime.datetime.strptime(cal,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc).timestamp()*1000000)
data={'units':{unit+'.service':{'LoadState':'loaded','ActiveState':'inactive','ExecStart':' '.join(cmd)}, unit+'.timer':{'LoadState':'loaded','ActiveState':'active','Triggers':unit+'.service','NextElapseUSecRealtime':str(next_us),'NextElapseUSecMonotonic':str(int((time.monotonic()+60)*1000000))}}}
reg.write_text(json.dumps(data)); print('Running timer as unit: '+unit+'.timer')
""")


def fake_systemctl(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-systemctl", """#!/usr/bin/env python3
import json, os, pathlib, sys
reg=pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY']); log=pathlib.Path(os.environ['FAKE_LOG'])
with log.open('a') as f: f.write(json.dumps({'kind':'systemctl','argv':sys.argv[1:]})+'\\n')
args=[a for a in sys.argv[1:] if a != '--system']; data=json.loads(reg.read_text()) if reg.exists() else {'units':{}}
if args[0] == 'show':
    unit=args[-1]; info=data['units'].get(unit, {'LoadState':'not-found','ActiveState':'inactive'})
    props=[args[i+1] for i,a in enumerate(args) if a == '-p' and i+1 < len(args)]
    for p in props: print(f'{p}={info.get(p, "")}')
    sys.exit(0 if info.get('LoadState') != 'not-found' else 1)
elif args[0] == 'stop':
    for unit in args[1:]: data['units'].setdefault(unit,{})['ActiveState']='inactive'
    reg.write_text(json.dumps(data)); sys.exit(0)
sys.exit(2)
""")


def base_cmd(tmp_path: Path, out: Path, c1_root=C1_FIXTURE_ROOT, k1_root=K1_FIXTURE_ROOT):
    root=tmp_path/'root'; root.mkdir(exist_ok=True); (root/'CONTROL').write_text('RUN\n'); (root/'RELEASE').write_text('c2-continuation-20260909\n')
    return [sys.executable, str(RUNNER), '--root', str(root), '--recipe', str(RECIPE), '--out', str(out), '--run-id', 'c2-continuation-20260909', '--c1-root', str(c1_root), '--k1-root', str(k1_root), '--docker', str(fake_docker(tmp_path)), '--bash', str(fake_bash(tmp_path)), '--python', str(fake_python(tmp_path)), '--health', str(fake_health(tmp_path)), '--api-probe', str(fake_api_probe(tmp_path)), '--nsys', str(fake_nsys(tmp_path)), '--host-operation-lock', str(tmp_path/'host.lock'), '--systemd-run', str(fake_systemd_run(tmp_path)), '--systemctl', str(fake_systemctl(tmp_path)), '--timer-python', 'python', '--command-timeout-sec', '5', '--window-deadline-sec', '20']


def run_runner(tmp_path: Path, env_extra=None, extra_args=()):
    log=tmp_path/'events.jsonl'; state=tmp_path/'docker-state.json'; state.write_text(json.dumps({INCUMBENT: True})+'\n')
    env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path/'systemd-registry.json'), BOUND_INCUMBENT_GREEDY=str(C1_FIXTURE_ROOT/'quality/incumbent-greedy.json'), **(env_extra or {}))
    result=subprocess.run(base_cmd(tmp_path, tmp_path/'receipts')+list(extra_args), env=env, text=True, capture_output=True, check=False)
    return result, log, state


def test_candidate_env_satisfies_real_launcher_pinned_image_contract(tmp_path):
    runner = load_runner_module()
    log = tmp_path / "events.jsonl"
    docker_state = tmp_path / "docker-state.json"
    docker_state.write_text(json.dumps({INCUMBENT: False}) + "\n")
    out = tmp_path / "capture"
    (out / "snapshots" / "c2-continuation").mkdir(parents=True)
    patched_runner = generate_patched_runner(tmp_path)
    args = types.SimpleNamespace(docker=str(fake_docker(tmp_path)), docker_context=None)
    env = runner.candidate_env(os.environ, out, args, patched_runner, sha256(patched_runner), PINNED_SOURCE, recipe_manifest_sha256())
    nsys_target = tmp_path / "nsys" / "target-linux-sbsa-armv8"
    nsys_target.mkdir(parents=True)
    make_executable(nsys_target / "nsys", "#!/usr/bin/env sh\nexit 0\n")
    api_key = tmp_path / ".glm_api_key"
    api_key.write_text("synthetic-secret\n")
    env.update({
        "FAKE_LOG": str(log),
        "FAKE_DOCKER_STATE": str(docker_state),
        "API_KEY_FILE": str(api_key),
        "NSYS_INSTALL_DIR": str(tmp_path / "nsys"),
    })

    result = subprocess.run(["bash", str(LAUNCHER), "c2-continuation", "112", "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}'], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    docker_run = next(e for e in read_events(log) if e["kind"] == "docker" and e["argv"][:2] == ["run", "-d"])
    assert IMAGE_ID in docker_run["argv"]
    assert any(v == f"SLOT_CACHE_IMAGE_SHA={IMAGE_ID}" for i, v in enumerate(docker_run["argv"]) if i and docker_run["argv"][i - 1] == "-e")


def test_c2_continuation_creates_host_snapshot_dir_before_launch(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    launch = next(e for e in read_events(log) if e["kind"] == "bash")
    assert launch["snapshot_host_exists_before_launch"] is True
    assert launch["env"]["SLOT_CACHE_SNAPSHOT_DIR"] == "/wcap/snapshots/c2-continuation"


def test_c2_continuation_binds_c1_k1_artifacts_launches_only_k2_and_restores(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    kinds = [e['kind'] for e in events]
    assert kinds.index('systemd-run') < next(i for i,e in enumerate(events) if e.get('argv') == ['stop', INCUMBENT])
    launches=[e for e in events if e['kind']=='bash']
    assert len(launches) == 1
    launch=launches[0]
    assert launch['env']['CONTAINER_NAME'] == C2
    assert launch['env']['STATS_SEC'] == '0'
    assert launch['env']['NSYS'] == '1'
    assert launch['env']['SLOT_CACHE_K_MODE'] == 'K2'
    assert launch['env']['SLOT_CACHE_EXPECTED_LAYERS'] == '75'
    assert launch['argv'][-2:] == ['--speculative-config', '{"method":"mtp","num_speculative_tokens":2}']
    assert json.dumps(events).count('num_speculative_tokens\\":2') == 1
    docker_calls=[e['argv'] for e in events if e['kind']=='docker']
    assert ['stop','-t','120', C2] in docker_calls[:docker_calls.index(['start', INCUMBENT])]
    assert json.loads(state.read_text())[INCUMBENT] is True
    receipt=json.loads((tmp_path/'receipts'/'c2-continuation-receipt.json').read_text())
    assert receipt['promotion_authorized'] is False
    assert receipt['c1_binding']['receipt_sha256'] == C1_RECEIPT_SHA
    assert receipt['c1_binding']['gate_sha256'] == C1_GATE_SHA
    assert receipt['c1_binding']['acceptance_sha256'] == C1_ACCEPTANCE_SHA
    assert receipt['c1_binding']['incumbent_greedy_sha256'] == GREEDY_SHA
    assert receipt['k1_binding']['evidence_sha256'] == K1_EVIDENCE_SHA
    assert receipt['k1_binding']['snapshot_sha256'] == K1_SNAPSHOT_SHA
    assert receipt['c1_binding']['c2_authorized'] is False
    assert receipt['c1_binding']['gate']['pass'] is True
    assert receipt['measurement']['g2_pass'] is True
    assert receipt['measurement']['g3_pass'] is True
    assert receipt['snapshot']['layer_count'] == 75
    assert receipt['snapshot']['endpoints'] == ['start', 'end']
    assert receipt['nvtx_correlation']['status'] == 'UNPROVEN'
    nsys_i = next(i for i,e in enumerate(events) if e['kind'] == 'nsys')
    restore_i = next(i for i,e in enumerate(events) if e.get('argv') == ['start', INCUMBENT])
    cancel_i = max(i for i,e in enumerate(events) if e['kind']=='systemctl' and 'stop' in e['argv'])
    assert restore_i < nsys_i < cancel_i


def test_c2_continuation_refuses_nonempty_output_before_any_action(tmp_path):
    out = tmp_path/'receipts'; out.mkdir(); (out/'old').write_text('x')
    log=tmp_path/'events.jsonl'; state=tmp_path/'docker-state.json'; state.write_text(json.dumps({INCUMBENT: True})+'\n')
    env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path/'systemd-registry.json'), BOUND_INCUMBENT_GREEDY=str(C1_FIXTURE_ROOT/'quality/incumbent-greedy.json'))
    result=subprocess.run(base_cmd(tmp_path, out), env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'output directory is not empty' in result.stderr
    assert not log.exists()


def test_c2_continuation_rejects_bad_c1_hash_before_timer_or_stop(tmp_path):
    bad_c1 = tmp_path/'bad-c1'; subprocess.run(['cp','-R',str(C1_FIXTURE_ROOT), str(bad_c1)], check=True)
    (bad_c1/'c1-gate.json').write_text('{"pass": true}\n')
    out=tmp_path/'receipts'; log=tmp_path/'events.jsonl'; state=tmp_path/'docker-state.json'; state.write_text(json.dumps({INCUMBENT: True})+'\n')
    env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path/'systemd-registry.json'), BOUND_INCUMBENT_GREEDY=str(bad_c1/'quality/incumbent-greedy.json'))
    result=subprocess.run(base_cmd(tmp_path, out, c1_root=bad_c1), env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'C1 gate hash mismatch' in result.stderr
    events = read_events(log) if log.exists() else []
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']


def test_c2_continuation_does_not_cancel_timer_when_restore_proof_fails(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_STOP_FAIL_AFTER_MUTATION':'1', 'FAKE_API_PROBE_FAIL':'1'})
    assert result.returncode != 0
    events=read_events(log)
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind']=='docker']
    assert not any(e['kind']=='systemctl' and 'stop' in e['argv'] for e in events)


def load_runner_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('window_c2_continuation_under_test', RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_instrumentation_helper():
    import importlib.util
    helper = RECIPE / 'patches/slot_cache_window_instrumentation.py'
    spec = importlib.util.spec_from_file_location('slot_cache_window_instrumentation_under_test', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def real_c2_trace_id(*, seq: int, step: int, phase_flags=None, graph_mode='unknown', k_mode='K2') -> str:
    helper = load_instrumentation_helper()
    return helper.make_slot_cache_trace_id(
        run_id='c2-continuation',
        engine_generation=1,
        seq=seq,
        engine_step=step,
        boundary='step_complete',
        phase_flags=phase_flags or {'canonical_phase': 'spec_verify', 'has_spec_verify': True},
        graph_mode=graph_mode,
        k_mode=k_mode,
        has_drafter_config=True,
        drafter_runs_model_forward=False,
    )


def write_snapshot(out: Path, *, trace_mutator=None, seqs=(2, 4), layer_mutator=None, phase='spec_verify', k_mode='K2', graph_mode='unknown', phase_flags=None):
    snapdir = out / 'snapshots' / 'c2-continuation'
    snapdir.mkdir(parents=True)
    rows = []
    for endpoint, step, seq in [('start', 100, seqs[0]), ('end', 164, seqs[1])]:
        layers = {str(i): {'misses': i, 'routes': i + 10, 'steps': step} for i in range(3, 78)}
        if layer_mutator:
            layer_mutator(endpoint, layers)
        trace_id = real_c2_trace_id(seq=seq, step=step, phase_flags=phase_flags, graph_mode=graph_mode, k_mode=k_mode)
        if trace_mutator:
            trace_id = trace_mutator(trace_id)
        rows.append({
            'schema': 'slot-cache-quiescent-snapshot-v1',
            'metadata': {
                'valid_for_campaign': False,
                'window_endpoint': endpoint,
                'engine_step': step,
                'boundary': 'step_complete',
                'phase_flags': phase_flags or {'canonical_phase': 'spec_verify', 'has_spec_verify': True},
                'graph_mode': graph_mode,
                'k_mode': k_mode,
                'trace_id': trace_id,
            },
            'provenance': {'seq': seq, 'engine_generation': 1, 'run_id': 'c2-continuation', 'phase': phase, 'k_mode': k_mode},
            'layers': layers,
        })
    (snapdir / 'slot-cache-snapshots-real.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


def test_candidate_identity_fails_closed_when_model_arg_absent(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_C2_OMIT_MODEL_ARG': '1'})
    assert result.returncode != 0
    assert 'candidate runtime shape mismatch' in result.stderr
    failure = json.loads((tmp_path / 'receipts' / 'failure.json').read_text())
    assert failure['code'] == 'CANDIDATE_IDENTITY_FAILED'


def test_candidate_identity_accepts_digest_only_config_image_from_real_launcher_shape(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_C2_CONFIG_IMAGE': 'digest'})
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / 'receipts' / 'c2-continuation-receipt.json').read_text())
    proof = receipt['candidate_proof']
    assert proof['observed_image_digest'] == IMAGE_ID
    assert proof['observed_config_image'] == IMAGE_ID


def test_candidate_identity_rejects_bad_config_images_and_wrong_image_field(tmp_path):
    cases = [
        ({'FAKE_C2_CONFIG_IMAGE': 'wrong_digest'}, 'candidate image/source mismatch'),
        ({'FAKE_C2_CONFIG_IMAGE': 'wrong_tag'}, 'candidate image/source mismatch'),
        ({'FAKE_C2_CONFIG_IMAGE': 'ambiguous'}, 'candidate image/source mismatch'),
        ({'FAKE_C2_CONFIG_IMAGE': 'non_string'}, 'candidate image/source mismatch'),
        ({'FAKE_C2_CONFIG_IMAGE': 'digest', 'FAKE_C2_IMAGE_FIELD_WRONG': '1'}, 'candidate image/source mismatch'),
    ]
    for index, (env_extra, message) in enumerate(cases):
        case_dir = tmp_path / f'case-{index}'
        case_dir.mkdir()
        result, log, state = run_runner(case_dir, env_extra=env_extra)
        assert result.returncode != 0, env_extra
        assert message in result.stderr, result.stderr
        failure = json.loads((case_dir / 'receipts' / 'failure.json').read_text())
        assert failure['code'] == 'CANDIDATE_IDENTITY_FAILED'


def test_snapshot_accepts_real_instrumentation_helper_trace_ids(tmp_path):
    runner = load_runner_module()
    write_snapshot(tmp_path)

    snapshot = runner.parse_snapshot(tmp_path)

    assert snapshot['sequences'] == [2, 4]
    assert snapshot['trace_ids'] == [
        real_c2_trace_id(seq=2, step=100),
        real_c2_trace_id(seq=4, step=164),
    ]


def test_snapshot_rejects_malformed_or_wrong_extended_trace_contract(tmp_path):
    runner = load_runner_module()
    cases = [
        ('simplified', lambda trace_id: trace_id.split(':phase:', 1)[0]),
        ('extra', lambda trace_id: trace_id + ':provenance:fake'),
        ('wrong-phase', lambda trace_id: trace_id.replace(':phase:spec_verify:', ':phase:decode:')),
        ('wrong-flags', lambda trace_id: trace_id.replace(':flags:spec_verify,drafter:', ':flags:drafter,spec_verify:')),
        ('wrong-graph', lambda trace_id: trace_id.replace(':graph:unknown:', ':graph:FULL:')),
        ('wrong-k', lambda trace_id: trace_id.replace(':k:K2', ':k:K1')),
        ('wrong-seq', lambda trace_id: trace_id.replace(':seq:2:', ':seq:3:')),
        ('wrong-step', lambda trace_id: trace_id.replace(':step:100:', ':step:101:')),
    ]
    for name, mutate in cases:
        case_dir = tmp_path / name
        write_snapshot(case_dir, trace_mutator=mutate)
        try:
            runner.parse_snapshot(case_dir)
        except Exception as exc:
            assert getattr(exc, 'code', '') == 'SNAPSHOT_INVALID', name
            assert 'trace id' in str(exc), name
        else:
            raise AssertionError(f'{name} trace id accepted')


def test_snapshot_rejects_fake_trace_suffix_wrong_sequences_and_negative_deltas(tmp_path):
    runner = load_runner_module()
    bad_suffix = tmp_path / 'bad-suffix'
    write_snapshot(bad_suffix, trace_mutator=lambda trace_id: trace_id + ':fake')
    try:
        runner.parse_snapshot(bad_suffix)
    except Exception as exc:
        assert getattr(exc, 'code', '') == 'SNAPSHOT_INVALID'
    else:
        raise AssertionError('fake trace suffix accepted')

    bad_seq = tmp_path / 'bad-seq'
    write_snapshot(bad_seq, seqs=(9, 10))
    try:
        runner.parse_snapshot(bad_seq)
    except Exception as exc:
        assert getattr(exc, 'code', '') == 'SNAPSHOT_INVALID'
    else:
        raise AssertionError('wrong provenance sequences accepted')

    bad_delta = tmp_path / 'bad-delta'
    def regress(endpoint, layers):
        if endpoint == 'end':
            layers['3']['misses'] = 0
    write_snapshot(bad_delta, layer_mutator=regress)
    try:
        runner.parse_snapshot(bad_delta)
    except Exception as exc:
        assert getattr(exc, 'code', '') == 'SNAPSHOT_INVALID'
    else:
        raise AssertionError('negative layer delta accepted')


def test_measurement_rejects_nan_infinity_and_nonpositive_speed():
    runner = load_runner_module()
    ref = {str(i): 'ok' for i in range(20)}
    for speed in ('NaN', 'Infinity', '-Infinity', '0', '-1'):
        try:
            runner.evaluate_measurement({'summary': {'decode_tok_s_median': speed}}, ref, ref)
        except Exception as exc:
            assert getattr(exc, 'code', '') == 'MEASUREMENT_INVALID'
        else:
            raise AssertionError(f'invalid speed accepted: {speed}')


def test_profiler_wraps_real_192_token_workload_and_keeps_512_acceptance(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    python_events = [(i, e) for i, e in enumerate(events) if e['kind'] in {'probe', 'nsys-control', 'greedy-capture'}]
    start_i = next(i for i, e in python_events if e['kind'] == 'nsys-control' and e['argv'][2] == 'START')
    stop_i = next(i for i, e in python_events if e['kind'] == 'nsys-control' and e['argv'][2] == 'STOP')
    profiled = [e for i, e in python_events if start_i < i < stop_i and e['kind'] == 'probe']
    assert len(profiled) == 1
    assert profiled[0]['argv'][profiled[0]['argv'].index('--max-tokens') + 1] == '192'
    assert profiled[0]['argv'][profiled[0]['argv'].index('--model') + 1] == 'glm-5.3-big'
    acceptance = [e for i, e in python_events if e['kind'] == 'probe' and e not in profiled]
    assert any(e['argv'][e['argv'].index('--max-tokens') + 1] == '512' for e in acceptance)
    assert all(not (start_i < i < stop_i) for i, e in python_events if e['kind'] == 'greedy-capture')


def test_launch_binds_exact_pinned_instrumented_runner_artifacts(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    launch = next(e for e in events if e['kind'] == 'bash')
    env = launch['env']
    assert env['SLOT_CACHE_PATCHED_RUNNER']
    assert Path(env['SLOT_CACHE_PATCHED_RUNNER']).is_file()
    assert env['SLOT_CACHE_PATCHED_RUNNER_SHA256'] == '2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4'
    assert env['SLOT_CACHE_SOURCE_SHA'] == '7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa'
    assert env['SLOT_CACHE_SOURCE_RUNNER'].endswith('sources/vllm/v1/worker/gpu_model_runner.py')
    receipt = json.loads((tmp_path / 'receipts' / 'c2-continuation-receipt.json').read_text())
    source = receipt['source']
    assert source['source_runner_sha256'] == env['SLOT_CACHE_SOURCE_SHA']
    assert source['patched_runner_sha256'] == env['SLOT_CACHE_PATCHED_RUNNER_SHA256']
    assert source['patch_generator_sha256'] == '8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b'
