import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
RUNNER = RECIPE / "scripts/window_c1_quality_gate.py"
INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"
CANDIDATE = "glm53-big-c1-quality-k1"
IMAGE_ID = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
IMAGE_TAG = "vllm-glm53-uva:v0.28.0-2cf0a691"
INCUMBENT_ID = "c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea"


def make_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def read_events(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wait_for_path(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def assert_process_exited(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} was still alive")


def valid_acceptance(max_tokens=512, speed=46.0):
    rows = []
    total_tokens = 0
    total_steps = 0
    speeds = [speed - 1, speed, speed, speed + 1]
    for i, kind in enumerate(("prose", "prose", "code", "code")):
        n = max_tokens
        duration = n / speeds[i]
        rows.append({
            "index": i,
            "kind": kind,
            "completion_tokens": n,
            "decode_seconds": duration,
            "decode_tok_s": speeds[i],
            "metric_delta": {"drafts": n, "draft_tokens": n, "accepted_tokens": n},
        })
        total_tokens += n
        total_steps += n
    return {
        "schema": "glm53-dflash2-uva-acceptance-v1",
        "model": "glm-5.3-big",
        "max_tokens": max_tokens,
        "rows": rows,
        "summary": {
            "requests": 4,
            "completion_tokens": total_tokens,
            "verification_steps": total_steps,
            "acceptance_length_weighted": total_tokens / total_steps,
            "decode_tok_s_median": (speeds[1] + speeds[2]) / 2,
        },
    }


def fake_python(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-python", """#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record(kind, argv, extra=None):
    with log.open('a') as f:
        payload = {'kind': kind, 'argv': argv, 'env': {k: os.environ.get(k) for k in ('BASE_URL','MODEL_NAME','MODEL','API_KEY','API_KEY_FILE','HEALTH_RETRIES','RETRY_SLEEP')}}
        if extra: payload.update(extra)
        f.write(json.dumps(payload) + '\\n')
args = sys.argv[1:]
if args and args[0].endswith('window_gate.py'):
    record('gate', args); print('WINDOW_GATE_OK ' + args[-1])
elif args and args[0].endswith('dflash2_acceptance_probe.py'):
    record('probe', args)
    out = pathlib.Path(args[args.index('--out') + 1]); max_tokens = int(args[args.index('--max-tokens') + 1])
    speed = float(os.environ.get('FAKE_MEDIAN_TOK_S', '46.0'))
    rows=[]; total_tokens=0; total_steps=0; speeds=[speed-1, speed, speed, speed+1]
    for i, kind in enumerate(('prose','prose','code','code')):
        n=max_tokens; duration=n/speeds[i]
        rows.append({'index':i,'kind':kind,'completion_tokens':n,'decode_seconds':duration,'decode_tok_s':speeds[i],'metric_delta':{'drafts':n,'draft_tokens':n,'accepted_tokens':n}})
        total_tokens += n; total_steps += n
    payload={'schema':'glm53-dflash2-uva-acceptance-v1','model':'glm-5.3-big','max_tokens':max_tokens,'rows':rows,'summary':{'requests':4,'completion_tokens':total_tokens,'verification_steps':total_steps,'acceptance_length_weighted':total_tokens/total_steps,'decode_tok_s_median':(speeds[1]+speeds[2])/2}}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload)+'\\n')
elif args and args[0].endswith('greedy_equiv.py') and len(args) >= 4 and args[1] == '--compare':
    record('greedy-compare', args)
    print('GREEDY_EQUIV identical=19/20' if os.environ.get('FAKE_GREEDY_MISMATCH') == '1' else 'GREEDY_EQUIV identical=20/20')
elif args and args[0].endswith('greedy_equiv.py'):
    record('greedy-capture', args)
    out = pathlib.Path(args[1]); out.parent.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get('FAKE_GREEDY_BAD')
    payload = {str(i): 'same' for i in range(20)}
    if mode == 'empty': payload['3'] = ''
    if mode == 'missing': payload.pop('19')
    if os.environ.get('FAKE_GREEDY_MISMATCH') == '1' and 'c1-greedy' in str(out): payload['7'] = 'different'
    out.write_text(json.dumps(payload)+'\\n'); print('GREEDY saved %d outputs to %s' % (len(payload), out))
else:
    record('unknown-python', args); sys.exit(2)
""")


def fake_docker(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-docker", f"""#!/usr/bin/env python3
import json, os, pathlib, sys
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE']); log = pathlib.Path(os.environ['FAKE_LOG'])
def load():
    if state.exists(): return json.loads(state.read_text())
    return {{'{INCUMBENT}': True}}
def save(s): state.write_text(json.dumps(s))
with log.open('a') as f: f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}})+'\\n')
args = sys.argv[1:]; s = load()
if args[:2] == ['inspect','-f']:
    name = args[-1]
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'ambiguous':
        print('Internal error for API_KEY=secret-token-1234567890', file=sys.stderr); sys.exit(42)
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'sensitive':
        print('bad API_KEY=stdout-secret-token Authorization: Bearer bearer-secret-token')
        print('bad VLLM_API_KEY=stderr-secret-token', file=sys.stderr); sys.exit(47)
    if name not in s:
        print('Error: No such object: ' + name, file=sys.stderr); sys.exit(1)
    print('true' if s[name] else 'false')
elif args[:3] == ['image','inspect','--format']:
    print('{IMAGE_ID}')
elif args and args[0] == 'inspect':
    name = args[-1]; running = bool(s.get(name, False)); k = 1
    if os.environ.get('FAKE_CANDIDATE_WRONG_K') == '1' and name == '{CANDIDATE}': k = 2
    model = 'WRONG-MODEL' if os.environ.get('FAKE_INCUMBENT_WRONG_MODEL') == '1' and name == '{INCUMBENT}' else 'glm-5.3-big'
    config_image = '{IMAGE_TAG}'
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_QUALIFIED_IMAGE') == '1': config_image += '@{IMAGE_ID}'
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_WRONG_QUALIFIED_IMAGE') == '1': config_image += '@sha256:' + '0' * 64
    cfg = '{{"method":"mtp","num_speculative_tokens":%d}}' % k
    cid = '{INCUMBENT_ID}' if name == '{INCUMBENT}' else 'container-' + name
    print(json.dumps([{{'Id':cid,'Image':'{IMAGE_ID}','Name':'/'+name,'Config':{{'Image':config_image,'Cmd':['/model','--served-model-name',model,'--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}},'State':{{'Running':running}},'Args':['/model','--served-model-name',model,'--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}}]))
elif args and args[0] == 'stop':
    name = args[-1]
    if name in s:
        s[name] = False; save(s)
        if name == '{INCUMBENT}' and os.environ.get('FAKE_STOP_FAIL_AFTER_MUTATION') == '1': sys.exit(8)
elif args and args[0] == 'start':
    if args[-1] == '{INCUMBENT}' and os.environ.get('FAKE_RESTORE_START_MARKER'):
        pathlib.Path(os.environ['FAKE_RESTORE_START_MARKER']).write_text('started')
        import time; time.sleep(2)
    s[args[-1]] = True; save(s)
elif args and args[0] == 'ps':
    print('{INCUMBENT}' if s.get('{INCUMBENT}') else '')
sys.exit(0)
""")


def fake_bash(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-bash", f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'bash','argv':sys.argv[1:], 'env': {{k: os.environ.get(k) for k in ('NSYS','IMAGE','DOCKER','STATS_SEC','CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL','MAX_MODEL_LEN','MAX_NUM_SEQS','SLOT_CACHE_PER_LAYER','AT_KEY','COMPILATION_CONFIG')}}}})+'\\n')
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
s = json.loads(state.read_text()) if state.exists() else {{'{INCUMBENT}': False}}
s[os.environ['CONTAINER_NAME']] = True; state.write_text(json.dumps(s))
print('launched fake ' + os.environ['CONTAINER_NAME'])
""")


def fake_health(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-health", """#!/usr/bin/env python3
import json, os, pathlib
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'health','env': {k: os.environ.get(k) for k in ('HEALTH_RETRIES','RETRY_SLEEP','BASE_URL','MODEL_NAME')}})+'\\n')
print('health ok')
""")


def fake_api_probe(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-api-probe", """#!/usr/bin/env python3
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
if os.environ.get('FAKE_API_PROBE_FAIL') == '1': sys.exit(9)
(out / 'models.json').write_text(json.dumps({'data':[{'id':'glm-5.3-big'}]})+'\\n')
(out / 'completion.json').write_text(json.dumps({'choices':[{'message':{'content':'WINDOW_RESTORE_OK'}}]})+'\\n')
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f: f.write(json.dumps({'kind':'api-probe','argv':sys.argv[1:]})+'\\n')
""")


def fake_systemd_run(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-systemd-run", """#!/usr/bin/env python3
import datetime, json, os, pathlib, subprocess, sys, time
registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY']); log = pathlib.Path(os.environ['FAKE_LOG'])
def load(): return json.loads(registry.read_text()) if registry.exists() else {'units': {}}
def save(data): registry.parent.mkdir(parents=True, exist_ok=True); tmp=registry.with_suffix('.tmp'); tmp.write_text(json.dumps(data, sort_keys=True)); os.replace(tmp, registry)
with log.open('a') as f: f.write(json.dumps({'kind':'systemd-run','argv':sys.argv[1:]})+'\\n')
args=sys.argv[1:]; unit=args[args.index('--unit')+1]; on_calendar=args[args.index('--on-calendar')+1]; cmd=args[args.index('--')+1:]
service=unit+'.service'; timer=unit+'.timer'; next_realtime=int(datetime.datetime.strptime(on_calendar, '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc).timestamp()*1000000)
delay=max(0.0, next_realtime/1000000-time.time()); exec_cmd=list(cmd)
if os.environ.get('FAKE_SYSTEMD_SWAP_EXECSTART') == '1': exec_cmd=[cmd[0], cmd[1], '--restore-only', '--out', '/tmp/bad', '--docker', cmd[cmd.index('--docker')+1], '--host-operation-lock', '/tmp/bad.lock']
if os.environ.get('FAKE_SYSTEMD_WRONG_ELAPSE_USEC'): next_realtime += int(os.environ['FAKE_SYSTEMD_WRONG_ELAPSE_USEC'])
data=load(); data['units'][service]={'LoadState':'loaded','ActiveState':'inactive','ExecStart':' '.join(exec_cmd)}; data['units'][timer]={'LoadState':'loaded','ActiveState':'active','Triggers':service,'NextElapseUSecRealtime':str(next_realtime),'NextElapseUSecMonotonic':str(int((time.monotonic()+delay)*1000000))}; save(data)
code = '''
import json, os, pathlib, subprocess, time
registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY']); unit=%r; service=unit+'.service'; timer=unit+'.timer'; delay=%r; cmd=%r
def load(): return json.loads(registry.read_text())
def save(data):
    tmp=registry.with_suffix('.tmp'); tmp.write_text(json.dumps(data, sort_keys=True)); os.replace(tmp, registry)
time.sleep(delay); data=load()
if data.get('units',{}).get(timer,{}).get('ActiveState') != 'active': raise SystemExit(0)
data['units'][timer]['ActiveState']='elapsed'; data['units'][service]['ActiveState']='activating'; save(data)
result=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
pathlib.Path(os.environ['FAKE_SYSTEMD_RESTORE_LOG']).write_text(result.stdout or '')
data=load(); data['units'][service]['ActiveState']='failed' if result.returncode else 'inactive'; data['units'][service]['ExecMainStatus']=str(result.returncode); save(data)
''' % (unit, delay, cmd)
subprocess.Popen([sys.executable, '-c', code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=open(str(registry)+'.daemon.err','w'), env=os.environ.copy())
print('Running timer as unit: '+timer)
""")


def fake_systemctl(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-systemctl", """#!/usr/bin/env python3
import json, os, pathlib, sys
registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY']); log = pathlib.Path(os.environ['FAKE_LOG'])
def load(): return json.loads(registry.read_text()) if registry.exists() else {'units': {}}
def save(data): tmp=registry.with_suffix('.tmp'); tmp.write_text(json.dumps(data, sort_keys=True)); os.replace(tmp, registry)
with log.open('a') as f: f.write(json.dumps({'kind':'systemctl','argv':sys.argv[1:]})+'\\n')
args=[a for a in sys.argv[1:] if a != '--system']; data=load(); units=data.get('units', {})
if args and args[0] == 'show':
    props=[]; unit=args[-1]
    for i,arg in enumerate(args):
        if arg == '-p' and i+1 < len(args): props.append(args[i+1])
        elif arg.startswith('-p') and arg != '-p': props.append(arg[2:])
    info=units.get(unit)
    if info is None:
        print('LoadState=not-found'); print('ActiveState=inactive'); sys.exit(1)
    for prop in props or sorted(info): print(f'{prop}={info.get(prop, "")}')
elif args and args[0] == 'stop':
    for unit in args[1:]: units.setdefault(unit, {})['ActiveState']='inactive'
    save(data)
else:
    print('unsupported', file=sys.stderr); sys.exit(2)
""")


def base_cmd(tmp_path: Path, out: Path):
    root = tmp_path / "root"; root.mkdir(exist_ok=True); (root / "CONTROL").write_text("RUN\n"); (root / "RELEASE").write_text("c1-quality-20260909\n")
    return [sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "c1-quality-20260909", "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)), "--host-operation-lock", str(tmp_path / "host.lock"), "--systemd-run", str(fake_systemd_run(tmp_path)), "--systemctl", str(fake_systemctl(tmp_path)), "--timer-python", "python", "--command-timeout-sec", "5", "--window-deadline-sec", "20"]


def run_runner(tmp_path: Path, env_extra=None, extra_args=()):
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"
    state.write_text(json.dumps({INCUMBENT: True}) + "\n")
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / "systemd-registry.json"), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / "systemd-restore.log"), **(env_extra or {}))
    result = subprocess.run(base_cmd(tmp_path, tmp_path / "receipts") + list(extra_args), env=env, text=True, capture_output=True, check=False)
    return result, log, state


def test_c1_quality_gate_orders_incumbent_greedy_and_timer_before_stop_then_restores_before_cancel(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    kinds = [e['kind'] for e in events]
    inc_greedy_i = next(i for i, e in enumerate(events) if e['kind'] == 'greedy-capture' and 'incumbent-greedy' in e['argv'][-1])
    systemd_i = kinds.index('systemd-run')
    stop_inc_i = next(i for i, e in enumerate(events) if e.get('argv') == ['stop', INCUMBENT])
    assert inc_greedy_i < systemd_i < stop_inc_i
    restore_i = next(i for i, e in enumerate(events) if e.get('argv') == ['start', INCUMBENT])
    cancel_i = max(i for i, e in enumerate(events) if e['kind'] == 'systemctl' and 'stop' in e['argv'])
    assert restore_i < cancel_i
    assert json.loads(state.read_text())[INCUMBENT] is True
    receipt = json.loads((tmp_path / 'receipts' / 'c1-quality-receipt.json').read_text())
    assert receipt['schema'] == 'glm53-c1-quality-gate-receipt-v1'
    assert receipt['verdict'] == 'PASS'
    assert receipt['promotion_authorized'] is False
    assert receipt['c2_authorized'] is False
    assert receipt['run_id'] == 'c1-quality-20260909'
    source_manifest = json.loads((tmp_path / 'receipts' / 'source-manifest.json').read_text())
    archived_paths = {item['path'] for item in source_manifest['files']}
    assert 'scripts/window_gate.py' in archived_paths
    timer = json.loads((tmp_path / 'receipts' / 'preflight' / 'restore-timer-armed-readback.json').read_text())
    timer_bundle = Path(timer['timer_restore_bundle']['path'])
    restore_cmd = timer['restore_cmd']
    health_path = Path(restore_cmd[restore_cmd.index('--health') + 1])
    assert health_path == timer_bundle / 'scripts' / 'health-check.sh'
    assert health_path.is_file()
    assert not health_path.is_symlink()


def test_c1_quality_gate_uses_k1_stats_sec_zero_and_acceptance_512_only(tmp_path):
    result, log, _ = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    launches = [e for e in events if e['kind'] == 'bash']
    assert len(launches) == 1
    launch = launches[0]
    assert launch['env']['CONTAINER_NAME'] == CANDIDATE
    assert launch['env']['STATS_SEC'] == '0'
    assert launch['env']['MAX_MODEL_LEN'] == '524288'
    assert launch['env']['MAX_NUM_SEQS'] == '1'
    assert launch['argv'][-2:] == ['--speculative-config', '{"method":"mtp","num_speculative_tokens":1}']
    probes = [e['argv'] for e in events if e['kind'] == 'probe']
    assert len(probes) == 1
    assert probes[0][probes[0].index('--max-tokens') + 1] == '512'
    assert 'bench' not in [e['kind'] for e in events]


def test_c1_quality_gate_wrong_incumbent_model_fails_before_timer_and_stop(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_INCUMBENT_WRONG_MODEL': '1'})
    assert result.returncode != 0
    assert json.loads(state.read_text())[INCUMBENT] is True
    events = read_events(log)
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert not any(e['kind'] == 'docker' and e['argv'][:2] == ['stop', INCUMBENT] for e in events)


def test_c1_quality_gate_accepts_only_exact_digest_qualified_candidate_image(tmp_path):
    accepted_path = tmp_path / 'accepted'; accepted_path.mkdir()
    accepted, _, _ = run_runner(accepted_path, env_extra={'FAKE_CANDIDATE_QUALIFIED_IMAGE': '1'})
    assert accepted.returncode == 0, accepted.stderr
    rejected_path = tmp_path / 'rejected'; rejected_path.mkdir()
    rejected, _, state = run_runner(rejected_path, env_extra={'FAKE_CANDIDATE_WRONG_QUALIFIED_IMAGE': '1'})
    assert rejected.returncode != 0
    assert json.loads(state.read_text())[INCUMBENT] is True


def test_c1_quality_gate_rejects_empty_missing_or_incomplete_greedy(tmp_path):
    for mode in ('empty', 'missing'):
        case = tmp_path / mode; case.mkdir()
        result, log, state = run_runner(case, env_extra={'FAKE_GREEDY_BAD': mode})
        assert result.returncode != 0
        assert 'quality' in result.stderr or 'greedy' in result.stderr
        events = read_events(log)
        assert not any(e['kind'] == 'bash' for e in events)
        assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_c1_quality_gate_19_of_20_fails_even_when_compare_exits_zero_and_restores(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_GREEDY_MISMATCH': '1'})
    assert result.returncode != 0
    assert 'C1 gate failed' in result.stderr or 'quality_pass' in result.stderr
    events = read_events(log)
    assert 'greedy-compare' in [e['kind'] for e in events]
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert json.loads(state.read_text())[INCUMBENT] is True


def test_c1_quality_gate_does_not_cancel_failsafe_when_restore_proof_fails(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={
        'FAKE_GREEDY_MISMATCH': '1',
        'FAKE_API_PROBE_FAIL': '1',
    }, extra_args=('--window-deadline-sec', '60'))
    assert result.returncode != 0
    assert json.loads(state.read_text())[INCUMBENT] is True
    events = read_events(log)
    assert not any(e['kind'] == 'systemctl' and 'stop' in e['argv'] for e in events)
    registry = json.loads((tmp_path / 'systemd-registry.json').read_text())
    timer = next(info for name, info in registry['units'].items() if name.endswith('.timer'))
    assert timer['ActiveState'] == 'active'


def test_c1_quality_gate_execstart_and_deadline_swap_rejected_before_stop(tmp_path):
    for env in ({'FAKE_SYSTEMD_SWAP_EXECSTART': '1'}, {'FAKE_SYSTEMD_WRONG_ELAPSE_USEC': '5000000'}):
        case = tmp_path / next(iter(env)); case.mkdir()
        result, log, state = run_runner(case, env_extra=env)
        assert result.returncode != 0
        assert 'ExecStart mismatch' in result.stderr or 'deadline mismatch' in result.stderr
        assert ['stop', INCUMBENT] not in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
        if state.exists(): assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_c1_quality_gate_stop_mutates_then_fails_restores_and_does_not_launch(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_STOP_FAIL_AFTER_MUTATION': '1'})
    assert result.returncode != 0
    events = read_events(log)
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'bash' for e in events)
    assert json.loads(state.read_text())[INCUMBENT] is True


def test_c1_quality_restore_only_is_independent_and_uses_no_release_gate(tmp_path):
    log = tmp_path / 'events.jsonl'; state = tmp_path / 'docker-state.json'; out = tmp_path / 'restore-only'
    result = subprocess.run([sys.executable, str(RUNNER), '--restore-only', '--out', str(out), '--docker', str(fake_docker(tmp_path)), '--health', str(fake_health(tmp_path)), '--api-probe', str(fake_api_probe(tmp_path)), '--host-operation-lock', str(tmp_path / 'host.lock')], env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state)), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert not any(e['kind'] == 'gate' for e in events)
    assert not any(e['kind'] == 'bash' for e in events)
    assert ['stop', '-t', '120', CANDIDATE] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']


def test_c1_quality_gate_ambiguous_candidate_inspect_redacts_and_fails_before_timer(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_CANDIDATE_INSPECT_FAILURE': 'sensitive'})
    assert result.returncode != 0
    assert 'candidate container inspect failed before launch' in result.stderr
    assert 'stdout-secret-token' not in result.stderr
    assert 'stderr-secret-token' not in result.stderr
    events = read_events(log)
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_c1_quality_gate_kills_active_child_group_on_sigterm_and_restores(tmp_path):
    active = tmp_path / 'active.json'
    hanging_bash = make_executable(tmp_path / 'hanging-bash', f"""#!/usr/bin/env python3
import json, os, pathlib, time
pathlib.Path({str(active)!r}).write_text(json.dumps({{'pid': os.getpid(), 'pgid': os.getpgrp()}}))
while True: time.sleep(1)
""")
    log = tmp_path / 'events.jsonl'; state = tmp_path / 'docker-state.json'; out = tmp_path / 'receipts'
    cmd = base_cmd(tmp_path, out); cmd[cmd.index('--bash') + 1] = str(hanging_bash)
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / 'systemd-registry.json'), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / 'systemd-restore.log'))
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(active)
    child_pid = json.loads(active.read_text())['pid']
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode != 0, stdout + stderr
    assert_process_exited(child_pid)
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
