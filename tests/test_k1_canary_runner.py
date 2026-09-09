import json
import os
import signal
import stat
import subprocess
import importlib.util
import runpy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
RUNNER = RECIPE / "scripts/window_k1_canary.py"
INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"
CANDIDATE = "glm53-big-k1-canary-instrumented"
ADAPTER = RECIPE / "patches/slot_cache_window_instrumentation.py"
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


def fake_docker(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
log = pathlib.Path(os.environ['FAKE_LOG'])
def load():
    if state.exists():
        return json.loads(state.read_text())
    return {{'{INCUMBENT}': True}}
def save(s): state.write_text(json.dumps(s))
def record():
    with log.open('a') as f:
        f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
record()
args = sys.argv[1:]
s = load()
if args[:2] == ['inspect', '-f']:
    name = args[-1]
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'permission':
        print('permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock', file=sys.stderr)
        sys.exit(1)
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'ambiguous':
        print('Error response from daemon: request returned Internal Server Error for API route and version', file=sys.stderr)
        sys.exit(1)
    if name == '{CANDIDATE}' and os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'sensitive':
        print('stdout diagnostic API_KEY=stdout-secret Authorization: Bearer stdout-token')
        print('stderr diagnostic VLLM_API_KEY=stderr-secret Authorization: Bearer stderr-token', file=sys.stderr)
        sys.exit(47)
    if name not in s:
        if os.environ.get('FAKE_CANDIDATE_INSPECT_FAILURE') == 'ubuntu-lowercase-absent':
            print('error: no such object: ' + name, file=sys.stderr)
        else:
            print('Error: No such object: ' + name, file=sys.stderr)
        sys.exit(1)
    print('true' if s[name] else 'false')
elif args[:3] == ['image', 'inspect', '--format']:
    print('{IMAGE_ID}')
elif args and args[0] == 'inspect':
    name = args[-1]
    running = bool(s.get(name, False))
    cfg = '{{"method":"mtp","num_speculative_tokens":1}}'
    container_id = '{INCUMBENT_ID}' if name == '{INCUMBENT}' else 'container-'+name
    print(json.dumps([{{'Id':container_id,'Image':'{IMAGE_ID}','Name':'/'+name,'Config':{{'Image':'{IMAGE_TAG}','Cmd':['/model','--served-model-name','glm-5.3-big','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}},'State':{{'Running': running}},'Args':['/model','--served-model-name','glm-5.3-big','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}}]))
elif args and args[0] == 'stop':
    name = args[-1]
    if name in s:
        s[name] = False
        save(s)
        if name == '{INCUMBENT}' and os.environ.get('FAKE_STOP_FAIL_AFTER_MUTATION') == '1':
            sys.exit(8)
elif args and args[0] == 'start':
    if args[-1] == '{INCUMBENT}' and os.environ.get('FAKE_RESTORE_START_MARKER'):
        pathlib.Path(os.environ['FAKE_RESTORE_START_MARKER']).write_text('started')
        import time; time.sleep(2)
    s[args[-1]] = True
    save(s)
elif args and args[0] == 'logs':
    print('candidate logs')
elif args and args[0] == 'ps':
    print('{INCUMBENT}' if s.get('{INCUMBENT}') else '')
sys.exit(0)
""",
    )


def fake_bash(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-bash",
        f"""#!/usr/bin/env python3
import importlib.util, json, os, pathlib, sys, types
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'bash','argv':sys.argv[1:], 'env': {{k: os.environ.get(k) for k in ('NSYS','IMAGE','DOCKER','STATS_SEC','CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL','SLOT_CACHE_QUIESCENT_SNAPSHOTS','SLOT_CACHE_PATCHED_RUNNER','SLOT_CACHE_PATCHED_RUNNER_SHA256','SLOT_CACHE_SOURCE_RUNNER','SLOT_CACHE_SOURCE_SHA','SLOT_CACHE_RECIPE_SHA','SLOT_CACHE_IMAGE_SHA','SLOT_CACHE_RUN_ID','SLOT_CACHE_ENGINE_GENERATION','SLOT_CACHE_WINDOW_STEPS','SLOT_CACHE_SNAPSHOT_DIR','SLOT_CACHE_K_MODE','SLOT_CACHE_EXPECTED_LAYERS','SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS')}}}}) + '\\n')
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
s = json.loads(state.read_text()) if state.exists() else {{'{INCUMBENT}': False}}
name = os.environ['CONTAINER_NAME']
if name in s:
    print('docker: Error response from daemon: Conflict. The container name /' + name + ' is already in use by container stale123. You have to remove (or rename) that container to be able to reuse that name.', file=sys.stderr)
    sys.exit(125)
s[name] = True
state.write_text(json.dumps(s))
cap = pathlib.Path(os.environ['CAPTURE_DIR']); cap.mkdir(parents=True, exist_ok=True)
(pathlib.Path(os.environ['SLOT_CACHE_PATCHED_RUNNER'])).read_text()
(cap / 'k1-canary.nsys-rep').write_bytes(b'REAL_NSYS_REPORT_BYTES')
snap = cap / 'snapshots' / 'k1-canary'; snap.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('slot_cache_window_instrumentation', pathlib.Path(sys.argv[1]).parents[1] / 'patches/slot_cache_window_instrumentation.py')
inst = importlib.util.module_from_spec(spec); sys.modules['slot_cache_window_instrumentation'] = inst; spec.loader.exec_module(inst)
class Hook:
    @staticmethod
    def slot_cache_snapshot_device(**kwargs):
        return types.SimpleNamespace(trace_id=kwargs['trace_id'], provenance=kwargs['provenance'], metadata=kwargs['metadata'], layer_ids=list(range(3,78)), counters_device=[[i, i+10, kwargs['metadata']['engine_step']] for i in range(3,78)])
sys.modules['slot_cache_hook'] = Hook
class Cached:
    req_ids=['d']
    def is_context_phase(self, req_id): return False
def sched(step):
    return types.SimpleNamespace(engine_step=step, total_num_scheduled_tokens=1, num_scheduled_tokens={{'d':1}}, scheduled_cached_reqs=Cached(), scheduled_new_reqs=[], scheduled_spec_decode_tokens={{}})
ctl = inst.SlotCacheWindowController(enabled=True, snapshot_dir=snap, windows=(inst.SlotCacheWindow(100,164),), run_id=os.environ['SLOT_CACHE_RUN_ID'], source_sha=os.environ['SLOT_CACHE_SOURCE_SHA'], engine_generation=int(os.environ['SLOT_CACHE_ENGINE_GENERATION']), expected_layers=int(os.environ['SLOT_CACHE_EXPECTED_LAYERS']), k_mode=os.environ['SLOT_CACHE_K_MODE'], recipe_sha=os.environ['SLOT_CACHE_RECIPE_SHA'], image_sha=os.environ['SLOT_CACHE_IMAGE_SHA'], counter_scope=os.environ['SLOT_CACHE_COUNTER_SCOPE'], target_forward_snapshots=True)
for step in (100, 164):
    scheduler_output = sched(step)
    ctl.mark_target_forward_boundary(scheduler_output=scheduler_output)
    meta = ctl.make_metadata(scheduler_output=scheduler_output, boundary='step_complete')
    snap_obj = ctl.maybe_snapshot_device(metadata=meta)
    inst.finalize_slot_cache_snapshot(ctl, snap_obj, counters_cpu=[[i, i+10, step] for i in range(3,78)])
print('launched fake canary with adapter snapshots')
""",
    )


def fake_python(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-python",
        """#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record(kind, argv):
    with log.open('a') as f:
        f.write(json.dumps({'kind': kind, 'argv': argv}) + '\\n')
args = sys.argv[1:]
if args and args[0].endswith('window_gate.py'):
    record('gate', args); print('WINDOW_GATE_OK ' + args[-1])
elif args and args[0].endswith('dflash2_acceptance_probe.py'):
    record('probe', args)
    out = pathlib.Path(args[args.index('--out') + 1]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'schema':'glm53-dflash2-uva-acceptance-v1','summary': {'verification_steps':192},'rows':[{'completion_tokens':192}]}) + '\\n')
elif args and args[0].endswith('nsys_capture_control.py'):
    record('nsys-control', args); print(f'ACK {args[2]} {args[3]}')
else:
    record('unknown-python', args); sys.exit(2)
""",
    )


def fake_health(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "fake-health", "#!/usr/bin/env python3\nprint('health ok')\n")


def fake_api_probe(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-api-probe",
        """#!/usr/bin/env python3
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / 'models.json').write_text(json.dumps({'data':[{'id':'glm-5.3-big'}]}) + '\\n')
(out / 'completion.json').write_text(json.dumps({'choices':[{'message':{'content':'WINDOW_RESTORE_OK'}}]}) + '\\n')
""",
    )


def fake_nsys(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-nsys",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'nsys','argv':sys.argv[1:]}) + '\\n')
report = pathlib.Path(sys.argv[-1])
if report.read_bytes() != b'REAL_NSYS_REPORT_BYTES':
    print('bad report', file=sys.stderr); sys.exit(8)
out_prefix = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
(out_prefix.parent / f'{out_prefix.name}_cuda_gpu_trace.csv').write_text('Name,Start (ns),End (ns)\\ntarget_forward_kernel,10,20\\n')
(out_prefix.parent / f'{out_prefix.name}_cuda_api_trace.csv').write_text('Name,Start (ns),End (ns),NVTX Range\\ncudaLaunchKernel,10,20,unproven-unparsed\\n')
""",
    )




def fake_systemd_run(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-systemd-run",
        """#!/usr/bin/env python3
import datetime, json, os, pathlib, subprocess, sys, time
registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY'])
log = pathlib.Path(os.environ['FAKE_LOG'])
def load():
    if registry.exists():
        return json.loads(registry.read_text())
    return {'units': {}}
def save(data):
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, sort_keys=True))
    os.replace(tmp, registry)
with log.open('a') as f:
    f.write(json.dumps({'kind':'systemd-run','argv':sys.argv[1:]}) + '\\n')
args = sys.argv[1:]
unit = args[args.index('--unit') + 1]
on_calendar = args[args.index('--on-calendar') + 1]
cmd_start = args.index('--') + 1 if '--' in args else next(i for i,a in enumerate(args) if not a.startswith('-') and i > args.index('--unit') + 1)
cmd = args[cmd_start:]
service = unit + '.service'
timer = unit + '.timer'
now = time.time()
next_realtime = int(datetime.datetime.strptime(on_calendar, '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc).timestamp() * 1000000)
delay = max(0.0, next_realtime / 1000000 - now)
data = load()
exec_cmd = list(cmd)
if os.environ.get('FAKE_SYSTEMD_SWAP_EXECSTART') == '1':
    exec_cmd = [cmd[0], cmd[1], '--restore-only', '--out', '/tmp/adversarial-restore-out', '--docker', cmd[cmd.index('--docker') + 1], '--host-operation-lock', '/tmp/adversarial.lock.timer']
if os.environ.get('FAKE_SYSTEMD_WRONG_ELAPSE_USEC'):
    next_realtime += int(os.environ['FAKE_SYSTEMD_WRONG_ELAPSE_USEC'])
next_monotonic = int((time.monotonic() + delay) * 1000000)
data['units'][service] = {'LoadState':'loaded','ActiveState':'inactive','ExecStart':' '.join(exec_cmd)}
data['units'][timer] = {'LoadState':'loaded','ActiveState':'active','Triggers':service,'NextElapseUSecRealtime':str(next_realtime),'NextElapseUSecMonotonic':str(next_monotonic)}
save(data)
code = (
    "import json, os, pathlib, subprocess, time\\n"
    "registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY'])\\n"
    + f"unit = {unit!r}; service = unit + '.service'; timer = unit + '.timer'; delay = {delay!r}; cmd = {cmd!r}\\n"
    "def load(): return json.loads(registry.read_text())\\n"
    "def save(data):\\n"
    "    tmp = registry.with_suffix('.tmp')\\n"
    "    tmp.write_text(json.dumps(data, sort_keys=True))\\n"
    "    os.replace(tmp, registry)\\n"
    "time.sleep(delay)\\n"
    "data = load()\\n"
    "if data.get('units', {}).get(timer, {}).get('ActiveState') != 'active':\\n"
    "    raise SystemExit(0)\\n"
    "data['units'][timer]['ActiveState'] = 'elapsed'\\n"
    "data['units'][service]['ActiveState'] = 'activating'\\n"
    "save(data)\\n"
    "result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())\\n"
    "(pathlib.Path(os.environ['FAKE_SYSTEMD_RESTORE_LOG'])).write_text(result.stdout or '')\\n"
    "data = load()\\n"
    "data['units'][service]['ActiveState'] = 'failed' if result.returncode else 'inactive'\\n"
    "data['units'][service]['ExecMainStatus'] = str(result.returncode)\\n"
    "save(data)\\n"
)
subprocess.Popen([sys.executable, '-c', code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=open(str(registry) + '.daemon.err', 'w'), env=os.environ.copy())
print('Running timer as unit: ' + timer)
""",
    )


def fake_systemctl(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-systemctl",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
registry = pathlib.Path(os.environ['FAKE_SYSTEMD_REGISTRY'])
log = pathlib.Path(os.environ['FAKE_LOG'])
def load():
    if registry.exists(): return json.loads(registry.read_text())
    return {'units': {}}
def save(data):
    tmp = registry.with_suffix('.tmp'); tmp.write_text(json.dumps(data, sort_keys=True)); os.replace(tmp, registry)
with log.open('a') as f:
    f.write(json.dumps({'kind':'systemctl','argv':sys.argv[1:]}) + '\\n')
args = [a for a in sys.argv[1:] if a != '--system']
data = load(); units = data.get('units', {})
if args and args[0] == 'show':
    props = []
    unit = args[-1]
    for arg in args[1:-1]:
        if arg.startswith('-p'):
            if arg == '-p': continue
            props.append(arg[2:])
    # also handle '-p', 'Name' pairs
    for i, arg in enumerate(args):
        if arg == '-p' and i + 1 < len(args): props.append(args[i + 1])
    info = units.get(unit)
    if info is None:
        print('LoadState=not-found')
        print('ActiveState=inactive')
        if os.environ.get('FAKE_SYSTEMD_NOT_FOUND_SHOW_EXIT_ZERO') == '1':
            sys.exit(0)
        sys.exit(1)
    for prop in props or sorted(info):
        print(f'{prop}={info.get(prop, "") }')
elif args and args[0] == 'stop':
    for unit in args[1:]:
        if unit not in units:
            sys.exit(4)
        units[unit]['ActiveState'] = 'inactive'
        if os.environ.get('FAKE_SYSTEMD_COLLECT_SERVICE_ON_TIMER_STOP') == '1' and unit.endswith('.timer'):
            units.pop(unit[:-6] + '.service', None)
    save(data)
elif args and args[0] == 'reset-failed':
    sys.exit(0)
else:
    print('unsupported systemctl ' + ' '.join(args), file=sys.stderr); sys.exit(2)
""",
    )

def base_cmd(tmp_path: Path, out: Path):
    root = tmp_path / "root"; root.mkdir(exist_ok=True)
    (root / "CONTROL").write_text("RUN\n")
    (root / "RELEASE").write_text("k1-canary-20260908\n")
    return [sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "k1-canary-20260908", "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--host-operation-lock", str(tmp_path / "host.lock"), "--systemd-run", str(fake_systemd_run(tmp_path)), "--systemctl", str(fake_systemctl(tmp_path)), "--timer-python", "python"]


def run_runner(tmp_path: Path, env_extra=None, extra_args=()):
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / "systemd-registry.json"), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / "systemd-restore.log"), **(env_extra or {}))
    result = subprocess.run(base_cmd(tmp_path, tmp_path / "receipts") + list(extra_args), env=env, text=True, capture_output=True, check=False)
    return result, log, state



def write_docker_state(path: Path, *, candidate_running: bool) -> None:
    path.write_text(json.dumps({INCUMBENT: True, CANDIDATE: candidate_running}))


def test_k1_canary_rejects_stopped_candidate_name_owner_before_arming_or_incumbent_stop(tmp_path):
    state = tmp_path / "docker-state.json"
    write_docker_state(state, candidate_running=False)
    result, log, state = run_runner(tmp_path)
    assert result.returncode != 0
    expected = (
        f"candidate container name is already owned before launch: {CANDIDATE}; "
        f"safe cleanup prerequisite: verify the incumbent is running, then manually stop the candidate if running and remove or rename the stale candidate container "
        f"(for example: docker stop {CANDIDATE} || true; docker rm {CANDIDATE}) before rerunning"
    )
    assert expected in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert json.loads(state.read_text())[INCUMBENT] is True

def test_k1_canary_rejects_running_candidate_name_owner_before_arming_or_incumbent_stop(tmp_path):
    state = tmp_path / "docker-state.json"
    write_docker_state(state, candidate_running=True)
    result, log, state = run_runner(tmp_path)
    assert result.returncode != 0
    expected = (
        f"candidate container name is already owned before launch: {CANDIDATE}; "
        f"safe cleanup prerequisite: verify the incumbent is running, then manually stop the candidate if running and remove or rename the stale candidate container "
        f"(for example: docker stop {CANDIDATE} || true; docker rm {CANDIDATE}) before rerunning"
    )
    assert expected in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert json.loads(state.read_text())[INCUMBENT] is True


def test_k1_canary_rejects_candidate_inspect_permission_denied_before_arming_or_incumbent_stop(tmp_path):
    (tmp_path / "docker-state.json").write_text(json.dumps({INCUMBENT: True}))
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_CANDIDATE_INSPECT_FAILURE': 'permission'})
    assert result.returncode != 0
    assert 'candidate container inspect failed before launch' in result.stderr
    assert 'permission denied' in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_k1_canary_rejects_ambiguous_candidate_inspect_failure_before_arming_or_incumbent_stop(tmp_path):
    (tmp_path / "docker-state.json").write_text(json.dumps({INCUMBENT: True}))
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_CANDIDATE_INSPECT_FAILURE': 'ambiguous'})
    assert result.returncode != 0
    assert 'candidate container inspect failed before launch' in result.stderr
    assert 'Internal Server Error' in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_k1_canary_redacts_supported_secret_shapes_from_candidate_inspect_failure(tmp_path):
    (tmp_path / "docker-state.json").write_text(json.dumps({INCUMBENT: True}))
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_CANDIDATE_INSPECT_FAILURE': 'sensitive'})

    assert result.returncode != 0
    assert 'candidate container inspect failed before launch' in result.stderr
    assert 'returncode=47' in result.stderr
    assert 'stdout diagnostic API_KEY=<redacted> Authorization: Bearer <redacted>' in result.stderr
    assert 'stderr diagnostic VLLM_API_KEY=<redacted> Authorization: Bearer <redacted>' in result.stderr
    for secret in ('stdout-secret', 'stdout-token', 'stderr-secret', 'stderr-token'):
        assert secret not in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'systemd-run' for e in events)
    assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_k1_canary_snapshot_validator_rejects_noncanonical_layer_keys():
    namespace = runpy.run_path(str(RUNNER))
    require_all75 = namespace['_require_all75_layers']
    failure = namespace['CanaryFailed']
    canonical = {str(layer): {'misses': 0, 'routes': 0, 'steps': 0} for layer in range(3, 78)}
    assert sorted(map(int, require_all75({'layers': canonical}))) == list(range(3, 78))

    for bad_key in ('03', '٣', '３'):
        malformed = dict(canonical)
        malformed[bad_key] = malformed.pop('3')
        try:
            require_all75({'layers': malformed})
        except failure:
            pass
        else:
            raise AssertionError(f'noncanonical layer key accepted: {bad_key!r}')


def test_k1_canary_vertical_fake_executable_restores_before_nsys_export_and_attests_snapshot_nvtx(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert json.loads(state.read_text())[INCUMBENT] is True
    docker_calls = [e['argv'] for e in events if e['kind'] == 'docker']
    stop_inc = docker_calls.index(['stop', INCUMBENT])
    start_inc = docker_calls.index(['start', INCUMBENT])
    candidate_stop = docker_calls.index(['stop', '-t', '120', CANDIDATE])
    assert candidate_stop < start_inc
    nsys_export = next(i for i, e in enumerate(events) if e['kind'] == 'nsys')
    event_start_inc = next(i for i, e in enumerate(events) if e.get('argv') == ['start', INCUMBENT])
    assert event_start_inc < nsys_export
    out = tmp_path / 'receipts'
    restore_obligation = json.loads((out / 'preflight' / 'restore-obligation.json').read_text())
    assert restore_obligation['restore_required_before_stop'] is True
    timer = json.loads((out / 'preflight' / 'restore-timer-armed-readback.json').read_text())
    assert timer['status'] == 'armed'
    assert timer['pid'] > 0
    assert timer['deadline_monotonic_ns'] == restore_obligation['timer']['window_deadline_monotonic_ns']
    assert restore_obligation['timer']['system_timer_state'].startswith('systemd:')
    launch = next(e for e in events if e['kind'] == 'bash')
    assert launch['argv'][-2:] == ['--speculative-config', '{"method":"mtp","num_speculative_tokens":1}']
    assert launch['env']['NSYS'] == '1'
    assert launch['env']['STATS_SEC'] == '0'
    assert launch['env']['SLOT_CACHE_QUIESCENT_SNAPSHOTS'] == '1'
    assert launch['env']['SLOT_CACHE_K_MODE'] == 'K1'
    assert launch['env']['SLOT_CACHE_EXPECTED_LAYERS'] == '75'
    assert launch['env']['SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS'] == '1'
    assert launch['env']['SLOT_CACHE_SNAPSHOT_DIR'] == '/wcap/snapshots/k1-canary'
    assert Path(launch['env']['SLOT_CACHE_PATCHED_RUNNER']).is_file()
    evidence = json.loads((out / 'k1-canary-evidence.json').read_text())
    assert evidence['campaign_valid'] is False
    assert evidence['snapshot']['window'] == {'start': 100, 'end': 164}
    assert evidence['snapshot']['endpoints'] == ['start', 'end']
    assert evidence['snapshot']['sequences'] == [2, 4]
    assert evidence['snapshot']['engine_generation'] == 1
    assert evidence['snapshot']['layer_count'] == 75
    assert all(t.startswith('slotcache:k1-canary:gen:1:seq:') for t in evidence['snapshot']['trace_ids'])
    assert evidence['snapshot']['layer_deltas']['3']['steps'] == 64
    assert evidence['snapshot']['layer_deltas']['77']['steps'] == 64
    assert evidence['nvtx_correlation']['status'] == 'UNPROVEN'
    assert evidence['nvtx_correlation']['raw_nsys_report_nonempty'] is True
    assert evidence['source']['incumbent_container_id'] == INCUMBENT_ID
    assert evidence['source']['image_id'] == IMAGE_ID
    manifest_paths = {item['path'] for item in json.loads((out / 'source-manifest.json').read_text())['files']}
    assert 'scripts/window_k1_canary.py' in manifest_paths
    assert 'scripts/window_e1_v2.py' in manifest_paths
    assert 'scripts/k1_canary_restore_timer.py' in manifest_paths
    assert 'patches/slot_cache_stats.py' in manifest_paths
    assert 'patches/slot_cache_profile_control.py' in manifest_paths
    assert 'results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py' in manifest_paths


def test_k1_canary_accepts_exact_ubuntu_lowercase_absent_candidate(tmp_path):
    result, log, state = run_runner(
        tmp_path,
        env_extra={'FAKE_CANDIDATE_INSPECT_FAILURE': 'ubuntu-lowercase-absent'},
    )

    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert any(e['kind'] == 'systemd-run' for e in events)
    assert any(
        e['kind'] == 'docker' and e['argv'] == ['stop', INCUMBENT]
        for e in events
    )
    assert json.loads(state.read_text())[INCUMBENT] is True


def test_k1_canary_accepts_transient_service_collected_during_timer_cancellation(tmp_path):
    result, log, state = run_runner(
        tmp_path,
        env_extra={
            'FAKE_SYSTEMD_COLLECT_SERVICE_ON_TIMER_STOP': '1',
            'FAKE_SYSTEMD_NOT_FOUND_SHOW_EXIT_ZERO': '1',
        },
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(state.read_text())[INCUMBENT] is True
    assert (tmp_path / 'receipts' / 'k1-canary-evidence.json').is_file()


def test_k1_canary_absolute_timer_deadline_fires_independent_archived_restore(tmp_path):
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"; out = tmp_path / "timer-out"
    root = tmp_path / "root"; root.mkdir(); (root / "CONTROL").write_text("RUN\n"); (root / "RELEASE").write_text("k1-canary-20260908\n")
    result = subprocess.run([sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "k1-canary-20260908", "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--host-operation-lock", str(tmp_path / "host.lock"), "--systemd-run", str(fake_systemd_run(tmp_path)), "--systemctl", str(fake_systemctl(tmp_path)), "--timer-python", "python", "--window-deadline-sec", "1.0", "--command-timeout-sec", "5"], env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / "systemd-registry.json"), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / "systemd-restore.log"), FAKE_STOP_FAIL_AFTER_MUTATION='1'), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    state_file = tmp_path / 'systemd-registry.json'
    wait_for_path(state_file)
    units = json.loads(state_file.read_text())['units']
    timer = next(v for k, v in units.items() if k.endswith('.timer'))
    assert timer['ActiveState'] in {'inactive', 'elapsed'}
    assert json.loads(state.read_text())[INCUMBENT] is True

def test_k1_canary_restores_if_incumbent_stop_mutates_then_fails_before_launch(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_STOP_FAIL_AFTER_MUTATION': '1'})
    assert result.returncode != 0
    assert json.loads(state.read_text())[INCUMBENT] is True
    events = read_events(log)
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'bash' for e in events)


def test_k1_canary_restore_only_ignores_release_gate_and_only_restores(tmp_path):
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"; out = tmp_path / "restore-only"
    result = subprocess.run([sys.executable, str(RUNNER), "--restore-only", "--out", str(out), "--docker", str(fake_docker(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)), "--host-operation-lock", str(tmp_path / "host.lock"), "--systemd-run", str(fake_systemd_run(tmp_path)), "--systemctl", str(fake_systemctl(tmp_path)), "--timer-python", "python"], env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state)), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert not any(e['kind'] == 'gate' for e in events)
    assert not any(e['kind'] == 'bash' for e in events)
    assert ['stop', '-t', '120', CANDIDATE] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']


def test_k1_canary_terminates_active_child_group_on_sigterm_and_restores(tmp_path):
    active = tmp_path / "active.json"
    hanging_bash = make_executable(tmp_path / "hanging-bash", f"""#!/usr/bin/env python3
import json, os, pathlib, time
pathlib.Path({str(active)!r}).write_text(json.dumps({{'pid': os.getpid(), 'pgid': os.getpgrp()}}))
while True: time.sleep(1)
""")
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"; out = tmp_path / "receipts"
    cmd = base_cmd(tmp_path, out); cmd[cmd.index('--bash') + 1] = str(hanging_bash)
    proc = subprocess.Popen(cmd, env=dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / "systemd-registry.json"), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / "systemd-restore.log")), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(active)
    child_pid = json.loads(active.read_text())['pid']
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode != 0, stdout + stderr
    assert_process_exited(child_pid)
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']



def test_k1_canary_arms_system_scope_systemd_restore_before_stop_with_execstart_readback(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / 'receipts'
    readback = json.loads((out / 'preflight' / 'restore-timer-armed-readback.json').read_text())
    assert readback['schema'] == 'glm53-k1-canary-systemd-restore-timer-readback-v1'
    assert readback['scope'] == 'system'
    assert readback['timer_unit'].endswith('.timer')
    assert readback['service_unit'].endswith('.service')
    assert readback['timer']['LoadState'] == 'loaded'
    assert readback['timer']['ActiveState'] == 'active'
    assert readback['timer']['Triggers'] == readback['service_unit']
    assert 'ExecStart' in readback['service']
    exec_start = readback['service']['ExecStart']
    assert str(out / 'restore-bundle' / 'scripts' / 'window_k1_canary.py') in exec_start
    assert '--restore-only' in exec_start
    assert '/Users/' not in exec_start
    obligation = json.loads((out / 'preflight' / 'restore-obligation.json').read_text())
    assert obligation['timer']['absolute_deadline_realtime_us'] == readback['absolute_deadline_realtime_us']
    events = read_events(log)
    systemd_run_i = next(i for i, e in enumerate(events) if e['kind'] == 'systemd-run')
    stop_inc_i = next(i for i, e in enumerate(events) if e.get('argv') == ['stop', INCUMBENT])
    assert systemd_run_i < stop_inc_i
    cancel_i = max(i for i, e in enumerate(events) if e['kind'] == 'systemctl' and 'stop' in e['argv'])
    restore_i = next(i for i, e in enumerate(events) if e.get('argv') == ['start', INCUMBENT])
    assert restore_i < cancel_i


def test_k1_canary_rejects_systemd_execstart_swap_before_incumbent_stop(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_SYSTEMD_SWAP_EXECSTART': '1'})
    assert result.returncode != 0
    assert 'ExecStart mismatch' in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    if state.exists():
        assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_k1_canary_rejects_systemd_realtime_deadline_mismatch_before_incumbent_stop(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_SYSTEMD_WRONG_ELAPSE_USEC': '5000000'})
    assert result.returncode != 0
    assert 'deadline mismatch' in result.stderr
    events = read_events(log)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    if state.exists():
        assert json.loads(state.read_text()).get(INCUMBENT, True) is True


def test_k1_canary_systemd_timer_survives_runner_sigkill_and_fires_restore(tmp_path):
    active = tmp_path / "active.json"
    hanging_bash = make_executable(tmp_path / "hanging-bash", f"""#!/usr/bin/env python3
import json, os, pathlib, time
pathlib.Path({str(active)!r}).write_text(json.dumps({{'pid': os.getpid(), 'pgid': os.getpgrp()}}))
while True: time.sleep(1)
""")
    log = tmp_path / "events.jsonl"; state = tmp_path / "docker-state.json"; out = tmp_path / "receipts"
    cmd = base_cmd(tmp_path, out); cmd[cmd.index('--bash') + 1] = str(hanging_bash)
    cmd += ["--window-deadline-sec", "3.0", "--command-timeout-sec", "5"]
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_SYSTEMD_REGISTRY=str(tmp_path / "systemd-registry.json"), FAKE_SYSTEMD_RESTORE_LOG=str(tmp_path / "systemd-restore.log"))
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(active)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    registry = tmp_path / "systemd-registry.json"
    wait_for_path(registry)
    units = json.loads(registry.read_text())['units']
    timer_units = [name for name in units if name.endswith('.timer')]
    assert timer_units
    assert units[timer_units[0]]['ActiveState'] in {'active', 'elapsed'}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if state.exists() and json.loads(state.read_text()).get(INCUMBENT) is True:
            break
        time.sleep(0.05)
    assert json.loads(state.read_text())[INCUMBENT] is True
    events = read_events(log)
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']


def test_k1_canary_staging_archive_contains_base_scripts_sources_and_extracts(tmp_path):
    release = tmp_path / 'k1-canary-release.tar'
    result = subprocess.run([sys.executable, str(RUNNER), '--build-release-archive', str(release), '--recipe', str(RECIPE)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    extract = tmp_path / 'extract'; extract.mkdir()
    subprocess.run(['tar', '-xf', str(release), '-C', str(extract)], check=True)
    root = extract / 'k1-canary-20260908' / 'recipe'
    assert (root / 'scripts/window_k1_canary.py').is_file()
    assert (root / 'scripts/k1_canary_restore_timer.py').is_file()
    assert (root / 'sources/vllm/v1/worker/gpu_model_runner.py').is_file()
    assert (root / 'patches/slot_cache_window_instrumentation.py').is_file()
    assert (root / 'README.md').is_file()
    assert (root / 'scripts/bench3.sh').is_file()
    manifest = json.loads((extract / 'k1-canary-20260908' / 'SHA256MANIFEST.json').read_text())
    paths = {row['path'] for row in manifest['files']}
    assert 'recipe/scripts/window_k1_canary.py' in paths
    assert 'recipe/sources/vllm/v1/worker/gpu_model_runner.py' in paths
    assert 'recipe/scripts/bench3.sh' in paths
    assert manifest['source_files_count'] >= 19


def test_k1_canary_default_source_runner_is_archived_not_users_path(tmp_path):
    result, log, state = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    launch = next(e for e in read_events(log) if e['kind'] == 'bash')
    assert '/Users/' not in launch['env']['SLOT_CACHE_SOURCE_RUNNER']
    assert launch['env']['SLOT_CACHE_SOURCE_RUNNER'].endswith('/restore-bundle/sources/vllm/v1/worker/gpu_model_runner.py')
