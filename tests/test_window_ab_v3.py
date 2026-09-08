import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
RUNNER = RECIPE / "scripts/window_ab_v3.py"
INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"
C1 = "glm53-big-k2v3-c1-mtp1"
C2 = "glm53-big-k2v3-c2-mtp2"
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


def recipe_with_fake_verdict(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe"
    shutil.copytree(RECIPE, recipe)
    make_executable(
        recipe / "scripts/window_ab_verdict.py",
        """#!/usr/bin/env python3
import json, pathlib, sys
def validate_lane(payload):
    rows = payload.get('rows')
    speed = payload.get('summary', {}).get('decode_tok_s_median')
    if not isinstance(rows, list) or not rows or not isinstance(speed, (int, float)):
        raise ValueError('malformed lane')
    return float(speed)
def quality_match(reference, candidate):
    keys = {str(i) for i in range(20)}
    return set(reference) == keys and set(candidate) == keys and all(reference[k] and reference[k] == candidate[k] for k in keys)
def c1_gate(probe, reference, candidate, baseline_speed=45.65):
    speed = validate_lane(probe)
    quality_pass = quality_match(reference, candidate)
    return {'pass': speed >= baseline_speed * 0.99 and quality_pass, 'speed': speed, 'quality_pass': quality_pass}
def evaluate(c1_probe, c2_probe, reference, c1_quality, c2_quality):
    return {'g1': c1_gate(c1_probe, reference, c1_quality), 'c2_speed': validate_lane(c2_probe), 'c2_quality': quality_match(reference, c2_quality)}
if __name__ == '__main__':
    out = pathlib.Path(sys.argv[2])
    dest = pathlib.Path(sys.argv[3])
    dest.write_text(json.dumps({'schema':'glm53-window-ab-v3-verdict-v1','verdict':'INCONCLUSIVE','valid':False,'integration':'fake'}) + '\\n')
    print('WINDOW_AB_VERDICT_FAKE')
""",
    )
    return recipe


def fake_python(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-python",
        """#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record(kind, argv):
    with log.open('a') as f:
        f.write(json.dumps({'kind': kind, 'argv': argv, 'env': {k: os.environ.get(k) for k in ('BASE_URL','MODEL_NAME','MODEL','API_KEY','API_KEY_FILE','HEALTH_RETRIES','RETRY_SLEEP')}}) + '\\n')
args = sys.argv[1:]
if args and args[0].endswith('window_gate.py'):
    record('gate', args)
    if os.environ.get('FAKE_GATE_FAIL') == '1':
        sys.exit(9)
    print('WINDOW_GATE_OK ' + args[-1])
elif args and args[0].endswith('dflash2_acceptance_probe.py'):
    record('probe', args)
    out = pathlib.Path(args[args.index('--out') + 1])
    max_tokens = int(args[args.index('--max-tokens') + 1])
    median = float(os.environ.get('FAKE_MEDIAN_TOK_S', '48'))
    payload = {'schema':'glm53-dflash2-uva-acceptance-v1','model':'glm-5.3-big','max_tokens':max_tokens,'rows':[{'index':0,'completion_tokens':max_tokens}], 'summary': {'requests':1,'completion_tokens': max_tokens,'verification_steps':128,'decode_tok_s_median': median}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + '\\n')
elif args and args[0].endswith('greedy_equiv.py') and len(args) >= 4 and args[1] == '--compare':
    record('greedy-compare', args)
    if os.environ.get('FAKE_GREEDY_MISMATCH') == '1':
        print('GREEDY_EQUIV identical=19/20')
    else:
        print('GREEDY_EQUIV identical=20/20')
elif args and args[0].endswith('greedy_equiv.py'):
    record('greedy-capture', args)
    out = pathlib.Path(args[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(i): 'same' for i in range(20)}
    if os.environ.get('FAKE_GREEDY_MISMATCH') == '1' and 'c1-greedy' in str(out):
        payload['7'] = 'different'
    out.write_text(json.dumps(payload) + '\\n')
    print('GREEDY saved 20 outputs to ' + str(out))
elif args and args[0].endswith('nsys_capture_control.py'):
    record('nsys-control', args)
    print(f'ACK {args[2]} {args[3]}')
elif args and args[0].endswith('bench_big.py') or (args and args[0].endswith('bench_big_code.py')):
    record('bench', args)
    print('BENCH {"C": 1, "agg_tok_s": 10.0}')
elif args and args[0].endswith('window_ab_verdict.py'):
    record('verdict', args)
    raise SystemExit(subprocess.run([sys.executable] + args).returncode)
elif args and args[0].endswith('nsys_bucket.py'):
    record('nsys-bucket', args)
    pathlib.Path(args[args.index('--output') + 1]).write_text(json.dumps({'schema':'glm53-nsys-buckets-v1'}) + '\\n')
else:
    record('unknown-python', args)
    sys.exit(2)
""",
    )


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
    return {{'{INCUMBENT}': True, '{C1}': False, '{C2}': False}}
def save(s): state.write_text(json.dumps(s))
def record():
    with log.open('a') as f:
        f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
record()
args = sys.argv[1:]
s = load()
if args[:2] == ['inspect', '-f']:
    name = args[-1]
    if os.environ.get('FAKE_INSPECT_AMBIGUOUS') == '1' and name in ('{C1}', '{C2}'):
        sys.exit(42)
    if name not in s:
        sys.exit(1)
    print('true' if s[name] else 'false')
elif args[:3] == ['image', 'inspect', '--format']:
    print('{IMAGE_ID}')
elif args and args[0] == 'inspect':
    name = args[-1]
    running = bool(s.get(name, False))
    k = 2 if name == '{C2}' else 1
    if os.environ.get('FAKE_C2_INSPECT_WRONG_K') == '1' and name == '{C2}':
        k = 1
    cfg = '{{"method":"mtp","num_speculative_tokens":%d}}' % k
    container_id = '{INCUMBENT_ID}' if name == '{INCUMBENT}' else 'container-'+name
    print(json.dumps([{{'Id':container_id,'Image':'{IMAGE_ID}','Name':'/'+name,'Config':{{'Image':'{IMAGE_TAG}','Env':['MODEL_NAME=glm-5.3-big'],'Cmd':['/model','--model','glm-5.3-big','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}},'State':{{'Running': running}},'Args':['/model','--model','glm-5.3-big','--max-model-len','524288','--max-num-seqs','1','--speculative-config',cfg]}}]))
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
    print('STATS {{"layers":{{"0":{{"misses":1,"routes":2,"steps":3}}}}}}')
elif args and args[0] == 'ps':
    print('{INCUMBENT}' if s.get('{INCUMBENT}') else '')
sys.exit(0)
""",
    )


def fake_bash(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-bash",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'bash','argv':sys.argv[1:], 'env': {{k: os.environ.get(k) for k in ('NSYS','IMAGE','DOCKER','ROUTER','CAPTURE','UNPACKED','LOGIT_RING','BYPASS','DRAFT_MODEL_DIR','MODEL_DIR','CACHE_DIR','API_KEY_FILE','KV_CACHE_MEMORY','MAX_MODEL_LEN','MAX_NUM_SEQS','SLOT_CACHE_PER_LAYER','AT_KEY','STATS_SEC','COMPILATION_CONFIG','CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL')}}}}) + '\\n')
state = pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
s = json.loads(state.read_text()) if state.exists() else {{'{INCUMBENT}': False, '{C1}': False, '{C2}': False}}
s[os.environ['CONTAINER_NAME']] = True
state.write_text(json.dumps(s))
cap = pathlib.Path(os.environ['CAPTURE_DIR'])
cap.mkdir(parents=True, exist_ok=True)
(cap / (os.environ['CONTAINER_NAME'] + '.nsys-rep')).write_bytes(b'REAL_NSYS_REPORT_BYTES')
(cap / (os.environ['CONTAINER_NAME'] + '-slot-stats.jsonl')).write_text('{{"layers":{{"0":{{"misses":1,"routes":2,"steps":3}}}}}}\\n')
print('launched fake ' + os.environ['CONTAINER_NAME'])
""",
    )


def fake_health(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-health",
        """#!/usr/bin/env python3
import json, os, pathlib
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'health','env': {k: os.environ.get(k) for k in ('HEALTH_RETRIES','RETRY_SLEEP','BASE_URL','MODEL_NAME')}}) + '\\n')
print('health ok')
if os.environ.get('FAKE_HEALTH_FAIL') == '1':
    raise SystemExit(77)
""",
    )


def fake_api_probe(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-api-probe",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / 'models.json').write_text(json.dumps({'data':[{'id':'glm-5.3-big'}]}) + '\\n')
(out / 'completion.json').write_text(json.dumps({'choices':[{'message':{'content':'WINDOW_RESTORE_OK'}}]}) + '\\n')
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'api-probe','argv':sys.argv[1:]}) + '\\n')
""",
    )


def fake_nsys(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-nsys",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'nsys','argv':sys.argv[1:]}) + '\\n')
out_prefix = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
report = pathlib.Path(sys.argv[-1])
if os.environ.get('FAKE_NSYS_MUTATE_REPORT') == '1':
    report.write_bytes(report.read_bytes() + b'MUTATED')
for suffix in ('cuda_gpu_kern_sum','cuda_kern_exec_sum','cuda_gpu_trace','cuda_api_trace'):
    (out_prefix.parent / f'{out_prefix.name}_{suffix}.csv').write_text('Name,Total Time (ms),Instances\\nfused_bookkeeping,160,128\\n')
""",
    )


def base_cmd(tmp_path: Path, out: Path, recipe: Path | None = None):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    (root / "CONTROL").write_text("RUN\n")
    (root / "RELEASE").write_text("k2-v3-20260908\n")
    return [
        # Test-only driver exercises the unreleased lifecycle with fake tools.
        # The production CLI has no bypass for its PREP_BLOCKED gate.
        sys.executable, '-c',
        'import sys; sys.path.insert(0, ' + repr(str(RUNNER.parent)) + '); '
        'import window_ab_v3 as w; raise SystemExit(w._run_ab(w.parse_args()))',
        "--root", str(root), "--recipe", str(recipe or recipe_with_fake_verdict(tmp_path)), "--out", str(out),
        "--run-id", "k2-v3-20260908", "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)),
        "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)), "--nsys", str(fake_nsys(tmp_path)),
    ]


def run_runner(tmp_path: Path, *, recipe: Path | None = None, extra_args=(), env_extra=None):
    log = tmp_path / "events.jsonl"
    state = tmp_path / "docker-state.json"
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), **(env_extra or {}))
    cmd = base_cmd(tmp_path, tmp_path / "receipts", recipe=recipe) + list(extra_args)
    return subprocess.run(cmd, env=env, text=True, capture_output=True, check=False), log, state


def test_stop_mutates_then_fails_still_restores_incumbent(tmp_path):
    result, log, state = run_runner(tmp_path, env_extra={'FAKE_STOP_FAIL_AFTER_MUTATION': '1'})
    assert result.returncode != 0
    events = read_events(log)
    assert any(e.get('argv') == ['start', INCUMBENT] for e in events)
    assert json.loads(state.read_text())[INCUMBENT] is True
    assert not any(e['kind'] == 'bash' for e in events)


def test_sigterm_during_restore_does_not_abort_restore_status(tmp_path):
    marker = tmp_path / 'restore-started'
    log = tmp_path / "events.jsonl"
    state = tmp_path / "docker-state.json"
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_STOP_FAIL_AFTER_MUTATION='1', FAKE_RESTORE_START_MARKER=str(marker))
    cmd = base_cmd(tmp_path, tmp_path / "receipts")
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(marker)
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode != 0, stdout + stderr
    assert (tmp_path / 'receipts' / 'restore' / 'restore-status.txt').exists()
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_ab_v3_fails_closed_prep_blocked_when_required_verdict_dependency_missing(tmp_path):
    recipe = tmp_path / "recipe-missing-verdict"
    shutil.copytree(RECIPE, recipe)
    (recipe / "scripts/window_ab_verdict.py").unlink(missing_ok=True)
    result, log, _ = run_runner(tmp_path, recipe=recipe)
    assert result.returncode != 0
    assert "PREP_BLOCKED" in result.stderr
    assert not log.exists() or ['stop', INCUMBENT] not in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    failure = json.loads((tmp_path / "receipts" / "failure.json").read_text())
    assert failure["code"] == "PREP_BLOCKED"


def test_ab_v3_runs_two_same_tree_candidates_and_binds_exact_launch_identity(tmp_path):
    result, log, _ = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    launches = [e for e in events if e['kind'] == 'bash']
    assert [e['env']['CONTAINER_NAME'] for e in launches] == [C1, C2]
    specs = [e['argv'][e['argv'].index('--speculative-config') + 1] for e in launches]
    assert specs == ['{"method":"mtp","num_speculative_tokens":1}', '{"method":"mtp","num_speculative_tokens":2}']
    assert json.dumps(events).count('num_speculative_tokens\\":2') == 1
    c1_env = {k: v for k, v in launches[0]['env'].items() if k not in ('CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL')}
    c2_env = {k: v for k, v in launches[1]['env'].items() if k not in ('CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL')}
    assert c1_env == c2_env
    assert c1_env['STATS_SEC'] == '0'
    assert c1_env['NSYS'] == '1'
    out = tmp_path / "receipts"
    manifest = json.loads((out / "source-manifest.json").read_text())
    paths = {item['path'] for item in manifest['files']}
    assert 'scripts/window_e1_v2.py' in paths
    assert 'scripts/window_ab_v3.py' in paths
    assert 'scripts/greedy_equiv.py' in paths
    assert 'scripts/window_ab_verdict.py' in paths
    assert manifest['helper_binding'] == 'archived-executed-source'
    assert str(out / 'executed-source' / 'scripts/window_e1_v2.py') == manifest['v2_helpers_imported_from']
    launch_proofs = json.loads((out / "launch-identity.json").read_text())
    assert [p['candidate'] for p in launch_proofs['candidates']] == ['c1', 'c2']
    assert launch_proofs['candidates'][0]['config_digest'] != launch_proofs['candidates'][1]['config_digest']
    assert launch_proofs['common_except_speculative_config'] is True
    assert all(p['source_manifest_sha256'] == launch_proofs['source_manifest_sha256'] for p in launch_proofs['candidates'])
    assert [p['observed_speculative_config']['num_speculative_tokens'] for p in launch_proofs['candidates']] == [1, 2]
    assert all(p['observed_image_digest'] == IMAGE_ID for p in launch_proofs['candidates'])
    assert all(p['observed_model'] == 'glm-5.3-big' for p in launch_proofs['candidates'])
    assert all(p['observed_max_model_len'] == '524288' for p in launch_proofs['candidates'])
    assert all(p['observed_max_num_seqs'] == '1' for p in launch_proofs['candidates'])


def test_ab_v3_rejects_candidate_identity_mismatch_from_actual_inspect(tmp_path):
    result, log, _ = run_runner(tmp_path, env_extra={'FAKE_C2_INSPECT_WRONG_K': '1'})
    assert result.returncode != 0
    assert 'candidate c2 speculative K mismatch' in result.stderr
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_ab_v3_requires_nonempty_nsys_reports_and_attests_stats_export_hashes(tmp_path):
    result, _, _ = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    attestation = json.loads((tmp_path / 'receipts' / 'nsys-export-attestation.json').read_text())
    assert [item['candidate'] for item in attestation['reports']] == ['c1', 'c2']
    assert all(item['report_bytes'] > 0 for item in attestation['reports'])
    assert all(item['pre_stats_sha256'] == item['post_stats_sha256'] for item in attestation['reports'])
    assert all(len(item['exported_csv_sha256']) == 4 for item in attestation['reports'])


def test_ab_v3_fails_if_nsys_stats_export_mutates_report(tmp_path):
    result, log, _ = run_runner(tmp_path, env_extra={'FAKE_NSYS_MUTATE_REPORT': '1'})
    assert result.returncode != 0
    assert 'nsys report changed during stats export' in result.stderr
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_restore_health_failure_makes_restore_fail(tmp_path):
    result, log, _ = run_runner(tmp_path, env_extra={'FAKE_HEALTH_FAIL': '1'})
    assert result.returncode != 0
    assert (tmp_path / 'receipts' / 'restore' / 'restore-status.txt').read_text() == 'WINDOW_RESTORE_STATUS=1\n'
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_ab_v3_stops_both_named_candidates_and_verifies_stopped_before_incumbent_starts(tmp_path):
    result, log, _ = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    start_inc = docker_calls.index(['start', INCUMBENT])
    for name in (C1, C2):
        assert ['stop', '-t', '120', name] in docker_calls[:start_inc]
        assert ['inspect', '-f', '{{.State.Running}}', name] in docker_calls[:start_inc]
    assert ['start', INCUMBENT] not in docker_calls[:docker_calls.index(['stop', '-t', '120', C2])]
    assert json.loads((tmp_path / "receipts" / "restore" / "service-proof.json").read_text())['observed_container_id'] == INCUMBENT_ID


def test_ab_v3_g1_c1_receipt_and_greedy_gate_blocks_c2_before_profiler_analysis(tmp_path):
    result, log, _ = run_runner(tmp_path, env_extra={'FAKE_GREEDY_MISMATCH':'1'})
    assert result.returncode != 0
    events = read_events(log)
    assert sum(1 for e in events if e['kind'] == 'bash') == 1
    kinds = [e['kind'] for e in events]
    assert 'greedy-compare' in kinds
    assert 'nsys' not in kinds
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']
    failure = json.loads((tmp_path / "receipts" / "failure.json").read_text())
    assert failure['code'] == 'G1_FAILED'


def test_restore_only_has_no_release_gate_or_host_lock_requirement_and_stops_both_candidates(tmp_path):
    log = tmp_path / "events.jsonl"
    state = tmp_path / "docker-state.json"
    out = tmp_path / "restore-only"
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state), FAKE_GATE_FAIL='1')
    result = subprocess.run([
        sys.executable, str(RUNNER), "--restore-only", "--out", str(out), "--docker", str(fake_docker(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert not any(e['kind'] == 'gate' for e in events)
    assert [e['argv'] for e in events if e['kind'] == 'docker'].count(['start', INCUMBENT]) == 1
    assert not (Path(str(out) + '.lock')).exists()


def test_restore_only_default_health_does_not_use_mutable_recipe_path(tmp_path):
    log = tmp_path / "events.jsonl"
    state = tmp_path / "docker-state.json"
    out = tmp_path / "restore-only"
    bad_recipe = tmp_path / 'missing-recipe'
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state))
    result = subprocess.run([
        sys.executable, str(RUNNER), "--restore-only", "--recipe", str(bad_recipe), "--out", str(out), "--docker", str(fake_docker(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    health_log = (out / 'restore' / 'health.txt').read_text()
    assert str(bad_recipe) not in health_log
    assert str(RECIPE / 'scripts/health-check.sh') in health_log


def test_ambiguous_candidate_inspect_during_stop_does_not_count_as_stopped(tmp_path):
    result, log, _ = run_runner(tmp_path, env_extra={'FAKE_INSPECT_AMBIGUOUS': '1'})
    assert result.returncode != 0
    assert 'candidate stop state ambiguous' in result.stderr
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    ambiguous_probe = docker_calls.index(['inspect', '-f', '{{.State.Running}}', C1])
    assert ['start', INCUMBENT] not in docker_calls[ambiguous_probe:]


def test_public_run_ab_is_not_an_ungated_lifecycle_entrypoint(tmp_path):
    out = tmp_path / 'out'
    result = subprocess.run([
        sys.executable, '-c',
        'import sys; sys.path.insert(0, ' + repr(str(RUNNER.parent)) + '); import window_ab_v3 as w; raise SystemExit(w.run_ab(w.parse_args()))',
        '--root', str(tmp_path), '--out', str(out), '--docker', str(fake_docker(tmp_path)), '--bash', str(fake_bash(tmp_path)), '--python', str(fake_python(tmp_path)), '--health', str(fake_health(tmp_path)), '--api-probe', str(fake_api_probe(tmp_path)), '--nsys', str(fake_nsys(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(tmp_path / 'events.jsonl'), FAKE_DOCKER_STATE=str(tmp_path / 'docker-state.json')), text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert 'PREP_BLOCKED' in result.stderr
    assert not (tmp_path / 'events.jsonl').exists()


def test_ab_v3_terminates_active_child_group_on_sigterm_and_restores(tmp_path):
    active = tmp_path / "active.json"
    hanging_bash = make_executable(
        tmp_path / "hanging-bash",
        f"""#!/usr/bin/env python3
import json, os, pathlib, time
pathlib.Path({str(active)!r}).write_text(json.dumps({{'pid': os.getpid(), 'pgid': os.getpgrp()}}))
while True:
    time.sleep(1)
""",
    )
    recipe = recipe_with_fake_verdict(tmp_path)
    out = tmp_path / "receipts"
    log = tmp_path / "events.jsonl"
    state = tmp_path / "docker-state.json"
    env = dict(os.environ, FAKE_LOG=str(log), FAKE_DOCKER_STATE=str(state))
    cmd = base_cmd(tmp_path, out, recipe=recipe)
    cmd[cmd.index('--bash') + 1] = str(hanging_bash)
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(active)
    child_pid = json.loads(active.read_text())["pid"]
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode != 0, stdout + stderr
    assert_process_exited(child_pid)
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_ab_v3_window_deadline_checked_before_each_subprocess_preserves_restore_budget(tmp_path):
    result, log, _ = run_runner(tmp_path, extra_args=("--window-deadline-sec", "0.001", "--restore-budget-sec", "20"))
    assert result.returncode != 0
    assert "WINDOW_DEADLINE_EXCEEDED" in result.stderr
    events = read_events(log)
    assert not any(e['kind'] == 'bash' for e in events)
    assert ['stop', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert ['start', INCUMBENT] not in [e['argv'] for e in events if e['kind'] == 'docker']
    assert "restore_budget_sec=20.0" in (tmp_path / "receipts" / "raw-logs" / "docker.log").read_text()
