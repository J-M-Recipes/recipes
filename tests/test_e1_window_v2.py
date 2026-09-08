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
RUNNER = RECIPE / "scripts/window_e1_v2.py"
LAUNCHER = RECIPE / "scripts/launch-slotcache-portable.sh"
VERDICT = RECIPE / "scripts/window_verdict.py"

INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"


def make_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fake_python(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-python",
        """#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record(kind, argv):
    with log.open('a') as f:
        f.write(json.dumps({'kind': kind, 'argv': argv, 'env': {k: os.environ.get(k) for k in ('HEALTH_RETRIES','RETRY_SLEEP','STARTUP_DEADLINE_SEC','BASE_URL','MODEL_NAME','MODEL','API_KEY','API_KEY_FILE')}}) + '\\n')
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
    rows = []
    for i, kind in enumerate(('prose','prose','code','code')):
        rows.append({'index': i, 'kind': kind, 'completion_tokens': max_tokens, 'metric_delta': {'drafts': 32, 'draft_tokens': 32, 'accepted_tokens': 30}})
    payload = {'schema':'glm53-dflash2-uva-acceptance-v1','model':'glm-5.3-big','max_tokens':max_tokens,'rows':rows,'summary': {'requests':4,'completion_tokens': max_tokens*4,'verification_steps':128,'acceptance_length_weighted': max_tokens/32,'decode_tok_s_median': 100.0}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + '\\n')
elif args and args[0].endswith('nsys_capture_control.py'):
    record('nsys-control', args)
    action = args[2]
    token = args[3]
    print(f'ACK {action} {token}')
elif args and args[0].endswith('nsys_bucket.py'):
    record('nsys-bucket', args)
    out = pathlib.Path(args[args.index('--output') + 1])
    buckets = {name: {'total_ms': 0.0, 'instances': 0, 'per_step_ms': 0.0} for name in ('fused_bookkeeping','masked_row_copy','routed_moe','scalar_gather','mla_attention','mtp_verify','other_gpu')}
    buckets['fused_bookkeeping']['per_step_ms'] = 1.25
    buckets['scalar_gather']['per_step_ms'] = 0.60
    out.write_text(json.dumps({'schema':'glm53-nsys-buckets-v1','steps':128,'wall_ms':1000.0,'source_report': str(pathlib.Path(args[0]).name), 'buckets':buckets}) + '\\n')
elif args and args[0].endswith('window_verdict.py'):
    record('verdict', args)
    sys.exit(subprocess.run([sys.executable] + args).returncode)
elif args and args[0].endswith('bench_big.py') or (args and args[0].endswith('bench_big_code.py')):
    record('bench', args)
    print('BENCH {"C": 1, "agg_tok_s": 10.0}')
    print('BENCH {"C": 4, "agg_tok_s": 30.0}')
    print('BENCH {"C": 8, "agg_tok_s": 50.0}')
else:
    record('unknown-python', args)
    sys.exit(2)
""",
    )


def fake_docker(tmp_path: Path) -> Path:
    path = make_executable(
        tmp_path / "fake-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
log = pathlib.Path(os.environ['FAKE_LOG'])
def record():
    with log.open('a') as f:
        f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
record()
args = sys.argv[1:]
if args[:2] == ['inspect', '-f']:
    name = args[-1]
    if name == '{INCUMBENT}':
        print('true')
    else:
        print('false')
elif args[:3] == ['image', 'inspect', '--format']:
    print('sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14')
elif args and args[0] == 'inspect':
    name = args[-1]
    print(json.dumps([{{'Id':'c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea','Image':'sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14','Name':'/'+name,'Config':{{'Image':'vllm-glm53-uva:v0.28.0-2cf0a691','Cmd':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}},'State':{{'Running': True, 'StartedAt':'now'}},'Args':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}}]))
elif args and args[0] == 'logs':
    print('engine core captured no profiler errors')
elif args and args[0] == 'ps':
    print('{INCUMBENT}')
sys.exit(0)
""",
    )
    docker_name = tmp_path / "docker"
    if not docker_name.exists():
        docker_name.symlink_to(path)
    return path


def fake_bash(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-bash",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'bash','argv':sys.argv[1:], 'env': {k: os.environ.get(k) for k in ('NSYS','IMAGE','DOCKER','ROUTER','CAPTURE','UNPACKED','LOGIT_RING','BYPASS','DRAFT_MODEL_DIR','MODEL_DIR','CACHE_DIR','API_KEY_FILE','KV_CACHE_MEMORY','MAX_MODEL_LEN','MAX_NUM_SEQS','SLOT_CACHE_PER_LAYER','AT_KEY','STATS_SEC','COMPILATION_CONFIG','CONTAINER_NAME','CAPTURE_DIR','NSYS_OUTPUT','NSYS_CONTROL')}}) + '\\n')
cap = pathlib.Path(os.environ['CAPTURE_DIR'])
cap.mkdir(parents=True, exist_ok=True)
(cap / 'e1-profile.nsys-rep').write_bytes(b'REAL_NSYS_REPORT_BYTES')
print('launched fake')
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
before = report.read_bytes()
if before != b'REAL_NSYS_REPORT_BYTES':
    print('nsys report bytes changed', file=sys.stderr)
    sys.exit(8)
out_prefix = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
for suffix in ('cuda_gpu_kern_sum','cuda_kern_exec_sum','cuda_gpu_trace','cuda_api_trace'):
    (out_prefix.parent / f'{out_prefix.name}_{suffix}.csv').write_text('Name,Total Time (ms),Instances\\nfused_bookkeeping,160,128\\n')
""",
    )


def fake_health(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-health.sh",
        """#!/usr/bin/env python3
import json, os, pathlib
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'health','env': {k: os.environ.get(k) for k in ('HEALTH_RETRIES','RETRY_SLEEP','STARTUP_DEADLINE_SEC','BASE_URL','MODEL_NAME','MODEL','API_KEY','API_KEY_FILE')}}) + '\\n')
print('health ok')
""",
    )


def fake_api_probe(tmp_path: Path) -> Path:
    return make_executable(
        tmp_path / "fake-api-probe",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'models.json').write_text(json.dumps({'data':[{'id':'glm-5.3-big'}]}) + '\\n')
(out / 'completion.json').write_text(json.dumps({'choices':[{'message':{'content':'WINDOW_RESTORE_OK'}}]}) + '\\n')
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'api-probe','argv':sys.argv[1:], 'env': {'BASE_URL': os.environ.get('BASE_URL'), 'MODEL_NAME': os.environ.get('MODEL_NAME'), 'MODEL': os.environ.get('MODEL'), 'API_KEY': os.environ.get('API_KEY'), 'API_KEY_FILE': os.environ.get('API_KEY_FILE')}}) + '\\n')
""",
    )


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


def test_launcher_preserves_compilation_config_json_and_does_not_append_extra_brace(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret")
    capture = tmp_path / "cap"
    docker = fake_docker(tmp_path)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", API_KEY_FILE=str(key), CAPTURE_DIR=str(capture), COMPILATION_CONFIG='{"mode":3,"backend":"eager"}', FAKE_LOG=str(tmp_path / "events.jsonl"))
    result = subprocess.run(["bash", str(LAUNCHER), "e1-nsys", "112", "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}'], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(Path(env['FAKE_LOG']))
    run = next(e for e in events if e['kind'] == 'docker' and e['argv'][0] == 'run')
    argv = run['argv']
    assert argv[argv.index('--compilation-config') + 1] == '{"mode":3,"backend":"eager"}'


def test_v2_runner_scrubs_launch_environment_uses_long_readiness_archives_and_never_launches_k2(tmp_path):
    log = tmp_path / "events.jsonl"
    out = tmp_path / "receipts"
    root = tmp_path / "root"
    root.mkdir()
    (root / "CONTROL").write_text("RUN\n")
    (root / "RELEASE").write_text("e0-e1-20260908-v2\n")
    docker = fake_docker(tmp_path)
    env = dict(os.environ, FAKE_LOG=str(log), NSYS='leak', IMAGE='bad:image', ROUTER='bad', CAPTURE='1', UNPACKED='1', LOGIT_RING='1', BYPASS='999', DRAFT_MODEL_DIR='/bad/draft')
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(docker), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    launch = next(e for e in events if e['kind'] == 'bash')
    assert launch['argv'][-2:] == ['--speculative-config', '{"method":"mtp","num_speculative_tokens":1}']
    assert 'num_speculative_tokens":2' not in json.dumps(events)
    assert launch['env']['IMAGE'] == 'vllm-glm53-uva:v0.28.0-2cf0a691@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14'
    assert launch['env']['DOCKER'] == str(docker)
    assert launch['argv'][0].startswith(str(out / 'executed-source'))
    for name in ('ROUTER','CAPTURE','UNPACKED','LOGIT_RING','BYPASS','DRAFT_MODEL_DIR'):
        assert launch['env'][name] in (None, '0', 'ffi', '16'), (name, launch['env'][name])
    assert launch['env']['NSYS'] == '1'
    assert launch['env']['COMPILATION_CONFIG'] == '{"mode":3,"backend":"eager"}'
    assert (out / 'e1-profile.nsys-rep').read_bytes() == b'REAL_NSYS_REPORT_BYTES'
    health = next(e for e in events if e['kind'] == 'health')
    assert health['env']['HEALTH_RETRIES'] == '360'
    assert health['env']['RETRY_SLEEP'] == '5'
    assert health['env']['STARTUP_DEADLINE_SEC'] is None
    bench = next(e for e in events if e['kind'] == 'bench')
    assert bench['env']['BASE_URL'] == 'http://127.0.0.1:30001/v1'
    assert bench['env']['MODEL'] == 'glm-5.3-big'
    assert bench['env']['API_KEY'] == '***test-only-key***'
    manifest = json.loads((out / 'source-manifest.json').read_text())
    assert manifest['schema'] == 'glm53-e1-v2-source-manifest-v1'
    assert all('__pycache__' not in item['path'] and not item['path'].endswith('.pyc') for item in manifest['files'])
    assert (out / 'raw-logs' / 'launch.log').is_file()
    verdict = json.loads((out / 'e1-verdict.json').read_text())
    assert verdict['valid'] is False
    assert verdict['verdict'] == 'INCONCLUSIVE'
    assert 'missing hash-bound CUDA API attribution' in verdict['issues']
    restore_proof = json.loads((out / 'restore' / 'service-proof.json').read_text())
    assert restore_proof['observed_container_id'] == 'c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea'
    assert restore_proof['observed_image'] == 'sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14'
    preflight_proof = json.loads((out / 'preflight' / 'incumbent-proof.json').read_text())
    assert preflight_proof['observed_container_id'] == restore_proof['observed_container_id']
    api_probe = next(e for e in events if e['kind'] == 'api-probe')
    assert api_probe['env']['MODEL_NAME'] == 'glm-5.3-big'
    assert json.loads((out / 'restore' / 'models.json').read_text())['data'][0]['id'] == 'glm-5.3-big'
    source_paths = {item['path'] for item in manifest['files']}
    assert 'results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py' in source_paths
    assert 'results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py' in source_paths
    for required_source in ('patches/sitecustomize.py', 'patches/exact_pin.py', 'patches/ffi_route.py'):
        assert required_source in source_paths
    run_evidence = json.loads((out / 'current-run-evidence.json').read_text())
    assert run_evidence['source_manifest_sha256']
    assert run_evidence['candidate_image'] == launch['env']['IMAGE']
    launch_index = next(i for i, e in enumerate(events) if e['kind'] == 'bash')
    restore_index = next(i for i, e in enumerate(events) if e['kind'] == 'docker' and e['argv'] == ['start', INCUMBENT])
    nsys_index = next(i for i, e in enumerate(events) if e['kind'] == 'nsys')
    assert launch_index < restore_index < nsys_index


def test_v2_runner_restores_incumbent_when_launch_fails(tmp_path):
    failing_bash = make_executable(tmp_path / "failing-bash", "#!/usr/bin/env python3\nimport sys\nprint('launch failed')\nsys.exit(7)\n")
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir()
    (root / "CONTROL").write_text("RUN\n")
    (root / "RELEASE").write_text("e0-e1-20260908-v2\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(failing_bash), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    assert ['stop', INCUMBENT] in docker_calls
    assert ['start', INCUMBENT] in docker_calls


def test_v2_runner_gate_failure_blocks_mutation_and_does_not_restore_unstopped_incumbent(tmp_path):
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir()
    (root / "CONTROL").write_text("HOLD\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log), FAKE_GATE_FAIL='1'), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    assert ['stop', INCUMBENT] not in docker_calls
    assert ['start', INCUMBENT] not in docker_calls



def test_v2_runner_refuses_existing_output_directory(tmp_path):
    log = tmp_path / "events.jsonl"
    out = tmp_path / "receipts"
    out.mkdir()
    (out / "existing.txt").write_text("do not overwrite")
    root = tmp_path / "root"
    root.mkdir()
    (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert (out / "existing.txt").read_text() == "do not overwrite"
    assert not log.exists()


def test_v2_runner_rejects_restore_identity_without_required_id_digest_and_runtime_flags(tmp_path):
    bad_docker = make_executable(
        tmp_path / "bad-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
args = sys.argv[1:]
if args[:2] == ['inspect', '-f']:
    print('true' if args[-1] == '{INCUMBENT}' else 'false')
elif args and args[0] == 'inspect':
    print(json.dumps([{{'Name':'/{INCUMBENT}','Config':{{'Image':'vllm-glm53-uva:v0.28.0-2cf0a691','Env':['VLLM_API_KEY=SECRET'], 'Cmd':['/model','--max-model-len','65536','--max-num-seqs','8']}},'State':{{'Running': True}}}}]))
elif args and args[0] in ('logs', 'ps'):
    print('ok')
sys.exit(0)
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(bad_docker), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    docker_log = (tmp_path / "receipts" / "raw-logs" / "docker.log").read_text()
    assert "SECRET" not in docker_log
    proof = json.loads((tmp_path / "receipts" / "preflight" / "incumbent-proof.json").read_text())
    assert "Config" not in json.dumps(proof)


def test_v2_runner_rejects_incumbent_identity_with_mtp_k2_before_stop(tmp_path):
    k2_docker = make_executable(
        tmp_path / "k2-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
args = sys.argv[1:]
if args[:2] == ['inspect', '-f']:
    print('true' if args[-1] == '{INCUMBENT}' else 'false')
elif args[:3] == ['image', 'inspect', '--format']:
    print('sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14')
elif args and args[0] == 'inspect':
    print(json.dumps([{{'Id':'c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea','Image':'sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14','Name':'/{INCUMBENT}','Config':{{'Image':'vllm-glm53-uva:v0.28.0-2cf0a691','Cmd':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":2}}']}},'State':{{'Running': True}},'Args':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":2}}']}}]))
elif args and args[0] in ('logs', 'ps'):
    print('ok')
sys.exit(0)
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(k2_docker), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert ['stop', INCUMBENT] not in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_v2_runner_redacts_candidate_inspect_and_env_output_from_receipts(tmp_path):
    secret = "SUPERSECRET1234567890"
    leaky_docker = make_executable(
        tmp_path / "leaky-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
args = sys.argv[1:]
if args[:2] == ['inspect', '-f']:
    print('true' if args[-1] == '{INCUMBENT}' else 'false')
elif args[:3] == ['image', 'inspect', '--format']:
    print('sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14')
elif args and args[0] == 'inspect':
    name = args[-1]
    print(json.dumps([{{'Id':'c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea','Image':'sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14','Name':'/'+name,'Config':{{'Image':'vllm-glm53-uva:v0.28.0-2cf0a691','Env':['VLLM_API_KEY={secret}'], 'Cmd':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}},'State':{{'Running': True}},'Args':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}}]))
elif args and args[0] == 'logs':
    print('container env VLLM_API_KEY={secret}')
elif args and args[0] == 'ps':
    print('{INCUMBENT}')
sys.exit(0)
""",
    )
    log = tmp_path / "events.jsonl"
    out = tmp_path / "receipts"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(leaky_docker), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    receipt_text = "\n".join(path.read_text(errors="replace") for path in out.rglob("*") if path.is_file())
    assert secret not in receipt_text
    assert "VLLM_API_KEY=<redacted>" in receipt_text


def test_v2_runner_times_out_subprocess_and_restores(tmp_path):
    fake_py = fake_python(tmp_path)
    slow_python = make_executable(
        tmp_path / "slow-python",
        f"""#!/usr/bin/env python3
import subprocess, sys, time
if sys.argv[1].endswith('dflash2_acceptance_probe.py'):
    time.sleep(5)
else:
    raise SystemExit(subprocess.run([{str(fake_py)!r}] + sys.argv[1:]).returncode)
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2", "--command-timeout-sec", "1",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(slow_python), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_v2_runner_uses_real_2400s_readiness_timeout_separate_from_command_timeout(tmp_path):
    slow_health = make_executable(
        tmp_path / "slow-health",
        """#!/usr/bin/env python3
import time
time.sleep(5)
print('late health ok')
""",
    )
    log = tmp_path / "events.jsonl"
    out = tmp_path / "receipts"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(out), "--run-id", "e0-e1-20260908-v2", "--command-timeout-sec", "20", "--readiness-timeout-sec", "1",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(slow_health), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    assert "exit=timeout after 1.0s" in (out / "health.txt").read_text()


def test_v2_runner_rejects_trailing_k2_speculative_config_args_before_launch(tmp_path):
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
    assert not log.exists()


def test_restore_only_is_independently_invocable_without_release_gate(tmp_path):
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("HOLD\n")
    out = tmp_path / "restore-only"
    result = subprocess.run([
        sys.executable, str(RUNNER), "--restore-only", "--out", str(out),
        "--docker", str(fake_docker(tmp_path)), "--health", str(fake_health(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log), FAKE_GATE_FAIL='1'), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    assert ['start', INCUMBENT] in docker_calls
    assert not any(e['kind'] == 'gate' for e in read_events(log))


def test_v2_runner_terminates_active_subprocess_group_on_sigterm_before_restore(tmp_path):
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
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    proc = subprocess.Popen([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(hanging_bash), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_path(active)
    child_pid = json.loads(active.read_text())["pid"]
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode != 0, stdout + stderr
    assert_process_exited(child_pid)
    assert ['start', INCUMBENT] in [e['argv'] for e in read_events(log) if e['kind'] == 'docker']


def test_v2_runner_runs_gate_and_health_from_archived_snapshot_after_live_mutation(tmp_path):
    recipe_copy = tmp_path / "recipe"
    shutil.copytree(RECIPE, recipe_copy)
    health_script = recipe_copy / "scripts/health-check.sh"
    make_executable(
        health_script,
        """#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({'kind':'health-script','argv0':sys.argv[0]}) + '\\n')
print('snapshot health ok')
""",
    )
    base_python = fake_python(tmp_path)
    snapshot_python = make_executable(
        tmp_path / "snapshot-python",
        f"""#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
if args and args[0].endswith('window_gate.py'):
    script = pathlib.Path(args[0])
    with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
        f.write(json.dumps({{'kind':'gate','argv':args}}) + '\\n')
    if 'FAIL_LIVE_MUTATION' in script.read_text():
        print('live mutated gate used', file=sys.stderr)
        sys.exit(13)
    print('WINDOW_GATE_OK ' + args[-1])
else:
    raise SystemExit(subprocess.run([{str(base_python)!r}] + args).returncode)
""",
    )
    mutating_bash = make_executable(
        tmp_path / "mutating-bash",
        f"""#!/usr/bin/env python3
import os, pathlib
pathlib.Path({str(recipe_copy / 'scripts/window_gate.py')!r}).write_text('FAIL_LIVE_MUTATION\\n')
pathlib.Path({str(health_script)!r}).write_text('#!/usr/bin/env bash\\necho live mutated health used >&2\\nexit 17\\n')
cap = pathlib.Path(os.environ['CAPTURE_DIR']); cap.mkdir(parents=True, exist_ok=True)
(cap / 'e1-profile.nsys-rep').write_bytes(b'REAL_NSYS_REPORT_BYTES')
print('mutated live source')
""",
    )
    log = tmp_path / "events.jsonl"
    out = tmp_path / "receipts"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(recipe_copy), "--out", str(out), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(mutating_bash), "--python", str(snapshot_python), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    events = read_events(log)
    assert 'FAIL_LIVE_MUTATION' in (recipe_copy / 'scripts/window_gate.py').read_text()
    assert 'live mutated health used' in health_script.read_text()
    assert all(e['argv'][0].startswith(str(out / 'executed-source')) for e in events if e['kind'] == 'gate')
    assert all(e['argv0'].startswith(str(out / 'executed-source')) for e in events if e['kind'] == 'health-script')


def test_v2_runner_fails_closed_when_archived_source_is_mutated_after_archive(tmp_path):
    mutating_bash = make_executable(
        tmp_path / "mutating-bash",
        """#!/usr/bin/env python3
import os, pathlib, sys
path = pathlib.Path(sys.argv[1]).parent / 'window_gate.py'
path.write_text(path.read_text() + '\\n# mutated after archive\\n')
cap = pathlib.Path(os.environ['CAPTURE_DIR']); cap.mkdir(parents=True, exist_ok=True)
(cap / 'e1-profile.nsys-rep').write_bytes(b'REAL_NSYS_REPORT_BYTES')
print('mutated archived source')
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(mutating_bash), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert '# mutated after archive' not in (RECIPE / 'scripts/window_gate.py').read_text()


def test_v2_runner_rejects_candidate_digest_drift_before_stopping_incumbent(tmp_path):
    bad_docker = make_executable(
        tmp_path / "bad-candidate-docker",
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f:
    f.write(json.dumps({{'kind':'docker','argv':sys.argv[1:]}}) + '\\n')
args = sys.argv[1:]
if args[:2] == ['inspect', '-f'] and args[-1] == '{INCUMBENT}':
    print('true')
elif args[:3] == ['image', 'inspect', '--format']:
    print('sha256:WRONG')
elif args and args[0] == 'inspect':
    print(json.dumps([{{'Id':'c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea','Image':'sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14','Name':'/'+args[-1],'Config':{{'Image':'vllm-glm53-uva:v0.28.0-2cf0a691','Cmd':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}},'State':{{'Running': True}},'Args':['/model','--max-model-len','524288','--max-num-seqs','1','--speculative-config','{{"method":"mtp","num_speculative_tokens":1}}']}}]))
sys.exit(0)
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(bad_docker), "--bash", str(fake_bash(tmp_path)), "--python", str(fake_python(tmp_path)), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    docker_calls = [e['argv'] for e in read_events(log) if e['kind'] == 'docker']
    assert ['stop', INCUMBENT] not in docker_calls



def test_v2_runner_restores_on_incorrect_profile_control_ack_before_analysis(tmp_path):
    fake_py = fake_python(tmp_path)
    bad_ack_python = make_executable(
        tmp_path / "bad-ack-python",
        f"""#!/usr/bin/env python3
import subprocess, sys
if sys.argv[1].endswith('nsys_capture_control.py'):
    print('ACK WRONG TOKEN')
    sys.exit(0)
raise SystemExit(subprocess.run([{str(fake_py)!r}] + sys.argv[1:]).returncode)
""",
    )
    log = tmp_path / "events.jsonl"
    root = tmp_path / "root"
    root.mkdir(); (root / "CONTROL").write_text("RUN\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "--root", str(root), "--recipe", str(RECIPE), "--out", str(tmp_path / "receipts"), "--run-id", "e0-e1-20260908-v2",
        "--docker", str(fake_docker(tmp_path)), "--bash", str(fake_bash(tmp_path)), "--python", str(bad_ack_python), "--health", str(fake_health(tmp_path)), "--nsys", str(fake_nsys(tmp_path)), "--api-probe", str(fake_api_probe(tmp_path)),
    ], env=dict(os.environ, FAKE_LOG=str(log)), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    events = read_events(log)
    assert ['start', INCUMBENT] in [e['argv'] for e in events if e['kind'] == 'docker']
    assert not any(e['kind'] == 'nsys' for e in events)
