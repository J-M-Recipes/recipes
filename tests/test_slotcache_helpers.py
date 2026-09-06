import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
SCRIPTS = RECIPE / "scripts"
INSPECT = RECIPE / "results/2026-09-06-sc13g-slotcache-eager/container-inspect-sanitized.json"


def _run(cmd, *, env=None, cwd=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        env=full_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)
    return path


def _fake_docker_recorder(fakebin: Path, record: Path) -> None:
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env python3
        import json, sys
        pathlib_record = {str(record)!r}
        with open(pathlib_record, "w") as f:
            json.dump(sys.argv[1:], f)
        print("fake-container-id")
        """,
    )


def _docker_env(fakebin: Path, home: Path, tmp_path: Path) -> dict:
    return {
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "HOME": str(home),
        "MODEL_DIR": "/models/glm",
        "CACHE_DIR": str(tmp_path / "cache"),
    }


def test_bigv1_launcher_rejects_missing_api_key_before_docker(tmp_path):
    record = tmp_path / "docker-argv.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()

    result = _run(
        ["bash", str(SCRIPTS / "launch-bigv1.sh"), "v1"],
        env=_docker_env(fakebin, home, tmp_path),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "missing readable API_KEY_FILE" in result.stderr
    assert not record.exists()


def test_bigv1_launcher_rejects_empty_api_key_before_docker_without_leaking_secret(tmp_path):
    record = tmp_path / "docker-argv.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".glm_api_key").write_text("\r\n")

    result = _run(
        ["bash", str(SCRIPTS / "launch-bigv1.sh"), "v1"],
        env=_docker_env(fakebin, home, tmp_path),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "API_KEY_FILE is empty" in result.stderr
    assert not record.exists()


def test_bigv1_launcher_trims_crlf_api_key_and_preserves_measured_flags(tmp_path):
    record = tmp_path / "docker-argv.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()
    secret = "synthetic-crlf-secret"
    (home / ".glm_api_key").write_text(f"{secret}\r\n")

    result = _run(
        ["bash", str(SCRIPTS / "launch-bigv1.sh"), "v1"],
        env=_docker_env(fakebin, home, tmp_path),
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert secret not in result.stderr
    argv = json.loads(record.read_text())
    env_values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-e"]
    assert f"VLLM_API_KEY={secret}" in env_values
    image_index = argv.index("vllm-glm53-uva:v0.28.0-2cf0a691")
    assert argv[image_index + 1 :] == [
        "/model",
        "--host",
        "0.0.0.0",
        "--port",
        "30001",
        "--served-model-name",
        "glm-5.3-big",
        "--trust-remote-code",
        "--quantization",
        "modelopt",
        "--load-format",
        "safetensors",
        "--offload-backend",
        "uva",
        "--cpu-offload-gb",
        "188",
        "--cpu-offload-params",
        "routed_experts.w13_weight",
        "routed_experts.w2_weight",
        "--gpu-memory-utilization",
        "0.95",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-memory",
        "8589934592",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "8192",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
    ]


def test_portable_launcher_preserves_sc13g_flags_and_uses_packaged_hook_paths(tmp_path):
    record = tmp_path / "docker-argv.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".glm_api_key").write_text("synthetic-secret-key\n")

    result = _run(
        ["bash", str(SCRIPTS / "launch-slotcache-portable.sh"), "sc13g", "112"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "HOME": str(home),
            "MODEL_DIR": "/models/glm",
            "CACHE_DIR": str(tmp_path / "cache"),
            "CAPTURE_DIR": str(tmp_path / "capture"),
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "synthetic-secret-key" not in result.stdout
    assert "synthetic-secret-key" not in result.stderr
    argv = json.loads(record.read_text())
    inspect_cmd = json.loads(INSPECT.read_text())["inspect"]["cmd"]

    assert argv[:2] == ["run", "-d"]
    assert "--name" in argv and argv[argv.index("--name") + 1] == "glm53-big-sc13g"
    assert "vllm-glm53-uva:v0.28.0-2cf0a691" in argv
    image_index = argv.index("vllm-glm53-uva:v0.28.0-2cf0a691")
    assert argv[image_index + 1 :] == inspect_cmd

    env_values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-e"]
    assert "EXACT_PIN=1" in env_values
    assert "EXACT_PIN_FILE=/w/patches/exact_pin.py" in env_values
    assert "SLOT_CACHE_HOOK=/w/patches/slot_cache_hook.py" in env_values
    assert "SLOT_CACHE_ROUTER=ffi" in env_values
    assert "SLOT_CACHE_PER_LAYER=/w/configs/slots-8400.json" in env_values
    assert "VLLM_API_KEY=synthetic-secret-key" in env_values

    mounts = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-v"]
    assert f"{RECIPE}:/w:ro" in mounts
    assert f"{RECIPE / 'patches/sitecustomize.py'}:/usr/lib/python3.12/sitecustomize.py:ro" in mounts


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_health_check_uses_bearer_token_retries_and_strictly_accepts_model(tmp_path):
    requests = []
    secret = "synthetic-health-secret"

    class Handler(BaseHTTPRequestHandler):
        health_count = 0

        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/health":
                Handler.health_count += 1
                if Handler.health_count == 1:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"warming")
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if self.path == "/v1/models":
                if self.headers.get("Authorization") != f"Bearer {secret}":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "glm-5.3-big"}]}).encode())
                return
            if self.path == "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    server = _serve(Handler)
    before_tmp_health = set(Path("/tmp").glob("glm53-health*"))
    before_tmp_metrics = set(Path("/tmp").glob("glm53-metrics*"))
    try:
        key_file = tmp_path / "api_key"
        key_file.write_text(secret + "\n")
        result = _run(
            ["bash", str(SCRIPTS / "health-check.sh")],
            env={
                "BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "MODEL_NAME": "glm-5.3-big",
                "API_KEY_FILE": str(key_file),
                "HEALTH_RETRIES": "3",
                "RETRY_SLEEP": "0.01",
                "TIMEOUT": "2",
            },
            cwd=tmp_path,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    assert "HEALTH_OK glm-5.3-big" in result.stdout
    assert "served_models= glm-5.3-big" in result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert [path for path, _ in requests].count("/health") == 2
    assert ("/v1/models", f"Bearer {secret}") in requests
    assert set(Path("/tmp").glob("glm53-health*")) == before_tmp_health
    assert set(Path("/tmp").glob("glm53-metrics*")) == before_tmp_metrics


def test_health_check_rejects_missing_expected_model(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(json.dumps({"data": [{"id": "other-model"}]}).encode())
            else:
                self.wfile.write(b"ok")

    server = _serve(Handler)
    try:
        key_file = tmp_path / "api_key"
        key_file.write_text("synthetic\n")
        result = _run(
            ["bash", str(SCRIPTS / "health-check.sh")],
            env={
                "BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "MODEL_NAME": "glm-5.3-big",
                "API_KEY_FILE": str(key_file),
                "HEALTH_RETRIES": "1",
                "RETRY_SLEEP": "0.01",
            },
            cwd=tmp_path,
        )
    finally:
        server.shutdown()

    assert result.returncode != 0
    assert "expected model 'glm-5.3-big'" in result.stderr


def test_rollback_dry_run_is_cwd_independent_and_has_no_docker_side_effects(tmp_path):
    record = tmp_path / "docker-called"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(record)!r}
        exit 99
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh")],
        env={"PATH": f"{fakebin}:{os.environ['PATH']}"},
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "ROLLBACK_DRY_RUN" in result.stdout
    assert str(SCRIPTS / "launch-bigv1.sh") in result.stdout
    assert str(SCRIPTS / "health-check.sh") in result.stdout
    assert not record.exists()


def test_rollback_execute_stops_candidate_starts_preserved_without_rm_then_health(tmp_path):
    docker_log = tmp_path / "docker.log"
    health_log = tmp_path / "health.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        if [ "$1" = "inspect" ]; then
          if [ "$4" = "glm53-big-v1-keep" ]; then printf 'false\n'; exit 0; fi
          exit 0
        fi
        if [ "$1" = "stop" ]; then exit 0; fi
        if [ "$1" = "start" ]; then printf '%s\n' "$2"; exit 0; fi
        exit 7
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        f"""
        #!/usr/bin/env bash
        printf 'health called BASE_URL=%s MODEL_NAME=%s\n' "${{BASE_URL:-}}" "${{MODEL_NAME:-}}" >> {str(health_log)!r}
        exit 0
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "HEALTH_CHECK_SCRIPT": str(health),
            "HEALTH_RETRIES": "1",
            "RETRY_SLEEP": "0.01",
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    commands = docker_log.read_text().splitlines()
    assert any(line == "stop glm53-big-sc13g" for line in commands)
    assert any(line == "inspect -f {{.State.Running}} glm53-big-v1-keep" for line in commands)
    assert any(line == "start glm53-big-v1-keep" for line in commands)
    assert not any(line.startswith("rm ") for line in commands)
    assert "health called" in health_log.read_text()


def test_rollback_fresh_launch_fails_closed_when_preserved_absence_is_unconfirmed(tmp_path):
    docker_log = tmp_path / "docker.log"
    launch_log = tmp_path / "launch.log"
    health_log = tmp_path / "health.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        if [ "$1" = "inspect" ]; then printf 'Cannot connect to Docker daemon\n' >&2; exit 1; fi
        if [ "$1" = "container" ] && [ "$2" = "ls" ]; then printf 'list transport failure\n' >&2; exit 55; fi
        if [ "$1" = "stop" ]; then exit 0; fi
        exit 7
        """,
    )
    launch = _write_executable(
        tmp_path / "launch-bigv1.sh",
        f"""
        #!/usr/bin/env bash
        printf 'launch %s\n' "$*" >> {str(launch_log)!r}
        exit 0
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        f"""
        #!/usr/bin/env bash
        printf 'health\n' >> {str(health_log)!r}
        exit 0
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute", "--fresh-launch"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "LAUNCH_SCRIPT": str(launch),
            "HEALTH_CHECK_SCRIPT": str(health),
        },
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "fresh launch requires confirmed preserved-container absence" in result.stderr
    assert "docker inspect/list failed" in result.stderr
    commands = docker_log.read_text().splitlines()
    assert "inspect -f {{.State.Running}} glm53-big-v1-keep" in commands
    assert "container ls -a --format {{.Names}}" in commands
    assert "stop glm53-big-sc13g" not in commands
    assert not launch_log.exists()
    assert not health_log.exists()


def test_rollback_rejects_candidate_equal_to_preserved_without_docker_side_effects(tmp_path):
    docker_log = tmp_path / "docker.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        exit 99
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        """
        #!/usr/bin/env bash
        exit 0
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "CANDIDATE_CONTAINER": "glm53-big-v1-keep",
            "PRESERVED_CONTAINER": "glm53-big-v1-keep",
            "HEALTH_CHECK_SCRIPT": str(health),
        },
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "refusing to stop preserved baseline" in result.stderr
    assert not docker_log.exists()


def test_rollback_passes_slow_cold_start_health_defaults(tmp_path):
    docker_log = tmp_path / "docker.log"
    health_log = tmp_path / "health.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        if [ "$1" = "inspect" ]; then printf 'true\n'; exit 0; fi
        if [ "$1" = "stop" ]; then exit 0; fi
        exit 7
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        f"""
        #!/usr/bin/env bash
        printf 'HEALTH_RETRIES=%s RETRY_SLEEP=%s\n' "${{HEALTH_RETRIES:-}}" "${{RETRY_SLEEP:-}}" >> {str(health_log)!r}
        exit 0
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute"],
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "HEALTH_CHECK_SCRIPT": str(health)},
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert health_log.read_text() == "HEALTH_RETRIES=720 RETRY_SLEEP=5\n"


def test_rollback_execute_requires_explicit_fresh_launch_when_preserved_missing(tmp_path):
    docker_log = tmp_path / "docker.log"
    launch_log = tmp_path / "launch.log"
    health_log = tmp_path / "health.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        if [ "$1" = "inspect" ]; then exit 1; fi
        if [ "$1" = "container" ] && [ "$2" = "ls" ]; then exit 0; fi
        if [ "$1" = "stop" ]; then exit 0; fi
        exit 7
        """,
    )
    launch = _write_executable(
        tmp_path / "launch-bigv1.sh",
        f"""
        #!/usr/bin/env bash
        printf 'launch %s\n' "$*" >> {str(launch_log)!r}
        exit 0
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        f"""
        #!/usr/bin/env bash
        printf 'health\n' >> {str(health_log)!r}
        exit 0
        """,
    )

    blocked = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute"],
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "LAUNCH_SCRIPT": str(launch), "HEALTH_CHECK_SCRIPT": str(health)},
        cwd=tmp_path,
    )
    assert blocked.returncode != 0
    assert "pass --fresh-launch" in blocked.stderr
    assert not launch_log.exists()
    blocked_commands = docker_log.read_text().splitlines()
    assert "stop glm53-big-sc13g" not in blocked_commands

    fresh = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute", "--fresh-launch"],
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "LAUNCH_SCRIPT": str(launch), "HEALTH_CHECK_SCRIPT": str(health)},
        cwd=tmp_path,
    )
    assert fresh.returncode == 0, fresh.stderr
    assert launch_log.read_text() == "launch v1\n"
    assert health_log.read_text() == "health\n"


def test_rollback_execute_fails_closed_when_candidate_stop_fails(tmp_path):
    docker_log = tmp_path / "docker.log"
    health_log = tmp_path / "health.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> {str(docker_log)!r}
        if [ "$1" = "inspect" ]; then printf 'false\n'; exit 0; fi
        if [ "$1" = "stop" ]; then printf 'synthetic stop failure\n' >&2; exit 42; fi
        if [ "$1" = "start" ]; then printf '%s\n' "$2"; exit 0; fi
        exit 7
        """,
    )
    health = _write_executable(
        tmp_path / "synthetic-health.sh",
        f"""
        #!/usr/bin/env bash
        printf 'health\n' >> {str(health_log)!r}
        exit 0
        """,
    )

    result = _run(
        ["bash", str(SCRIPTS / "rollback.sh"), "--execute"],
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "HEALTH_CHECK_SCRIPT": str(health)},
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "failed to stop candidate container: glm53-big-sc13g" in result.stderr
    assert "set CANDIDATE_CONTAINER" in result.stderr
    commands = docker_log.read_text().splitlines()
    assert "start glm53-big-v1-keep" not in commands
    assert not health_log.exists()
