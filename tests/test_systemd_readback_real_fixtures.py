import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
RUNNER = RECIPE / "scripts/window_c2_continuation.py"
FIXTURES = REPO_ROOT / "tests/fixtures/systemd"
E5D0EBC_LOG = FIXTURES / "execstart-real-e5d0ebc-r2.txt"
D02233B_LOG = FIXTURES / "execstart-real-d02233b-r1.txt"
D02233B_READBACK = FIXTURES / "restore-timer-armed-readback-d02233b-r1.json"


def load_runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("window_c2_continuation_real_fixtures", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_execstart(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            return line.split("=", 1)[1]
    raise AssertionError("fixture is missing ExecStart=")


def accept_execstart(runner, tmp_path, monkeypatch, exec_start: str, expected: list[str]):
    def show(args, out, unit, props):
        if unit.endswith(".timer"):
            return {"ActiveState": "active", "NextElapseUSecRealtime": "1060000000"}
        return {"ActiveState": "inactive", "ExecStart": exec_start}

    monkeypatch.setattr(runner, "systemctl_show", show)
    payload = runner.read_systemd_timer(None, tmp_path, "test.timer", "test.service", 1060000000, expected)
    assert payload["status"] == "armed"
    assert payload["exec_start_argv"] == expected
    assert payload["exec_start_sha256"] == payload["expected_restore_cmd_sha256"]
    assert payload["service"]["ExecStart"] == exec_start
    return payload


def test_real_e5d0ebc_execstart_accepted(monkeypatch, tmp_path):
    runner = load_runner_module()
    exec_start = first_execstart(E5D0EBC_LOG.read_text())
    expected = runner._extract_execstart(exec_start)
    assert expected[0] == "/usr/bin/python3.12"
    assert "--restore-only" in expected
    assert exec_start.endswith("status=0/0 }")
    accept_execstart(runner, tmp_path, monkeypatch, exec_start, expected)


def test_real_d02233b_execstart_accepted(monkeypatch, tmp_path):
    runner = load_runner_module()
    exec_start = first_execstart(D02233B_LOG.read_text())
    expected = runner._extract_execstart(exec_start)
    assert expected[0] == "/usr/bin/python3.12"
    assert "--restore-only" in expected
    assert exec_start.endswith("status=0/0 }")
    accept_execstart(runner, tmp_path, monkeypatch, exec_start, expected)


def test_real_d02233b_armed_readback_execstart_accepted(monkeypatch, tmp_path):
    runner = load_runner_module()
    payload = json.loads(D02233B_READBACK.read_text())
    exec_start = payload["service"]["ExecStart"]
    expected = list(payload["exec_start_argv"])
    assert expected == runner._extract_execstart(exec_start)
    assert payload["exec_start_sha256"] == payload["expected_restore_cmd_sha256"]
    accepted = accept_execstart(runner, tmp_path, monkeypatch, exec_start, expected)
    assert accepted["exec_start_sha256"] == payload["exec_start_sha256"]


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("wrong-path", lambda text: text.replace("path=/usr/bin/python3.12", "path=/opt/wrong/python3", 1)),
        ("duplicate-path", lambda text: text.replace("ignore_errors=no", "path=/usr/bin/python3.12")),
        ("unstructured", lambda text: "/usr/bin/python3.12 /tmp/bundle/runner.py --restore-only"),
    ],
)
def test_real_shape_rejects_path_field_failures(monkeypatch, tmp_path, label, mutate):
    runner = load_runner_module()
    original = first_execstart(D02233B_LOG.read_text())
    expected = runner._extract_execstart(original)
    exec_start = mutate(original)
    if label != "unstructured":
        assert runner._extract_execstart(exec_start) == expected

    def show(args, out, unit, props):
        if unit.endswith(".timer"):
            return {"ActiveState": "active", "NextElapseUSecRealtime": "1060000000"}
        return {"ActiveState": "inactive", "ExecStart": exec_start}

    monkeypatch.setattr(runner, "systemctl_show", show)
    with pytest.raises(runner.C2Failed, match="restore service ExecStart mismatch") as error:
        runner.read_systemd_timer(None, tmp_path, "test.timer", "test.service", 1060000000, expected)
    assert error.value.code == "RESTORE_TIMER_INVALID"


def test_real_shape_rejects_missing_restore_only(monkeypatch, tmp_path):
    runner = load_runner_module()
    expected = ["/usr/bin/python3.12", "/tmp/bundle/runner.py"]
    exec_start = (
        "{ path=/usr/bin/python3.12 ; argv[]=/usr/bin/python3.12 /tmp/bundle/runner.py"
        " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
    )
    assert "--restore-only" not in exec_start

    def show(args, out, unit, props):
        if unit.endswith(".timer"):
            return {"ActiveState": "active", "NextElapseUSecRealtime": "1060000000"}
        return {"ActiveState": "inactive", "ExecStart": exec_start}

    monkeypatch.setattr(runner, "systemctl_show", show)
    with pytest.raises(runner.C2Failed, match="not independent restore-only") as error:
        runner.read_systemd_timer(None, tmp_path, "test.timer", "test.service", 1060000000, expected)
    assert error.value.code == "RESTORE_TIMER_INVALID"


def test_real_shape_rejects_users_argv(monkeypatch, tmp_path):
    runner = load_runner_module()
    expected = ["/usr/bin/python3.12", "/Users/jamesmeadlock/runner.py", "--restore-only"]
    exec_start = (
        "{ path=/usr/bin/python3.12 ; argv[]=/usr/bin/python3.12 /Users/jamesmeadlock/runner.py --restore-only"
        " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
    )

    def show(args, out, unit, props):
        if unit.endswith(".timer"):
            return {"ActiveState": "active", "NextElapseUSecRealtime": "1060000000"}
        return {"ActiveState": "inactive", "ExecStart": exec_start}

    monkeypatch.setattr(runner, "systemctl_show", show)
    with pytest.raises(runner.C2Failed, match="not independent restore-only") as error:
        runner.read_systemd_timer(None, tmp_path, "test.timer", "test.service", 1060000000, expected)
    assert error.value.code == "RESTORE_TIMER_INVALID"


def test_fixture_files_have_no_credentials():
    cred = re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}")
    for path in (E5D0EBC_LOG, D02233B_LOG, D02233B_READBACK):
        text = path.read_text()
        assert cred.search(text) is None, path
        assert "sk-" not in text
        assert "VLLM_API_KEY=" not in text
