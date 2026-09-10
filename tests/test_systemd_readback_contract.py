import json
import os
import shlex
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
CONTRACT = RECIPE / "scripts/systemd_readback_contract.py"
C2_RUNNER = RECIPE / "scripts/window_c2_continuation.py"


def load_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location("systemd_readback_contract_under_test", CONTRACT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_script_exists():
    assert CONTRACT.is_file()


def test_contract_refuses_without_dry_run(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CONTRACT), "--unit-prefix", "glm53-contract-test", "--out", str(tmp_path / "out")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "dry-run" in (result.stderr + result.stdout).lower()
    assert "CONTRACT_OK" not in result.stdout


def test_contract_refuses_wrong_unit_prefix(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(CONTRACT),
            "--dry-run",
            "--unit-prefix",
            "glm53-c2-continuation-restore",
            "--out",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unit-prefix" in (result.stderr + result.stdout).lower()


def test_dummy_cmd_is_true_restore_only_without_incumbent(tmp_path, monkeypatch):
    contract = load_contract()
    dummy = tmp_path / "true"
    dummy.write_text("#!/bin/sh\nexit 0\n")
    dummy.chmod(dummy.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(contract.shutil, "which", lambda command: str(dummy) if command in {"true", dummy.name} else None)
    cmd = contract.dummy_cmd()
    assert cmd[0] == str(dummy.resolve())
    assert cmd[1:] == ["--restore-only"]
    joined = " ".join(cmd)
    for forbidden in contract.FORBIDDEN_SUBSTRINGS:
        assert forbidden not in joined


def test_systemd_run_flags_match_arm_restore_timer():
    contract = load_contract()
    source = C2_RUNNER.read_text()
    start = source.index("def arm_restore_timer")
    end = source.index("\ndef cancel_restore_timer")
    body = source[start:end]
    for token in (
        '"--system"',
        '"--unit"',
        '"--property"',
        '"Type=oneshot"',
        '"CollectMode=inactive-or-failed"',
        '"--timer-property"',
        '"Persistent=true"',
        '"--on-calendar"',
    ):
        assert token in body
    argv = contract.build_systemd_run_argv(
        systemd_run="systemd-run",
        unit="glm53-contract-test-1",
        description="dummy",
        calendar="2026-01-01 00:00:00 UTC",
        cmd=["/bin/true", "--restore-only"],
    )
    assert argv[:2] == ["systemd-run", "--system"]
    assert argv[argv.index("--property") + 1] == "Type=oneshot"
    assert "CollectMode=inactive-or-failed" in argv
    assert argv[argv.index("--timer-property") + 1] == "Persistent=true"
    assert argv[argv.index("--") + 1 :] == ["/bin/true", "--restore-only"]
    assert "window_c2_continuation.py" not in argv


def test_dry_run_arms_reads_cancels_and_prints_contract_ok(tmp_path, monkeypatch, capsys):
    contract = load_contract()
    dummy = tmp_path / "true"
    dummy.write_text("#!/bin/sh\nexit 0\n")
    dummy.chmod(dummy.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(contract.shutil, "which", lambda command: str(dummy) if command == "true" else None)
    monkeypatch.setattr(contract.time, "time", lambda: 1000.0)
    monkeypatch.setattr(os, "getpid", lambda: 4242)

    expected = [str(dummy.resolve()), "--restore-only"]
    exec_start = (
        "{ path="
        + expected[0]
        + " ; argv[]="
        + shlex.join(expected)
        + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
    )
    submitted = []
    leftover = {"armed": True}

    def run_logged(argv, log, timeout=None, **kwargs):
        submitted.append(list(argv))
        if argv[0] == "systemd-run":
            leftover["armed"] = True
            return subprocess.CompletedProcess(argv, 0, "Running timer as unit: glm53-contract-test-4242.timer\n", "")
        if argv[0] == "systemctl" and "show" in argv:
            unit = argv[-1]
            if leftover.get("armed") and unit.endswith(".timer"):
                text = "LoadState=loaded\nActiveState=active\nTriggers=glm53-contract-test-4242.service\nNextElapseUSecRealtime=1120000000\n"
            elif leftover.get("armed") and unit.endswith(".service"):
                text = "LoadState=loaded\nActiveState=inactive\nExecStart=" + exec_start + "\n"
            else:
                text = "LoadState=not-found\nActiveState=inactive\n"
            return subprocess.CompletedProcess(argv, 0, text, "")
        if argv[0] == "systemctl" and "stop" in argv:
            leftover["armed"] = False
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "systemctl" and "list-units" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "systemd" and "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "systemd 249 (249)\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(contract.v2, "run_logged", run_logged)
    out = tmp_path / "out"
    rc = contract.main(
        [
            "--dry-run",
            "--unit-prefix",
            "glm53-contract-test",
            "--out",
            str(out),
            "--c2-runner",
            str(C2_RUNNER),
            "--delay-sec",
            "120",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "CONTRACT_OK" in captured.out
    assert leftover["armed"] is False
    receipt = json.loads((out / "contract-receipt.json").read_text())
    assert receipt["promotion_authorized"] is False
    assert receipt["exec_start_sha256"] == receipt["expected_cmd_sha256"]
    assert receipt["leftover_units"] == []
    assert any(argv[0] == "systemd-run" for argv in submitted)
    systemd_run = next(argv for argv in submitted if argv[0] == "systemd-run")
    assert systemd_run[systemd_run.index("--") + 1 :] == expected
    assert "window_c2_continuation.py" not in " ".join(systemd_run)
    assert "glm53-big-sc13g" not in " ".join(systemd_run)


def test_dry_run_fails_if_units_remain(tmp_path, monkeypatch):
    contract = load_contract()
    dummy = tmp_path / "true"
    dummy.write_text("#!/bin/sh\nexit 0\n")
    dummy.chmod(dummy.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(contract.shutil, "which", lambda command: str(dummy) if command == "true" else None)
    monkeypatch.setattr(contract.time, "time", lambda: 1000.0)
    monkeypatch.setattr(os, "getpid", lambda: 7)
    expected = [str(dummy.resolve()), "--restore-only"]
    exec_start = "{ path=" + expected[0] + " ; argv[]=" + shlex.join(expected) + " ; ignore_errors=no ; status=0/0 }"

    def run_logged(argv, log, timeout=None, **kwargs):
        if argv[0] == "systemd-run":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "systemctl" and "show" in argv:
            unit = argv[-1]
            if unit.endswith(".timer"):
                text = "LoadState=loaded\nActiveState=active\nTriggers=glm53-contract-test-7.service\nNextElapseUSecRealtime=1120000000\n"
            else:
                text = "LoadState=loaded\nActiveState=inactive\nExecStart=" + exec_start + "\n"
            return subprocess.CompletedProcess(argv, 0, text, "")
        if argv[0] == "systemctl" and "stop" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "systemctl" and "list-units" in argv:
            return subprocess.CompletedProcess(argv, 0, "glm53-contract-test-7.timer loaded active running\n", "")
        if argv[0] == "systemd":
            return subprocess.CompletedProcess(argv, 0, "systemd 249\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(contract.v2, "run_logged", run_logged)
    rc = contract.main(["--dry-run", "--out", str(tmp_path / "out"), "--c2-runner", str(C2_RUNNER)])
    assert rc != 0
