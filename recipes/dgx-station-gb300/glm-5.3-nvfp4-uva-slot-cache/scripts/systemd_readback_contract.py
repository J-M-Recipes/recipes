#!/usr/bin/env python3
"""Dummy system-scope systemd timer contract for the production C2 readback parser.

Arms a transient timer whose service is `/bin/true --restore-only`, reads it back
with window_c2_continuation.read_systemd_timer, then cancels it. Never references
the incumbent or the real restore command. Refuses to run without --dry-run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_C2_RUNNER = SCRIPT_DIR / "window_c2_continuation.py"
DEFAULT_UNIT_PREFIX = "glm53-contract-test"
FORBIDDEN_SUBSTRINGS = (
    "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907",
    "glm53-big-c2-continuation-k2",
    "window_c2_continuation.py",
    "/Users/",
)


def _load_runner(path: Path):
    path = path.resolve()
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("c2_runner_for_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C2 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_runner = _load_runner(DEFAULT_C2_RUNNER)
v2 = _runner.v2
run_unit = _runner.run_unit
read_systemd_timer = _runner.read_systemd_timer
systemctl_show = _runner.systemctl_show
digest_json = _runner.digest_json


class ContractFailed(RuntimeError):
    pass


def dummy_cmd() -> list[str]:
    resolved = shutil.which("true")
    if resolved is None:
        raise ContractFailed("cannot resolve true")
    path = Path(resolved).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ContractFailed(f"true is not an executable regular file: {path}")
    cmd = [str(path), "--restore-only"]
    joined = " ".join(cmd)
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in joined:
            raise ContractFailed(f"dummy command contains forbidden substring {forbidden}")
    return cmd


def build_systemd_run_argv(
    *,
    systemd_run: str,
    unit: str,
    description: str,
    calendar: str,
    cmd: list[str],
) -> list[str]:
    return [
        systemd_run,
        "--system",
        "--unit",
        unit,
        "--description",
        description,
        "--property",
        "Type=oneshot",
        "--property",
        "CollectMode=inactive-or-failed",
        "--timer-property",
        "Persistent=true",
        "--on-calendar",
        calendar,
        "--",
        *cmd,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--c2-runner", type=Path, default=DEFAULT_C2_RUNNER)
    parser.add_argument("--delay-sec", type=float, default=120.0)
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--systemd-run", default="systemd-run")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--sudo-command", default="sudo")
    parser.add_argument("--command-timeout-sec", type=float, default=30.0)
    return parser.parse_args(argv)


def _ns(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        systemd_run=args.systemd_run,
        systemctl=args.systemctl,
        sudo=bool(args.sudo),
        sudo_command=args.sudo_command,
        command_timeout_sec=float(args.command_timeout_sec),
    )


def _systemd_version(args: argparse.Namespace, out: Path) -> str:
    result = v2.run_logged(["systemd", "--version"], out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    text = (result.stdout or "").strip().splitlines()
    return text[0] if text else "unknown"


def _list_leftover(args: argparse.Namespace, out: Path, prefix: str) -> list[str]:
    cmd = [args.systemctl, "--system", "list-units", "--all", "--plain", "--no-legend", prefix + "*"]
    actual = [args.sudo_command, *cmd] if args.sudo else cmd
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    lines = []
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if stripped and prefix in stripped:
            lines.append(stripped)
    return lines


def run_contract(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ns = _ns(args)
    cmd = dummy_cmd()
    unit = f"{args.unit_prefix}-{os.getpid()}"
    deadline_us = int((time.time() + max(args.delay_sec, 0.001)) * 1_000_000)
    calendar = dt.datetime.fromtimestamp(deadline_us / 1_000_000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    description = f"GLM53 systemd readback contract dummy {unit}"
    systemd_run_argv = build_systemd_run_argv(
        systemd_run=args.systemd_run,
        unit=unit,
        description=description,
        calendar=calendar,
        cmd=cmd,
    )
    joined = " ".join(systemd_run_argv)
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in joined:
            raise ContractFailed(f"systemd-run argv contains forbidden substring {forbidden}")
    version = _systemd_version(ns, out)
    armed = False
    try:
        run_unit(ns, systemd_run_argv, out, "systemd-run contract dummy timer")
        armed = True
        payload = read_systemd_timer(ns, out, unit + ".timer", unit + ".service", deadline_us, cmd)
        if payload.get("exec_start_sha256") != payload.get("expected_restore_cmd_sha256"):
            raise ContractFailed("exec_start_sha256 != expected_restore_cmd_sha256")
        (out / "restore-timer-armed-readback.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        run_unit(ns, [args.systemctl, "--system", "stop", unit + ".timer"], out, "systemctl stop contract dummy timer")
        armed = False
        leftover = _list_leftover(ns, out, args.unit_prefix)
        if leftover:
            raise ContractFailed("leftover units remain: " + " | ".join(leftover))
        receipt = {
            "schema": "glm53-systemd-readback-contract-v1",
            "status": "CONTRACT_OK",
            "promotion_authorized": False,
            "station_mutation_authorized": False,
            "unit": unit,
            "timer_unit": unit + ".timer",
            "service_unit": unit + ".service",
            "dummy_cmd": cmd,
            "exec_start_sha256": payload["exec_start_sha256"],
            "expected_cmd_sha256": payload["expected_restore_cmd_sha256"],
            "absolute_deadline_realtime_us": deadline_us,
            "systemd_version": version,
            "leftover_units": leftover,
            "c2_runner": str(Path(args.c2_runner).resolve()),
        }
        (out / "contract-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("CONTRACT_OK")
        print("SYSTEMD_VERSION=" + version)
        print("UNIT=" + unit)
        return 0
    finally:
        if armed:
            try:
                run_unit(ns, [args.systemctl, "--system", "stop", unit + ".timer"], out, "systemctl stop contract dummy timer")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        print("refusing to run without --dry-run", file=sys.stderr)
        return 2
    if args.unit_prefix != DEFAULT_UNIT_PREFIX:
        print("refusing unit-prefix; must be glm53-contract-test", file=sys.stderr)
        return 2
    try:
        return run_contract(args)
    except Exception as exc:
        print(f"CONTRACT_FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
