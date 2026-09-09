#!/usr/bin/env python3
"""Independent deadline restore helper for window_k1_canary.

This process is launched before the incumbent is stopped. It is intentionally
small and release-gate independent: after its absolute deadline it executes the
archived canary runner in --restore-only mode unless a cancel file exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def _now_ns() -> int:
    return time.monotonic_ns()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-monotonic-ns", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--cancel", type=Path, required=True)
    parser.add_argument("--restore-runner", type=Path, required=True)
    parser.add_argument("--restore-out", type=Path, required=True)
    parser.add_argument("--docker", required=True)
    parser.add_argument("--docker-context", default=None)
    parser.add_argument("--health", default=None)
    parser.add_argument("--api-probe", default=None)
    parser.add_argument("--host-operation-lock", required=True)
    parser.add_argument("--command-timeout-sec", type=float, required=True)
    parser.add_argument("--readiness-timeout-sec", type=float, required=True)
    return parser.parse_args(argv)


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    armed_ns = _now_ns()
    payload = {
        "schema": "glm53-k1-canary-independent-restore-timer-v1",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "armed_monotonic_ns": armed_ns,
        "deadline_monotonic_ns": args.deadline_monotonic_ns,
        "restore_runner": str(args.restore_runner),
        "restore_out": str(args.restore_out),
        "cancel": str(args.cancel),
        "status": "armed",
    }
    write_state(args.state, payload)
    while _now_ns() < args.deadline_monotonic_ns:
        if args.cancel.exists():
            payload.update({"status": "cancelled", "cancelled_monotonic_ns": _now_ns()})
            write_state(args.state, payload)
            return 0
        time.sleep(min(1.0, max((args.deadline_monotonic_ns - _now_ns()) / 1_000_000_000, 0.01)))
    if args.cancel.exists():
        payload.update({"status": "cancelled", "cancelled_monotonic_ns": _now_ns()})
        write_state(args.state, payload)
        return 0
    cmd = [
        sys.executable,
        str(args.restore_runner),
        "--restore-only",
        "--out",
        str(args.restore_out),
        "--docker",
        args.docker,
        "--host-operation-lock",
        args.host_operation_lock,
        "--command-timeout-sec",
        str(args.command_timeout_sec),
        "--readiness-timeout-sec",
        str(args.readiness_timeout_sec),
    ]
    if args.docker_context:
        cmd += ["--docker-context", args.docker_context]
    if args.health:
        cmd += ["--health", args.health]
    if args.api_probe:
        cmd += ["--api-probe", args.api_probe]
    payload.update({"status": "deadline_fired", "fired_monotonic_ns": _now_ns(), "restore_cmd": cmd})
    write_state(args.state, payload)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (args.state.parent / "restore-timer-run.log").write_text(result.stdout or "")
    payload.update({"status": "restore_executed", "restore_returncode": int(result.returncode), "completed_monotonic_ns": _now_ns()})
    write_state(args.state, payload)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
