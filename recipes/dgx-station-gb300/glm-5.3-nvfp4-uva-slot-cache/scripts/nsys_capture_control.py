#!/usr/bin/env python3
"""Atomically request START/STOP from the slot-cache CUDA profiler controller."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_file", type=Path)
    parser.add_argument("action", choices=("START", "STOP"))
    parser.add_argument("run_id")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--poll", type=float, default=0.05)
    args = parser.parse_args()

    if not _RUN_ID.fullmatch(args.run_id):
        parser.error("invalid run id")
    if args.timeout <= 0 or args.poll <= 0:
        parser.error("timeout and poll must be positive")
    if not args.control_file.parent.is_dir():
        parser.error(f"control directory does not exist: {args.control_file.parent}")

    command = f"{args.action} {args.run_id}"
    expected = f"ACK {command}"
    temporary = args.control_file.with_name(args.control_file.name + f".tmp.{os.getpid()}")
    temporary.write_text(command + "\n")
    os.replace(temporary, args.control_file)

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            response = args.control_file.read_text().strip()
        except FileNotFoundError:
            response = ""
        if response == expected:
            print(response)
            return 0
        if response.startswith("ERROR "):
            print(response, file=sys.stderr)
            return 2
        time.sleep(args.poll)

    print(f"timed out waiting for {expected}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
