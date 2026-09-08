#!/usr/bin/env python3
"""Fail-closed release gate for the E0/E1/E5 GB300 window."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _exact_value(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"missing {path.name}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular non-symlink file")
    value = path.read_text()
    if value.endswith("\n"):
        value = value[:-1]
    if "\n" in value or "\r" in value:
        raise ValueError(f"{path.name} must contain exactly one line")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()

    if not _RUN_ID.fullmatch(args.run_id):
        parser.error("invalid run id")
    try:
        control = _exact_value(args.root / "CONTROL")
        if control != "RUN":
            raise ValueError("CONTROL must equal RUN")
        release = _exact_value(args.root / "RELEASE")
        if release != args.run_id:
            raise ValueError(f"RELEASE does not equal {args.run_id}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"WINDOW_GATE_OK {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
