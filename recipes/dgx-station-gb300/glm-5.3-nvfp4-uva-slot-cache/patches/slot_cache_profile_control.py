"""State machine for file-triggered Nsight Systems capture control."""

from __future__ import annotations

import re
from collections.abc import Callable

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def handle_command(
    command: str,
    *,
    state: str,
    start: Callable[[], object],
    stop: Callable[[], object],
) -> tuple[str, str]:
    """Apply one START/STOP command and return the new state and acknowledgement."""
    parts = command.strip().split()
    if len(parts) != 2 or parts[0] not in {"START", "STOP"}:
        raise ValueError("expected START or STOP followed by one run id")
    action, run_id = parts
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run id")

    running = state.startswith("running:")
    stopped = state.startswith("stopped:")
    active_run = state.split(":", 1)[1] if running or stopped else None

    if action == "START":
        if running:
            if active_run != run_id:
                raise ValueError("run id does not match active run")
            return state, f"ACK START {run_id}"
        if stopped:
            raise ValueError("capture has already stopped and cannot restart")
        status = start()
        if status not in (None, 0):
            raise RuntimeError(f"CUDA profiler START failed with status {status}")
        return f"running:{run_id}", f"ACK START {run_id}"

    if stopped:
        if active_run != run_id:
            raise ValueError("run id does not match active run")
        return state, f"ACK STOP {run_id}"
    if not running:
        raise ValueError("cannot STOP before START")
    if active_run != run_id:
        raise ValueError("run id does not match active run")
    status = stop()
    if status not in (None, 0):
        raise RuntimeError(f"CUDA profiler STOP failed with status {status}")
    return f"stopped:{run_id}", f"ACK STOP {run_id}"
