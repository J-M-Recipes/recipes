#!/usr/bin/env python3
"""Restore-protected single-K1 slot-cache instrumentation canary runner.

This is a narrow live canary entrypoint. It collects only safe snapshot/NVTX
correlation evidence for K1 instrumentation and leaves all rows campaign-invalid.
It does not run K2 and does not issue a performance verdict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import csv
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import shlex
import subprocess
import tarfile
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import window_e1_v2 as v2  # noqa: E402

RUN_ID = "k1-canary-20260908"
CANDIDATE = "glm53-big-k1-canary-instrumented"
SPEC_K1 = '{"method":"mtp","num_speculative_tokens":1}'
TRACE_TOKEN = "k1-canary-trace-0001"
EXPERT_LAYER_IDS = tuple(range(3, 78))
ARCHIVED_SOURCE_RUNNER = Path("sources/vllm/v1/worker/gpu_model_runner.py")
SOURCE_RUNNER = None
PINNED_SOURCE_SHA = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
PINNED_PATCHED_SHA = "2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4"
PINNED_GENERATOR_SHA = "8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b"
PINNED_ADAPTER_SHA = "9f0c75b25438c63511a5b2580a4c0a77520f232e2affe109dd0ba3908477e453"
PINNED_LAUNCHER_SHA = "aebe4fab6272a8ded9d2e871d5b9c536b641634ae9b10232db9fa5c33bcac04d"
SOURCE_FILES = tuple(dict.fromkeys((
    "scripts/window_k1_canary.py",
    "scripts/k1_canary_restore_timer.py",
    "scripts/window_e1_v2.py",
    "scripts/launch-slotcache-portable.sh",
    "scripts/apply_slot_cache_instrumentation_patch.py",
    "scripts/window_gate.py",
    "scripts/nsys_capture_control.py",
    "scripts/dflash2_acceptance_probe.py",
    "scripts/health-check.sh",
    "configs/slots-5792-ctx512k.json",
    "patches/slot_cache_window_instrumentation.py",
    "patches/slot_cache_hook.py",
    "patches/slot_cache_stats.py",
    "patches/slot_cache_profile_control.py",
    "patches/sitecustomize.py",
    "patches/exact_pin.py",
    "patches/ffi_route.py",
    "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py",
    "sources/vllm/v1/worker/gpu_model_runner.py",
)))


class CanaryFailed(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_manifest_sha256(recipe: Path) -> str:
    ignored_dirs = {".git", "__pycache__", "capture", "results"}
    rows: list[str] = []
    for path in sorted(recipe.rglob("*")):
        rel_path = path.relative_to(recipe)
        if any(part in ignored_dirs for part in rel_path.parts):
            continue
        if path.is_symlink():
            raise CanaryFailed(f"recipe artifact input must not be a symlink: {rel_path.as_posix()}")
        if not path.is_file():
            continue
        rel = rel_path.as_posix()
        if rel.endswith(".pyc"):
            continue
        data = path.read_bytes()
        rows.append(f"{rel}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}")
    manifest = "\n".join(rows).encode() + b"\n"
    return hashlib.sha256(manifest).hexdigest()


def archive_source(recipe: Path, out: Path, archive_name: str = "restore-bundle") -> Path:
    archive = out / archive_name
    files: list[dict[str, str | int]] = []
    for rel in SOURCE_FILES:
        src = recipe / rel
        if not src.is_file():
            raise CanaryFailed(f"missing source for archive: {rel}")
        if src.is_symlink():
            raise CanaryFailed(f"source archive input is symlink: {rel}")
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest, follow_symlinks=False)
        files.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    manifest = {
        "schema": "glm53-k1-canary-source-manifest-v1",
        "run_id": RUN_ID,
        "files": files,
        "excluded": [".git", "__pycache__", "*.pyc", "capture", "results"],
    }
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def verify_archive_integrity(runtime_recipe: Path, out: Path) -> None:
    manifest = json.loads((out / "source-manifest.json").read_text())
    paths = [item["path"] for item in manifest.get("files", [])]
    if set(paths) != set(SOURCE_FILES) or len(paths) != len(SOURCE_FILES):
        raise CanaryFailed("source manifest file set mismatch")
    for item in manifest["files"]:
        rel = str(item["path"])
        target = runtime_recipe / rel
        if not target.is_file() or target.is_symlink():
            raise CanaryFailed(f"executed source missing or symlink: {rel}")
        if sha256(target) != item["sha256"]:
            raise CanaryFailed(f"executed source hash mismatch: {rel}")


def load_archived_v2(runtime_recipe: Path):
    path = runtime_recipe / "scripts/window_e1_v2.py"
    spec = importlib.util.spec_from_file_location("window_e1_v2_k1_archived", path)
    if spec is None or spec.loader is None:
        raise CanaryFailed("cannot load archived v2 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("INCUMBENT", "INCUMBENT_ID", "IMAGE_DIGEST", "IMAGE", "MODEL", "CONTROLLED_ENV"):
        if getattr(module, name, None) != getattr(v2, name, None):
            raise CanaryFailed(f"archived v2 helper constant mismatch: {name}")
    return module


@contextmanager
def restoring_signal_shield():
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def write_restore_obligation(args: argparse.Namespace, out: Path, timer_state: Path | None = None, deadline_ns: int | None = None, *, absolute_deadline_realtime_us: int | None = None, systemd_readback: dict[str, Any] | None = None) -> None:
    now_ns = time.monotonic_ns()
    wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    deadline = deadline_ns if deadline_ns is not None else now_ns + int(args.window_deadline_sec * 1_000_000_000)
    payload = {
        "schema": "glm53-k1-canary-restore-obligation-v1",
        "run_id": args.run_id,
        "candidate": CANDIDATE,
        "incumbent": v2.INCUMBENT,
        "restore_required_before_stop": True,
        "release_independent_restore_only": True,
        "timer": {
            "armed_wall_time": wall,
            "armed_monotonic_ns": now_ns,
            "window_deadline_sec": float(args.window_deadline_sec),
            "restore_budget_sec": float(args.restore_budget_sec),
            "window_deadline_monotonic_ns": deadline,
            "system_timer_state": str(timer_state) if timer_state is not None else None,
            "absolute_deadline_realtime_us": absolute_deadline_realtime_us,
            "systemd_readback": systemd_readback,
        },
    }
    path = out / "preflight" / "restore-obligation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_systemctl_show(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _restore_command_hash(cmd: list[str]) -> str:
    encoded = json.dumps(cmd, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _decode_systemd_escapes(value: str) -> str:
    def repl(match):
        return chr(int(match.group(1), 16))
    import re
    return re.sub(r"\\x([0-9A-Fa-f]{2})", repl, value)


def _extract_systemd_execstart_argv(exec_start: str) -> list[str]:
    raw = exec_start.strip()
    if not raw:
        return []
    if "argv[]=" in raw:
        argv = raw.split("argv[]=", 1)[1]
        if " ; " in argv:
            argv = argv.split(" ; ", 1)[0]
        elif argv.endswith("}"):
            argv = argv[:-1].strip()
        return shlex.split(_decode_systemd_escapes(argv))
    return shlex.split(_decode_systemd_escapes(raw))


def _parse_systemd_usec(value: str, prop: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped in {"0", "n/a", "[n/a]"}:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise CanaryFailed(f"systemd {prop} is not an integer usec value: {value!r}") from exc


def _parse_systemd_realtime_usec(value: str, expected_us: int) -> int | None:
    """Parse systemctl's integer or human-formatted realtime timer property."""
    stripped = value.strip()
    if not stripped or stripped in {"0", "n/a", "[n/a]"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        expected_local = dt.datetime.fromtimestamp(expected_us / 1_000_000).astimezone()
        expected_utc = dt.datetime.fromtimestamp(expected_us / 1_000_000, dt.timezone.utc)
        displays = {
            expected_local.strftime("%a %Y-%m-%d %H:%M:%S %Z"),
            expected_utc.strftime("%a %Y-%m-%d %H:%M:%S UTC"),
        }
        if stripped not in displays:
            raise CanaryFailed(
                "systemd NextElapseUSecRealtime does not match expected deadline: "
                f"observed={stripped!r} expected_one_of={sorted(displays)!r}"
            )
        return int(expected_local.replace(microsecond=0).timestamp() * 1_000_000)


def _run_unit_command(args: argparse.Namespace, cmd: list[str], out: Path, label: str) -> subprocess.CompletedProcess[str]:
    actual = list(cmd)
    if getattr(args, "sudo", False):
        actual = [args.sudo_command, *actual]
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    if result.returncode != 0:
        raise CanaryFailed(f"{label} failed with exit {result.returncode}")
    return result


def _systemctl_show(args: argparse.Namespace, out: Path, unit: str, props: tuple[str, ...]) -> dict[str, str]:
    cmd = [args.systemctl, "--system", "show"]
    for prop in props:
        cmd += ["-p", prop]
    cmd.append(unit)
    result = _run_unit_command(args, cmd, out, f"systemctl show {unit}")
    values = _parse_systemctl_show(result.stdout)
    if values.get("LoadState") not in {"loaded", "transient"}:
        raise CanaryFailed(f"systemd unit not loaded: {unit}: {values}")
    return values


def _systemctl_show_after_cancel(args: argparse.Namespace, out: Path, unit: str, props: tuple[str, ...]) -> dict[str, str]:
    cmd = [args.systemctl, "--system", "show"]
    for prop in props:
        cmd += ["-p", prop]
    cmd.append(unit)
    actual = [args.sudo_command, *cmd] if getattr(args, "sudo", False) else cmd
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    values = _parse_systemctl_show(result.stdout)
    if values.get("LoadState") == "not-found" and values.get("ActiveState") == "inactive":
        return values
    if result.returncode == 0 and values.get("LoadState") in {"loaded", "transient"}:
        return values
    raise CanaryFailed(f"systemctl show after cancellation failed for {unit} with exit {result.returncode}: {values}")


SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC = 2_000_000


def _read_systemd_timer(args: argparse.Namespace, out: Path, timer_unit: str, service_unit: str, absolute_deadline_realtime_us: int, expected_restore_cmd: list[str] | None = None, expected_deadline_monotonic_us: int | None = None) -> dict[str, Any]:
    timer = _systemctl_show(args, out, timer_unit, ("LoadState", "ActiveState", "Triggers", "NextElapseUSecRealtime", "NextElapseUSecMonotonic"))
    service = _systemctl_show(args, out, service_unit, ("LoadState", "ActiveState", "ExecStart"))
    if timer.get("ActiveState") != "active":
        raise CanaryFailed(f"systemd restore timer is not active: {timer}")
    if timer.get("Triggers") and timer.get("Triggers") != service_unit:
        raise CanaryFailed(f"systemd restore timer trigger mismatch: {timer}")
    exec_start = service.get("ExecStart", "")
    exec_start_argv = _extract_systemd_execstart_argv(exec_start)
    if expected_restore_cmd is not None and exec_start_argv != expected_restore_cmd:
        raise CanaryFailed(
            "systemd restore service ExecStart mismatch: "
            f"expected_sha256={_restore_command_hash(expected_restore_cmd)} "
            f"observed_sha256={_restore_command_hash(exec_start_argv)}"
        )
    elif "--restore-only" not in exec_start_argv:
        raise CanaryFailed("systemd restore service ExecStart does not restore-only")
    if any(arg.startswith("/Users/") for arg in exec_start_argv):
        raise CanaryFailed("systemd restore service ExecStart depends on /Users")
    next_realtime_us = _parse_systemd_realtime_usec(
        timer.get("NextElapseUSecRealtime", ""), absolute_deadline_realtime_us
    )
    if next_realtime_us is None:
        raise CanaryFailed("systemd restore timer has no realtime elapse deadline")
    realtime_delta_us = abs(next_realtime_us - absolute_deadline_realtime_us)
    if realtime_delta_us > SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC:
        raise CanaryFailed(
            "systemd restore timer deadline mismatch: "
            f"expected_realtime_us={absolute_deadline_realtime_us} observed_realtime_us={next_realtime_us} "
            f"tolerance_us={SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC}"
        )
    next_monotonic_us = _parse_systemd_usec(timer.get("NextElapseUSecMonotonic", ""), "NextElapseUSecMonotonic")
    monotonic_delta_us = None
    if next_monotonic_us is not None and expected_deadline_monotonic_us is not None:
        now_monotonic_us = time.monotonic_ns() // 1000
        local_clock_window_us = int(float(args.window_deadline_sec) * 1_000_000) + SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC
        if abs(next_monotonic_us - now_monotonic_us) <= local_clock_window_us:
            monotonic_delta_us = abs(next_monotonic_us - expected_deadline_monotonic_us)
            if monotonic_delta_us > SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC:
                raise CanaryFailed(
                    "systemd restore timer monotonic deadline mismatch: "
                    f"expected_monotonic_us={expected_deadline_monotonic_us} observed_monotonic_us={next_monotonic_us} "
                    f"tolerance_us={SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC}"
                )
    payload: dict[str, Any] = {
        "schema": "glm53-k1-canary-systemd-restore-timer-readback-v1",
        "scope": "system",
        "status": "armed",
        "pid": os.getpid(),
        "deadline_monotonic_ns": int(time.monotonic_ns()),
        "timer_unit": timer_unit,
        "service_unit": service_unit,
        "absolute_deadline_realtime_us": absolute_deadline_realtime_us,
        "deadline_tolerance_us": SYSTEMD_TIMER_DEADLINE_TOLERANCE_USEC,
        "exec_start_argv": exec_start_argv,
        "exec_start_sha256": _restore_command_hash(exec_start_argv),
        "expected_restore_cmd_sha256": _restore_command_hash(expected_restore_cmd) if expected_restore_cmd is not None else None,
        "realtime_deadline_delta_us": realtime_delta_us,
        "monotonic_deadline_delta_us": monotonic_delta_us,
        "timer": timer,
        "service": service,
    }
    return payload


def arm_independent_restore_timer(args: argparse.Namespace, out: Path, runtime_recipe: Path, deadline_ns: int) -> dict[str, Any]:
    restore_out = out / "timer-restore"
    restore_runner = runtime_recipe / "scripts/window_k1_canary.py"
    if not restore_runner.is_file() or restore_runner.is_symlink():
        raise CanaryFailed("restore runner must be an immutable archived file")
    unit = f"glm53-k1-canary-restore-{args.run_id}-{os.getpid()}"
    service_unit = unit + ".service"
    timer_unit = unit + ".timer"
    delay_sec = max(float(args.window_deadline_sec), 0.001)
    absolute_deadline_realtime_us = int((time.time() + delay_sec) * 1_000_000)
    restore_cmd = [
        args.timer_python,
        str(restore_runner),
        "--restore-only",
        "--out", str(restore_out),
        "--docker", args.docker,
        "--host-operation-lock", str(args.host_operation_lock) + ".timer",
        "--command-timeout-sec", str(min(args.command_timeout_sec, args.restore_budget_sec)),
        "--readiness-timeout-sec", str(min(args.readiness_timeout_sec, args.restore_budget_sec)),
    ]
    if args.docker_context:
        restore_cmd += ["--docker-context", args.docker_context]
    if args.health:
        restore_cmd += ["--health", str(args.health)]
    if args.api_probe:
        restore_cmd += ["--api-probe", str(args.api_probe)]
    # systemd.time calendar syntax does not accept an epoch prefixed with '@'.
    # Use an explicit UTC timestamp; systemd timer readback is validated below.
    on_calendar = dt.datetime.fromtimestamp(
        absolute_deadline_realtime_us / 1_000_000, dt.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    cmd = [
        args.systemd_run,
        "--system",
        "--unit", unit,
        "--description", f"GLM53 K1 canary restore failsafe {args.run_id}",
        "--property", "Type=oneshot",
        "--property", "CollectMode=inactive-or-failed",
        "--timer-property", "Persistent=true",
        # Calendar timers expose a realtime deadline for pre-stop readback.
        # Station's --on-active timer only populated the monotonic field.
        "--on-calendar", on_calendar,
        "--",
        *restore_cmd,
    ]
    _run_unit_command(args, cmd, out, "systemd-run restore timer")
    payload = _read_systemd_timer(args, out, timer_unit, service_unit, absolute_deadline_realtime_us, restore_cmd, deadline_ns // 1000)
    deadline_monotonic_ns = int(time.monotonic_ns() + delay_sec * 1_000_000_000)
    payload["deadline_monotonic_ns"] = deadline_monotonic_ns
    payload["restore_cmd"] = restore_cmd
    (out / "preflight").mkdir(parents=True, exist_ok=True)
    (out / "preflight" / "restore-timer-armed-readback.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_restore_obligation(args, out, Path(f"systemd:{timer_unit}"), deadline_monotonic_ns, absolute_deadline_realtime_us=absolute_deadline_realtime_us, systemd_readback=payload)
    return payload


def cancel_independent_restore_timer(out: Path, process: dict[str, Any] | None, args: argparse.Namespace | None = None) -> None:
    if process is None or args is None:
        raise CanaryFailed("cannot cancel restore timer without systemd readback")
    pre = _read_systemd_timer(
        args,
        out,
        str(process["timer_unit"]),
        str(process["service_unit"]),
        int(process["absolute_deadline_realtime_us"]),
        list(process.get("restore_cmd", [])) or None,
        int(process["deadline_monotonic_ns"]) // 1000 if process.get("deadline_monotonic_ns") is not None else None,
    )
    if pre["service"].get("ActiveState") != "inactive":
        raise CanaryFailed("restore service was not inactive before timer cancellation")
    cmd = [args.systemctl, "--system", "stop", str(process["timer_unit"])]
    _run_unit_command(args, cmd, out, "systemctl stop restore timer")
    post_timer = _systemctl_show_after_cancel(args, out, str(process["timer_unit"]), ("LoadState", "ActiveState", "Triggers"))
    post_service = _systemctl_show_after_cancel(args, out, str(process["service_unit"]), ("LoadState", "ActiveState", "ExecStart"))
    if post_timer.get("ActiveState") == "active":
        raise CanaryFailed("restore timer remained active after cancellation")
    if post_service.get("ActiveState") != "inactive":
        raise CanaryFailed("restore service became active during timer cancellation")
    (out / "preflight" / "restore-timer-cancelled-readback.json").write_text(json.dumps({"schema":"glm53-k1-canary-systemd-restore-timer-cancelled-v1", "pre": pre, "post_timer": post_timer, "post_service": post_service}, indent=2, sort_keys=True) + "\n")

def assert_pinned_artifact_hashes(runtime_recipe: Path, source_runner: Path) -> dict[str, str]:
    hashes = {
        "source_runner_sha256": sha256(source_runner),
        "patch_generator_sha256": sha256(runtime_recipe / "scripts/apply_slot_cache_instrumentation_patch.py"),
        "instrumentation_adapter_sha256": sha256(runtime_recipe / "patches/slot_cache_window_instrumentation.py"),
        "launcher_sha256": sha256(runtime_recipe / "scripts/launch-slotcache-portable.sh"),
    }
    expected = {
        "source_runner_sha256": PINNED_SOURCE_SHA,
        "patch_generator_sha256": PINNED_GENERATOR_SHA,
        "instrumentation_adapter_sha256": PINNED_ADAPTER_SHA,
        "launcher_sha256": PINNED_LAUNCHER_SHA,
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise CanaryFailed(f"pinned artifact hash mismatch: {key}")
    return hashes


def generate_patched_runner(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> tuple[Path, str]:
    generated = out / "generated-runtime" / "gpu_model_runner.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    result = v2.run_logged([
        sys.executable,
        str(runtime_recipe / "scripts/apply_slot_cache_instrumentation_patch.py"),
        "--source", str(args.source_runner),
        "--output", str(generated),
        "--expected-sha256", PINNED_SOURCE_SHA,
    ], out / "raw-logs/generate-runner.log", timeout=args.command_timeout_sec)
    v2.require_ok(result, "generate patched runner")
    digest = sha256(generated)
    if digest != PINNED_PATCHED_SHA:
        raise CanaryFailed("generated patched runner sha mismatch")
    return generated, digest


def docker(args: argparse.Namespace, out: Path, *docker_args: str, allow_fail: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    actual_timeout = args.command_timeout_sec if timeout is None else timeout
    result = v2.run_logged([args.docker, *docker_args], out / "raw-logs/docker.log", env=env, timeout=actual_timeout, redact_stdout=(docker_args[:1] == ("inspect",)))
    if not allow_fail and result.returncode != 0:
        raise CanaryFailed("docker " + " ".join(docker_args) + f" failed with exit {result.returncode}")
    return result


def candidate_name_cleanup_message() -> str:
    return (
        f"candidate container name is already owned before launch: {CANDIDATE}; "
        "safe cleanup prerequisite: verify the incumbent is running, then manually stop the candidate if running and remove or rename the stale candidate container "
        f"(for example: docker stop {CANDIDATE} || true; docker rm {CANDIDATE}) before rerunning"
    )


def _is_exact_absent_candidate_inspect(result: subprocess.CompletedProcess[str]) -> bool:
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    absent_messages = {
        f"Error: No such object: {CANDIDATE}",
        f"Error: No such container: {CANDIDATE}",
        f"error: no such object: {CANDIDATE}",
        f"error: no such container: {CANDIDATE}",
    }
    return result.returncode != 0 and combined in absent_messages


def _describe_inspect_failure(result: subprocess.CompletedProcess[str]) -> str:
    stdout = v2.redact_sensitive((result.stdout or "").strip())
    stderr = v2.redact_sensitive((result.stderr or "").strip())
    return (
        f"returncode={result.returncode} "
        f"stdout={stdout!r} "
        f"stderr={stderr!r}"
    )


def require_candidate_name_free_before_launch(args: argparse.Namespace, out: Path) -> None:
    result = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if result.returncode == 0:
        raise CanaryFailed(candidate_name_cleanup_message())
    if not _is_exact_absent_candidate_inspect(result):
        raise CanaryFailed(f"candidate container inspect failed before launch: {_describe_inspect_failure(result)}")


def ensure_candidate_stopped(args: argparse.Namespace, out: Path) -> None:
    docker(args, out, "stop", "-t", "120", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    inspected = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if inspected.returncode != 0:
        if not _is_exact_absent_candidate_inspect(inspected):
            raise CanaryFailed(f"candidate container inspect failed during restore: {_describe_inspect_failure(inspected)}")
        return
    state = inspected.stdout.strip()
    if state == "true":
        raise CanaryFailed(f"candidate did not stop: {CANDIDATE}")
    if state != "false":
        raise CanaryFailed(f"candidate stop state ambiguous: {CANDIDATE}")


def health_command(args: argparse.Namespace, runtime_recipe: Path | None = None) -> str:
    if args.health:
        return str(args.health)
    if runtime_recipe is not None:
        return str(runtime_recipe / "scripts/health-check.sh")
    return str(SCRIPT_DIR / "health-check.sh")


def restore(args: argparse.Namespace, out: Path, runtime_recipe: Path | None = None) -> int:
    status = 0
    restore_dir = out / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S%z") + f"-{time.monotonic_ns()}"
    restore_log = out / "raw-logs" / f"restore-{stamp}.log"
    try:
        with restoring_signal_shield():
            (out / "raw-logs").mkdir(parents=True, exist_ok=True)
            with (out / "raw-logs" / "docker.log").open("a") as handle:
                handle.write(f"restore_budget_sec={float(args.restore_budget_sec)}\n")
            ensure_candidate_stopped(args, out)
            env = dict(os.environ)
            if args.docker_context:
                env["DOCKER_CONTEXT"] = args.docker_context
            result = v2.run_logged([args.docker, "start", v2.INCUMBENT], restore_log, env=env, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
            if result.returncode != 0:
                status = 1
            health = v2.run_logged([health_command(args, runtime_recipe)], restore_dir / "health.txt", env=v2.health_env(os.environ), timeout=min(args.readiness_timeout_sec, args.restore_budget_sec))
            if health.returncode != 0:
                status = 1
            ps = v2.run_logged([args.docker, "ps", "--filter", f"name=^/{v2.INCUMBENT}$"], restore_dir / "docker-ps.txt", env=env, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
            if ps.returncode != 0:
                status = 1
            try:
                v2.verify_incumbent_identity(args, out)
                v2.restore_api_proof(args, out)
            except Exception as exc:  # noqa: BLE001
                restore_log.parent.mkdir(parents=True, exist_ok=True)
                with restore_log.open("a") as handle:
                    handle.write(f"restore proof failed: {exc}\n")
                status = 1
    except BaseException as exc:  # noqa: BLE001
        restore_log.parent.mkdir(parents=True, exist_ok=True)
        with restore_log.open("a") as handle:
            handle.write(f"restore exception: {exc}\n")
        status = 1
    (restore_dir / "restore-status.txt").write_text(f"WINDOW_RESTORE_STATUS={status}\n")
    return status


def gate(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> None:
    verify_archive_integrity(runtime_recipe, out)
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/window_gate.py"), str(args.root), args.run_id], out / "raw-logs/gate.log", timeout=args.command_timeout_sec)
    v2.require_ok(result, "release gate")


def candidate_env(parent: dict[str, str], out: Path, args: argparse.Namespace, patched_runner: Path, patched_sha: str, recipe_sha: str) -> dict[str, str]:
    snapshot_host = out / "snapshots" / "k1-canary"
    snapshot_host.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": parent.get("PATH", "/usr/bin:/bin"),
        "HOME": parent.get("HOME", "/home/milo"),
        "MODEL_DIR": "/home/exx/models/GLM-5.3-NVFP4-big",
        "CACHE_DIR": "/home/milo/vllm-cache",
        "API_KEY_FILE": "/home/milo/.glm_api_key",
        "KV_CACHE_MEMORY": "51539607552",
        "MAX_MODEL_LEN": "524288",
        "MAX_NUM_SEQS": "1",
        "SLOT_CACHE_PER_LAYER": "/w/configs/slots-5792-ctx512k.json",
        "AT_KEY": "slotcache-S112",
        "STATS_SEC": "0",
        "COMPILATION_CONFIG": '{"mode":3,"backend":"eager"}',
        "IMAGE": v2.IMAGE_DIGEST,
        "ROUTER": "ffi",
        "CAPTURE": "0",
        "UNPACKED": "0",
        "LOGIT_RING": "0",
        "BYPASS": "16",
        "CAPTURE_DIR": str(out),
        "CONTAINER_NAME": CANDIDATE,
        "DOCKER": args.docker,
        "NSYS": "1",
        "NSYS_OUTPUT": "/wcap/k1-canary",
        "NSYS_CONTROL": "/wcap/k1-canary-nsys-control",
        "SLOT_CACHE_QUIESCENT_SNAPSHOTS": "1",
        "SLOT_CACHE_PATCHED_RUNNER": str(patched_runner),
        "SLOT_CACHE_PATCHED_RUNNER_SHA256": patched_sha,
        "SLOT_CACHE_SOURCE_RUNNER": str(args.source_runner),
        "SLOT_CACHE_RUN_ID": "k1-canary",
        "SLOT_CACHE_SOURCE_SHA": PINNED_SOURCE_SHA,
        "SLOT_CACHE_IMAGE_SHA": v2.IMAGE_DIGEST,
        "SLOT_CACHE_RECIPE_SHA": recipe_sha,
        "SLOT_CACHE_ENGINE_GENERATION": "1",
        "SLOT_CACHE_WINDOW_STEPS": "100:164",
        "SLOT_CACHE_SNAPSHOT_DIR": "/wcap/snapshots/k1-canary",
        "SLOT_CACHE_K_MODE": "K1",
        "SLOT_CACHE_EXPECTED_LAYERS": "75",
        "SLOT_CACHE_COUNTER_SCOPE": "target_slot_cache",
        "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS": "1",
    }
    for key in ("FAKE_LOG", "FAKE_DOCKER_STATE"):
        if key in parent:
            env[key] = parent[key]
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    return env


def inspect_candidate(args: argparse.Namespace, out: Path) -> dict[str, object]:
    result = docker(args, out, "inspect", CANDIDATE)
    data = json.loads(result.stdout)[0]
    cmd = [str(part) for part in (data.get("Args") or data.get("Config", {}).get("Cmd") or [])]
    def opt(name: str) -> str | None:
        return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None
    spec_raw = opt("--speculative-config")
    spec = json.loads(spec_raw) if spec_raw else None
    proof = {
        "container_name": str(data.get("Name", "")).lstrip("/"),
        "container_id": data.get("Id") or data.get("ID"),
        "image_id": data.get("Image"),
        "config_image": data.get("Config", {}).get("Image"),
        "model": opt("--served-model-name") or opt("--model") or v2.MODEL,
        "max_model_len": opt("--max-model-len"),
        "max_num_seqs": opt("--max-num-seqs"),
        "speculative_config": spec,
        "running": bool(data.get("State", {}).get("Running")),
    }
    if proof["container_name"] != CANDIDATE:
        raise CanaryFailed("candidate container name mismatch")
    if proof["image_id"] != v2.IMAGE_DIGEST:
        raise CanaryFailed("candidate image id mismatch")
    if proof["model"] != v2.MODEL or proof["max_model_len"] != "524288" or proof["max_num_seqs"] != "1":
        raise CanaryFailed("candidate runtime shape mismatch")
    if spec != {"method": "mtp", "num_speculative_tokens": 1}:
        raise CanaryFailed("candidate speculative K1 mismatch")
    if not proof["running"]:
        raise CanaryFailed("candidate is not running")
    (out / "preflight" / "candidate-proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    return proof


def _read_snapshot_rows(out: Path) -> tuple[Path, list[dict[str, Any]]]:
    directory = out / "snapshots" / "k1-canary"
    paths = sorted(directory.glob("slot-cache-snapshots-*-*.jsonl"))
    if len(paths) != 1:
        raise CanaryFailed(f"expected exactly one adapter-produced slot-cache snapshot JSONL, found {len(paths)}")
    snap = paths[0]
    if not snap.is_file() or snap.stat().st_size == 0:
        raise CanaryFailed("missing real slot-cache snapshot JSONL")
    rows = [json.loads(line) for line in snap.read_text().splitlines() if line.strip()]
    if not rows:
        raise CanaryFailed("empty slot-cache snapshot JSONL")
    return snap, rows


def _require_all75_layers(row: dict[str, Any]) -> dict[str, dict[str, int]]:
    layers_raw = row.get("layers")
    if not isinstance(layers_raw, dict):
        raise CanaryFailed("snapshot row missing layers dict")
    expected_keys = {str(layer) for layer in EXPERT_LAYER_IDS}
    if set(layers_raw) != expected_keys:
        raise CanaryFailed("snapshot does not contain canonical all75 layer keys")
    result: dict[str, dict[str, int]] = {}
    for key, value in layers_raw.items():
        if not isinstance(value, dict):
            raise CanaryFailed("snapshot layer counters malformed")
        result[str(int(key))] = {name: int(value.get(name, 0)) for name in ("misses", "routes", "steps")}
    return result


def parse_snapshot(out: Path) -> dict[str, object]:
    snap, rows = _read_snapshot_rows(out)
    snapshots = [row for row in rows if row.get("schema") == "slot-cache-quiescent-snapshot-v1"]
    receipts = [row for row in rows if row.get("schema") == "slot-cache-quiescent-receipt-v1"]
    if len(snapshots) != 2:
        raise CanaryFailed(f"expected start/end snapshot rows, found {len(snapshots)}")
    endpoints: list[str] = []
    steps: list[int] = []
    sequences: list[int] = []
    trace_ids: list[str] = []
    layer_sets: list[dict[str, dict[str, int]]] = []
    for row in snapshots:
        metadata = row.get("metadata", {})
        provenance = row.get("provenance", {})
        endpoint = metadata.get("window_endpoint")
        engine_step = int(metadata.get("engine_step", -1))
        seq = int(provenance.get("seq", -1))
        generation = int(provenance.get("engine_generation", -1))
        trace_id = str(metadata.get("trace_id", ""))
        if metadata.get("valid_for_campaign") is not False:
            raise CanaryFailed("snapshot must remain campaign-invalid")
        if generation != 1:
            raise CanaryFailed("snapshot engine generation mismatch")
        if not trace_id.startswith(f"slotcache:k1-canary:gen:1:seq:{seq}:step:{engine_step}:"):
            raise CanaryFailed("snapshot trace id is not adapter structured trace id")
        endpoints.append(str(endpoint))
        steps.append(engine_step)
        sequences.append(seq)
        trace_ids.append(trace_id)
        layer_sets.append(_require_all75_layers(row))
    if endpoints != ["start", "end"] or steps != [100, 164] or sequences != [2, 4]:
        raise CanaryFailed("snapshot endpoint/step/sequence mismatch")
    deltas: dict[str, dict[str, int]] = {}
    for layer in map(str, EXPERT_LAYER_IDS):
        deltas[layer] = {
            name: layer_sets[1][layer][name] - layer_sets[0][layer][name]
            for name in ("misses", "routes", "steps")
        }
    return {
        "path": str(snap),
        "sha256": sha256(snap),
        "window": {"start": 100, "end": 164},
        "endpoints": endpoints,
        "endpoint_statuses": {"100:start": "snapshot", "164:end": "snapshot"},
        "sequences": sequences,
        "engine_generation": 1,
        "trace_ids": trace_ids,
        "layer_count": 75,
        "layer_deltas": deltas,
        "receipt_count": len(receipts),
        "campaign_valid": False,
    }


def verify_probe_reaches_window(path: Path, *, min_steps: int = 164) -> None:
    payload = json.loads(path.read_text())
    steps = int(payload.get("summary", {}).get("verification_steps", 0))
    if steps < min_steps:
        raise CanaryFailed(f"probe did not reach configured endpoint {min_steps}: verification_steps={steps}")


def export_nsys_stats(args: argparse.Namespace, out: Path) -> dict[str, object]:
    report = out / "k1-canary.nsys-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise CanaryFailed("missing real nsys report: k1-canary.nsys-rep")
    before = sha256(report)
    result = v2.run_logged([
        args.nsys,
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--report", "cuda_gpu_trace,cuda_api_trace",
        "--format", "csv",
        "--output", str(out / "k1_canary"),
        str(report),
    ], out / "raw-logs/nsys-stats.log", timeout=args.command_timeout_sec)
    v2.require_ok(result, "nsys stats")
    after = sha256(report)
    if after != before:
        raise CanaryFailed("nsys report changed during stats export")
    api_csv = out / "k1_canary_cuda_api_trace.csv"
    gpu_csv = out / "k1_canary_cuda_gpu_trace.csv"
    for path in (api_csv, gpu_csv):
        if not path.is_file() or path.stat().st_size == 0:
            raise CanaryFailed(f"missing nsys export: {path.name}")
    return {
        "status": "UNPROVEN",
        "reason": "raw Nsight report/export collected; CUDA/NVTX/GPU correlation requires a separate real correlator",
        "report": str(report),
        "report_sha256": before,
        "raw_nsys_report_nonempty": True,
        "cuda_api_trace_sha256": sha256(api_csv),
        "cuda_gpu_trace_sha256": sha256(gpu_csv),
    }


def run_canary(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"WINDOW_K1_CANARY_FAILED: output directory is not empty: {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    host_lock_fd: int | None = None
    runtime_recipe: Path | None = None
    restore_required = False
    restore_timer_process: dict[str, Any] | None = None
    previous_handlers: dict[int, object] = {}

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} out={out} run_id={args.run_id}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"WINDOW_K1_CANARY_FAILED: output lock already exists: {lock_path}", file=sys.stderr)
        return 1

    def _signal_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)

    try:
        runtime_recipe = archive_source(args.recipe, out)
        if args.source_runner_was_default:
            args.source_runner = runtime_recipe / ARCHIVED_SOURCE_RUNNER
        archived_v2 = load_archived_v2(runtime_recipe)
        globals()["v2"] = archived_v2
        verify_archive_integrity(runtime_recipe, out)
        host_lock_fd = v2.acquire_host_lock(args, out)
        require_candidate_name_free_before_launch(args, out)
        absolute_deadline_ns = time.monotonic_ns() + int(args.window_deadline_sec * 1_000_000_000)
        restore_timer_process = arm_independent_restore_timer(args, out, runtime_recipe, absolute_deadline_ns)
        incumbent_before = None
        v2.verify_incumbent_identity(args, out, "preflight/incumbent-proof.json")
        incumbent_before = json.loads((out / "preflight" / "incumbent-proof.json").read_text())
        v2.verify_candidate_image(args, out)
        artifact_hashes = assert_pinned_artifact_hashes(runtime_recipe, args.source_runner)
        recipe_sha = recipe_manifest_sha256(runtime_recipe)
        patched_runner, patched_sha = generate_patched_runner(args, out, runtime_recipe)
        if patched_sha != PINNED_PATCHED_SHA:
            raise CanaryFailed("patched runner not pinned")
        gate(args, out, runtime_recipe)
        restore_required = True
        docker(args, out, "stop", v2.INCUMBENT)
        gate(args, out, runtime_recipe)
        env = candidate_env(os.environ, out, args, patched_runner, patched_sha, recipe_sha)
        launch_cmd = [args.bash, str(runtime_recipe / "scripts/launch-slotcache-portable.sh"), "k1-canary", "112", "--speculative-config", SPEC_K1]
        launch = v2.run_logged(launch_cmd, out / "raw-logs/launch.log", env=env, timeout=args.command_timeout_sec)
        (out / "launch.txt").write_text(v2.redact_sensitive(launch.stdout))
        v2.require_ok(launch, "K1 canary launch")
        candidate_proof = inspect_candidate(args, out)
        v2.require_ok(v2.run_logged([health_command(args, runtime_recipe)], out / "health.txt", env=v2.health_env(os.environ), timeout=args.readiness_timeout_sec), "readiness")
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "192", "--out", str(out / "k1-canary-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), "canary warmup probe")
        verify_probe_reaches_window(out / "k1-canary-probe.json")
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / "k1-canary-nsys-control"), "START", TRACE_TOKEN], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys start")
        v2.check_control_ack(out / "nsys-control.log", "START", TRACE_TOKEN)
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "192", "--out", str(out / "k1-canary-profiled-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), "profiled canary probe")
        verify_probe_reaches_window(out / "k1-canary-profiled-probe.json")
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / "k1-canary-nsys-control"), "STOP", TRACE_TOKEN], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys stop")
        v2.check_control_ack(out / "nsys-control.log", "STOP", TRACE_TOKEN)
        snapshot = parse_snapshot(out)
        docker(args, out, "stop", "-t", "120", CANDIDATE)
        stopped = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True).stdout.strip()
        if stopped != "false":
            raise CanaryFailed(f"candidate stop state ambiguous: {CANDIDATE}")
        restore_status = restore(args, out, runtime_recipe)
        restore_required = False
        if restore_status != 0:
            return restore_status
        cancel_independent_restore_timer(out, restore_timer_process, args)
        nsys = export_nsys_stats(args, out)
        evidence = {
            "schema": "glm53-k1-canary-evidence-v1",
            "run_id": args.run_id,
            "campaign_valid": False,
            "purpose": "engine_safe_snapshot_nvtx_correlation_canary_only_not_k2_verdict",
            "source": {
                "published_source_commit": "f7912f9055d524b387d154db5defe8683a50c9ce",
                "source_manifest_sha256": sha256(out / "source-manifest.json"),
                "recipe_manifest_sha256": recipe_sha,
                "incumbent_container": v2.INCUMBENT,
                "incumbent_container_id": (incumbent_before or {}).get("observed_container_id"),
                "incumbent_image_tag": v2.IMAGE,
                "image_id": v2.IMAGE_DIGEST,
                **artifact_hashes,
                "patched_runner_sha256": patched_sha,
            },
            "candidate": candidate_proof,
            "snapshot": snapshot,
            "nvtx_correlation": nsys,
            "restore_status": json.loads((out / "restore" / "service-proof.json").read_text()),
            "rows_campaign_valid": False,
        }
        (out / "k1-canary-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print("WINDOW_K1_CANARY_COLLECTION_OK")
        return 0
    except BaseException as exc:  # noqa: BLE001
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-k1-canary-failure-v1", "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        if restore_required:
            restore(args, out, runtime_recipe)
            cancel_independent_restore_timer(out, restore_timer_process, args)
        print(f"WINDOW_K1_CANARY_FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        v2.release_host_lock(args, host_lock_fd)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass



def build_release_archive(args: argparse.Namespace) -> int:
    archive_path = args.build_release_archive.resolve()
    if archive_path.exists():
        raise CanaryFailed(f"release archive already exists: {archive_path}")
    tmp = archive_path.parent / (archive_path.name + ".tmpdir")
    if tmp.exists():
        shutil.rmtree(tmp)
    root = tmp / RUN_ID
    recipe_dst = root / "recipe"
    try:
        copied: set[str] = set()
        for path in sorted(args.recipe.rglob("*")):
            rel_path = path.relative_to(args.recipe)
            rel = rel_path.as_posix()
            if any(part in {".git", "__pycache__"} for part in rel_path.parts) or rel.endswith(".pyc"):
                continue
            if path.is_symlink():
                raise CanaryFailed(f"release archive input must not be a symlink: {rel}")
            if not path.is_file():
                continue
            dest = recipe_dst / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest, follow_symlinks=False)
            copied.add(rel)
        missing = [rel for rel in SOURCE_FILES if rel not in copied]
        if missing:
            raise CanaryFailed("missing release archive SOURCE_FILES: " + ", ".join(missing))
        manifest_files = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                manifest_files.append({"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size})
        manifest = {
            "schema": "glm53-k1-canary-release-archive-manifest-v1",
            "run_id": RUN_ID,
            "published_source_commit": "f7912f9055d524b387d154db5defe8683a50c9ce",
            "source_files_count": len(SOURCE_FILES),
            "source_files": list(SOURCE_FILES),
            "files": manifest_files,
        }
        (root / "SHA256MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_files.append({"path": "SHA256MANIFEST.json", "sha256": sha256(root / "SHA256MANIFEST.json"), "bytes": (root / "SHA256MANIFEST.json").stat().st_size})
        manifest["files"] = manifest_files
        (root / "SHA256MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with tarfile.open(archive_path, "w") as tar:
            for path in sorted(root.rglob("*")):
                tar.add(path, arcname=path.relative_to(tmp), recursive=False)
        print(json.dumps({"archive": str(archive_path), "sha256": sha256(archive_path), "manifest_sha256": sha256(root / "SHA256MANIFEST.json")}, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--build-release-archive", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--docker-context", default=None)
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--health", default=None)
    parser.add_argument("--nsys", default="/opt/nvidia/nsight-systems/2025.6.3/bin/nsys")
    parser.add_argument("--api-probe", default=None)
    parser.add_argument("--source-runner", type=Path, default=SOURCE_RUNNER)
    parser.add_argument("--systemd-run", default="systemd-run")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--timer-python", default="python3")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--sudo-command", default="sudo")
    parser.add_argument("--command-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--readiness-timeout-sec", type=float, default=v2.READINESS_TIMEOUT_SEC)
    parser.add_argument("--window-deadline-sec", type=float, default=7200.0)
    parser.add_argument("--restore-budget-sec", type=float, default=14400.0)
    parser.add_argument("--host-operation-lock", default="/tmp/glm53-k1-canary-host-operation.lock")
    args = parser.parse_args(argv)
    if args.run_id != RUN_ID:
        parser.error(f"K1 canary runner requires run id {RUN_ID}")
    if args.build_release_archive is None and args.out is None:
        parser.error("--out is required unless --build-release-archive is set")
    if args.build_release_archive is None and not args.restore_only and args.root is None:
        parser.error("--root is required unless --restore-only is set")
    args.recipe = args.recipe.resolve()
    if args.root is not None:
        args.root = args.root.resolve()
    if args.out is not None:
        args.out = args.out.resolve()
    args.source_runner_was_default = args.source_runner is None
    if args.source_runner is None:
        args.source_runner = args.recipe / ARCHIVED_SOURCE_RUNNER
    else:
        args.source_runner = args.source_runner.resolve()
    if not args.source_runner.is_file() or args.source_runner.is_symlink():
        parser.error(f"--source-runner must be an explicit existing regular file or archived recipe source: {args.source_runner}")
    if args.source_runner_was_default and "/Users/" in str(args.source_runner):
        # Local development checkouts may live under /Users, but the default itself is recipe-relative and will be rebound to restore-bundle before execution.
        pass
    return args


if __name__ == "__main__":
    _args = parse_args()
    raise SystemExit(build_release_archive(_args) if _args.build_release_archive else (restore(_args, _args.out) if _args.restore_only else run_canary(_args)))
