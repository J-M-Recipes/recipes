#!/usr/bin/env python3
"""Restore-protected C1-only 20/20 quality gate for the GLM-5.3 slot-cache K1 candidate.

This public entrypoint runs only the C1 lifecycle. It proves the incumbent before
any stop, arms an independent system-scope restore timer from archived source,
launches one K1 candidate, enforces acceptance-512 plus exact greedy 20/20, then
restores and cancels the timer only after authenticated restore proof.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import window_e1_v2 as v2  # noqa: E402

RUN_ID = "c1-quality-20260909"
CANDIDATE = "glm53-big-c1-quality-k1"
SPEC_K1 = '{"method":"mtp","num_speculative_tokens":1}'
BASELINE_SPEED = 45.65
PUBLISHED_SOURCE_COMMIT = "f7912f9055d524b387d154db5defe8683a50c9ce"
HOST_OPERATION_LOCK = "/tmp/glm53-c1-quality-host-operation.lock"

SOURCE_FILES = tuple(dict.fromkeys((
    "scripts/window_c1_quality_gate.py",
    "scripts/window_gate.py",
    "scripts/k1_canary_restore_timer.py",
    "scripts/window_k1_canary.py",
    "scripts/window_e1_v2.py",
    "scripts/greedy_equiv.py",
    "scripts/window_ab_verdict.py",
    "scripts/launch-slotcache-portable.sh",
    "scripts/dflash2_acceptance_probe.py",
    "scripts/health-check.sh",
    "configs/slots-5792-ctx512k.json",
    "sources/vllm/v1/worker/gpu_model_runner.py",
    "patches/slot_cache_window_instrumentation.py",
    "patches/slot_cache_hook.py",
    "patches/slot_cache_stats.py",
    "patches/slot_cache_profile_control.py",
    "patches/sitecustomize.py",
    "patches/exact_pin.py",
    "patches/ffi_route.py",
)))


class GateFailed(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _decode_systemd_escapes(value: str) -> str:
    return re.sub(r"\\x([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), value)


def _extract_systemd_execstart_argv(exec_start: str) -> list[str]:
    raw = exec_start.strip()
    if not raw:
        return []
    if "argv[]=" in raw:
        raw = raw.split("argv[]=", 1)[1].split(" ; ", 1)[0].rstrip("}").strip()
    return shlex.split(_decode_systemd_escapes(raw))


def _restore_command_hash(cmd: list[str]) -> str:
    return digest_json(cmd)


def _parse_systemctl_show(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _parse_realtime_usec(value: str, expected_us: int) -> int | None:
    stripped = value.strip()
    if not stripped or stripped in {"0", "n/a", "[n/a]"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        expected_utc = dt.datetime.fromtimestamp(expected_us / 1_000_000, dt.timezone.utc)
        expected_local = expected_utc.astimezone()
        allowed = {
            expected_utc.strftime("%a %Y-%m-%d %H:%M:%S UTC"),
            expected_local.strftime("%a %Y-%m-%d %H:%M:%S %Z"),
        }
        if stripped not in allowed:
            raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore timer deadline mismatch")
        return int(expected_utc.replace(microsecond=0).timestamp() * 1_000_000)


def run_unit_command(args: argparse.Namespace, argv: list[str], out: Path, label: str) -> subprocess.CompletedProcess[str]:
    actual = [args.sudo_command, *argv] if args.sudo else argv
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    if result.returncode != 0:
        raise GateFailed("RESTORE_TIMER_INVALID", f"{label} failed with exit {result.returncode}")
    return result


def systemctl_show(args: argparse.Namespace, out: Path, unit: str, props: tuple[str, ...], *, allow_not_found: bool = False) -> dict[str, str]:
    cmd = [args.systemctl, "--system", "show"]
    for prop in props:
        cmd += ["-p", prop]
    cmd.append(unit)
    actual = [args.sudo_command, *cmd] if args.sudo else cmd
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    values = _parse_systemctl_show(result.stdout)
    if allow_not_found and values.get("LoadState") == "not-found" and values.get("ActiveState") == "inactive":
        return values
    if result.returncode != 0 or values.get("LoadState") not in {"loaded", "transient"}:
        raise GateFailed("RESTORE_TIMER_INVALID", f"systemd unit not loaded: {unit}")
    return values


def read_systemd_timer(args: argparse.Namespace, out: Path, timer_unit: str, service_unit: str, deadline_us: int, expected_cmd: list[str]) -> dict[str, Any]:
    timer = systemctl_show(args, out, timer_unit, ("LoadState", "ActiveState", "Triggers", "NextElapseUSecRealtime", "NextElapseUSecMonotonic"))
    service = systemctl_show(args, out, service_unit, ("LoadState", "ActiveState", "ExecStart"))
    if timer.get("ActiveState") != "active":
        raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore timer is not active")
    if timer.get("Triggers") and timer.get("Triggers") != service_unit:
        raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore timer trigger mismatch")
    observed = _extract_systemd_execstart_argv(service.get("ExecStart", ""))
    if observed != expected_cmd:
        raise GateFailed("RESTORE_TIMER_INVALID", f"systemd restore service ExecStart mismatch: expected_sha256={_restore_command_hash(expected_cmd)} observed_sha256={_restore_command_hash(observed)}")
    if "--restore-only" not in observed:
        raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore service ExecStart does not restore-only")
    if any(arg.startswith("/Users/") for arg in observed):
        raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore service ExecStart depends on /Users")
    observed_us = _parse_realtime_usec(timer.get("NextElapseUSecRealtime", ""), deadline_us)
    if observed_us is None or abs(observed_us - deadline_us) > 2_000_000:
        raise GateFailed("RESTORE_TIMER_INVALID", "systemd restore timer deadline mismatch")
    return {
        "schema": "glm53-c1-quality-systemd-restore-timer-readback-v1",
        "scope": "system",
        "status": "armed",
        "timer_unit": timer_unit,
        "service_unit": service_unit,
        "absolute_deadline_realtime_us": deadline_us,
        "exec_start_argv": observed,
        "exec_start_sha256": _restore_command_hash(observed),
        "expected_restore_cmd_sha256": _restore_command_hash(expected_cmd),
        "timer": timer,
        "service": service,
    }


def arm_independent_restore_timer(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> dict[str, Any]:
    runner = runtime_recipe / "scripts/window_c1_quality_gate.py"
    helper = runtime_recipe / "scripts/window_e1_v2.py"
    health = runtime_recipe / "scripts/health-check.sh"
    if any(not path.is_file() or path.is_symlink() for path in (runner, helper, health)):
        raise GateFailed("RESTORE_TIMER_INVALID", "restore runner/helpers must be archived regular files")
    unit = f"glm53-c1-quality-restore-{args.run_id}-{os.getpid()}"
    timer_bundle = Path("/tmp") / f"glm53-c1-quality-restore-bundle-{args.run_id}-{os.getpid()}"
    timer_scripts = timer_bundle / "scripts"
    if timer_bundle.exists():
        shutil.rmtree(timer_bundle)
    timer_scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runner, timer_scripts / "window_c1_quality_gate.py", follow_symlinks=False)
    shutil.copy2(helper, timer_scripts / "window_e1_v2.py", follow_symlinks=False)
    shutil.copy2(health, timer_scripts / "health-check.sh", follow_symlinks=False)
    runner = timer_scripts / "window_c1_quality_gate.py"
    health = timer_scripts / "health-check.sh"
    deadline_us = int((time.time() + max(args.window_deadline_sec, 0.001)) * 1_000_000)
    restore_out = Path("/tmp") / f"glm53-c1-quality-restore-{args.run_id}-{os.getpid()}"
    restore_cmd = [
        args.timer_python,
        str(runner),
        "--restore-only",
        "--out", str(restore_out),
        "--docker", args.docker,
        "--host-operation-lock", str(args.host_operation_lock) + ".timer",
        "--command-timeout-sec", str(min(args.command_timeout_sec, args.restore_budget_sec)),
        "--readiness-timeout-sec", str(min(args.readiness_timeout_sec, args.restore_budget_sec)),
        "--health", str(health),
    ]
    if args.docker_context:
        restore_cmd += ["--docker-context", args.docker_context]

    if any(arg.startswith("/Users/") for arg in restore_cmd):
        raise GateFailed("RESTORE_TIMER_INVALID", "restore command depends on /Users")
    on_calendar = dt.datetime.fromtimestamp(deadline_us / 1_000_000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cmd = [
        args.systemd_run, "--system", "--unit", unit,
        "--description", f"GLM53 C1 quality restore failsafe {args.run_id}",
        "--property", "Type=oneshot",
        "--property", "CollectMode=inactive-or-failed",
        "--timer-property", "Persistent=true",
        "--on-calendar", on_calendar,
        "--", *restore_cmd,
    ]
    run_unit_command(args, cmd, out, "systemd-run restore timer")
    payload = read_systemd_timer(args, out, unit + ".timer", unit + ".service", deadline_us, restore_cmd)
    payload["restore_cmd"] = restore_cmd
    payload["timer_restore_bundle"] = {
        "path": str(timer_bundle),
        "runner_sha256": sha256(runner),
        "v2_helper_sha256": sha256(helper),
        "health_sha256": sha256(health),
        "no_symlinks": not any(path.is_symlink() for path in timer_bundle.rglob("*")),
    }
    preflight = out / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    (preflight / "restore-timer-armed-readback.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (preflight / "restore-obligation.json").write_text(json.dumps({
        "schema": "glm53-c1-quality-restore-obligation-v1",
        "run_id": args.run_id,
        "candidate": CANDIDATE,
        "incumbent": v2.INCUMBENT,
        "restore_required_before_stop": True,
        "release_independent_restore_only": True,
        "timer": payload,
    }, indent=2, sort_keys=True) + "\n")
    return payload


def cancel_independent_restore_timer(args: argparse.Namespace, out: Path, timer_payload: dict[str, Any] | None) -> None:
    if timer_payload is None:
        return
    pre = read_systemd_timer(args, out, str(timer_payload["timer_unit"]), str(timer_payload["service_unit"]), int(timer_payload["absolute_deadline_realtime_us"]), list(timer_payload["restore_cmd"]))
    if pre["service"].get("ActiveState") != "inactive":
        raise GateFailed("RESTORE_TIMER_INVALID", "restore service was not inactive before timer cancellation")
    run_unit_command(args, [args.systemctl, "--system", "stop", str(timer_payload["timer_unit"])], out, "systemctl stop restore timer")
    post_timer = systemctl_show(args, out, str(timer_payload["timer_unit"]), ("LoadState", "ActiveState", "Triggers"), allow_not_found=True)
    post_service = systemctl_show(args, out, str(timer_payload["service_unit"]), ("LoadState", "ActiveState", "ExecStart"), allow_not_found=True)
    if post_timer.get("ActiveState") == "active" or post_service.get("ActiveState") not in {"inactive", ""}:
        raise GateFailed("RESTORE_TIMER_INVALID", "restore timer cancellation readback failed")
    (out / "preflight" / "restore-timer-cancelled-readback.json").write_text(json.dumps({"schema": "glm53-c1-quality-systemd-restore-timer-cancelled-v1", "pre": pre, "post_timer": post_timer, "post_service": post_service}, indent=2, sort_keys=True) + "\n")


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


def archive_source(recipe: Path, out: Path) -> Path:
    missing = [rel for rel in SOURCE_FILES if not (recipe / rel).is_file()]
    if missing:
        raise GateFailed("PREP_BLOCKED", "missing required runtime dependencies: " + ", ".join(missing))
    archive = out / "restore-bundle"
    files: list[dict[str, str | int]] = []
    for rel in SOURCE_FILES:
        src = recipe / rel
        if src.is_symlink():
            raise GateFailed("PREP_BLOCKED", f"source archive input is symlink: {rel}")
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest, follow_symlinks=False)
        files.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    manifest = {
        "schema": "glm53-c1-quality-source-manifest-v1",
        "run_id": RUN_ID,
        "published_source_commit": PUBLISHED_SOURCE_COMMIT,
        "files": files,
        "helper_binding": "pending",
        "v2_helpers_imported_from": str(Path(inspect.getfile(v2)).resolve()),
        "v2_source_sha256": sha256(Path(inspect.getfile(v2)).resolve()),
        "excluded": [".git", "__pycache__", "*.pyc", "capture", "results runtime outputs"],
    }
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def bind_archived_v2(runtime_recipe: Path, out: Path) -> None:
    global v2
    helper = runtime_recipe / "scripts/window_e1_v2.py"
    spec = importlib.util.spec_from_file_location("window_e1_v2_c1_archived", helper)
    if spec is None or spec.loader is None:
        raise GateFailed("PREP_BLOCKED", "cannot load archived window_e1_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("INCUMBENT", "INCUMBENT_ID", "IMAGE_DIGEST", "IMAGE", "MODEL", "CONTROLLED_ENV"):
        if getattr(module, name, None) != getattr(v2, name, None):
            raise GateFailed("PREP_BLOCKED", f"archived v2 helper constant mismatch: {name}")
    v2 = module
    manifest_path = out / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["helper_binding"] = "archived-restore-bundle"
    manifest["v2_helpers_imported_from"] = str(helper.resolve())
    manifest["v2_source_sha256"] = sha256(helper)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_archive_integrity(runtime_recipe: Path, out: Path, expected_manifest_sha: str | None = None) -> None:
    manifest_path = out / "source-manifest.json"
    if expected_manifest_sha is not None and sha256(manifest_path) != expected_manifest_sha:
        raise GateFailed("PREP_BLOCKED", "source manifest hash changed after archive")
    manifest = json.loads(manifest_path.read_text())
    paths = [str(item.get("path")) for item in manifest.get("files", [])]
    if set(paths) != set(SOURCE_FILES) or len(paths) != len(SOURCE_FILES):
        raise GateFailed("PREP_BLOCKED", "source manifest file set mismatch")
    for item in manifest["files"]:
        rel = str(item["path"])
        target = runtime_recipe / rel
        if not target.is_file() or target.is_symlink():
            raise GateFailed("PREP_BLOCKED", f"executed source missing or symlink: {rel}")
        if sha256(target) != item["sha256"]:
            raise GateFailed("PREP_BLOCKED", f"executed source hash mismatch: {rel}")


def load_archived_verdict(runtime_recipe: Path):
    path = runtime_recipe / "scripts/window_ab_verdict.py"
    spec = importlib.util.spec_from_file_location("window_ab_verdict_c1_archived", path)
    if spec is None or spec.loader is None:
        raise GateFailed("PREP_BLOCKED", "cannot load archived window_ab_verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("validate_lane", "quality_match", "c1_gate"):
        if not callable(getattr(module, name, None)):
            raise GateFailed("PREP_BLOCKED", f"archived window_ab_verdict.py missing callable {name}")
    return module


def docker(args: argparse.Namespace, out: Path, *docker_args: str, allow_fail: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    actual_timeout = args.command_timeout_sec if timeout is None else timeout
    if args._window_deadline_at is not None:
        actual_timeout = min(float(actual_timeout), max(args._window_deadline_at - time.monotonic(), 0.001))
    result = v2.run_logged([args.docker, *docker_args], out / "raw-logs/docker.log", env=env, timeout=actual_timeout, redact_stdout=(docker_args[:1] == ("inspect",) and "-f" not in docker_args))
    if not allow_fail and result.returncode != 0:
        raise GateFailed("DOCKER_FAILED", "docker " + " ".join(docker_args) + f" failed with exit {result.returncode}")
    return result


def _is_exact_absent_candidate_inspect(result: subprocess.CompletedProcess[str]) -> bool:
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    return result.returncode != 0 and combined in {f"Error: No such object: {CANDIDATE}", f"Error: No such container: {CANDIDATE}", f"error: no such object: {CANDIDATE}", f"error: no such container: {CANDIDATE}"}


def _describe_inspect_failure(result: subprocess.CompletedProcess[str]) -> str:
    return f"returncode={result.returncode} stdout={v2.redact_sensitive((result.stdout or '').strip())!r} stderr={v2.redact_sensitive((result.stderr or '').strip())!r}"


def require_candidate_name_free_before_launch(args: argparse.Namespace, out: Path) -> None:
    result = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if result.returncode == 0:
        raise GateFailed("CANDIDATE_NAME_OWNED", f"candidate container name is already owned before launch: {CANDIDATE}")
    if not _is_exact_absent_candidate_inspect(result):
        raise GateFailed("CANDIDATE_INSPECT_AMBIGUOUS", f"candidate container inspect failed before launch: {_describe_inspect_failure(result)}")


def ensure_candidate_stopped(args: argparse.Namespace, out: Path) -> None:
    docker(args, out, "stop", "-t", "120", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    result = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if result.returncode != 0:
        if _is_exact_absent_candidate_inspect(result):
            return
        raise GateFailed("RESTORE_FAILED", f"candidate container inspect failed during restore: {_describe_inspect_failure(result)}")
    state = result.stdout.strip()
    if state == "true":
        raise GateFailed("RESTORE_FAILED", f"candidate did not stop: {CANDIDATE}")
    if state != "false":
        raise GateFailed("RESTORE_FAILED", f"candidate stop state ambiguous: {CANDIDATE}")


def health_command(args: argparse.Namespace, runtime_recipe: Path | None = None) -> str:
    if args.health:
        return str(args.health)
    recipe = runtime_recipe if runtime_recipe is not None else Path(__file__).resolve().parents[1]
    return str(recipe / "scripts/health-check.sh")


def verify_exact_incumbent_identity(
    args: argparse.Namespace,
    out: Path,
    proof_rel: str = "restore/service-proof.json",
) -> None:
    v2.verify_incumbent_identity(args, out, proof_rel=proof_rel)
    proof = json.loads((out / proof_rel).read_text())
    cmd = [str(part) for part in proof.get("args", [])]
    if option_value(cmd, "--served-model-name") != v2.MODEL:
        raise GateFailed("INCUMBENT_IDENTITY_FAILED", "incumbent served model mismatch")


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
                verify_exact_incumbent_identity(args, out)
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
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/window_gate.py"), str(args.root), args.run_id], out / "raw-logs/gate.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise GateFailed("RELEASE_GATE_FAILED", f"release gate failed with exit {result.returncode}")


def candidate_env(parent: dict[str, str], out: Path, args: argparse.Namespace) -> dict[str, str]:
    env = v2.controlled_launch_env(parent, out, args.docker, args.docker_context)
    for key in ("FAKE_DOCKER_STATE",):
        if key in parent:
            env[key] = parent[key]
    env.update({
        "CONTAINER_NAME": CANDIDATE,
        "STATS_SEC": "0",
        "NSYS": "1",
        "NSYS_OUTPUT": f"/wcap/{CANDIDATE}",
        "NSYS_CONTROL": f"/wcap/{CANDIDATE}-nsys-control",
    })
    return env


def option_value(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None


def inspect_candidate_identity(args: argparse.Namespace, out: Path, intended: dict[str, object]) -> dict[str, object]:
    result = docker(args, out, "inspect", CANDIDATE)
    data = json.loads(result.stdout)[0]
    cmd = [str(part) for part in (data.get("Args") or data.get("Config", {}).get("Cmd") or [])]
    try:
        spec = json.loads(option_value(cmd, "--speculative-config") or "null")
    except json.JSONDecodeError as exc:
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate speculative config is not JSON") from exc
    proof = dict(intended)
    proof.update({
        "observed_container_name": str(data.get("Name", "")).lstrip("/"),
        "observed_container_id": data.get("Id") or data.get("ID"),
        "observed_image_digest": data.get("Image"),
        "observed_config_image": data.get("Config", {}).get("Image"),
        "observed_model": option_value(cmd, "--model") or option_value(cmd, "--served-model-name") or v2.MODEL,
        "observed_max_model_len": option_value(cmd, "--max-model-len"),
        "observed_max_num_seqs": option_value(cmd, "--max-num-seqs"),
        "observed_speculative_config": spec,
        "running": bool(data.get("State", {}).get("Running")),
        "actual_args": cmd,
    })
    if proof["observed_container_name"] != CANDIDATE:
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate container name mismatch")
    expected_config_images = {v2.IMAGE, f"{v2.IMAGE}@{v2.IMAGE_DIGEST}"}
    if proof["observed_image_digest"] != v2.IMAGE_DIGEST or proof["observed_config_image"] not in expected_config_images:
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate image/source mismatch")
    if proof["observed_model"] != v2.MODEL or proof["observed_max_model_len"] != "524288" or proof["observed_max_num_seqs"] != "1":
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate runtime shape mismatch")
    if spec != {"method": "mtp", "num_speculative_tokens": 1}:
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate speculative K1 mismatch")
    if not proof["running"]:
        raise GateFailed("CANDIDATE_IDENTITY_FAILED", "candidate is not running")
    proof["config_digest"] = digest_json(proof)
    (out / "candidate-identity.json").write_text(json.dumps({"schema": "glm53-c1-quality-launch-identity-v1", "candidate": proof}, indent=2, sort_keys=True) + "\n")
    return proof


def validate_greedy_file(path: Path, runtime_recipe: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    module = load_archived_verdict(runtime_recipe)
    # quality_match validates exact key-set/nonempty when comparing a file to itself.
    module.quality_match(payload, payload)
    return payload


def run_greedy_capture(args: argparse.Namespace, out: Path, runtime_recipe: Path, dest: Path) -> dict[str, str]:
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), str(dest)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise GateFailed("QUALITY_FAILED", f"greedy capture failed with exit {result.returncode}")
    return validate_greedy_file(dest, runtime_recipe)


def run_greedy_compare(args: argparse.Namespace, out: Path, runtime_recipe: Path, a: Path, b: Path) -> str:
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), "--compare", str(a), str(b)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    (out / "quality" / "c1-compare.txt").write_text(result.stdout)
    if result.returncode != 0:
        raise GateFailed("QUALITY_FAILED", f"greedy compare failed with exit {result.returncode}")
    return result.stdout


def run_acceptance(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> dict[str, Any]:
    dest = out / "c1" / "acceptance-512.json"
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "512", "--out", str(dest)], out / "raw-logs/probes.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise GateFailed("ACCEPTANCE_FAILED", f"acceptance-512 failed with exit {result.returncode}")
    return json.loads(dest.read_text())


def evaluate_c1(out: Path, runtime_recipe: Path) -> dict[str, Any]:
    module = load_archived_verdict(runtime_recipe)
    reference = json.loads((out / "quality" / "incumbent-greedy.json").read_text())
    candidate = json.loads((out / "quality" / "c1-greedy.json").read_text())
    probe = json.loads((out / "c1" / "acceptance-512.json").read_text())
    try:
        gate_result = module.c1_gate(probe, reference, candidate, baseline_speed=BASELINE_SPEED)
    except Exception as exc:  # noqa: BLE001
        raise GateFailed("QUALITY_FAILED", f"C1 gate validation failed: {exc}") from exc
    (out / "c1-gate.json").write_text(json.dumps(gate_result, indent=2, sort_keys=True) + "\n")
    if gate_result.get("pass") is not True or gate_result.get("quality_pass") is not True:
        raise GateFailed("QUALITY_FAILED", f"C1 gate failed: {gate_result}")
    return gate_result


def write_receipt(args: argparse.Namespace, out: Path, gate_result: dict[str, Any], candidate_proof: dict[str, Any]) -> None:
    files = {}
    for rel in ("source-manifest.json", "preflight/incumbent-proof.json", "preflight/candidate-image-proof.json", "preflight/restore-timer-armed-readback.json", "preflight/restore-timer-cancelled-readback.json", "quality/incumbent-greedy.json", "quality/c1-greedy.json", "quality/c1-compare.txt", "c1/acceptance-512.json", "c1-gate.json", "restore/service-proof.json", "restore/models.json", "restore/completion.json"):
        path = out / rel
        if path.is_file():
            files[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    receipt = {
        "schema": "glm53-c1-quality-gate-receipt-v1",
        "run_id": args.run_id,
        "verdict": "PASS",
        "candidate": CANDIDATE,
        "incumbent": v2.INCUMBENT,
        "published_source_commit": PUBLISHED_SOURCE_COMMIT,
        "source_manifest_sha256": sha256(out / "source-manifest.json"),
        "gate": gate_result,
        "candidate_proof": candidate_proof,
        "files": files,
        "promotion_authorized": False,
        "c2_authorized": False,
    }
    (out / "c1-quality-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run_quality_gate(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"WINDOW_C1_QUALITY_FAILED: output directory is not empty: {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    output_lock = out.with_suffix(out.suffix + ".lock")
    host_fd: int | None = None
    runtime_recipe: Path | None = None
    timer_payload: dict[str, Any] | None = None
    restore_required = False
    previous_handlers: dict[int, object] = {}
    try:
        fd = os.open(output_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} out={out} run_id={args.run_id}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"WINDOW_C1_QUALITY_FAILED: output lock already exists: {output_lock}", file=sys.stderr)
        return 1

    def _signal_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)

    try:
        args._window_deadline_at = time.monotonic() + args.window_deadline_sec
        runtime_recipe = archive_source(args.recipe, out)
        bind_archived_v2(runtime_recipe, out)
        args._manifest_sha256 = sha256(out / "source-manifest.json")
        verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
        host_fd = v2.acquire_host_lock(args, out)
        require_candidate_name_free_before_launch(args, out)
        verify_exact_incumbent_identity(args, out, proof_rel="preflight/incumbent-proof.json")
        v2.verify_candidate_image(args, out)
        (out / "quality").mkdir(parents=True, exist_ok=True)
        run_greedy_capture(args, out, runtime_recipe, out / "quality" / "incumbent-greedy.json")
        timer_payload = arm_independent_restore_timer(args, out, runtime_recipe)
        gate(args, out, runtime_recipe)
        restore_required = True
        docker(args, out, "stop", v2.INCUMBENT)
        gate(args, out, runtime_recipe)
        launch_cmd = [args.bash, str(runtime_recipe / "scripts/launch-slotcache-portable.sh"), "c1-quality", "112", "--speculative-config", SPEC_K1]
        env = candidate_env(os.environ, out, args)
        intended = {"candidate": "c1", "container_name": CANDIDATE, "speculative_config": SPEC_K1, "argv": launch_cmd, "env": {k: env.get(k) for k in ("MODEL_DIR", "CACHE_DIR", "API_KEY_FILE", "KV_CACHE_MEMORY", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "SLOT_CACHE_PER_LAYER", "AT_KEY", "STATS_SEC", "COMPILATION_CONFIG", "IMAGE", "ROUTER", "CAPTURE", "UNPACKED", "LOGIT_RING", "BYPASS", "NSYS")}, "source_manifest_sha256": args._manifest_sha256}
        intended["config_digest"] = digest_json(intended)
        launch = v2.run_logged(launch_cmd, out / "raw-logs/launch-c1.log", env=env, timeout=args.command_timeout_sec)
        (out / "c1" / "launch.txt").parent.mkdir(parents=True, exist_ok=True)
        (out / "c1" / "launch.txt").write_text(v2.redact_sensitive(launch.stdout))
        if launch.returncode != 0:
            raise GateFailed("LAUNCH_FAILED", f"C1 candidate launch failed with exit {launch.returncode}")
        candidate_proof = inspect_candidate_identity(args, out, intended)
        health = v2.run_logged([health_command(args, runtime_recipe)], out / "c1" / "health.txt", env=v2.health_env(os.environ), timeout=args.readiness_timeout_sec)
        if health.returncode != 0:
            raise GateFailed("READINESS_FAILED", f"C1 readiness failed with exit {health.returncode}")
        run_acceptance(args, out, runtime_recipe)
        run_greedy_capture(args, out, runtime_recipe, out / "quality" / "c1-greedy.json")
        run_greedy_compare(args, out, runtime_recipe, out / "quality" / "incumbent-greedy.json", out / "quality" / "c1-greedy.json")
        gate_result = evaluate_c1(out, runtime_recipe)
        docker(args, out, "stop", "-t", "120", CANDIDATE)
        stopped = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True).stdout.strip()
        if stopped != "false":
            raise GateFailed("RESTORE_FAILED", f"candidate stop state ambiguous: {CANDIDATE}")
        restore_status = restore(args, out, runtime_recipe)
        restore_required = False
        if restore_status != 0:
            return restore_status
        cancel_independent_restore_timer(args, out, timer_payload)
        write_receipt(args, out, gate_result, candidate_proof)
        print("WINDOW_C1_QUALITY_GATE_PASS")
        return 0
    except GateFailed as exc:
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-c1-quality-failure-v1", "code": exc.code, "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        safe_to_cancel = not restore_required
        if restore_required:
            safe_to_cancel = restore(args, out, runtime_recipe) == 0
        if timer_payload is not None and safe_to_cancel:
            try:
                cancel_independent_restore_timer(args, out, timer_payload)
            except Exception as cancel_exc:  # noqa: BLE001
                failure = json.loads((out / "failure.json").read_text())
                failure["timer_cancellation_error"] = str(cancel_exc)
                (out / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1 if exc.code != "PREP_BLOCKED" else 2
    except BaseException as exc:  # noqa: BLE001
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-c1-quality-failure-v1", "code": "WINDOW_C1_QUALITY_FAILED", "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        safe_to_cancel = not restore_required
        if restore_required:
            safe_to_cancel = restore(args, out, runtime_recipe) == 0
        if timer_payload is not None and safe_to_cancel:
            try:
                cancel_independent_restore_timer(args, out, timer_payload)
            except Exception as cancel_exc:  # noqa: BLE001
                failure = json.loads((out / "failure.json").read_text())
                failure["timer_cancellation_error"] = str(cancel_exc)
                (out / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"WINDOW_C1_QUALITY_FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        v2.release_host_lock(args, host_fd)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            output_lock.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--docker-context", default=None)
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--health", default=None)
    parser.add_argument("--api-probe", default=None)
    parser.add_argument("--systemd-run", default="systemd-run")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--timer-python", default="python3")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--sudo-command", default="sudo")
    parser.add_argument("--command-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--readiness-timeout-sec", type=float, default=v2.READINESS_TIMEOUT_SEC)
    parser.add_argument("--window-deadline-sec", type=float, default=7200.0)
    parser.add_argument("--restore-budget-sec", type=float, default=14400.0)
    parser.add_argument("--host-operation-lock", default=HOST_OPERATION_LOCK)
    args = parser.parse_args(argv)
    if args.run_id != RUN_ID:
        parser.error(f"C1 quality runner requires run id {RUN_ID}")
    if not args.restore_only and args.root is None:
        parser.error("--root is required unless --restore-only is set")
    args.recipe = args.recipe.resolve()
    args.out = args.out.resolve()
    if args.root is not None:
        args.root = args.root.resolve()
    args._window_deadline_at = None
    args._manifest_sha256 = None
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(restore(parsed, parsed.out) if parsed.restore_only else run_quality_gate(parsed))
