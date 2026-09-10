#!/usr/bin/env python3
"""C2-only restore-protected K=2 continuation runner.

This runner consumes already-completed K1/C1 evidence by exact hash, then runs
only the C2 K=2 lane. It never reruns C1 and never authorizes promotion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
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

RUN_ID = "c2-continuation-20260909"
CANDIDATE = "glm53-big-c2-continuation-k2"
SPEC_K2 = '{"method":"mtp","num_speculative_tokens":2}'
HOST_OPERATION_LOCK = "/tmp/glm53-c2-continuation-host-operation.lock"
BASELINE_C1_SPEED = 45.747
EXPERT_LAYER_IDS = tuple(range(3, 78))
ARCHIVED_SOURCE_RUNNER = Path("sources/vllm/v1/worker/gpu_model_runner.py")
PINNED_SOURCE_SHA = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
PINNED_PATCHED_SHA = "2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4"
PINNED_GENERATOR_SHA = "8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b"
PINNED_ADAPTER_SHA = "9f0c75b25438c63511a5b2580a4c0a77520f232e2affe109dd0ba3908477e453"
PINNED_LAUNCHER_SHA = "aebe4fab6272a8ded9d2e871d5b9c536b641634ae9b10232db9fa5c33bcac04d"

EXPECTED = {
    "c1_receipt": "38a650666c747d36bd40f91d77a8d73a00bb3f4330a0b43d00fe4fa0a34bb0b8",
    "c1_gate": "0bf616eb2a7962ce54135065233d899c25fa6844fda8839aeb1ded52a1ca0994",
    "c1_acceptance": "64271ced8af2e96d3ffcb5392794fdb6157f291233bd522ce403679d01c6b7ae",
    "c1_incumbent_greedy": "c8172e6f286d6394aca925ca969dc656fb38f724addd42425f6871036c0116d5",
    "c1_candidate_greedy": "c8172e6f286d6394aca925ca969dc656fb38f724addd42425f6871036c0116d5",
    "k1_evidence": "43945bb91fa3cd0e1176d0f7e4399d4b8891e901ccebc80bdb2f4c52a0912be7",
    "k1_snapshot": "26419c77a278e86e6e5aafeb2289209573be45872965bf13222dab62da80f5bf",
}

SOURCE_FILES = tuple(dict.fromkeys((
    "scripts/window_c2_continuation.py",
    "scripts/window_c1_quality_gate.py",
    "scripts/window_k1_canary.py",
    "scripts/k1_canary_restore_timer.py",
    "scripts/window_e1_v2.py",
    "scripts/window_gate.py",
    "scripts/greedy_equiv.py",
    "scripts/window_ab_verdict.py",
    "scripts/launch-slotcache-portable.sh",
    "scripts/dflash2_acceptance_probe.py",
    "scripts/nsys_capture_control.py",
    "scripts/health-check.sh",
    "scripts/apply_slot_cache_instrumentation_patch.py",
    "configs/slots-5792-ctx512k.json",
    "patches/slot_cache_window_instrumentation.py",
    "patches/slot_cache_hook.py",
    "patches/slot_cache_stats.py",
    "patches/slot_cache_profile_control.py",
    "patches/sitecustomize.py",
    "patches/exact_pin.py",
    "patches/ffi_route.py",
    "C2-CONTINUATION-CONTRACT.md",
    "sources/vllm/v1/worker/gpu_model_runner.py",
)))


class C2Failed(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _decode_systemd_escapes(value: str) -> str:
    return re.sub(r"\\x([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), value)


def _extract_execstart(exec_start: str) -> list[str]:
    raw = exec_start.strip()
    if not raw:
        return []
    if "argv[]=" in raw:
        raw = raw.split("argv[]=", 1)[1].split(" ; ", 1)[0].rstrip("}").strip()
    return shlex.split(_decode_systemd_escapes(raw))


def _parse_show(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def _parse_realtime_usec(value: str, expected_us: int) -> int | None:
    stripped = value.strip()
    if not stripped or stripped in {"0", "n/a", "[n/a]"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        expected_utc = dt.datetime.fromtimestamp(expected_us / 1_000_000, dt.timezone.utc)
        expected_local = expected_utc.astimezone()
        if stripped not in {expected_utc.strftime("%a %Y-%m-%d %H:%M:%S UTC"), expected_local.strftime("%a %Y-%m-%d %H:%M:%S %Z")}:
            raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore timer deadline mismatch")
        return int(expected_utc.replace(microsecond=0).timestamp() * 1_000_000)


def run_unit(args: argparse.Namespace, argv: list[str], out: Path, label: str) -> subprocess.CompletedProcess[str]:
    actual = [args.sudo_command, *argv] if args.sudo else argv
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    if result.returncode != 0:
        raise C2Failed("RESTORE_TIMER_INVALID", f"{label} failed with exit {result.returncode}")
    return result


def systemctl_show(args: argparse.Namespace, out: Path, unit: str, props: tuple[str, ...], *, allow_not_found: bool = False) -> dict[str, str]:
    cmd = [args.systemctl, "--system", "show"]
    for prop in props:
        cmd += ["-p", prop]
    cmd.append(unit)
    actual = [args.sudo_command, *cmd] if args.sudo else cmd
    result = v2.run_logged(actual, out / "raw-logs/systemd.log", timeout=min(args.command_timeout_sec, 30.0))
    values = _parse_show(result.stdout)
    if allow_not_found and values.get("LoadState") == "not-found" and values.get("ActiveState") == "inactive":
        return values
    if result.returncode != 0 or values.get("LoadState") not in {"loaded", "transient"}:
        raise C2Failed("RESTORE_TIMER_INVALID", f"systemd unit not loaded: {unit}")
    return values


def read_systemd_timer(args: argparse.Namespace, out: Path, timer_unit: str, service_unit: str, deadline_us: int, expected_cmd: list[str]) -> dict[str, Any]:
    timer = systemctl_show(args, out, timer_unit, ("LoadState", "ActiveState", "Triggers", "NextElapseUSecRealtime", "NextElapseUSecMonotonic"))
    service = systemctl_show(args, out, service_unit, ("LoadState", "ActiveState", "ExecStart"))
    if timer.get("ActiveState") != "active" or (timer.get("Triggers") and timer.get("Triggers") != service_unit):
        raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore timer readback mismatch")
    exec_start = service.get("ExecStart", "")
    # Require one structured executable field; never infer it from argv[0].
    path_match = re.fullmatch(r"\{ path=(/[^\s;{}\\]+) ; argv\[\]=[^{}]* \}", exec_start.strip())
    path_fields = re.findall(r"(?:\{\s*|;\s*)path\b", exec_start)
    if not path_match or len(path_fields) != 1 or not expected_cmd or path_match[1] != expected_cmd[0]:
        raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore service ExecStart mismatch")
    observed = _extract_execstart(exec_start)
    if observed != expected_cmd:
        raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore service ExecStart mismatch")
    if "--restore-only" not in observed or any(arg.startswith("/Users/") for arg in observed):
        raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore service is not independent restore-only")
    observed_us = _parse_realtime_usec(timer.get("NextElapseUSecRealtime", ""), deadline_us)
    if observed_us is None or abs(observed_us - deadline_us) > 2_000_000:
        raise C2Failed("RESTORE_TIMER_INVALID", "systemd restore timer deadline mismatch")
    return {"schema": "glm53-c2-continuation-systemd-restore-timer-readback-v1", "scope": "system", "status": "armed", "timer_unit": timer_unit, "service_unit": service_unit, "absolute_deadline_realtime_us": deadline_us, "exec_start_argv": observed, "exec_start_sha256": digest_json(observed), "expected_restore_cmd_sha256": digest_json(expected_cmd), "timer": timer, "service": service}


def archive_source(recipe: Path, out: Path) -> Path:
    missing = [rel for rel in SOURCE_FILES if not (recipe / rel).is_file()]
    if missing:
        raise C2Failed("PREP_BLOCKED", "missing required runtime dependencies: " + ", ".join(missing))
    archive = out / "restore-bundle"
    files = []
    for rel in SOURCE_FILES:
        src = recipe / rel
        if src.is_symlink():
            raise C2Failed("PREP_BLOCKED", f"source archive input is symlink: {rel}")
        dst = archive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
        files.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    manifest = {"schema": "glm53-c2-continuation-source-manifest-v1", "run_id": RUN_ID, "files": files, "helper_binding": "pending", "v2_helpers_imported_from": str(Path(inspect.getfile(v2)).resolve()), "v2_source_sha256": sha256(Path(inspect.getfile(v2)).resolve()), "excluded": [".git", "__pycache__", "*.pyc", "runtime outputs"]}
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def bind_archived_v2(runtime_recipe: Path, out: Path) -> None:
    global v2
    helper = runtime_recipe / "scripts/window_e1_v2.py"
    spec = importlib.util.spec_from_file_location("window_e1_v2_c2_archived", helper)
    if spec is None or spec.loader is None:
        raise C2Failed("PREP_BLOCKED", "cannot load archived window_e1_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("INCUMBENT", "INCUMBENT_ID", "IMAGE_DIGEST", "IMAGE", "MODEL", "CONTROLLED_ENV"):
        if getattr(module, name, None) != getattr(v2, name, None):
            raise C2Failed("PREP_BLOCKED", f"archived v2 helper constant mismatch: {name}")
    v2 = module
    path = out / "source-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["helper_binding"] = "archived-restore-bundle"
    manifest["v2_helpers_imported_from"] = str(helper.resolve())
    manifest["v2_source_sha256"] = sha256(helper)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_archive_integrity(runtime_recipe: Path, out: Path, expected_manifest_sha: str | None = None) -> None:
    manifest_path = out / "source-manifest.json"
    if expected_manifest_sha is not None and sha256(manifest_path) != expected_manifest_sha:
        raise C2Failed("PREP_BLOCKED", "source manifest hash changed after archive")
    manifest = json.loads(manifest_path.read_text())
    paths = [str(row.get("path")) for row in manifest.get("files", [])]
    if set(paths) != set(SOURCE_FILES) or len(paths) != len(SOURCE_FILES):
        raise C2Failed("PREP_BLOCKED", "source manifest file set mismatch")
    for row in manifest["files"]:
        rel = str(row["path"])
        target = runtime_recipe / rel
        if not target.is_file() or target.is_symlink() or sha256(target) != row["sha256"]:
            raise C2Failed("PREP_BLOCKED", f"executed source hash mismatch: {rel}")


def recipe_manifest_sha256(recipe: Path) -> str:
    rows: list[str] = []
    for path in sorted(recipe.rglob("*")):
        rel_path = path.relative_to(recipe)
        if any(part in {".git", "__pycache__", "results", "capture"} for part in rel_path.parts):
            continue
        if path.is_symlink():
            raise C2Failed("PREP_BLOCKED", f"recipe artifact input must not be a symlink: {rel_path.as_posix()}")
        if not path.is_file() or rel_path.as_posix().endswith(".pyc"):
            continue
        data = path.read_bytes()
        rows.append(f"{rel_path.as_posix()}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


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
            raise C2Failed("PREP_BLOCKED", f"pinned artifact hash mismatch: {key}")
    return hashes


def generate_patched_runner(args: argparse.Namespace, out: Path, runtime_recipe: Path, source_runner: Path) -> tuple[Path, str]:
    generated = out / "generated-runtime" / "gpu_model_runner.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    result = v2.run_logged([
        sys.executable,
        str(runtime_recipe / "scripts/apply_slot_cache_instrumentation_patch.py"),
        "--source", str(source_runner),
        "--output", str(generated),
        "--expected-sha256", PINNED_SOURCE_SHA,
    ], out / "raw-logs/generate-runner.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("PREP_BLOCKED", f"generate patched runner failed with exit {result.returncode}")
    digest = sha256(generated)
    if digest != PINNED_PATCHED_SHA:
        raise C2Failed("PREP_BLOCKED", "generated patched runner sha mismatch")
    return generated, digest


def _hash_check(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise C2Failed("BOUND_EVIDENCE_INVALID", f"missing {label}")
    observed = sha256(path)
    if observed != expected:
        raise C2Failed("BOUND_EVIDENCE_INVALID", f"{label} hash mismatch: expected {expected} observed {observed}")
    return observed


def bind_prior_evidence(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    c1_root = args.c1_root.resolve()
    k1_root = args.k1_root.resolve()
    c1_receipt_path = c1_root / "c1-quality-receipt.json"
    c1_gate_path = c1_root / "c1-gate.json"
    c1_acceptance_path = c1_root / "c1/acceptance-512.json"
    inc_greedy_path = c1_root / "quality/incumbent-greedy.json"
    c1_greedy_path = c1_root / "quality/c1-greedy.json"
    k1_evidence_path = k1_root / "k1-canary-evidence.json"
    snapshot_paths = sorted(k1_root.glob("slot-cache-snapshots-*.jsonl"))
    if len(snapshot_paths) != 1:
        raise C2Failed("BOUND_EVIDENCE_INVALID", "expected exactly one K1 snapshot JSONL")

    hashes = {
        "receipt_sha256": _hash_check(c1_receipt_path, EXPECTED["c1_receipt"], "C1 receipt"),
        "gate_sha256": _hash_check(c1_gate_path, EXPECTED["c1_gate"], "C1 gate"),
        "acceptance_sha256": _hash_check(c1_acceptance_path, EXPECTED["c1_acceptance"], "C1 acceptance"),
        "incumbent_greedy_sha256": _hash_check(inc_greedy_path, EXPECTED["c1_incumbent_greedy"], "C1 incumbent greedy"),
        "candidate_greedy_sha256": _hash_check(c1_greedy_path, EXPECTED["c1_candidate_greedy"], "C1 candidate greedy"),
    }
    k1_hashes = {
        "evidence_sha256": _hash_check(k1_evidence_path, EXPECTED["k1_evidence"], "K1 evidence"),
        "snapshot_sha256": _hash_check(snapshot_paths[0], EXPECTED["k1_snapshot"], "K1 snapshot"),
    }
    c1_receipt = json.loads(c1_receipt_path.read_text())
    c1_gate = json.loads(c1_gate_path.read_text())
    c1_acceptance = json.loads(c1_acceptance_path.read_text())
    k1_evidence = json.loads(k1_evidence_path.read_text())
    if c1_receipt.get("verdict") != "PASS" or c1_receipt.get("c2_authorized") is not False:
        raise C2Failed("BOUND_EVIDENCE_INVALID", "C1 receipt must be PASS with c2_authorized=false preparation semantics")
    if c1_gate.get("pass") is not True or c1_gate.get("quality_pass") is not True or float(c1_gate.get("speed")) != BASELINE_C1_SPEED:
        raise C2Failed("BOUND_EVIDENCE_INVALID", "C1 gate internal PASS/speed mismatch")
    if c1_acceptance.get("max_tokens") != 512 or c1_acceptance.get("summary", {}).get("decode_tok_s_median") != BASELINE_C1_SPEED:
        raise C2Failed("BOUND_EVIDENCE_INVALID", "C1 acceptance exact 512/speed mismatch")
    if k1_evidence.get("campaign_valid") is not False or k1_evidence.get("nvtx_correlation", {}).get("status") != "UNPROVEN":
        raise C2Failed("BOUND_EVIDENCE_INVALID", "K1 canary evidence semantics mismatch")
    binding = {"schema": "glm53-c2-continuation-bound-prior-evidence-v1", "c1_binding": {**hashes, "root": str(c1_root), "gate": c1_gate, "c2_authorized": False}, "k1_binding": {**k1_hashes, "root": str(k1_root), "snapshot": str(snapshot_paths[0])}}
    (out / "bound-prior-evidence.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    return binding


def docker(args: argparse.Namespace, out: Path, *docker_args: str, allow_fail: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    actual_timeout = args.command_timeout_sec if timeout is None else timeout
    result = v2.run_logged([args.docker, *docker_args], out / "raw-logs/docker.log", env=env, timeout=actual_timeout, redact_stdout=(docker_args[:1] == ("inspect",) and "-f" not in docker_args))
    if not allow_fail and result.returncode != 0:
        raise C2Failed("DOCKER_FAILED", "docker " + " ".join(docker_args) + f" failed with exit {result.returncode}")
    return result


def _is_absent(result: subprocess.CompletedProcess[str]) -> bool:
    combined = "\n".join(p.strip() for p in (result.stdout, result.stderr) if p and p.strip())
    return result.returncode != 0 and combined in {f"Error: No such object: {CANDIDATE}", f"Error: No such container: {CANDIDATE}", f"error: no such object: {CANDIDATE}", f"error: no such container: {CANDIDATE}"}


def require_candidate_name_free_before_timer(args: argparse.Namespace, out: Path) -> None:
    result = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if result.returncode == 0:
        raise C2Failed("CANDIDATE_NAME_OWNED", f"candidate container name is already owned before launch: {CANDIDATE}")
    if not _is_absent(result):
        raise C2Failed("CANDIDATE_INSPECT_AMBIGUOUS", "candidate container inspect failed before launch")


def ensure_candidate_stopped(args: argparse.Namespace, out: Path) -> None:
    docker(args, out, "stop", "-t", "120", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    result = docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
    if result.returncode != 0:
        if _is_absent(result):
            return
        raise C2Failed("RESTORE_FAILED", "candidate container inspect failed during restore")
    if result.stdout.strip() != "false":
        raise C2Failed("RESTORE_FAILED", f"candidate stop state ambiguous: {CANDIDATE}")


def option_value(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None


def verify_exact_incumbent_identity(args: argparse.Namespace, out: Path, proof_rel: str) -> None:
    v2.verify_incumbent_identity(args, out, proof_rel=proof_rel)
    proof = json.loads((out / proof_rel).read_text())
    cmd = [str(p) for p in proof.get("args", [])]
    if option_value(cmd, "--served-model-name") != v2.MODEL and option_value(cmd, "--model") != v2.MODEL:
        raise C2Failed("INCUMBENT_IDENTITY_FAILED", "incumbent served model mismatch")


def verify_candidate_image(args: argparse.Namespace, out: Path) -> None:
    v2.verify_candidate_image(args, out)


def candidate_env(parent: dict[str, str], out: Path, args: argparse.Namespace, patched_runner: Path, patched_sha: str, source_runner: Path, recipe_sha: str) -> dict[str, str]:
    env = v2.controlled_launch_env(parent, out, args.docker, args.docker_context)
    for key in ("FAKE_DOCKER_STATE",):
        if key in parent:
            env[key] = parent[key]
    env.update({
        "IMAGE": v2.IMAGE_DIGEST,
        "SLOT_CACHE_IMAGE_SHA": v2.IMAGE_DIGEST,
        "CONTAINER_NAME": CANDIDATE,
        "STATS_SEC": "0",
        "NSYS": "1",
        "NSYS_OUTPUT": f"/wcap/{CANDIDATE}",
        "NSYS_CONTROL": f"/wcap/{CANDIDATE}-nsys-control",
        "SLOT_CACHE_QUIESCENT_SNAPSHOTS": "1",
        "SLOT_CACHE_SNAPSHOT_DIR": "/wcap/snapshots/c2-continuation",
        "SLOT_CACHE_RUN_ID": "c2-continuation",
        "SLOT_CACHE_K_MODE": "K2",
        "SLOT_CACHE_EXPECTED_LAYERS": "75",
        "SLOT_CACHE_WINDOW_STEPS": "100:164",
        "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS": "1",
        "SLOT_CACHE_PATCHED_RUNNER": str(patched_runner),
        "SLOT_CACHE_PATCHED_RUNNER_SHA256": patched_sha,
        "SLOT_CACHE_SOURCE_RUNNER": str(source_runner),
        "SLOT_CACHE_SOURCE_SHA": PINNED_SOURCE_SHA,
        "SLOT_CACHE_RECIPE_SHA": recipe_sha,
        "SLOT_CACHE_ENGINE_GENERATION": "1",
        "SLOT_CACHE_COUNTER_SCOPE": "target_slot_cache",
    })
    return env


def inspect_candidate_identity(args: argparse.Namespace, out: Path, intended: dict[str, object]) -> dict[str, object]:
    data = json.loads(docker(args, out, "inspect", CANDIDATE).stdout)[0]
    cmd = [str(p) for p in (data.get("Args") or data.get("Config", {}).get("Cmd") or [])]
    try:
        spec = json.loads(option_value(cmd, "--speculative-config") or "null")
    except json.JSONDecodeError as exc:
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate speculative config is not JSON") from exc
    observed_model = option_value(cmd, "--model") or option_value(cmd, "--served-model-name")
    proof = dict(intended)
    proof.update({"observed_container_name": str(data.get("Name", "")).lstrip("/"), "observed_container_id": data.get("Id") or data.get("ID"), "observed_image_digest": data.get("Image"), "observed_config_image": data.get("Config", {}).get("Image"), "observed_model": observed_model, "observed_max_model_len": option_value(cmd, "--max-model-len"), "observed_max_num_seqs": option_value(cmd, "--max-num-seqs"), "observed_speculative_config": spec, "running": bool(data.get("State", {}).get("Running")), "actual_args": cmd})
    if proof["observed_container_name"] != CANDIDATE:
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate container name mismatch")
    expected_config_images = {v2.IMAGE, v2.IMAGE_DIGEST, f"{v2.IMAGE}@{v2.IMAGE_DIGEST}"}
    if proof["observed_image_digest"] != v2.IMAGE_DIGEST or not isinstance(proof["observed_config_image"], str) or proof["observed_config_image"] not in expected_config_images:
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate image/source mismatch")
    if proof["observed_model"] != v2.MODEL or proof["observed_max_model_len"] != "524288" or proof["observed_max_num_seqs"] != "1":
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate runtime shape mismatch")
    if spec != {"method": "mtp", "num_speculative_tokens": 2}:
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate speculative K2 mismatch")
    if not proof["running"]:
        raise C2Failed("CANDIDATE_IDENTITY_FAILED", "candidate is not running")
    proof["config_digest"] = digest_json(proof)
    (out / "candidate-identity.json").write_text(json.dumps({"schema": "glm53-c2-continuation-launch-identity-v1", "candidate": proof}, indent=2, sort_keys=True) + "\n")
    return proof


def health_command(args: argparse.Namespace, runtime_recipe: Path | None = None) -> str:
    if args.health:
        return str(args.health)
    recipe = runtime_recipe if runtime_recipe is not None else Path(__file__).resolve().parents[1]
    return str(recipe / "scripts/health-check.sh")


def check_deadline(args: argparse.Namespace, out: Path, what: str, *, restoring: bool = False) -> None:
    if restoring or args._window_deadline_at is None:
        return
    if time.monotonic() >= args._window_deadline_at:
        (out / "raw-logs").mkdir(parents=True, exist_ok=True)
        with (out / "raw-logs/docker.log").open("a") as h:
            h.write(f"restore_budget_sec={float(args.restore_budget_sec)}\n")
        raise C2Failed("WINDOW_DEADLINE_EXCEEDED", f"deadline before {what}; restore_budget_sec={float(args.restore_budget_sec)}")


def gate(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> None:
    check_deadline(args, out, "release gate")
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/window_gate.py"), str(args.root), args.run_id], out / "raw-logs/gate.log", timeout=min(args.command_timeout_sec, max(args._window_deadline_at - time.monotonic(), 0.001)) if args._window_deadline_at else args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("RELEASE_GATE_FAILED", f"release gate failed with exit {result.returncode}")


def arm_restore_timer(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> dict[str, Any]:
    try:
        resolved_python = shutil.which(args.timer_python)
        if resolved_python is None:
            raise ValueError("executable not found")
        timer_python = Path(resolved_python).resolve(strict=True)
        if not timer_python.is_file() or not os.access(timer_python, os.X_OK):
            raise ValueError("not an executable regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise C2Failed("RESTORE_TIMER_INVALID", f"cannot resolve timer Python executable: {args.timer_python}") from exc
    runner = runtime_recipe / "scripts/window_c2_continuation.py"
    helper = runtime_recipe / "scripts/window_e1_v2.py"
    health = runtime_recipe / "scripts/health-check.sh"
    if any(not p.is_file() or p.is_symlink() for p in (runner, helper, health)):
        raise C2Failed("RESTORE_TIMER_INVALID", "restore runner/helpers must be archived regular files")
    bundle = Path("/tmp") / f"glm53-c2-continuation-restore-bundle-{args.run_id}-{os.getpid()}"
    scripts = bundle / "scripts"
    shutil.rmtree(bundle, ignore_errors=True)
    scripts.mkdir(parents=True, exist_ok=True)
    for src in (runner, helper, health):
        shutil.copy2(src, scripts / src.name, follow_symlinks=False)
    restore_runner = scripts / "window_c2_continuation.py"
    restore_health = scripts / "health-check.sh"
    unit = f"glm53-c2-continuation-restore-{args.run_id}-{os.getpid()}"
    deadline_us = int((time.time() + max(args.window_deadline_sec, 0.001)) * 1_000_000)
    restore_out = Path("/tmp") / f"glm53-c2-continuation-restore-{args.run_id}-{os.getpid()}"
    restore_cmd = [str(timer_python), str(restore_runner), "--restore-only", "--out", str(restore_out), "--docker", args.docker, "--host-operation-lock", str(args.host_operation_lock) + ".timer", "--command-timeout-sec", str(min(args.command_timeout_sec, args.restore_budget_sec)), "--readiness-timeout-sec", str(min(args.readiness_timeout_sec, args.restore_budget_sec)), "--health", str(restore_health)]
    if args.docker_context:
        restore_cmd += ["--docker-context", args.docker_context]
    if any(arg.startswith("/Users/") for arg in restore_cmd):
        raise C2Failed("RESTORE_TIMER_INVALID", "restore command depends on /Users")
    cal = dt.datetime.fromtimestamp(deadline_us / 1_000_000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_unit(args, [args.systemd_run, "--system", "--unit", unit, "--description", f"GLM53 C2 continuation restore failsafe {args.run_id}", "--property", "Type=oneshot", "--property", "CollectMode=inactive-or-failed", "--timer-property", "Persistent=true", "--on-calendar", cal, "--", *restore_cmd], out, "systemd-run restore timer")
    payload = read_systemd_timer(args, out, unit + ".timer", unit + ".service", deadline_us, restore_cmd)
    payload["restore_cmd"] = restore_cmd
    payload["timer_restore_bundle"] = {"path": str(bundle), "runner_sha256": sha256(restore_runner), "v2_helper_sha256": sha256(scripts / "window_e1_v2.py"), "health_sha256": sha256(restore_health), "no_symlinks": not any(p.is_symlink() for p in bundle.rglob("*"))}
    pre = out / "preflight"; pre.mkdir(parents=True, exist_ok=True)
    (pre / "restore-timer-armed-readback.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (pre / "restore-obligation.json").write_text(json.dumps({"schema": "glm53-c2-continuation-restore-obligation-v1", "run_id": args.run_id, "candidate": CANDIDATE, "incumbent": v2.INCUMBENT, "restore_required_before_stop": True, "release_independent_restore_only": True, "timer": payload}, indent=2, sort_keys=True) + "\n")
    return payload


def cancel_restore_timer(args: argparse.Namespace, out: Path, timer: dict[str, Any] | None) -> None:
    if timer is None:
        return
    pre = read_systemd_timer(args, out, str(timer["timer_unit"]), str(timer["service_unit"]), int(timer["absolute_deadline_realtime_us"]), list(timer["restore_cmd"]))
    if pre["service"].get("ActiveState") != "inactive":
        raise C2Failed("RESTORE_TIMER_INVALID", "restore service was not inactive before timer cancellation")
    run_unit(args, [args.systemctl, "--system", "stop", str(timer["timer_unit"])], out, "systemctl stop restore timer")
    post_timer = systemctl_show(args, out, str(timer["timer_unit"]), ("LoadState", "ActiveState", "Triggers"), allow_not_found=True)
    post_service = systemctl_show(args, out, str(timer["service_unit"]), ("LoadState", "ActiveState", "ExecStart"), allow_not_found=True)
    if post_timer.get("ActiveState") == "active" or post_service.get("ActiveState") not in {"inactive", ""}:
        raise C2Failed("RESTORE_TIMER_INVALID", "restore timer cancellation readback failed")
    (out / "preflight" / "restore-timer-cancelled-readback.json").write_text(json.dumps({"schema": "glm53-c2-continuation-systemd-restore-timer-cancelled-v1", "pre": pre, "post_timer": post_timer, "post_service": post_service}, indent=2, sort_keys=True) + "\n")


@contextmanager
def restoring_signal_shield():
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous:
        signal.signal(s, signal.SIG_IGN)
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def restore(args: argparse.Namespace, out: Path, runtime_recipe: Path | None = None) -> int:
    status = 0
    restore_dir = out / "restore"; restore_dir.mkdir(parents=True, exist_ok=True)
    log = out / "raw-logs" / ("restore-" + time.strftime("%Y%m%dT%H%M%S%z") + f"-{time.monotonic_ns()}.log")
    try:
        with restoring_signal_shield():
            (out / "raw-logs").mkdir(parents=True, exist_ok=True)
            with (out / "raw-logs/docker.log").open("a") as h:
                h.write(f"restore_budget_sec={float(args.restore_budget_sec)}\n")
            ensure_candidate_stopped(args, out)
            env = dict(os.environ)
            if args.docker_context: env["DOCKER_CONTEXT"] = args.docker_context
            if v2.run_logged([args.docker, "start", v2.INCUMBENT], log, env=env, timeout=min(args.command_timeout_sec, args.restore_budget_sec)).returncode != 0:
                status = 1
            if v2.run_logged([health_command(args, runtime_recipe)], restore_dir / "health.txt", env=v2.health_env(os.environ), timeout=min(args.readiness_timeout_sec, args.restore_budget_sec)).returncode != 0:
                status = 1
            if v2.run_logged([args.docker, "ps", "--filter", f"name=^/{v2.INCUMBENT}$"], restore_dir / "docker-ps.txt", env=env, timeout=min(args.command_timeout_sec, args.restore_budget_sec)).returncode != 0:
                status = 1
            try:
                verify_exact_incumbent_identity(args, out, "restore/service-proof.json")
                v2.restore_api_proof(args, out)
            except Exception as exc:  # noqa: BLE001
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as h: h.write(f"restore proof failed: {exc}\n")
                status = 1
    except BaseException as exc:  # noqa: BLE001
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as h: h.write(f"restore exception: {exc}\n")
        status = 1
    (restore_dir / "restore-status.txt").write_text(f"WINDOW_RESTORE_STATUS={status}\n")
    return status


def run_greedy_capture(args: argparse.Namespace, out: Path, runtime_recipe: Path, dest: Path) -> dict[str, str]:
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), str(dest)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("QUALITY_FAILED", f"greedy capture failed with exit {result.returncode}")
    payload = json.loads(dest.read_text())
    if set(payload) != {str(i) for i in range(20)} or any(not isinstance(v, str) or not v.replace("\u241f", "").strip() for v in payload.values()):
        raise C2Failed("QUALITY_FAILED", "C2 greedy output invalid")
    return payload


def run_greedy_compare(args: argparse.Namespace, out: Path, runtime_recipe: Path, a: Path, b: Path) -> None:
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), "--compare", str(a), str(b)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    (out / "quality/c2-compare.txt").write_text(result.stdout)
    if result.returncode != 0 or "20/20" not in result.stdout:
        raise C2Failed("QUALITY_FAILED", "C2 greedy 20/20 failed")


def run_acceptance(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> dict[str, Any]:
    dest = out / "c2/acceptance-512.json"
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "512", "--out", str(dest)], out / "raw-logs/probes.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("ACCEPTANCE_FAILED", f"acceptance-512 failed with exit {result.returncode}")
    return json.loads(dest.read_text())


def run_profiled_probe(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> dict[str, Any]:
    dest = out / "c2/profiled-192.json"
    result = v2.run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "192", "--out", str(dest)], out / "raw-logs/probes.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("ACCEPTANCE_FAILED", f"profiled-192 failed with exit {result.returncode}")
    payload = json.loads(dest.read_text())
    steps = int(payload.get("summary", {}).get("verification_steps", 0))
    if steps < 164:
        raise C2Failed("ACCEPTANCE_FAILED", f"profiled workload did not reach endpoint 164: verification_steps={steps}")
    return payload


def _coerce_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise C2Failed("SNAPSHOT_INVALID", f"snapshot counter is not an integer: {label}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise C2Failed("SNAPSHOT_INVALID", f"snapshot counter is not an integer: {label}") from exc
    if parsed < 0:
        raise C2Failed("SNAPSHOT_INVALID", f"snapshot counter is negative: {label}")
    return parsed


def _load_instrumentation_helper(runtime_recipe: Path | None = None) -> Any:
    recipe = runtime_recipe if runtime_recipe is not None else Path(__file__).resolve().parents[1]
    helper = recipe / "patches/slot_cache_window_instrumentation.py"
    spec = importlib.util.spec_from_file_location("slot_cache_window_instrumentation_c2_contract", helper)
    if spec is None or spec.loader is None:
        raise C2Failed("SNAPSHOT_INVALID", "cannot load archived slot-cache instrumentation helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_c2_snapshot_trace_id(seq: int, step: int, runtime_recipe: Path | None = None) -> str:
    helper = _load_instrumentation_helper(runtime_recipe)
    return helper.make_slot_cache_trace_id(
        run_id="c2-continuation",
        engine_generation=1,
        seq=seq,
        engine_step=step,
        boundary="step_complete",
        phase_flags={"canonical_phase": "spec_verify", "has_spec_verify": True},
        graph_mode="unknown",
        k_mode="K2",
        has_drafter_config=True,
        drafter_runs_model_forward=False,
    )


def parse_snapshot(out: Path, runtime_recipe: Path | None = None) -> dict[str, Any]:
    paths = sorted((out / "snapshots/c2-continuation").glob("slot-cache-snapshots-*.jsonl"))
    if len(paths) != 1:
        raise C2Failed("SNAPSHOT_INVALID", f"expected exactly one C2 snapshot JSONL, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    snaps = [row for row in rows if row.get("schema") == "slot-cache-quiescent-snapshot-v1"]
    if len(snaps) != 2:
        raise C2Failed("SNAPSHOT_INVALID", "expected start/end snapshot rows")
    endpoints=[]; steps=[]; sequences=[]; trace_ids=[]
    expected = {str(i) for i in EXPERT_LAYER_IDS}
    layer_sets=[]
    for row in snaps:
        meta = row.get("metadata", {}); prov = row.get("provenance", {})
        endpoint = meta.get("window_endpoint")
        step = _coerce_nonnegative_int(meta.get("engine_step"), "engine_step")
        seq = _coerce_nonnegative_int(prov.get("seq"), "provenance.seq")
        generation = _coerce_nonnegative_int(prov.get("engine_generation"), "provenance.engine_generation")
        if meta.get("valid_for_campaign") is not False or generation != 1:
            raise C2Failed("SNAPSHOT_INVALID", "snapshot provenance invalid")
        if prov.get("run_id") != "c2-continuation" or prov.get("phase") != "spec_verify" or prov.get("k_mode") != "K2":
            raise C2Failed("SNAPSHOT_INVALID", "snapshot provenance contract mismatch")
        trace_id = str(meta.get("trace_id", ""))
        if trace_id != _expected_c2_snapshot_trace_id(seq, step, runtime_recipe):
            raise C2Failed("SNAPSHOT_INVALID", "snapshot trace id is not exact finalized dynamic trace id")
        layers = row.get("layers")
        if not isinstance(layers, dict) or set(layers) != expected:
            raise C2Failed("SNAPSHOT_INVALID", "snapshot does not contain canonical all75 layer keys")
        canonical_layers: dict[str, dict[str, int]] = {}
        for key in expected:
            value = layers.get(key)
            if not isinstance(value, dict):
                raise C2Failed("SNAPSHOT_INVALID", "snapshot layer counters malformed")
            canonical_layers[key] = {name: _coerce_nonnegative_int(value.get(name), f"layer {key} {name}") for name in ("misses", "routes", "steps")}
        endpoints.append(endpoint); steps.append(step); sequences.append(seq); trace_ids.append(trace_id)
        layer_sets.append(canonical_layers)
    if endpoints != ["start", "end"] or steps != [100, 164] or sequences != [2, 4]:
        raise C2Failed("SNAPSHOT_INVALID", "snapshot endpoint/step/sequence mismatch")
    deltas: dict[str, dict[str, int]] = {}
    for layer in map(str, EXPERT_LAYER_IDS):
        layer_delta = {name: layer_sets[1][layer][name] - layer_sets[0][layer][name] for name in ("misses", "routes", "steps")}
        if any(value < 0 for value in layer_delta.values()):
            raise C2Failed("SNAPSHOT_INVALID", "snapshot layer counters are not monotonic")
        deltas[layer] = layer_delta
    return {"path": str(paths[0]), "sha256": sha256(paths[0]), "endpoints": endpoints, "endpoint_statuses": {"100:start": "snapshot", "164:end": "snapshot"}, "window": {"start": 100, "end": 164}, "sequences": sequences, "engine_generation": 1, "trace_ids": trace_ids, "layer_count": 75, "layer_deltas": deltas, "campaign_valid": False}


def export_nsys_stats(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    rep = out / f"{CANDIDATE}.nsys-rep"
    if not rep.is_file() or rep.stat().st_size == 0:
        raise C2Failed("NSYS_INVALID", f"missing real nsys report: {rep.name}")
    before = sha256(rep)
    result = v2.run_logged([args.nsys, "stats", "--force-export=true", "--force-overwrite=true", "--report", "cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_gpu_trace,cuda_api_trace", "--format", "csv", "--output", str(out / "c2"), str(rep)], out / "raw-logs/nsys-stats.log", timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise C2Failed("NSYS_INVALID", f"nsys stats failed with exit {result.returncode}")
    if sha256(rep) != before:
        raise C2Failed("NSYS_INVALID", "nsys report changed during stats export")
    csvs = {}
    for suffix in ("cuda_gpu_kern_sum", "cuda_kern_exec_sum", "cuda_gpu_trace", "cuda_api_trace"):
        path = out / f"c2_{suffix}.csv"
        if not path.is_file() or path.stat().st_size == 0:
            raise C2Failed("NSYS_INVALID", f"missing nsys export: {path.name}")
        csvs[path.name] = sha256(path)
    return {"status": "UNPROVEN", "reason": "Nsight CUDA/NVTX/GPU correlation requires exact dynamic trace IDs linked to GPU events; this runner fails closed unless proven by future correlator", "raw_nsys_report_nonempty": True, "report_sha256": before, "exported_csv_sha256": csvs}


def evaluate_measurement(c2_probe: dict[str, Any], reference: dict[str, str], c2_quality: dict[str, str]) -> dict[str, Any]:
    try:
        speed = float(c2_probe.get("summary", {}).get("decode_tok_s_median"))
    except (TypeError, ValueError) as exc:
        raise C2Failed("MEASUREMENT_INVALID", "C2 speed is not numeric") from exc
    if not math.isfinite(speed) or speed <= 0:
        raise C2Failed("MEASUREMENT_INVALID", "C2 speed must be finite and positive")
    g2 = speed >= BASELINE_C1_SPEED * 1.05
    g3 = reference == c2_quality and set(reference) == {str(i) for i in range(20)}
    return {"schema": "glm53-c2-continuation-measurement-v1", "c1_speed_bound": BASELINE_C1_SPEED, "c2_speed": speed, "g2_pass": g2, "g3_pass": g3, "measurement_gates_pass": bool(g2 and g3), "promotion_authorized": False, "verdict": "INCONCLUSIVE"}


def run_c2(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"WINDOW_C2_CONTINUATION_FAILED: output directory is not empty: {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    host_fd: int | None = None; runtime_recipe: Path | None = None; timer: dict[str, Any] | None = None; restore_required = False
    previous: dict[int, object] = {}
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"pid={os.getpid()} out={out} run_id={args.run_id}\n".encode()); os.close(fd)
    except FileExistsError:
        print(f"WINDOW_C2_CONTINUATION_FAILED: output lock already exists: {lock_path}", file=sys.stderr); return 1
    def handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"received signal {signum}")
    for s in (signal.SIGINT, signal.SIGTERM):
        previous[s] = signal.getsignal(s); signal.signal(s, handler)
    try:
        args._window_deadline_at = time.monotonic() + args.window_deadline_sec
        prior = bind_prior_evidence(args, out)
        runtime_recipe = archive_source(args.recipe, out)
        bind_archived_v2(runtime_recipe, out)
        args._manifest_sha256 = sha256(out / "source-manifest.json")
        verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
        host_fd = v2.acquire_host_lock(args, out)
        require_candidate_name_free_before_timer(args, out)
        verify_exact_incumbent_identity(args, out, "preflight/incumbent-proof.json")
        verify_candidate_image(args, out)
        (out / "quality").mkdir(exist_ok=True)
        shutil.copy2(args.c1_root / "quality/incumbent-greedy.json", out / "quality/incumbent-greedy.json", follow_symlinks=False)
        recipe_sha = recipe_manifest_sha256(runtime_recipe)
        source_runner = runtime_recipe / ARCHIVED_SOURCE_RUNNER
        artifact_hashes = assert_pinned_artifact_hashes(runtime_recipe, source_runner)
        patched_runner, patched_sha = generate_patched_runner(args, out, runtime_recipe, source_runner)
        timer = arm_restore_timer(args, out, runtime_recipe)
        gate(args, out, runtime_recipe)
        restore_required = True
        docker(args, out, "stop", v2.INCUMBENT)
        gate(args, out, runtime_recipe)
        env = candidate_env(os.environ, out, args, patched_runner, patched_sha, source_runner, recipe_sha)
        (out / "snapshots" / "c2-continuation").mkdir(parents=True, exist_ok=False)
        launch_cmd = [args.bash, str(runtime_recipe / "scripts/launch-slotcache-portable.sh"), "c2-continuation", "112", "--speculative-config", SPEC_K2]
        intended = {"candidate": "c2", "container_name": CANDIDATE, "speculative_config": SPEC_K2, "argv": launch_cmd, "env": {k: env.get(k) for k in ("MODEL_DIR", "CACHE_DIR", "API_KEY_FILE", "KV_CACHE_MEMORY", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "SLOT_CACHE_PER_LAYER", "AT_KEY", "STATS_SEC", "COMPILATION_CONFIG", "IMAGE", "ROUTER", "CAPTURE", "UNPACKED", "LOGIT_RING", "BYPASS", "NSYS", "SLOT_CACHE_QUIESCENT_SNAPSHOTS", "SLOT_CACHE_EXPECTED_LAYERS", "SLOT_CACHE_K_MODE", "SLOT_CACHE_IMAGE_SHA", "SLOT_CACHE_PATCHED_RUNNER", "SLOT_CACHE_PATCHED_RUNNER_SHA256", "SLOT_CACHE_SOURCE_RUNNER", "SLOT_CACHE_SOURCE_SHA", "SLOT_CACHE_RECIPE_SHA", "SLOT_CACHE_ENGINE_GENERATION", "SLOT_CACHE_COUNTER_SCOPE")}, "source_manifest_sha256": args._manifest_sha256}
        intended["config_digest"] = digest_json(intended)
        launch = v2.run_logged(launch_cmd, out / "raw-logs/launch-c2.log", env=env, timeout=args.command_timeout_sec)
        (out / "c2/launch.txt").parent.mkdir(parents=True, exist_ok=True)
        (out / "c2/launch.txt").write_text(v2.redact_sensitive(launch.stdout))
        if launch.returncode != 0:
            raise C2Failed("LAUNCH_FAILED", f"C2 candidate launch failed with exit {launch.returncode}")
        candidate = inspect_candidate_identity(args, out, intended)
        if v2.run_logged([health_command(args, runtime_recipe)], out / "c2/health.txt", env=v2.health_env(os.environ), timeout=args.readiness_timeout_sec).returncode != 0:
            raise C2Failed("READINESS_FAILED", "C2 readiness failed")
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / f"{CANDIDATE}-nsys-control"), "START", "c2-continuation"], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys start")
        v2.check_control_ack(out / "nsys-control.log", "START", "c2-continuation")
        run_profiled_probe(args, out, runtime_recipe)
        v2.require_ok(v2.run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / f"{CANDIDATE}-nsys-control"), "STOP", "c2-continuation"], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys stop")
        v2.check_control_ack(out / "nsys-control.log", "STOP", "c2-continuation")
        c2_probe = run_acceptance(args, out, runtime_recipe)
        if c2_probe.get("max_tokens") != 512:
            raise C2Failed("ACCEPTANCE_FAILED", "C2 acceptance was not exact 512")
        c2_quality = run_greedy_capture(args, out, runtime_recipe, out / "quality/c2-greedy.json")
        run_greedy_compare(args, out, runtime_recipe, out / "quality/incumbent-greedy.json", out / "quality/c2-greedy.json")
        snapshot = parse_snapshot(out, runtime_recipe)
        docker(args, out, "stop", "-t", "120", CANDIDATE)
        if docker(args, out, "inspect", "-f", "{{.State.Running}}", CANDIDATE, allow_fail=True).stdout.strip() != "false":
            raise C2Failed("RESTORE_FAILED", f"candidate stop state ambiguous: {CANDIDATE}")
        restore_status = restore(args, out, runtime_recipe)
        restore_required = False
        if restore_status != 0:
            return restore_status
        nsys = export_nsys_stats(args, out)
        reference = json.loads((out / "quality/incumbent-greedy.json").read_text())
        measurement = evaluate_measurement(c2_probe, reference, c2_quality)
        receipt = {"schema": "glm53-c2-continuation-receipt-v1", "run_id": args.run_id, "candidate": CANDIDATE, "incumbent": v2.INCUMBENT, "promotion_authorized": False, "c2_authorized_by_prior_c1": False, "source_manifest_sha256": sha256(out / "source-manifest.json"), "source": {"recipe_manifest_sha256": recipe_sha, **artifact_hashes, "patched_runner_sha256": patched_sha}, "c1_binding": prior["c1_binding"], "k1_binding": prior["k1_binding"], "candidate_proof": candidate, "measurement": measurement, "snapshot": snapshot, "nvtx_correlation": nsys, "files": {rel: {"sha256": sha256(out / rel), "bytes": (out / rel).stat().st_size} for rel in ("source-manifest.json", "bound-prior-evidence.json", "generated-runtime/gpu_model_runner.py", "c2/profiled-192.json", "c2/acceptance-512.json", "quality/incumbent-greedy.json", "quality/c2-greedy.json", "quality/c2-compare.txt", "restore/service-proof.json", "restore/models.json", "restore/completion.json") if (out / rel).is_file()}}
        (out / "c2-continuation-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        cancel_restore_timer(args, out, timer)
        print("WINDOW_C2_CONTINUATION_COLLECTION_OK")
        return 0
    except C2Failed as exc:
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-c2-continuation-failure-v1", "code": exc.code, "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        safe_to_cancel = not restore_required
        if restore_required:
            safe_to_cancel = restore(args, out, runtime_recipe) == 0
        if timer is not None and safe_to_cancel:
            try: cancel_restore_timer(args, out, timer)
            except Exception as cancel_exc:  # noqa: BLE001
                failure = json.loads((out / "failure.json").read_text()); failure["timer_cancellation_error"] = str(cancel_exc); (out / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2 if exc.code == "PREP_BLOCKED" else 1
    except BaseException as exc:  # noqa: BLE001
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-c2-continuation-failure-v1", "code": "WINDOW_C2_CONTINUATION_FAILED", "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        safe_to_cancel = not restore_required
        if restore_required:
            safe_to_cancel = restore(args, out, runtime_recipe) == 0
        if timer is not None and safe_to_cancel:
            try: cancel_restore_timer(args, out, timer)
            except Exception as cancel_exc:  # noqa: BLE001
                failure = json.loads((out / "failure.json").read_text()); failure["timer_cancellation_error"] = str(cancel_exc); (out / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"WINDOW_C2_CONTINUATION_FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        v2.release_host_lock(args, host_fd)
        for s, h in previous.items(): signal.signal(s, h)
        try: lock_path.unlink()
        except FileNotFoundError: pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--restore-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--c1-root", type=Path, default=Path("/tmp/c1-live-76fff65"))
    p.add_argument("--k1-root", type=Path, default=Path("/tmp/k1-live-aa3c645"))
    p.add_argument("--docker", default="docker")
    p.add_argument("--docker-context", default=None)
    p.add_argument("--bash", default="bash")
    p.add_argument("--python", default="python3")
    p.add_argument("--health", default=None)
    p.add_argument("--api-probe", default=None)
    p.add_argument("--nsys", default="/opt/nvidia/nsight-systems/2025.6.3/bin/nsys")
    p.add_argument("--systemd-run", default="systemd-run")
    p.add_argument("--systemctl", default="systemctl")
    p.add_argument("--timer-python", default="python3")
    p.add_argument("--sudo", action="store_true")
    p.add_argument("--sudo-command", default="sudo")
    p.add_argument("--command-timeout-sec", type=float, default=1800.0)
    p.add_argument("--readiness-timeout-sec", type=float, default=v2.READINESS_TIMEOUT_SEC)
    p.add_argument("--window-deadline-sec", type=float, default=10800.0)
    p.add_argument("--restore-budget-sec", type=float, default=14400.0)
    p.add_argument("--host-operation-lock", default=HOST_OPERATION_LOCK)
    args = p.parse_args(argv)
    if args.run_id != RUN_ID:
        p.error(f"C2 continuation runner requires run id {RUN_ID}")
    if not args.restore_only and args.root is None:
        p.error("--root is required unless --restore-only is set")
    args.recipe = args.recipe.resolve(); args.out = args.out.resolve(); args.c1_root = args.c1_root.resolve(); args.k1_root = args.k1_root.resolve()
    if args.root is not None: args.root = args.root.resolve()
    args._manifest_sha256 = None; args._window_deadline_at = None
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(restore(parsed, parsed.out) if parsed.restore_only else run_c2(parsed))
