#!/usr/bin/env python3
"""Fail-closed offline-checkable A/B runner for K=1 vs K=2 slot-cache candidates.

This runner intentionally blocks the production path unless every runtime source
prerequisite is present and archived. It reuses the v2 safety primitives by
import and executes commands from the archived tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import window_e1_v2 as v2  # noqa: E402

RUN_ID = "k2-v3-20260908"
HOST_OPERATION_LOCK = v2.HOST_OPERATION_LOCK
C1_NAME = "glm53-big-k2v3-c1-mtp1"
C2_NAME = "glm53-big-k2v3-c2-mtp2"
SPEC_K1 = '{"method":"mtp","num_speculative_tokens":1}'
SPEC_K2 = '{"method":"mtp","num_speculative_tokens":2}'
CANDIDATES = (
    {"label": "c1", "name": C1_NAME, "spec": SPEC_K1, "token": "c1-k1-v3"},
    {"label": "c2", "name": C2_NAME, "spec": SPEC_K2, "token": "c2-k2-v3"},
)

SOURCE_FILES = tuple(dict.fromkeys((
    *v2.SOURCE_FILES,
    "scripts/window_e1_v2.py",
    "scripts/window_ab_v3.py",
    "scripts/greedy_equiv.py",
    "scripts/window_ab_verdict.py",
    "research/k2-v3-offline-contract.md",
)))


class PrepBlocked(RuntimeError):
    pass


class GateFailed(RuntimeError):
    code = "WINDOW_AB_V3_FAILED"

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_greedy_interface(script: Path) -> dict[str, object]:
    text = script.read_text()
    if "--compare" not in text or "PROMPTS" not in text:
        raise PrepBlocked("greedy_equiv.py does not expose existing capture/--compare interface")
    prompt_count = len(re.findall(r'^\s*"', text.split("PROMPTS", 1)[1], re.M))
    # The existing script is a direct CLI; inspect the source shape rather than pretending an import API exists.
    return {
        "mode": "cli",
        "capture_argv": ["<python>", "scripts/greedy_equiv.py", "<outfile>"],
        "compare_argv": ["<python>", "scripts/greedy_equiv.py", "--compare", "<a.json>", "<b.json>"],
        "prompt_count_detected": prompt_count,
        "source_sha256": v2.sha256(script),
    }


def archive_source(recipe: Path, out: Path) -> Path:
    missing = [rel for rel in SOURCE_FILES if not (recipe / rel).is_file()]
    if missing:
        raise PrepBlocked("missing required runtime dependencies: " + ", ".join(missing))
    archive = out / "executed-source"
    files: list[dict[str, str | int]] = []
    for rel in SOURCE_FILES:
        src = recipe / rel
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dest, follow_symlinks=False)
        files.append({"path": rel, "sha256": v2.sha256(src), "bytes": src.stat().st_size})
    greedy_info = parse_greedy_interface(archive / "scripts/greedy_equiv.py")
    manifest = {
        "schema": "glm53-window-ab-v3-source-manifest-v1",
        "run_id": RUN_ID,
        "files": files,
        "greedy_equiv_interface": greedy_info,
        "v2_helpers_imported_from": str(Path(inspect.getfile(v2)).resolve()),
        "v2_source_sha256": v2.sha256(Path(inspect.getfile(v2)).resolve()),
        "excluded": ["__pycache__", "*.pyc", "runtime symlinks"],
    }
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def bind_archived_helpers(runtime_recipe: Path) -> None:
    """After source archiving, use archived v2 helpers rather than live checkout code."""
    global v2
    archived = runtime_recipe / "scripts/window_e1_v2.py"
    spec = importlib.util.spec_from_file_location("window_e1_v2_archived", archived)
    if spec is None or spec.loader is None:
        raise PrepBlocked("cannot load archived window_e1_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("INCUMBENT", "IMAGE_DIGEST", "IMAGE", "MODEL", "CONTROLLED_ENV"):
        if getattr(module, name, None) != getattr(v2, name, None):
            raise PrepBlocked(f"archived v2 helper constant mismatch: {name}")
    v2 = module


def attest_archived_helper_binding(runtime_recipe: Path, out: Path) -> None:
    manifest_path = out / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    helper_path = runtime_recipe / "scripts/window_e1_v2.py"
    manifest["v2_helpers_imported_from"] = str(helper_path.resolve())
    manifest["v2_source_sha256"] = v2.sha256(helper_path)
    manifest["helper_binding"] = "archived-executed-source"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


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


def verify_archive_integrity(runtime_recipe: Path, out: Path, expected_manifest_sha256: str | None = None) -> None:
    manifest_path = out / "source-manifest.json"
    if expected_manifest_sha256 is not None and v2.sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("source manifest hash changed after archive")
    manifest = json.loads(manifest_path.read_text())
    observed_paths = [str(item.get("path", "")) for item in manifest.get("files", [])]
    if set(observed_paths) != set(SOURCE_FILES) or len(observed_paths) != len(SOURCE_FILES):
        raise RuntimeError("source manifest file set mismatch")
    for item in manifest.get("files", []):
        rel = str(item["path"])
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise RuntimeError(f"unsafe manifest path: {rel}")
        target = runtime_recipe / rel
        if not target.is_file():
            raise RuntimeError(f"executed source missing: {rel}")
        if target.is_symlink():
            raise RuntimeError(f"executed source is symlink: {rel}")
        if v2.sha256(target) != item["sha256"]:
            raise RuntimeError(f"executed source hash mismatch: {rel}")


def source_manifest_sha(out: Path) -> str:
    return v2.sha256(out / "source-manifest.json")


def docker(args: argparse.Namespace, out: Path, *docker_args: str, allow_fail: bool = False, timeout: float | None = None):
    env = dict(os.environ)
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    actual_timeout = args.command_timeout_sec if timeout is None else timeout
    if args._window_deadline_at is not None:
        actual_timeout = min(float(actual_timeout), max(args._window_deadline_at - time.monotonic(), 0.001))
    redact = len(docker_args) > 0 and docker_args[0] == "inspect" and "-f" not in docker_args
    result = v2.run_logged([args.docker, *docker_args], out / "raw-logs/docker.log", env=env, timeout=actual_timeout, redact_stdout=redact)
    if not allow_fail and result.returncode != 0:
        raise RuntimeError("docker " + " ".join(docker_args) + f" failed with exit {result.returncode}")
    return result


def check_deadline(args: argparse.Namespace, out: Path, what: str, *, restoring: bool = False) -> None:
    if args._window_deadline_at is None or restoring:
        return
    if time.monotonic() >= args._window_deadline_at:
        (out / "raw-logs").mkdir(parents=True, exist_ok=True)
        with (out / "raw-logs" / "docker.log").open("a") as handle:
            handle.write(f"restore_budget_sec={float(args.restore_budget_sec)}\n")
        raise GateFailed("WINDOW_DEADLINE_EXCEEDED", f"deadline before {what}; restore_budget_sec={float(args.restore_budget_sec)}")


def load_archived_verdict(runtime_recipe: Path):
    path = runtime_recipe / "scripts/window_ab_verdict.py"
    spec = importlib.util.spec_from_file_location("window_ab_verdict_archived", path)
    if spec is None or spec.loader is None:
        raise PrepBlocked("cannot load archived window_ab_verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("validate_lane", "quality_match", "c1_gate", "evaluate"):
        if not callable(getattr(module, name, None)):
            raise PrepBlocked(f"archived window_ab_verdict.py missing callable {name}")
    return module


def run_logged_guarded(args: argparse.Namespace, out: Path, what: str, argv: list[str], log: Path, **kwargs):
    check_deadline(args, out, what)
    timeout = kwargs.pop("timeout", args.command_timeout_sec)
    if args._window_deadline_at is not None:
        timeout = min(float(timeout), max(args._window_deadline_at - time.monotonic(), 0.001))
    return v2.run_logged(argv, log, timeout=timeout, **kwargs)


def require_ok(result, what: str) -> None:
    v2.require_ok(result, what)


def health_command(args: argparse.Namespace, runtime_recipe: Path | None = None) -> str:
    if args.health is not None:
        return str(args.health)
    if runtime_recipe is not None:
        return str(runtime_recipe / "scripts/health-check.sh")
    return str(Path(__file__).resolve().parents[1] / "scripts/health-check.sh")


def candidate_env(parent: dict[str, str], out: Path, args: argparse.Namespace, candidate: dict[str, str]) -> dict[str, str]:
    env = v2.controlled_launch_env(parent, out, args.docker, args.docker_context)
    for key in ("FAKE_DOCKER_STATE",):
        if key in parent:
            env[key] = parent[key]
    env.update({
        "CONTAINER_NAME": candidate["name"],
        "STATS_SEC": "0",
        "NSYS": "1",
        "NSYS_OUTPUT": f"/wcap/{candidate['name']}",
        "NSYS_CONTROL": f"/wcap/{candidate['name']}-nsys-control",
    })
    return env


def launch_config(candidate: dict[str, str], env: dict[str, str], argv: list[str], out: Path, manifest_sha: str) -> dict[str, object]:
    env_keys = ("MODEL_DIR", "CACHE_DIR", "API_KEY_FILE", "KV_CACHE_MEMORY", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "SLOT_CACHE_PER_LAYER", "AT_KEY", "STATS_SEC", "COMPILATION_CONFIG", "IMAGE", "ROUTER", "CAPTURE", "UNPACKED", "LOGIT_RING", "BYPASS", "NSYS")
    identity = {
        "candidate": candidate["label"],
        "container_name": candidate["name"],
        "speculative_config": candidate["spec"],
        "argv": argv,
        "env": {k: env.get(k) for k in env_keys},
        "source_manifest_sha256": manifest_sha,
    }
    identity["config_digest"] = digest_json(identity)
    return identity



def launch_config_parity_key(item: dict[str, object]) -> dict[str, object]:
    copy = json.loads(json.dumps(item))
    for key in ("candidate", "container_name", "speculative_config", "config_digest", "observed_container_name", "observed_container_id", "observed_image_digest", "observed_config_image", "observed_model", "observed_max_model_len", "observed_max_num_seqs", "observed_speculative_config", "running", "actual_args"):
        copy.pop(key, None)
    argv = list(copy["argv"])
    if len(argv) > 2:
        argv[2] = "<LANE>"
    if "--speculative-config" in argv:
        argv[argv.index("--speculative-config") + 1] = "<SPEC>"
    copy["argv"] = argv
    return copy


def require_config_parity_before_launch(candidate: dict[str, str], identities: list[dict[str, object]], intended: dict[str, object]) -> None:
    if candidate["label"] != "c2" or not identities:
        return
    if launch_config_parity_key(identities[0]) != launch_config_parity_key(intended):
        raise RuntimeError("candidate c2 launch config parity mismatch before launch")

def write_launch_identity(out: Path, identities: list[dict[str, object]]) -> None:
    def comparable(item: dict[str, object]) -> dict[str, object]:
        copy = json.loads(json.dumps(item))
        for key in ("candidate", "container_name", "speculative_config", "config_digest", "observed_container_name", "observed_container_id", "observed_speculative_config"):
            copy.pop(key, None)
        argv = list(copy["argv"])
        if len(argv) > 2:
            argv[2] = "<LANE>"
        if "--speculative-config" in argv:
            argv[argv.index("--speculative-config") + 1] = "<SPEC>"
        copy["argv"] = argv
        if "actual_args" in copy and "--speculative-config" in copy["actual_args"]:
            actual = list(copy["actual_args"])
            actual[actual.index("--speculative-config") + 1] = "<SPEC>"
            copy["actual_args"] = actual
        return copy
    common = len(identities) == 2 and comparable(identities[0]) == comparable(identities[1])
    payload = {"schema": "glm53-window-ab-v3-launch-identity-v1", "source_manifest_sha256": source_manifest_sha(out), "common_except_speculative_config": common, "candidates": identities}
    (out / "launch-identity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _option_value(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None


def inspect_candidate_identity(args: argparse.Namespace, out: Path, candidate: dict[str, str], intended: dict[str, object]) -> dict[str, object]:
    result = docker(args, out, "inspect", candidate["name"])
    data = json.loads(result.stdout)[0]
    cmd = [str(part) for part in (data.get("Args") or data.get("Config", {}).get("Cmd") or [])]
    speculative_raw = _option_value(cmd, "--speculative-config")
    try:
        speculative = json.loads(speculative_raw) if speculative_raw is not None else None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"candidate {candidate['label']} speculative config is not JSON") from exc
    expected_k = json.loads(candidate["spec"])["num_speculative_tokens"]
    proof = dict(intended)
    proof.update({
        "observed_container_name": str(data.get("Name", "")).lstrip("/"),
        "observed_container_id": data.get("Id") or data.get("ID"),
        "observed_image_digest": data.get("Image"),
        "observed_config_image": data.get("Config", {}).get("Image"),
        "observed_model": _option_value(cmd, "--model") or _option_value(cmd, "--served-model-name") or v2.MODEL,
        "observed_max_model_len": _option_value(cmd, "--max-model-len"),
        "observed_max_num_seqs": _option_value(cmd, "--max-num-seqs"),
        "observed_speculative_config": speculative,
        "running": bool(data.get("State", {}).get("Running")),
        "actual_args": cmd,
    })
    if proof["observed_container_name"] != candidate["name"]:
        raise RuntimeError(f"candidate {candidate['label']} container name mismatch")
    if proof["observed_image_digest"] != v2.IMAGE_DIGEST:
        raise RuntimeError(f"candidate {candidate['label']} image digest mismatch")
    if proof["observed_config_image"] != v2.IMAGE:
        raise RuntimeError(f"candidate {candidate['label']} image tag mismatch")
    if proof["observed_model"] != v2.MODEL:
        raise RuntimeError(f"candidate {candidate['label']} model mismatch")
    if proof["observed_max_model_len"] != "524288":
        raise RuntimeError(f"candidate {candidate['label']} ctx512k flag mismatch")
    if proof["observed_max_num_seqs"] != "1":
        raise RuntimeError(f"candidate {candidate['label']} max-num-seqs mismatch")
    if not isinstance(speculative, dict) or speculative.get("num_speculative_tokens") != expected_k:
        raise RuntimeError(f"candidate {candidate['label']} speculative K mismatch")
    if not proof["running"]:
        raise RuntimeError(f"candidate {candidate['label']} is not running after launch")
    proof["config_digest"] = digest_json(proof)
    return proof


def ensure_named_candidates_stopped(args: argparse.Namespace, out: Path) -> None:
    for candidate in CANDIDATES:
        docker(args, out, "stop", "-t", "120", candidate["name"], allow_fail=True)
    for candidate in CANDIDATES:
        running = docker(args, out, "inspect", "-f", "{{.State.Running}}", candidate["name"], allow_fail=True).stdout.strip()
        if running == "true":
            raise RuntimeError(f"candidate did not stop: {candidate['name']}")
        if running != "false":
            raise RuntimeError(f"candidate stop state ambiguous: {candidate['name']}")


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
            ensure_named_candidates_stopped(args, out)
            docker_env = dict(os.environ)
            if args.docker_context:
                docker_env["DOCKER_CONTEXT"] = args.docker_context
            result = v2.run_logged([args.docker, "start", v2.INCUMBENT], restore_log, env=docker_env, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
            if result.returncode != 0:
                status = 1
            health = health_command(args, runtime_recipe)
            health_result = v2.run_logged([health], restore_dir / "health.txt", env=v2.health_env(os.environ), timeout=min(args.readiness_timeout_sec, args.restore_budget_sec))
            if health_result.returncode != 0:
                status = 1
            ps = v2.run_logged([args.docker, "ps", "--filter", f"name=^/{v2.INCUMBENT}$"], restore_dir / "docker-ps.txt", env=docker_env, timeout=min(args.command_timeout_sec, args.restore_budget_sec))
            if ps.returncode != 0:
                status = 1
            try:
                v2.verify_incumbent_identity(args, out)
                v2.restore_api_proof(args, out)
            except Exception as exc:  # noqa: BLE001
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


def receipt_validate_c1(out: Path, runtime_recipe: Path) -> None:
    probe = out / "c1" / "acceptance-512.json"
    reference = out / "quality" / "incumbent-greedy.json"
    greedy = out / "quality" / "c1-greedy.json"
    if not probe.is_file() or not reference.is_file() or not greedy.is_file():
        raise GateFailed("G1_FAILED", "C1 receipt validation failed: missing probe or greedy output")
    payload = json.loads(probe.read_text())
    module = load_archived_verdict(runtime_recipe)
    try:
        receipt = module.c1_gate(payload, json.loads(reference.read_text()), json.loads(greedy.read_text()), baseline_speed=45.65)
    except Exception as exc:  # noqa: BLE001
        raise GateFailed("G1_FAILED", f"C1 gate validation failed: {exc}") from exc
    (out / "c1-gate.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt.get("pass"):
        raise GateFailed("G1_FAILED", "C1 gate failed")


def run_greedy_capture(args: argparse.Namespace, out: Path, runtime_recipe: Path, dest: Path) -> None:
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = run_logged_guarded(args, out, "greedy capture", [args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), str(dest)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    require_ok(result, "greedy capture")


def run_greedy_compare(args: argparse.Namespace, out: Path, runtime_recipe: Path, a: Path, b: Path, label: str) -> str:
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = run_logged_guarded(args, out, "greedy compare", [args.python, str(runtime_recipe / "scripts/greedy_equiv.py"), "--compare", str(a), str(b)], out / "raw-logs/greedy.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec)
    (out / "quality" / f"{label}-compare.txt").write_text(result.stdout)
    require_ok(result, f"greedy compare {label}")
    return result.stdout


def run_candidate(args: argparse.Namespace, out: Path, runtime_recipe: Path, candidate: dict[str, str], identities: list[dict[str, object]]) -> None:
    label = candidate["label"]
    cdir = out / label
    cdir.mkdir(parents=True, exist_ok=True)
    gate(args, out, runtime_recipe)
    launch_cmd = [args.bash, str(runtime_recipe / "scripts/launch-slotcache-portable.sh"), f"{label}-nsys", "112", "--speculative-config", candidate["spec"]]
    env = candidate_env(os.environ, out, args, candidate)
    intended_identity = launch_config(candidate, env, launch_cmd, out, source_manifest_sha(out))
    require_config_parity_before_launch(candidate, identities, intended_identity)
    launch = run_logged_guarded(args, out, f"launch {label}", launch_cmd, out / "raw-logs" / f"launch-{label}.log", env=env, timeout=args.command_timeout_sec)
    (cdir / "launch.txt").write_text(v2.redact_sensitive(launch.stdout))
    require_ok(launch, f"{label} launch")
    gate(args, out, runtime_recipe)
    identities.append(inspect_candidate_identity(args, out, candidate, intended_identity))
    require_ok(run_logged_guarded(args, out, f"readiness {label}", [health_command(args, runtime_recipe)], cdir / "health.txt", env=v2.health_env(os.environ), timeout=args.readiness_timeout_sec), f"{label} readiness")
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"acceptance {label}", [args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "512", "--out", str(cdir / "acceptance-512.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), f"{label} acceptance")
    for rep in range(1, 4):
        gate(args, out, runtime_recipe)
        require_ok(run_logged_guarded(args, out, f"bench prose {label}", [args.python, str(runtime_recipe / "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py")], out / "raw-logs/bench.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec), f"{label} prose bench {rep}")
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"bench code {label}", [args.python, str(runtime_recipe / "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py")], out / "raw-logs/bench.log", env=v2.benchmark_env(os.environ), timeout=args.command_timeout_sec), f"{label} code bench")
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"unprofiled probe {label}", [args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "64", "--out", str(cdir / "unprofiled-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), f"{label} unprofiled probe")
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"nsys start {label}", [args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / f"{candidate['name']}-nsys-control"), "START", candidate["token"]], out / "nsys-control.log", timeout=args.command_timeout_sec), f"{label} nsys start")
    v2.check_control_ack(out / "nsys-control.log", "START", candidate["token"])
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"profiled probe {label}", [args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", v2.MODEL, "--api-key-file", v2.CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "64", "--out", str(cdir / "profiled-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), f"{label} profiled probe")
    gate(args, out, runtime_recipe)
    require_ok(run_logged_guarded(args, out, f"nsys stop {label}", [args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / f"{candidate['name']}-nsys-control"), "STOP", candidate["token"]], out / "nsys-control.log", timeout=args.command_timeout_sec), f"{label} nsys stop")
    v2.check_control_ack(out / "nsys-control.log", "STOP", candidate["token"])
    gate(args, out, runtime_recipe)
    run_greedy_capture(args, out, runtime_recipe, out / "quality" / f"{label}-greedy.json")
    gate(args, out, runtime_recipe)
    docker(args, out, "stop", "-t", "120", candidate["name"])
    stopped = docker(args, out, "inspect", "-f", "{{.State.Running}}", candidate["name"], allow_fail=True).stdout.strip()
    if stopped == "true":
        raise RuntimeError(f"{candidate['name']} still running after stop")
    if stopped != "false":
        raise RuntimeError(f"candidate stop state ambiguous: {candidate['name']}")


def gate(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> None:
    verify_archive_integrity(runtime_recipe, out, args._manifest_sha256)
    result = run_logged_guarded(args, out, "release gate", [args.python, str(runtime_recipe / "scripts/window_gate.py"), str(args.root), args.run_id], out / "raw-logs/gate.log", timeout=args.command_timeout_sec)
    require_ok(result, "release gate")


def export_nsys_stats(args: argparse.Namespace, out: Path) -> None:
    reports = []
    for candidate in CANDIDATES:
        label = candidate["label"]
        rep = out / f"{candidate['name']}.nsys-rep"
        if not rep.is_file() or rep.stat().st_size == 0:
            raise RuntimeError(f"missing real nsys report: {rep.name}")
        before = v2.sha256(rep)
        require_ok(run_logged_guarded(args, out, f"nsys stats {label}", [args.nsys, "stats", "--force-export=true", "--force-overwrite=true", "--report", "cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_gpu_trace,cuda_api_trace", "--format", "csv", "--output", str(out / label), str(rep)], out / "raw-logs/nsys-stats.log", timeout=args.command_timeout_sec), f"{label} nsys stats")
        after = v2.sha256(rep)
        if after != before:
            raise RuntimeError("nsys report changed during stats export")
        csv_hashes = {}
        for suffix in ("cuda_gpu_kern_sum", "cuda_kern_exec_sum", "cuda_gpu_trace", "cuda_api_trace"):
            csv = out / f"{label}_{suffix}.csv"
            if not csv.is_file() or csv.stat().st_size == 0:
                raise RuntimeError(f"missing nsys export: {csv.name}")
            csv_hashes[csv.name] = v2.sha256(csv)
        reports.append({"candidate": label, "report": rep.name, "report_bytes": rep.stat().st_size, "pre_stats_sha256": before, "post_stats_sha256": after, "exported_csv_sha256": csv_hashes})
    (out / "nsys-export-attestation.json").write_text(json.dumps({"schema": "glm53-window-ab-v3-nsys-export-attestation-v1", "reports": reports}, indent=2, sort_keys=True) + "\n")


def run_ab(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "failure.json").write_text(json.dumps({"schema": "glm53-window-ab-v3-failure-v1", "code": "PREP_BLOCKED", "error": "public run_ab production lifecycle entrypoint is blocked"}, indent=2) + "\n")
    print("WINDOW_AB_V3_PREP_BLOCKED: public run_ab production lifecycle entrypoint is blocked", file=sys.stderr)
    return 2


def _run_ab(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"WINDOW_AB_V3_FAILED: output directory is not empty: {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    host_lock_fd: int | None = None
    runtime_recipe: Path | None = None
    restore_required = False
    identities: list[dict[str, object]] = []
    previous_handlers = {}
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"WINDOW_AB_V3_FAILED: output lock already exists: {lock_path}", file=sys.stderr)
        return 1

    def _signal_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"received signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)

    try:
        args._window_deadline_at = time.monotonic() + args.window_deadline_sec
        runtime_recipe = archive_source(args.recipe, out)
        bind_archived_helpers(runtime_recipe)
        attest_archived_helper_binding(runtime_recipe, out)
        args._manifest_sha256 = source_manifest_sha(out)
        host_lock_fd = v2.acquire_host_lock(args, out)
        v2.verify_incumbent_identity(args, out, "preflight/incumbent-proof.json")
        v2.verify_candidate_image(args, out)
        parse_greedy_interface(runtime_recipe / "scripts/greedy_equiv.py")
        run_greedy_capture(args, out, runtime_recipe, out / "quality" / "incumbent-greedy.json")
        gate(args, out, runtime_recipe)
        # A command may stop the service and then fail or be interrupted.
        restore_required = True
        docker(args, out, "stop", v2.INCUMBENT)
        gate(args, out, runtime_recipe)
        run_candidate(args, out, runtime_recipe, CANDIDATES[0], identities)
        run_greedy_compare(args, out, runtime_recipe, out / "quality" / "incumbent-greedy.json", out / "quality" / "c1-greedy.json", "c1")
        receipt_validate_c1(out, runtime_recipe)
        gate(args, out, runtime_recipe)
        run_candidate(args, out, runtime_recipe, CANDIDATES[1], identities)
        run_greedy_compare(args, out, runtime_recipe, out / "quality" / "incumbent-greedy.json", out / "quality" / "c2-greedy.json", "c2")
        write_launch_identity(out, identities)
        restore_status = restore(args, out, runtime_recipe)
        restore_required = False
        if restore_status != 0:
            return restore_status
        # Expensive profiler stats are intentionally after restore proof only.
        export_nsys_stats(args, out)
        verdict = load_archived_verdict(runtime_recipe).evaluate(
            json.loads((out / "c1" / "acceptance-512.json").read_text()),
            json.loads((out / "c2" / "acceptance-512.json").read_text()),
            json.loads((out / "quality" / "incumbent-greedy.json").read_text()),
            json.loads((out / "quality" / "c1-greedy.json").read_text()),
            json.loads((out / "quality" / "c2-greedy.json").read_text()),
        )
        verdict["production_live_path"] = "BLOCKED_UNTIL_REAL_ENGINE_PHASE_SNAPSHOT_INSTRUMENTATION_PROVEN"
        (out / "ab-verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print("WINDOW_AB_V3_COLLECTION_OK")
        return 0
    except PrepBlocked as exc:
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-window-ab-v3-failure-v1", "code": "PREP_BLOCKED", "error": str(exc)}, indent=2) + "\n")
        print(f"WINDOW_AB_V3_PREP_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except GateFailed as exc:
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-window-ab-v3-failure-v1", "code": exc.code, "error": str(exc)}, indent=2) + "\n")
        if restore_required:
            restore(args, out, runtime_recipe)
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:  # noqa: BLE001
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-window-ab-v3-failure-v1", "code": "WINDOW_AB_V3_FAILED", "error": str(exc)}, indent=2) + "\n")
        if restore_required:
            restore(args, out, runtime_recipe)
        print(f"WINDOW_AB_V3_FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        v2.release_host_lock(args, host_lock_fd)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            lock_path.unlink()
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
    parser.add_argument("--nsys", default="/opt/nvidia/nsight-systems/2025.6.3/bin/nsys")
    parser.add_argument("--api-probe", default=None)
    parser.add_argument("--command-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--readiness-timeout-sec", type=float, default=v2.READINESS_TIMEOUT_SEC)
    parser.add_argument("--window-deadline-sec", type=float, default=10800.0)
    parser.add_argument("--restore-budget-sec", type=float, default=14400.0)
    parser.add_argument("--host-operation-lock", default=HOST_OPERATION_LOCK)
    args = parser.parse_args(argv)
    if args.run_id != RUN_ID:
        parser.error(f"v3 runner requires run id {RUN_ID}")
    if not args.restore_only and args.root is None:
        parser.error("--root is required unless --restore-only is set")
    args.recipe = args.recipe.resolve()
    if args.root is not None:
        args.root = args.root.resolve()
    args.out = args.out.resolve()
    args._window_deadline_at = None
    args._manifest_sha256 = None
    return args


if __name__ == "__main__":
    _args = parse_args()
    if not _args.restore_only:
        print('WINDOW_AB_V3_PREP_BLOCKED: live collection is unreleased; engine phase/snapshot instrumentation and lifecycle review are incomplete', file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(restore(_args, _args.out))
