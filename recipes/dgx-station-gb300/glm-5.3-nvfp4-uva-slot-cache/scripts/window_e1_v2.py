#!/usr/bin/env python3
"""Offline-checkable E1-only window runner, version 2.

Runs only the patched K=1 baseline plus one Nsight capture, then restores the
pinned incumbent before offline analysis. It never launches K=2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

RUN_ID = "e0-e1-20260908-v2"
INCUMBENT = "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907"
INCUMBENT_ID = "c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea"
IMAGE = "vllm-glm53-uva:v0.28.0-2cf0a691"
IMAGE_DIGEST = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
CANDIDATE_IMAGE = f"{IMAGE}@{IMAGE_DIGEST}"
HOST_OPERATION_LOCK = "/tmp/glm53-e1-v2-host-operation.lock"
MODEL = "glm-5.3-big"
E1_CANDIDATE = "glm53-big-e1-nsys"
READINESS_TIMEOUT_SEC = 2400.0
SPEC_K1 = '{"method":"mtp","num_speculative_tokens":1}'
CONTROLLED_ENV = {
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
    "IMAGE": CANDIDATE_IMAGE,
    "ROUTER": "ffi",
    "CAPTURE": "0",
    "UNPACKED": "0",
    "LOGIT_RING": "0",
    "BYPASS": "16",
}

SOURCE_FILES = (
    "scripts/window_e1_v2.py",
    "scripts/launch-slotcache-portable.sh",
    "scripts/window_gate.py",
    "scripts/window_verdict.py",
    "scripts/nsys_capture_control.py",
    "scripts/nsys_bucket.py",
    "scripts/dflash2_acceptance_probe.py",
    "scripts/health-check.sh",
    "configs/slots-5792-ctx512k.json",
    "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py",
    "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py",
    "patches/slot_cache_hook.py",
    "patches/slot_cache_stats.py",
    "patches/slot_cache_profile_control.py",
    "patches/sitecustomize.py",
    "patches/exact_pin.py",
    "patches/ffi_route.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SENSITIVE_PATTERNS = (
    re.compile(r"(VLLM_API_KEY=)[^\s,'\"\]]+"),
    re.compile(r"(API_KEY=)[^\s,'\"\]]+"),
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s,'\"\]]+", re.IGNORECASE),
)


def redact_sensitive(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(r"\1<redacted>", redacted)
    return redacted


def terminate_process_group(process: subprocess.Popen[str], *, grace_sec: float = 10.0) -> tuple[str, bool]:
    terminated = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        terminated = True
    except ProcessLookupError:
        pass
    try:
        stdout, _ = process.communicate(timeout=grace_sec)
        return stdout or "", terminated
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
        stdout, _ = process.communicate()
        return stdout or "", terminated


def run_logged(argv: list[str], log: Path, *, env: dict[str, str] | None = None, cwd: Path | None = None, timeout: float | None = None, redact_stdout: bool = False) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log.open("a") as handle:
        handle.write(f"$ {' '.join(argv)}\nstarted={started}\n")
        process = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, terminated = terminate_process_group(process)
            handle.write("[stdout redacted]\n" if redact_stdout else redact_sensitive(stdout or ""))
            handle.write(f"\nexit=timeout after {timeout}s; process_group_terminated={str(terminated).lower()}\n")
            return subprocess.CompletedProcess(argv, 124, stdout or "", "")
        except BaseException:
            stdout, terminated = terminate_process_group(process)
            handle.write("[stdout redacted]\n" if redact_stdout else redact_sensitive(stdout or ""))
            handle.write(f"\nexit=interrupted; process_group_terminated={str(terminated).lower()}\n")
            raise
        stdout = stdout or ""
        handle.write("[stdout redacted]\n" if redact_stdout else redact_sensitive(stdout))
        handle.write(f"\nexit={process.returncode}\n")
    return subprocess.CompletedProcess(argv, int(process.returncode or 0), stdout, "")


def require_ok(result: subprocess.CompletedProcess[str], what: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed with exit {result.returncode}")


def verify_archive_integrity(runtime_recipe: Path, out: Path) -> None:
    manifest_path = out / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest.get("files", []):
        rel = str(item["path"])
        target = runtime_recipe / rel
        if not target.is_file():
            raise RuntimeError(f"executed source missing: {rel}")
        if sha256(target) != item["sha256"]:
            raise RuntimeError(f"executed source hash mismatch: {rel}")


def gate(args: argparse.Namespace, out: Path, runtime_recipe: Path) -> None:
    verify_archive_integrity(runtime_recipe, out)
    result = run_logged([args.python, str(runtime_recipe / "scripts/window_gate.py"), str(args.root), args.run_id], out / "raw-logs/gate.log", timeout=args.command_timeout_sec)
    require_ok(result, "release gate")


def docker(args: argparse.Namespace, out: Path, *docker_args: str, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    redact = len(docker_args) > 0 and docker_args[0] == "inspect" and "-f" not in docker_args
    env = dict(os.environ)
    if args.docker_context:
        env["DOCKER_CONTEXT"] = args.docker_context
    result = run_logged([args.docker, *docker_args], out / "raw-logs/docker.log", env=env, timeout=args.command_timeout_sec, redact_stdout=redact)
    if not allow_fail:
        require_ok(result, "docker " + " ".join(docker_args))
    return result


def verify_incumbent_identity(args: argparse.Namespace, out: Path, proof_rel: str = "restore/service-proof.json") -> None:
    result = docker(args, out, "inspect", INCUMBENT)
    data = json.loads(result.stdout)[0]
    proof = {
        "expected_container_name": INCUMBENT,
        "expected_container_id": INCUMBENT_ID,
        "expected_image_digest": IMAGE_DIGEST,
        "observed_container_name": data.get("Name", "").lstrip("/"),
        "observed_container_id": data.get("Id") or data.get("ID"),
        "observed_image": data.get("Image"),
        "observed_config_image": data.get("Config", {}).get("Image"),
        "running": bool(data.get("State", {}).get("Running")),
        "args": data.get("Args") or data.get("Config", {}).get("Cmd") or [],
    }
    proof_path = out / proof_rel
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if proof["observed_container_name"] != INCUMBENT:
        raise RuntimeError("restore container name mismatch")
    observed_id = str(proof["observed_container_id"] or "")
    if not observed_id:
        raise RuntimeError("restore container id missing")
    if observed_id != INCUMBENT_ID:
        raise RuntimeError("restore container id mismatch")
    observed_image = str(proof["observed_image"] or "")
    if not observed_image:
        raise RuntimeError("restore image digest missing")
    if observed_image != IMAGE_DIGEST:
        raise RuntimeError("restore image digest mismatch")
    if proof["observed_config_image"] != IMAGE:
        raise RuntimeError("restore config image tag mismatch")
    cmd = [str(part) for part in proof["args"]]
    def option_value(name: str) -> str | None:
        return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None
    if option_value("--max-model-len") != "524288":
        raise RuntimeError("restore ctx512k flag mismatch")
    if option_value("--max-num-seqs") != "1":
        raise RuntimeError("restore max-num-seqs flag mismatch")
    speculative = option_value("--speculative-config")
    try:
        speculative_config = json.loads(speculative) if speculative is not None else None
    except json.JSONDecodeError as exc:
        raise RuntimeError("restore speculative config is not JSON") from exc
    proof["observed_speculative_config"] = speculative_config
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if not isinstance(speculative_config, dict) or speculative_config.get("method") != "mtp" or speculative_config.get("num_speculative_tokens") != 1:
        raise RuntimeError("restore MTP K=1 flag missing")
    if not proof["running"]:
        raise RuntimeError("restore incumbent is not running")



def verify_candidate_image(args: argparse.Namespace, out: Path) -> None:
    result = docker(args, out, "image", "inspect", "--format", "{{.Id}}", CANDIDATE_IMAGE)
    observed = result.stdout.strip()
    proof = {"candidate_image": CANDIDATE_IMAGE, "expected_image_id": IMAGE_DIGEST, "observed_image_id": observed}
    proof_path = out / "preflight/candidate-image-proof.json"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if observed != IMAGE_DIGEST:
        raise RuntimeError("candidate image digest mismatch")


def acquire_host_lock(args: argparse.Namespace, out: Path) -> int:
    lock_path = Path(args.host_operation_lock)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"host operation lock already exists: {lock_path}") from exc
    os.write(fd, f"pid={os.getpid()} out={out} run_id={args.run_id}\n".encode())
    return fd


def release_host_lock(args: argparse.Namespace, fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        Path(args.host_operation_lock).unlink()
    except FileNotFoundError:
        pass


def write_current_run_evidence(out: Path, **values: str) -> None:
    evidence = dict(values)
    for name in ("source-manifest.json", "profiled-probe.json", "unprofiled-probe.json", "nsys-buckets.json", "e1-profile.nsys-rep", "e1_cuda_gpu_kern_sum.csv", "e1_cuda_kern_exec_sum.csv", "e1_cuda_gpu_trace.csv", "e1_cuda_api_trace.csv"):
        path = out / name
        if path.is_file():
            evidence[name.replace("-", "_").replace(".", "_") + "_sha256"] = sha256(path)
    (out / "current-run-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

def archive_source(recipe: Path, out: Path) -> Path:
    archive = out / "executed-source"
    files: list[dict[str, str | int]] = []
    for rel in SOURCE_FILES:
        src = recipe / rel
        if not src.is_file():
            raise RuntimeError(f"missing source for archive: {rel}")
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # copy2 does not dereference unrelated runtime symlinks because only exact files are listed.
        shutil.copy2(src, dest, follow_symlinks=False)
        files.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    manifest = {
        "schema": "glm53-e1-v2-source-manifest-v1",
        "run_id": RUN_ID,
        "files": files,
        "excluded": ["__pycache__", "*.pyc", "runtime nsys symlink"],
    }
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def controlled_launch_env(parent: dict[str, str], out: Path, docker_bin: str, docker_context: str | None = None) -> dict[str, str]:
    env = {"PATH": parent.get("PATH", "/usr/bin:/bin"), "HOME": parent.get("HOME", "/home/milo")}
    for key in ("FAKE_LOG",):
        if key in parent:
            env[key] = parent[key]
    env.update(CONTROLLED_ENV)
    env.update({
        "CAPTURE_DIR": str(out),
        "CONTAINER_NAME": E1_CANDIDATE,
        "NSYS": "1",
        "NSYS_OUTPUT": "/wcap/e1-profile",
        "NSYS_CONTROL": "/wcap/nsys-control",
        "DOCKER": docker_bin,
    })
    if docker_context:
        env["DOCKER_CONTEXT"] = docker_context
    env.pop("DRAFT_MODEL_DIR", None)
    return env


def health_env(parent: dict[str, str]) -> dict[str, str]:
    env = dict(parent)
    env.update({
        "HEALTH_RETRIES": "360",
        "RETRY_SLEEP": "5",
        "BASE_URL": "http://127.0.0.1:30001",
        "MODEL_NAME": MODEL,
        "API_KEY_FILE": CONTROLLED_ENV["API_KEY_FILE"],
    })
    return env


def health_command(args: argparse.Namespace, runtime_recipe: Path | None = None) -> str:
    if args.health is not None:
        return str(args.health)
    recipe = runtime_recipe if runtime_recipe is not None else args.recipe
    return str(recipe / "scripts/health-check.sh")


def read_api_key(path: str) -> str:
    key = Path(path).read_text().strip()
    if not key:
        raise RuntimeError(f"API key file is empty: {path}")
    return key


def benchmark_env(parent: dict[str, str]) -> dict[str, str]:
    env = dict(parent)
    env.update({
        "BASE_URL": "http://127.0.0.1:30001/v1",
        "MODEL": MODEL,
        "MODEL_NAME": MODEL,
        "API_KEY_FILE": CONTROLLED_ENV["API_KEY_FILE"],
    })
    env["API_KEY"] = "***test-only-key***" if "FAKE_LOG" in parent else read_api_key(CONTROLLED_ENV["API_KEY_FILE"])
    return env


def restore_api_proof(args: argparse.Namespace, out: Path) -> None:
    restore_dir = out / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    if args.api_probe:
        result = run_logged([args.api_probe, str(restore_dir)], out / "raw-logs/api-probe.log", env=health_env(os.environ), timeout=args.command_timeout_sec)
        require_ok(result, "restore API proof")
        return
    key = Path(CONTROLLED_ENV["API_KEY_FILE"]).read_text().strip()
    headers = {"Authorization": "Bearer " + key}
    request = urllib.request.Request("http://127.0.0.1:30001/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    (restore_dir / "models.json").write_bytes(raw)
    models = json.loads(raw)
    ids = [item.get("id") for item in models.get("data", [])]
    if MODEL not in ids:
        raise RuntimeError("restore model id missing")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly WINDOW_RESTORE_OK"}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:30001/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
    (restore_dir / "completion.json").write_bytes(raw)
    content = json.loads(raw)["choices"][0]["message"]["content"]
    if content != "WINDOW_RESTORE_OK":
        raise RuntimeError("restore completion mismatch")


def restore(args: argparse.Namespace, out: Path, runtime_recipe: Path | None = None) -> int:
    status = 0
    restore_dir = out / "restore"
    stamp = time.strftime("%Y%m%dT%H%M%S%z") + f"-{time.monotonic_ns()}"
    restore_log = out / "raw-logs" / f"restore-{stamp}.log"
    try:
        docker(args, out, "inspect", "-f", "{{.State.Running}}", E1_CANDIDATE, allow_fail=True)
        docker(args, out, "stop", "-t", "120", E1_CANDIDATE, allow_fail=True)
        docker_env = dict(os.environ)
        if args.docker_context:
            docker_env["DOCKER_CONTEXT"] = args.docker_context
        result = run_logged([args.docker, "start", INCUMBENT], restore_log, env=docker_env, timeout=args.command_timeout_sec)
        if result.returncode != 0:
            status = 1
        run_logged([health_command(args, runtime_recipe)], restore_dir / "health.txt", env=health_env(os.environ), timeout=args.readiness_timeout_sec)
        ps = run_logged([args.docker, "ps", "--filter", f"name=^/{INCUMBENT}$"], restore_dir / "docker-ps.txt", env=docker_env, timeout=args.command_timeout_sec)
        if ps.returncode != 0:
            status = 1
        try:
            verify_incumbent_identity(args, out)
            restore_api_proof(args, out)
        except Exception as exc:  # noqa: BLE001
            with restore_log.open("a") as handle:
                handle.write(f"restore proof failed: {exc}\n")
            status = 1
    except Exception as exc:  # noqa: BLE001
        restore_log.parent.mkdir(parents=True, exist_ok=True)
        with restore_log.open("a") as handle:
            handle.write(f"restore exception: {exc}\n")
        status = 1
    (restore_dir / "restore-status.txt").write_text(f"WINDOW_RESTORE_STATUS={status}\n")
    return status


def profiled_steps_and_wall(out: Path) -> tuple[str, str]:
    probe = json.loads((out / "profiled-probe.json").read_text())
    steps = int(probe["summary"]["verification_steps"])
    wall_ms = float((out / "profile-wall-seconds.txt").read_text().strip()) * 1000.0
    if steps < 50:
        raise RuntimeError("profiled probe has fewer than 50 verification steps")
    return str(steps), str(wall_ms)


def check_control_ack(log: Path, action: str, token: str) -> None:
    text = log.read_text(errors="replace") if log.is_file() else ""
    if f"ACK {action} {token}" not in text:
        raise RuntimeError(f"missing nsys control ACK {action} {token}")


def run_e1(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"WINDOW_E1_V2_FAILED: output directory is not empty: {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    host_lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"WINDOW_E1_V2_FAILED: output lock already exists: {lock_path}", file=sys.stderr)
        return 1
    os.close(lock_fd)
    runtime_recipe = archive_source(args.recipe, out)
    restore_required = False
    previous_handlers = {}
    def _signal_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"received signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)
    try:
        host_lock_fd = acquire_host_lock(args, out)
        verify_incumbent_identity(args, out, "preflight/incumbent-proof.json")
        verify_candidate_image(args, out)
        for candidate in (E1_CANDIDATE, "glm53-big-e5-mtp2"):
            running = docker(args, out, "inspect", "-f", "{{.State.Running}}", candidate, allow_fail=True).stdout.strip()
            if running == "true":
                raise RuntimeError(f"candidate already running: {candidate}")
        gate(args, out, runtime_recipe)
        docker(args, out, "stop", INCUMBENT)
        restore_required = True
        gate(args, out, runtime_recipe)

        verify_archive_integrity(runtime_recipe, out)
        launch_cmd = [args.bash, str(runtime_recipe / "scripts/launch-slotcache-portable.sh"), "e1-nsys", "112", "--speculative-config", SPEC_K1]
        launch = run_logged(launch_cmd, out / "raw-logs/launch.log", env=controlled_launch_env(os.environ, out, args.docker, args.docker_context), timeout=args.command_timeout_sec)
        (out / "launch.txt").write_text(redact_sensitive(launch.stdout))
        require_ok(launch, "E1 K=1 launch")

        gate(args, out, runtime_recipe)
        require_ok(run_logged([health_command(args, runtime_recipe)], out / "health.txt", env=health_env(os.environ), timeout=args.readiness_timeout_sec), "readiness")
        gate(args, out, runtime_recipe)
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", MODEL, "--api-key-file", CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "512", "--out", str(out / "k1-acceptance.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), "k1-acceptance.json")
        for rep in range(1, 4):
            gate(args, out, runtime_recipe)
            result = run_logged([args.python, str(runtime_recipe / "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py")], out / "raw-logs/bench.log", env=benchmark_env(os.environ), timeout=args.command_timeout_sec)
            (out / f"bench-prose-rep{rep}.txt").write_text(result.stdout)
            require_ok(result, f"prose bench rep {rep}")
        gate(args, out, runtime_recipe)
        result = run_logged([args.python, str(runtime_recipe / "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py")], out / "raw-logs/bench.log", env=benchmark_env(os.environ), timeout=args.command_timeout_sec)
        (out / "bench-code-rep1.txt").write_text(result.stdout)
        require_ok(result, "code bench")

        gate(args, out, runtime_recipe)
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", MODEL, "--api-key-file", CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "64", "--out", str(out / "unprofiled-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec), "unprofiled-probe.json")
        gate(args, out, runtime_recipe)
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / "nsys-control"), "START", "e1-profile-v2"], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys start")
        check_control_ack(out / "nsys-control.log", "START", "e1-profile-v2")
        start = time.monotonic()
        result = run_logged([args.python, str(runtime_recipe / "scripts/dflash2_acceptance_probe.py"), "--base-url", "http://127.0.0.1:30001", "--model", MODEL, "--api-key-file", CONTROLLED_ENV["API_KEY_FILE"], "--max-tokens", "64", "--out", str(out / "profiled-probe.json")], out / "raw-logs/probes.log", timeout=args.command_timeout_sec)
        (out / "profile-wall-seconds.txt").write_text(f"{max(time.monotonic() - start, 0.001):.6f}\n")
        require_ok(result, "profiled probe")
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/nsys_capture_control.py"), str(out / "nsys-control"), "STOP", "e1-profile-v2"], out / "nsys-control.log", timeout=args.command_timeout_sec), "nsys stop")
        check_control_ack(out / "nsys-control.log", "STOP", "e1-profile-v2")
        gate(args, out, runtime_recipe)
        docker(args, out, "stop", "-t", "120", E1_CANDIDATE)
        result = docker(args, out, "logs", E1_CANDIDATE)
        (out / "container.log").write_text(redact_sensitive(result.stdout))
        nsys_report = out / "e1-profile.nsys-rep"
        if not nsys_report.is_file() or nsys_report.stat().st_size == 0:
            raise RuntimeError("missing real nsys report: e1-profile.nsys-rep")
        before_nsys_report_sha = sha256(nsys_report)
        restore_status = restore(args, out, runtime_recipe)
        restore_required = False
        if restore_status != 0:
            return restore_status
        require_ok(run_logged([args.nsys, "stats", "--force-export=true", "--force-overwrite=true", "--report", "cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_gpu_trace,cuda_api_trace", "--format", "csv", "--output", str(out / "e1"), str(nsys_report)], out / "raw-logs/nsys-stats.log", timeout=args.command_timeout_sec), "nsys stats")
        if sha256(nsys_report) != before_nsys_report_sha:
            raise RuntimeError("nsys report changed during stats export")
        steps, wall_ms = profiled_steps_and_wall(out)
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/nsys_bucket.py"), str(out / "e1_cuda_gpu_kern_sum.csv"), "--steps", steps, "--wall-ms", wall_ms, "--output", str(out / "nsys-buckets.json")], out / "raw-logs/nsys-bucket.log", timeout=args.command_timeout_sec), "nsys bucket")
        write_current_run_evidence(out, run_id=args.run_id, candidate_image=CANDIDATE_IMAGE, source_manifest_sha256=sha256(out / "source-manifest.json"))
        require_ok(run_logged([args.python, str(runtime_recipe / "scripts/window_verdict.py"), "e1", str(out), str(out / "e1-verdict.json")], out / "raw-logs/verdict.log"), "E1 verdict")
        print("WINDOW_E1_V2_COLLECTION_OK")
        return 0
    except BaseException as exc:  # noqa: BLE001
        (out / "failure.json").write_text(json.dumps({"schema": "glm53-e1-v2-failure-v1", "error": str(exc)}, indent=2) + "\n")
        if restore_required:
            restore(args, out, runtime_recipe)
        print(f"WINDOW_E1_V2_FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        release_host_lock(args, host_lock_fd)
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
    parser.add_argument("--restore-only", action="store_true", help="unconditionally restore incumbent and prove it; no release gate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--docker-context", default=None)
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--health", default=None)
    parser.add_argument("--nsys", default="/opt/nvidia/nsight-systems/2025.6.3/bin/nsys")
    parser.add_argument("--api-probe", default=None, help="test hook executable that writes restore API proofs")
    parser.add_argument("--command-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--readiness-timeout-sec", type=float, default=READINESS_TIMEOUT_SEC)
    parser.add_argument("--host-operation-lock", default=HOST_OPERATION_LOCK)
    args = parser.parse_args(argv)
    if args.run_id != RUN_ID:
        parser.error(f"v2 runner requires run id {RUN_ID}")
    if not args.restore_only and args.root is None:
        parser.error("--root is required unless --restore-only is set")
    args.recipe = args.recipe.resolve()
    if args.root is not None:
        args.root = args.root.resolve()
    args.out = args.out.resolve()
    return args


if __name__ == "__main__":
    _args = parse_args()
    raise SystemExit(restore(_args, _args.out) if _args.restore_only else run_e1(_args))
