#!/usr/bin/env python3
"""Bounded Round 7 Station campaign controller."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class CmdResult:
    stdout: str
    stderr: str
    returncode: int


class SubprocessRuntime:
    def run(self, argv: List[str], timeout_s: Optional[float] = None) -> CmdResult:
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_s, check=False)
        return CmdResult(proc.stdout, proc.stderr, proc.returncode)


class ControllerError(RuntimeError):
    """Refusal or campaign failure."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControllerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"JSON file must contain an object: {path}")
    return value


def _safe_manifest_path(release: pathlib.Path, rel: str) -> pathlib.Path:
    if not isinstance(rel, str) or rel == "":
        raise ControllerError("manifest path must be a nonempty string")
    pure = pathlib.PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        raise ControllerError(f"manifest path rejects traversal/absolute path: {rel}")
    target = release.joinpath(*pure.parts)
    current = release
    if current.is_symlink():
        raise ControllerError("release directory must not be a symlink")
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ControllerError(f"manifest path rejects symlink: {rel}")
    try:
        target.relative_to(release)
    except ValueError as exc:
        raise ControllerError(f"manifest path escapes release: {rel}") from exc
    return target


def verify_manifest(release: pathlib.Path) -> Dict[str, str]:
    manifest_path = release / "manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ControllerError("manifest files must be a nonempty object")
    verified: Dict[str, str] = {}
    for rel, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ControllerError(f"manifest hash is not lowercase sha256 for {rel}")
        path = _safe_manifest_path(release, rel)
        if not path.is_file():
            raise ControllerError(f"manifest file missing: {rel}")
        actual = sha256_hex(path.read_bytes())
        if actual != expected:
            raise ControllerError(f"manifest hash mismatch for {rel}")
        verified[rel] = actual
    return verified


def read_control(release: pathlib.Path, run_id: str) -> None:
    control = (release / "CONTROL").read_text(encoding="utf-8")
    if control not in ("RUN", "RUN\n"):
        raise ControllerError("CONTROL is not RUN")
    release_id = (release / "RELEASE").read_text(encoding="utf-8")
    if release_id not in (run_id, run_id + "\n"):
        raise ControllerError("RELEASE does not match run-id")


def verify_release_inputs(
    release: pathlib.Path,
    contract_path: pathlib.Path,
    fixture_path: pathlib.Path,
    output: pathlib.Path,
    run_id: str,
) -> Dict[str, Any]:
    release = pathlib.Path(release)
    contract_path = pathlib.Path(contract_path)
    fixture_path = pathlib.Path(fixture_path)
    output = pathlib.Path(output)
    if output.exists():
        raise ControllerError("output directory must be new")
    verified = verify_manifest(release)
    if "CONTROL" not in verified or "RELEASE" not in verified:
        raise ControllerError("manifest must include CONTROL and RELEASE")
    contract = _read_json(contract_path)
    expected_fixture = contract.get("fixture_sha256")
    if not isinstance(expected_fixture, str):
        raise ControllerError("contract missing fixture_sha256")
    actual_fixture = sha256_hex(fixture_path.read_bytes())
    if actual_fixture != expected_fixture:
        raise ControllerError("fixture sha256 mismatch")
    read_control(release, run_id)
    try:
        contract_rel = contract_path.resolve().relative_to(release.resolve()).as_posix()
    except ValueError as exc:
        raise ControllerError("contract must be inside release manifest") from exc
    if contract_rel not in verified:
        raise ControllerError("contract is not bound by manifest")
    required = {"controller/controller.py", "replay/replay_matched.py", "replay/analyze_runs.py",
                "gauntlet/gauntlet.py", "gauntlet/oracle.py", "gauntlet/run_suite.py"}
    if not required.issubset(verified):
        raise ControllerError("manifest omits an executed helper or analyzer")
    undeclared = {p.relative_to(release).as_posix() for p in release.rglob("*.py")} - set(verified)
    if undeclared:
        raise ControllerError("manifest omits Python files: " + ", ".join(sorted(undeclared)))
    return contract


def parse_boot_order(contract: Dict[str, Any]) -> list[tuple[int, str]]:
    order = contract.get("order")
    if not isinstance(order, list):
        raise ControllerError("contract order must be a list")
    parsed: list[tuple[int, str]] = []
    profiles = contract.get("profiles", {})
    for item in order:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], str)):
            raise ControllerError("contract order entries must be [pair, profile]")
        if item[1] not in profiles:
            raise ControllerError(f"contract order references unknown profile {item[1]}")
        parsed.append((item[0], item[1]))
    expected = [(1, "v14"), (1, "v13"), (2, "v13"), (2, "v14"), (3, "v14"), (3, "v13")]
    if parsed != expected:
        raise ControllerError("contract order is not the exact Round 7 six-boot order")
    return parsed


def _unit_names(unit: str) -> tuple[str, str]:
    if unit.endswith(".timer"):
        return unit, unit[:-6] + ".service"
    if unit.endswith(".service"):
        return unit[:-8] + ".timer", unit
    return unit + ".timer", unit + ".service"


def verify_guard_timer(runtime: Any, guard_unit: str, v14_id: str, v13_id: str) -> None:
    timer_unit, service_unit = _unit_names(guard_unit)
    active = runtime.run(["systemctl", "is-active", timer_unit], timeout_s=30)
    if active.returncode != 0 or active.stdout.strip() != "active":
        raise ControllerError(f"guard timer is not active: {timer_unit}")
    shown = runtime.run(["systemctl", "show", "--value", "-p", "ExecStart", service_unit], timeout_s=30)
    if shown.returncode != 0:
        raise ControllerError(f"cannot read guard service ExecStart: {service_unit}")
    text = shown.stdout.strip()
    if "path=/usr/bin/docker" not in text:
        raise ControllerError("guard ExecStart path is not /usr/bin/docker")
    match = re.search(r"argv\[\]=([^;]+);", text)
    if not match:
        raise ControllerError("guard ExecStart has no argv[]")
    try:
        argv = shlex.split(match.group(1).strip())
    except ValueError as exc:
        raise ControllerError("guard ExecStart argv[] is not parseable") from exc
    expected = ["/usr/bin/docker", "stop", "--time", "30", v14_id, v13_id]
    if argv != expected:
        raise ControllerError("guard ExecStart argv[] does not exactly stop v14 then v13")


def _docker_state(doc: Dict[str, Any]) -> Dict[str, Any]:
    state = doc.get("State")
    return state if isinstance(state, dict) else {}


class DockerController:
    def __init__(self, contract: Dict[str, Any], runtime: Any, port_is_dark: Callable[[], bool]):
        self.contract = contract
        self.runtime = runtime
        self.port_is_dark = port_is_dark
        self.owned_started: Dict[str, str] = {}

    def _run_ok(self, argv: List[str], timeout_s: Optional[float] = None) -> CmdResult:
        result = self.runtime.run(argv, timeout_s=timeout_s)
        if result.returncode != 0:
            raise ControllerError(f"command failed: {' '.join(argv)}: {result.stderr.strip()}")
        return result

    def _inspect_doc(self, cid: str) -> Dict[str, Any]:
        result = self._run_ok(["docker", "inspect", cid], timeout_s=30)
        try:
            docs = json.loads(result.stdout)
        except Exception as exc:
            raise ControllerError(f"docker inspect did not return JSON for {cid}") from exc
        if not isinstance(docs, list) or len(docs) != 1 or not isinstance(docs[0], dict):
            raise ControllerError(f"docker inspect returned unexpected shape for {cid}")
        return docs[0]

    def _verify_identity(self, profile_key: str, doc: Dict[str, Any]) -> Dict[str, str]:
        profile = self.contract.get("profiles", {}).get(profile_key)
        if not isinstance(profile, dict):
            raise ControllerError(f"missing profile in contract: {profile_key}")
        ident = runtime_identity(doc)
        checks = {
            "id": profile.get("id"),
            "name": profile.get("name"),
            "image": profile.get("image"),
            "config_image": profile.get("config_image"),
            "runtime_sha256": profile.get("runtime_sha256"),
        }
        for key, expected in checks.items():
            if ident.get(key) != expected:
                raise ControllerError(f"container identity mismatch for {profile_key}: {key}")
        return ident

    def assert_dark_before_start(self) -> None:
        running = self._run_ok(["docker", "ps", "--format", "{{json .}}"], timeout_s=30).stdout.strip()
        if running:
            raise ControllerError("refusing start: a container is already running")
        gpu = self._run_ok(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            timeout_s=30,
        ).stdout.strip()
        if gpu:
            raise ControllerError("refusing start: GPU compute process is active")
        if not self.port_is_dark():
            raise ControllerError("refusing start: port 30006 is not dark")

    def start_profile(self, profile_key: str) -> Dict[str, str]:
        self.assert_dark_before_start()
        cid = str(self.contract["profiles"][profile_key]["id"])
        before = self._inspect_doc(cid)
        self._verify_identity(profile_key, before)
        if _docker_state(before).get("Running"):
            raise ControllerError("refusing start: target container is already running")
        self.owned_started[cid] = profile_key
        self._run_ok(["docker", "start", cid], timeout_s=60)
        after = self._inspect_doc(cid)
        ident = self._verify_identity(profile_key, after)
        if not _docker_state(after).get("Running"):
            raise ControllerError("docker start returned but container is not running")
        ident["started_at"] = str(_docker_state(after).get("StartedAt", ""))
        return ident

    def collect_logs_since(self, cid: str, since: str, boot_dir: pathlib.Path) -> Dict[str, Any]:
        stdout_path = pathlib.Path(boot_dir) / "docker-logs.stdout.log"
        stderr_path = pathlib.Path(boot_dir) / "docker-logs.stderr.log"
        result = self.runtime.run(["docker", "logs", "--since", since, cid], timeout_s=60)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        receipt = {"id": cid, "since": since, "returncode": result.returncode, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        write_json(pathlib.Path(boot_dir) / "docker-logs.receipt.json", receipt)
        return receipt

    def stop_owned(self) -> List[Dict[str, Any]]:
        proofs: List[Dict[str, Any]] = []
        for cid, profile_key in list(self.owned_started.items())[::-1]:
            doc = self._inspect_doc(cid)
            ident = self._verify_identity(profile_key, doc)
            if ident["id"] != cid:
                raise ControllerError("owned container id changed before stop")
            if _docker_state(doc).get("Running"):
                self._run_ok(["docker", "stop", "--time", "30", cid], timeout_s=90)
            stopped = self._inspect_doc(cid)
            self._verify_identity(profile_key, stopped)
            running = bool(_docker_state(stopped).get("Running"))
            if running:
                raise ControllerError("container did not stop")
            port_dark = self.port_is_dark()
            if not port_dark:
                raise ControllerError("port 30006 is not dark after stop")
            proof = {"profile": profile_key, "id": cid, "running": running, "port_30006_dark": port_dark}
            proofs.append(proof)
            self.owned_started.pop(cid, None)
        return proofs


class CampaignController:
    def __init__(
        self,
        contract: Dict[str, Any],
        release: pathlib.Path,
        fixture: pathlib.Path,
        output: pathlib.Path,
        run_id: str,
        guard_unit: str,
        runtime: Any,
        phase_runner: Any,
        port_is_dark: Callable[[], bool],
        guard_check: Optional[Callable[[], None]] = None,
        readiness_check: Optional[Callable[[str, pathlib.Path], None]] = None,
        receipts: Optional[Any] = None,
    ):
        self.contract = contract
        self.release = pathlib.Path(release)
        self.fixture = pathlib.Path(fixture)
        self.output = pathlib.Path(output)
        self.run_id = run_id
        self.guard_unit = guard_unit
        self.runtime = runtime
        self.phase_runner = phase_runner
        self.guard_check = guard_check or (lambda: None)
        self.readiness_check = readiness_check or (lambda profile_key, boot_dir: None)
        self.receipts = receipts
        self.docker = DockerController(contract, runtime, port_is_dark)

    def check_control_and_guard(self) -> None:
        read_control(self.release, self.run_id)
        self.guard_check()

    def run_one_boot(self, pair: int, profile_key: str) -> Dict[str, Any]:
        boot_dir = self.output / "boots" / f"pair{pair}-{profile_key}"
        boot_dir.mkdir(parents=True, exist_ok=False)
        if self.receipts is not None:
            self.receipts.append({"event": "boot_start", "pair": pair, "profile": profile_key})
        self.check_control_and_guard()
        identity = {}
        try:
            identity = self.docker.start_profile(profile_key)
            write_json(boot_dir / "identity.json", {"pair": pair, "profile": profile_key, "identity": identity})
            if self.receipts is not None:
                self.receipts.append({"event": "boot_identity", "pair": pair, "profile": profile_key, "identity": identity})
            self.check_control_and_guard()
            readiness = self.readiness_check(profile_key, boot_dir)
            write_json(boot_dir / "readiness.json", {"ok": True, "detail": readiness})
            self.check_control_and_guard()
            result = self.phase_runner.run_boot(self, pair, profile_key, boot_dir)
            write_json(boot_dir / "phases.json", result)
            if self.receipts is not None:
                self.receipts.append({"event": "boot_phases_complete", "pair": pair, "profile": profile_key, "phases_path": str(boot_dir / "phases.json")})
            return {"pair": pair, "profile": profile_key, "identity": identity, "phases": result}
        finally:
            try:
                cid = identity.get("id", "")
                since = identity.get("started_at", "")
                if cid and since:
                    try:
                        logs = self.docker.collect_logs_since(cid, since, boot_dir)
                        if self.receipts is not None:
                            self.receipts.append({"event": "boot_logs_saved", "pair": pair, "profile": profile_key, "logs": logs})
                    except Exception as exc:
                        write_json(boot_dir / "docker-logs.error.json", {"error_type": type(exc).__name__, "error": str(exc)})
            finally:
                stop_proof = self.docker.stop_owned()
                write_json(boot_dir / "stop-proof.json", {"proof": stop_proof})
                if self.receipts is not None:
                    self.receipts.append({"event": "boot_stopped", "pair": pair, "profile": profile_key, "proof": stop_proof})


class ReceiptWriter:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("t", time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ReleaseLock:
    def __init__(self, release: pathlib.Path):
        self.path = pathlib.Path(release).parent / ".round7-controller.lock"
        self.handle: Optional[Any] = None

    def __enter__(self) -> "ReleaseLock":
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ControllerError("another Round 7 controller holds the release lock") from exc
        self.handle.write(f"pid={os.getpid()} time={time.time()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def port_30006_dark() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        return sock.connect_ex(("127.0.0.1", 30006)) != 0
    finally:
        sock.close()


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _model_context_ok(models_doc: Dict[str, Any], model: str, context: int) -> bool:
    data = models_doc.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if not isinstance(item, dict) or item.get("id") != model:
            continue
        for key in ("max_model_len", "context_length", "max_context_length", "max_context", "model_len"):
            value = item.get(key)
            if isinstance(value, (int, float)) and int(value) == context:
                return True
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            for key in ("max_model_len", "context_length", "max_context_length", "max_context", "model_len"):
                value = metadata.get(key)
                if isinstance(value, (int, float)) and int(value) == context:
                    return True
    return False


def wait_for_readiness(
    docker: DockerController,
    profile_key: str,
    model: str,
    context: int,
    base_url: str,
    control_check: Callable[[], None],
    timeout_s: float = 20 * 60,
) -> Dict[str, Any]:
    cid = str(docker.contract["profiles"][profile_key]["id"])
    deadline = time.monotonic() + timeout_s
    last_error = "not checked"
    url = _models_url(base_url)
    while time.monotonic() < deadline:
        control_check()
        doc = docker._inspect_doc(cid)
        docker._verify_identity(profile_key, doc)
        if not _docker_state(doc).get("Running"):
            raise ControllerError("container exited before readiness")
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if getattr(response, "status", 200) < 200 or getattr(response, "status", 200) >= 300:
                    last_error = f"HTTP {getattr(response, 'status', 'unknown')}"
                else:
                    body = response.read().decode("utf-8")
                    models = json.loads(body)
                    if isinstance(models, dict) and _model_context_ok(models, model, context):
                        return {"ok": True, "model": model, "context": context, "url": url}
                    last_error = "model/context not present in /v1/models"
        except Exception as exc:
            last_error = str(exc)[:300]
        time.sleep(5)
    raise ControllerError(f"readiness timed out after {timeout_s}s: {last_error}")


def _load_replay_module(release: pathlib.Path) -> Any:
    path = pathlib.Path(release) / "replay" / "replay_matched.py"
    spec = importlib.util.spec_from_file_location("round7_replay_matched", path)
    if spec is None or spec.loader is None:
        raise ControllerError("cannot load replay module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sized_prompt(case: str, size: int) -> str:
    header = f"Round 7 deterministic excluded warmup {case}. Reply with the final marker only.\n"
    filler_unit = f"[{case}-seed42-warmup] "
    repeat = max(0, (size - len(header)) // len(filler_unit) + 1)
    return (header + filler_unit * repeat)[:size]


def run_prompt_warmups(
    release: pathlib.Path,
    model: str,
    base_url: str,
    cache_salt: str,
    boot_dir: pathlib.Path,
    control_check: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    replay = _load_replay_module(release)
    chat_url, _metrics_url = replay.normalize_urls(base_url) if hasattr(replay, "normalize_urls") else (base_url.rstrip("/") + "/chat/completions", "")
    tools = getattr(replay, "TOOLS", [])
    canonical = getattr(replay, "canonical_json_bytes", canonical_json_bytes)
    stream = getattr(replay, "stream_chat_completion")
    from concurrent.futures import ThreadPoolExecutor
    records = []
    for concurrency in range(1, 5):
        case = f"C{concurrency}"
        for size in (4096, 16384, 49152):
            if control_check is not None:
                control_check()
            def request(index):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Deterministic excluded warmup. Return WARMUP-OK."},
                        {"role": "user", "content": _sized_prompt(f"{case}-{index}", size)},
                    ],
                    "temperature": 0, "seed": 42, "max_tokens": 64,
                    "reasoning_effort": "low", "tools": tools, "tool_choice": "auto",
                    "stream": True, "stream_options": {"include_usage": True},
                    "cache_salt": cache_salt,
                }
                result = stream(chat_url, canonical(payload), timeout_s=180)
                usage = result.get("usage", {})
                if not isinstance(usage, dict) or int(usage.get("completion_tokens", 0)) <= 0:
                    raise ControllerError("prompt warmup did not produce completion tokens")
                return {"case":case,"request_index":index,"prompt_chars":size,
                        "request_sha256":sha256_hex(canonical(payload)),
                        "wall_s":result["duration_s"],"ttft_s":result.get("ttft_s"),
                        "usage":usage,"finish_reason":result.get("finish_reason"),
                        "response":result.get("response")}
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                records.extend(pool.map(request, range(concurrency)))
    summary = {
        "ok": True,
        "excluded_from_scoring": True,
        "settings": {"temperature": 0, "seed": 42, "reasoning_effort": "low", "max_tokens": 64},
        "cache_salt_sha256": sha256_hex(cache_salt.encode("utf-8")),
        "count": len(records),
        "records": records,
    }
    write_json(pathlib.Path(boot_dir) / "prompt-warmup.json", summary)
    return summary


def resolve_sandbox_image(contract: Dict[str, Any], release: pathlib.Path, explicit: Optional[str] = None) -> str:
    expected = contract.get("sandbox_image")
    if not isinstance(expected, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
        raise ControllerError("contract must pin an immutable sandbox image ID")
    if explicit is not None and explicit != expected:
        raise ControllerError("sandbox image override differs from reviewed contract")
    return expected


def verify_final_dark(docker: DockerController) -> Dict[str, Any]:
    stopped = []
    for profile_key in ("v14", "v13"):
        profile = docker.contract.get("profiles", {}).get(profile_key, {})
        cid = str(profile.get("id", ""))
        doc = docker._inspect_doc(cid)
        ident = docker._verify_identity(profile_key, doc)
        running = bool(_docker_state(doc).get("Running"))
        stopped.append({"profile": profile_key, "id": ident["id"], "running": running})
        if running:
            raise ControllerError(f"final state is not dark: {profile_key} is running")
    port_dark = docker.port_is_dark()
    if not port_dark:
        raise ControllerError("final state is not dark: port 30006 is open")
    return {"containers": stopped, "port_30006_dark": True}


def run_campaign(
    release: pathlib.Path,
    contract_path: pathlib.Path,
    fixture: pathlib.Path,
    output: pathlib.Path,
    run_id: str,
    guard_unit: str,
    runtime: Optional[Any] = None,
    phase_runner: Optional[Any] = None,
    sandbox_image: Optional[str] = None,
    base_url: str = "http://127.0.0.1:30006/v1",
) -> Dict[str, Any]:
    release = pathlib.Path(release)
    contract_path = pathlib.Path(contract_path)
    fixture = pathlib.Path(fixture)
    output = pathlib.Path(output)
    contract = verify_release_inputs(release, contract_path, fixture, output, run_id)
    order = parse_boot_order(contract)
    runtime = runtime or SubprocessRuntime()
    sandbox = resolve_sandbox_image(contract, release, sandbox_image)
    model = str(contract.get("model"))
    context = int(contract.get("context"))
    fixture_sha = str(contract.get("fixture_sha256"))
    with ReleaseLock(release):
        output.mkdir(parents=True, exist_ok=False)
        receipts = ReceiptWriter(output / "receipts.jsonl")
        deadline = time.monotonic() + 4 * 60 * 60
        v14_id = str(contract["profiles"]["v14"]["id"])
        v13_id = str(contract["profiles"]["v13"]["id"])

        def guard_check() -> None:
            if time.monotonic() > deadline:
                raise ControllerError("controller exceeded 4h bound")
            verify_guard_timer(runtime, guard_unit, v14_id, v13_id)

        runner = phase_runner or PhaseRunner(release, fixture, run_id, model, fixture_sha, sandbox, base_url=base_url)
        campaign = CampaignController(
            contract=contract,
            release=release,
            fixture=fixture,
            output=output,
            run_id=run_id,
            guard_unit=guard_unit,
            runtime=runtime,
            phase_runner=runner,
            port_is_dark=port_30006_dark,
            guard_check=guard_check,
            readiness_check=lambda profile_key, boot_dir: None,
            receipts=receipts,
        )
        campaign.readiness_check = lambda profile_key, boot_dir: wait_for_readiness(
            campaign.docker, profile_key, model, context, base_url, campaign.check_control_and_guard
        )
        receipts.append({"event": "campaign_start", "run_id": run_id, "measurement_only": True, "promotion_authorized": False})
        boots = []
        try:
            for pair, profile_key in order:
                boots.append(campaign.run_one_boot(pair, profile_key))
            final_dark = verify_final_dark(campaign.docker)
            summary = {
                "run_id": run_id,
                "measurement_only": True,
                "promotion_authorized": False,
                "boot_count": len(boots),
                "order": [[pair, profile] for pair, profile in order],
                "boots": boots,
                "final_state": final_dark,
                "statistics_note": "measurement-only campaign summary; no inferential verdicts or promotion authorization",
            }
            write_json(output / "summary.json", summary)
            receipts.append({"event": "campaign_complete", "run_id": run_id, "final_state": final_dark, "promotion_authorized": False})
            return summary
        except BaseException as exc:
            failure = {"event": "campaign_failed", "run_id": run_id, "error_type": type(exc).__name__, "error": str(exc), "promotion_authorized": False}
            receipts.append(failure)
            write_json(output / "failure.json", failure)
            raise


class ProcessRunner:
    def run(
        self,
        argv: List[str],
        cwd: pathlib.Path,
        timeout_s: float,
        control_check: Callable[[], None],
        stdout_path: pathlib.Path,
        stderr_path: pathlib.Path,
    ) -> Dict[str, Any]:
        started = time.time()
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_s
            try:
                while True:
                    code = proc.poll()
                    if code is not None:
                        if code != 0:
                            raise ControllerError(f"phase subprocess failed with exit {code}: {' '.join(argv)}")
                        return {"returncode": code, "argv": argv, "started": started, "ended": time.time()}
                    control_check()
                    if time.monotonic() >= deadline:
                        self._kill_group(proc.pid)
                        raise ControllerError(f"phase subprocess timed out after {timeout_s}s: {' '.join(argv)}")
                    time.sleep(1.0)
            except BaseException:
                if proc.poll() is None:
                    self._kill_group(proc.pid)
                raise

    @staticmethod
    def _kill_group(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _load_summary(path: pathlib.Path) -> Dict[str, Any]:
    return _read_json(path)


def validate_replay_summary(path: pathlib.Path, expected_fixture_sha: str) -> Dict[str, Any]:
    summary = _load_summary(path)
    if summary.get("fixture_sha256") != expected_fixture_sha:
        raise ControllerError("replay summary fixture hash mismatch")
    if summary.get("turn_count") != 300 or summary.get("expected_turn_count") != 300:
        raise ControllerError("replay summary does not contain exact 300/300 turns")
    if not _finite_positive(summary.get("wall_s")):
        raise ControllerError("replay summary wall_s must be finite positive")
    rate = summary.get("aggregate_completion_tok_s", summary.get("completion_tok_s"))
    if not _finite_positive(rate):
        raise ControllerError("replay summary rate must be finite positive")
    return summary


def validate_gauntlet_summary(path: pathlib.Path) -> Dict[str, Any]:
    summary = _load_summary(path)
    if summary.get("task_count") != 12 or summary.get("expected_task_count") != 12:
        raise ControllerError("gauntlet summary does not contain exact 12/12 tasks")
    if summary.get("capture_error_count") != 0:
        raise ControllerError("gauntlet summary has capture errors")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 12:
        raise ControllerError("gauntlet summary results must contain 12 records")
    ids = [record.get("task_id") for record in results if isinstance(record, dict)]
    if len(set(ids)) != 12:
        raise ControllerError("gauntlet summary task IDs are not unique")
    if not _finite_positive(summary.get("wall_s")):
        raise ControllerError("gauntlet summary wall_s must be finite positive")
    return summary


class PhaseRunner:
    def __init__(
        self,
        release: pathlib.Path,
        fixture: pathlib.Path,
        run_id: str,
        model: str,
        fixture_sha256: str,
        sandbox_image: str,
        process_runner: Optional[Any] = None,
        warmup_runner: Optional[Callable[..., Dict[str, Any]]] = None,
        base_url: str = "http://127.0.0.1:30006/v1",
    ):
        self.release = pathlib.Path(release)
        self.fixture = pathlib.Path(fixture)
        self.run_id = run_id
        self.model = model
        self.fixture_sha256 = fixture_sha256
        self.sandbox_image = sandbox_image
        self.process_runner = process_runner or ProcessRunner()
        self.warmup_runner = warmup_runner or run_prompt_warmups
        self.base_url = base_url

    def run_boot(self, campaign: "CampaignController", pair: int, profile_key: str, boot_dir: pathlib.Path) -> Dict[str, Any]:
        boot_dir.mkdir(parents=True, exist_ok=True)
        namespace = f"r7-{self.run_id}-pair{pair}"
        warmups = self.warmup_runner(
            self.release,
            self.model,
            self.base_url,
            f"{namespace}-warm-{profile_key}",
            boot_dir,
            control_check=campaign.check_control_and_guard,
        )
        warm_gauntlet = self._run_gauntlet(
            campaign,
            boot_dir,
            tag=f"R7-p{pair}-{profile_key}-agent-warmup",
            salt=f"r7-agent-warm-{self.run_id}-pair{pair}-{profile_key}",
            subdir="gauntlet-warmup",
            timeout_s=15 * 60,
        )
        replay_initial = self._run_replay(
            campaign,
            boot_dir,
            tag=f"R7-p{pair}-{profile_key}-fresh",
            salt=namespace,
            cache_state="initial",
            subdir="replay-initial",
        )
        replay_repeat = self._run_replay(
            campaign,
            boot_dir,
            tag=f"R7-p{pair}-{profile_key}-repeat",
            salt=namespace,
            cache_state="repeat",
            subdir="replay-repeat",
        )
        gauntlet_scored = self._run_gauntlet(
            campaign,
            boot_dir,
            tag=f"R7-p{pair}-{profile_key}-agent",
            salt=f"r7-agent-{self.run_id}-pair{pair}",
            subdir="gauntlet-scored",
            timeout_s=15 * 60,
        )
        return {
            "prompt_warmups": warmups,
            "gauntlet_warmup": warm_gauntlet,
            "replay_initial": replay_initial,
            "replay_repeat": replay_repeat,
            "gauntlet_scored": gauntlet_scored,
        }

    def _run_replay(self, campaign: "CampaignController", boot_dir: pathlib.Path, tag: str, salt: str, cache_state: str, subdir: str) -> Dict[str, Any]:
        out = boot_dir / subdir
        argv = [
            sys.executable,
            str(self.release / "replay" / "replay_matched.py"),
            "--fixture", str(self.fixture),
            "--base-url", self.base_url,
            "--model", self.model,
            "--output", str(out),
            "--tag", tag,
            "--workers", "4",
            "--max-tokens", "400",
            "--reasoning-effort", "low",
            "--cache-state", cache_state,
            "--cache-salt", salt,
        ]
        self.process_runner.run(argv, self.release, 20 * 60, campaign.check_control_and_guard, boot_dir / f"{subdir}.stdout.log", boot_dir / f"{subdir}.stderr.log")
        return validate_replay_summary(out / "summary.json", self.fixture_sha256)

    def _run_gauntlet(self, campaign: "CampaignController", boot_dir: pathlib.Path, tag: str, salt: str, subdir: str, timeout_s: float) -> Dict[str, Any]:
        out = boot_dir / subdir
        argv = [
            sys.executable,
            str(self.release / "gauntlet" / "run_suite.py"),
            "--base-url", self.base_url,
            "--model", self.model,
            "--output", str(out),
            "--tag", tag,
            "--cache-salt", salt,
            "--sandbox-image", self.sandbox_image,
            "--workers", "4",
        ]
        self.process_runner.run(argv, self.release, timeout_s, campaign.check_control_and_guard, boot_dir / f"{subdir}.stdout.log", boot_dir / f"{subdir}.stderr.log")
        return validate_gauntlet_summary(out / "summary.json")


def runtime_identity(doc: Dict[str, Any]) -> Dict[str, str]:
    material = {
        "Id": doc["Id"],
        "Name": doc["Name"],
        "Image": doc["Image"],
        "Config": doc["Config"],
        "HostConfig": doc["HostConfig"],
        "Mounts": sorted(doc["Mounts"], key=lambda x: x["Destination"]),
    }
    return {
        "id": str(doc["Id"]),
        "name": str(doc["Name"]),
        "image": str(doc["Image"]),
        "config_image": str(doc.get("Config", {}).get("Image", "")),
        "runtime_sha256": sha256_hex(canonical_json_bytes(material)),
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description="Measurement-only six-boot v13/v14 comparison; no promotion or service restoration.")
    for name in ("release","contract","fixture","output"):
        parser.add_argument("--"+name,required=True,type=pathlib.Path)
    for name in ("run-id","guard-unit"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--sandbox-image")
    parser.add_argument("--validate-only",action="store_true",help="Validate frozen local inputs only; never query or change the runtime.")
    args=parser.parse_args(argv)
    terminating=False
    def interrupt(_signum,_frame):
        nonlocal terminating
        if terminating:
            return
        terminating=True
        raise KeyboardInterrupt("operator stop")
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    try:
        if args.validate_only:
            verify_release_inputs(args.release,args.contract,args.fixture,args.output,args.run_id)
            print(json.dumps({"source_inputs_valid":True,"runtime_validated":False,"launch_authorized":False}))
        else:
            summary=run_campaign(args.release,args.contract,args.fixture,args.output,args.run_id,args.guard_unit,sandbox_image=args.sandbox_image)
            print(json.dumps({k:summary[k] for k in ("run_id","boot_count","measurement_only","promotion_authorized","final_state")},sort_keys=True))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        print(json.dumps({"error_type":type(exc).__name__,"error":str(exc),"promotion_authorized":False}),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
