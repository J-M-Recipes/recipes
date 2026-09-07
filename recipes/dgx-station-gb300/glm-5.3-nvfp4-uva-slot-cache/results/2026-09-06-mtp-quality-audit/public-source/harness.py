#!/usr/bin/env python3
"""Local-only objective task evaluation harness.

Stdlib-only OpenAI-compatible /v1/chat/completions client plus deterministic
oracles for HumanEval and GSM8K fixtures. Generated code is never executed in
host Python; HumanEval scoring uses a Docker CPU sandbox supplied by CLI.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import secrets as _secrets
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

DEFAULT_SEED = 20260906
DEFAULT_REPEATS = 2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "low"


class ProtocolError(Exception):
    """A remote API response or model output violated the harness protocol."""


class ResumeError(Exception):
    """Existing JSONL evidence cannot be safely resumed or appended to."""


# ------------------------- redaction and JSONL -------------------------


def redact_text(text: Any, secrets: Iterable[str] = ()) -> Any:
    if not isinstance(text, str):
        return text
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "<redacted>")
    return out


_TRANSPORT_SECRET_FIELDS = {"authorization", "api_key", "api-key", "x-api-key", "key", "token"}


def redact_obj(obj: Any, secrets: Iterable[str] = ()) -> Any:
    if isinstance(obj, dict):
        return {k: ("<redacted>" if isinstance(k, str) and k.lower() in _TRANSPORT_SECRET_FIELDS else redact_obj(v, secrets)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, secrets) for v in obj]
    return redact_text(obj, secrets)


def safe_error(exc: BaseException, secrets: Iterable[str] = ()) -> str:
    return redact_text(str(exc), secrets)


def _ensure_jsonl_appendable(path: str, validate_json: bool = False) -> None:
    if not path or not os.path.exists(path):
        return
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            raise ResumeError(f"JSONL file has a partial final line: {path}")
    if validate_json:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    raise ResumeError(f"JSONL line {line_no} is blank")
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ResumeError(f"JSONL line {line_no} is corrupt: {e}") from e


def append_jsonl(path: str, record: Dict[str, Any], secrets: Iterable[str] = ()) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _ensure_jsonl_appendable(path, validate_json=True)
    safe = redact_obj(record, secrets)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _request_identity(model: str, max_tokens: int, reasoning_effort: str) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }


def resume_identity(
    fixture: Dict[str, Any],
    lane: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    return {
        "schema": "gb300-mtp-quality-result-v1",
        "lane": lane,
        "model": model,
        "fixture_set_hash": fixture.get("metadata", {}).get("fixture_set_hash"),
        "request": _request_identity(model, max_tokens, reasoning_effort),
        "task_fixture_hashes": {t.get("id"): t.get("fixture_hash") for t in fixture.get("tasks", []) if isinstance(t.get("id"), str)},
    }


def _resume_request_matches(record_request: Any, expected_request: Dict[str, Any]) -> bool:
    if not isinstance(record_request, dict):
        return False
    normalized = {k: v for k, v in record_request.items() if k not in {"seed", "messages"}}
    return normalized == expected_request


def _validate_resume_record(rec: Any, line_no: int, expected: Optional[Dict[str, Any]]) -> Tuple[str, str, int]:
    if not isinstance(rec, dict):
        raise ResumeError(f"JSONL line {line_no} is not an object")
    lane = rec.get("lane")
    task_id = rec.get("task_id")
    repeat = rec.get("repeat")
    if not (isinstance(lane, str) and isinstance(task_id, str) and isinstance(repeat, int)):
        raise ResumeError(f"JSONL line {line_no} is missing lane/task_id/repeat identity")
    if expected is not None:
        if rec.get("schema") != expected.get("schema"):
            raise ResumeError(f"JSONL line {line_no} schema does not match current run")
        if lane != expected.get("lane"):
            raise ResumeError(f"JSONL line {line_no} lane does not match current run")
        if rec.get("model") != expected.get("model"):
            raise ResumeError(f"JSONL line {line_no} model does not match current run")
        if rec.get("fixture_set_hash") != expected.get("fixture_set_hash"):
            raise ResumeError(f"JSONL line {line_no} fixture_set_hash does not match current run")
        task_hashes = expected.get("task_fixture_hashes") or {}
        if task_id not in task_hashes or rec.get("fixture_hash") != task_hashes[task_id]:
            raise ResumeError(f"JSONL line {line_no} fixture_hash does not match current run")
        if not _resume_request_matches(rec.get("request"), expected.get("request") or {}):
            raise ResumeError(f"JSONL line {line_no} request config does not match current run")
    return lane, task_id, repeat


def load_completed(path: str, expected: Optional[Dict[str, Any]] = None) -> set[Tuple[str, str, int]]:
    done: set[Tuple[str, str, int]] = set()
    if not path or not os.path.exists(path):
        return done
    _ensure_jsonl_appendable(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ResumeError(f"JSONL line {line_no} is blank")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ResumeError(f"JSONL line {line_no} is corrupt: {e}") from e
            done.add(_validate_resume_record(rec, line_no, expected))
    return done


def iter_jobs(fixture: Dict[str, Any], lane: str, repeats: int, limit: Optional[int], completed: set[Tuple[str, str, int]]) -> Iterator[Dict[str, Any]]:
    emitted = 0
    for task in fixture.get("tasks", []):
        task_id = task["id"]
        for repeat in range(repeats):
            if (lane, task_id, repeat) in completed:
                continue
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield {"task": task, "repeat": repeat}


# ------------------------- request/response -------------------------


def read_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("key file is empty")
    return key


def stable_seed(task_id: str, repeat: int, base_seed: int = DEFAULT_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task_id}:{repeat}".encode("utf-8")).digest()
    # OpenAI-compatible servers commonly accept signed 32-bit seeds.
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_chat_request(model: str, prompt: str, max_tokens: int, reasoning_effort: str, seed: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


def call_chat(base_url: str, key: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(base_url),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            status = getattr(resp, "status", None)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            detail = ""
        raise ProtocolError(redact_text(f"HTTP {e.code}: {e.reason}: {detail}", [key])) from None
    except urllib.error.URLError as e:
        raise ProtocolError(redact_text(f"URL error: {e.reason}", [key])) from None
    except TimeoutError as e:
        raise ProtocolError("request timeout") from e
    wall = time.monotonic() - start

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except Exception as e:
        raise ProtocolError(f"response was not valid JSON: {e}") from e

    try:
        choice = raw["choices"][0]
        message = choice.get("message")
    except Exception as e:
        raise ProtocolError(f"response missing choices[0].message: {e}") from e
    if not isinstance(message, dict):
        raise ProtocolError("response missing choices[0].message object")

    content = message.get("content")
    raw_reasoning = (
        message.get("reasoning_content")
        if "reasoning_content" in message
        else message.get("reasoning")
    )
    finish = choice.get("finish_reason")
    usage = raw.get("usage") or {}
    tool_calls = message.get("tool_calls")
    invalid_reasons: List[str] = []
    if content is None:
        invalid_reasons.append("null-content")
    elif not isinstance(content, str):
        invalid_reasons.append("non-string-content")
    if finish is None:
        invalid_reasons.append("finish-missing")
    elif finish == "length":
        invalid_reasons.append("finish-length")
    elif finish in {"content_filter", "error"}:
        invalid_reasons.append(f"finish-{finish}")
    elif finish != "stop":
        invalid_reasons.append(f"finish-unexpected-{finish}")
    if tool_calls:
        invalid_reasons.append("unexpected-tool-calls")

    return {
        "http_status": status,
        "wall_seconds": wall,
        "content": content,
        "raw_reasoning": raw_reasoning,
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "usage": usage,
        "raw_response": raw,
        "invalid": bool(invalid_reasons),
        "invalid_reasons": invalid_reasons,
    }


# ------------------------- prompts and oracles -------------------------


def make_prompt(task: Dict[str, Any]) -> str:
    typ = task.get("type")
    if typ == "humaneval":
        return (
            "Write a complete Python function or module that solves the task below. "
            "Include the required function signature exactly as shown. Return only Python code; "
            "no explanations.\n\n"
            + task["prompt"]
        )
    if typ == "gsm8k":
        return (
            "Solve the grade-school math problem. You may reason internally, but the final "
            "answer must be on its own line tagged exactly as FINAL: number.\n\n"
            + task["question"]
        )
    raise ValueError(f"unsupported task type: {typ!r}")


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.-]*)\s*\n(.*?)```", re.DOTALL)


def extract_python(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("empty code content")
    matches = list(_FENCE_RE.finditer(text))
    if matches:
        if len(matches) != 1:
            raise ProtocolError("expected exactly one Python code fence")
        lang = matches[0].group(1).strip().lower()
        if lang not in {"", "python", "py"}:
            raise ProtocolError(f"unsupported code fence language: {lang}")
        code = matches[0].group(2).strip() + "\n"
        outside = (text[: matches[0].start()] + text[matches[0].end() :]).strip()
        if outside:
            raise ProtocolError("prose outside Python code fence")
        return code
    stripped = text.strip() + "\n"
    if not re.search(r"(^|\n)\s*(def|class|import|from)\s+", stripped):
        raise ProtocolError("plain response does not look like Python code")
    return stripped


def _number_to_fraction(text: str) -> Optional[Fraction]:
    s = text.strip().replace(",", "")
    try:
        if "/" in s:
            left, right = s.split("/", 1)
            return Fraction(Decimal(left.strip())) / Fraction(Decimal(right.strip()))
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


_FINAL_RE = re.compile(
    r"(?im)^\s*FINAL:\s*([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)\s*$"
)


def parse_final_number(text: str) -> Optional[Fraction]:
    if not isinstance(text, str):
        return None
    matches = _FINAL_RE.findall(text)
    if len(matches) != 1:
        return None
    return _number_to_fraction(matches[0])


def parse_gsm_expected(answer: str) -> Fraction:
    m = re.search(r"####\s*([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)", answer)
    if not m:
        raise ValueError("GSM8K answer missing #### numeric final answer")
    val = _number_to_fraction(m.group(1))
    if val is None:
        raise ValueError("GSM8K expected answer is not numeric")
    return val


def score_gsm(content: str, expected: Any) -> Tuple[bool, Dict[str, Any]]:
    predicted = parse_final_number(content)
    if predicted is None:
        return False, {"oracle": "gsm8k-final-number", "malformed": True, "parsed": None, "expected": str(expected)}
    expected_fraction = expected if isinstance(expected, Fraction) else _number_to_fraction(str(expected))
    if expected_fraction is None:
        raise ValueError(f"bad expected numeric value: {expected!r}")
    return predicted == expected_fraction, {
        "oracle": "gsm8k-final-number",
        "malformed": False,
        "parsed": str(predicted),
        "expected": str(expected_fraction),
    }


_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_local_image_id(image: str) -> bool:
    return bool(isinstance(image, str) and _IMAGE_ID_RE.fullmatch(image.strip()))


def resolve_sandbox_image(
    image: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    timeout: int = 10,
) -> str:
    if not image:
        raise ValueError("--sandbox-image is required for Python code execution")
    image = image.strip()
    if is_local_image_id(image):
        return image
    run = runner or subprocess.run
    cmd = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    try:
        proc = run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"failed to resolve sandbox image locally before run: timeout inspecting {image!r}") from e
    except FileNotFoundError as e:
        raise ValueError(f"failed to resolve sandbox image locally before run: {e}") from e
    if getattr(proc, "returncode", 1) != 0:
        stderr = getattr(proc, "stderr", "") or ""
        raise ValueError(f"failed to resolve sandbox image locally before run: {stderr.strip() or image}")
    resolved = (getattr(proc, "stdout", "") or "").strip()
    if not is_local_image_id(resolved):
        raise ValueError(f"docker image inspect did not return a sha256 image ID for {image!r}")
    return resolved


def docker_command(image: str) -> List[str]:
    if not is_local_image_id(image):
        raise ValueError("sandbox image must be a local sha256 image ID")
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--entrypoint",
        "python3",
        image,
        "-",
    ]


def run_python_in_sandbox(
    program: str,
    image: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    timeout: int = 10,
    required_stdout: Optional[str] = None,
) -> Dict[str, Any]:
    if not image:
        raise ValueError("--sandbox-image is required for Python code execution")
    try:
        image_id = resolve_sandbox_image(image, runner=runner)
        cmd = docker_command(image_id)
    except ValueError as e:
        return {"ok": False, "timeout": False, "error": str(e), "stdout": "", "stderr": "", "returncode": None}
    run = runner or subprocess.run
    try:
        proc = run(cmd, input=program, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "error": "timeout", "stdout": "", "stderr": "", "returncode": None}
    except FileNotFoundError as e:
        return {"ok": False, "timeout": False, "error": str(e), "stdout": "", "stderr": "", "returncode": None}
    returncode = getattr(proc, "returncode", 1)
    stdout = getattr(proc, "stdout", "")
    stderr = getattr(proc, "stderr", "")
    sentinel_ok = True
    if required_stdout is not None:
        sentinel_ok = any(line.strip() == required_stdout for line in stdout.splitlines())
    ok = returncode == 0 and sentinel_ok
    error = None
    if returncode != 0:
        error = "nonzero-exit"
    elif not sentinel_ok:
        error = "missing-sentinel"
    return {
        "ok": ok,
        "timeout": False,
        "error": error,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


def humaneval_program(code: str, task: Dict[str, Any], sentinel: Optional[str] = None) -> Tuple[str, str]:
    entry = task["entry_point"]
    test = task["test"]
    sentinel = sentinel or "GB300_HUMANEVAL_SENTINEL_" + _secrets.token_urlsafe(32)
    child_ok_code = 40 + _secrets.randbelow(80)
    child_code = (
        "import json, sys, traceback\n"
        "payload = json.loads(sys.stdin.read())\n"
        "_NS = {}\n"
        "try:\n"
        "    exec(compile(payload['code'], '<candidate>', 'exec'), _NS)\n"
        "    exec(compile(payload['test'], '<hidden-tests>', 'exec'), _NS)\n"
        "    _NS['check'](_NS[payload['entry']])\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
        f"sys.exit({child_ok_code})\n"
    )
    payload = json.dumps({"code": code, "test": test, "entry": entry}, ensure_ascii=False)
    return (
        "import json as _gb300_json\n"
        "import subprocess as _gb300_subprocess\n"
        "import sys as _gb300_sys\n"
        f"_CHILD_CODE = {child_code!r}\n"
        f"_PAYLOAD = {payload!r}\n"
        f"_CHILD_OK_CODE = {child_ok_code!r}\n"
        "_proc = _gb300_subprocess.run([_gb300_sys.executable, '-I', '-c', _CHILD_CODE], input=_PAYLOAD, text=True, capture_output=True)\n"
        "if _proc.stdout:\n"
        "    _gb300_sys.stdout.write(_proc.stdout)\n"
        "if _proc.stderr:\n"
        "    _gb300_sys.stderr.write(_proc.stderr)\n"
        "if _proc.returncode != _CHILD_OK_CODE:\n"
        "    _gb300_sys.exit(1)\n"
        f"print({sentinel!r})\n",
        sentinel,
    )


def score_humaneval(
    content: str,
    task: Dict[str, Any],
    sandbox_image: str,
    sandbox: Callable[[str, str], Dict[str, Any]] = run_python_in_sandbox,
) -> Tuple[bool, Dict[str, Any]]:
    try:
        code = extract_python(content)
    except ProtocolError as e:
        return False, {"oracle": "humaneval-hidden-tests", "malformed": True, "error": str(e)}
    program, sentinel = humaneval_program(code, task)
    result = sandbox(program, sandbox_image, required_stdout=sentinel)
    safe_result = {k: v for k, v in result.items() if k in {"ok", "timeout", "error", "returncode", "stdout", "stderr"}}
    return bool(result.get("ok")), {"oracle": "humaneval-hidden-tests", "malformed": False, "sandbox": safe_result}


def score_task(content: str, task: Dict[str, Any], sandbox_image: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
    if task.get("type") == "gsm8k":
        expected = task.get("expected_numeric")
        if expected is None:
            expected = str(parse_gsm_expected(task["answer"]))
        return score_gsm(content, expected)
    if task.get("type") == "humaneval":
        if not sandbox_image:
            raise ValueError("--sandbox-image is required when fixtures include HumanEval tasks")
        return score_humaneval(content, task, sandbox_image)
    raise ValueError(f"unsupported task type: {task.get('type')!r}")


# ------------------------- CLI -------------------------


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    if not isinstance(fixture.get("tasks"), list):
        raise ValueError("fixture JSON must contain a tasks list")
    return fixture


def run_eval(args: argparse.Namespace) -> int:
    fixture = load_fixture(args.fixtures)
    if any(t.get("type") == "humaneval" for t in fixture["tasks"]) and not args.sandbox_image:
        raise SystemExit("--sandbox-image is required when evaluating HumanEval tasks")
    key = read_key(args.key_file)
    identity = resume_identity(fixture, args.lane, args.model, args.max_tokens, args.reasoning_effort)
    completed = load_completed(args.out, identity)
    total = 0
    summary = {
        "total_count": 0,
        "valid_count": 0,
        "pass_count": 0,
        "fail_count": 0,
        "invalid_count": 0,
        "protocol_error_count": 0,
        "oracle_error_count": 0,
        "error_count": 0,
    }
    for job in iter_jobs(fixture, args.lane, args.repeats, args.limit, completed):
        task = job["task"]
        repeat = job["repeat"]
        task_id = task["id"]
        seed = stable_seed(task_id, repeat, fixture.get("metadata", {}).get("seed", DEFAULT_SEED))
        payload = build_chat_request(args.model, make_prompt(task), args.max_tokens, args.reasoning_effort, seed)
        base_record: Dict[str, Any] = {
            "schema": "gb300-mtp-quality-result-v1",
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "lane": args.lane,
            "task_id": task_id,
            "task_type": task.get("type"),
            "repeat": repeat,
            "seed": seed,
            "model": args.model,
            "fixture_hash": task.get("fixture_hash"),
            "fixture_set_hash": fixture.get("metadata", {}).get("fixture_set_hash"),
            "request": {k: v for k, v in payload.items() if k != "messages"},
        }
        try:
            response = call_chat(args.base_url, key, payload, timeout=args.http_timeout)
            content = response.get("content")
            invalid = bool(response.get("invalid"))
            invalid_reasons = list(response.get("invalid_reasons") or [])
            passed = False
            oracle: Dict[str, Any] = {"skipped": True}
            if not invalid:
                try:
                    passed, oracle = score_task(content or "", task, args.sandbox_image)
                    if oracle.get("malformed"):
                        invalid = True
                        invalid_reasons.append("malformed-output")
                except Exception as e:
                    invalid = True
                    invalid_reasons.append("oracle-error")
                    oracle = {"error": safe_error(e, [key])}
            record = {
                **base_record,
                "protocol_error": False,
                "invalid": invalid,
                "invalid_reasons": invalid_reasons,
                "pass": bool(passed) if not invalid else False,
                "oracle": oracle,
                "response": response,
            }
        except ProtocolError as e:
            record = {
                **base_record,
                "protocol_error": True,
                "invalid": True,
                "invalid_reasons": ["protocol-error"],
                "pass": False,
                "error": safe_error(e, [key]),
            }
        append_jsonl(args.out, record, secrets=[key])
        total += 1
        summary["total_count"] += 1
        if record.get("protocol_error"):
            summary["protocol_error_count"] += 1
        if record.get("invalid"):
            summary["invalid_count"] += 1
        else:
            summary["valid_count"] += 1
            if record.get("pass"):
                summary["pass_count"] += 1
            else:
                summary["fail_count"] += 1
        if "oracle-error" in (record.get("invalid_reasons") or []):
            summary["oracle_error_count"] += 1
        if record.get("error") and not record.get("protocol_error"):
            summary["error_count"] += 1
    print(json.dumps({"wrote": total, "out": args.out, "lane": args.lane, "summary": summary}, sort_keys=True))
    return 1 if (summary["invalid_count"] or summary["protocol_error_count"] or summary["oracle_error_count"] or summary["error_count"]) else 0


def run_selfcheck(args: argparse.Namespace) -> int:
    fixture = load_fixture(args.fixtures)
    failures = []
    count = 0
    for task in fixture.get("tasks", []):
        if task.get("type") != "humaneval":
            continue
        count += 1
        canonical = task.get("prompt", "") + task.get("canonical_solution", "")
        ok, detail = score_humaneval(canonical, task, args.sandbox_image)
        if not ok:
            failures.append({"task_id": task.get("id"), "detail": detail})
            if args.fail_fast:
                break
    result = {"humaneval_checked": count, "failures": failures, "ok": not failures}
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local-only GB300 MTP objective eval harness")
    sub = p.add_subparsers(dest="command", required=True)

    def add_run_args(r: argparse.ArgumentParser) -> None:
        r.add_argument("--base-url", required=True)
        r.add_argument("--model", required=True)
        r.add_argument("--key-file", required=True)
        r.add_argument("--lane", required=True)
        r.add_argument("--out", required=True, help="JSONL output path; appends and checkpoints")
        r.add_argument("--fixtures", required=True)
        r.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
        r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        r.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
        r.add_argument("--limit", type=int, default=None, help="pilot limit on uncompleted task-repeat jobs")
        r.add_argument("--sandbox-image", default=None, help="required for HumanEval code scoring")
        r.add_argument("--http-timeout", type=int, default=600)

    run = sub.add_parser("run", help="run evaluation (default when no subcommand is supplied)")
    add_run_args(run)
    run.set_defaults(func=run_eval)

    selfcheck = sub.add_parser("selfcheck", help="validate HumanEval canonical solutions through Docker sandbox")
    selfcheck.add_argument("--fixtures", required=True)
    selfcheck.add_argument("--sandbox-image", required=True)
    selfcheck.add_argument("--fail-fast", action="store_true")
    selfcheck.set_defaults(func=run_selfcheck)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Backward-compatible default command: `harness.py --base-url ...` means run.
    if raw and raw[0] not in {"run", "selfcheck", "-h", "--help"}:
        raw.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(raw)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
