#!/usr/bin/env python3
"""Portable closed-loop tool-execution gauntlet."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    files: Mapping[str, str]
    expected: Mapping[str, Any]


def _final_instruction(shape: str) -> str:
    return (
        "Return the final answer as exactly one compact JSON object, with no markdown, "
        "no code fences, and no prose. Expected shape: " + shape
    )


def _chain_task(task_id: str, paths: List[str], answer: str) -> Task:
    files: Dict[str, str] = {}
    for index, path in enumerate(paths[:-1]):
        files[path] = (
            f"chain-step={index + 1}\n"
            f"Read the next file named exactly: {paths[index + 1]}\n"
            "Do not guess the final nonce.\n"
        )
    files[paths[-1]] = (
        f"chain-step={len(paths)}\n"
        f"final nonce: {answer}\n"
        "Return it in the required JSON object.\n"
    )
    prompt = (
        "This is a serial file-chain task. Use read_file(path) starting at "
        f"{paths[0]!r}; each file names the next file. You need the opaque final nonce. "
        + _final_instruction('{"answer":"<nonce>"}')
    )
    return Task(task_id, "file_chain", prompt, files, {"answer": answer, "required_reads": paths})


def build_tasks() -> List[Task]:
    """Return the fixed public-safe task corpus. Expected answers are fixed, not model-derived."""
    tasks: List[Task] = [
        _chain_task(
            "chain-amber",
            ["start.txt", "rungs/a17.txt", "rungs/cobalt-42.txt", "notes/kite.txt", "vault/final.txt"],
            "nonce-amber-4d2f8a",
        ),
        _chain_task(
            "chain-basil",
            ["entry.txt", "maze/left-03.txt", "maze/green-owl.txt", "maze/hinge.txt", "out/last.txt"],
            "nonce-basil-91c0ee",
        ),
        _chain_task(
            "chain-cinder",
            ["open-me.txt", "map/river.txt", "map/stone-19.txt", "map/lamp.txt", "answer/omega.txt"],
            "nonce-cinder-7aa135",
        ),
        _chain_task(
            "chain-delta",
            ["first.txt", "trail/blue.txt", "trail/fern-88.txt", "trail/quiet.txt", "done/nonce.txt"],
            "nonce-delta-b60f2c",
        ),
        Task(
            "code-slugify",
            "code_fix",
            "Fix slugify.py. Public tests: slugify(' Hello, Mars! ')== 'hello-mars'; "
            "slugify('GPU__Box  42') == 'gpu-box-42'; slugify('...') == ''. "
            + _final_instruction('{"status":"fixed","summary":"<short>"}'),
            {"slugify.py": "def slugify(text):\n    return text.replace(' ', '-')\n"},
            {"module": "slugify.py"},
        ),
        Task(
            "code-intervals",
            "code_fix",
            "Fix intervals.py. Public tests: merge_intervals([[5,7],[1,3],[2,4],[9,10]]) == "
            "[[1,4],[5,7],[9,10]]; merge_intervals([[1,2],[2,5]]) == [[1,5]]; "
            "merge_intervals([]) == []. "
            + _final_instruction('{"status":"fixed","summary":"<short>"}'),
            {"intervals.py": "def merge_intervals(intervals):\n    return intervals\n"},
            {"module": "intervals.py"},
        ),
        Task(
            "code-totals",
            "code_fix",
            "Fix totals.py. Public tests: totals_by_category reads CSV text with header category,amount; "
            "sums integer amounts per category; ignores blank lines. "
            + _final_instruction('{"status":"fixed","summary":"<short>"}'),
            {"totals.py": "def totals_by_category(csv_text):\n    return {}\n"},
            {"module": "totals.py"},
        ),
        Task(
            "code-brackets",
            "code_fix",
            "Fix brackets.py. Public tests: is_balanced handles (), [], {}, nesting, and ignores other characters; "
            "'([{}])' is true; '([)]' is false; 'notes (v2) [ok]' is true. "
            + _final_instruction('{"status":"fixed","summary":"<short>"}'),
            {"brackets.py": "def is_balanced(text):\n    return True\n"},
            {"module": "brackets.py"},
        ),
        Task(
            "struct-inventory",
            "structured",
            "Read inventory.csv and compute total_qty, reorder_skus sorted alphabetically where reorder is yes, "
            "and largest_sku by qty. " + _final_instruction('{"answer":{"total_qty":0,"reorder_skus":[],"largest_sku":""}}'),
            {"inventory.csv": "sku,qty,reorder\ngear,7,no\nseal,2,yes\naxle,4,yes\ncam,10,no\n"},
            {"final_json": {"answer": {"total_qty": 23, "reorder_skus": ["axle", "seal"], "largest_sku": "cam"}}},
        ),
        Task(
            "struct-incidents",
            "structured",
            "Read incidents.json. Return open_p1_ids sorted ascending and closed_count. "
            + _final_instruction('{"answer":{"open_p1_ids":[],"closed_count":0}}'),
            {"incidents.json": "{\"incidents\":[{\"id\":\"INC-004\",\"priority\":1,\"status\":\"open\"},{\"id\":\"INC-002\",\"priority\":2,\"status\":\"closed\"},{\"id\":\"INC-001\",\"priority\":1,\"status\":\"open\"},{\"id\":\"INC-003\",\"priority\":1,\"status\":\"closed\"}]}\n"},
            {"final_json": {"answer": {"open_p1_ids": ["INC-001", "INC-004"], "closed_count": 2}}},
        ),
        Task(
            "struct-shipments",
            "structured",
            "Read shipments.tsv. Return total_weight_kg as an integer and delayed_routes sorted alphabetically. "
            + _final_instruction('{"answer":{"total_weight_kg":0,"delayed_routes":[]}}'),
            {"shipments.tsv": "route\tweight_kg\tstatus\nA7\t11\ton-time\nB2\t5\tdelayed\nC9\t13\tdelayed\nD4\t7\ton-time\n"},
            {"final_json": {"answer": {"total_weight_kg": 36, "delayed_routes": ["B2", "C9"]}}},
        ),
        Task(
            "struct-readings",
            "structured",
            "Read readings.txt. Each row is sensor=value. Return sensors_over_50 sorted and min_sensor. "
            + _final_instruction('{"answer":{"sensors_over_50":[],"min_sensor":""}}'),
            {"readings.txt": "alpha=48\nbravo=72\ncharlie=15\ndelta=51\n"},
            {"final_json": {"answer": {"sensors_over_50": ["bravo", "delta"], "min_sensor": "charlie"}}},
        ),
    ]
    return tasks


def parse_final_json(content: str) -> tuple[Optional[Any], bool, List[str]]:
    def unique_object(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj:
                raise ValueError("duplicate JSON key")
            obj[k] = v
        return obj
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")
    try:
        parsed = json.loads(content.strip(), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except Exception:
        return None, False, ["malformed_final_json"]
    if not isinstance(parsed, dict):
        return parsed, False, ["final_json_not_object"]
    return parsed, True, []


def _same_json_types(value: Any, example: Any) -> bool:
    """The fixed task rubric uses exact decoded JSON types and object keys."""
    if type(value) is not type(example):
        return False
    if isinstance(example, dict):
        return set(value)==set(example) and all(_same_json_types(value[k],example[k]) for k in example)
    if isinstance(example, list):
        return not example or all(_same_json_types(item,example[0]) for item in value)
    return True


def evaluate_task_result(task: Task, final_content: str, trace: List[Mapping[str, Any]], oracle_result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    parsed, final_valid, issues = parse_final_json(final_content or "")
    reasons: List[str] = []
    correct = False
    schema_valid = False
    if not final_valid:
        reasons.append("invalid_final_answer")
    elif task.kind == "file_chain":
        schema_valid = _same_json_types(parsed, {"answer": task.expected["answer"]})
        answer_ok = parsed == {"answer": task.expected["answer"]}
        required = list(task.expected["required_reads"])
        first_reads = {}
        for event in trace:
            if event.get("event") != "tool_call" or event.get("name") != "read_file" or event.get("result", {}).get("ok") is not True:
                continue
            args = event.get("args")
            path = args.get("path") if isinstance(args, Mapping) else None
            if isinstance(path, str) and path in required and path not in first_reads:
                first_reads[path] = event.get("model_turn")
        first_turns = list(first_reads.values())
        chain_ok = (
            list(first_reads) == required
            and all(type(turn) is int and turn > 0 for turn in first_turns)
            and first_turns == sorted(first_turns)
            and len(set(first_turns)) >= 4
        )
        if not answer_ok:
            reasons.append("wrong_answer")
        if not chain_ok:
            reasons.append("required_read_chain_missing")
        correct = answer_ok and chain_ok
    elif task.kind == "structured":
        schema_valid = _same_json_types(parsed, task.expected["final_json"])
        answer_ok = json.dumps(parsed,sort_keys=True) == json.dumps(task.expected["final_json"],sort_keys=True)
        reads = {e.get("args",{}).get("path") for e in trace if e.get("event")=="tool_call" and e.get("name")=="read_file" and e.get("result",{}).get("ok") is True}
        grounded = set(task.files).issubset(reads)
        correct = answer_ok and grounded
        if not answer_ok:
            reasons.append("wrong_structured_answer")
        if not grounded:
            reasons.append("required_file_read_missing")
    elif task.kind == "code_fix":
        shape_ok = set(parsed)=={"status","summary"} and parsed.get("status") == "fixed" and isinstance(parsed.get("summary"), str) and bool(parsed.get("summary"))
        schema_valid = shape_ok
        oracle_ok = bool(oracle_result and oracle_result.get("ok") is True)
        if not shape_ok:
            reasons.append("wrong_code_final_shape")
        if not oracle_ok:
            reasons.append("tests_failed_or_not_run")
        correct = shape_ok and oracle_ok
    else:
        reasons.append("unknown_task_kind")
    return {
        "correct": bool(correct),
        "reasons": reasons,
        "validity": {"final_answer_valid": final_valid, "schema_valid": bool(schema_valid), "issues": issues},
        "parsed_final": parsed if final_valid else None,
    }


MAX_CAPTURE_BYTES = 8192
MAX_READ_BYTES = 65536
MAX_WRITE_BYTES = 65536

TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from this task directory only. Path must be safe relative; no symlinks/traversal.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 content into this task directory only. Path must be safe relative; no symlinks/traversal.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the fixed public evaluator for code-fix tasks inside the configured Docker sandbox.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

SYSTEM_PROMPT = """You are inside a bounded tool-execution gauntlet.
Use only the provided tools: read_file(path), write_file(path, content), and run_tests().
Paths are task-local safe relative paths; traversal, absolute paths, and symlinks are rejected.
For code-fix tasks, edit files with write_file and use run_tests() for feedback. Tests run in an isolated Docker sandbox, not on the host.
When finished, answer with exactly one JSON object matching the task's requested shape and no prose.
""".strip()


def prepare_task_dir(task: Task, task_dir: Path) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    fs = SafeTaskFS(task_dir, MAX_READ_BYTES, MAX_WRITE_BYTES)
    for relpath, content in task.files.items():
        result = fs.write_file(relpath, content)
        if not result.get("ok"):
            raise RuntimeError(f"failed to prepare {task.id}:{relpath}: {result}")


def _endpoint_for_base_url(base_url: str) -> str:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'replay'))
    import replay_matched as wire
    return wire.normalize_urls(base_url)[0]


class OpenAIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout_s: int = 180):
        self.endpoint = _endpoint_for_base_url(base_url)
        if api_key is not None:
            raise ValueError("Round 7 does not support credentials")
        self.timeout_s = timeout_s
        self.cache_salt = "r7-gauntlet-default"

    def chat(self, messages: List[Mapping[str, Any]], model: str, tools: List[Mapping[str, Any]], reasoning_effort: str, max_tokens: int) -> Dict[str, Any]:
        payload = {
            "model": model,
            "temperature": 0,
            "seed": 42,
            "cache_salt": self.cache_salt,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        # Share the reviewed SSE/usage boundary with recorded-history replay.
        import sys, hashlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'replay'))
        import replay_matched as wire
        body = wire.canonical_json_bytes(payload)
        result = wire.stream_chat_completion(self.endpoint, body, self.timeout_s)
        content = result['response']['content']
        message = {'role':'assistant', 'content': content or None}
        if result['response']['tool_calls']:
            message['tool_calls'] = result['response']['tool_calls']
        return {
            'message': message,
            'finish_reason': result['finish_reason'],
            'usage': result['usage'],
            'reasoning': result['response']['reasoning'],
            'raw_response': result['response'],
            'timing': {'wall_s':result['duration_s'], 'ttft_s':result['ttft_s'], 'ttfa_s':result['ttfa_s']},
            'request_body_sha256': hashlib.sha256(body).hexdigest(),
        }


class GauntletRunner:
    def __init__(
        self,
        client: Any,
        sandbox_runner: Optional[SandboxRunner],
        *,
        model: str = "test-model",
        reasoning_effort: str = "low",
        max_turns: int = 12,
        max_tool_calls: int = 32,
        task_timeout_s: int = 180,
        max_tokens: int = 1500,
        max_read_bytes: int = MAX_READ_BYTES,
        max_write_bytes: int = MAX_WRITE_BYTES,
    ):
        self.client = client
        self.sandbox_runner = sandbox_runner
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.task_timeout_s = task_timeout_s
        self.max_tokens = max_tokens
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def run_task(self, task: Task, task_dir: Path, raw_dir: Optional[Path]) -> Dict[str, Any]:
        raw_dir = Path(raw_dir) if raw_dir is not None else None
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{task.id}.jsonl" if raw_dir is not None else None
        fs = SafeTaskFS(task_dir, self.max_read_bytes, self.max_write_bytes)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
        trace: List[Dict[str, Any]] = []
        validity_issues: List[str] = []
        tool_errors: List[Dict[str, Any]] = []
        tool_counts: Dict[str, int] = {"read_file": 0, "write_file": 0, "run_tests": 0, "unknown": 0}
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason: Optional[str] = None
        first_ttft: Optional[float] = None
        final_content = ""
        tool_call_count = 0
        started = time.monotonic()
        for turn in range(1, self.max_turns + 1):
            if time.monotonic() - started > self.task_timeout_s:
                validity_issues.append("task_timeout")
                break
            _append_jsonl(raw_path, {"event": "request", "turn": turn, "messages": messages})
            remaining = self.task_timeout_s - (time.monotonic() - started)
            if hasattr(self.client, 'timeout_s'):
                self.client.timeout_s = min(self.client.timeout_s, max(.001, remaining))
            try:
                response = self.client.chat(messages, self.model, TOOL_DEFS, self.reasoning_effort, self.max_tokens)
            except Exception as exc:
                _append_jsonl(raw_path, {"partial_response": getattr(exc, "partial_response", None), "event":"transport_error", "turn":turn, "error":str(exc)})
                validity_issues.append("transport_error")
                break
            _append_jsonl(raw_path, {"event": "response", "turn": turn, "response": response})
            if time.monotonic() - started > self.task_timeout_s:
                validity_issues.append("task_timeout")
                break
            finish_reason = response.get("finish_reason")
            timing = response.get("timing") or {}
            if first_ttft is None and timing.get("ttft_s") is not None:
                first_ttft = timing.get("ttft_s")
            _merge_usage(usage, response.get("usage") or {})
            assistant_message = response.get("message") or {"role": "assistant", "content": ""}
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            if finish_reason == "length":
                validity_issues.append("token_budget_exhausted")
                final_content = assistant_message.get("content") or ""
                break
            if not tool_calls:
                final_content = assistant_message.get("content") or ""
                break
            exhausted_tools = False
            for call in tool_calls:
                if tool_call_count >= self.max_tool_calls:
                    validity_issues.append("tool_call_limit_exhausted")
                    exhausted_tools = True
                    break
                tool_call_count += 1
                result, event, issue = self._execute_tool(call, fs, task, Path(task_dir))
                event["model_turn"] = turn
                if issue:
                    validity_issues.append(issue)
                if not result.get("ok"):
                    tool_errors.append({"id": event.get("id"), "name": event.get("name"), "error": result.get("error")})
                trace.append(event)
                tool_counts[event.get("name") if event.get("name") in tool_counts else "unknown"] += 1
                tool_message = {
                    "role": "tool",
                    "tool_call_id": event.get("id") or "missing-tool-call-id",
                    "content": json.dumps({k:v for k,v in result.items() if k != "command"}, sort_keys=True, separators=(",", ":")),
                }
                messages.append(tool_message)
                _append_jsonl(raw_path, {"event": "tool_result", "turn": turn, "tool_call": event, "tool_message": tool_message})
            if exhausted_tools:
                break
        else:
            validity_issues.append("budget_exhausted")
        if not final_content:
            validity_issues.append("budget_exhausted")
        oracle_result: Optional[Mapping[str, Any]] = None
        if task.kind == "code_fix":
            if self.sandbox_runner is None:
                oracle_result = {"ok": False, "error": "sandbox_not_configured"}
            else:
                oracle_result = self.sandbox_runner.run_tests(task.id, Path(task_dir))
            _append_jsonl(raw_path, {"event": "oracle_after_final", "task_id": task.id, "result": oracle_result})
        score = evaluate_task_result(task, final_content, trace, oracle_result)
        validity_issues.extend(score["validity"].get("issues", []))
        transport_protocol_issues = {
            "transport_error",
            "token_budget_exhausted",
            "unknown_tool",
            "malformed_tool_arguments",
            "tool_arguments_not_object",
            "missing_tool_call_id",
            "tool_call_limit_exhausted",
            "budget_exhausted",
            "task_timeout",
        }
        transport_protocol_valid = not any(issue in transport_protocol_issues for issue in validity_issues)
        wall_s = time.monotonic() - started
        if wall_s > self.task_timeout_s:
            if "task_timeout" not in validity_issues:
                validity_issues.append("task_timeout")
            transport_protocol_valid = False
        result = {
            "task_id": task.id,
            "kind": task.kind,
            "correct": score["correct"],
            "success": bool(score["correct"] and transport_protocol_valid and score["validity"]["final_answer_valid"]),
            "reasons": score["reasons"],
            "validity": {
                "transport_protocol_valid": transport_protocol_valid,
                "final_answer_valid": score["validity"]["final_answer_valid"],
                "schema_valid": score["validity"]["schema_valid"],
                "tool_error_count": len(tool_errors),
                "tool_errors": tool_errors,
                "issues": sorted(set(validity_issues)),
            },
            "score_count": 1,
            "wall_s": wall_s,
            "ttft_s": first_ttft,
            "tool_counts": tool_counts,
            "tool_call_count": tool_call_count,
            "usage": usage,
            "finish_reason": finish_reason,
            "final_content": final_content,
            "oracle_result": oracle_result,
            "trace": trace,
        }
        _append_jsonl(raw_path, {"event": "score", "result": result})
        if raw_path is not None:
            raw_path.chmod(0o400)
        return result

    def _execute_tool(self, call: Mapping[str, Any], fs: SafeTaskFS, task: Task, task_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        call_id = call.get("id")
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or ""
        event: Dict[str, Any] = {"event": "tool_call", "id": call_id, "name": name, "raw_arguments": raw_args, "args": None}
        issue: Optional[str] = None
        if not call_id:
            issue = "missing_tool_call_id"
        if name not in {"read_file", "write_file", "run_tests"}:
            event["name"] = name or "unknown"
            result = {"ok": False, "error": "unknown_tool"}
            event["result"] = result
            return result, event, "unknown_tool"
        try:
            args = json.loads(raw_args or "{}")
        except Exception:
            result = {"ok": False, "error": "malformed_tool_arguments"}
            event["result"] = result
            return result, event, "malformed_tool_arguments"
        if not isinstance(args, dict):
            result = {"ok": False, "error": "tool_arguments_not_object"}
            event["result"] = result
            return result, event, "tool_arguments_not_object"
        event["args"] = args
        if name == "read_file":
            result = fs.read_file(args.get("path"))
        elif name == "write_file":
            result = fs.write_file(args.get("path"), args.get("content"))
        else:
            if task.kind != "code_fix":
                result = {"ok": False, "error": "run_tests_not_applicable"}
            elif self.sandbox_runner is None:
                result = {"ok": False, "error": "sandbox_not_configured"}
            else:
                result = self.sandbox_runner.run_tests(task.id, task_dir)
        event["result"] = result
        return result, event, issue


def _merge_usage(total: Dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _append_jsonl(path: Optional[Path], obj: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def _contains_subsequence(observed: List[str], required: List[str]) -> bool:
    pos = 0
    for item in observed:
        if pos < len(required) and item == required[pos]:
            pos += 1
    return pos == len(required)


def _truncate_text(value: str, limit: int = MAX_CAPTURE_BYTES) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    keep = max(0, limit - 80)
    return encoded[:keep].decode("utf-8", errors="replace") + f"\n...[truncated to {limit} bytes]"


def parse_evaluator_process_result(returncode: int, stdout: str, stderr: str, timed_out: bool, output_limit: int = MAX_CAPTURE_BYTES) -> Dict[str, Any]:
    stdout_t = _truncate_text(stdout or "", output_limit)
    stderr_t = _truncate_text(stderr or "", output_limit)
    base: Dict[str, Any] = {"exit_code": returncode, "stdout": stdout_t, "stderr": stderr_t}
    if timed_out:
        return {**base, "ok": False, "error": "timeout"}
    markers = [line[len("GAUNTLET_RESULT ") :] for line in stdout_t.splitlines() if line.startswith("GAUNTLET_RESULT ")]
    if len(markers) != 1:
        return {**base, "ok": False, "error": "missing_evaluator_marker" if not markers else "multiple_evaluator_markers"}
    try:
        payload = json.loads(markers[0])
    except Exception:
        return {**base, "ok": False, "error": "malformed_evaluator_marker"}
    if not isinstance(payload, dict):
        return {**base, "ok": False, "error": "malformed_evaluator_marker"}
    base["evaluator"] = payload
    if payload.get("ok") is True and returncode == 0:
        return {**base, "ok": True}
    if payload.get("ok") is False:
        return {**base, "ok": False, "error": "evaluator_reported_failure"}
    return {**base, "ok": False, "error": "evaluator_exit_nonzero" if returncode != 0 else "evaluator_not_ok"}


class SandboxRunner:
    def __init__(self, sandbox_image: str, evaluator_path: Path, timeout_s: int = 20, docker_bin: str = "docker"):
        self.sandbox_image = sandbox_image
        self.evaluator_path = Path(evaluator_path).resolve()
        self.timeout_s = timeout_s
        self.docker_bin = docker_bin

    def build_command(self, task_id: str, task_dir: Path) -> List[str]:
        task_dir = Path(task_dir).resolve()
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--name", "r7-eval-" + uuid.uuid4().hex,
            "--pull", "never",
            "--entrypoint", "python3",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory-swap", "256m",
            "--runtime=runc",
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
            "256m",
            "--cpus",
            "1",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{task_dir}:/task:ro",
            "-v",
            f"{self.evaluator_path}:/trusted_evaluator/evaluator.py:ro",
            self.sandbox_image,
            "/trusted_evaluator/evaluator.py",
            task_id,
        ]

    def run_tests(self, task_id: str, task_dir: Path) -> Dict[str, Any]:
        cmd = self.build_command(task_id, task_dir)
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False)
            result = parse_evaluator_process_result(completed.returncode, completed.stdout, completed.stderr, False)
        except subprocess.TimeoutExpired as exc:
            result = parse_evaluator_process_result(-1, exc.stdout or "", exc.stderr or "", True)
        except Exception as exc:
            result = {"ok": False, "error": "docker_invocation_failed", "exception": str(exc), "stdout": "", "stderr": ""}
        finally:
            # Killing the Docker client does not kill the container. Cleanup owns
            # only this unique disposable CPU sandbox, never a model container.
            name = cmd[cmd.index("--name") + 1]
            clean = subprocess.run([self.docker_bin, "rm", "-f", name], capture_output=True, text=True, timeout=15, check=False)
            absent = f"No such container: {name}" in clean.stderr
            if clean.returncode != 0 and not absent:
                raise RuntimeError("CPU sandbox cleanup could not be verified: " + name)
        result["command"] = cmd
        return result


class SafeTaskFS:
    def __init__(self, root, max_read_bytes=65536, max_write_bytes=65536):
        self.root = Path(root).resolve()
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def _resolve(self, relpath: str) -> Path:
        if not isinstance(relpath, str) or not relpath or "\x00" in relpath:
            raise ValueError("unsafe_path")
        if "\\" in relpath or len(relpath.encode("utf-8")) > 240:
            raise ValueError("unsafe_path")
        parsed = PurePosixPath(relpath)
        if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
            raise ValueError("unsafe_path")
        candidate = self.root.joinpath(*parsed.parts)
        probe = self.root
        for part in parsed.parts:
            probe = probe / part
            if probe.exists() and probe.is_symlink():
                raise ValueError("unsafe_path_symlink")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError("unsafe_path")
        return resolved

    def read_file(self, path: str) -> dict:
        try:
            target = self._resolve(path)
            if target.is_symlink():
                return {"ok": False, "error": "unsafe_path_symlink"}
            if not target.is_file():
                return {"ok": False, "error": "not_file"}
            data = target.read_bytes()
            if len(data) > self.max_read_bytes:
                return {"ok": False, "error": "read_limit"}
            return {"ok": True, "content": data.decode("utf-8"), "bytes": len(data)}
        except UnicodeDecodeError:
            return {"ok": False, "error": "decode_error"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def write_file(self, path: str, content: str) -> dict:
        try:
            if not isinstance(content, str):
                return {"ok": False, "error": "content_must_be_string"}
            data = content.encode("utf-8")
            if len(data) > self.max_write_bytes:
                return {"ok": False, "error": "write_limit"}
            target = self._resolve(path)
            if target.exists() and target.is_symlink():
                return {"ok": False, "error": "unsafe_path_symlink"}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {"ok": True, "bytes": len(data)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
