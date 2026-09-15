#!/usr/bin/env python3
"""Matched-session replay instrument for OpenAI-compatible chat SSE endpoints.

The fixture is treated as a RECORDED-HISTORY PROXY: prior tool results are prose
inside recorded messages, not native Hermes closed-loop state.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import queue
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXPECTED_SESSION_COUNT = 20
EXPECTED_TURNS_PER_SESSION = 15
EXPECTED_TURN_COUNT = EXPECTED_SESSION_COUNT * EXPECTED_TURNS_PER_SESSION
CHAT_TIMEOUT_S = 900
METRICS_TIMEOUT_S = 15

SYSTEM_PROMPT = (
    "You are Milo, James's operations agent. This is a RECORDED-HISTORY PROXY replay: "
    "prior tool results are represented as user prose, using a simplified system prompt. "
    "It is not native Hermes closed-loop operation and is not evidence of production capacity. "
    "Be terse. Use tools when needed. Quote CDT times. Never rephrase a blocked command. "
    "Prefer tables for numbers."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

REQUIRED_METRICS = (
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)

METRIC_ALIASES = {
    "vllm:spec_decode_num_draft_tokens_total": "spec_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted_tokens",
    "vllm:spec_decode_num_drafts_total": "spec_draft_steps",
    "vllm:prefix_cache_queries_total": "prefix_cache_query_tokens",
    "vllm:prefix_cache_hits_total": "prefix_cache_hit_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens_metric",
    "vllm:generation_tokens_total": "generation_tokens_metric",
}

METRIC_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)


class ReplayError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Turn:
    index: int
    turn_id: str
    ctx: List[Dict[str, Any]]
    has_tool: bool


@dataclasses.dataclass(frozen=True)
class Session:
    entry_index: int
    identity: str
    entry_sha256: str
    source_session_sha256: Optional[str]
    turns: List[Turn]


@dataclasses.dataclass(frozen=True)
class LoadedFixture:
    fixture_sha256: str
    sessions: List[Session]
    expected_turn_ids: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    fixture: pathlib.Path
    base_url: str
    model: str
    output: pathlib.Path
    tag: str
    workers: int
    max_tokens: int
    reasoning_effort: str
    cache_state: str
    cache_salt: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixture_bytes(raw: bytes) -> LoadedFixture:
    fixture_sha = sha256_hex(raw)
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ReplayError(f"fixture is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ReplayError("fixture must be a JSON list")
    if len(data) != EXPECTED_SESSION_COUNT:
        raise ReplayError(f"fixture must contain exactly {EXPECTED_SESSION_COUNT} sessions, got {len(data)}")

    sessions: List[Session] = []
    expected: List[str] = []
    for entry_index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ReplayError(f"fixture entry {entry_index} must be an object")
        turns_data = entry.get("turns")
        if not isinstance(turns_data, list):
            raise ReplayError(f"fixture entry {entry_index} missing turns list")
        if len(turns_data) != EXPECTED_TURNS_PER_SESSION:
            raise ReplayError(
                f"fixture entry {entry_index} must contain exactly {EXPECTED_TURNS_PER_SESSION} turns, got {len(turns_data)}"
            )
        entry_sha = sha256_hex(canonical_json_bytes(entry))
        source_session = entry.get("session")
        source_session_sha = sha256_hex(str(source_session).encode("utf-8")) if source_session is not None else None
        identity = f"entry-{entry_index:03d}-{entry_sha[:16]}"
        turns: List[Turn] = []
        for turn_index, turn_data in enumerate(turns_data):
            if not isinstance(turn_data, dict):
                raise ReplayError(f"fixture entry {entry_index} turn {turn_index} must be an object")
            ctx = turn_data.get("ctx")
            if not isinstance(ctx, list):
                raise ReplayError(f"fixture entry {entry_index} turn {turn_index} missing ctx list")
            for msg_index, msg in enumerate(ctx):
                if not isinstance(msg, dict):
                    raise ReplayError(f"fixture entry {entry_index} turn {turn_index} message {msg_index} must be an object")
                if set(msg.keys()) != {"role", "content"}:
                    raise ReplayError(
                        f"fixture entry {entry_index} turn {turn_index} message {msg_index} must have role/content only"
                    )
                if msg.get("role") not in {"user", "assistant"}:
                    raise ReplayError(f"fixture entry {entry_index} turn {turn_index} message {msg_index} has invalid role")
                if not isinstance(msg.get("content"), str):
                    raise ReplayError(f"fixture entry {entry_index} turn {turn_index} message {msg_index} content must be string")
            has_tool = turn_data.get("has_tool")
            if not isinstance(has_tool, bool):
                raise ReplayError(f"fixture entry {entry_index} turn {turn_index} has_tool must be boolean")
            turn_id = f"{identity}/turn-{turn_index:02d}"
            turns.append(Turn(index=turn_index, turn_id=turn_id, ctx=ctx, has_tool=has_tool))
            expected.append(turn_id)
        sessions.append(Session(entry_index, identity, entry_sha, source_session_sha, turns))

    if len(expected) != EXPECTED_TURN_COUNT:
        raise ReplayError(f"fixture must contain exactly {EXPECTED_TURN_COUNT} turns, got {len(expected)}")
    if len(set(s.identity for s in sessions)) != EXPECTED_SESSION_COUNT:
        raise ReplayError("entry-derived session identities are not distinct")
    if len(set(expected)) != EXPECTED_TURN_COUNT:
        raise ReplayError("entry-derived turn identities are not distinct")
    return LoadedFixture(fixture_sha, sessions, tuple(expected))


def load_fixture_path(path: pathlib.Path) -> LoadedFixture:
    return load_fixture_bytes(path.read_bytes())


def validate_loopback_url(url: str) -> str:
    if not isinstance(url, str) or any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise ReplayError("Round 7 requires a literal loopback HTTP URL")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ReplayError("invalid loopback URL") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or port == 0):
        raise ReplayError("Round 7 requires credential-free HTTP on 127.0.0.1")
    return url


def normalize_urls(base_url: str) -> Tuple[str, str]:
    validate_loopback_url(base_url)
    if urllib.parse.urlsplit(base_url).path not in ("", "/", "/v1", "/v1/"):
        raise ReplayError("base URL must use the API root or /v1")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        chat_root = base
        metrics_root = base[: -len("/v1")]
    else:
        chat_root = base + "/v1"
        metrics_root = base
    return chat_root + "/chat/completions", metrics_root + "/metrics"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirects are forbidden", headers, fp)


def open_loopback(request: Any, timeout: float):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    validate_loopback_url(url)
    if isinstance(request, urllib.request.Request):
        allowed = {key.lower(): value for key, value in auth_headers().items()}
        if any(allowed.get(key.lower()) != value for key, value in request.header_items()):
            raise ReplayError("custom request headers are unavailable in Round 7")
    # A private opener cannot inherit the process-global proxy/auth/cookie state.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def auth_headers() -> Dict[str, str]:
    """Fixed non-authenticated headers; ambient credentials are never read."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    return headers


def request_static_config(config: ReplayConfig) -> Dict[str, Any]:
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": 0,
        "seed": 42,
        "tools": TOOLS,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": config.reasoning_effort,
        "cache_salt": config.cache_salt,
        "system_prompt_sha256": sha256_hex(SYSTEM_PROMPT.encode("utf-8")),
        "history_proxy": "recorded-history-proxy-not-native-hermes-closed-loop",
    }


def request_config_hash(config: ReplayConfig) -> str:
    return sha256_hex(canonical_json_bytes(request_static_config(config)))


def comparison_config_hash(config: ReplayConfig) -> str:
    settings = request_static_config(config).copy()
    settings.pop("cache_salt")
    return sha256_hex(canonical_json_bytes(settings))


def request_public_static_config(config: ReplayConfig) -> Dict[str, Any]:
    public = request_static_config(config).copy()
    public.pop("cache_salt", None)
    public["cache_salt_sha256"] = sha256_hex(config.cache_salt.encode("utf-8"))
    return public


def build_request_payload(config: ReplayConfig, turn: Turn) -> Dict[str, Any]:
    payload = request_static_config(config).copy()
    payload.pop("system_prompt_sha256")
    payload.pop("history_proxy")
    payload["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + turn.ctx
    return payload


def request_shape_digest(payload: Dict[str, Any]) -> str:
    shaped: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "messages":
            shaped[key] = [
                {
                    "role": msg.get("role"),
                    "content_sha256": sha256_hex(str(msg.get("content", "")).encode("utf-8")),
                    "content_bytes": len(str(msg.get("content", "")).encode("utf-8")),
                }
                for msg in value
            ]
        else:
            shaped[key] = value
    return sha256_hex(canonical_json_bytes(shaped))


def _merge_tool_call_delta(acc: List[Dict[str, Any]], delta_calls: Any) -> None:
    if not isinstance(delta_calls, list):
        raise ReplayError("tool_calls delta must be a list")
    for delta in delta_calls:
        if not isinstance(delta, dict):
            raise ReplayError("tool_call delta entries must be objects")
        idx = delta.get("index", len(acc))
        if not isinstance(idx, int) or idx < 0:
            raise ReplayError("tool_call delta index must be a nonnegative integer")
        while len(acc) <= idx:
            acc.append({"index": len(acc), "id": "", "type": "", "function": {"name": "", "arguments": ""}})
        target = acc[idx]
        for key in ("id", "type"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                target[key] += value if target[key] and target[key] != value else ("" if target[key] == value else value)
        fn = delta.get("function")
        if isinstance(fn, dict):
            if isinstance(fn.get("name"), str) and fn.get("name"):
                if target["function"]["name"] and target["function"]["name"] != fn["name"]:
                    target["function"]["name"] += fn["name"]
                elif not target["function"]["name"]:
                    target["function"]["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                target["function"]["arguments"] += fn["arguments"]


def validate_usage(usage: Any) -> Dict[str, int]:
    if not isinstance(usage, dict):
        raise ReplayError("terminal stream did not include usage object")
    out: Dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ReplayError(f"usage.{key} must be finite number")
        if value < 0:
            raise ReplayError(f"usage.{key} must be nonnegative")
        if int(value) != value:
            raise ReplayError(f"usage.{key} must be integral")
        out[key] = int(value)
    if out["completion_tokens"] <= 0:
        raise ReplayError("usage.completion_tokens must be positive")
    return out


def stream_chat_completion(chat_url: str, body_bytes: bytes, timeout_s: int = CHAT_TIMEOUT_S) -> Dict[str, Any]:
    validate_loopback_url(chat_url)
    request = urllib.request.Request(chat_url, data=body_bytes, headers=auth_headers(), method="POST")
    wall_start = time.time()
    mono_start = time.monotonic()
    first_offset: Optional[float] = None
    answer_offset: Optional[float] = None
    terminal_seen = False
    usage: Any = None
    finish_reason: Optional[str] = None
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    event_data_lines: List[str] = []
    raw_lines: List[bytes] = []

    def response_snapshot():
        return {"content":"".join(content_parts), "reasoning":"".join(reasoning_parts),
                "tool_calls":tool_calls,
                "raw_sse_base64":base64.b64encode(b"".join(raw_lines)).decode("ascii")}

    def fail(message):
        exc = ReplayError(message)
        exc.partial_response = response_snapshot()
        return exc

    def process_event(data: str) -> None:
        nonlocal terminal_seen, usage, finish_reason, first_offset, answer_offset
        if data == "[DONE]":
            terminal_seen = True
            return
        try:
            event = json.loads(data)
        except Exception as exc:
            raise fail(f"invalid SSE JSON event: {exc}") from exc
        if event.get("usage") is not None:
            usage = event.get("usage")
        choices = event.get("choices", [])
        if choices is None:
            choices = []
        if not isinstance(choices, list):
            raise fail("SSE choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise fail("SSE choice must be an object")
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason"))
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise fail("SSE choice delta must be an object")
            content = delta.get("content")
            if isinstance(content, str):
                content_parts.append(content)
                if content and answer_offset is None:
                    answer_offset = time.monotonic() - mono_start
                if content and first_offset is None:
                    first_offset = answer_offset
            for key in ("reasoning_content", "reasoning"):
                reasoning = delta.get(key)
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                    if reasoning and first_offset is None:
                        first_offset = time.monotonic() - mono_start
            if delta.get("tool_calls"):
                if answer_offset is None:
                    answer_offset = time.monotonic() - mono_start
                if first_offset is None:
                    first_offset = answer_offset
                _merge_tool_call_delta(tool_calls, delta.get("tool_calls"))

    try:
        with open_loopback(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            if status < 200 or status >= 300:
                raise fail(f"chat HTTP status {status}")
            for raw_line in response:
                raw_lines.append(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise fail(f"SSE line is not UTF-8: {exc}") from exc
                line = line.rstrip("\r\n")
                if line == "":
                    if event_data_lines:
                        process_event("\n".join(event_data_lines))
                        event_data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event_data_lines.append(line[5:].lstrip())
                    continue
                # Allow standard SSE fields but reject arbitrary non-SSE text.
                if line.startswith(("event:", "id:", "retry:")):
                    continue
                raise fail(f"unexpected SSE line: {line[:80]}")
            if event_data_lines:
                process_event("\n".join(event_data_lines))
    except Exception as exc:
        raise fail(f"chat request failed: {exc}") from exc

    mono_end = time.monotonic()
    if not terminal_seen:
        raise fail("SSE stream ended without terminal [DONE] event")
    if first_offset is None:
        raise fail("SSE stream had no meaningful reasoning/content/tool delta")
    if finish_reason is None or finish_reason == "":
        raise fail("SSE stream had no finish_reason")
    try:
        valid_usage = validate_usage(usage)
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    return {
        "t_start": wall_start,
        "t_first": wall_start + first_offset,
        "t_end": wall_start + (mono_end - mono_start),
        "duration_s": mono_end - mono_start,
        "ttft_s": first_offset,
        "ttfa_s": answer_offset,
        "ttft_definition": "first nonempty reasoning/content/tool delta; ttfa excludes reasoning",
        "usage": valid_usage,
        "finish_reason": finish_reason,
        "response": response_snapshot(),
    }


def parse_metrics_text(text: str, required_metrics: Iterable[str] = REQUIRED_METRICS) -> Dict[str, float]:
    required = tuple(required_metrics)
    values: Dict[str, float] = {name: 0.0 for name in required}
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(stripped)
        if not match:
            continue
        name = match.group(1)
        if name not in values:
            continue
        raw_value = match.group(3)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ReplayError(f"metric {name} has nonnumeric value {raw_value!r}") from exc
        if not math.isfinite(value):
            raise ReplayError(f"metric {name} has nonfinite value {raw_value!r}")
        if value < 0:
            raise ReplayError(f"metric {name} has negative counter value {raw_value!r}")
        values[name] += value
        seen.add(name)
    missing = [name for name in required if name not in seen]
    if missing:
        raise ReplayError("missing required metrics: " + ", ".join(missing))
    return values


def fetch_metrics(metrics_url: str, timeout_s: int = METRICS_TIMEOUT_S) -> Dict[str, float]:
    validate_loopback_url(metrics_url)
    try:
        with open_loopback(metrics_url, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            if status < 200 or status >= 300:
                raise ReplayError(f"metrics HTTP status {status}")
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ReplayError(f"metrics request failed: {exc}") from exc
    return parse_metrics_text(body)


def metric_deltas(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for name in REQUIRED_METRICS:
        if name not in before or name not in after:
            raise ReplayError(f"metric {name} missing from before/after sample")
        delta = after[name] - before[name]
        if not math.isfinite(delta):
            raise ReplayError(f"metric {name} delta is nonfinite")
        if delta < 0:
            raise ReplayError(f"metric {name} counter decreased: before={before[name]} after={after[name]}")
        deltas[METRIC_ALIASES[name]] = delta
    return deltas


class JsonlWriter:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.lock = threading.Lock()
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()


def nearest_rank(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def quantile_summary(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": nearest_rank(values, 0.50),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
    }


def validate_turn_record_coverage(expected_turn_ids: Iterable[str], records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    expected = list(expected_turn_ids)
    expected_set = set(expected)
    seen_counts: Dict[str, int] = {}
    errors = 0
    unexpected: List[str] = []
    for record in records:
        turn_id = record.get("turn_id")
        if record.get("status") == "error":
            errors += 1
        if record.get("status") != "ok":
            continue
        if turn_id not in expected_set:
            unexpected.append(str(turn_id))
            continue
        seen_counts[str(turn_id)] = seen_counts.get(str(turn_id), 0) + 1
    duplicate = sorted([turn_id for turn_id, count in seen_counts.items() if count > 1])
    missing = [turn_id for turn_id in expected if seen_counts.get(turn_id, 0) == 0]
    return {
        "ok": not duplicate and not missing and not unexpected and errors == 0,
        "missing_turn_ids": missing,
        "duplicate_turn_ids": duplicate,
        "unexpected_turn_ids": sorted(unexpected),
        "errors": errors,
        "ok_turn_count": sum(1 for count in seen_counts.values() if count == 1),
    }


def common_record(config: ReplayConfig, session: Session, turn: Turn, worker_index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = canonical_json_bytes(payload)
    return {
        "tag": config.tag,
        "cache_state": config.cache_state,
        "worker_index": worker_index,
        "session_entry_index": session.entry_index,
        "session_id": session.identity,
        "session_entry_sha256": session.entry_sha256,
        "source_session_sha256": session.source_session_sha256,
        "turn_index": turn.index,
        "turn_id": turn.turn_id,
        "fixture_turn_has_tool": turn.has_tool,
        "request_body_sha256": sha256_hex(body),
        "request_shape_sha256": request_shape_digest(payload),
        "request_bytes": len(body),
        "model": config.model,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "cache_salt_sha256": sha256_hex(config.cache_salt.encode("utf-8")),
    }


def prepare_output_dir(output: pathlib.Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ReplayError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ReplayError(f"output directory must be new/empty: {output}")
    else:
        output.mkdir(parents=True)


def write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_summary(
    config: ReplayConfig,
    loaded: LoadedFixture,
    records: List[Dict[str, Any]],
    wall_s: float,
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, Any]:
    deltas = metric_deltas(metrics_before, metrics_after)
    ok_records = [r for r in records if r.get("status") == "ok"]
    completion_tokens = sum(int(r["usage"]["completion_tokens"]) for r in ok_records)
    prompt_tokens = sum(int(r["usage"]["prompt_tokens"]) for r in ok_records)
    duration_sum = sum(float(r["duration_s"]) for r in ok_records)
    aggregate = completion_tokens / wall_s if wall_s > 0 else 0.0
    busy = completion_tokens / (duration_sum / config.workers) if duration_sum > 0 and config.workers > 0 else 0.0

    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for record in ok_records:
        by_session.setdefault(record["session_id"], []).append(record)
    per_session = []
    for session in loaded.sessions:
        session_records = sorted(by_session.get(session.identity, []), key=lambda r: r["turn_index"])
        ttfts = [float(r["ttft_s"]) for r in session_records]
        if session_records:
            session_wall = max(float(r["t_end"]) for r in session_records) - min(float(r["t_start"]) for r in session_records)
        else:
            session_wall = None
        per_session.append(
            {
                "session_id": session.identity,
                "session_entry_index": session.entry_index,
                "turn_count": len(session_records),
                "wall_s": session_wall,
                "ttft_s": quantile_summary(ttfts),
            }
        )
    all_ttfts = [float(r["ttft_s"]) for r in ok_records]
    acceptance = None
    if deltas["spec_draft_tokens"] > 0:
        acceptance = deltas["spec_accepted_tokens"] / deltas["spec_draft_tokens"]
    accepted_per_step = None
    if deltas["spec_draft_steps"] > 0:
        accepted_per_step = deltas["spec_accepted_tokens"] / deltas["spec_draft_steps"]
    cache_token_hit_rate = None
    if deltas["prefix_cache_query_tokens"] > 0:
        cache_token_hit_rate = deltas["prefix_cache_hit_tokens"] / deltas["prefix_cache_query_tokens"]

    return {
        "tag": config.tag,
        "cache_state": config.cache_state,
        "fixture_sha256": loaded.fixture_sha256,
        "fixture_session_count": len(loaded.sessions),
        "fixture_turns_per_session": EXPECTED_TURNS_PER_SESSION,
        "turn_count": len(ok_records),
        "expected_turn_count": EXPECTED_TURN_COUNT,
        "model": config.model,
        "workers": config.workers,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "cache_salt_sha256": sha256_hex(config.cache_salt.encode("utf-8")),
        "request_config_hash": request_config_hash(config),
        "comparison_config_hash": comparison_config_hash(config),
        "request_static_config": request_public_static_config(config),
        "history_proxy_notice": "RECORDED-HISTORY PROXY: old tool results are user prose; simplified system prompt; not native Hermes; not production capacity evidence.",
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "aggregate_completion_tok_s": aggregate,
        "diagnostic_busy_worker_completion_tok_s": busy,
        "diagnostic_busy_worker_note": "diagnostic only: completion_tokens / (sum(request_durations) / workers); wall aggregate above uses measured run makespan",
        "quantile_method": "nearest-rank sorted[ceil(p*n)-1]",
        "ttft_s": quantile_summary(all_ttfts),
        "per_session": per_session,
        "metrics_delta": deltas,
        "acceptance": {
            "accepted_over_draft_tokens": acceptance,
            "accepted_tokens": deltas["spec_accepted_tokens"],
            "draft_tokens": deltas["spec_draft_tokens"],
            "accepted_per_step": accepted_per_step,
            "draft_steps": deltas["spec_draft_steps"],
        },
        "prefix_cache": {
            "units": "tokens (not request counts)",
            "query_tokens": deltas["prefix_cache_query_tokens"],
            "hit_tokens": deltas["prefix_cache_hit_tokens"],
            "token_hit_rate": cache_token_hit_rate,
        },
    }


def failure_summary(
    config: ReplayConfig,
    loaded: Optional[LoadedFixture],
    records: List[Dict[str, Any]],
    fatal_error: str,
    wall_s: Optional[float] = None,
) -> Dict[str, Any]:
    expected = loaded.expected_turn_ids if loaded is not None else ()
    coverage = validate_turn_record_coverage(expected, records) if loaded is not None else {}
    return {
        "tag": config.tag,
        "cache_state": config.cache_state,
        "cache_salt_sha256": sha256_hex(config.cache_salt.encode("utf-8")),
        "fatal_error": fatal_error,
        "fixture_sha256": loaded.fixture_sha256 if loaded is not None else None,
        "expected_turn_count": EXPECTED_TURN_COUNT,
        "record_count": len(records),
        "ok_turn_count": coverage.get("ok_turn_count", 0),
        "errors": coverage.get("errors", sum(1 for r in records if r.get("status") == "error")),
        "missing_turn_count": len(coverage.get("missing_turn_ids", [])),
        "duplicate_turn_ids": coverage.get("duplicate_turn_ids", []),
        "unexpected_turn_ids": coverage.get("unexpected_turn_ids", []),
        "wall_s": wall_s,
    }


def run_replay(config: ReplayConfig) -> int:
    prepare_output_dir(config.output)
    records_path = config.output / "turn_records.jsonl"
    records: List[Dict[str, Any]] = []
    loaded: Optional[LoadedFixture] = None
    run_wall_s: Optional[float] = None
    writer = JsonlWriter(records_path)
    try:
        loaded = load_fixture_path(config.fixture)
        chat_url, metrics_url = normalize_urls(config.base_url)
        metrics_before = fetch_metrics(metrics_url)
        sessions_queue: "queue.Queue[Session]" = queue.Queue()
        for session in loaded.sessions:
            sessions_queue.put(session)
        records_lock = threading.Lock()
        failure_event = threading.Event()
        first_error_lock = threading.Lock()
        first_error: List[str] = []

        def record_append(record: Dict[str, Any]) -> None:
            writer.append(record)
            with records_lock:
                records.append(record)

        def worker(worker_index: int) -> None:
            while not failure_event.is_set():
                try:
                    session = sessions_queue.get_nowait()
                except queue.Empty:
                    return
                for turn in session.turns:
                    if failure_event.is_set():
                        return
                    payload = build_request_payload(config, turn)
                    base_record = common_record(config, session, turn, worker_index, payload)
                    body = canonical_json_bytes(payload)
                    try:
                        result = stream_chat_completion(chat_url, body)
                        record = dict(base_record)
                        record.update(
                            {
                                "status": "ok",
                                "t_start": result["t_start"],
                                "t_first": result["t_first"],
                                "t_end": result["t_end"],
                                "duration_s": result["duration_s"],
                                "ttft_s": result["ttft_s"],
                                "ttfa_s": result["ttfa_s"],
                                "usage": result["usage"],
                                "finish_reason": result["finish_reason"],
                                "response": result["response"],
                            }
                        )
                        record_append(record)
                    except Exception as exc:  # Preserve partial evidence and stop cleanly.
                        record = dict(base_record)
                        record.update(
                            {
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc(limit=5),
                                "t_error": time.time(),
                                "partial_response": getattr(exc, "partial_response", None),
                            }
                        )
                        record_append(record)
                        with first_error_lock:
                            if not first_error:
                                first_error.append(str(exc))
                        failure_event.set()
                        return

        mono_start = time.monotonic()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(config.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        run_wall_s = time.monotonic() - mono_start

        metrics_after = fetch_metrics(metrics_url)
        coverage = validate_turn_record_coverage(loaded.expected_turn_ids, records)
        if failure_event.is_set() or not coverage["ok"]:
            reason = first_error[0] if first_error else "turn coverage validation failed"
            write_json(config.output / "failure_summary.json", failure_summary(config, loaded, records, reason, run_wall_s))
            return 1
        summary = build_summary(config, loaded, records, run_wall_s, metrics_before, metrics_after)
        summary["records_path"] = str(records_path)
        write_json(config.output / "summary.json", summary)
        print(
            f"{config.tag}: ok {summary['turn_count']} turns, "
            f"{summary['completion_tokens']} completion tokens, "
            f"{summary['aggregate_completion_tok_s']:.3f} tok/s wall; summary={config.output / 'summary.json'}",
            flush=True,
        )
        return 0
    except Exception as exc:
        write_json(config.output / "failure_summary.json", failure_summary(config, loaded, records, str(exc), run_wall_s))
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        writer.close()


CACHE_STATE_ALIASES = {
    "initial": "cache_namespace_initial",
    "repeat": "identical_request_repeat",
    "cache_namespace_initial": "cache_namespace_initial",
    "identical_request_repeat": "identical_request_repeat",
}


def parse_args(argv: Optional[List[str]] = None) -> ReplayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--cache-state", choices=tuple(CACHE_STATE_ALIASES), required=True)
    parser.add_argument("--cache-salt", required=True, help="vLLM cache_salt namespace, 1..1024 bytes/chars")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ReplayError("--workers must be positive")
    if args.max_tokens <= 0:
        raise ReplayError("--max-tokens must be positive")
    if not (1 <= len(args.cache_salt) <= 1024):
        raise ReplayError("--cache-salt must be 1..1024 characters")
    return ReplayConfig(
        fixture=pathlib.Path(args.fixture),
        base_url=args.base_url,
        model=args.model,
        output=pathlib.Path(args.output),
        tag=args.tag,
        workers=args.workers,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        cache_state=CACHE_STATE_ALIASES[args.cache_state],
        cache_salt=args.cache_salt,
    )


def main(argv: Optional[List[str]] = None) -> int:
    try:
        config = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except ReplayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return run_replay(config)


if __name__ == "__main__":
    raise SystemExit(main())
