#!/usr/bin/env python3
"""Diag matrix: E1 cells + GB300 probe prompts. Spec metrics optional (AR boot).

Acceptance counters match dflash2_acceptance_probe.py:
  vllm:spec_decode_num_drafts_total
  vllm:spec_decode_num_draft_tokens_total
  vllm:spec_decode_num_accepted_tokens_total
Acceptance length = completion_tokens_total / drafts_delta.
Accept rate = accepted_tokens_delta / draft_tokens_delta.
tok/s per stream = usage.completion_tokens / wall.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

# GB300 probe prompts, verbatim from dflash2_acceptance_probe.py (512 tokens, C1).
PROBE_PROMPTS = [
    ("probe_prose0", "Write a detailed 700-word explanation of how Roman aqueducts maintained gradient and water quality."),
    ("probe_prose1", "Explain speculative decoding to an inference engineer, including verification, acceptance length, and failure modes."),
    ("probe_code0", "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings, and tests. Output only code."),
    ("probe_code1", "Write a complete Rust implementation of a bounded work queue with cancellation and unit tests. Output only code."),
]

PROSE = "Write a 300-word essay on how tidal locking shapes planetary climates."
STRUCTURED = "Count from 1 to 300 as a comma-separated list, nothing else."
AGENT = "Use the read_file tool to read /etc/hosts and then summarize what you found."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the local filesystem and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute filesystem path to read",
                    }
                },
                "required": ["path"],
            },
        },
    }
]


def date_is() -> str:
    return subprocess.check_output(["date", "-Is"], text=True).strip()


def mem_available() -> dict[str, int]:
    # P3 GB300 single-GPU lane: there is no rank1 worker (the Spark version ssh'd milo@<rdma-peer-2>).
    # Record host MemAvailable and GB300 (GPU index 1) HBM used instead.
    head = int(
        next(
            line.split()[1]
            for line in open("/proc/meminfo")
            if line.startswith("MemAvailable:")
        )
    )
    hbm = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "1"],
            text=True,
            timeout=20,
        ).strip()
    )
    return {"head_kB": head, "gpu1_hbm_used_MiB": hbm}


def get_metrics(base_url: str) -> tuple[dict[str, float], list[str], dict[str, int]]:
    text = urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=30).read().decode()
    totals = {name: 0.0 for name in METRICS}
    last = {}
    seen = {name: 0 for name in METRICS}
    raw_lines: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name, metric in METRICS.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                value = float(line.rsplit(" ", 1)[1])
                totals[name] += value
                last[name] = value
                seen[name] += 1
                raw_lines.append(line)
    missing = sorted(name for name, count in seen.items() if count == 0)
    if missing:
        # AR boot: no speculative series. Record absence, do not invent zeros as deltas.
        totals["_missing"] = missing
    per_pos: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith(PER_POS + "{"):
            labels = line[len(PER_POS) + 1 : line.index("}")]
            pos = None
            for part in labels.split(","):
                k, _, v = part.partition("=")
                if k.strip() == "position":
                    pos = v.strip().strip('"')
            if pos is not None:
                per_pos[pos] = per_pos.get(pos, 0.0) + float(line.rsplit(" ", 1)[1])
    totals["_per_pos"] = per_pos
    return totals, raw_lines, seen


def post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, Any, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        parsed: Any
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body[:4000]}
        return exc.code, parsed, body


def one_request(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    started_iso = date_is()
    status, parsed, raw = post_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    ended = time.monotonic()
    wall = ended - started
    usage = (parsed or {}).get("usage") if isinstance(parsed, dict) else None
    completion = int((usage or {}).get("completion_tokens") or 0)
    choice = None
    message = {}
    if isinstance(parsed, dict):
        choices = parsed.get("choices") or []
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
    tool_calls = message.get("tool_calls")
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if tool_calls:
        response_kind = "tool_call"
    elif content:
        response_kind = "text"
    else:
        response_kind = "empty"
    return {
        "http_status": status,
        "started_iso": started_iso,
        "wall_seconds": round(wall, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion,
        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "tok_s": round(completion / wall, 3) if completion and wall else None,
        "finish_reason": (choice or {}).get("finish_reason") if choice else None,
        "response_kind": response_kind,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "error": None if status == 200 else parsed,
    }


def run_cell(
    base_url: str,
    model: str,
    fixture: str,
    prompt: str,
    concurrency: int,
    max_tokens: int,
    timeout: int,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    before, before_lines, before_seen = get_metrics(base_url)
    mem_box: dict[str, Any] = {}

    def sample_mem() -> None:
        time.sleep(5)
        try:
            mem_box["during"] = mem_available()
            mem_box["during_iso"] = date_is()
        except Exception as exc:  # noqa: BLE001 — record, do not invent
            mem_box["during_error"] = repr(exc)

    sampler = None
    if concurrency >= 8:
        sampler = threading.Thread(target=sample_mem, daemon=True)
        sampler.start()
    t0 = time.monotonic()
    t0_iso = date_is()
    rows: list[dict[str, Any] | None] = [None] * concurrency
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(one_request, base_url, payload, timeout): index
            for index in range(concurrency)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                rows[index] = {"error": repr(exc), "completion_tokens": 0, "tok_s": None, "wall_seconds": None}
    t1 = time.monotonic()
    t1_iso = date_is()
    if sampler is not None:
        sampler.join(timeout=30)
    time.sleep(1.0)
    after, after_lines, after_seen = get_metrics(base_url)
    spec_present = not after.get("_missing")
    delta = {name: (after[name] - before[name]) if spec_present else None for name in METRICS}
    per_pos_delta = {
        pos: after["_per_pos"].get(pos, 0.0) - before["_per_pos"].get(pos, 0.0)
        for pos in sorted(set(after["_per_pos"]) | set(before["_per_pos"]), key=int)
    }
    ok = [row for row in rows if row and row.get("http_status") == 200]
    completion_total = sum(int(row.get("completion_tokens") or 0) for row in rows if row)
    walls = [float(row["wall_seconds"]) for row in ok if row.get("wall_seconds") is not None]
    batch_wall = t1 - t0
    drafts = delta["drafts"] if spec_present else None
    draft_tokens = delta["draft_tokens"] if spec_present else None
    cell = {
        "fixture": fixture,
        "concurrency": concurrency,
        "started_iso": t0_iso,
        "ended_iso": t1_iso,
        "batch_wall_seconds": round(batch_wall, 3),
        "mean_wall_seconds": round(statistics.mean(walls), 3) if walls else None,
        "completion_tokens_total": completion_total,
        "aggregate_tok_s": round(completion_total / batch_wall, 3) if completion_total and batch_wall else None,
        "per_stream_tok_s": [row.get("tok_s") if row else None for row in rows],
        "spec_metrics_present": spec_present,
        "metric_before": {k: v for k, v in before.items() if not k.startswith("_")},
        "metric_after": {k: v for k, v in after.items() if not k.startswith("_")},
        "metric_delta": delta,
        "per_pos_accepted_delta": per_pos_delta,
        "per_pos_accept_prob": (
            {pos: (v / drafts) for pos, v in per_pos_delta.items()} if drafts else None
        ),
        "metric_series_seen_before": before_seen,
        "metric_series_seen_after": after_seen,
        "metric_lines_before": before_lines,
        "metric_lines_after": after_lines,
        "acceptance_length": (completion_total / drafts) if drafts else None,
        "accept_rate": (delta["accepted_tokens"] / draft_tokens) if draft_tokens else None,
        "ar_tok_s_from_wall": round(completion_total / batch_wall, 3) if (not spec_present and completion_total and batch_wall) else None,
        "response_kinds": [row.get("response_kind") if row else None for row in rows],
        "finish_reasons": [row.get("finish_reason") if row else None for row in rows],
        "errors": [row.get("error") for row in rows if row and row.get("error")],
        "mem_during": mem_box,
        "requests": rows,
        "request_payload_template": payload,
    }
    return cell


def smoke(base_url: str, model: str) -> dict[str, Any]:
    models_raw = urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=30).read().decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one sentence: what is a DGX Spark?"}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 64,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    row = one_request(base_url, payload, timeout=300)
    return {"models": json.loads(models_raw), "completion": row, "iso": date_is()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--mode", required=True, choices=["ar", "dflash"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"MATRIX_BEGIN mode={args.mode} {date_is()}", flush=True)
    print("MEM_AT_MATRIX_START " + json.dumps(mem_available()), flush=True)
    smoke_row = smoke(args.base_url, args.model)
    (out / "smoke.json").write_text(json.dumps(smoke_row, indent=2) + "\n")
    text = ((smoke_row["completion"].get("content") or "") if smoke_row["completion"] else "")
    print("SMOKE_TEXT " + json.dumps(text), flush=True)
    print("SMOKE_KIND " + str(smoke_row["completion"].get("response_kind")), flush=True)

    plan = [
        ("prose", PROSE, 1, None, 3),
        ("prose", PROSE, 4, None, 1),
        ("prose", PROSE, 8, None, 1),
        ("structured", STRUCTURED, 1, None, 1),
        ("structured", STRUCTURED, 4, None, 1),
        ("structured", STRUCTURED, 8, None, 1),
        ("agent", AGENT, 1, TOOLS, 1),
        ("agent", AGENT, 4, TOOLS, 1),
        ("agent", AGENT, 8, TOOLS, 1),
    ]
    probe_plan = [(name, prompt) for name, prompt in PROBE_PROMPTS]
    summaries = []
    for fixture, prompt, concurrency, tools, reps in plan:
        for rep in range(1, reps + 1):
            suffix = f"_r{rep}" if reps > 1 else ""
            name = f"{fixture}_c{concurrency}{suffix}"
            print(f"CELL_BEGIN {name} {date_is()}", flush=True)
            cell = run_cell(
                args.base_url,
                args.model,
                fixture,
                prompt,
                concurrency,
                300,
                args.timeout,
                tools,
            )
            cell["name"] = name
            cell["rep"] = rep
            cell["mode"] = args.mode
            (out / f"{name}.json").write_text(json.dumps(cell, indent=2) + "\n")
            brief = {
                "name": name,
                "acceptance_length": cell["acceptance_length"],
                "accept_rate": cell["accept_rate"],
                "aggregate_tok_s": cell["aggregate_tok_s"],
                "mean_wall_seconds": cell["mean_wall_seconds"],
                "completion_tokens_total": cell["completion_tokens_total"],
                "per_stream_tok_s": cell["per_stream_tok_s"],
                "metric_delta": cell["metric_delta"],
                "per_pos_accepted_delta": cell["per_pos_accepted_delta"],
                "response_kinds": cell["response_kinds"],
                "errors": cell["errors"],
                "mem_during": cell["mem_during"],
            }
            summaries.append(brief)
            print("CELL_DONE " + json.dumps(brief), flush=True)
    for name, prompt in probe_plan:
        print(f"CELL_BEGIN {name} {date_is()}", flush=True)
        cell = run_cell(args.base_url, args.model, name, prompt, 1, 512, args.timeout, None)
        cell["name"] = name
        cell["rep"] = 1
        cell["mode"] = args.mode
        (out / f"{name}.json").write_text(json.dumps(cell, indent=2) + "\n")
        brief = {
            "name": name,
            "acceptance_length": cell["acceptance_length"],
            "accept_rate": cell["accept_rate"],
            "aggregate_tok_s": cell["aggregate_tok_s"],
            "mean_wall_seconds": cell["mean_wall_seconds"],
            "completion_tokens_total": cell["completion_tokens_total"],
            "per_stream_tok_s": cell["per_stream_tok_s"],
            "metric_delta": cell["metric_delta"],
            "per_pos_accepted_delta": cell["per_pos_accepted_delta"],
            "response_kinds": cell["response_kinds"],
            "finish_reasons": cell["finish_reasons"],
            "errors": cell["errors"],
            "mem_during": cell["mem_during"],
        }
        summaries.append(brief)
        print("CELL_DONE " + json.dumps(brief), flush=True)
    prose_c1 = [row for row in summaries if row["name"].startswith("prose_c1_")]
    toks = [row["aggregate_tok_s"] for row in prose_c1 if row["aggregate_tok_s"] is not None]
    acc = [row["acceptance_length"] for row in prose_c1 if row["acceptance_length"] is not None]
    summary = {
        "mode": args.mode,
        "ended_iso": date_is(),
        "mem_at_end": mem_available(),
        "c1_prose_tok_s_reps": toks,
        "c1_prose_tok_s_median": statistics.median(toks) if toks else None,
        "c1_prose_accept_len_reps": acc,
        "c1_prose_accept_len_median": statistics.median(acc) if acc else None,
        "cells": summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MATRIX_DONE " + json.dumps({k: summary[k] for k in summary if k != "cells"}), flush=True)


if __name__ == "__main__":
    main()
