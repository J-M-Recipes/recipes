#!/usr/bin/env python3
"""Measure DFlash acceptance length and decode speed on prose/code.

Acceptance length follows the draft model card definition: completion tokens divided
by verification steps. vLLM exposes verification steps as the delta of
spec_decode_num_drafts_total for these one-request-at-a-time probes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

PROMPTS = [
    ("prose", "Write a detailed 700-word explanation of how Roman aqueducts maintained gradient and water quality."),
    ("prose", "Explain speculative decoding to an inference engineer, including verification, acceptance length, and failure modes."),
    ("code", "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings, and tests. Output only code."),
    ("code", "Write a complete Rust implementation of a bounded work queue with cancellation and unit tests. Output only code."),
]
METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def get_metrics(base_url: str) -> dict[str, float]:
    text = urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=20).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name, metric in METRICS.items():
            if line.startswith(metric + "{"):
                out[name] = float(line.rsplit(" ", 1)[1])
    missing = sorted(set(METRICS) - set(out))
    if missing:
        raise RuntimeError(f"missing metrics: {missing}")
    return out


def run_one(base_url: str, model: str, key: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    before = get_metrics(base_url)
    started = time.monotonic()
    first = None
    usage = None
    finish = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            datum = json.loads(line[5:])
            if datum.get("usage"):
                usage = datum["usage"]
            choices = datum.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.monotonic()
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    ended = time.monotonic()
    after = get_metrics(base_url)
    md = {name: after[name] - before[name] for name in METRICS}
    completion = int((usage or {}).get("completion_tokens") or 0)
    decode_seconds = ended - (first or started)
    return {
        "wall_seconds": round(ended - started, 3),
        "ttft_seconds": round((first or ended) - started, 3),
        "decode_seconds": round(decode_seconds, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion,
        "decode_tok_s": round(completion / decode_seconds, 3) if completion and decode_seconds else None,
        "finish_reason": finish,
        "metric_delta": md,
        "acceptance_length": completion / md["drafts"] if md["drafts"] else None,
        "accepted_over_drafted": md["accepted_tokens"] / md["draft_tokens"] if md["draft_tokens"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:30001"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "glm-5.3-big"))
    parser.add_argument("--api-key-file", default=os.environ.get("API_KEY_FILE", str(Path.home() / ".glm_api_key")))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    key = Path(args.api_key_file).read_text().strip()
    rows = []
    for index, (kind, prompt) in enumerate(PROMPTS):
        row = run_one(args.base_url, args.model, key, f"[dflash2-probe-{index}-{time.time_ns()}] {prompt}", args.max_tokens)
        row.update({"index": index, "kind": kind})
        rows.append(row)
        print(json.dumps(row), flush=True)
    lengths = [row["acceptance_length"] for row in rows if row["acceptance_length"] is not None]
    speeds = [row["decode_tok_s"] for row in rows if row["decode_tok_s"] is not None]
    summary = {
        "schema": "glm53-dflash2-uva-acceptance-v1",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "rows": rows,
        "summary": {
            "requests": len(rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "verification_steps": sum(row["metric_delta"]["drafts"] for row in rows),
            "acceptance_length_weighted": (
                sum(row["completion_tokens"] for row in rows) / sum(row["metric_delta"]["drafts"] for row in rows)
                if sum(row["metric_delta"]["drafts"] for row in rows)
                else None
            ),
            "acceptance_length_median": statistics.median(lengths),
            "decode_tok_s_median": statistics.median(speeds),
            "stop_gate_pass": (
                sum(row["completion_tokens"] for row in rows) / sum(row["metric_delta"]["drafts"] for row in rows)
            ) >= 3.0,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
