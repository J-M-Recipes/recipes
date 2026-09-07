"""Secondary supplied-data structured-output V2 validation.

Versioned separately from the frozen V1 runner.  Uses the original V1
fixtures/settings and changes only the chat-completions request by adding
OpenAI response_format json_schema for grounded/structured fixtures.
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request

SETTINGS = {
    "temperature": 0,
    "top_p": 1,
    "seed": 20260906,
    "max_tokens": 4096,
    "stream": False,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
FORBIDDEN_VISIBLE = ["<think>", "</think>", "<tool_call>", "</tool_call>"]


def fixtures():
    """Exact original V1 supplied-data fixtures from secondary_gates.py."""
    out = []
    rng = random.Random(20260906)
    for i in range(10):
        records = [
            {
                "id": f"item-{i}-{j}",
                "amount": rng.randrange(10, 500),
                "status": rng.choice(["open", "closed"]),
                "region": rng.choice(["east", "west"]),
            }
            for j in range(35)
        ]
        chosen = [r for r in records if r["status"] == "open" and r["region"] == "east"]
        want = {
            "count": len(chosen),
            "total": sum(r["amount"] for r in chosen),
            "ids": sorted(r["id"] for r in chosen),
        }
        prompt = (
            "From ONLY these supplied records, select status=open AND region=east. "
            "Return one JSON object with count, total (sum of amount), and ids sorted "
            "lexicographically. No extra keys or prose.\n" + json.dumps(records)
        )
        out.append({"id": f"grounded-{i}", "category": "grounded", "prompt": prompt, "want": want})
    for i in range(10):
        rows = [
            {"name": f"k{j}", "priority": rng.randrange(1, 6), "enabled": bool(rng.randrange(2))}
            for j in range(12)
        ]
        want = [r["name"] for r in sorted(rows, key=lambda x: (-x["priority"], x["name"])) if r["enabled"]]
        out.append(
            {
                "id": f"structure-{i}",
                "category": "structured",
                "prompt": "Filter enabled=true, sort by priority descending then name lexicographically ascending. "
                "Return ONLY a JSON array of names. Data: " + json.dumps(rows),
                "want": want,
            }
        )
    return out


def schema_for(item_or_category):
    category = item_or_category if isinstance(item_or_category, str) else item_or_category["category"]
    if category == "grounded":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "total": {"type": "integer"},
                        "ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["count", "total", "ids"],
                    "additionalProperties": False,
                },
            },
        }
    if category == "structured":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "enabled_names",
                "schema": {"type": "array", "items": {"type": "string"}},
            },
        }
    raise ValueError(category)


def request_body(model, item):
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}], **SETTINGS}
    body["response_format"] = schema_for(item)
    return body


def request(base, key, model, item):
    body = request_body(model, item)
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as r:
        raw_body = r.read()
    wall = time.monotonic() - t
    return json.loads(raw_body.decode("utf-8")), wall, body


def protocol(response):
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return False, None, "missing choices[0].message"
    if choice.get("finish_reason") != "stop":
        return False, message, "finish_reason != stop"
    if not isinstance(message.get("content"), str):
        return False, message, "message.content not string"
    if message.get("tool_calls"):
        return False, message, "tool_calls present"
    visible = message.get("content") or ""
    if any(tag in visible for tag in FORBIDDEN_VISIBLE):
        return False, message, "visible forbidden tag"
    return True, message, None


def parse_full_string(content):
    # json.loads on the whole string rejects trailing bytes and leading junk.
    return json.loads(content)


def schema_validate(category, value):
    if category == "grounded":
        if not isinstance(value, dict):
            return False, "not object"
        if set(value.keys()) != {"count", "total", "ids"}:
            return False, "keys mismatch"
        if not isinstance(value["count"], int) or isinstance(value["count"], bool):
            return False, "count not integer"
        if not isinstance(value["total"], int) or isinstance(value["total"], bool):
            return False, "total not integer"
        if not isinstance(value["ids"], list) or not all(isinstance(x, str) for x in value["ids"]):
            return False, "ids not string array"
        return True, None
    if category == "structured":
        if not isinstance(value, list):
            return False, "not array"
        if not all(isinstance(x, str) for x in value):
            return False, "array item not string"
        return True, None
    return False, "unknown category"


def score_response(item, response, wall_s, req_body=None, error=None):
    row = {
        "fixture": item["id"],
        "category": item["category"],
        "fixture_sha256": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
        "request": req_body,
        "response": response,
        "wall_s": wall_s,
        "protocol_ok": False,
        "parse_ok": False,
        "schema_ok": False,
        "correctness_ok": False,
        "parsed_answer": None,
        "parse_error": None,
        "schema_error": None,
        "protocol_error": None,
        "failure_class": None,
        "passed": False,
    }
    if error is not None:
        row.update(error_type=type(error).__name__, error=str(error), failure_class="transport_fail")
        return row
    ok, message, protocol_error = protocol(response)
    row["protocol_ok"] = ok
    row["protocol_error"] = protocol_error
    if not ok:
        row["failure_class"] = "protocol_fail"
        return row
    try:
        answer = parse_full_string(message["content"])
        row["parse_ok"] = True
        row["parsed_answer"] = answer
    except Exception as exc:
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
        row["failure_class"] = "format_fail"
        return row
    schema_ok, schema_error = schema_validate(item["category"], answer)
    row["schema_ok"] = schema_ok
    row["schema_error"] = schema_error
    if not schema_ok:
        row["failure_class"] = "schema_fail"
        return row
    row["correctness_ok"] = answer == item["want"]
    if not row["correctness_ok"]:
        row["failure_class"] = "oracle_fail"
        return row
    row["failure_class"] = "pass"
    row["passed"] = True
    return row


def run(args):
    key = pathlib.Path(args.key_file).read_text().strip()
    if not key:
        raise SystemExit("empty key file")
    target = pathlib.Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        for repeat in range(args.repeats):
            for item in fixtures():
                try:
                    response, wall_s, body = request(args.base_url, key, args.model, item)
                    row = score_response(item, response, wall_s, body)
                except Exception as exc:
                    row = score_response(item, None, None, request_body(args.model, item), exc)
                row.update(lane=args.lane, repeat=repeat, runner_version="secondary_gates_v2")
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                f.flush()
                print(json.dumps({k: row[k] for k in ["lane", "repeat", "fixture", "failure_class", "passed"]}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--model", default="glm-5.3-big")
    p.add_argument("--lane", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args(argv)
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
