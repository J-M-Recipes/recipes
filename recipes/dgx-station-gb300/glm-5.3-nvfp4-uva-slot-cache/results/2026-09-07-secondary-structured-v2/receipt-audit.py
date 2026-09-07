#!/usr/bin/env python3
"""Audit the public V2 secondary structured-output receipts.

This script reads only files in this directory. It verifies receipt shape,
request pairing, fixture identity, structural JSON-schema usage, and a small
public-safety marker scan. It does not contact a model or regenerate results.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent
LANES = {
    "v1": "v1-secondary-structured-v2.jsonl",
    "sc13g-nomtp": "sc13g-nomtp-v2.jsonl",
    "sc13g-mtp": "sc13g-mtp-v2.jsonl",
}
EXPECTED = {
    "v1": {"rows": 40, "schema_ok": 40, "correctness_ok": 37},
    "sc13g-nomtp": {"rows": 40, "schema_ok": 40, "correctness_ok": 38},
    "sc13g-mtp": {"rows": 40, "schema_ok": 40, "correctness_ok": 37},
}
SECRET_OR_PRIVATE_MARKERS = [
    "glm" + "_api_key",
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "BEGIN " + "RSA PRIVATE KEY",
]
AUTH_HEADER_LITERALS = ["Author" + "ization", "Bearer" + " "]
PRIVATE_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
ORACLE_MARKER_RE = re.compile(r"item-\d+-\d+|\bk(?:[1-9]|1[0-2])\b|\b(?:902|1002)\b")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(name: str):
    path = ROOT / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def response_format_is_structural(row: dict) -> bool:
    rf = (row.get("request") or {}).get("response_format")
    if not isinstance(rf, dict):
        return False
    text = json.dumps(rf, sort_keys=True)
    # Schema may include structural field names like total/ids/count, but must
    # not include fixture data or answer values.
    return ORACLE_MARKER_RE.search(text) is None


def main() -> int:
    report = {
        "ok": True,
        "package_dir": ".",
        "lanes": {},
        "pairing": {},
        "privacy_scan": {},
        "hashes": {},
    }
    errors = []
    rows_by_lane = {}

    for lane, filename in LANES.items():
        path = ROOT / filename
        rows = read_jsonl(filename)
        rows_by_lane[lane] = rows
        keyset = {(r.get("fixture"), r.get("repeat")) for r in rows}
        lane_report = {
            "file": filename,
            "sha256": sha256(path),
            "rows": len(rows),
            "unique_fixture_repeat_pairs": len(keyset),
            "unique_tasks": len({r.get("fixture") for r in rows}),
            "protocol_ok": sum(bool(r.get("protocol_ok")) for r in rows),
            "parse_ok": sum(bool(r.get("parse_ok")) for r in rows),
            "schema_ok": sum(bool(r.get("schema_ok")) for r in rows),
            "correctness_ok": sum(bool(r.get("correctness_ok")) for r in rows),
            "failure_classes": dict(Counter(r.get("failure_class") for r in rows)),
            "response_format_oracle_leak_count": sum(not response_format_is_structural(r) for r in rows),
        }
        report["lanes"][lane] = lane_report
        exp = EXPECTED[lane]
        for field, value in exp.items():
            if lane_report[field] != value:
                errors.append(f"{lane}: expected {field}={value}, got {lane_report[field]}")
        if lane_report["protocol_ok"] != 40 or lane_report["parse_ok"] != 40:
            errors.append(f"{lane}: protocol/parse count not 40/40")
        if lane_report["unique_fixture_repeat_pairs"] != 40 or lane_report["unique_tasks"] != 20:
            errors.append(f"{lane}: fixture pairing cardinality mismatch")
        if lane_report["response_format_oracle_leak_count"]:
            errors.append(f"{lane}: response_format contains fixture/oracle-like values")

    for left, right in [("v1", "sc13g-nomtp"), ("sc13g-nomtp", "sc13g-mtp"), ("v1", "sc13g-mtp")]:
        L = {(r["fixture"], r["repeat"]): r for r in rows_by_lane[left]}
        R = {(r["fixture"], r["repeat"]): r for r in rows_by_lane[right]}
        common = set(L) & set(R)
        sha_mismatch = [k for k in common if L[k].get("fixture_sha256") != R[k].get("fixture_sha256")]
        request_mismatch = []
        for k in common:
            lreq = dict(L[k].get("request") or {})
            rreq = dict(R[k].get("request") or {})
            lreq.pop("model", None)
            rreq.pop("model", None)
            if lreq != rreq:
                request_mismatch.append(k)
        pair_report = {
            "common_pairs": len(common),
            "fixture_sha256_mismatches": len(sha_mismatch),
            "request_mismatches_ignoring_model": len(request_mismatch),
        }
        report["pairing"][f"{left}_vs_{right}"] = pair_report
        if pair_report != {"common_pairs": 40, "fixture_sha256_mismatches": 0, "request_mismatches_ignoring_model": 0}:
            errors.append(f"pairing mismatch: {left} vs {right}: {pair_report}")

    paired = json.loads((ROOT / "paired-summary.json").read_text())
    report["paired_summary"] = paired
    expected_paired = {
        "unique_tasks": 20,
        "pairs": 40,
        "task_comparison": {"tie": 18, "mtp_win": 1, "mtp_loss": 1},
        "no_mtp_correct": 38,
        "mtp_correct": 37,
    }
    if paired != expected_paired:
        errors.append(f"paired-summary mismatch: {paired}")

    manifest = json.loads((ROOT / "public-evidence-manifest.json").read_text())
    for filename, meta in manifest["files"].items():
        # receipt-audit.json is regenerated by this script, so its hash is a
        # post-run receipt value maintained by the manifest, not an input
        # invariant the script can check while producing that same file.
        if filename == "receipt-audit.json":
            continue
        path = ROOT / filename
        if path.exists():
            report["hashes"][filename] = sha256(path)
            if meta.get("published_sha256") != report["hashes"][filename]:
                errors.append(f"manifest published_sha256 mismatch for {filename}")

    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.suffix in {".jsonl", ".json", ".md", ".py"}:
            text = path.read_text(errors="ignore")
            hits = [m for m in SECRET_OR_PRIVATE_MARKERS if m in text]
            private_paths = sorted(set(PRIVATE_PATH_RE.findall(text)))
            auth_literals = [m for m in AUTH_HEADER_LITERALS if m in text]
            report["privacy_scan"][path.name] = {
                "secret_marker_hits": hits,
                "private_path_hits": private_paths,
                "auth_header_literal_hits": auth_literals,
            }
            if hits or private_paths:
                errors.append(f"private/secret markers in {path.name}: markers={hits} private_paths={private_paths}")

    report["errors"] = errors
    report["ok"] = not errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
