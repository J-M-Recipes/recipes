#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re

evidence = pathlib.Path(__file__).resolve().parent
recipe = evidence.parents[1]
result = json.loads((evidence / "offline-sim-v1.json").read_text())
manifest = json.loads((evidence / "input-manifest.json").read_text())
live = json.loads((evidence / "live-service-proof.json").read_text())

assert result["schema"] == "glm53-slotcache-offline-sim-v1"
assert result["input"] == {
    "decode_tokens": 71210,
    "experts": 256,
    "layers": 78,
    "prefill_tokens": 7909,
    "steps": 48363,
    "tokens": 79119,
    "topk": 8,
}
cache = result["cache_policies"]["gate"]
assert cache["pass"] is False
assert cache["best_candidate"] == "optimized:lru"
assert cache["baseline_hit_rate"] == 0.7097356268039628
assert cache["best_hit_rate"] == 0.7099777673765504
assert cache["gain_points"] == 0.024214057258753474
assert cache["agent_segment"] == "tools_low"
assert cache["agent_gain_points"] == 0.014184397163119478
held_out_tokens = sum(segment["held_out_tokens"] for segment in result["segments"].values())
expected_accesses = held_out_tokens * 75 * 8
assert expected_accesses == 12819000
for policy in result["cache_policies"]["frozen_allocation"].values():
    assert policy["aggregate"]["accesses"] == expected_accesses
for policy in result["cache_policies"]["optimized_allocation"].values():
    allocation = {int(layer): int(slots) for layer, slots in policy["allocation"].items()}
    assert sum(allocation.values()) == 5792
    assert all(32 <= slots <= 256 and slots % 16 == 0 for slots in allocation.values())
    assert policy["aggregate"]["accesses"] == expected_accesses

prefetch = result["prefetch"]
gate = prefetch["gate"]
assert gate == {
    "best_limit": 1,
    "best_strategy": "adjacent_layer",
    "pass": False,
    "precision": 0.08408456197831345,
    "recall": 0.036721409234329934,
    "threshold_precision": 0.5,
}
assert prefetch["adjacent_layer"]["training_scope"] == "that workload segment's chronological training prefix only"
assert prefetch["adjacent_layer"]["evaluation_scope"] == "all held-out decode rows"
assert prefetch["previous_token"]["evaluation_scope"].startswith("unambiguous consecutive single-token")
tools = prefetch["adjacent_layer"]["limits"]["1"]["per_segment"]["tools_low"]
assert tools["precision"] == 0.1571063829787234

assert manifest["remote_local_hashes_matched"] is True
expected_hashes = {
    "trace-361.i16": "8603fa87aa8845bb89b368b45bae4a370277df3170913fab3b241e359dd56480",
    "trace-361.steps": "00ce58df2ab763aefdcbc7be298591a7047e0adb52fab2f5b507d05d91b31ef1",
    "trace-361.slots.i64": "7f9eee9e46b29352ea5f3ae2f4817aa89b14f5a9753dff14af2d615ba59c286e",
    "slots-5792-ctx512k.json": "73962276498626af16baaa7c8737384960aff8718d254e66ab92610570c9743c",
    "slots-8400.json": "4ee071670e13f199658776ddb7b508c9a068657ea631a0cb4287db0a2afeeaed",
    "marks.txt": "eb7346e706e4bedcbb3b2b4e693c0b52feb09e2ce440c99e6193514b2e31a7e2",
}
assert {item["name"]: item["sha256"] for item in manifest["files"]} == expected_hashes

assert live["live_container_was_stopped_by_offline_run"] is False
assert live["models"] == [{"id": "glm-5.3-big", "max_model_len": 524288}]
assert live["completion_content"] == "OFFLINE_WORK_OK"
assert live["completion_ok"] is True

required = {
    "CONTRACT.md",
    "README.md",
    "input-manifest.json",
    "live-service-proof.json",
    "offline-sim-v1.json",
    "run-command.txt",
    "verify-package.py",
}
assert required.issubset({path.name for path in evidence.iterdir()})
public_text = "\n".join(
    path.read_text(errors="replace")
    for path in evidence.iterdir()
    if path.is_file() and path.name not in {"SHA256SUMS", "verify-package.py"}
)
for pattern in (
    r"/Users/",
    r"\.glm_api_key",
    r"Authorization:",
    r"Bearer [A-Za-z0-9]",
    r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY",
):
    assert not re.search(pattern, public_text, re.IGNORECASE), pattern

recipe_text = (recipe / "recipe.yaml").read_text()
assert "value: 0.08408456197831345" in recipe_text
assert "value: 0.7099777673765504" in recipe_text
print("OFFLINE_PUBLIC_PACKAGE_OK")
