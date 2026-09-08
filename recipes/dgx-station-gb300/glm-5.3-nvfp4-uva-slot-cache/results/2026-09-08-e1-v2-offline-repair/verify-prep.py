#!/usr/bin/env python3
"""Verify the E1 v2 offline repair prep package without touching Station state."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1]
REQUIRED = (
    "scripts/window_e1_v2.py",
    "scripts/launch-slotcache-portable.sh",
    "scripts/window_gate.py",
    "scripts/window_verdict.py",
    "scripts/nsys_capture_control.py",
    "scripts/nsys_bucket.py",
    "scripts/dflash2_acceptance_probe.py",
    "scripts/health-check.sh",
    "configs/slots-5792-ctx512k.json",
    "patches/sitecustomize.py",
    "patches/exact_pin.py",
    "patches/ffi_route.py",
    "patches/slot_cache_hook.py",
    "patches/slot_cache_stats.py",
    "patches/slot_cache_profile_control.py",
    "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py",
    "results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py",
    "results/2026-09-08-e1-v2-offline-repair/CONTRACT.md",
    "results/2026-09-08-e1-v2-offline-repair/README.md",
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_manifest() -> None:
    for line_number, line in enumerate((HERE / "PREP-SHA256SUMS").read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum manifest line {line_number}")
        expected, relative = match.groups()
        require("__pycache__" not in relative and not relative.endswith(".pyc"), f"pycache entry in manifest: {relative}")
        target = RECIPE / relative
        require(target.is_file(), f"checksum target missing: {relative}")
        require(sha256(target) == expected, f"checksum mismatch: {relative}")

def main() -> int:
    verify_manifest()
    missing = [path for path in REQUIRED if not (RECIPE / path).is_file()]
    require(not missing, f"missing required files: {missing}")
    runner = (RECIPE / "scripts/window_e1_v2.py").read_text()
    contract = (HERE / "CONTRACT.md").read_text()
    launcher = (RECIPE / "scripts/launch-slotcache-portable.sh").read_text()
    require("num_speculative_tokens\":2" not in runner, "runner must not contain a K=2 launch")
    require("e0-e1-20260908-v2" in runner and "e0-e1-20260908-v2" in contract, "v2 run id missing")
    require("c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea" in runner, "incumbent id pin missing")
    require("sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14" in runner, "image digest pin missing")
    require("COMPILATION_CONFIG='{" in launcher, "launcher default compilation config must avoid nested brace expansion")
    require("source-manifest.json" in runner and "raw-logs" in runner, "runner must archive source and logs")
    require("placeholder" not in runner, "runner must not fabricate nsys report placeholders")
    require("command-timeout-sec" in runner and "os.O_EXCL" in runner and "HOST_OPERATION_LOCK" in runner, "runner must enforce timeouts and output/host locking")
    require("start_new_session=True" in runner and "killpg" in runner, "runner must terminate timed-out subprocess groups")
    require("--restore-only" in runner, "runner must expose independent restore command")
    require("CANDIDATE_IMAGE" in runner and "IMAGE_DIGEST" in runner and "verify_candidate_image" in runner, "runner must launch digest-pinned candidate image")
    require("preflight/incumbent-proof.json" in runner, "runner must verify incumbent identity before stop")
    require(runner.index("docker(args, out, \"stop\", \"-t\", \"120\", E1_CANDIDATE)") < runner.index("restore_status = restore(args, out, runtime_recipe)") < runner.index("args.nsys"), "runner must stop candidate then restore before profiler analysis")
    require("executed-source" in runner and "runtime_recipe" in runner, "runner must execute archived source snapshot")
    require("engine-core/graph-node evidence" in (RECIPE / "scripts/window_verdict.py").read_text(), "E1 verdict must require engine-core/graph-node evidence")
    require("api-attribution.json" in (RECIPE / "scripts/window_verdict.py").read_text(), "E1 verdict must require API attribution")
    for rel in REQUIRED:
        path = RECIPE / rel
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    print("WINDOW_E1_V2_PREP_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
