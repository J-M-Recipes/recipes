#!/usr/bin/env python3
"""Static verifier for the prepared (not released) E0/E1/E5 window."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1]
REPO = RECIPE.parents[2]

EXPECTED_FIXTURES = {
    "fixtures/bench_big.py": "3ac6e7ec174d38d73dad6c9130b014225da1c7c57ffc2d1ec2a8660964ac4757",
    "fixtures/bench_big_code.py": "102d04536f4d181732b176707e758063f956ce58f28706af9bd441df6699344c",
}
REQUIRED = [
    HERE / "CONTRACT.md",
    HERE / "README.md",
    HERE / "PREP-SHA256SUMS",
    RECIPE / "patches/slot_cache_hook.py",
    RECIPE / "patches/slot_cache_stats.py",
    RECIPE / "patches/slot_cache_profile_control.py",
    RECIPE / "scripts/launch-slotcache-portable.sh",
    RECIPE / "scripts/nsys_bucket.py",
    RECIPE / "scripts/nsys_capture_control.py",
    RECIPE / "scripts/window_gate.py",
    RECIPE / "scripts/window_verdict.py",
    RECIPE / "scripts/dflash2_acceptance_probe.py",
    RECIPE / "research/window-plan-e0-e1-e5-2026-09-07.md",
    RECIPE / "research/window-live-baseline-2026-09-07.json",
    REPO / "tests/test_slotcache_helpers.py",
    REPO / "tests/test_nsys_bucket.py",
    REPO / "tests/test_window_gate.py",
    REPO / "tests/test_window_verdict.py",
    REPO / "tests/test_verify_window_prep.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_manifest(manifest: Path) -> None:
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum manifest line {line_number}")
        if match is None:  # narrow the type after the fail-closed check
            raise ValueError(f"invalid checksum manifest line {line_number}")
        expected, relative = match.groups()
        target = manifest.parent / relative
        require(target.is_file(), f"checksum target missing: {relative}")
        require(sha256(target) == expected, f"checksum mismatch: {relative}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    require(spec.loader is not None, f"cannot load module: {path}")
    if spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    spec.loader.exec_module(module)
    return module


def main() -> int:
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    require(not missing, f"missing required files: {missing}")
    verify_manifest(HERE / "PREP-SHA256SUMS")
    require(not (HERE / "CONTROL").exists(), "release CONTROL must not be committed")
    require(not (HERE / "RELEASE").exists(), "release token must not be committed")

    contract = (HERE / "CONTRACT.md").read_text()
    for phrase in (
        "PREPARED, NOT RELEASED",
        "CONTROL=RUN",
        "RELEASE=e0-e1-e5-20260907-v1",
        "recoverable work >=2.0 ms/step",
        "accepted length +0.15",
        "K=2 changes only",
        "never promotes a candidate to production",
    ):
        require(phrase in contract, f"contract is missing frozen phrase: {phrase}")
    for unwanted in ("imageulk", "--porn", "to=functions", "aderíanout"):
        require(unwanted not in contract, f"contract contains corrupt token: {unwanted}")
    bash_blocks = re.findall(r"```bash\n(.*?)```", contract, flags=re.DOTALL)
    require(bool(bash_blocks), "contract has no executable bash blocks")
    for index, block in enumerate(bash_blocks, 1):
        syntax = subprocess.run(
            ["bash", "-n"], input=block, text=True, capture_output=True, check=False
        )
        require(syntax.returncode == 0, f"bash block {index} has invalid syntax: {syntax.stderr}")

    plan = (RECIPE / "research/window-plan-e0-e1-e5-2026-09-07.md").read_text()
    require("nothing launched" in plan, "plan must preserve the nothing-launched status")
    require("separate explicit window go" in plan, "plan must require a separate window go")
    require("force aggregate kernel duration to sum" not in plan, "plan contains invalid overlap accounting")

    baseline = json.loads((RECIPE / "research/window-live-baseline-2026-09-07.json").read_text())
    require(baseline["schema"] == "glm53-window-live-baseline-v1", "baseline schema mismatch")
    require(baseline["running"] is True, "baseline must record the running incumbent")
    require(
        baseline["container"] == "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907",
        "baseline container mismatch",
    )
    require(baseline["image"] == "vllm-glm53-uva:v0.28.0-2cf0a691", "baseline image mismatch")
    require(
        baseline["args"][baseline["args"].index("--max-model-len") + 1] == "524288",
        "baseline max-model-len mismatch",
    )
    spec_index = baseline["args"].index("--speculative-config")
    require(
        json.loads(baseline["args"][spec_index + 1])
        == {"method": "mtp", "num_speculative_tokens": 1},
        "baseline speculative configuration mismatch",
    )

    for relative, expected in EXPECTED_FIXTURES.items():
        require(sha256(HERE / relative) == expected, f"fixture hash mismatch: {relative}")

    hook = (RECIPE / "patches/slot_cache_hook.py").read_text()
    require("tl.atomic_add(route_count_ptr, N)" in hook, "hook lacks observed-route accounting")
    require("summarize_window(delta_misses=dm, delta_routes=dr" in hook, "hook lacks telemetry summary")
    require("tot_m / (tot_s * 8)" not in hook, "hook retains the legacy fixed-route denominator")
    require("SLOT_CACHE_PROFILE_CONTROL" in hook, "hook lacks profiler control")

    launcher = (RECIPE / "scripts/launch-slotcache-portable.sh").read_text()
    for phrase in (
        'if [ "${NSYS:-0}" = "1" ]',
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--entrypoint /wcap/nsys-cli",
        "SLOT_CACHE_PROFILE_CONTROL",
    ):
        require(phrase in launcher, f"launcher missing: {phrase}")

    stats = load_module("slot_cache_stats_verify", RECIPE / "patches/slot_cache_stats.py")
    require(
        stats.summarize_window(delta_misses=400, delta_routes=1600, delta_steps=100)["hit_rate"] == 0.75,
        "observed-route telemetry smoke check failed",
    )

    for path in REQUIRED:
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))

    public_paths = list(HERE.rglob("*")) + [
        RECIPE / "research/window-plan-e0-e1-e5-2026-09-07.md",
        RECIPE / "research/window-live-baseline-2026-09-07.json",
    ]
    private_home = "/Users/" + "jamesmeadlock"
    credential_assignment = "VLLM_API" + "_KEY="
    for path in public_paths:
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".sh", ".txt"}:
            continue
        text = path.read_text(errors="replace")
        require(private_home not in text, f"private local path in {path}")
        require(credential_assignment not in text, f"credential assignment in {path}")

    print("WINDOW_PREP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
