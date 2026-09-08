import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PREP = (
    REPO_ROOT
    / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-07-e0-e1-e5-window/verify-prep.py"
)


def load_verifier():
    spec = importlib.util.spec_from_file_location("window_verify_prep", VERIFY_PREP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_verification_uses_python_and_rejects_tampering(tmp_path):
    verifier = load_verifier()
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen\n")
    digest = verifier.sha256(payload)
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{digest}  payload.txt\n")

    verifier.verify_manifest(manifest)

    payload.write_text("tampered\n")
    with pytest.raises((AssertionError, ValueError), match="checksum mismatch: payload.txt"):
        verifier.verify_manifest(manifest)


def test_manifest_verification_cannot_be_disabled_by_python_optimize(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("tampered\n")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{'0' * 64}  payload.txt\n")
    code = (
        "import importlib.util, pathlib; "
        f"p=pathlib.Path({str(VERIFY_PREP)!r}); "
        "s=importlib.util.spec_from_file_location('v',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.verify_manifest(pathlib.Path({str(manifest)!r}))"
    )
    env = dict(os.environ, PYTHONOPTIMIZE="1")

    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False, env=env
    )

    assert result.returncode != 0
    assert "checksum mismatch: payload.txt" in result.stderr


def test_static_verifier_contains_no_optimizable_assert_statements():
    tree = ast.parse(VERIFY_PREP.read_text())
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
