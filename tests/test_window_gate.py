import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/window_gate.py"


def run_gate(root: Path, run_id: str = "e0-e1-e5-20260907-v1"):
    return subprocess.run(
        [sys.executable, str(GATE), str(root), run_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_window_gate_requires_exact_control_and_release(tmp_path):
    (tmp_path / "CONTROL").write_text("HOLD\n")
    held = run_gate(tmp_path)
    assert held.returncode == 2
    assert "CONTROL must equal RUN" in held.stderr

    (tmp_path / "CONTROL").write_text("RUN\n")
    missing = run_gate(tmp_path)
    assert missing.returncode == 2
    assert "missing RELEASE" in missing.stderr

    (tmp_path / "RELEASE").write_text("wrong-run\n")
    wrong = run_gate(tmp_path)
    assert wrong.returncode == 2
    assert "RELEASE does not equal" in wrong.stderr

    (tmp_path / "RELEASE").write_text("e0-e1-e5-20260907-v1\n")
    passed = run_gate(tmp_path)
    assert passed.returncode == 0, passed.stderr
    assert passed.stdout.strip() == "WINDOW_GATE_OK e0-e1-e5-20260907-v1"


def test_window_gate_rejects_unsafe_run_id_without_reading_files(tmp_path):
    result = run_gate(tmp_path, "../unsafe")
    assert result.returncode == 2
    assert "invalid run id" in result.stderr
