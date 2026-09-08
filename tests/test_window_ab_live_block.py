"""The unreleased v3 CLI must reject before any external executable."""
import os
from pathlib import Path
import subprocess
import sys

RUNNER = Path(__file__).resolve().parents[1] / 'recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/window_ab_v3.py'


def test_unreleased_cli_blocks_before_docker_or_output_mutation(tmp_path):
    marker = tmp_path / 'docker-called'
    docker = tmp_path / 'docker'
    docker.write_text('#!' + sys.executable + '\nfrom pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\nraise SystemExit(1)\n')
    docker.chmod(0o700)
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(RUNNER), '--root', str(tmp_path / 'root'),
                             '--out', str(out), '--docker', str(docker),
                             '--host-operation-lock', str(tmp_path / 'host.lock')],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'PREP_BLOCKED' in result.stderr
    assert not marker.exists(), 'unreleased CLI invoked Docker'
    assert not out.exists(), 'unreleased CLI wrote an execution directory'
