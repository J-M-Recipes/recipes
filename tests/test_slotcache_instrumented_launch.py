import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
LAUNCHER = RECIPE / "scripts/launch-slotcache-portable.sh"
PATCHED_CONTAINER_RUNNER = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
ORIGINAL_RUNNER_SHA = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
SOURCE_SHA = ORIGINAL_RUNNER_SHA
IMAGE_SHA = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
DEFAULT_IMAGE_TAG = "vllm-glm53-uva:v0.28.0-2cf0a691"
PINNED_SOURCE = Path(
    os.environ.get(
        "K2_V3_RUNTIME_GPU_MODEL_RUNNER",
        REPO_ROOT / "tests/fixtures/k2-v3-runtime-source/vllm/v1/worker/gpu_model_runner.py",
    )
)
PATCH_GENERATOR = RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py"


def _recipe_manifest_sha256(recipe: Path = RECIPE) -> str:
    """Hash the launch recipe source artifact mounted at /w, excluding runtime outputs."""
    ignored_dirs = {".git", "__pycache__", "capture", "results"}
    rows = []
    for path in sorted(recipe.rglob("*")):
        rel_path = path.relative_to(recipe)
        if any(part in ignored_dirs for part in rel_path.parts):
            continue
        if path.is_symlink():
            raise AssertionError(f"recipe manifest input must not be a symlink: {rel_path}")
        if not path.is_file():
            continue
        rel = rel_path.as_posix()
        if rel.endswith(".pyc"):
            continue
        data = path.read_bytes()
        rows.append(f"{rel}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}")
    manifest = "\n".join(rows).encode() + b"\n"
    return hashlib.sha256(manifest).hexdigest()


RECIPE_SHA = _recipe_manifest_sha256()


def _run(cmd, *, env=None, cwd=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        env=full_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)
    return path


def _fake_docker_recorder(fakebin: Path, record: Path) -> None:
    _write_executable(
        fakebin / "docker",
        f"""
        #!/usr/bin/env python3
        import json, os, sys
        with open({str(record)!r}, "w") as f:
            json.dump({{"argv": sys.argv[1:], "env": dict(os.environ)}}, f, sort_keys=True)
        print("fake-container-id")
        """,
    )


def _generate_patched_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "gpu_model_runner.patched.py"
    result = _run([
        sys.executable, str(PATCH_GENERATOR),
        "--source", str(PINNED_SOURCE),
        "--output", str(runner),
    ])
    assert result.returncode == 0, result.stderr
    return runner


def _copy_recipe_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "recipe-fixture"
    shutil.copytree(RECIPE, fixture, ignore=shutil.ignore_patterns("capture", "results", "__pycache__"))
    return fixture


def _base_env(tmp_path: Path, *, fakebin: Path, record: Path, secret="synthetic-secret", recipe: Path = RECIPE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".glm_api_key").write_text(secret + "\n")
    capture = tmp_path / "capture"
    capture.mkdir(exist_ok=True)
    snapshot = capture / "quiescent-run-a"
    snapshot.mkdir(exist_ok=True)
    runner = _generate_patched_runner(tmp_path)
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    return {
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "HOME": str(home),
        "MODEL_DIR": "/models/glm",
        "CACHE_DIR": str(tmp_path / "cache"),
        "CAPTURE_DIR": str(capture),
        "SLOT_CACHE_QUIESCENT_SNAPSHOTS": "1",
        "SLOT_CACHE_PATCHED_RUNNER": str(runner),
        "SLOT_CACHE_PATCHED_RUNNER_SHA256": runner_sha,
        "SLOT_CACHE_SOURCE_RUNNER": str(PINNED_SOURCE),
        "IMAGE": IMAGE_SHA,
        "SLOT_CACHE_RUN_ID": "run-a",
        "SLOT_CACHE_SOURCE_SHA": SOURCE_SHA,
        "SLOT_CACHE_IMAGE_SHA": IMAGE_SHA,
        "SLOT_CACHE_RECIPE_SHA": _recipe_manifest_sha256(recipe),
        "SLOT_CACHE_ENGINE_GENERATION": "1",
        "SLOT_CACHE_WINDOW_STEPS": "100:120,200:220",
        "SLOT_CACHE_SNAPSHOT_DIR": "/wcap/quiescent-run-a",
        "SLOT_CACHE_K_MODE": "K1",
        "STATS_SEC": "0",
    }


def test_enabled_quiescent_rejects_reviewer_bogus_metadata_before_key_or_docker(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    Path(env["HOME"], ".glm_api_key").unlink()
    env.update({
        "IMAGE": "totally-different-tag:no-digest",
        "SLOT_CACHE_SOURCE_SHA": "a" * 64,
        "SLOT_CACHE_IMAGE_SHA": "sha256:" + "b" * 64,
        "SLOT_CACHE_RECIPE_SHA": "c" * 64,
    })

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "pinned local Docker image ID" in result.stderr or "mismatch" in result.stderr
    assert "API_KEY" not in result.stderr
    assert not record.exists()


def test_enabled_quiescent_launch_requires_image_source_and_recipe_bound_to_actual_artifacts(tmp_path):
    for idx, (updates, expected) in enumerate([
        ({"IMAGE": IMAGE_SHA, "SLOT_CACHE_IMAGE_SHA": "sha256:" + "0" * 64}, "SLOT_CACHE_IMAGE_SHA must match IMAGE local ID"),
        ({"IMAGE": f"vllm-glm53-uva@{IMAGE_SHA}"}, "IMAGE must be the pinned local Docker image ID"),
        ({"SLOT_CACHE_SOURCE_SHA": "a" * 64}, "SLOT_CACHE_SOURCE_SHA must match pinned gpu_model_runner.py source"),
        ({"SLOT_CACHE_RECIPE_SHA": "c" * 64}, "SLOT_CACHE_RECIPE_SHA must match recipe manifest"),
    ]):
        record = tmp_path / f"docker-{idx}.json"
        fakebin = tmp_path / f"bin-{idx}"
        fakebin.mkdir()
        env = _base_env(tmp_path / f"case-{idx}", fakebin=fakebin, record=record)
        env["IMAGE"] = IMAGE_SHA
        env.update(updates)

        result = _launch(tmp_path, env)

        assert result.returncode == 2
        assert expected in result.stderr
        assert not record.exists()


def _launch(tmp_path: Path, env: dict, *extra: str, launcher: Path = LAUNCHER):
    return _run(
        [
            "bash", str(launcher), "run-a", "112",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}',
            *extra,
        ],
        env=env,
        cwd=tmp_path,
    )


def _docker_record(record: Path):
    data = json.loads(record.read_text())
    return data["argv"], data["env"]


def _env_values(argv):
    return [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-e"]


def _mounts(argv):
    return [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-v"]


def test_default_portable_launch_does_not_mount_or_enable_quiescent_instrumentation(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".glm_api_key").write_text("synthetic-secret\n")

    result = _run(
        ["bash", str(LAUNCHER), "sc13g", "112"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "HOME": str(home),
            "MODEL_DIR": "/models/glm",
            "CACHE_DIR": str(tmp_path / "cache"),
            "CAPTURE_DIR": str(tmp_path / "capture"),
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    argv, _ = _docker_record(record)
    assert DEFAULT_IMAGE_TAG in argv
    image_index = argv.index(DEFAULT_IMAGE_TAG)
    assert argv[image_index + 1 : image_index + 3] == ["/model", "--host"]
    assert PATCHED_CONTAINER_RUNNER not in " ".join(argv)
    env_values = _env_values(argv)
    assert "SLOT_CACHE_QUIESCENT_SNAPSHOTS=1" not in env_values
    assert "SLOT_CACHE_STATS_MODE=quiescent" not in env_values
    assert not any(v.startswith("PYTHONPATH=") for v in env_values)
    assert "--run-live" not in argv


def test_disabled_quiescent_launch_preserves_explicit_image_override(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _fake_docker_recorder(fakebin, record)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".glm_api_key").write_text("synthetic-secret\n")

    result = _run(
        ["bash", str(LAUNCHER), "sc13g", "112"],
        env={
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "HOME": str(home),
            "MODEL_DIR": "/models/glm",
            "CACHE_DIR": str(tmp_path / "cache"),
            "CAPTURE_DIR": str(tmp_path / "capture"),
            "IMAGE": "custom-local-image:dev",
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    argv, _ = _docker_record(record)
    assert "custom-local-image:dev" in argv
    assert DEFAULT_IMAGE_TAG not in argv
    assert IMAGE_SHA not in argv


def test_enabled_quiescent_no_image_defaults_to_pinned_local_image_id(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    env.pop("IMAGE")

    result = _launch(tmp_path, env)

    assert result.returncode == 0, result.stderr
    argv, _ = _docker_record(record)
    assert IMAGE_SHA in argv
    assert DEFAULT_IMAGE_TAG not in argv


def test_enabled_quiescent_rejects_explicit_image_tag_even_when_image_sha_matches(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    env["IMAGE"] = DEFAULT_IMAGE_TAG

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "IMAGE must be the pinned local Docker image ID" in result.stderr
    assert not record.exists()


def test_enabled_quiescent_launch_mounts_verified_runner_readonly_and_passes_allowlisted_metadata(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    env.update({
        "SLOT_CACHE_COUNTER_SCOPE": "target_slot_cache",
        "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS": "1",
        "SLOT_CACHE_EXPECTED_LAYERS": "75",
        "SLOT_CACHE_STATS_MODE": "production-should-not-leak",
        "SLOT_CACHE_VALID_FOR_CAMPAIGN": "1",
        "SLOT_CACHE_SECRET_SENTINEL": "do-not-pass",
    })

    result = _launch(tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert "synthetic-secret" not in result.stdout + result.stderr
    argv, process_env = _docker_record(record)
    mounts = _mounts(argv)
    assert f"{env['SLOT_CACHE_PATCHED_RUNNER']}:{PATCHED_CONTAINER_RUNNER}:ro" in mounts
    assert f"{RECIPE / 'patches'}:/w/patches:ro" not in mounts  # /w already provides patches read-only
    env_values = _env_values(argv)
    expected = {
        "SLOT_CACHE_QUIESCENT_SNAPSHOTS=1",
        "SLOT_CACHE_STATS_MODE=quiescent",
        "SLOT_CACHE_EXPECTED_LAYERS=75",
        "SLOT_CACHE_SNAPSHOT_DIR=/wcap/quiescent-run-a",
        "SLOT_CACHE_WINDOW_STEPS=100:120,200:220",
        "SLOT_CACHE_RUN_ID=run-a",
        f"SLOT_CACHE_SOURCE_SHA={SOURCE_SHA}",
        f"SLOT_CACHE_IMAGE_SHA={IMAGE_SHA}",
        f"SLOT_CACHE_RECIPE_SHA={RECIPE_SHA}",
        "SLOT_CACHE_ENGINE_GENERATION=1",
        "SLOT_CACHE_K_MODE=K1",
        "SLOT_CACHE_COUNTER_SCOPE=target_slot_cache",
        "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS=1",
        "PYTHONPATH=/w/patches",
        "SLOT_CACHE_CAPTURE=0",
        "SLOT_CACHE_STATS_SEC=0",
    }
    assert expected.issubset(set(env_values))
    assert "SLOT_CACHE_VALID_FOR_CAMPAIGN=1" not in env_values
    assert "SLOT_CACHE_SECRET_SENTINEL=do-not-pass" not in env_values
    assert "SLOT_CACHE_STATS_MODE=production-should-not-leak" not in env_values
    assert process_env["SLOT_CACHE_SECRET_SENTINEL"] == "do-not-pass"  # inherited by fake docker process, not passed via -e


def test_enabled_quiescent_checks_runner_hash_before_api_key_or_docker(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    Path(env["HOME"], ".glm_api_key").unlink()
    env["SLOT_CACHE_PATCHED_RUNNER_SHA256"] = "0" * 64

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "patched runner sha256 mismatch" in result.stderr
    assert "API_KEY" not in result.stderr
    assert not record.exists()



def test_enabled_quiescent_rejects_forged_patched_runner_receipt_before_key_or_docker(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    forged = tmp_path / "forged-gpu_model_runner.py"
    forged.write_text("# caller supplied bytes with matching caller supplied hash\n")
    env["SLOT_CACHE_PATCHED_RUNNER"] = str(forged)
    env["SLOT_CACHE_PATCHED_RUNNER_SHA256"] = hashlib.sha256(forged.read_bytes()).hexdigest()
    Path(env["HOME"], ".glm_api_key").unlink()

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "patched runner does not match deterministic generator output" in result.stderr
    assert "API_KEY" not in result.stderr
    assert not record.exists()


def test_enabled_quiescent_rejects_external_patch_generator_before_execution_key_or_docker(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    forged_generator = tmp_path / "forged-generator.py"
    marker = tmp_path / "forged-generator-executed"
    _write_executable(
        forged_generator,
        f"""
        #!/usr/bin/env python3
        import shutil
        from pathlib import Path

        Path({str(marker)!r}).write_text("executed\\n")
        output = Path(__import__("sys").argv[__import__("sys").argv.index("--output") + 1])
        shutil.copyfile({env["SLOT_CACHE_PATCHED_RUNNER"]!r}, output)
        """,
    )
    env["SLOT_CACHE_PATCH_GENERATOR"] = str(forged_generator)
    Path(env["HOME"], ".glm_api_key").unlink()

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "SLOT_CACHE_PATCH_GENERATOR must be the staged recipe generator" in result.stderr
    assert "API_KEY" not in result.stderr
    assert not marker.exists()
    assert not record.exists()


def test_enabled_quiescent_rejects_symlinked_source_patched_and_recipe_inputs_before_docker(tmp_path):
    cases = []

    record = tmp_path / "docker-patched-link.json"
    fakebin = tmp_path / "bin-patched-link"
    fakebin.mkdir()
    env = _base_env(tmp_path / "patched-link", fakebin=fakebin, record=record)
    real_runner = Path(env["SLOT_CACHE_PATCHED_RUNNER"])
    linked_runner = real_runner.with_name("linked-runner.py")
    linked_runner.symlink_to(real_runner)
    env["SLOT_CACHE_PATCHED_RUNNER"] = str(linked_runner)
    cases.append((env, record, "SLOT_CACHE_PATCHED_RUNNER must not contain symlinks"))

    record = tmp_path / "docker-source-dir-link.json"
    fakebin = tmp_path / "bin-source-dir-link"
    fakebin.mkdir()
    env = _base_env(tmp_path / "source-dir-link", fakebin=fakebin, record=record)
    src_root = tmp_path / "source-real"
    src_file = src_root / "gpu_model_runner.py"
    src_root.mkdir()
    src_file.write_bytes(PINNED_SOURCE.read_bytes())
    src_link_dir = tmp_path / "source-link"
    src_link_dir.symlink_to(src_root, target_is_directory=True)
    env["SLOT_CACHE_SOURCE_RUNNER"] = str(src_link_dir / "gpu_model_runner.py")
    cases.append((env, record, "SLOT_CACHE_SOURCE_RUNNER must not contain symlinks"))

    temp_recipe = _copy_recipe_fixture(tmp_path)
    record = tmp_path / "docker-recipe-link.json"
    fakebin = tmp_path / "bin-recipe-link"
    fakebin.mkdir()
    env = _base_env(tmp_path / "recipe-link", fakebin=fakebin, record=record, recipe=temp_recipe)
    (temp_recipe / "zz-test-launcher-symlink-input").symlink_to(temp_recipe / "README.md")
    cases.append((env, record, "recipe artifact input must not be a symlink", temp_recipe / "scripts/launch-slotcache-portable.sh"))

    for case in cases:
        if len(case) == 3:
            env, record, expected = case
            launcher = LAUNCHER
        else:
            env, record, expected, launcher = case
        result = _launch(tmp_path, env, launcher=launcher)
        assert result.returncode == 2
        assert expected in result.stderr
        assert not record.exists()


def test_enabled_quiescent_rejects_missing_runner_before_docker(tmp_path):
    record = tmp_path / "docker.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env = _base_env(tmp_path, fakebin=fakebin, record=record)
    Path(env["SLOT_CACHE_PATCHED_RUNNER"]).unlink()

    result = _launch(tmp_path, env)

    assert result.returncode == 2
    assert "missing readable SLOT_CACHE_PATCHED_RUNNER" in result.stderr
    assert not record.exists()


def test_enabled_quiescent_rejects_path_escape_symlink_snapshot_and_nonzero_stats(tmp_path):
    for case, updates, expected in [
        ("relative", {"SLOT_CACHE_SNAPSHOT_DIR": "quiescent-run-a"}, "must be under /wcap"),
        ("traversal", {"SLOT_CACHE_SNAPSHOT_DIR": "/wcap/../escape"}, "must not contain traversal"),
        ("missing", {"SLOT_CACHE_SNAPSHOT_DIR": "/wcap/missing"}, "must already exist"),
        ("stats", {"STATS_SEC": "20"}, "STATS_SEC=0 is required"),
    ]:
        record = tmp_path / f"docker-{case}.json"
        fakebin = tmp_path / f"bin-{case}"
        fakebin.mkdir()
        env = _base_env(tmp_path / case, fakebin=fakebin, record=record)
        env.update(updates)
        result = _launch(tmp_path, env)
        assert result.returncode == 2, case
        assert expected in result.stderr, result.stderr
        assert not record.exists()

    record = tmp_path / "docker-symlink.json"
    fakebin = tmp_path / "bin-symlink"
    fakebin.mkdir()
    env = _base_env(tmp_path / "symlink", fakebin=fakebin, record=record)
    symlink = Path(env["CAPTURE_DIR"]) / "linked"
    symlink.symlink_to(Path(env["CAPTURE_DIR"]) / "quiescent-run-a", target_is_directory=True)
    env["SLOT_CACHE_SNAPSHOT_DIR"] = "/wcap/linked"
    result = _launch(tmp_path, env)
    assert result.returncode == 2
    assert "must not be a symlink" in result.stderr
    assert not record.exists()


def test_enabled_quiescent_rejects_malformed_or_inconsistent_metadata_and_duplicate_speculative_config(tmp_path):
    cases = [
        ({"SLOT_CACHE_SOURCE_SHA": "ABC"}, "SLOT_CACHE_SOURCE_SHA must be a canonical lowercase SHA"),
        ({"SLOT_CACHE_IMAGE_SHA": "not-a-digest"}, "SLOT_CACHE_IMAGE_SHA must be sha256:"),
        ({"SLOT_CACHE_ENGINE_GENERATION": "0"}, "SLOT_CACHE_ENGINE_GENERATION must be positive"),
        ({"SLOT_CACHE_EXPECTED_LAYERS": "74"}, "SLOT_CACHE_EXPECTED_LAYERS=75 is required"),
        ({"SLOT_CACHE_K_MODE": "K2"}, "SLOT_CACHE_K_MODE does not match speculative config"),
    ]
    for idx, (updates, expected) in enumerate(cases):
        record = tmp_path / f"docker-{idx}.json"
        fakebin = tmp_path / f"bin-{idx}"
        fakebin.mkdir()
        env = _base_env(tmp_path / f"case-{idx}", fakebin=fakebin, record=record)
        env.update(updates)
        result = _launch(tmp_path, env)
        assert result.returncode == 2
        assert expected in result.stderr
        assert not record.exists()

    record = tmp_path / "docker-dupe.json"
    fakebin = tmp_path / "bin-dupe"
    fakebin.mkdir()
    env = _base_env(tmp_path / "dupe", fakebin=fakebin, record=record)
    result = _launch(tmp_path, env, "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}')
    assert result.returncode == 2
    assert "exactly one --speculative-config is required" in result.stderr
    assert not record.exists()


def test_sitecustomize_does_not_import_instrumentation_adapter_so_launcher_sets_pythonpath():
    text = (RECIPE / "patches/sitecustomize.py").read_text()
    assert "slot_cache_window_instrumentation" not in text
    assert "SLOT_CACHE_QUIESCENT_SNAPSHOTS" not in text
