import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import controller


class IdentityTests(unittest.TestCase):
    def test_runtime_identity_uses_exact_docker_material_without_leaking_env(self):
        doc = {
            "Id": "abc123",
            "Name": "/synthetic-v14",
            "Image": "sha256:image",
            "Config": {"Image": "synthetic:tag", "Env": ["SECRET_TOKEN=do-not-print"], "Cmd": ["serve"]},
            "HostConfig": {"Runtime": "nvidia"},
            "Mounts": [
                {"Destination": "/z", "Source": "/tmp/z"},
                {"Destination": "/a", "Source": "/tmp/a"},
            ],
        }
        material = {
            "Id": doc["Id"],
            "Name": doc["Name"],
            "Image": doc["Image"],
            "Config": doc["Config"],
            "HostConfig": doc["HostConfig"],
            "Mounts": sorted(doc["Mounts"], key=lambda x: x["Destination"]),
        }
        expected = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

        ident = controller.runtime_identity(doc)

        self.assertEqual(expected, ident["runtime_sha256"])
        self.assertEqual("abc123", ident["id"])
        self.assertEqual("/synthetic-v14", ident["name"])
        self.assertEqual("sha256:image", ident["image"])
        self.assertEqual("synthetic:tag", ident["config_image"])
        self.assertNotIn("Env", json.dumps(ident))
        self.assertNotIn("SECRET_TOKEN", json.dumps(ident))


class ContractOrderTests(unittest.TestCase):
    def test_contract_boot_order_is_exact_six_profiles(self):
        contract = {
            "order": [[1, "v14"], [1, "v13"], [2, "v13"], [2, "v14"], [3, "v14"], [3, "v13"]],
            "profiles": {"v13": {}, "v14": {}},
        }

        order = controller.parse_boot_order(contract)

        self.assertEqual(
            [(1, "v14"), (1, "v13"), (2, "v13"), (2, "v14"), (3, "v14"), (3, "v13")],
            order,
        )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class ReleaseInputTests(unittest.TestCase):
    def test_corrupt_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "release"
            fixture = root / "fixture.json"
            contract = root / "contract.json"
            output = root / "out"
            _write(release / "CONTROL", b"RUN\n")
            _write(release / "RELEASE", b"run-1\n")
            _write(release / "replay" / "replay_matched.py", b"print('ok')\n")
            _write(
                release / "manifest.json",
                json.dumps({"files": {"CONTROL": _sha(b"RUN\n"), "RELEASE": _sha(b"run-1\n"), "replay/replay_matched.py": "0" * 64}}).encode(),
            )
            _write(fixture, b"[]")
            _write(contract, json.dumps({"fixture_sha256": _sha(b"[]")}).encode())

            with self.assertRaisesRegex(controller.ControllerError, "manifest hash mismatch"):
                controller.verify_release_inputs(release, contract, fixture, output, "run-1")

    def test_fixture_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "release"
            fixture = root / "fixture.json"
            contract = root / "contract.json"
            output = root / "out"
            _write(release / "CONTROL", b"RUN\n")
            _write(release / "RELEASE", b"run-1\n")
            manifest_files = {"CONTROL": _sha(b"RUN\n"), "RELEASE": _sha(b"run-1\n")}
            _write(release / "manifest.json", json.dumps({"files": manifest_files}).encode())
            _write(fixture, b"actual")
            _write(contract, json.dumps({"fixture_sha256": _sha(b"expected")}).encode())

            with self.assertRaisesRegex(controller.ControllerError, "fixture sha256 mismatch"):
                controller.verify_release_inputs(release, contract, fixture, output, "run-1")

    def test_stale_control_or_release_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "release"
            fixture = root / "fixture.json"
            contract = root / "contract.json"
            output = root / "out"
            _write(release / "CONTROL", b"STOP\n")
            _write(release / "RELEASE", b"run-1\n")
            manifest_files = {"CONTROL": _sha(b"STOP\n"), "RELEASE": _sha(b"run-1\n")}
            _write(release / "manifest.json", json.dumps({"files": manifest_files}).encode())
            _write(fixture, b"[]")
            _write(contract, json.dumps({"fixture_sha256": _sha(b"[]")}).encode())

            with self.assertRaisesRegex(controller.ControllerError, "CONTROL is not RUN"):
                controller.verify_release_inputs(release, contract, fixture, output, "run-1")

    def test_campaign_refuses_bad_release_before_runtime_calls(self):
        class NoCallRuntime:
            def __init__(self):
                self.commands = []
            def run(self, argv, timeout_s=None):
                self.commands.append(list(argv))
                raise AssertionError("runtime should not be called")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "release"
            fixture = root / "fixture.json"
            contract = root / "contract.json"
            output = root / "out"
            _write(release / "CONTROL", b"RUN\n")
            _write(release / "RELEASE", b"run-1\n")
            _write(release / "controller.py", b"ok\n")
            _write(release / "manifest.json", json.dumps({"files": {"CONTROL": _sha(b"RUN\n"), "RELEASE": _sha(b"run-1\n"), "controller.py": "0" * 64}}).encode())
            _write(fixture, b"[]")
            _write(contract, json.dumps({"fixture_sha256": _sha(b"[]")}).encode())
            runtime = NoCallRuntime()

            with self.assertRaisesRegex(controller.ControllerError, "manifest hash mismatch"):
                controller.run_campaign(release, contract, fixture, output, "run-1", "guard", runtime=runtime)

            self.assertEqual([], runtime.commands)


class FakeRuntime:
    def __init__(self, docs, running_ps=""):
        self.docs = {doc["Id"]: json.loads(json.dumps(doc)) for doc in docs}
        self.commands = []
        self.started = set()
        self.running_ps = running_ps

    def run(self, argv, timeout_s=None):
        self.commands.append(list(argv))
        if argv[:3] == ["docker", "ps", "--format"]:
            return controller.CmdResult(self.running_ps, "", 0)
        if argv[0] == "nvidia-smi":
            return controller.CmdResult("", "", 0)
        if argv[:2] == ["docker", "inspect"]:
            doc = json.loads(json.dumps(self.docs[argv[2]]))
            doc.setdefault("State", {})["Running"] = argv[2] in self.started
            doc["State"].setdefault("StartedAt", "2026-09-15T12:00:00.000000000Z")
            return controller.CmdResult(json.dumps([doc]), "", 0)
        if argv[:2] == ["docker", "start"]:
            self.started.add(argv[2])
            return controller.CmdResult(argv[2] + "\n", "", 0)
        if argv[:3] == ["docker", "stop", "--time"]:
            self.started.discard(argv[4])
            return controller.CmdResult(argv[4] + "\n", "", 0)
        raise AssertionError(f"unexpected command: {argv}")


def _docker_doc(cid="cid-v14", name="/profile-v14", image="sha256:image", config_image="image:tag"):
    return {
        "Id": cid,
        "Name": name,
        "Image": image,
        "Config": {"Image": config_image, "Env": ["SECRET=hidden"]},
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
        "Mounts": [{"Destination": "/model", "Source": "/models/synthetic"}],
        "State": {"Running": False, "StartedAt": "2026-09-15T12:00:00.000000000Z"},
    }


def _profile_from_doc(doc):
    ident = controller.runtime_identity(doc)
    return {
        "id": ident["id"],
        "name": ident["name"],
        "image": ident["image"],
        "config_image": ident["config_image"],
        "runtime_sha256": ident["runtime_sha256"],
    }


class DockerSeamTests(unittest.TestCase):
    def test_start_inspect_stop_use_exact_preserved_container_commands(self):
        doc = _docker_doc()
        fake = FakeRuntime([doc])
        contract = {"profiles": {"v14": _profile_from_doc(doc)}}
        docker = controller.DockerController(contract, fake, port_is_dark=lambda: True)

        docker.start_profile("v14")
        docker.stop_owned()

        self.assertIn(["docker", "start", "cid-v14"], fake.commands)
        self.assertIn(["docker", "stop", "--time", "30", "cid-v14"], fake.commands)
        forbidden = {"run", "create", "rename", "delete", "rm"}
        self.assertFalse(any(cmd[:1] == ["docker"] and len(cmd) > 1 and cmd[1] in forbidden for cmd in fake.commands))

    def test_foreign_running_container_is_refused_and_never_stopped(self):
        doc = _docker_doc()
        fake = FakeRuntime([doc], running_ps='{"ID":"foreign","Image":"other"}\n')
        contract = {"profiles": {"v14": _profile_from_doc(doc)}}
        docker = controller.DockerController(contract, fake, port_is_dark=lambda: True)

        with self.assertRaisesRegex(controller.ControllerError, "container is already running"):
            docker.start_profile("v14")

        self.assertNotIn(["docker", "stop", "--time", "30", "foreign"], fake.commands)
        self.assertFalse(any(cmd[:2] == ["docker", "stop"] for cmd in fake.commands))


class ExplodingPhaseRunner:
    def run_boot(self, *args, **kwargs):
        raise controller.ControllerError("phase failed")


class CampaignFailureTests(unittest.TestCase):
    def test_phase_failure_stops_owned_container(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            doc = _docker_doc()
            fake = FakeRuntime([doc])
            contract = {"profiles": {"v14": _profile_from_doc(doc)}, "model": "synthetic", "context": 10}
            release = root / "release"
            _write(release / "CONTROL", b"RUN\n")
            _write(release / "RELEASE", b"run-1\n")
            campaign = controller.CampaignController(
                contract=contract,
                release=release,
                fixture=root / "fixture.json",
                output=root / "out",
                run_id="run-1",
                guard_unit="guard",
                runtime=fake,
                phase_runner=ExplodingPhaseRunner(),
                port_is_dark=lambda: True,
                guard_check=lambda: None,
                readiness_check=lambda profile_key, boot_dir: None,
            )

            with self.assertRaisesRegex(controller.ControllerError, "phase failed"):
                campaign.run_one_boot(1, "v14")

            self.assertIn(["docker", "stop", "--time", "30", "cid-v14"], fake.commands)
            self.assertNotIn("cid-v14", fake.started)


class GuardRuntime:
    def __init__(self, exec_start):
        self.exec_start = exec_start
        self.commands = []

    def run(self, argv, timeout_s=None):
        self.commands.append(list(argv))
        if argv[:2] == ["systemctl", "is-active"]:
            return controller.CmdResult("active\n", "", 0)
        if argv[:4] == ["systemctl", "show", "--value", "-p"]:
            return controller.CmdResult(self.exec_start + "\n", "", 0)
        raise AssertionError(argv)


class GuardTests(unittest.TestCase):
    def test_guard_timer_active_and_execstart_exact_ids(self):
        runtime = GuardRuntime("{ path=/usr/bin/docker ; argv[]=/usr/bin/docker stop --time 30 id14 id13 ; ignore_errors=no ; }")

        controller.verify_guard_timer(runtime, "dsv41-r7-hardstop-run-1", "id14", "id13")

        self.assertIn(["systemctl", "is-active", "dsv41-r7-hardstop-run-1.timer"], runtime.commands)
        self.assertIn(["systemctl", "show", "--value", "-p", "ExecStart", "dsv41-r7-hardstop-run-1.service"], runtime.commands)


class FakeProcessRunner:
    def __init__(self):
        self.commands = []

    def run(self, argv, cwd, timeout_s, control_check, stdout_path, stderr_path):
        self.commands.append(list(argv))
        control_check()
        out_dir = Path(argv[argv.index("--output") + 1])
        out_dir.mkdir(parents=True, exist_ok=False)
        tag = argv[argv.index("--tag") + 1]
        if "replay_matched.py" in argv[1]:
            (out_dir / "summary.json").write_text(json.dumps({
                "tag": tag,
                "fixture_sha256": "fixture-sha",
                "fixture_session_count": 20,
                "fixture_turns_per_session": 15,
                "turn_count": 300,
                "expected_turn_count": 300,
                "wall_s": 1.5,
                "aggregate_completion_tok_s": 2.0,
            }))
        else:
            (out_dir / "summary.json").write_text(json.dumps({
                "tag": tag,
                "task_count": 12,
                "expected_task_count": 12,
                "capture_error_count": 0,
                "wall_s": 1.0,
                "results": [{"task_id": f"t{i}"} for i in range(12)],
            }))
        return {"returncode": 0, "argv": list(argv)}


class PhaseRunnerTests(unittest.TestCase):
    def test_phase_runner_uses_pair_namespace_and_exact_replay_gauntlet_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "release"
            (release / "replay").mkdir(parents=True)
            (release / "gauntlet").mkdir()
            fixture = root / "fixture.json"
            fixture.write_text("fixture")
            fake = FakeProcessRunner()
            phase = controller.PhaseRunner(
                release=release,
                fixture=fixture,
                run_id="run-1",
                model="dsv41-flash-uva",
                fixture_sha256="fixture-sha",
                sandbox_image="synthetic-sandbox",
                process_runner=fake,
                warmup_runner=lambda *args, **kwargs: {"warmups": 7},
            )
            campaign = type("CampaignStub", (), {"check_control_and_guard": lambda self: None})()

            summary = phase.run_boot(campaign, 1, "v14", root / "boot")

            replay_cmds = [cmd for cmd in fake.commands if "replay_matched.py" in cmd[1]]
            gauntlet_cmds = [cmd for cmd in fake.commands if "run_suite.py" in cmd[1]]
            self.assertEqual(2, len(replay_cmds))
            self.assertEqual(2, len(gauntlet_cmds))
            self.assertEqual("r7-run-1-pair1", replay_cmds[0][replay_cmds[0].index("--cache-salt") + 1])
            self.assertEqual("r7-run-1-pair1", replay_cmds[1][replay_cmds[1].index("--cache-salt") + 1])
            self.assertIn("--cache-state", replay_cmds[0]); self.assertIn("initial", replay_cmds[0])
            self.assertIn("--cache-state", replay_cmds[1]); self.assertIn("repeat", replay_cmds[1])
            self.assertIn("--workers", replay_cmds[0]); self.assertIn("4", replay_cmds[0])
            self.assertIn("--max-tokens", replay_cmds[0]); self.assertIn("400", replay_cmds[0])
            self.assertIn("--sandbox-image", gauntlet_cmds[-1]); self.assertIn("synthetic-sandbox", gauntlet_cmds[-1])
            self.assertEqual("R7-p1-v14-fresh", summary["replay_initial"]["tag"])
            self.assertEqual("R7-p1-v14-repeat", summary["replay_repeat"]["tag"])
            self.assertEqual("R7-p1-v14-agent", summary["gauntlet_scored"]["tag"])


if __name__ == "__main__":
    unittest.main()
