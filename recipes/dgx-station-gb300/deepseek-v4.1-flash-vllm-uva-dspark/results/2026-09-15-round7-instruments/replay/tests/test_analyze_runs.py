import importlib.util
import io
import json
import pathlib
import shutil
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analyze_runs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def receipt(tag, wall, tok_s, fixture="fixture-a", cfg="cfg-a", cache_state="cache_namespace_initial", salt_hash="salt-a"):
    return {
        "tag": tag,
        "cache_state": cache_state,
        "cache_salt_sha256": salt_hash,
        "fixture_sha256": fixture,
        "fixture_session_count": 20,
        "expected_turn_count": 300,
        "turn_count": 300,
        "model": "same-model",
        "workers": 4,
        "max_tokens": 400,
        "reasoning_effort": "low",
        "request_config_hash": cfg,
        "comparison_config_hash": cfg,
        "wall_s": wall,
        "completion_tokens": 1500,
        "aggregate_completion_tok_s": tok_s,
    }


class AnalyzeRunsTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.case_dir = ROOT / "_test_runs" / self._testMethodName
        shutil.rmtree(self.case_dir, ignore_errors=True)
        self.case_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def write_receipt(self, name, obj):
        path = self.case_dir / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def call_main_quiet(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return self.mod.main(argv)

    def test_compares_paired_receipts_after_label_checks_without_significance(self):
        runs = [
            ("v13", self.write_receipt("v13-1.json", receipt("v13-1", 10.0, 150.0))),
            ("v14", self.write_receipt("v14-1.json", receipt("v14-1", 8.0, 187.5))),
            ("v13", self.write_receipt("v13-2.json", receipt("v13-2", 11.0, 136.36))),
            ("v14", self.write_receipt("v14-2.json", receipt("v14-2", 9.0, 166.67))),
            ("v13", self.write_receipt("v13-3.json", receipt("v13-3", 12.0, 125.0))),
            ("v14", self.write_receipt("v14-3.json", receipt("v14-3", 7.0, 214.29))),
        ]
        out = self.case_dir / "analysis.json"
        argv = []
        for label, path in runs:
            argv.extend(["--run", f"{label}={path}"])
        argv.extend(["--output", str(out)])
        rc = self.call_main_quiet(argv)
        self.assertEqual(0, rc)
        analysis = json.loads(out.read_text())
        self.assertEqual("no significance test; mean/range only", analysis["statistics_note"])
        self.assertNotIn("p_value", json.dumps(analysis))
        self.assertEqual(3, analysis["by_label"]["v13"]["n"])
        self.assertEqual(11.0, analysis["by_label"]["v13"]["wall_s"]["mean"])
        self.assertEqual([10.0, 12.0], analysis["by_label"]["v13"]["wall_s"]["range"])
        self.assertEqual(3, len(analysis["pairs"]))
        self.assertEqual(-2.0, analysis["pairs"][0]["wall_delta_s"])

    def test_new_namespace_per_pair_is_valid_when_semantics_match(self):
        rows = []
        for pair in range(3):
            for label in ('v13','v14'):
                obj = receipt(f'{label}-{pair}', 10.0, 150.0,
                              cfg=f'exact-request-{pair}', salt_hash=f'namespace-{pair}')
                obj['comparison_config_hash'] = 'same-semantic-settings'
                rows.append((label, self.write_receipt(f'{label}-{pair}.json', obj)))
        out = self.case_dir / 'analysis-pairs.json'
        argv = sum((['--run', f'{label}={path}'] for label,path in rows), [])
        rc = self.call_main_quiet(argv + ['--output',str(out)])
        self.assertEqual(0, rc)
        self.assertEqual(3,len(json.loads(out.read_text())['pairs']))

    def test_rejects_mismatched_fixture_or_request_config(self):
        good = self.write_receipt("good.json", receipt("good", 10.0, 150.0))
        bad_fixture = self.write_receipt("bad-fixture.json", receipt("bad", 9.0, 166.7, fixture="different"))
        rc = self.call_main_quiet(["--run", f"a={good}", "--run", f"b={bad_fixture}"])
        self.assertNotEqual(0, rc)
        bad_cfg = self.write_receipt("bad-cfg.json", receipt("bad-cfg", 9.0, 166.7, cfg="different"))
        rc = self.call_main_quiet(["--run", f"a={good}", "--run", f"b={bad_cfg}"])
        self.assertNotEqual(0, rc)
        bad_salt = self.write_receipt("bad-salt.json", receipt("bad-salt", 9.0, 166.7, salt_hash="different"))
        rc = self.call_main_quiet(["--run", f"a={good}", "--run", f"b={bad_salt}"])
        self.assertNotEqual(0, rc)


if __name__ == "__main__":
    unittest.main()
