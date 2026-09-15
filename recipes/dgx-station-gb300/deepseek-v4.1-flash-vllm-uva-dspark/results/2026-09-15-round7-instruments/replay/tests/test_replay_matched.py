import contextlib
import importlib.util
import io
import json
import math
import pathlib
import shutil
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "replay_matched.py"


def load_module():
    spec = importlib.util.spec_from_file_location("replay_matched", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fixture_bytes():
    sessions = []
    for i in range(20):
        turns = []
        for j in range(15):
            turns.append({
                "ctx": [{"role": "user", "content": f"public-session-{i:02d}-turn-{j:02d}"}],
                "has_tool": (j % 5 == 0),
            })
        # Deliberately duplicate the source id; loader must not trust it as identity.
        sessions.append({"session": "truncated-duplicate", "turns": turns})
    return json.dumps(sessions, sort_keys=True).encode("utf-8")


class LocalSSEServer:
    def __init__(self, mode="ok", metrics_mode="ok"):
        self.mode = mode
        self.metrics_mode = metrics_mode
        self.lock = threading.Lock()
        self.requests = []
        self.metrics_calls = 0
        self.successful_requests = 0
        self.httpd = None
        self.thread = None
        self.base_url = None

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                with outer.lock:
                    outer.metrics_calls += 1
                    metrics_call = outer.metrics_calls
                    n = outer.successful_requests
                text = outer.metrics_text(n, metrics_call)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(text.encode("utf-8"))

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                with outer.lock:
                    outer.requests.append(payload)
                    request_index = len(outer.requests)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                if outer.mode == "truncated" and request_index == 1:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}],"usage":null}\n\n')
                    self.wfile.flush()
                    return
                if outer.mode == "missing_usage" and request_index == 1:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}],"usage":null}\n\n')
                    self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":null}\n\n')
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                last_content = payload["messages"][-1]["content"]
                if last_content.endswith("19-turn-14"):
                    time.sleep(0.05)
                elif last_content.endswith("00-turn-00"):
                    time.sleep(0.01)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}],"usage":null}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                with outer.lock:
                    outer.successful_requests += 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()

    def metrics_text(self, n, metrics_call):
        if self.metrics_mode == "nan":
            query_tokens = "NaN"
        else:
            query_tokens = str(7 * n)
        if self.metrics_mode == "reset":
            base = 100 if metrics_call == 1 else 1
            n = base
        values = {
            "vllm:spec_decode_num_draft_tokens_total": 10 * n,
            "vllm:spec_decode_num_accepted_tokens_total": 5 * n,
            "vllm:spec_decode_num_drafts_total": 2 * n,
            "vllm:prefix_cache_queries_total": query_tokens,
            "vllm:prefix_cache_hits_total": 3 * n,
            "vllm:prompt_tokens_total": 3 * n,
            "vllm:generation_tokens_total": 5 * n,
        }
        lines = ["# HELP sibling metric should not match"]
        lines.append(f"vllm:prefix_cache_queries_total_extra 999999")
        for name, value in values.items():
            lines.append(f'{name}{{worker="0"}} {value}')
        return "\n".join(lines) + "\n"


class ReplayTestCase(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.case_dir = ROOT / "_test_runs" / self._testMethodName
        shutil.rmtree(self.case_dir, ignore_errors=True)
        self.case_dir.mkdir(parents=True)
        self.fixture = self.case_dir / "fixture.json"
        self.fixture.write_bytes(make_fixture_bytes())

    def tearDown(self):
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def call_main_quiet(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return self.mod.main(args)


class FixtureIdentityTests(ReplayTestCase):
    def test_duplicate_source_sessions_get_entry_hash_identities(self):
        loaded = self.mod.load_fixture_bytes(make_fixture_bytes())
        self.assertEqual(20, len(loaded.sessions))
        self.assertEqual(300, len(loaded.expected_turn_ids))
        identities = [s.identity for s in loaded.sessions]
        self.assertEqual(20, len(set(identities)))
        self.assertTrue(all(identity.startswith(f"entry-{i:03d}-") for i, identity in enumerate(identities)))
        self.assertRegex(loaded.fixture_sha256, r"^[0-9a-f]{64}$")


class ReplayExecutionTests(ReplayTestCase):
    def test_successful_cli_replay_writes_300_turn_summary_uses_wall_not_busy(self):
        out = self.case_dir / "out-success"
        with LocalSSEServer() as server:
            rc = self.call_main_quiet([
                "--fixture", str(self.fixture),
                "--base-url", server.base_url,
                "--model", "public-test-model",
                "--output", str(out),
                "--tag", "test-tag",
                "--workers", "4",
                "--max-tokens", "400",
                "--reasoning-effort", "low",
                "--cache-state", "cache_namespace_initial",
                "--cache-salt", "test-cache-namespace",
            ])
        self.assertEqual(0, rc)
        records = (out / "turn_records.jsonl").read_text().splitlines()
        self.assertEqual(300, len(records))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(300, summary["turn_count"])
        self.assertEqual("cache_namespace_initial", summary["cache_state"])
        self.assertEqual(self.mod.sha256_hex(b"test-cache-namespace"), summary["cache_salt_sha256"])
        self.assertNotIn("test-cache-namespace", json.dumps(summary))
        self.assertEqual(1500, summary["completion_tokens"])
        self.assertAlmostEqual(1500 / summary["wall_s"], summary["aggregate_completion_tok_s"], delta=0.001)
        self.assertNotEqual(summary["aggregate_completion_tok_s"], summary["diagnostic_busy_worker_completion_tok_s"])
        self.assertEqual("nearest-rank sorted[ceil(p*n)-1]", summary["quantile_method"])
        self.assertEqual(20, len(summary["per_session"]))
        self.assertIn("prefix_cache_query_tokens", summary["metrics_delta"])
        self.assertEqual(300, len(server.requests))
        first_payload = server.requests[0]
        self.assertEqual("public-test-model", first_payload["model"])
        self.assertEqual(400, first_payload["max_tokens"])
        self.assertEqual("low", first_payload["reasoning_effort"])
        self.assertEqual("test-cache-namespace", first_payload["cache_salt"])
        self.assertTrue(first_payload["stream"])
        self.assertEqual({"include_usage": True}, first_payload["stream_options"])

    def test_truncated_stream_preserves_partial_evidence_and_fails(self):
        out = self.case_dir / "out-truncated"
        with LocalSSEServer(mode="truncated") as server:
            rc = self.call_main_quiet([
                "--fixture", str(self.fixture),
                "--base-url", server.base_url,
                "--model", "public-test-model",
                "--output", str(out),
                "--tag", "bad-stream",
                "--workers", "1",
                "--max-tokens", "400",
                "--reasoning-effort", "low",
                "--cache-state", "identical_request_repeat",
                "--cache-salt", "test-cache-namespace",
            ])
        self.assertNotEqual(0, rc)
        self.assertTrue((out / "turn_records.jsonl").exists())
        failure = json.loads((out / "failure_summary.json").read_text())
        self.assertGreaterEqual(failure["errors"], 1)
        self.assertLess(failure["ok_turn_count"], 300)
        rows = [json.loads(x) for x in (out / "turn_records.jsonl").read_text().splitlines()]
        self.assertTrue(any((x.get("partial_response") or {}).get("raw_sse_base64") for x in rows))

    def test_missing_usage_fails_closed(self):
        out = self.case_dir / "out-missing-usage"
        with LocalSSEServer(mode="missing_usage") as server:
            rc = self.call_main_quiet([
                "--fixture", str(self.fixture),
                "--base-url", server.base_url,
                "--model", "public-test-model",
                "--output", str(out),
                "--tag", "missing-usage",
                "--workers", "1",
                "--max-tokens", "400",
                "--reasoning-effort", "low",
                "--cache-state", "cache_namespace_initial",
                "--cache-salt", "test-cache-namespace",
            ])
        self.assertNotEqual(0, rc)
        first = json.loads((out / "turn_records.jsonl").read_text().splitlines()[0])
        self.assertEqual("error", first["status"])
        self.assertIn("usage", first["error"])

    def test_nan_metric_and_counter_reset_fail_closed(self):
        for metrics_mode in ("nan", "reset"):
            out = self.case_dir / f"out-{metrics_mode}"
            with LocalSSEServer(metrics_mode=metrics_mode) as server:
                rc = self.call_main_quiet([
                    "--fixture", str(self.fixture),
                    "--base-url", server.base_url,
                    "--model", "public-test-model",
                    "--output", str(out),
                    "--tag", f"metrics-{metrics_mode}",
                    "--workers", "1",
                    "--max-tokens", "400",
                    "--reasoning-effort", "low",
                    "--cache-state", "cache_namespace_initial",
                    "--cache-salt", "test-cache-namespace",
                ])
            self.assertNotEqual(0, rc)
            failure = json.loads((out / "failure_summary.json").read_text())
            self.assertIn("metric", failure["fatal_error"].lower())

    def test_requires_cache_salt(self):
        out = self.case_dir / "out-no-salt"
        with LocalSSEServer() as server:
            rc = self.call_main_quiet([
                "--fixture", str(self.fixture),
                "--base-url", server.base_url,
                "--model", "public-test-model",
                "--output", str(out),
                "--tag", "no-salt",
                "--workers", "1",
                "--max-tokens", "400",
                "--reasoning-effort", "low",
                "--cache-state", "cache_namespace_initial",
            ])
        self.assertNotEqual(0, rc)
        self.assertEqual([], server.requests)

    def test_duplicate_turn_records_are_rejected(self):
        expected = ("a", "b")
        records = [{"status": "ok", "turn_id": "a"}, {"status": "ok", "turn_id": "a"}]
        result = self.mod.validate_turn_record_coverage(expected, records)
        self.assertFalse(result["ok"])
        self.assertEqual(["a"], result["duplicate_turn_ids"])
        self.assertEqual(["b"], result["missing_turn_ids"])


if __name__ == "__main__":
    unittest.main()
