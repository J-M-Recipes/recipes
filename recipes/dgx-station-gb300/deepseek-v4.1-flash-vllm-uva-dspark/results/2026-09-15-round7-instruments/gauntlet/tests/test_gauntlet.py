import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path

import gauntlet


class SafeTaskFSTests(unittest.TestCase):
    def test_read_file_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "task"
            root.mkdir()
            (root / "safe.txt").write_text("ok", encoding="utf-8")
            outside = Path(td) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            fs = gauntlet.SafeTaskFS(root, max_read_bytes=100, max_write_bytes=100)
            result = fs.read_file("../outside.txt")

            self.assertFalse(result["ok"])
            self.assertIn("unsafe_path", result["error"])

    def test_read_file_rejects_symlink_even_inside_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "task"
            root.mkdir()
            (root / "real.txt").write_text("ok", encoding="utf-8")
            (root / "link.txt").symlink_to(root / "real.txt")

            fs = gauntlet.SafeTaskFS(root, max_read_bytes=100, max_write_bytes=100)
            result = fs.read_file("link.txt")

            self.assertFalse(result["ok"])
            self.assertIn("symlink", result["error"])

    def test_write_file_creates_safe_relative_file_and_enforces_limit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "task"
            root.mkdir()
            fs = gauntlet.SafeTaskFS(root, max_read_bytes=100, max_write_bytes=5)

            ok = fs.write_file("sub/out.txt", "hello")
            too_large = fs.write_file("sub/big.txt", "toolong")

            self.assertTrue(ok["ok"])
            self.assertEqual((root / "sub" / "out.txt").read_text(encoding="utf-8"), "hello")
            self.assertFalse(too_large["ok"])
            self.assertEqual(too_large["error"], "write_limit")


class CorpusAndEvaluationTests(unittest.TestCase):
    def test_fixed_corpus_has_12_unique_tasks_with_required_mix(self):
        tasks = gauntlet.build_tasks()
        ids = [task.id for task in tasks]
        by_kind = {kind: sum(1 for task in tasks if task.kind == kind) for kind in {task.kind for task in tasks}}

        self.assertEqual(len(tasks), 12)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(by_kind, {"file_chain": 4, "code_fix": 4, "structured": 4})
        for task in tasks:
            if task.kind == "file_chain":
                self.assertGreaterEqual(len(task.expected["required_reads"]), 4)
                self.assertIn(task.expected["required_reads"][0], task.files)

    def test_file_chain_evaluator_accepts_known_correct_nonce_only_after_chain_reads(self):
        task = next(task for task in gauntlet.build_tasks() if task.kind == "file_chain")
        trace = [
            {"event": "tool_call", "name": "read_file", "args": {"path": path}, "result": {"ok": True}, "model_turn": index}
            for index,path in enumerate(task.expected["required_reads"],1)
        ]
        correct = gauntlet.evaluate_task_result(task, json.dumps({"answer": task.expected["answer"]}), trace, None)
        wrong = gauntlet.evaluate_task_result(task, json.dumps({"answer": "wrong"}), trace, None)
        skipped_reads = gauntlet.evaluate_task_result(task, json.dumps({"answer": task.expected["answer"]}), [], None)

        self.assertTrue(correct["correct"])
        self.assertFalse(wrong["correct"])
        self.assertFalse(skipped_reads["correct"])
        self.assertIn("required_read_chain_missing", skipped_reads["reasons"])

    def test_structured_evaluator_accepts_exact_object_and_rejects_malformed_or_prefix_json(self):
        task = next(task for task in gauntlet.build_tasks() if task.kind == "structured")
        trace = [{"event":"tool_call","name":"read_file","args":{"path":p},"result":{"ok":True}} for p in task.files]
        correct = gauntlet.evaluate_task_result(task, json.dumps(task.expected["final_json"]), trace, None)
        wrong = gauntlet.evaluate_task_result(task, json.dumps({"answer": {"not": "it"}}), [], None)
        malformed = gauntlet.evaluate_task_result(task, "Sure: " + json.dumps(task.expected["final_json"]), [], None)

        self.assertTrue(correct["correct"])
        self.assertFalse(wrong["correct"])
        self.assertFalse(malformed["validity"]["final_answer_valid"])
        self.assertIn("malformed_final_json", malformed["validity"]["issues"])


class SandboxRunnerTests(unittest.TestCase):
    def test_docker_command_is_cpu_only_and_mounts_only_task_and_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td) / "task"
            evaluator = Path(td) / "trusted" / "evaluator.py"
            task_dir.mkdir()
            evaluator.parent.mkdir()
            evaluator.write_text("print('x')", encoding="utf-8")

            runner = gauntlet.SandboxRunner("sha256:00d577", evaluator, timeout_s=7)
            cmd = runner.build_command("code-slugify", task_dir)

            joined = " ".join(cmd)
            self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
            for fragment in [
                "--runtime=runc",
                "--network none",
                "--read-only",
                "--cap-drop ALL",
                "--security-opt no-new-privileges",
                "--pids-limit 64",
                "--memory 256m",
                "--cpus 1",
                "--env NVIDIA_VISIBLE_DEVICES=void",
                "--env CUDA_VISIBLE_DEVICES=",
                "--env PYTHONDONTWRITEBYTECODE=1",
            ]:
                self.assertIn(fragment, joined)
            mounts = [cmd[index + 1] for index, part in enumerate(cmd) if part == "-v"]
            self.assertEqual(len(mounts), 2)
            self.assertTrue(all(mount.endswith(":ro") for mount in mounts))
            self.assertNotIn("docker.sock", joined)
            self.assertNotIn(str(Path.home()), joined)
            self.assertNotIn("nvidia", joined.lower().replace("nvidia_visible_devices=void", ""))

    def test_evaluator_result_requires_marker_and_rejects_wrong_marker(self):
        early = gauntlet.parse_evaluator_process_result(0, "plain success\n", "", False, 1000)
        wrong = gauntlet.parse_evaluator_process_result(1, "GAUNTLET_RESULT {\"ok\": false, \"failures\": [\"x\"]}\n", "", False, 1000)
        good = gauntlet.parse_evaluator_process_result(0, "noise\nGAUNTLET_RESULT {\"ok\": true}\n", "", False, 1000)

        self.assertFalse(early["ok"])
        self.assertEqual(early["error"], "missing_evaluator_marker")
        self.assertFalse(wrong["ok"])
        self.assertEqual(wrong["error"], "evaluator_reported_failure")
        self.assertTrue(good["ok"])


class TransportAndLoopTests(unittest.TestCase):
    def test_openai_sse_client_parses_split_tool_call_and_usage(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                seen["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "read_file", "arguments": '{"pa'},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th":"start.txt"}'}}]},
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_args):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = gauntlet.OpenAIClient(f"http://127.0.0.1:{server.server_port}/v1", api_key=None, timeout_s=5)
            response = client.chat([], "model-x", gauntlet.TOOL_DEFS, "low", 1500)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(0, seen["payload"].get("temperature"))
        self.assertEqual(42, seen["payload"].get("seed"))
        self.assertTrue(seen["payload"].get("cache_salt"))
        self.assertTrue(seen["payload"]["stream"])
        self.assertEqual(seen["payload"]["stream_options"], {"include_usage": True})
        self.assertEqual(seen["payload"]["reasoning_effort"], "low")
        self.assertEqual(response["message"]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(response["message"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(response["message"]["tool_calls"][0]["function"]["arguments"], '{"path":"start.txt"}')
        self.assertEqual(response["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        self.assertIsNotNone(response["timing"]["ttft_s"])

    def test_loop_feeds_tool_response_ids_and_scores_once_on_bad_json_final(self):
        task = gauntlet.build_tasks()[0]

        class ScriptedClient:
            def __init__(self):
                self.requests = []
                self.responses = [
                    {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"start.txt"}'}}]}, "finish_reason": "tool_calls", "usage": {}, "timing": {"ttft_s": 0.01, "wall_s": 0.02}, "reasoning": ""},
                    {"message": {"role": "assistant", "content": "not json"}, "finish_reason": "stop", "usage": {}, "timing": {"ttft_s": 0.01, "wall_s": 0.02}, "reasoning": ""},
                ]

            def chat(self, messages, *_args):
                self.requests.append(json.loads(json.dumps(messages)))
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td) / "task"
            raw_dir = Path(td) / "raw"
            gauntlet.prepare_task_dir(task, task_dir)
            runner = gauntlet.GauntletRunner(ScriptedClient(), None, max_turns=4, max_tool_calls=4, task_timeout_s=10)
            result = runner.run_task(task, task_dir, raw_dir)

        second_request = runner.client.requests[1]
        tool_message = next(message for message in second_request if message.get("role") == "tool")
        self.assertEqual(tool_message["tool_call_id"], "tc1")
        self.assertEqual(result["score_count"], 1)
        self.assertFalse(result["correct"])
        self.assertFalse(result["validity"]["final_answer_valid"])

    def test_loop_retains_malformed_call_and_budget_exhaustion_without_retrying_score(self):
        task = gauntlet.build_tasks()[0]

        class BadLoopClient:
            def __init__(self):
                self.requests = []

            def chat(self, messages, *_args):
                self.requests.append(json.loads(json.dumps(messages)))
                return {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "read_file", "arguments": "{"}}]}, "finish_reason": "tool_calls", "usage": {}, "timing": {"ttft_s": 0.0, "wall_s": 0.0}, "reasoning": ""}

        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td) / "task"
            raw_dir = Path(td) / "raw"
            gauntlet.prepare_task_dir(task, task_dir)
            runner = gauntlet.GauntletRunner(BadLoopClient(), None, max_turns=2, max_tool_calls=5, task_timeout_s=10)
            result = runner.run_task(task, task_dir, raw_dir)

        self.assertEqual(result["score_count"], 1)
        self.assertFalse(result["validity"]["transport_protocol_valid"])
        self.assertIn("malformed_tool_arguments", result["validity"]["issues"])
        self.assertIn("budget_exhausted", result["validity"]["issues"])
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main()
