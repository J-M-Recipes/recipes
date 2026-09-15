import base64,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gauntlet as g
class RawCaptureTests(unittest.TestCase):
    def test_real_decoder_bytes_survive_complete_and_failed_tool_runs(self):
        packet=b'data: {"choices":[{"delta":{"content":"synthetic"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
        for done in (True,False):
            with self.subTest(done=done),tempfile.TemporaryDirectory() as d:
                raw=packet+(b'data: [DONE]\n\n' if done else b'');response=io.BytesIO(raw);response.status=200
                task=next(t for t in g.build_tasks() if t.kind=='structured');root=Path(d);fs=root/'task';fs.mkdir()
                for name,content in task.files.items():g.SafeTaskFS(fs).write_file(name,content)
                client=g.OpenAIClient('http://127.0.0.1:9/v1',api_key=None,timeout_s=5)
                with patch('urllib.request.OpenerDirector.open',return_value=response):
                    g.GauntletRunner(client,None).run_task(task,fs,root/'raw')
                files=list((root/'raw').glob('*.jsonl'));self.assertEqual(1,len(files))
                rows=[json.loads(line) for line in files[0].read_text().splitlines()]
                if done:
                    event=next(e for e in rows if e['event']=='response')['response']
                    self.assertIn('raw_response',event)
                    saved=event['raw_response']
                else:
                    event=next(e for e in rows if e['event']=='transport_error')
                    self.assertIn('partial_response',event)
                    saved=event['partial_response']
                self.assertEqual(raw,base64.b64decode(saved['raw_sse_base64']))
