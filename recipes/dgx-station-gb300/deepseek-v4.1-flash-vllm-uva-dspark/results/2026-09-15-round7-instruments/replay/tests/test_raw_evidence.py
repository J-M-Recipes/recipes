import base64,io,json,unittest
from unittest.mock import patch
import replay_matched as r

class RawEvidenceTests(unittest.TestCase):
    def test_complete_and_truncated_streams_preserve_exact_sse_bytes(self):
        event={'choices':[{'delta':{'content':'π'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}}
        prefix=('data: '+json.dumps(event,ensure_ascii=False)+'\n\n').encode()
        for suffix in (b'',b'data: [DONE]\n\n'):
            raw=prefix+suffix
            with self.subTest(complete=bool(suffix)),patch.object(r.urllib.request.OpenerDirector,'open',return_value=io.BytesIO(raw)):
                if suffix:
                    result=r.stream_chat_completion('http://127.0.0.1/v1/chat/completions',b'{}')['response']
                else:
                    with self.assertRaises(r.ReplayError) as caught:
                        r.stream_chat_completion('http://127.0.0.1/v1/chat/completions',b'{}')
                    self.assertTrue(hasattr(caught.exception,'partial_response'),'truncated content was lost')
                    result=caught.exception.partial_response
                self.assertEqual('π',result['content'])
                self.assertEqual(raw,base64.b64decode(result.get('raw_sse_base64','')))
