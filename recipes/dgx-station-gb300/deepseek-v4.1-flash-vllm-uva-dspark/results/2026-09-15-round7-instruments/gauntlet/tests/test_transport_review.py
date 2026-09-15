import io,json,unittest
from unittest.mock import patch
import gauntlet as g
class TransportReviewTests(unittest.TestCase):
    def test_incomplete_sse_cannot_be_a_successful_chat(self):
        event={'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
        body=('data: '+json.dumps(event)+'\n\n').encode()
        with patch.object(g.urllib.request,'urlopen',return_value=io.BytesIO(body)):
            with self.assertRaises(RuntimeError):
                g.OpenAIClient('http://127.0.0.1:1/v1').chat([], 'test', [], 'low', 1500)
