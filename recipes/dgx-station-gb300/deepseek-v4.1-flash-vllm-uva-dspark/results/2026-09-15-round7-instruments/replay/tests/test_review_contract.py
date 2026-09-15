"""Regression checks added by parent audit; all fixtures synthetic."""
import dataclasses
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_matched as r

class RequestContractTests(unittest.TestCase):
    def test_effort_does_not_conflict_with_thinking_false(self):
        cfg = r.ReplayConfig(Path('private.json'), 'http://127.0.0.1:30006/v1', 'dsv41-flash-uva', Path('out'), 'test', 4, 400, 'low', 'cache_namespace_initial', 'synthetic')
        turn = r.Turn(0, 't', [{'role': 'user', 'content': 'hello'}], False)
        payload = r.build_request_payload(cfg, turn)
        self.assertEqual('low', payload['reasoning_effort'])
        self.assertEqual(42, payload.get('seed'))
        self.assertNotIn('thinking', payload.get('chat_template_kwargs', {}))

class RealMetricContractTests(unittest.TestCase):
    def test_pinned_metrics_are_token_counters_not_invented_names(self):
        # Names/units from the actual pinned vLLM v1/metrics/loggers.py.
        names = ['spec_decode_num_draft_tokens', 'spec_decode_num_accepted_tokens',
                 'spec_decode_num_drafts', 'prefix_cache_queries', 'prefix_cache_hits',
                 'prompt_tokens', 'generation_tokens']
        before = '\n'.join(f'vllm:{n}_total 0' for n in names)
        after = '\n'.join(f'vllm:{n}_total {i+1}' for i,n in enumerate(names))
        try:
            d = r.metric_deltas(r.parse_metrics_text(before), r.parse_metrics_text(after))
        except r.ReplayError as exc:
            self.fail(str(exc))
        self.assertEqual(4, d['prefix_cache_query_tokens'])
        self.assertEqual(5, d['prefix_cache_hit_tokens'])

class ReasoningBoundaryTests(unittest.TestCase):
    def test_reasoning_only_length_is_a_complete_capture_not_transport_failure(self):
        import io
        events = [
            {'choices':[{'delta':{'reasoning':'working'}, 'finish_reason':None}]},
            {'choices':[{'delta':{}, 'finish_reason':'length'}],
             'usage':{'prompt_tokens':3, 'completion_tokens':4, 'total_tokens':7}},
        ]
        data = ''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
        with patch.object(r.urllib.request,'urlopen',return_value=io.BytesIO(data.encode())):
            try:
                out = r.stream_chat_completion('http://127.0.0.1/v1/chat/completions', b'{}')
            except r.ReplayError as exc:
                self.fail(str(exc))
        self.assertEqual('length', out['finish_reason'])
        self.assertGreaterEqual(out['ttft_s'], 0)
        self.assertIsNone(out['ttfa_s'])
        self.assertEqual('working', out['response']['reasoning'])

if __name__ == '__main__':
    unittest.main()
