import os,pathlib,sys,unittest
from unittest import mock
import gauntlet as g
R=pathlib.Path(__file__).resolve().parents[2]/'replay'
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tests'))
import importlib
receiver=importlib.import_module('test_http_policy').receiver

class GauntletCredentialTests(unittest.TestCase):
    def test_ambient_key_is_not_transmitted(self):
        with receiver() as (base,seen),mock.patch.dict(os.environ,{'OPENAI_API_KEY':'SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL'}):
            result=g.OpenAIClient(base+'/v1',timeout_s=3).chat([{'role':'user','content':'synthetic'}],'synthetic',[],'low',2)
        self.assertEqual('synthetic',result['message']['content'])
        self.assertNotIn('Authorization',list(seen[0]))

    def test_explicit_credentials_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'credentials'):
            g.OpenAIClient('http://127.0.0.1:30006/v1',api_key='SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL')

    def test_non_loopback_client_cannot_be_constructed(self):
        with self.assertRaises(RuntimeError):
            g.OpenAIClient('http://example.com/v1')
