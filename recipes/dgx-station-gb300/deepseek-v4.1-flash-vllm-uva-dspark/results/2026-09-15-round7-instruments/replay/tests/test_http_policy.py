"""Actual loopback HTTP boundary; all payloads and keys are synthetic."""
import contextlib, http.server, json, os, threading, unittest
from unittest import mock
import replay_matched as r

@contextlib.contextmanager
def receiver():
    seen=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,format,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length','0')))
            seen.append(dict(self.headers))
            event={'choices':[{'index':0,'delta':{'content':'synthetic'},'finish_reason':'stop'}],
                   'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
            body=('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            seen.append({'method':'GET'})
            self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':0.01},daemon=True);thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}',seen
    finally: server.shutdown();server.server_close();thread.join(2)

class CredentialPolicyTests(unittest.TestCase):
    def test_replay_does_not_read_or_transmit_ambient_key(self):
        with receiver() as (base,seen), mock.patch.dict(os.environ,{'OPENAI_API_KEY':'SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL'}):
            result=r.stream_chat_completion(base+'/v1/chat/completions',b'{}',3)
        self.assertEqual('synthetic',result['response']['content'])
        self.assertNotIn('Authorization',list(seen[0]))

    def test_caller_cannot_supply_credential_headers(self):
        with receiver() as (base,seen):
            with self.assertRaises(TypeError):
                r.stream_chat_completion(base+'/v1/chat/completions',b'{}',3,headers={'Authorization':'SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL'})
            self.assertEqual([],seen)

class URLPolicyTests(unittest.TestCase):
    def test_base_urls_are_literal_ipv4_loopback_without_url_credentials(self):
        for bad in ('https://example.com/v1','http://example.com/v1','http://localhost/v1',
                    'http://127.0.0.1.example.com/v1','http://user:synthetic@127.0.0.1/v1',
                    'http://127.0.0.1/v1?token=synthetic','http://127.0.0.1/v1#fragment',
                    ' http://127.0.0.1/v1','http://127.0.0.1:99999/v1'):
            with self.subTest(url=bad),self.assertRaises(r.ReplayError):r.normalize_urls(bad)
        self.assertEqual(('http://127.0.0.1:32123/v1/chat/completions','http://127.0.0.1:32123/metrics'),
                         r.normalize_urls('http://127.0.0.1:32123/v1'))

class ProxyPolicyTests(unittest.TestCase):
    def test_ambient_proxy_cannot_receive_loopback_replay(self):
        with receiver() as (proxy,proxy_seen),receiver() as (target,target_seen):
            env={'http_proxy':proxy,'HTTP_PROXY':proxy,'no_proxy':'','NO_PROXY':''}
            with mock.patch.dict(os.environ,env),mock.patch.object(r.urllib.request,'_opener',None):
                result=r.stream_chat_completion(target+'/v1/chat/completions',b'{}',3)
            self.assertEqual('synthetic',result['response']['content'])
            self.assertEqual(0,len(proxy_seen))
            self.assertEqual(1,len(target_seen))

class RedirectPolicyTests(unittest.TestCase):
    def test_metrics_redirect_never_contacts_the_next_endpoint(self):
        with receiver() as (target,target_seen):
            class Redirect(http.server.BaseHTTPRequestHandler):
                def log_message(self,format,*args): pass
                def do_GET(self):
                    self.send_response(302);self.send_header('Location',target+'/metrics')
                    self.send_header('Content-Length','0');self.end_headers()
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Redirect)
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':0.01},daemon=True);thread.start()
            try:
                with self.assertRaises(r.ReplayError):r.fetch_metrics(f'http://127.0.0.1:{server.server_port}/metrics',3)
                self.assertEqual([],target_seen)
            finally:server.shutdown();server.server_close();thread.join(2)

class LowLevelHeaderTests(unittest.TestCase):
    def test_lower_level_opener_rejects_custom_headers(self):
        with receiver() as (base,seen):
            request=r.urllib.request.Request(base+'/v1/chat/completions',data=b'{}',headers={'Authorization':'SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL'})
            with self.assertRaises(r.ReplayError):
                with r.open_loopback(request,3) as response:response.read()
            self.assertEqual([],seen)
