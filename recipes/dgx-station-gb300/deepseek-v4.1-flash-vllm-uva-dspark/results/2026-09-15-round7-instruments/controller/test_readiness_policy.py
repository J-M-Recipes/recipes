"""Real HTTP readiness transport; Docker lifecycle is a separate tested seam."""
import contextlib,http.server,json,os,threading,unittest
from unittest import mock
from typing import cast
import controller as c

@contextlib.contextmanager
def model_endpoint():
    seen=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass
        def do_GET(self):
            seen.append(self.path)
            data=json.dumps({'data':[{'id':'synthetic','max_model_len':42}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    t=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':0.01},daemon=True);t.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',seen
    finally:server.shutdown();server.server_close();t.join(2)

class ReadinessHTTPTests(unittest.TestCase):
    def test_readiness_cannot_be_satisfied_by_ambient_proxy(self):
        class DockerStub:
            contract={'profiles':{'v14':{'id':'synthetic'}}}
            def _inspect_doc(self,cid):return {'State':{'Running':True}}
            def _verify_identity(self,profile,doc):pass
        with model_endpoint() as (origin,origin_seen),model_endpoint() as (proxy,proxy_seen):
            env={'http_proxy':proxy.removesuffix('/v1'),'HTTP_PROXY':proxy.removesuffix('/v1'),'no_proxy':'','NO_PROXY':''}
            with mock.patch.dict(os.environ,env),mock.patch.object(c.urllib.request,'_opener',None):
                result=c.wait_for_readiness(cast(c.DockerController,DockerStub()),'v14','synthetic',42,origin,lambda:None,2)
            self.assertTrue(result['ok']);self.assertEqual([],proxy_seen);self.assertEqual(['/v1/models'],origin_seen)
