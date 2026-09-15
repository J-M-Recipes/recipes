import json,tempfile,unittest
from pathlib import Path
import gauntlet as g

class ToolFeedbackTests(unittest.TestCase):
    def test_model_gets_actual_result_without_run_specific_launcher_metadata(self):
        task=next(t for t in g.build_tasks() if t.kind=='code_fix')
        class Sandbox:
            def run_tests(self,*a):
                return {'ok':True,'evaluator':{'ok':True,'passed_cases':5},
                        'command':['docker','--name','random-local-run','/private/output/path']}
        class Client:
            n=0;seen=None
            def chat(self,messages,*a):
                self.n+=1
                if self.n==1:
                    m={'content':'','tool_calls':[{'id':'check','type':'function','function':{'name':'run_tests','arguments':'{}'}}]};f='tool_calls'
                else:
                    self.seen=json.loads(messages[-1]['content'])
                    m={'content':json.dumps({'status':'fixed','summary':'fixed'})};f='stop'
                return {'message':m,'finish_reason':f,'usage':{},'timing':{}}
        client=Client()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);g.prepare_task_dir(task,p)
            r=g.GauntletRunner(client,Sandbox()).run_task(task,p,None)
        self.assertNotIn('command',client.seen)
        self.assertTrue(client.seen['ok'])
        self.assertIn('command',r['trace'][0]['result'])
