import tempfile,time,unittest
from pathlib import Path
import gauntlet as g
class LoopDeadlineTests(unittest.TestCase):
    def test_token_budget_exhaustion_is_separate_from_correct_answer(self):
        import json
        task=next(t for t in g.build_tasks() if t.id=='struct-shipments')
        class Client:
            n=0
            def chat(self,*a):
                self.n+=1
                if self.n==1:
                    message={'content':'','tool_calls':[{'id':f'read_source_{i}','type':'function','function':{'name':'read_file','arguments':json.dumps({'path':name})}} for i,name in enumerate(task.files)]}
                    reason='tool_calls'
                else:
                    message={'content':json.dumps(task.expected['final_json'])};reason='length'
                return {'message':message,'finish_reason':reason,'usage':{},'timing':{}}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);g.prepare_task_dir(task,p)
            r=g.GauntletRunner(Client(),None).run_task(task,p,None)
        self.assertTrue(r['correct'])
        self.assertFalse(r['success'])
        self.assertIn('token_budget_exhausted',r['validity']['issues'])

    def test_transport_failure_is_retained_as_invalid_task_not_lost(self):
        class Broken:
            def chat(self,*a): raise RuntimeError('synthetic transport failure')
        task=g.build_tasks()[0]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);g.prepare_task_dir(task,p/'task')
            try: r=g.GauntletRunner(Broken(),None).run_task(task,p/'task',p/'raw')
            except RuntimeError as exc: self.fail(str(exc))
            self.assertFalse(r['success'])
            self.assertIn('transport_error',r['validity']['issues'])
            self.assertEqual(1,r['score_count'])
    def test_late_answer_cannot_pass_task_deadline(self):
        class Late:
            def chat(self,*a):
                time.sleep(.025)
                return {'message':{'content':'{}'},'finish_reason':'stop','usage':{},'timing':{}}
        task=g.build_tasks()[0]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);g.prepare_task_dir(task,p)
            r=g.GauntletRunner(Late(),None,task_timeout_s=.005).run_task(task,p,None)
        self.assertIn('task_timeout',r['validity']['issues'])
