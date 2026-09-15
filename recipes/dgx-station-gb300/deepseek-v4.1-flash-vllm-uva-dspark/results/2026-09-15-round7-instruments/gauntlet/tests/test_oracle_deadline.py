import tempfile,time,unittest
from pathlib import Path
import gauntlet as g
class OracleDeadlineTests(unittest.TestCase):
    def test_late_oracle_cannot_turn_into_under_budget_success(self):
        class Client:
            def chat(self,*args):return {'message':{'content':'{"status":"fixed","summary":"synthetic"}'},'finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},'timing':{'ttft_s':.001,'wall_s':.001}}
        class Sandbox:
            def run_tests(self,*args):time.sleep(.025);return {'ok':True}
        task=next(t for t in g.build_tasks() if t.kind=='code_fix')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,value in task.files.items():g.SafeTaskFS(root).write_file(name,value)
            result=g.GauntletRunner(Client(),Sandbox(),task_timeout_s=.01).run_task(task,root,None)
        self.assertFalse(result['success'])
        self.assertIn('task_timeout',result['validity']['issues'])
