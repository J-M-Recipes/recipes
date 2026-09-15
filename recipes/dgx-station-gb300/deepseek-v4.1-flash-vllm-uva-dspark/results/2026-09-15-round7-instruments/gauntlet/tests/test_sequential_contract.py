import json,unittest
import gauntlet as g
class SequentialContractTests(unittest.TestCase):
    def test_one_batch_is_not_four_sequential_model_turns(self):
        task=next(t for t in g.build_tasks() if t.kind=='file_chain')
        trace=[{'event':'tool_call','name':'read_file','args':{'path':p},'result':{'ok':True},'model_turn':1} for p in task.expected['required_reads']]
        final=json.dumps({'answer':task.expected['answer']})
        self.assertFalse(g.evaluate_task_result(task,final,trace,None)['correct'])
        for i,e in enumerate(trace,1):e['model_turn']=i
        self.assertTrue(g.evaluate_task_result(task,final,trace,None)['correct'])
