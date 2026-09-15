import json,unittest
import gauntlet as g
class FinalContractTests(unittest.TestCase):
    def test_structured_success_requires_file_evidence_and_exact_json_types(self):
        task=next(t for t in g.build_tasks() if t.id=='struct-inventory')
        answer=json.dumps(task.expected['final_json'])
        trace=[{'event':'tool_call','name':'read_file','args':{'path':'inventory.csv'},'result':{'ok':True}}]
        self.assertFalse(g.evaluate_task_result(task,answer,[],None)['correct'])
        self.assertTrue(g.evaluate_task_result(task,answer,trace,None)['correct'])
        wrong=json.loads(answer);wrong['answer']['total_qty']=23.0
        self.assertFalse(g.evaluate_task_result(task,json.dumps(wrong),trace,None)['correct'])
    def test_duplicate_json_keys_are_not_strict_json_success(self):
        _,valid,_=g.parse_final_json('{"answer":1,"answer":1}')
        self.assertFalse(valid)
