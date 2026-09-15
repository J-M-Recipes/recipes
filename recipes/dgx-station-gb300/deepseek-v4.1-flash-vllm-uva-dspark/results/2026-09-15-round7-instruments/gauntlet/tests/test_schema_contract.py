import json,unittest
import gauntlet as g
class SchemaContractTests(unittest.TestCase):
    def test_schema_validity_is_separate_from_answer_correctness(self):
        code=next(t for t in g.build_tasks() if t.kind=='code_fix')
        extra=g.evaluate_task_result(code,json.dumps({'status':'fixed','summary':'x','extra':1}),[],{'ok':True})
        self.assertFalse(extra['correct'])
        self.assertTrue(extra['validity']['final_answer_valid'])
        self.assertFalse(extra['validity']['schema_valid'])
        chain=next(t for t in g.build_tasks() if t.kind=='file_chain')
        wrong=g.evaluate_task_result(chain,json.dumps({'answer':'wrong'}),[],None)
        self.assertFalse(wrong['correct'])
        self.assertTrue(wrong['validity']['schema_valid'])
        for task in [t for t in g.build_tasks() if t.kind=='structured']:
            expected=task.expected['final_json']
            valid=g.evaluate_task_result(task,json.dumps(expected),[],None)
            self.assertTrue(valid['validity']['schema_valid'])
            invalid=g.evaluate_task_result(task,json.dumps({'unrequested':True}),[],None)
            self.assertFalse(invalid['validity']['schema_valid'])
