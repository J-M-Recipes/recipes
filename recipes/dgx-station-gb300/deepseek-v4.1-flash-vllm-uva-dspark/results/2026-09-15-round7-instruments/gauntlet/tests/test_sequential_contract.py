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


    def test_later_rereads_or_unrelated_reads_cannot_repair_a_batch(self):
        task=next(t for t in g.build_tasks() if t.kind=='file_chain')
        required=task.expected['required_reads']
        final=json.dumps({'answer':task.expected['answer']})
        for filler_path in [required[0], 'unrelated.txt']:
            with self.subTest(filler_path=filler_path):
                batch=[{'event':'tool_call','name':'read_file','args':{'path':p},'result':{'ok':True},'model_turn':1} for p in required]
                filler=[{'event':'tool_call','name':'read_file','args':{'path':filler_path},'result':{'ok':True},'model_turn':t} for t in (2,3,4)]
                self.assertFalse(g.evaluate_task_result(task,final,batch+filler,None)['correct'])


    def test_first_access_turn_assignments_remain_decisive_after_rereads(self):
        import itertools
        task=next(t for t in g.build_tasks() if t.kind=='file_chain')
        required=task.expected['required_reads'];final=json.dumps({'answer':task.expected['answer']})
        for turns in itertools.product(range(1,6),repeat=len(required)):
            first=[{'event':'tool_call','name':'read_file','args':{'path':p},'result':{'ok':True},'model_turn':turn} for p,turn in zip(required,turns)]
            later=[dict(e,model_turn=i+10) for i,e in enumerate(first)]
            expected=turns==tuple(sorted(turns)) and len(set(turns))>=4
            self.assertEqual(expected,g.evaluate_task_result(task,final,first+later,None)['correct'],turns)

    def test_failed_reads_do_not_replace_successful_first_access_evidence(self):
        task=next(t for t in g.build_tasks() if t.kind=='file_chain')
        required=task.expected['required_reads'];final=json.dumps({'answer':task.expected['answer']})
        good=[{'event':'tool_call','name':'read_file','args':{'path':p},'result':{'ok':True},'model_turn':i+2} for i,p in enumerate(required)]
        failed=[dict(e,result={'ok':False},model_turn=1) for e in good]
        self.assertTrue(g.evaluate_task_result(task,final,failed+good,None)['correct'])
        premature=[dict(e,model_turn=1) for e in reversed(good)]
        self.assertFalse(g.evaluate_task_result(task,final,premature+good,None)['correct'])
        for invalid_turn in (None,True,0,-1,1.5,'2'):
            broken=[dict(e) for e in good];broken[0]['model_turn']=invalid_turn
            self.assertFalse(g.evaluate_task_result(task,final,broken+good,None)['correct'])
