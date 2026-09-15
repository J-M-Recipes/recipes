import argparse,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gauntlet as g
class SuiteTests(unittest.TestCase):
    def test_full_suite_preserves_failed_task_and_refuses_output_reuse(self):
        path=Path(g.__file__).with_name('run_suite.py')
        self.assertTrue(path.exists(),'suite driver missing')
        spec=importlib.util.spec_from_file_location('suite_under_test',path)
        suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
        def fake_task(runner,task,task_dir,raw_dir):
            return {'task_id':task.id,'kind':task.kind,'success':task.id!='code-slugify',
                    'validity':{'transport_protocol_valid':True},'wall_s':1.0,'ttft_s':0.01}
        with tempfile.TemporaryDirectory() as td:
            args=argparse.Namespace(output=str(Path(td)/'out'),base_url='http://127.0.0.1:1/v1',model='synthetic-test',workers=4,tag='synthetic',cache_salt='synthetic',sandbox_image='sha256:synthetic')
            with patch.object(g.GauntletRunner,'run_task',fake_task),patch.object(g.SandboxRunner,'run_tests',return_value={'ok':True}):
                self.assertEqual(0,suite.run_suite(args))
            s=json.loads((Path(args.output)/'summary.json').read_text())
            self.assertEqual(12,s['task_count']);self.assertEqual(11,s['success_count'])
            self.assertEqual(12,len({x['task_id'] for x in s['results']}))
            with self.assertRaises(FileExistsError): suite.run_suite(args)
