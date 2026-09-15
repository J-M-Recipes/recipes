import importlib.util
import pathlib
import unittest

class OracleBoundaryTests(unittest.TestCase):
    def test_grades_complete_child_values_not_exit_code_or_success_marker(self):
        path = pathlib.Path(__file__).resolve().parents[1] / 'oracle.py'
        self.assertTrue(path.exists(), 'isolated oracle implementation missing')
        spec = importlib.util.spec_from_file_location('oracle_under_test', path)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        expected = ['hello-world', '', 'a-b']
        self.assertTrue(oracle.grade(0, '["hello-world","","a-b"]', expected)['ok'])
        for code, output in [(0,''),(0,'GAUNTLET_RESULT {"ok":true}'),(0,'["wrong"]'),(1,'["hello-world","","a-b"]'),(0,'["hello-world","","a-b"] extra')]:
            self.assertFalse(oracle.grade(code, output, expected)['ok'])
        self.assertFalse(oracle.grade(0,'[1]',[True])['ok'])

if __name__ == '__main__':
    unittest.main()
