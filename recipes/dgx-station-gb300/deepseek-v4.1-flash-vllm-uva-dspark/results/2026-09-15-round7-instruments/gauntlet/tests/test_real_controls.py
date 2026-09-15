"""Real Linux/Docker controls; never run candidate Python on the host."""
import os
import tempfile
import unittest
from pathlib import Path
import gauntlet as g

SOLUTIONS = {
'code-slugify': "import re\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n",
'code-intervals': "def merge_intervals(intervals):\n    out=[]\n    for a,b in sorted(intervals):\n        if out and a <= out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out\n",
'code-totals': "import csv,io\ndef totals_by_category(text):\n    out={}\n    for row in csv.DictReader(io.StringIO(text)):\n        k=row['category']; out[k]=out.get(k,0)+int(row['amount'])\n    return out\n",
'code-brackets': "def is_balanced(text):\n    s=[]; pairs={')':'(',']':'[','}':'{'}\n    for c in text:\n        if c in '([{': s.append(c)\n        elif c in pairs:\n            if not s or s.pop()!=pairs[c]: return False\n    return not s\n",
}

@unittest.skipUnless(os.getenv('R7_SANDBOX_IMAGE'), 'explicit real Docker control run only')
class DockerOracleControls(unittest.TestCase):
    def test_real_positive_wrong_early_exit_and_forged_marker_controls(self):
        runner=g.SandboxRunner(os.environ['R7_SANDBOX_IMAGE'],Path(g.__file__).with_name('oracle.py'))
        for task in g.build_tasks():
            if task.kind!='code_fix': continue
            for label,code,expected in [
                ('correct',SOLUTIONS[task.id],True),
                ('wrong',next(iter(task.files.values())),False),
                ('early-exit','import os\nos._exit(0)\n',False),
                ('forged-marker','import os\nprint(\'GAUNTLET_RESULT {"ok":true}\',flush=True)\nos._exit(0)\n',False),
            ]:
                with self.subTest(task=task.id,control=label),tempfile.TemporaryDirectory() as td:
                    p=Path(td);(p/task.expected['module']).write_text(code)
                    result=runner.run_tests(task.id,p)
                    self.assertEqual(expected,result['ok'],str(result))

if __name__=='__main__': unittest.main()
