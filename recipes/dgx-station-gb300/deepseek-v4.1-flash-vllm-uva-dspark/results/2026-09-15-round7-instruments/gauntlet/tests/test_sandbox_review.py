import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import gauntlet as g

class SandboxReviewTests(unittest.TestCase):
    def test_pinned_image_entrypoint_is_overridden_and_sandbox_named(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); (p/'oracle.py').write_text('')
            cmd=g.SandboxRunner('sha256:fixed',p/'oracle.py').build_command('code-slugify',p)
        self.assertIn('--entrypoint',cmd)
        self.assertEqual('python3',cmd[cmd.index('--entrypoint')+1])
        self.assertEqual('never',cmd[cmd.index('--pull')+1])
        self.assertTrue(cmd[cmd.index('--name')+1].startswith('r7-eval-'))
        self.assertEqual(f'{os.getuid()}:{os.getgid()}',cmd[cmd.index('--user')+1])

    def test_timeout_removes_named_cpu_sandbox_and_preserves_partial_bytes(self):
        import subprocess
        calls=[]
        def run(cmd,**kwargs):
            calls.append(cmd)
            if len(calls)==1:
                raise subprocess.TimeoutExpired(cmd,1,output=b'partial',stderr=b'deadline')
            return subprocess.CompletedProcess(cmd,0,'','')
        with tempfile.TemporaryDirectory() as td, patch.object(g.subprocess,'run',side_effect=run):
            result=g.SandboxRunner('sha256:fixed',Path(td)/'oracle.py',timeout_s=1).run_tests('code-slugify',Path(td))
        self.assertFalse(result['ok'])
        self.assertEqual('timeout',result['error'])
        self.assertEqual('partial',result['stdout'])
        name=calls[0][calls[0].index('--name')+1]
        self.assertEqual(['docker','rm','-f',name],calls[1])
