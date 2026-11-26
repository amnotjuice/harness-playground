"""Scheduler code must not be able to read the workload trace (true requirements). Real worker, short run."""
import json, subprocess, sys, unittest
from pathlib import Path
from src.config import ROOT, TRACE, get_canonical_base_params

SEED = (ROOT / 'src/seed.py').read_text()


def run(code, tmp):
    path = Path(tmp) / 'scheduler.py'
    path.write_text(code)
    params = json.dumps(get_canonical_base_params(prototype=True))
    out = subprocess.run([sys.executable, '-m', 'src.worker', str(path), str(TRACE), params], cwd=ROOT,
                         capture_output=True, text=True, timeout=300).stdout
    return json.loads([l for l in out.splitlines() if l.startswith('__RESULT__')][-1][10:])


class TraceGuard(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reading_the_trace_is_refused(self):
        code = SEED.replace('s.waiting_queue = []', f's.waiting_queue = []\n    s.truth = open({str(TRACE)!r}).read()')
        result = run(code, self.tmp.name)
        self.assertFalse(result['ok'])
        self.assertIn('may not read', result['error'])

    def test_append_plus_cannot_read_back(self):
        code = SEED.replace('s.waiting_queue = []', f's.waiting_queue = []\n    s.truth = open({str(TRACE)!r}, "a+")')
        self.assertIn('may not read', run(code, self.tmp.name)['error'])

    def test_file_system_imports_are_refused(self):
        result = run('import os\n' + SEED, self.tmp.name)
        self.assertFalse(result['ok'])
        self.assertIn('may not import', result['error'])

    def test_ordinary_scheduler_still_runs(self):
        result = run('import math\nimport heapq\nfrom collections import deque\n' + SEED, self.tmp.name)
        self.assertTrue(result['ok'])


if __name__ == '__main__':
    unittest.main()
