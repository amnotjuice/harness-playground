"""Offline tests for fixed-start agent sessions and the early-signal analysis."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import early_signal, run
from src.config import ROOT

START_CODE = (ROOT / 'src/seed.py').read_text()


def write_session_run(directory, start, final, tests, status='improved', first_edit_lines=0, cut_off=False):
    """One run: the starting scheduler (iteration 0) and one agent session with timed self-tests.

    `tests` are scores in time order; the first tested code adds `first_edit_lines` lines.
    """
    seed_dir = directory / 'candidates/opencode/000_seed'
    session_dir = directory / 'candidates/opencode/001'
    seed_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (seed_dir / 'scheduler.py').write_text(START_CODE)
    if cut_off:
        (session_dir / 'stderr.txt').write_text('! permission requested: external_directory (/x/*); auto-rejecting\n')
    for k, score in enumerate(tests):
        # Directory names run opposite to time order; the timestamp decides the order.
        test = session_dir / 'internal_evals' / f'{len(tests) - k:02d}'
        test.mkdir(parents=True)
        edit = ''.join(f'# change {i}\n' for i in range(first_edit_lines if k == 0 else k + 50))
        (test / 'scheduler.py').write_text(START_CODE + edit)
        (test / 'eval.json').write_text(json.dumps({'ok': score is not None, 'score': score,
                                                     'started_at_utc': f'2026-10-27T00:{k:02d}:00'}))
    rows = [{'iteration': 0, 'path': str(seed_dir), 'parent': None, 'score': start, 'status': 'seed',
             'elapsed_seconds': 20},
            {'iteration': 1, 'path': str(session_dir), 'parent': str(seed_dir), 'score': final,
             'status': status, 'elapsed_seconds': 900, 'observed_cost_usd': .3}]
    (directory / 'results.json').write_text(json.dumps({'status': 'complete', 'arms': {
        'opencode': {'trajectory': rows, 'stop_reason': 'iterations'}}}))


def write_sweep(root, sessions):
    cells = []
    for i, session in enumerate(sessions, 1):
        out = root / 'train-simple' / f'rep{i}'
        out.mkdir(parents=True)
        write_session_run(out, *session[:4], **(session[4] if len(session) > 4 else {}))
        cells.append({'repeat': i, 'output': str(out)})
    (root / 'sweep.json').write_text(json.dumps({'cells': cells}))


class EarlySignalTests(unittest.TestCase):
    def test_best_tested_follows_time_order_and_is_not_floored_at_the_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_sweep(root, [(180, 150, [576, None, 230, 190, 150], 'improved', {'first_edit_lines': 12})])
            row = early_signal.session_rows(root)[0]
            self.assertEqual([row[f'best_tested_after_{k}'] for k in range(1, 7)], [576, 576, 230, 190, 150, None])
            self.assertEqual((row['self_tests'], row['valid_self_tests']), (5, 4))
            self.assertEqual(row['lines_changed_before_first_test'], 12)
            self.assertFalse(row['final_invalid'])

    def test_exclusions_and_invalid_finals_are_counted_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_sweep(root, [(180, 160, [175, 165]),                    # usable
                               (180, None, [None, 170]),                  # final invalid; no valid test at k=1
                               (180, 200, []),                            # no self-tests
                               (180, None, [], 'generation_failed'),
                               (180, 200, [190], 'improved', {'cut_off': True})])
            rows = early_signal.session_rows(root)
            summary = early_signal.summarize(rows)
            self.assertEqual(summary['usable'], 2)
            self.assertEqual(summary['excluded'], {'no self-tests': 1, 'generation failed': 1,
                                                   'cut off by a rejected permission': 1})
            self.assertEqual(summary['final_invalid'], 1)
            self.assertEqual(rows[1]['final_score'], 180)
            self.assertEqual((summary['by_k'][1]['n'], summary['by_k'][1]['no_valid_test_yet']), (1, 1))
            self.assertEqual((summary['by_k'][2]['n'], summary['by_k'][3]['fewer_than_k_tests']), (2, 2))
            self.assertIsNone(rows[2]['lines_changed_before_first_test'])

    def test_fit_statistics_on_known_lines(self):
        exact = early_signal.fit([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(exact['slope'], 2)
        self.assertAlmostEqual(exact['intercept'], 1)
        self.assertAlmostEqual(exact['r2'], 1)
        self.assertAlmostEqual(exact['residual_sd'], 0)
        noisy = early_signal.fit([1, 2, 3, 4], [1, 3, 2, 4])
        self.assertAlmostEqual(noisy['slope'], .8)
        self.assertAlmostEqual(noisy['residual_sd'], (1.8 / 2) ** .5)
        self.assertIsNone(early_signal.fit([1, 2], [1, 2]))
        self.assertIsNone(early_signal.fit([5, 5, 5], [1, 2, 3]))

    def test_outputs_and_plots_are_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # The first self-test is a noisy guide to the final score; by the fifth it matches it.
            sessions = [(180, 177 - 3 * i, [180 - 2 * i - (i % 3) * 4, 180 - 3 * i, 179 - 3 * i, 178 - 3 * i,
                                            177 - 3 * i], 'improved', {'first_edit_lines': 10 + 5 * i})
                        for i in range(12)]
            write_sweep(root, sessions)
            _, summary = early_signal.write_early_signal(root)
            out = root / 'early_signal'
            for name in ('sessions.csv', 'summary.json', 'A_final_vs_early.png', 'B_prediction_by_k.png',
                         'D_code_changed_before_first_test.png', 'E_self_tests.png'):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(summary['by_k'][5]['n'], 12)
            self.assertGreater(summary['by_k'][5]['r2'], summary['by_k'][1]['r2'])
            self.assertAlmostEqual(summary['lines_changed_before_first_test']['r2'], 1)
            lo, hi = summary['by_k'][1]['r2_ci']
            self.assertLessEqual(lo, hi)


class StartSchedulerTests(unittest.TestCase):
    def test_iteration_zero_evaluates_the_start_scheduler_instead_of_the_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            start = Path(temp) / 'start.py'
            start.write_text(START_CODE + '\n# a better starting policy\n')
            args = argparse.Namespace(output=Path(temp) / 'out', trace=ROOT / 'scenarios/train/simple.csv',
                                      iterations=0, prototype=True, budget_usd=None, timeout=1,
                                      agent_timeout=2, start_scheduler=start)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'OPENCODE_MODEL': 'fake'}), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 180}):
                run.run_arm('opencode', args, summary, lambda: None)
            first = Path(summary['arms']['opencode']['trajectory'][0]['path']) / 'scheduler.py'
            self.assertIn('# a better starting policy', first.read_text())

    def test_without_the_option_the_seed_is_used(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=ROOT / 'scenarios/train/simple.csv',
                                      iterations=0, prototype=True, budget_usd=None, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            with patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 700}):
                run.run_arm('direct', args, summary, lambda: None)
            first = Path(summary['arms']['direct']['trajectory'][0]['path']) / 'scheduler.py'
            self.assertEqual(first.read_bytes(), (ROOT / 'src/seed.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
