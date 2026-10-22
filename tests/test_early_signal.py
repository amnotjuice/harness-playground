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


def write_session_run(directory, start, final, tests, status='improved'):
    """One run: the starting scheduler (iteration 0) and one agent session with timed self-tests."""
    seed_dir = directory / 'candidates/opencode/000_seed'
    session_dir = directory / 'candidates/opencode/001'
    for k, score in enumerate(tests):
        test = session_dir / 'internal_evals' / f'{k:02d}'
        test.mkdir(parents=True)
        # Written out of order on disk; the timestamp decides the order.
        (test / 'eval.json').write_text(json.dumps({'ok': score is not None, 'score': score,
                                                     'started_at_utc': f'2026-10-22T00:{59 - k:02d}:00'}))
    seed_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
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
        write_session_run(out, *session)
        cells.append({'repeat': i, 'output': str(out)})
    (root / 'sweep.json').write_text(json.dumps({'cells': cells}))


class EarlySignalTests(unittest.TestCase):
    def test_best_after_k_follows_time_order_and_keeps_the_start_as_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Time order (reverse of disk order): 190 (worse than start), 170, failed, 150.
            write_sweep(root, [(180, 150, [150, None, 170, 190])])
            row = early_signal.session_rows(root)[0]
            self.assertEqual([row[f'best_after_{k}'] for k in (1, 2, 3, 4, 5)], [180, 170, 170, 150, None])
            self.assertEqual((row['self_tests'], row['valid_self_tests']), (4, 3))
            self.assertFalse(row['final_invalid'])

    def test_exclusions_and_invalid_finals_are_counted_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_sweep(root, [(180, 160, [175, 165]),        # usable
                               (180, None, [170]),            # final invalid -> start score
                               (180, 200, []),                # no self-tests
                               (180, None, [], 'generation_failed')])
            rows = early_signal.session_rows(root)
            summary = early_signal.summarize(rows)
            self.assertEqual(summary['usable'], 2)
            self.assertEqual(summary['excluded'], {'no self-tests': 1, 'generation failed': 1})
            self.assertEqual(summary['final_invalid'], 1)
            self.assertEqual(rows[1]['final_score'], 180)
            self.assertEqual(summary['by_k'][1]['n'], 2)
            self.assertEqual(summary['by_k'][2]['short_sessions'], 1)

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
            # Listed in time order, reversed because the helper writes them newest first. The first
            # self-test is a noisy guide to the final score; by the fifth it matches it exactly.
            sessions = [(180, 177 - 3 * i,
                         list(reversed([180 - 2 * i - (i % 3) * 4, 180 - 3 * i, 179 - 3 * i, 178 - 3 * i, 177 - 3 * i])))
                        for i in range(12)]
            write_sweep(root, sessions)
            rows, summary = early_signal.write_early_signal(root)
            out = root / 'early_signal'
            for name in ('sessions.csv', 'summary.json', 'A_final_vs_early.png', 'B_prediction_by_k.png',
                         'C_further_improvement.png'):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(summary['by_k'][5]['n'], 12)
            self.assertGreater(summary['by_k'][5]['r2'], summary['by_k'][1]['r2'])
            lo, hi = summary['by_k'][1]['r2_ci']
            self.assertLessEqual(lo, hi)


class StartSchedulerTests(unittest.TestCase):
    def test_iteration_zero_evaluates_the_start_scheduler_instead_of_the_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            start = Path(temp) / 'start.py'
            start.write_text((ROOT / 'src/seed.py').read_text() + '\n# a better starting policy\n')
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
