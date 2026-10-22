"""Offline tests for repeated runs, held-out scoring and sweep aggregation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import holdout, run, sweep
from src.analysis import write_sweep_analysis
from src.config import ROOT


def write_run(directory, trajectories, *, trace=ROOT / 'scenarios/train/hard.csv', budget=1):
    """A finished run: each arm's trajectory is (score, best_score, cost) per iteration."""
    directory.mkdir(parents=True)
    (directory / 'run_config.json').write_text(json.dumps(
        {'trace_source': str(trace), 'prototype': True, 'budget_usd_per_arm': budget}))
    arms = {}
    for arm, steps in trajectories.items():
        rows = []
        for iteration, (score, best, cost) in enumerate(steps):
            path = directory / 'candidates' / arm / f'{iteration:03d}'
            path.mkdir(parents=True)
            (path / 'scheduler.py').write_text(f'# {arm} {iteration}\n')
            best_iteration = next(i for i, step in enumerate(steps) if step[0] == best)
            rows.append({'iteration': iteration, 'score': score, 'best_score': best,
                         'best_path': str(directory / 'candidates' / arm / f'{best_iteration:03d}'),
                         'observed_cost_usd': cost, 'estimated_cost_usd': cost,
                         'elapsed_seconds': 1, 'status': 'improved'})
        arms[arm] = {'trajectory': rows, 'stop_reason': 'budget'}
    (directory / 'results.json').write_text(json.dumps({'status': 'complete', 'arms': arms}))


class SnapshotTests(unittest.TestCase):
    def test_train_and_test_traces_with_one_name_stay_distinct(self):
        train = run.snapshot_trace_name(ROOT / 'scenarios/train/hard.csv')
        test = run.snapshot_trace_name(ROOT / 'scenarios/test/hard.csv')
        self.assertEqual(train, 'scenarios/train/hard.csv')
        self.assertEqual(test, 'scenarios/test/hard.csv')
        self.assertEqual(run.snapshot_trace_name('/elsewhere/custom.csv'), 'scenarios/custom.csv')


class HoldoutTests(unittest.TestCase):
    def test_matching_trace_for_split_folders_and_canonical_names(self):
        self.assertEqual(holdout.matching_test_trace(ROOT / 'scenarios/train/medium.csv'),
                         ROOT / 'scenarios/test/medium.csv')
        self.assertEqual(holdout.matching_test_trace(ROOT / 'scenarios/bench_canonical_train.csv'),
                         ROOT / 'scenarios/bench_canonical_test.csv')
        with self.assertRaisesRegex(ValueError, '--test-trace'):
            holdout.matching_test_trace('/elsewhere/custom.csv')

    def test_chain_is_scored_relative_to_seed_and_cached_by_scheduler_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            write_run(directory, {'opencode': [(10, 10, 0), (8, 8, .3), (9, 8, .3), (5, 5, .3)]})
            test_scores = {'# opencode 0\n': 20, '# opencode 1\n': 18, '# opencode 3\n': 16}
            calls = []
            def evaluate(path, *, prototype, trace):
                calls.append((path, prototype, trace))
                return {'ok': True, 'score': test_scores[path.read_text()]}
            report = holdout.score_run(directory, evaluate=evaluate)
            chain = report['arms']['opencode']['chain']
            self.assertEqual([e['iteration'] for e in chain], [0, 1, 3])
            self.assertEqual(calls[0][2], ROOT / 'scenarios/test/hard.csv')
            self.assertTrue(all(prototype for _, prototype, _ in calls))
            self.assertAlmostEqual(chain[-1]['train_ratio'], .5)
            self.assertAlmostEqual(chain[-1]['test_ratio'], .8)
            self.assertAlmostEqual(chain[-1]['generalization_gap'], .3)
            holdout.score_run(directory, evaluate=evaluate)
            self.assertEqual(len(calls), 3)

    def test_failed_test_evaluation_leaves_ratio_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            write_run(directory, {'direct': [(10, 10, 0), (8, 8, .3)]})
            def evaluate(path, **_):
                seed = path.parent.name == '000'
                return {'ok': seed, 'score': 20 if seed else None}
            chain = holdout.score_run(directory, evaluate=evaluate)['arms']['direct']['chain']
            self.assertIsNone(chain[-1]['test_ratio'])
            self.assertIsNone(chain[-1]['generalization_gap'])


class SweepTests(unittest.TestCase):
    def test_cells_forward_arguments_and_existing_results_are_never_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'sweep'
            traces = [ROOT / 'scenarios/train/simple.csv', ROOT / 'scenarios/test/simple.csv']
            write_run(output / 'train-simple/rep1', {'direct': [(10, 10, 0)]})
            commands = []
            def runner(command, cwd):
                commands.append(command)
                return SimpleNamespace(returncode=0)
            manifest = sweep.run_sweep(output, traces, 2, 'both', ['--budget-usd', '2'], runner=runner)
            self.assertEqual(len(commands), 3)
            self.assertNotIn(str(output / 'train-simple/rep1'), [c[c.index('--output') + 1] for c in commands])
            self.assertIn(str(output / 'test-simple/rep2'), [c[c.index('--output') + 1] for c in commands])
            self.assertTrue(all(c[-2:] == ['--budget-usd', '2'] for c in commands))
            self.assertEqual([c['status'] for c in manifest['cells']],
                             ['complete', 'not_started', 'not_started', 'not_started'])
            saved = json.loads((output / 'sweep.json').read_text())
            self.assertEqual(saved['repeats'], 2)

    def test_parallel_cells_use_separate_state_slots_and_logs(self):
        import threading
        import time
        from src.generators import opencode_environment
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'sweep'
            active, peak, states = [0], [0], []
            lock = threading.Lock()
            def runner(command, cwd, env, stdout, stderr):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                    states.append(env['HARNESS_OPENCODE_STATE'])
                stdout.write('run output\n')
                time.sleep(.05)
                with lock:
                    active[0] -= 1
                return SimpleNamespace(returncode=0)
            manifest = sweep.run_sweep(output, [ROOT / 'scenarios/train/simple.csv'], 4, 'all', [],
                                       jobs=2, runner=runner)
            self.assertEqual(peak[0], 2)
            self.assertEqual(set(states), {str(ROOT / '.tools/state/job0'), str(ROOT / '.tools/state/job1')})
            self.assertEqual((output / 'train-simple/rep3/sweep.log').read_text(), 'run output\n')
            self.assertEqual(len(manifest['cells']), 4)
            self.assertTrue(all('exit_code' in c for c in manifest['cells']))
            with patch.dict('os.environ', {'HARNESS_OPENCODE_STATE': states[0]}):
                self.assertEqual(opencode_environment()['XDG_DATA_HOME'], states[0] + '/data')

    def test_cli_rejects_per_cell_flags_after_separator(self):
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            sweep.main(['--repeats', '1', '--', '--trace', 'x.csv'])

    def test_aggregation_reports_median_ratios_and_holdout_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'sweep'
            trace = ROOT / 'scenarios/train/hard.csv'
            for repeat, final in [(1, 5), (2, 6), (3, 8)]:
                write_run(output / f'train-hard/rep{repeat}',
                          {'direct': [(10, 10, 0), (final, final, .5)]})
            # Past budget, so the endpoint stays at the seed for this repeat.
            write_run(output / 'train-hard/rep4', {'direct': [(10, 10, 0), (2, 2, 1.5)]})
            cells = [{'trace': str(trace), 'repeat': r, 'output': str(output / f'train-hard/rep{r}')}
                     for r in range(1, 5)]
            (output / 'sweep.json').write_text(json.dumps({'cells': cells}))
            holdout.score_run(output / 'train-hard/rep1',
                              evaluate=lambda path, **_: {'ok': True, 'score': 20 if path.parent.name == '000' else 14})
            rows = write_sweep_analysis(output)
            self.assertEqual([r['train_ratio'] for r in rows], [.5, .6, .8, 1.0])
            self.assertAlmostEqual(rows[0]['generalization_gap'], .2)
            self.assertIsNone(rows[1]['test_ratio'])
            report = (output / 'report.md').read_text()
            self.assertIn('| train-hard | direct | 4 | 4 | 0.700 [0.500, 1.000] | 0.700 [0.700, 0.700] |', report)
            self.assertTrue((output / 'aggregate.csv').exists())


if __name__ == '__main__':
    unittest.main()
