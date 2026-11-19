"""OpenEvolve arm end to end with a fake model and evaluator (no paid calls, no simulation).

The real OpenEvolve controller and worker process run. The worker is forked so the patches
made here reach it; production runs use Python's default start method.
"""
import argparse
import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')

from src import openevolve_arm
from src.analysis import write_analysis
from src.config import TRACE, get_canonical_base_params

TWEAK = '        # tweak\n'
DIFF = ('<<<<<<< SEARCH\n        avail_ram = pool.avail_ram_pool\n=======\n'
        '        avail_ram = pool.avail_ram_pool\n' + TWEAK + '>>>>>>> REPLACE\n')


def fake_evaluation(path, **kwargs):
    score = 100.0 - Path(path).read_text().count(TWEAK)
    return {'ok': True, 'score': score, 'params': get_canonical_base_params(True), 'scales': [1, 2],
            'trace': 'fake.csv', 'scale_results': {
                str(s): {'ok': True, 'latency': score, 'failures': 0, 'oom_count': 0,
                         'query': {'completed': 1, 'arrived': 1, 'mean_s': 1.0, 'p99_s': 1.0}} for s in (1, 2)}}


def fake_llm(response, cost=.3, error=None):
    def call(directory, prompt, model, timeout, accounting, *, reasoning_effort=None, system=None):
        accounting['model_invocations'] = 1
        (directory / 'prompt.txt').write_text(prompt)
        if error:
            raise error
        accounting.update(input_tokens=10, output_tokens=5, estimated_cost_usd=cost, observed_cost_usd=cost or 0.0)
        assert 'Evaluation objective' in system
        return response
    return call


class ForkedOpenEvolveTest(unittest.TestCase):
    """Forks OpenEvolve's worker so the patches reach it; shared checks on OpenEvolve's own records."""

    @classmethod
    def setUpClass(cls):
        cls.start_method = multiprocessing.get_start_method(allow_none=True)
        multiprocessing.set_start_method('fork', force=True)

    @classmethod
    def tearDownClass(cls):
        multiprocessing.set_start_method(cls.start_method, force=True)

    def assert_openevolve_kept_children(self, args, rows):
        """OpenEvolve's own database, not just the harness records, must contain the children."""
        info = json.loads((args.output / 'openevolve/output/best/best_program_info.json').read_text())
        self.assertEqual(-info['metrics']['combined_score'], min(r['score'] for r in rows if r['score'] is not None))
        self.assertLess(-info['metrics']['combined_score'], rows[0]['score'])


class OpenEvolveArmTests(ForkedOpenEvolveTest):

    def run_arm(self, temp, llm, *, budget=1.0, iterations=12, guard=False):
        args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=iterations, prototype=True,
                                  budget_usd=budget, budget_guard=guard, budget_slack_usd=0, timeout=5)
        summary = {'arms': {}}
        with patch.object(openevolve_arm, 'evaluate_scheduler', side_effect=fake_evaluation), \
             patch.object(openevolve_arm, 'call_llm', side_effect=llm):
            openevolve_arm.run_openevolve_arm('openevolve', args, summary, lambda: None,
                                              model='fake-model', reasoning=None)
        return args, summary['arms']['openevolve']

    def test_records_rows_like_other_arms_and_stops_at_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            args, arm = self.run_arm(temp, fake_llm(DIFF))
            rows = arm['trajectory']
            self.assertEqual(arm['stop_reason'], 'budget')
            # 0.3 per call: calls stop once 1.2 >= 1.0, so 4 paid iterations after the seed.
            self.assertEqual([r['iteration'] for r in rows], [0, 1, 2, 3, 4])
            self.assertAlmostEqual(arm['known_cost_usd'], 1.2)
            self.assertEqual(rows[0]['status'], 'seed')
            self.assertEqual(rows[1]['parent'], rows[0]['path'])
            self.assertEqual(rows[1]['status'], 'improved')
            self.assertEqual(rows[1]['score'], 99.0)
            for row in rows[1:]:
                directory = Path(row['path'])
                self.assertEqual(json.loads((directory / 'meta.json').read_text()), row)
                self.assertEqual(json.loads((directory / 'eval.json').read_text())['score'], row['score'])
                self.assertIsNotNone(row['openevolve']['program_id'])
                self.assertIn(row['parent'], [r['path'] for r in rows])
            self.assertEqual(arm['best_path'], min(rows, key=lambda r: r['score'])['path'])
            self.assertEqual(arm['openevolve']['parallel_evaluations'], 1)
            self.assert_openevolve_kept_children(args, rows)
            # The shared analysis reads the arm unchanged.
            (args.output / 'run_config.json').write_text(json.dumps({'trace': 'fake.csv'}))
            write_analysis(args.output, {'arms': {'openevolve': arm}})

    def test_children_reach_openevolve_when_the_provider_reports_no_tokens(self):
        def llm(directory, prompt, model, timeout, accounting, *, reasoning_effort=None, system=None):
            accounting.update(model_invocations=1, estimated_cost_usd=.1, observed_cost_usd=.1)
            return DIFF
        with tempfile.TemporaryDirectory() as temp:
            args, arm = self.run_arm(temp, llm, budget=None, iterations=3)
            self.assertEqual(len(arm['trajectory']), 4)
            self.assert_openevolve_kept_children(args, arm['trajectory'])

    def test_rejected_response_is_a_failed_generation_not_a_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            _, arm = self.run_arm(temp, fake_llm('no diff here', cost=.25), budget=.5)
            rows = arm['trajectory'][1:]
            self.assertEqual(arm['stop_reason'], 'budget')
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['status'], 'generation_failed')
                self.assertFalse(row['error_fatal'])
                self.assertIn('No valid diffs', row['error'])
                self.assertIsNone(row['score'])

    def test_provider_failure_stops_the_arm_after_recording_the_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            _, arm = self.run_arm(temp, fake_llm(DIFF, error=RuntimeError('provider down')))
            rows = arm['trajectory']
            self.assertEqual(arm['stop_reason'], 'generation_error')
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[1]['error_fatal'])
            self.assertIn('provider down', rows[1]['error'])

    def test_unknown_cost_stops_a_guarded_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            _, arm = self.run_arm(temp, fake_llm(DIFF, cost=None), guard=True)
            self.assertEqual(arm['stop_reason'], 'cost_unknown')
            self.assertEqual(len(arm['trajectory']), 2)
            self.assertFalse(arm['cost_complete'])


def fake_agent(cost=.3, error=None, seen=None):
    """Stands in for generate_opencode: edits the parent in directory/scheduler.py, as a session would."""
    def session(directory, task, model, timeout, accounting, *, prototype=False, trace=None):
        accounting.update(model_invocations=1)
        (directory / 'prompt.txt').write_text(task)
        if seen is not None:
            seen.append({'task': task, 'parent': (directory / 'scheduler.py').read_text(), 'timeout': timeout})
        if error:
            raise error
        path = directory / 'scheduler.py'
        # Mid-file: OpenEvolve's parser strips the file's leading and trailing whitespace.
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n' + TWEAK + rest)
        accounting.update(input_tokens=10, output_tokens=5, estimated_cost_usd=cost, observed_cost_usd=cost)
    return session


class OpenEvolveAgentArmTests(ForkedOpenEvolveTest):
    """OpenEvolve's search with an OpenCode session making each edit."""

    def run_arm(self, temp, session, *, budget=1.0, iterations=12, islands=2):
        args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=iterations, prototype=True,
                                  budget_usd=budget, budget_guard=False, budget_slack_usd=0, timeout=5,
                                  agent_timeout=900, openevolve_islands=islands)
        summary = {'arms': {}}
        with patch.object(openevolve_arm, 'evaluate_scheduler', side_effect=fake_evaluation), \
             patch.object(openevolve_arm, 'generate_opencode', side_effect=session), \
             patch.object(openevolve_arm, 'call_llm', side_effect=AssertionError('no direct calls')):
            openevolve_arm.run_openevolve_arm('openevolve_agent', args, summary, lambda: None,
                                              model='fake-model', reasoning=None, agent=True)
        return args, summary['arms']['openevolve_agent']

    def test_each_iteration_is_an_agent_session_whose_file_becomes_the_child(self):
        seen = []
        with tempfile.TemporaryDirectory() as temp:
            args, arm = self.run_arm(temp, fake_agent(seen=seen))
            rows = arm['trajectory']
            self.assertEqual(arm['stop_reason'], 'budget')
            self.assertEqual([r['iteration'] for r in rows], [0, 1, 2, 3, 4])
            self.assertEqual({r['generator'] for r in rows}, {'openevolve_agent'})
            self.assertEqual(rows[1]['status'], 'improved')
            self.assertEqual(rows[1]['score'], 99.0)
            self.assertLess(min(r['score'] for r in rows), 99.0)
            self.assert_openevolve_kept_children(args, rows)
            # The session starts from OpenEvolve's chosen parent and gets OpenEvolve's prompt plus the note.
            parents = {Path(r['path']).joinpath('scheduler.py').read_text() for r in rows}
            for call in seen:
                self.assertIn(call['parent'], parents)
                self.assertIn('Evaluation objective', call['task'])
                self.assertIn('# Current Program', call['task'])
                self.assertIn('already in scheduler.py', call['task'])
                self.assertNotIn('Follow the output format requested', call['task'])
                self.assertEqual(call['timeout'], 900)
            record = arm['openevolve']
            self.assertFalse(record['diff_based_evolution'])
            self.assertEqual(record['num_islands'], 2)
            self.assertGreaterEqual(record['evaluator_timeout_seconds'], 900 + 2 * 120)

    def test_a_timed_out_session_is_a_failed_attempt_not_a_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            _, arm = self.run_arm(temp, fake_agent(error=TimeoutError('OpenCode session timed out')), budget=None,
                                  iterations=2)
            rows = arm['trajectory'][1:]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['status'], 'generation_failed')
                self.assertFalse(row['error_fatal'])
                self.assertIn('timed out', row['error'])

    def test_a_file_openevolve_would_misparse_is_refused(self):
        self.assertEqual(openevolve_arm.agent_rewrite_response('x = 1\n'), '```python\nx = 1\n\n```')
        with self.assertRaisesRegex(ValueError, 'intact'):
            openevolve_arm.agent_rewrite_response('"""Use ```python fences```."""\nx = 1\n')


if __name__ == '__main__':
    unittest.main()
