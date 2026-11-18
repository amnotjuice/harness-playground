"""Offline tests for the self-test ablations: score-only feedback and a cap on ./evaluate runs."""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import evaluator, generators, run
from src.config import ROOT


def full_evaluation(ok=True):
    """An evaluation with per-scale details, the 2x scale failing when not ok."""
    scale = {'ok': True, 'latency': 30.0, 'failures': 12, 'oom_count': 12, 'batch': {'completed': 5, 'arrived': 9}}
    results = {'1': dict(scale), '2': dict(scale) if ok else {'ok': False, 'error': 'subprocess timeout'}}
    return {'ok': ok, 'score': 30.0 if ok else None, 'scale_results': results, 'scales': [1, 2],
            'params': {'cpus_per_pool': 64, 'ram_gb_per_pool': 500}}


class EvaluatorAblationTests(unittest.TestCase):
    def run_main(self, args, evaluation):
        printed = io.StringIO()
        with patch.object(evaluator, 'evaluate_scheduler', return_value=evaluation), \
             contextlib.redirect_stdout(printed):
            code = evaluator.main(args)
        return code, printed.getvalue()

    def test_score_feedback_hides_per_scale_results_from_the_agent_but_archives_them(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'scheduler.py').write_text('policy')
            archive, copies = base / 'internal_evals', base / 'work/evals'
            args = ['--scheduler', str(base / 'scheduler.py'), '--output-dir', str(archive),
                    '--copy-to', str(copies), '--feedback', 'score']
            code, printed = self.run_main(args, full_evaluation())
            self.assertEqual(code, 0)
            self.assertNotIn('oom_count', printed)
            self.assertNotIn('latency', printed)
            self.assertEqual(json.loads((copies / '1/eval.json').read_text()), {'ok': True, 'score': 30.0})
            archived = json.loads(next(archive.glob('*/eval.json')).read_text())
            self.assertEqual(archived['scale_results']['1']['oom_count'], 12)

    def test_score_feedback_still_reports_which_scale_failed_and_why(self):
        view = evaluator.agent_view(full_evaluation(ok=False), 'score')
        self.assertEqual(view, {'ok': False, 'score': None, 'failed_scales': {'2': 'subprocess timeout'}})
        self.assertIs(evaluator.agent_view(full_evaluation(), 'full')['scale_results']['1']['oom_count'], 12)

    def test_run_cap_refuses_without_evaluating_and_counts_outside_the_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'scheduler.py').write_text('policy')
            archive, copies = base / 'internal_evals', base / 'work/evals'
            args = ['--scheduler', str(base / 'scheduler.py'), '--output-dir', str(archive),
                    '--copy-to', str(copies), '--max-runs', '2']
            outputs = [self.run_main(args, full_evaluation()) for _ in range(2)]
            self.assertIn('self-test 1 of 2', outputs[0][1])
            self.assertIn('self-test 2 of 2', outputs[1][1])
            # Deleting the agent's own copies does not reset the count.
            for path in copies.glob('*/*'):
                path.unlink()
            with patch.object(evaluator, 'evaluate_scheduler') as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(evaluator.main(args), 3)
            evaluate.assert_not_called()
            self.assertIn('Self-test limit reached', printed.getvalue())
            self.assertEqual(len(list(archive.glob('*/eval.json'))), 2)

    def test_launcher_passes_ablation_flags_only_when_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'src').mkdir()
            (root / 'src/__init__.py').write_text('')
            (root / 'src/evaluator.py').write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            work = root / 'candidate/workspace'
            work.mkdir(parents=True)
            for kwargs, extra in (({}, []), ({'feedback': 'score', 'max_self_tests': 3},
                                             ['--feedback', 'score', '--max-runs', '3'])):
                with patch.object(generators, 'ROOT', root):
                    launcher = generators.write_evaluation_launcher(work, **kwargs)
                argv = json.loads(subprocess.run([str(launcher)], capture_output=True, text=True, check=True).stdout)
                self.assertEqual(argv[argv.index('--copy-to') + 2:], extra)


class PromptAblationTests(unittest.TestCase):
    def test_task_text_states_each_limit_only_when_it_applies(self):
        plain = generators.render_opencode_prompt('task', timeout=60)
        self.assertNotIn('aggregate score, plus', plain)
        self.assertNotIn('at most', plain)
        score = generators.render_opencode_prompt('task', timeout=60, feedback='score')
        self.assertIn('only whether the scheduler is eligible and its aggregate score', score)
        capped = generators.render_opencode_prompt('task', timeout=60, max_self_tests=3)
        self.assertIn('at most 3 times', capped)
        self.assertIn('at most 1 time in', generators.render_opencode_prompt('task', timeout=60, max_self_tests=1))

    def test_score_feedback_also_hides_the_parent_diagnostics_in_the_prompt(self):
        evaluation = dict(full_evaluation(), trace='hard.csv',
                          params={'cpus_per_pool': 64, 'ram_gb_per_pool': 500, 'num_pools': 1, 'duration': 3600,
                                  'ticks_per_second': 100, 'max_job_seconds': 360, 'subprocess_timeout': 180,
                                  'multi_operator_containers': False})
        full = generators.build_prompt('code', evaluation)
        score = generators.build_prompt('code', evaluation, 'score')
        self.assertIn('OOMs: 12', full)
        parent_section = score.split('## Parent evaluation')[1]
        self.assertNotIn('OOMs', parent_section)
        self.assertNotIn('Weighted latency', parent_section)
        self.assertIn('aggregate score: 30.0000 s', score)
        self.assertIn('Per-scale results are not shown', score)
        # The objective and pool sizes are the task definition and stay.
        self.assertIn('64 CPUs / 500 GB', score)


class RunAblationTests(unittest.TestCase):
    def test_agent_arm_receives_the_ablation_and_its_prompt_hides_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=ROOT / 'scenarios/train/simple.csv',
                                      iterations=1, prototype=True, budget_usd=None, timeout=1, agent_timeout=2,
                                      self_test_feedback='score', max_self_tests=3)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'OPENCODE_MODEL': 'fake'}), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                 patch.object(run, 'build_prompt', return_value='task') as prompt, \
                 patch.object(run, 'generate_opencode') as agent:
                run.run_arm('opencode', args, summary, lambda: None)
            self.assertEqual(prompt.call_args.args[2], 'score')
            self.assertEqual((agent.call_args.kwargs['feedback'], agent.call_args.kwargs['max_self_tests']),
                             ('score', 3))

    def test_ablation_flags_are_refused_outside_the_agent_arm(self):
        for flags in (['--self-test-feedback', 'score'], ['--max-self-tests', '3']):
            result = subprocess.run([sys.executable, '-m', 'src.run', '--mode', 'direct', '--iterations', '0',
                                     '--prototype', *flags], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('apply only to --mode opencode', result.stderr)


if __name__ == '__main__':
    unittest.main()
