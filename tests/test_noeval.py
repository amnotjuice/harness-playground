"""Offline tests for the OpenCode arm without self-evaluation and the access audit."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import generators, run
from src.analysis import sweep_rows
from src.config import ROOT


def tool_event(command):
    return json.dumps({'type': 'tool_use', 'part': {'tool': 'bash', 'state': {
        'input': {'command': command}, 'output': 'python -m src.evaluator was mentioned here'}}})


class NoEvalTests(unittest.TestCase):
    def test_prompt_forbids_evaluation_and_omits_launcher(self):
        rendered = generators.render_opencode_prompt('task', timeout=45, self_eval=False)
        self.assertNotIn('./evaluate', rendered)
        self.assertIn('do not run the simulator', rendered)
        self.assertIn('45-second', rendered)
        self.assertIn('./evaluate', generators.render_opencode_prompt('task', timeout=45))

    def test_agent_without_evaluator_gets_no_launcher_and_audit_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cli = base / 'fake-opencode'
            # A fake agent that edits the policy and then tries to reach the evaluator.
            cli.write_text('#!/usr/bin/env python3\nimport json, os\nfrom pathlib import Path\n'
                           'assert not Path("evaluate").exists()\n'
                           'p = Path("scheduler.py"); p.write_text(p.read_text() + "\\n# edited\\n")\n'
                           f'print({tool_event("python -m src.evaluator --scheduler scheduler.py")!r})\n')
            cli.chmod(0o755)
            directory = base / 'candidate'
            directory.mkdir()
            (directory / 'scheduler.py').write_text((ROOT / 'src/seed.py').read_text())
            accounting = {}
            with patch('src.generators.detect_opencode', return_value=(str(cli), '--model --format')):
                generators.generate_opencode(directory, 'task', 'fake', 5, accounting, self_eval=False)
            self.assertIn('# edited', (directory / 'scheduler.py').read_text())
            self.assertFalse((directory / 'workspace/evaluate').exists())
            self.assertFalse(json.loads((directory / 'command.json').read_text())['self_eval'])
            self.assertEqual(accounting['agent_access']['markers'], ['src.evaluator'])

    def test_audit_reads_tool_inputs_only_and_counts_self_evaluation_separately(self):
        stdout = '\n'.join([
            tool_event('./evaluate'), tool_event('./evaluate'),
            tool_event(f'cat {ROOT}/scenarios/train/hard.csv'),
            json.dumps({'type': 'text', 'part': {'text': 'I will not import eudoxia'}}),
            'not json'])
        audit = generators.audit_agent_access(stdout)
        self.assertEqual(audit['self_eval_calls'], 2)
        self.assertEqual(audit['markers'], sorted(['scenarios/', str(ROOT)]))
        self.assertEqual(len(audit['tool_calls']), 1)
        self.assertEqual(generators.audit_agent_access(tool_event('ls'))['markers'], [])

    def test_modes_select_arms_and_noeval_arm_runs_without_self_evaluation(self):
        self.assertEqual(run.MODES['all'], ['direct', 'opencode', 'opencode_noeval'])
        self.assertEqual(run.MODES['both'], ['direct', 'opencode'])
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=ROOT / 'scenarios/train/simple.csv',
                                      iterations=1, prototype=True, budget_usd=None, timeout=1,
                                      agent_timeout=2)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'OPENCODE_MODEL': 'fake'}), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'generate_opencode') as agent:
                run.run_arm('opencode_noeval', args, summary, lambda: None)
                run.run_arm('opencode', args, summary, lambda: None)
            self.assertEqual([c.kwargs['self_eval'] for c in agent.call_args_list], [False, True])
            self.assertEqual(summary['arms']['opencode_noeval']['generator'], 'opencode_noeval')

    def test_sweep_rows_count_attempts_that_reached_the_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            run_dir = output / 'train-hard/rep1'
            run_dir.mkdir(parents=True)
            (run_dir / 'run_config.json').write_text(json.dumps({'budget_usd_per_arm': None}))
            def row(iteration, markers):
                return {'iteration': iteration, 'score': 10, 'best_score': 10, 'best_path': 'seed',
                        'status': 'seed', 'estimated_cost_usd': 0, 'elapsed_seconds': 1,
                        'generation_usage': {'agent_access': {'markers': markers}}}
            (run_dir / 'results.json').write_text(json.dumps({'status': 'complete', 'arms': {
                'opencode_noeval': {'trajectory': [row(0, []), row(1, ['eudoxia']), row(2, [])],
                                    'stop_reason': 'iterations'}}}))
            (output / 'sweep.json').write_text(json.dumps({'cells': [{'repeat': 1, 'output': str(run_dir)}]}))
            self.assertEqual(sweep_rows(output)[0]['attempts_with_repo_access'], 1)


if __name__ == '__main__':
    unittest.main()
