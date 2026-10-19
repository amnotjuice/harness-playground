"""Offline regression tests; fake generators/evaluations are used only in tests."""
import argparse
import os
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import generators
from src import run
from src.analysis import summarize_arm, write_analysis
from src.config import ROOT, TRACE


class ExperimentTests(unittest.TestCase):
    def test_evaluator_launch_error_propagates_but_timeout_is_a_candidate_failure(self):
        from src import evaluator
        import subprocess
        with patch.object(evaluator.subprocess, 'run', side_effect=FileNotFoundError('python missing')) as call:
            with self.assertRaisesRegex(FileNotFoundError, 'python missing'):
                evaluator.evaluate_scheduler(ROOT / 'src/seed.py')
            self.assertEqual(call.call_count, 1)
        with patch.object(evaluator.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 180)):
            result = evaluator.evaluate_scheduler(ROOT / 'src/seed.py')
        self.assertFalse(result['ok'])
        self.assertIsNone(result['score'])
        self.assertEqual(len(result['scale_results']), 5)

    def test_harness_failure_records_paid_attempt_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=3,
                                      prototype=True, budget_usd=2, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            def generate(directory, prompt, model, timeout, usage):
                usage.update(estimated_cost_usd=.2, observed_cost_usd=.2, model_invocations=1)
            with patch.dict(os.environ, {'DIRECT_MODEL': 'fake'}), \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'generate_direct', side_effect=generate) as call, \
                 patch.object(run, 'evaluate_scheduler', side_effect=[
                     {'ok': True, 'score': 10}, FileNotFoundError('worker missing')]):
                run.run_arm('direct', args, summary, lambda: None)
            arm = summary['arms']['direct']
            self.assertEqual(call.call_count, 1)
            self.assertEqual(arm['stop_reason'], 'harness_error')
            self.assertAlmostEqual(arm['known_cost_usd'], .2)
            row = json.loads((args.output / 'candidates/direct/001/meta.json').read_text())
            self.assertEqual(row['status'], 'evaluation_failed')
            self.assertTrue(row['error_fatal'])
            self.assertIn('worker missing', row['error'])
            self.assertEqual(row['best_score'], 10)

    def test_pricing_failure_keeps_response_and_is_recorded_as_fatal(self):
        from types import SimpleNamespace
        response = SimpleNamespace(model_dump=lambda **kw: {'fixture': True},
            choices=[SimpleNamespace(message=SimpleNamespace(content=(ROOT / 'src/seed.py').read_text()))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=3,
                                      prototype=True, budget_usd=2, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'DIRECT_MODEL': 'fake'}), \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                 patch('litellm.completion', return_value=response) as call, \
                 patch('litellm.completion_cost', side_effect=RuntimeError('broken pricing')):
                run.run_arm('direct', args, summary, lambda: None)
            arm = summary['arms']['direct']
            self.assertEqual(call.call_count, 1)
            self.assertEqual(arm['stop_reason'], 'generation_error')
            self.assertFalse(arm['cost_complete'])
            self.assertIn('broken pricing', arm['trajectory'][-1]['error'])
            self.assertTrue((args.output / 'candidates/direct/001/response.json').exists())

    def test_direct_pricing_uses_opencode_configuration(self):
        from types import SimpleNamespace
        response = SimpleNamespace(model_dump=lambda **kw: {}, service_tier='default',
            choices=[SimpleNamespace(message=SimpleNamespace(content='candidate'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.loads((ROOT / 'src/opencode.json').read_text())
            config['provider']['openai']['models']['gpt-6.1-sol']['cost']['input'] = 7
            (root / 'src').mkdir()
            (root / 'src/opencode.json').write_text(json.dumps(config))
            with patch.object(generators, 'ROOT', root), \
                 patch('litellm.register_model') as register, \
                 patch('litellm.completion', return_value=response), \
                 patch('litellm.completion_cost', return_value=.01):
                usage = {}
                generators.call_llm(root, 'prompt', 'openai/gpt-6.1-sol', 1, usage)
            self.assertEqual(register.call_args.args[0]['gpt-6.1-sol']['input_cost_per_token'], 7e-6)
            self.assertEqual(usage['estimated_cost_usd'], .01)

    def test_resume_keeps_pool_cost_and_existing_artifacts_and_allows_small_overshoot(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=20,
                                      prototype=True, budget_usd=.7, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            parents = []
            def generate(directory, prompt, model, timeout, usage, **kwargs):
                parents.append((directory / 'scheduler.py').read_text())
                (directory / 'scheduler.py').write_text(parents[-1] + '\n# child\n')
                usage.update(estimated_cost_usd=.3, observed_cost_usd=.3, model_invocations=1)
            def evaluate(path, **kwargs):
                return {'ok': True, 'score': 10 - path.read_text().count('# child')}
            with patch.dict(os.environ, {'DIRECT_MODEL': 'fake'}), \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'evaluate_scheduler', side_effect=evaluate) as ev, \
                 patch.object(run, 'generate_direct', side_effect=generate) as call:
                run.run_arm('direct', args, summary, lambda: None)
                arm_dir = args.output / 'candidates/direct'
                before = {p: p.read_bytes() for p in arm_dir.rglob('*') if p.is_file()}
                old_rows = json.loads(json.dumps(summary['arms']['direct']['trajectory']))
                args.resume = True
                args.budget_usd = 1.1
                args.budget_guard = True
                args.budget_slack_usd = .2
                run.run_arm('direct', args, summary, lambda: None)
                arm = summary['arms']['direct']
                self.assertEqual(call.call_count, 4)
                self.assertEqual(ev.call_count, 5)  # seed once plus four new children
                self.assertAlmostEqual(arm['known_cost_usd'], 1.2)
                self.assertEqual(arm['stop_reason'], 'budget')
                self.assertEqual(arm['trajectory'][:len(old_rows)], old_rows)
                self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
                self.assertEqual(arm['trajectory'][-1]['parent'], old_rows[-1]['best_path'])
                (arm_dir / '005').mkdir()
                with self.assertRaisesRegex(ValueError, 'Unrecorded'):
                    run.restore_pool(arm, arm_dir)
                (arm_dir / '005').rmdir()
                (arm_dir / '001/scheduler.py').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    run.restore_pool(arm, arm_dir)

    def test_relaxed_budget_reserve_uses_recent_mean_without_large_floor(self):
        rows = [{'iteration': i+1, 'observed_cost_usd': c} for i,c in enumerate([.8,.1,.2,.3])]
        self.assertAlmostEqual(run.next_round_reserve(rows, 'opencode', slack=.1), .2)
        self.assertAlmostEqual(run.next_round_reserve(rows, 'opencode'), 1.0)

    def test_budget_guard_stops_before_unaffordable_round_or_unknown_cost(self):
        for cost, stop, count in [(.49, 'budget_reserve', 3), (None, 'cost_unknown', 1)]:
            with self.subTest(cost=cost), tempfile.TemporaryDirectory() as temp:
                args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=20,
                                          prototype=True, budget_usd=2, budget_guard=True,
                                          timeout=1, agent_timeout=2)
                summary = {'arms': {}}
                def generate(directory, prompt, model, timeout, usage, **kwargs):
                    usage.update(estimated_cost_usd=cost, observed_cost_usd=cost or .2,
                                 model_invocations=1)
                with patch.dict(os.environ, {'OPENCODE_MODEL': 'fake'}), \
                     patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                     patch.object(run, 'build_prompt', return_value='task'), \
                     patch.object(run, 'generate_opencode', side_effect=generate) as call:
                    run.run_arm('opencode', args, summary, lambda: None)
                arm = summary['arms']['opencode']
                self.assertEqual(call.call_count, count)
                self.assertEqual(arm['stop_reason'], stop)
                self.assertLess(arm['known_cost_usd'], 2)
        # The previous problematic $1.988/$2 endpoint must not start another session.
        self.assertGreater(run.next_round_reserve([{'iteration': 1, 'observed_cost_usd': .45}], 'opencode'), .012)

    def test_confirmed_budget_prefix_survives_later_missing_costs(self):
        rows = [dict(iteration=i, observed_cost_usd=cost or .2, estimated_cost_usd=cost,
                     elapsed_seconds=1, score=score, best_score=score, best_path=str(i), status='improved')
                for i, (cost, score) in enumerate([(.4, 10), (None, 8), (.1, 6)])]
        result = summarize_arm('opencode', {'trajectory': rows, 'stop_reason': 'iterations'}, 5)
        self.assertEqual(result['best_at_or_below_budget']['best_score'], 10)
        self.assertEqual(result['final_best_score'], 6)
        self.assertFalse(result['cost_complete'])
        self.assertIsNone(result['points'][1]['within_budget'])
        self.assertIsNone(result['points'][2]['within_budget'])

    def test_resource_rounding_is_tolerated_but_invalid_allocations_are_rejected(self):
        from types import SimpleNamespace
        from src.worker import verify_valid_assignment
        pool = SimpleNamespace(avail_cpu_pool=.3, avail_ram_pool=.3, allow_memory_overcommit=False)
        assignments = [SimpleNamespace(cpu=.1, ram=.1), SimpleNamespace(cpu=.2, ram=.2)]
        # Ordinary binary-float addition exceeds .3 by one representable step.
        self.assertGreater(sum(a.cpu for a in assignments), pool.avail_cpu_pool)
        verify_valid_assignment(pool, assignments)
        for resource in ('cpu', 'ram'):
            for value in (.20001, float('inf'), float('nan'), -1, 0):
                with self.subTest(resource=resource, value=value):
                    setattr(assignments[1], resource, value)
                    with self.assertRaises(ValueError):
                        verify_valid_assignment(pool, assignments)
                    setattr(assignments[1], resource, .2)
        # Respect the simulator's explicit RAM overcommit setting, never CPU overcommit.
        pool.allow_memory_overcommit = True
        assignments[1].ram = 1
        verify_valid_assignment(pool, assignments)
        assignments[1].cpu = 1
        with self.assertRaisesRegex(ValueError, 'CPU'):
            verify_valid_assignment(pool, assignments)

    def test_registration_contract_accepts_python_spacing_and_rejects_duplicates(self):
        from src.worker import validate_code
        code = (ROOT / 'src/seed.py').read_text().replace('key="experiment"', ' key = "experiment", ')
        self.assertEqual(validate_code(code), code)
        with self.assertRaises(ValueError):
            validate_code(code + '\n@register_scheduler("experiment")\ndef extra(s): pass\n')
        with self.assertRaises(ValueError):
            validate_code(code.replace('"experiment"', '"other"', 1))

    def test_custom_trace_is_snapshotted_and_forwarded_to_both_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / 'custom.csv'
            trace.write_text('archived workload')
            manifest = run.snapshot_experiment(root, trace)
            saved = root / 'snapshot/scenarios/custom.csv'
            trace.write_text('later source edit')
            self.assertEqual(saved.read_text(), 'archived workload')
            self.assertIn('scenarios/custom.csv', manifest['sha256'])
            self.assertNotIn('scenarios/bench_canonical_train.csv', manifest['sha256'])
            args = argparse.Namespace(output=root, trace=saved, iterations=1, prototype=True,
                                      budget_usd=1, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'DIRECT_MODEL': 'fake', 'OPENCODE_MODEL': 'fake'}), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}) as evaluate, \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'generate_direct'), patch.object(run, 'generate_opencode') as agent:
                for arm in ('direct', 'opencode'):
                    run.run_arm(arm, args, summary, lambda: None)
            self.assertEqual(evaluate.call_count, 4)
            self.assertTrue(all(call.kwargs['trace'] == saved for call in evaluate.call_args_list))
            self.assertEqual(agent.call_args.kwargs['trace'], saved)

    def test_analysis_uses_saved_config_and_preserves_research_report(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'run_config.json').write_text(json.dumps({'budget_usd_per_arm': 1}))
            row = {'iteration': 0, 'estimated_cost_usd': 0, 'elapsed_seconds': 1,
                   'score': 10, 'best_score': 10, 'best_path': 'seed', 'status': 'seed'}
            summary = {'status': 'complete', 'arms': {'direct': {
                'trajectory': [row], 'stop_reason': 'iterations'}}}
            report = write_analysis(directory, summary)
            self.assertEqual(report['budget_usd_per_arm'], 1)
            self.assertTrue((directory / 'plots/curve.csv').exists())
            self.assertTrue((directory / 'report.md').exists())
            self.assertFalse((directory / 'comparison.json').exists())
            (directory / 'report.md').write_text('Research findings and limitations.\n')
            write_analysis(directory, summary)
            self.assertEqual((directory / 'report.md').read_text(), 'Research findings and limitations.\n')

    def test_cost_comparison_charges_failed_attempt_and_excludes_budget_crossing_child(self):
        rows = []
        for iteration, cost, score, best, status in [
            (0, 0, 10, 10, 'seed'), (1, .4, 8, 8, 'improved'),
            (2, .4, None, 8, 'generation_failed'), (3, .4, 6, 6, 'improved'),
        ]:
            rows.append({'iteration': iteration, 'observed_cost_usd': cost,
                         'estimated_cost_usd': cost, 'score': score, 'best_score': best,
                         'best_path': f'candidate-{best}', 'status': status, 'elapsed_seconds': 1})
        result = summarize_arm('direct', {'trajectory': rows, 'stop_reason': 'budget'}, 1)
        self.assertEqual(result['best_at_or_below_budget']['best_score'], 8)
        self.assertEqual(result['final_best_score'], 6)
        self.assertAlmostEqual(result['known_cost_usd'], 1.2)
        self.assertAlmostEqual(result['overshoot_usd'], .2)
        self.assertEqual(result['points'][2]['status'], 'generation_failed')

    def test_incomplete_cost_cannot_establish_within_budget_score(self):
        row = {'iteration': 1, 'observed_cost_usd': .4, 'estimated_cost_usd': None,
               'score': 8, 'best_score': 8, 'best_path': 'candidate',
               'status': 'improved', 'elapsed_seconds': 2}
        result = summarize_arm('opencode', {'trajectory': [row], 'stop_reason': 'iterations'}, 5)
        self.assertFalse(result['cost_complete'])
        self.assertIsNone(result['best_at_or_below_budget'])
        self.assertEqual(result['known_cost_usd'], .4)

    def test_worse_child_and_independent_pools(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=2, prototype=True,
                                      budget_usd=None, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            parents = []
            def generate(directory, prompt, model, timeout, usage, **kwargs):
                if directory.parent.name == 'opencode':
                    self.assertTrue(kwargs['prototype'])
                    self.assertEqual(timeout, 2)
                else:
                    self.assertEqual(timeout, 1)
                parents.append((directory / 'scheduler.py').read_text())
                (directory / 'scheduler.py').write_text('# worse child\n' + parents[-1])
                usage.update(estimated_cost_usd=0.1, model_invocations=1)
            def evaluate(path, **kwargs):
                return {'ok': True, 'score': 10 if '000_seed' in str(path) else 20,
                        'scale_results': {}, 'params': {}}
            with patch.dict('os.environ', {'DIRECT_MODEL': 'test', 'OPENCODE_MODEL': 'test'}), \
                 patch.object(run, 'evaluate_scheduler', side_effect=evaluate), \
                 patch.object(run, 'build_prompt', return_value='same context'), \
                 patch.object(run, 'generate_direct', side_effect=generate), \
                 patch.object(run, 'generate_opencode', side_effect=generate):
                for arm in ['direct', 'opencode']:
                    run.run_arm(arm, args, summary, lambda: None)
            self.assertEqual(len(parents), 4)
            self.assertEqual(len(set(parents)), 1)
            for arm in summary['arms'].values():
                self.assertEqual([r['best_score'] for r in arm['trajectory']], [10, 10, 10])
                self.assertEqual(len(arm['trajectory']), 3)
            self.assertNotEqual(summary['arms']['direct']['best_path'], summary['arms']['opencode']['best_path'])

    def test_budget_stops_after_known_cost_and_unknown_uses_iterations(self):
        for cost, expected in [(0.6, 3), (None, 5)]:
            with tempfile.TemporaryDirectory() as temp:
                args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=4, prototype=True,
                                          budget_usd=1.0, timeout=1, agent_timeout=2)
                summary = {'arms': {}}
                def generate(directory, prompt, model, timeout, usage):
                    usage.update(estimated_cost_usd=cost, model_invocations=1)
                with patch.dict('os.environ', {'DIRECT_MODEL': 'fake'}), \
                     patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                     patch.object(run, 'build_prompt', return_value='task'), \
                     patch.object(run, 'generate_direct', side_effect=generate):
                    run.run_arm('direct', args, summary, lambda: None)
                arm = summary['arms']['direct']
                self.assertEqual(len(arm['trajectory']), expected)
                self.assertEqual(arm['stop_reason'], 'budget' if cost else 'iterations')
                if cost is None:
                    self.assertIsNone(arm['estimated_cost_usd'])

    def test_failed_scale_is_ineligible_and_resources_scale(self):
        import subprocess
        from src import evaluator
        calls = []
        def worker(command, **kwargs):
            calls.append(json.loads(command[-1]))
            data = {'ok': True, 'latency': 10} if len(calls) == 1 else {'ok': False, 'error': 'test'}
            return subprocess.CompletedProcess(command, 0, '__RESULT__' + json.dumps(data), '')
        with patch('src.evaluator.subprocess.run', side_effect=worker):
            result = evaluator.evaluate_scheduler(ROOT / 'src/seed.py', prototype=True, scales=[1, 2])
        self.assertFalse(result['ok'])
        self.assertIsNone(result['score'])
        self.assertEqual([c['cpus_per_pool'] for c in calls], [64, 128])
        self.assertEqual([c['ram_gb_per_pool'] for c in calls], [500, 1000])
        self.assertEqual([c['random_seed'] for c in calls], [42, 42])

    def test_usage_events_deduplicated_and_missing_cost(self):
        event = {'type': 'step_finish', 'part': {'id': 'a', 'cost': 0.2,
                 'tokens': {'input': 10, 'output': 5, 'reasoning': 2, 'cache': {'read': 8, 'write': 0}}}}
        usage = generators.parse_opencode_usage('\n'.join([json.dumps(event)] * 2))
        self.assertEqual(usage['model_invocations'], 1)
        self.assertEqual(usage['estimated_cost_usd'], 0.2)
        event['part']['cost'] = 0
        self.assertIsNone(generators.parse_opencode_usage(json.dumps(event))['estimated_cost_usd'])
        self.assertIsNone(generators.parse_opencode_usage('garbage')['estimated_cost_usd'])

    def test_detection_missing(self):
        with patch('src.generators.shutil.which', return_value=None), patch('src.generators.Path.is_file', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'not installed'):
                generators.detect_opencode()

    def test_partial_cost_keeps_deduplicated_observations(self):
        paid = {'type': 'step_finish', 'part': {'id': 'paid', 'cost': .4}}
        for ending in [
            {'type': 'step_finish', 'part': {'id': 'missing'}},
            {'type': 'step_finish', 'part': {'id': 'zero', 'cost': 0}},
            {'type': 'error', 'error': 'interrupted'},
        ]:
            with self.subTest(ending=ending):
                usage = generators.parse_opencode_usage('\n'.join(map(json.dumps, [paid, paid, ending])))
                self.assertAlmostEqual(usage['observed_cost_usd'], .4)
                self.assertIsNone(usage['estimated_cost_usd'])
        self.assertEqual(generators.parse_opencode_usage('')['observed_cost_usd'], 0)

    def test_agent_timeout_counts_paid_steps_and_stops_at_budget(self):
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cli = root / 'fake-cli'
            event = {'type': 'step_finish', 'part': {'id': 'paid', 'cost': .4,
                     'tokens': {'input': 10, 'output': 5, 'reasoning': 0}}}
            cli.write_text('#!' + sys.executable + '\nimport time\nprint(' +
                           repr(json.dumps(event)) + ', flush=True)\ntime.sleep(30)\n')
            cli.chmod(0o700)
            args = argparse.Namespace(output=root / 'run', trace=TRACE, iterations=2, prototype=True,
                                      budget_usd=.3, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            with patch.dict(os.environ, {'OPENCODE_MODEL': 'fake'}), \
                 patch.object(generators, 'detect_opencode', return_value=(str(cli), '--model --format')), \
                 patch.object(run, 'build_prompt', return_value='test'), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}):
                run.run_arm('opencode', args, summary, lambda: None)
            arm = summary['arms']['opencode']
            self.assertEqual(len(arm['trajectory']), 2)  # seed + one paid timeout
            self.assertEqual(arm['stop_reason'], 'budget')
            self.assertAlmostEqual(arm['known_cost_usd'], .4)
            self.assertFalse(arm['cost_complete'])
            self.assertIsNone(arm['estimated_cost_usd'])
            attempt = arm['trajectory'][1]
            self.assertEqual(attempt['status'], 'generation_failed')
            self.assertFalse(attempt['error_fatal'])
            self.assertAlmostEqual(attempt['observed_cost_usd'], .4)
            self.assertIsNone(attempt['estimated_cost_usd'])

    def test_seed_ignores_idle_ticks_and_drops_expired_and_failed_work(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from eudoxia.workload import OperatorState
        from eudoxia.workload.runtime_status import ASSIGNABLE_STATES
        namespace = {'register_scheduler': lambda **kw: lambda f: f,
                     'register_scheduler_init': lambda **kw: lambda f: f,
                     'OperatorState': OperatorState, 'ASSIGNABLE_STATES': ASSIGNABLE_STATES,
                     'Assignment': lambda **kw: SimpleNamespace(**kw)}
        exec((ROOT / 'src/seed.py').read_text(), namespace)
        pool = SimpleNamespace(avail_cpu_pool=64, avail_ram_pool=500)
        state = SimpleNamespace(params={'multi_operator_containers': False},
                                executor=SimpleNamespace(num_pools=1, pools=[pool]))
        namespace['init'](state)
        idle = Mock()
        state.waiting_queue = [idle]
        self.assertEqual(namespace['schedule'](state, [], []), ([], []))
        idle.runtime_status.assert_not_called()
        state.waiting_queue = []
        work = []
        for failed, expired in [(False, True), (True, False), (False, False)]:
            status = Mock(state_counts={OperatorState.FAILED: int(failed)})
            status.is_pipeline_successful.return_value = False
            status.has_timed_out.return_value = expired
            status.get_ops.return_value = ['ready-op']
            pipeline = Mock(priority='query', pipeline_id=len(work))
            pipeline.runtime_status.return_value = status
            work.append(pipeline)
        suspensions, assignments = namespace['schedule'](state, [], work)
        self.assertEqual(suspensions, [])
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].pipeline_id, 2)
        self.assertEqual(state.waiting_queue, [work[2]])
        for pipeline in work[:2]:
            pipeline.runtime_status().get_ops.assert_not_called()

    def test_invalid_scheduler(self):
        with self.assertRaises(ValueError):
            generators.validate_code('def x(): pass')
        generators.validate_code((ROOT / 'src/seed.py').read_text())

    def test_opencode_process_failures_and_edit(self):
        # Exercise real subprocess execution with an explicitly fake local CLI.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cli = base / 'fake-opencode'
            for behavior in ['unchanged', 'missing', 'invalid', 'nonzero', 'timeout', 'success']:
                directory = base / behavior
                directory.mkdir()
                (directory / 'scheduler.py').write_text((ROOT / 'src/seed.py').read_text())
                body = {'unchanged': 'pass', 'nonzero': 'raise SystemExit(2)',
                        'missing': 'from pathlib import Path\nPath("scheduler.py").unlink()',
                        'invalid': 'from pathlib import Path\nPath("scheduler.py").write_text("def broken(:")',
                        'timeout': 'import time; time.sleep(5)',
                        'success': 'from pathlib import Path\np=Path("scheduler.py")\np.write_text(p.read_text()+"\\n# edited\\n")'}[behavior]
                cli.write_text('#!/usr/bin/env python3\nimport os, sys\n'
                               'assert os.environ["PWD"] == os.getcwd()\n'
                               'assert sys.argv[sys.argv.index("--dir") + 1] == os.getcwd()\n' + body + '\n')
                cli.chmod(0o755)
                with patch('src.generators.detect_opencode', return_value=(str(cli), '--model --format')):
                    if behavior in ('success', 'unchanged'):
                        generators.generate_opencode(directory, 'task', 'fake', 2, {}, prototype=True)
                        if behavior == 'success':
                            self.assertIn('# edited', (directory / 'scheduler.py').read_text())
                        else:
                            self.assertEqual((directory / 'scheduler.py').read_bytes(), (ROOT / 'src/seed.py').read_bytes())
                        rendered = (directory / 'workspace/task.md').read_text()
                        self.assertEqual(rendered, (directory / 'prompt.txt').read_text())
                        self.assertIn('./evaluate', rendered)
                        self.assertIn('--prototype', (directory / 'workspace/evaluate').read_text())
                        command = json.loads((directory / 'command.json').read_text())
                        self.assertEqual(command['argv'][command['argv'].index('--dir') + 1],
                                         str((directory / 'workspace').resolve()))
                    else:
                        with self.assertRaises((RuntimeError, ValueError, SyntaxError, TimeoutError)):
                            generators.generate_opencode(directory, 'task', 'fake', .1 if behavior == 'timeout' else 2, {})
                self.assertTrue((directory / 'response.txt').exists())

    def test_direct_one_call_no_retries(self):
        from unittest.mock import Mock
        response = Mock()
        response.model_dump.return_value = {'test_fixture': True}
        response.choices = [Mock(message=Mock(content='```python\n' + (ROOT / 'src/seed.py').read_text() + '\n```'))]
        response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        with tempfile.TemporaryDirectory() as temp, \
             patch('litellm.completion', return_value=response) as call, \
             patch('litellm.completion_cost', return_value=.01), \
             patch.dict('os.environ', {'DIRECT_REASONING_EFFORT': 'medium'}):
            usage = {}
            generators.generate_direct(Path(temp), 'context', 'fake', 1, usage)
            call.assert_called_once()
            self.assertEqual(call.call_args.kwargs['num_retries'], 0)
            self.assertEqual(call.call_args.kwargs['reasoning_effort'], 'medium')
            self.assertNotIn('temperature', call.call_args.kwargs)
            self.assertEqual(usage['model_invocations'], 1)
            self.assertEqual(call.call_args.kwargs['messages'][0]['content'],
                             (Path(temp) / 'prompt.txt').read_text())

    def test_failures_continue_from_best_and_unchanged_is_valid_in_both_arms(self):
        for arm in ('direct', 'opencode'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=6, prototype=True,
                                          budget_usd=1, timeout=1, agent_timeout=2)
                summary = {'arms': {}}
                parents = []
                def generate(directory, prompt, model, timeout, usage, **kwargs):
                    code = (directory / 'scheduler.py').read_text()
                    parents.append(code)
                    usage.update(estimated_cost_usd=.1, model_invocations=1)
                    attempt = int(directory.name)
                    failure = {1: SyntaxError('broken code'), 2: ValueError('missing scheduler'),
                               3: TimeoutError('deadline')}.get(attempt)
                    if failure:
                        raise failure
                    if attempt == 5:
                        (directory / 'scheduler.py').write_text(code + '\n# improved\n')
                def evaluate(path, **kwargs):
                    return {'ok': True, 'score': 5 if '# improved' in path.read_text() else 10}
                with patch.dict('os.environ', {'DIRECT_MODEL': 'fake', 'OPENCODE_MODEL': 'fake'}), \
                     patch.object(run, 'evaluate_scheduler', side_effect=evaluate) as eval_call, \
                     patch.object(run, 'build_prompt', return_value='task'), \
                     patch.object(run, 'generate_' + arm, side_effect=generate):
                    run.run_arm(arm, args, summary, lambda: None)
                result = summary['arms'][arm]
                rows = result['trajectory']
                self.assertEqual(result['stop_reason'], 'iterations')
                self.assertEqual(result['failed_attempts'], 3)
                self.assertAlmostEqual(result['estimated_cost_usd'], .6)
                self.assertEqual([r['status'] for r in rows],
                                 ['seed', 'generation_failed', 'generation_failed', 'generation_failed',
                                  'unchanged', 'improved', 'unchanged'])
                self.assertEqual([r['best_score'] for r in rows], [10, 10, 10, 10, 10, 5, 5])
                self.assertTrue(all(not r['error_fatal'] for r in rows))
                self.assertTrue(all(r['parent'] == rows[0]['path'] for r in rows[1:6]))
                self.assertEqual(rows[6]['parent'], rows[5]['path'])
                self.assertEqual(len(set(parents[:5])), 1)
                self.assertEqual(eval_call.call_count, 4)  # seed + both unchanged + improved
                saved = json.loads((Path(rows[1]['path']) / 'meta.json').read_text())
                self.assertEqual(saved['status'], 'generation_failed')

    def test_provider_or_cli_failure_stops_only_affected_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output=Path(temp), trace=TRACE, iterations=3, prototype=True,
                                      budget_usd=5, timeout=1, agent_timeout=2)
            summary = {'arms': {}}
            with patch.dict('os.environ', {'DIRECT_MODEL': 'fake', 'OPENCODE_MODEL': 'fake'}), \
                 patch.object(run, 'evaluate_scheduler', return_value={'ok': True, 'score': 10}), \
                 patch.object(run, 'build_prompt', return_value='task'), \
                 patch.object(run, 'generate_direct', side_effect=RuntimeError('authentication failed')) as direct, \
                 patch.object(run, 'generate_opencode') as agent:
                run.run_arm('direct', args, summary, lambda: None)
                run.run_arm('opencode', args, summary, lambda: None)
            self.assertEqual(direct.call_count, 1)
            self.assertEqual(agent.call_count, 3)
            failed = summary['arms']['direct']
            self.assertEqual(failed['stop_reason'], 'generation_error')
            self.assertTrue(failed['trajectory'][-1]['error_fatal'])
            self.assertIsNone(failed['estimated_cost_usd'])
            self.assertEqual(summary['arms']['opencode']['stop_reason'], 'iterations')

    def test_direct_api_timeouts_are_recoverable_but_provider_errors_are_fatal(self):
        import litellm
        errors = [(litellm.Timeout(message='deadline', model='fake', llm_provider='test'), TimeoutError),
                  (ValueError('unsupported provider configuration'), RuntimeError)]
        for error, expected_type in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp, \
                 patch('litellm.completion', side_effect=error) as call:
                usage = {'estimated_cost_usd': None}
                with self.assertRaises(expected_type):
                    generators.generate_direct(Path(temp), 'task', 'fake', 1, usage)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(usage['model_invocations'], 1)
                self.assertIsNone(usage['estimated_cost_usd'])

    def test_agent_timeout_allows_full_evaluations_and_is_shown_in_prompt(self):
        from src.config import default_agent_timeout
        self.assertEqual(default_agent_timeout(), 2100)
        self.assertEqual(default_agent_timeout(prototype=True), 1500)
        self.assertEqual(default_agent_timeout(direct_timeout=60), 1860)
        self.assertIn('2100-second', generators.render_opencode_prompt('task'))
        self.assertIn('1500-second', generators.render_opencode_prompt('task', prototype=True))
        self.assertIn('45-second', generators.render_opencode_prompt('task', timeout=45))

    def test_prompt_retains_evolved_metric_and_diagnostics_without_archive_metadata(self):
        from copy import deepcopy
        from src.config import get_canonical_base_params
        stats = {
            'ok': True, 'latency': 123.4567, 'failures': 9, 'oom_count': 7,
            'failure_error_counts': {'OOM': 7, 'OTHER': 2},
            'mean_memory_allocated_percent': 87.5, 'mean_memory_consumed_percent': 43.25,
            'assignments': 111, 'containers_completed': 90, 'suspensions': 3,
            'query': {'arrived': 13, 'completed': 12, 'mean_s': 8.25, 'p99_s': 27.5},
            'interactive': {'arrived': 23, 'completed': 21, 'mean_s': 18.5, 'p99_s': 60.0},
            'batch': {'arrived': 31, 'completed': 0, 'mean_s': None, 'p99_s': None},
        }
        evaluation = {
            'ok': True, 'score': 123.4567, 'scales': [1, 2, 4, 8, 16],
            'scale_results': {str(s): stats for s in [1, 2, 4, 8, 16]},
            'params': get_canonical_base_params(True), 'trace': 'fixture.csv',
            'elapsed_seconds': 1.0, 'trace_sha256': 'archival-only',
        }
        # Nondefault penalty/settings catch stale canonical constants in prompts.
        evaluation['params']['max_job_seconds'] = 37
        code = (ROOT / 'src/seed.py').read_text()
        before = deepcopy(evaluation)
        prompt = generators.build_prompt(code, evaluation)
        self.assertEqual(evaluation, before)
        self.assertTrue(prompt.startswith('# Eudoxia + Bauplan'))
        self.assertLess(prompt.index('## Instructions'), prompt.index('## Improvement request'))
        self.assertLess(prompt.index('## Improvement request'), prompt.index('## Evaluation objective'))
        self.assertNotIn('## Task\n', prompt)
        for expected in ['74 seconds', 'query=10, interactive=5, batch=1',
                         'geometric mean', 'across ALL', 'score=null',
                         'Simulation duration: 60 s; ticks per second: 10',
                         'wall-clock timeout: 120 s per scale',
                         'aggregate score: 123.4567 s', '16x: 1024 CPUs, 8000 GB RAM',
                         'OOMs: 7', "'OTHER': 2", '87.5000%', '43.2500%',
                         'Assignments: 111', 'containers completed: 90', 'suspensions: 3',
                         '| query | 12/13 | 8.2500 | 27.5000 |',
                         '| interactive | 21/23 | 18.5000 | 60.0000 |',
                         '| batch | 0/31 | n/a | n/a |']:
            self.assertIn(expected, prompt)
        self.assertEqual(prompt.count(code.rstrip()), 1)
        self.assertNotIn('archival-only', prompt)
        evaluation.update(elapsed_seconds=999, trace_sha256='other-archive-hash')
        self.assertEqual(prompt, generators.build_prompt(code, evaluation))
        for rendered in [generators.render_direct_prompt(prompt),
                         generators.render_opencode_prompt(prompt, prototype=True)]:
            self.assertEqual(rendered.split('\n## Delivery\n')[0], prompt)
        self.assertIn('./evaluate', generators.render_opencode_prompt(prompt, prototype=True))
        self.assertNotIn(str(ROOT), generators.render_opencode_prompt(prompt, prototype=True))
        evaluation['ok'] = False
        evaluation['score'] = None
        evaluation['scale_results']['16'] = {'ok': False, 'error': 'subprocess timeout'}
        failed = generators.build_prompt(code, evaluation)
        self.assertIn('Eligible: False; aggregate score: n/a', failed)
        self.assertIn('Simulation FAILED: subprocess timeout', failed)

    def test_evaluation_launcher_resolves_paths_and_quotes_them_in_both_modes(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / "project with spaces and 'quote $value"
            root.mkdir()
            # A stand-in evaluator reports its argv; no simulator or model calls.
            (root / 'src').mkdir()
            (root / 'src/__init__.py').write_text('')
            (root / 'src/evaluator.py').write_text(
                'import json, sys\nprint(json.dumps({"python": sys.executable, "args": sys.argv[1:]}))\n')
            work = root / 'candidate' / 'workspace'
            work.mkdir(parents=True)
            for prototype in (False, True):
                with patch.object(generators, 'ROOT', root):
                    launcher = generators.write_evaluation_launcher(work, prototype=prototype)
                # Launch from a different cwd to catch accidental relative-path dependencies.
                result = subprocess.run([str(launcher)], cwd=temp, capture_output=True, text=True, check=True)
                actual = json.loads(result.stdout)
                expected = ['--scheduler', str(work / 'scheduler.py'),
                            '--output-dir', str(work.parent / 'internal_evals'), '--trace', str(TRACE.resolve())]
                if prototype:
                    expected.append('--prototype')
                self.assertEqual(actual['args'], expected)
                self.assertEqual(actual['python'], sys.executable)

    def test_self_evaluation_archives_each_attempt_and_uses_matching_mode(self):
        import contextlib
        import io
        from src import evaluator
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / 'scheduler.py'
            source.write_text('first candidate')
            seen = []
            def evaluate(path, **kwargs):
                seen.append((path, path.read_text(), kwargs['prototype']))
                return {'ok': kwargs['prototype'], 'score': 7 if kwargs['prototype'] else None}
            base_args = ['--scheduler', str(source), '--output-dir', str(directory / 'evals')]
            with patch.object(evaluator, 'evaluate_scheduler', side_effect=evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(base_args + ['--prototype']), 0)
                source.write_text('second candidate')
                self.assertEqual(evaluator.main(base_args), 1)
            self.assertEqual([(text, mode) for _, text, mode in seen],
                             [('first candidate', True), ('second candidate', False)])
            self.assertNotEqual(seen[0][0], seen[1][0])
            self.assertEqual(seen[0][0].read_text(), 'first candidate')
            self.assertEqual(json.loads((seen[0][0].parent / 'eval.json').read_text())['score'], 7)
            self.assertFalse(json.loads((seen[1][0].parent / 'eval.json').read_text())['ok'])


if __name__ == '__main__':
    unittest.main()
