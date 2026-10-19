"""No network: selector decisions, isolation, fallback and joint accounting."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ROOT, TRACE
from src import run
from src import selection


class SelectionTests(unittest.TestCase):
    def make_pool(self, root):
        pool = []
        for name, score in [('000_seed', 10), ('001', 20)]:
            folder = root / name
            folder.mkdir()
            (folder / 'scheduler.py').write_text((ROOT / 'src/seed.py').read_text())
            pool.append({'path': str(folder), 'iteration': len(pool),
                         'evaluation': {'ok': True, 'score': score, 'params': {}, 'scale_results': {}}})
        return pool

    def choose(self, root, pool, usage):
        directory = root / 'attempt'
        directory.mkdir()
        return selection.select_parent(pool, [], directory, strategy='llm', generator='opencode',
                                       model='fake', reasoning_effort='medium', timeout=1,
                                       remaining_budget=2, usage=usage)

    def test_single_parent_is_free_and_invalid_candidates_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool = self.make_pool(root)
            pool[1]['evaluation'] = {'ok': False, 'score': None}
            with patch.object(selection, 'call_llm') as call:
                chosen = self.choose(root, pool, {})
            self.assertIs(chosen, pool[0])
            call.assert_not_called()

    def test_llm_can_choose_worse_parent_and_receives_all_valid_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool = self.make_pool(root)
            with patch.object(selection, 'call_llm', return_value='{"parent_id":"001","reason":"Explore structure"}') as call:
                self.assertIs(self.choose(root, pool, {}), pool[1])
            prompt = call.call_args.args[1]
            self.assertIn('"id": "000_seed"', prompt)
            self.assertIn('"id": "001"', prompt)
            self.assertEqual(call.call_args.kwargs['reasoning_effort'], 'medium')
            decision = json.loads((root / 'attempt/selection.json').read_text())
            self.assertEqual(decision['parent_rank'], 2)
            self.assertFalse(decision['fallback'])

    def test_bad_json_or_unknown_id_falls_back_without_retry(self):
        for raw in ['not json', '[]', '{"parent_id":"999","reason":"unknown"}']:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pool = self.make_pool(root)
                with patch.object(selection, 'call_llm', return_value=raw) as call:
                    self.assertIs(self.choose(root, pool, {}), pool[0])
                call.assert_called_once()
                self.assertTrue(json.loads((root / 'attempt/selection.json').read_text())['fallback'])

    def run_fixture(self, root, selection_cost, budget=1, raw=None):
        args = argparse.Namespace(output=root, trace=TRACE, iterations=2, prototype=True,
                                  budget_usd=budget, timeout=1, agent_timeout=2, selector='compare')
        summary = {'arms': {}}
        parents = []
        def generate(directory, prompt, model, timeout, usage, **kwargs):
            parents.append((directory / 'scheduler.py').read_text())
            self.assertNotIn('SECRET_RATIONALE', prompt)
            (directory / 'scheduler.py').write_text(f'# child {directory.name}\n' + parents[-1])
            usage.update(estimated_cost_usd=.1, observed_cost_usd=.1, model_invocations=1)
        def call(directory, prompt, model, timeout, usage, **kwargs):
            usage.update(estimated_cost_usd=selection_cost, observed_cost_usd=selection_cost or 0,
                         model_invocations=1, input_tokens=10, output_tokens=5)
            return raw or '{"parent_id":"001","reason":"SECRET_RATIONALE"}'
        def evaluate(path, **kwargs):
            return {'ok': True, 'score': 10 if path.parent.name == '000_seed' else 20,
                    'params': {}, 'scale_results': {}}
        with patch.dict('os.environ', {'OPENCODE_MODEL': 'fake', 'OPENCODE_VARIANT': 'medium'}), \
             patch.object(run, 'evaluate_scheduler', side_effect=evaluate), \
             patch.object(run, 'build_prompt', side_effect=lambda code, evaluation: code), \
             patch.object(run, 'generate_opencode', side_effect=generate), \
             patch.object(selection, 'call_llm', side_effect=call):
            run.run_arm('opencode_llm', args, summary, lambda: None, generator='opencode', selector='llm')
        return summary['arms']['opencode_llm'], parents

    def test_selection_and_generation_costs_add_and_reason_is_not_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            arm, parents = self.run_fixture(Path(temp), .05)
            self.assertEqual(len(parents), 2)
            self.assertIn('# child 001', parents[1])
            self.assertAlmostEqual(arm['known_cost_usd'], .25)
            last = arm['trajectory'][-1]
            self.assertEqual(last['model_invocations'], 2)
            self.assertAlmostEqual(last['selection_usage']['estimated_cost_usd'], .05)
            self.assertAlmostEqual(last['generation_usage']['estimated_cost_usd'], .1)
            self.assertEqual(Path(last['parent']).name, '001')
            self.assertEqual(last['best_score'], 10)

    def test_budget_spent_on_selection_does_not_start_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            arm, parents = self.run_fixture(Path(temp), .2, budget=.25)
            self.assertEqual(len(parents), 1)
            self.assertEqual(arm['stop_reason'], 'budget')
            self.assertEqual(arm['trajectory'][-1]['status'], 'selection_only')
            self.assertAlmostEqual(arm['known_cost_usd'], .3)
            self.assertEqual(arm['trajectory'][-1]['generation_usage']['model_invocations'], 0)

    def test_unknown_selector_cost_stops_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            arm, parents = self.run_fixture(Path(temp), None)
            self.assertEqual(len(parents), 1)
            self.assertEqual(arm['stop_reason'], 'generation_error')
            self.assertFalse(arm['cost_complete'])
            self.assertEqual(arm['trajectory'][-1]['status'], 'selection_failed')


if __name__ == '__main__':
    unittest.main()
