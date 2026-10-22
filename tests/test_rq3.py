"""Offline tests for the RQ3 signals table and the post-search LLM judge."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import predict, signals
from src.config import CANONICAL_SIM_PARAMS, PROTOTYPE_OVERRIDES, ROOT

SEED = (ROOT / 'src/seed.py').read_text()


def evaluation(score, *, ooms=0, completed=8):
    ok = score is not None
    scale = {'ok': ok, 'latency': score, 'oom_count': ooms, 'failures': ooms,
             'query': {'arrived': 10, 'completed': completed}} if ok else {'ok': False, 'error': 'boom'}
    return {'ok': ok, 'score': score, 'scales': [1], 'scale_results': {'1': scale},
            'params': {**CANONICAL_SIM_PARAMS, **PROTOTYPE_OVERRIDES}, 'trace': 'hard.csv'}


def write_search(directory, candidates, arm='opencode'):
    """candidates: (parent index or None, score, status, code suffix, self-test scores)."""
    rows = []
    for i, (parent, score, status, suffix, tests) in enumerate(candidates):
        path = directory / 'candidates' / arm / f'{i:03d}'
        path.mkdir(parents=True)
        (path / 'scheduler.py').write_text(SEED + suffix)
        (path / 'eval.json').write_text(json.dumps(evaluation(score, ooms=i)))
        for k, test in enumerate(tests):
            (path / 'internal_evals' / str(k)).mkdir(parents=True)
            (path / 'internal_evals' / str(k) / 'eval.json').write_text(json.dumps(
                {**evaluation(test), 'started_at_utc': f'2026-10-21T00:00:0{k}'}))
        rows.append({'iteration': i, 'path': str(path), 'parent': rows[parent]['path'] if parent is not None else None,
                     'score': score, 'best_score': score, 'best_path': str(path), 'status': status,
                     'generation_usage': {'input_tokens': 100 * i, 'observed_cost_usd': .1}})
    (directory / 'results.json').write_text(json.dumps({'status': 'complete', 'arms': {
        arm: {'trajectory': rows, 'stop_reason': 'iterations'}}}))
    return rows


# seed(10) -> 1 (8, improved; self-tests 9 then 8) -> 3 (6, improved)
#          -> 2 (failed evaluation)        1 -> 4 (unchanged code)
SEARCH = [(None, 10, 'seed', '', []), (0, 8, 'improved', '\n# s.queue sort(\n', [9, 8]),
          (0, None, 'evaluation_failed', '\n# Suspend(\n', []), (1, 6, 'improved', '\n# memory\n', []),
          (1, 8, 'unchanged', '\n# s.queue sort(\n', [])]


class SignalsTests(unittest.TestCase):
    def test_rows_join_predictors_with_parent_diagnostics_and_descendant_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_search(directory, SEARCH)
            rows = {r['iteration']: r for r in signals.write_signals(directory)}
            self.assertTrue((directory / 'signals.csv').exists())
            seed, first, failed = rows[0], rows[1], rows[2]
            self.assertEqual(seed['out_descendants'], 4)
            self.assertAlmostEqual(seed['out_best_descendant_ratio'], .6)
            self.assertTrue(first['out_better_than_parent'])
            self.assertAlmostEqual(first['out_best_descendant_ratio'], .75)
            self.assertIsNone(failed['out_better_than_parent'])
            self.assertEqual(failed['out_failed_scales'], 1)
            # Predictors come from the parent's evaluation, not the candidate's own.
            self.assertEqual(rows[3]['parent_ooms'], 1)
            self.assertAlmostEqual(first['parent_completion_1x'], .8)
            self.assertEqual(first['agent_self_tests'], 2)
            self.assertEqual(first['agent_test_improving_steps'], 1.0)
            self.assertEqual(first['code_diff_lines'], 2)
            self.assertTrue(failed['code_uses_suspend'])
            self.assertEqual(first['code_sort_calls'], 1)
            self.assertIsNone(first['pred_score'])


def fake_llm(responses, prompts):
    def call(folder, prompt, model, timeout, usage, **kwargs):
        prompts.append(prompt)
        usage.update(estimated_cost_usd=.01, observed_cost_usd=.01)
        return responses.pop(0)
    return call


def answer(score, p, fail=False):
    return json.dumps({'will_fail': fail, 'predicted_score': score, 'p_better_than_parent': p, 'reason': 'r'})


class PredictTests(unittest.TestCase):
    def test_judge_sees_parent_results_but_never_the_child_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            candidates = [list(c) for c in SEARCH]
            candidates[3][1] = 6.123456  # a sentinel the judge must not see
            write_search(directory, [tuple(c) for c in candidates])
            prompts = []
            responses = [answer(7, .9), answer(None, .1, True), answer(5, .7)]
            with patch.object(predict, 'call_llm', fake_llm(responses, prompts)):
                predict.predict_run(directory, 'judge')
            self.assertEqual(len(prompts), 3)  # seed and unchanged code are not judged
            self.assertTrue(all('6.1234' not in p for p in prompts))
            self.assertIn('8.0000 s', prompts[2])  # the parent's score is shown
            self.assertNotIn('## Improvement request', prompts[0])
            self.assertIn('# memory', prompts[2])
            report = json.loads((directory / 'predictions.json').read_text())
            self.assertEqual(report['summary']['better_accuracy'], 1.0)
            self.assertEqual(report['summary']['fail_accuracy'], 1.0)
            self.assertAlmostEqual(report['summary']['better_base_rate'], 2 / 3)
            self.assertEqual(report['summary']['score_pairs'], 2)
            rows = {r['iteration']: r for r in signals.write_signals(directory)}
            self.assertEqual(rows[1]['pred_score'], 7)
            self.assertTrue(rows[2]['pred_will_fail'])

    def test_reruns_skip_judged_candidates_and_budget_and_model_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_search(directory, SEARCH)
            prompts = []
            responses = ['not json', answer(9, .2), answer(5, .7)]
            with patch.object(predict, 'call_llm', fake_llm(responses, prompts)):
                spent = predict.predict_run(directory, 'judge', max_usd=.015)
                self.assertEqual(len(prompts), 2)
                predict.predict_run(directory, 'judge')
                self.assertEqual(len(prompts), 3)
                predict.predict_run(directory, 'judge')
                self.assertEqual(len(prompts), 3)
                with self.assertRaisesRegex(ValueError, '--tag'):
                    predict.predict_run(directory, 'other-judge')
            self.assertAlmostEqual(spent, .02)
            report = json.loads((directory / 'predictions.json').read_text())
            self.assertEqual(report['summary']['invalid_predictions'], 1)
            self.assertEqual(report['summary']['judged'], 2)

    def test_parse_rejects_malformed_predictions(self):
        for raw in ['[]', json.dumps({'will_fail': False}),
                    answer(-1, .5), answer(5, 1.5), answer(5, .5).replace('false', '"no"')]:
            with self.assertRaises(ValueError):
                predict.parse_prediction(raw)

    def test_spearman_handles_ties_and_small_samples(self):
        self.assertAlmostEqual(predict.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(predict.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(predict.ranks([5, 1, 5]), [2.5, 1.0, 2.5])
        self.assertIsNone(predict.spearman([1, 2], [1, 2]))


if __name__ == '__main__':
    unittest.main()
