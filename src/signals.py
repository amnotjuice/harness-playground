"""Build one table row per candidate for RQ3: what predicts further improvement?

Columns are grouped by prefix:
  code_*    features of the candidate's scheduler code and its diff from the parent
  parent_*  the parent's evaluation diagnostics (what the generator was shown)
  agent_*   generation behavior: tokens, cost, self-tests and their score trajectory
  pred_*    LLM judge predictions, when predict.py has been run
  out_*     outcomes: the candidate's own score and how its descendants fared

Predictors use only what existed before the candidate was evaluated; out_* never
feeds back into them. Never calls a model.
"""
import argparse
import ast
import csv
import difflib
import json
import re
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def code_features(code, parent_code):
    lines = [line for line in code.splitlines() if line.strip()]
    diff = [line for line in difflib.unified_diff(parent_code.splitlines(), code.splitlines(), lineterm='')
            if line[:1] in '+-' and line[:3] not in ('+++', '---')]
    try:
        tree = ast.parse(code)
        functions = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
        syntax_ok = True
    except SyntaxError:
        functions = loops = None
        syntax_ok = False
    def count(pattern):
        return len(re.findall(pattern, code, flags=re.IGNORECASE))
    return {'code_lines': len(lines), 'code_diff_lines': len(diff), 'code_syntax_ok': syntax_ok,
            'code_functions': functions, 'code_loops': loops,
            'code_state_fields': len(set(re.findall(r'\bs\.([A-Za-z_]\w*)', code))),
            'code_uses_suspend': 'Suspend(' in code,
            'code_priority_terms': count(r'priority'),
            'code_memory_terms': count(r'\bram\b|memory'),
            'code_sort_calls': count(r'\bsort(ed)?\('),
            'code_failure_terms': count(r'FAILED|\boom\b|retry')}


def evaluation_features(evaluation, prefix):
    """Totals across scales; completion at the smallest scale, where resources are tightest."""
    results = (evaluation or {}).get('scale_results', {})
    ok = [r for r in results.values() if r.get('ok')]
    features = {f'{prefix}_ok': (evaluation or {}).get('ok'), f'{prefix}_score': (evaluation or {}).get('score'),
                f'{prefix}_failed_scales': len(results) - len(ok),
                f'{prefix}_ooms': sum(r.get('oom_count', 0) for r in ok) if ok else None,
                f'{prefix}_failures': sum(r.get('failures', 0) for r in ok) if ok else None,
                f'{prefix}_completion_1x': None}
    smallest = results.get(str(min((int(s) for s in results), default=1)), {})
    if smallest.get('ok'):
        groups = [smallest.get(p) or {} for p in ('query', 'interactive', 'batch')]
        arrived = sum(g.get('arrived', 0) for g in groups)
        features[f'{prefix}_completion_1x'] = (sum(g.get('completed', 0) for g in groups) / arrived
                                              if arrived else None)
    return features


def agent_features(row, directory):
    usage = row.get('generation_usage') or {}
    access = usage.get('agent_access') or {}
    tests = [read_json(path) for path in sorted((directory / 'internal_evals').glob('*/eval.json'))]
    tests = sorted((t for t in tests if t), key=lambda t: t.get('started_at_utc', ''))
    scores = [t['score'] for t in tests if t.get('ok') and t.get('score') is not None]
    return {'agent_input_tokens': usage.get('input_tokens'), 'agent_output_tokens': usage.get('output_tokens'),
            'agent_reasoning_tokens': usage.get('reasoning_tokens'),
            'agent_cache_read_tokens': usage.get('cache_read_tokens'),
            'agent_cache_write_tokens': usage.get('cache_write_tokens'),
            'agent_model_invocations': usage.get('model_invocations'),
            'agent_cost_usd': usage.get('observed_cost_usd'),
            'agent_repo_access': bool(access.get('markers')),
            'agent_self_tests': len(tests), 'agent_self_tests_valid': len(scores),
            'agent_first_test_score': scores[0] if scores else None,
            'agent_best_test_score': min(scores) if scores else None,
            'agent_last_test_score': scores[-1] if scores else None,
            # Fraction of consecutive valid self-tests that improved on the previous one.
            'agent_test_improving_steps': (sum(b < a for a, b in zip(scores, scores[1:])) / (len(scores) - 1)
                                           if len(scores) > 1 else None)}


def outcome_features(row, parent_row, children):
    score = row['score']
    parent_score = parent_row['score'] if parent_row else None
    descendants, frontier = [], list(children.get(row['path'], []))
    while frontier:
        child = frontier.pop()
        descendants.append(child)
        frontier.extend(children.get(child['path'], []))
    descendant_scores = [d['score'] for d in descendants if d['score'] is not None]
    best_below = min(descendant_scores) if descendant_scores else None
    return {'out_status': row['status'], 'out_score': score,
            'out_better_than_parent': (score < parent_score) if score is not None and parent_score else None,
            'out_ratio_to_parent': score / parent_score if score is not None and parent_score else None,
            'out_children': len(children.get(row['path'], [])), 'out_descendants': len(descendants),
            # The improvement target: best descendant relative to this candidate (<1 means it led somewhere).
            'out_best_descendant_ratio': best_below / score if best_below is not None and score else None}


def run_rows(directory, *, condition=None, repeat=None):
    directory = Path(directory)
    summary = json.loads((directory / 'results.json').read_text())
    predictions = read_json(directory / 'predictions.json') or {}
    rows = []
    for arm, data in summary['arms'].items():
        trajectory = data['trajectory']
        by_path = {r['path']: r for r in trajectory}
        children = {}
        for r in trajectory:
            if r.get('parent'):
                children.setdefault(r['parent'], []).append(r)
        for r in trajectory:
            candidate = Path(r['path'])
            parent = by_path.get(r.get('parent'))
            code_path = candidate / 'scheduler.py'
            code = code_path.read_text() if code_path.exists() else ''
            parent_code = (Path(parent['path']) / 'scheduler.py').read_text() if parent else code
            prediction = (predictions.get('candidates', {}).get(r['path']) or {}).get('prediction') or {}
            parent_eval = read_json(Path(parent['path']) / 'eval.json') if parent else None
            rows.append({'run': directory.name, 'condition': condition, 'repeat': repeat, 'arm': arm,
                         'iteration': r['iteration'],
                         'parent_iteration': parent['iteration'] if parent else None,
                         'cumulative_cost_usd': r.get('cumulative_cost_usd'),
                         **code_features(code, parent_code),
                         **evaluation_features(parent_eval, 'parent'),
                         **agent_features(r, candidate),
                         'pred_score': prediction.get('predicted_score'),
                         'pred_p_better': prediction.get('p_better_than_parent'),
                         'pred_will_fail': prediction.get('will_fail'),
                         **outcome_features(r, parent, children),
                         **{k.replace('child_', 'out_'): v for k, v in
                            evaluation_features(read_json(candidate / 'eval.json'), 'child').items()
                            if k not in ('child_ok', 'child_score')}})
    return rows


def write_signals(directory):
    """A run directory, or a sweep directory (sweep.json) combining all of its runs."""
    directory = Path(directory)
    manifest = read_json(directory / 'sweep.json')
    if manifest:
        rows = [row for cell in manifest['cells'] if (Path(cell['output']) / 'results.json').exists()
                for row in run_rows(cell['output'], condition=Path(cell['output']).parent.name,
                                    repeat=cell['repeat'])]
    else:
        rows = run_rows(directory)
    if rows:
        with (directory / 'signals.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path, help='a run directory or a sweep directory')
    args = parser.parse_args(argv)
    rows = write_signals(args.directory)
    print(f'{len(rows)} candidates -> {args.directory / "signals.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
