"""Offline LLM judge for RQ3: predict each candidate's score before seeing it.

Runs after a search, so it never changes what the search did or spent. For each
generated candidate it shows a model the parent's code and evaluation plus the
child's code (never the child's evaluation or the agent's self-tests) and asks
for a predicted score, failure, and probability of beating the parent. It then
compares predictions with the recorded outcomes.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import ROOT
from .generators import call_llm, get_iteration_feedback_prompt, write_json

PREDICTOR_MAX_TOKENS = 4096
# Generated candidates with code worth judging; unchanged code is trivially the parent.
JUDGED_STATUSES = ('improved', 'not_improved', 'evaluation_failed')


def build_prediction_prompt(parent_code, parent_evaluation, child_code):
    feedback = get_iteration_feedback_prompt(parent_code, parent_evaluation)
    # Keep the objective, settings, parent code and parent results; drop the request to improve.
    feedback = feedback[feedback.index('## Evaluation objective'):]
    return '\n\n'.join([
        (ROOT / 'src/prompts/prediction.md').read_text(encoding='utf-8-sig').rstrip(),
        '## Background\n\n' + (ROOT / 'src/prompts/context.md').read_text(encoding='utf-8-sig').rstrip(),
        feedback.rstrip(),
        f'## Child scheduler (not yet evaluated)\n\n```python\n{child_code.rstrip()}\n```\n'])


def parse_prediction(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'will_fail', 'predicted_score', 'p_better_than_parent', 'reason'}:
        raise ValueError('Expected will_fail, predicted_score, p_better_than_parent and reason')
    if not isinstance(value['will_fail'], bool):
        raise ValueError('will_fail must be a boolean')
    score = value['predicted_score']
    if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0):
        raise ValueError('predicted_score must be a nonnegative number or null')
    p = value['p_better_than_parent']
    if not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError('p_better_than_parent must be between 0 and 1')
    return value


def ranks(values):
    """Average ranks, so ties share a rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var else None


def summarize(records):
    """Agreement between predictions and outcomes, with the majority-class rate for reference."""
    judged = [r for r in records.values() if r.get('prediction')]
    scored = [r for r in judged if r['actual_score'] is not None and r['prediction']['predicted_score'] is not None]
    compared = [r for r in judged if r['parent_score'] is not None]
    better = [r['actual_score'] is not None and r['actual_score'] < r['parent_score'] for r in compared]
    failed = [r['actual_score'] is None for r in judged]
    def rate(flags):
        return sum(flags) / len(flags) if flags else None
    return {'judged': len(judged), 'invalid_predictions': sum(bool(r.get('error')) for r in records.values()),
            'score_spearman': spearman([r['prediction']['predicted_score'] for r in scored],
                                       [r['actual_score'] for r in scored]),
            'score_pairs': len(scored),
            'better_accuracy': rate([(r['prediction']['p_better_than_parent'] >= .5) == b
                                     for r, b in zip(compared, better)]),
            'better_brier': (sum((r['prediction']['p_better_than_parent'] - b) ** 2 for r, b in zip(compared, better))
                             / len(compared) if compared else None),
            'better_base_rate': rate(better),
            'fail_accuracy': rate([r['prediction']['will_fail'] == f for r, f in zip(judged, failed)]),
            'fail_base_rate': rate(failed)}


def predict_run(directory, model, *, tag='predictions', reasoning_effort=None, timeout=300,
                max_usd=None, spent=0.0, limit=None):
    """Judge every eligible candidate once per tag; returns the updated total spend."""
    directory = Path(directory).resolve()
    summary = json.loads((directory / 'results.json').read_text())
    output = directory / f'{tag}.json'
    report = json.loads(output.read_text()) if output.exists() else {'model': model, 'candidates': {}}
    if report['model'] != model:
        raise ValueError(f'{output.name} holds predictions from {report["model"]}; use another --tag')
    report.update(reasoning_effort=reasoning_effort, prompt='src/prompts/prediction.md')
    records = report['candidates']
    done = 0
    for arm, data in summary['arms'].items():
        by_path = {r['path']: r for r in data['trajectory']}
        for row in data['trajectory']:
            parent = by_path.get(row.get('parent'))
            if row['status'] not in JUDGED_STATUSES or not parent or row['path'] in records:
                continue
            if (max_usd is not None and spent >= max_usd) or (limit is not None and done >= limit):
                return spent
            candidate, parent_dir = Path(row['path']), Path(parent['path'])
            prompt = build_prediction_prompt((parent_dir / 'scheduler.py').read_text(),
                                             json.loads((parent_dir / 'eval.json').read_text()),
                                             (candidate / 'scheduler.py').read_text())
            folder = candidate / tag
            folder.mkdir(exist_ok=True)
            usage = {'estimated_cost_usd': None, 'observed_cost_usd': 0.0}
            record = {'arm': arm, 'iteration': row['iteration'], 'parent_score': parent['score'],
                      'actual_score': row['score'], 'prediction': None, 'usage': usage}
            try:
                raw = call_llm(folder, prompt, model, timeout, usage,
                               reasoning_effort=reasoning_effort, max_tokens=PREDICTOR_MAX_TOKENS)
                record['prediction'] = parse_prediction(raw)
            except (ValueError, TypeError, TimeoutError) as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            records[row['path']] = record
            done += 1
            spent += max(usage['observed_cost_usd'], usage['estimated_cost_usd'] or 0)
            report['spent_usd'] = report.get('spent_usd', 0) + max(usage['observed_cost_usd'],
                                                                    usage['estimated_cost_usd'] or 0)
            report['summary'] = summarize(records)
            write_json(output, report)
            if usage['estimated_cost_usd'] is None:
                raise RuntimeError(f'Prediction cost unknown for {row["path"]}; stopped')
    report['summary'] = summarize(records)
    write_json(output, report)
    return spent


def main(argv=None):
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', type=Path, nargs='+', help='run directories, or a sweep directory')
    parser.add_argument('--model', default=os.getenv('PREDICT_MODEL') or os.getenv('DIRECT_MODEL'))
    parser.add_argument('--reasoning-effort',
                        default=os.getenv('PREDICT_REASONING_EFFORT') or os.getenv('DIRECT_REASONING_EFFORT'))
    parser.add_argument('--tag', default='predictions', help='output name; use one tag per judge model')
    parser.add_argument('--max-usd', type=float, required=True, help='stop starting new predictions past this spend')
    parser.add_argument('--limit', type=int, default=None, help='at most this many new predictions per run')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error('Set --model, PREDICT_MODEL or DIRECT_MODEL')
    directories = []
    for directory in args.directories:
        manifest = directory / 'sweep.json'
        if manifest.exists():
            directories += [Path(c['output']) for c in json.loads(manifest.read_text())['cells']
                            if (Path(c['output']) / 'results.json').exists()]
        else:
            directories.append(directory)
    spent = 0.0
    for directory in directories:
        spent = predict_run(directory, args.model, tag=args.tag, reasoning_effort=args.reasoning_effort,
                            timeout=args.timeout, max_usd=args.max_usd, spent=spent, limit=args.limit)
        summary = json.loads((directory / f'{args.tag}.json').read_text()).get('summary', {})
        print(f'{directory}: {json.dumps(summary)}')
    print(f'Prediction spend: ${spent:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
