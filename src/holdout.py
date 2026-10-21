"""Re-score each arm's best-so-far chain on the held-out test trace; never calls a model.

Absolute latencies differ between traces, so each score is also reported relative
to the seed on the same trace. The train/test gap compares those ratios.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .evaluator import evaluate_scheduler
from .generators import write_json


def matching_test_trace(trace):
    """train/<name>.csv -> test/<name>.csv; bench_canonical_train -> bench_canonical_test."""
    trace = Path(trace)
    if trace.parent.name == 'train':
        candidate = trace.parent.parent / 'test' / trace.name
    elif '_train' in trace.stem:
        candidate = trace.with_name(trace.name.replace('_train', '_test'))
    else:
        raise ValueError(f'No held-out trace is known for {trace}; pass --test-trace')
    if not candidate.is_file():
        raise FileNotFoundError(f'Held-out trace missing: {candidate}')
    return candidate


def best_chain(trajectory):
    """Each distinct best-so-far candidate in the order it became best."""
    chain = []
    for row in trajectory:
        if row.get('best_score') is not None and (not chain or chain[-1]['path'] != row['best_path']):
            chain.append({'path': row['best_path'], 'iteration': row['iteration'],
                          'train_score': row['best_score']})
    return chain


def score_run(directory, *, test_trace=None, final_only=False, evaluate=evaluate_scheduler):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'run_config.json').read_text())
    summary = json.loads((directory / 'results.json').read_text())
    test_trace = Path(test_trace or matching_test_trace(config['trace_source'])).resolve()
    trace_sha = hashlib.sha256(test_trace.read_bytes()).hexdigest()
    output = directory / 'holdout.json'
    previous = json.loads(output.read_text()) if output.exists() else {}
    # Reuse earlier scores only for identical scheduler bytes on an identical trace.
    cache = {}
    if previous.get('test_trace_sha256') == trace_sha and previous.get('prototype') == config['prototype']:
        for arm in previous['arms'].values():
            for entry in arm['chain']:
                cache[entry['scheduler_sha256']] = entry['test_evaluation']
    report = {'test_trace': str(test_trace), 'test_trace_sha256': trace_sha,
              'prototype': config['prototype'], 'arms': {}}
    for arm, data in summary['arms'].items():
        chain = best_chain(data['trajectory'])
        if final_only and len(chain) > 2:
            chain = [chain[0], chain[-1]]
        for entry in chain:
            scheduler = Path(entry['path']) / 'scheduler.py'
            entry['scheduler_sha256'] = hashlib.sha256(scheduler.read_bytes()).hexdigest()
            if entry['scheduler_sha256'] not in cache:
                cache[entry['scheduler_sha256']] = evaluate(scheduler, prototype=config['prototype'],
                                                            trace=test_trace)
            entry['test_evaluation'] = cache[entry['scheduler_sha256']]
            entry['test_score'] = entry['test_evaluation'].get('score')
        for entry in chain:
            seed = chain[0]
            entry['train_ratio'] = entry['train_score'] / seed['train_score']
            entry['test_ratio'] = (entry['test_score'] / seed['test_score']
                                   if entry['test_score'] is not None and seed['test_score'] else None)
            # Positive gap: the candidate improved less on unseen jobs than on the search trace.
            entry['generalization_gap'] = (entry['test_ratio'] - entry['train_ratio']
                                           if entry['test_ratio'] is not None else None)
        report['arms'][arm] = {'chain': chain}
        write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+', help='run directories containing results.json')
    parser.add_argument('--test-trace', type=Path, default=None, help='default: the matching test/ trace')
    parser.add_argument('--final-only', action='store_true', help='score only the seed and the final best')
    args = parser.parse_args(argv)
    for directory in args.directories:
        report = score_run(directory, test_trace=args.test_trace, final_only=args.final_only)
        for arm, data in report['arms'].items():
            final = data['chain'][-1] if data['chain'] else {}
            print(f"{directory.name} {arm}: train x{final.get('train_ratio', float('nan')):.3f} "
                  f"test x{final.get('test_ratio') or float('nan'):.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
