"""Evaluate one scheduler with a top-level constant set to each of several values (the agent's ./sweep).

The agent names a module-level constant (`RAM_GB = 48.0`) and the values to try. Each value gets
its own copy of scheduler.py, evaluated exactly as ./evaluate would, several at a time. The agent's
scheduler.py is never changed. Each sweep is archived under --output-dir with full results; the
agent gets a compact table and a copy of each variant under --copy-to.
"""
import argparse
import ast
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from .config import TRACE
from .evaluator import evaluate_scheduler

MAX_VALUES = 8
PARALLEL = 4


def substitute(code, name, value):
    """code with the single module-level `name = ...` assignment set to the literal `value`."""
    tree = ast.parse(code)
    targets = [node.value for node in tree.body
               if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
    targets += [node.value for node in tree.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == name and node.value is not None]
    if len(targets) != 1:
        found = 'no' if not targets else f'{len(targets)}'
        raise ValueError(f'scheduler.py has {found} module-level assignments to {name}; '
                         f'define it exactly once at the top level, e.g. `{name} = 48.0`')
    node = targets[0]
    if node.lineno != node.end_lineno:
        raise ValueError(f'The value assigned to {name} must fit on one line')
    lines = code.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    # ast offsets are UTF-8 byte offsets.
    raw = line.encode()
    lines[node.lineno - 1] = (raw[:node.col_offset] + value.encode() + raw[node.end_col_offset:]).decode()
    return ''.join(lines)


def parse_value(text):
    """A Python literal as written by the agent (numbers, strings, tuples, ...), kept as its source."""
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError):
        raise ValueError(f'Not a Python literal: {text!r}') from None
    return text


def summary_row(value, evaluation):
    if not evaluation['ok']:
        failed = {s: r.get('error', 'unknown error') for s, r in evaluation['scale_results'].items() if not r.get('ok')}
        return {'value': value, 'ok': False, 'score': None, 'failed_scales': failed}
    results = evaluation['scale_results']
    return {'value': value, 'ok': True, 'score': round(evaluation['score'], 3),
            'latency_by_scale': {s: round(r['latency'], 2) for s, r in results.items()},
            'ooms_by_scale': {s: r.get('oom_count') for s, r in results.items()}}


def table(rows):
    scales = next((list(r['latency_by_scale']) for r in rows if r['ok']), [])
    head = ['value', 'score'] + [f'{s}x latency' for s in scales] + ['OOMs (all scales)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    for r in rows:
        if r['ok']:
            cells = [r['value'], f"{r['score']:.3f}"] + [f"{r['latency_by_scale'][s]:.2f}" for s in scales]
            cells.append(str(sum(v or 0 for v in r['ooms_by_scale'].values())))
        else:
            cells = [r['value'], 'ineligible'] + [''] * len(scales) + [json.dumps(r['failed_scales'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='a module-level constant in scheduler.py')
    parser.add_argument('values', nargs='+', help='Python literals to try, e.g. 40 48 56 64')
    parser.add_argument('--scheduler', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='archive; a fresh subdirectory per sweep')
    parser.add_argument('--copy-to', type=Path, help="the agent's copy, numbered per sweep")
    parser.add_argument('--trace', type=Path, default=TRACE)
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--max-values', type=int, default=MAX_VALUES)
    parser.add_argument('--parallel', type=int, default=PARALLEL)
    args = parser.parse_args(argv)
    try:
        if not args.name.isidentifier():
            raise ValueError(f'Not a Python name: {args.name!r}')
        if len(args.values) > args.max_values:
            raise ValueError(f'At most {args.max_values} values per sweep; got {len(args.values)}')
        if len(set(args.values)) != len(args.values):
            raise ValueError('Values repeat')
        values = [parse_value(v) for v in args.values]
        code = args.scheduler.read_text()
        variants = {v: substitute(code, args.name, v) for v in values}
    except (ValueError, SyntaxError, OSError) as exc:
        print(f'Sweep not run: {exc}')
        return 2
    sweep = args.output_dir.resolve() / uuid4().hex
    sweep.mkdir(parents=True)
    (sweep / 'request.json').write_text(json.dumps({'name': args.name, 'values': values}, indent=2) + '\n')
    paths = []
    for index, value in enumerate(values, 1):
        directory = sweep / f'{index:02d}'
        directory.mkdir()
        (directory / 'scheduler.py').write_text(variants[value])
        paths.append(directory)

    def run(directory):
        evaluation = evaluate_scheduler(directory / 'scheduler.py', prototype=args.prototype, trace=args.trace)
        (directory / 'eval.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n')
        return evaluation

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        evaluations = list(pool.map(run, paths))
    rows = [summary_row(v, e) for v, e in zip(values, evaluations)]
    (sweep / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Sweep of {args.name} (scheduler.py unchanged):')
    print(table(rows))
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        number = 1 + max((int(p.name) for p in args.copy_to.iterdir() if p.name.isdigit()), default=0)
        shutil.copytree(sweep, args.copy_to / str(number))
        print(f'Saved each variant and its full result to ./{args.copy_to.name}/{number}/<nn>/ '
              '(scheduler.py, eval.json; summary.json lists the values in order)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
