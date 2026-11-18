"""Shared evaluator adapted from summer2026/tool/iterate.py and its worker."""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import ROOT, SCALES, TRACE, RESOURCE_TOLERANCE, get_canonical_base_params


def evaluate_scheduler(scheduler_path: Path, *, prototype=False, trace=TRACE, scales=None):
    for path in (scheduler_path, trace, ROOT / 'src/worker.py'):
        if not path.is_file():
            raise FileNotFoundError(f'Evaluation input missing: {path}')
    params = get_canonical_base_params(prototype)
    scales = SCALES if scales is None else scales
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    results = {}
    for scale in scales:
        scale_started = time.monotonic()
        scaled = dict(params, cpus_per_pool=params['cpus_per_pool'] * scale,
                      ram_gb_per_pool=params['ram_gb_per_pool'] * scale)
        try:
            proc = subprocess.run(
                [sys.executable, '-m', 'src.worker', str(scheduler_path.resolve()),
                 str(trace.resolve()), json.dumps(scaled)],
                capture_output=True, text=True, cwd=ROOT,
                timeout=params['subprocess_timeout'])
            markers = [s[10:] for s in proc.stdout.splitlines() if s.startswith('__RESULT__')]
            result = json.loads(markers[-1]) if markers else {
                'ok': False, 'error': f'worker exit {proc.returncode}: {proc.stderr[-1000:]}'}
            if proc.returncode:
                result = {'ok': False, 'error': f'worker exit {proc.returncode}: {proc.stderr[-1000:]}'}
            if result.get('ok') and (not isinstance(result.get('latency'), (int, float))
                                    or not math.isfinite(result['latency']) or result['latency'] < 0):
                result = {'ok': False, 'error': 'nonfinite or negative latency'}
        except subprocess.TimeoutExpired:
            result = {'ok': False, 'error': 'subprocess timeout'}
        results[str(scale)] = dict(result, elapsed_seconds=time.monotonic() - scale_started)
    ok = all(r.get('ok') for r in results.values()) and bool(results)
    score = (math.exp(sum(math.log(max(r['latency'], 1e-12)) for r in results.values())
                      / len(results)) if ok else None)
    return {'ok': ok, 'score': score, 'scale_results': results, 'params': params,
            'started_at_utc': started_at, 'resource_tolerance': RESOURCE_TOLERANCE,
            'scales': scales, 'trace': trace.name,
            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'scheduler_sha256': hashlib.sha256(scheduler_path.read_bytes()).hexdigest(),
            'elapsed_seconds': time.monotonic() - started}


FEEDBACK = ('full', 'score')


def agent_view(evaluation, feedback='full'):
    """What a self-testing agent is shown. 'score' keeps eligibility and the aggregate score, plus the
    error of any failed scale so a broken scheduler can still be fixed; no other per-scale results."""
    if feedback == 'full':
        return evaluation
    view = {'ok': evaluation['ok'], 'score': evaluation['score']}
    failed = {scale: r.get('error', 'unknown error')
              for scale, r in evaluation.get('scale_results', {}).items() if not r.get('ok')}
    if failed:
        view['failed_scales'] = failed
    return view


def claim_run(directory, limit):
    """Number of this self-test (1..limit), or None once all are used.

    Each slot is a file created exclusively in the archive directory, which is outside the agent's
    workspace, so concurrent runs cannot share a slot and the agent cannot reset the count."""
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, limit + 1):
        try:
            os.close(os.open(directory / f'.self-test-{number}', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return number
        except FileExistsError:
            continue
    return None


def main(argv=None):
    """Agent self-evaluation uses the same evaluator and archives each input."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='a fresh subdirectory is created per evaluation')
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--trace', type=Path, default=TRACE)
    parser.add_argument('--copy-to', type=Path,
                        help="also copy each tested version and result to a numbered subdirectory here (the agent's "
                             'workspace), so the agent never needs to read outside it')
    parser.add_argument('--feedback', choices=FEEDBACK, default='full',
                        help="what is printed and copied: 'full' results, or only eligibility and the aggregate "
                             "score ('score'); the archive under --output-dir always keeps the full result")
    parser.add_argument('--max-runs', type=int,
                        help='refuse to evaluate once this many runs have used this --output-dir')
    args = parser.parse_args(argv)
    if args.max_runs is not None and args.max_runs < 1:
        parser.error('--max-runs must be at least 1')
    if not args.scheduler.is_file():
        parser.error('Scheduler file does not exist')
    if not args.trace.is_file():
        parser.error('Trace file does not exist')
    run_number = None
    if args.max_runs is not None:
        run_number = claim_run(args.output_dir.resolve(), args.max_runs)
        if run_number is None:
            print(f'Self-test limit reached: all {args.max_runs} runs of ./evaluate allowed in this session '
                  'are used. Nothing was evaluated. Leave your final policy in scheduler.py.')
            return 3
    directory = args.output_dir.resolve() / uuid4().hex
    directory.mkdir(parents=True)
    scheduler = directory / 'scheduler.py'
    shutil.copyfile(args.scheduler, scheduler)
    # The archived bytes, rather than an agent's subsequently edited file, are evaluated.
    evaluation = evaluate_scheduler(scheduler, prototype=args.prototype, trace=args.trace)
    (directory / 'eval.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n')
    rendered = json.dumps(agent_view(evaluation, args.feedback), indent=2, allow_nan=False) + '\n'
    if args.copy_to:
        # OpenCode run ends the session when the agent reads outside its workspace, so the
        # archive's location is never printed; the archive itself stays the record.
        args.copy_to.mkdir(parents=True, exist_ok=True)
        number = 1 + max((int(p.name) for p in args.copy_to.iterdir() if p.name.isdigit()), default=0)
        copy = args.copy_to / str(number)
        shutil.copytree(directory, copy)
        # The agent's copy holds only what it is shown.
        (copy / 'eval.json').write_text(rendered)
        print(f'Saved this version and result to ./{args.copy_to.name}/{number}/ (scheduler.py, eval.json)')
    print(rendered, end='')
    if run_number is not None:
        print(f'This was self-test {run_number} of {args.max_runs} allowed in this session.')
    return 0 if evaluation['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
