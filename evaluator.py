"""Shared evaluator adapted from summer2026/tool/iterate.py and its worker."""
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import ROOT, SCALES, TRACE, RESOURCE_TOLERANCE, get_canonical_base_params


def evaluate_scheduler(scheduler_path: Path, *, prototype=False, trace=TRACE, scales=None):
    for path in (scheduler_path, trace, ROOT / 'worker.py'):
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
                [sys.executable, str(ROOT / 'worker.py'), str(scheduler_path.resolve()),
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


def main(argv=None):
    """Agent self-evaluation uses the same evaluator and archives each input."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='a fresh subdirectory is created per evaluation')
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--trace', type=Path, default=TRACE)
    args = parser.parse_args(argv)
    if not args.scheduler.is_file():
        parser.error('Scheduler file does not exist')
    if not args.trace.is_file():
        parser.error('Trace file does not exist')
    directory = args.output_dir.resolve() / uuid4().hex
    directory.mkdir(parents=True)
    scheduler = directory / 'scheduler.py'
    shutil.copyfile(args.scheduler, scheduler)
    # The archived bytes, rather than an agent's subsequently edited file, are evaluated.
    evaluation = evaluate_scheduler(scheduler, prototype=args.prototype, trace=args.trace)
    rendered = json.dumps(evaluation, indent=2, allow_nan=False) + '\n'
    (directory / 'eval.json').write_text(rendered)
    print(f'Evaluation artifacts: {directory}')
    print(rendered, end='')
    return 0 if evaluation['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
