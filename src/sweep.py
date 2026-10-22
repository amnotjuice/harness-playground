"""Repeat run.py over traces and repeats, one fresh output directory per cell.

Cells run sequentially, or --jobs at a time; with more than one job each cell's
output goes to sweep.log in its own directory, and each concurrent slot gets its
own OpenCode state directory. A cell whose directory already holds results is
never re-run, so an interrupted sweep resumes without repeating paid attempts; a
cell left unfinished must be inspected and removed by hand before it runs again.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .generators import write_json

DEFAULT_TRACES = [ROOT / f'scenarios/train/{name}.csv' for name in ('simple', 'medium', 'hard')]


def cell_name(trace):
    """train/hard.csv -> train-hard, so repeats on each split and level stay apart."""
    trace = Path(trace)
    return f'{trace.parent.name}-{trace.stem}' if trace.parent.name in ('train', 'test') else trace.stem


def plan_cells(output, traces, repeats):
    return [{'trace': str(Path(trace).resolve()), 'repeat': repeat,
             'output': str(output / cell_name(trace) / f'rep{repeat}')}
            for trace in traces for repeat in range(1, repeats + 1)]


def run_sweep(output, traces, repeats, mode, run_args, *, jobs=1, model=None, runner=subprocess.run):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'sweep.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                {'created_at_utc': datetime.now(timezone.utc).isoformat()})
    cells = plan_cells(output, traces, repeats)
    manifest.update(mode=mode, model=model, repeats=repeats, jobs=jobs, run_args=run_args, cells=cells)
    write_json(manifest_path, manifest)
    pending = []
    for cell in cells:
        results = Path(cell['output']) / 'results.json'
        if results.exists():
            cell['status'] = json.loads(results.read_text()).get('status', 'unknown')
            cell['skipped'] = True
        else:
            pending.append(cell)
    write_json(manifest_path, manifest)
    lock = threading.Lock()
    slots = queue.Queue()
    for slot in range(jobs):
        slots.put(slot)

    def execute(cell):
        name = Path(cell['output']).relative_to(output)
        command = [sys.executable, '-m', 'src.run', '--mode', mode, '--trace', cell['trace'],
                   '--output', cell['output'], *run_args]
        # Explicit, so a lost shell variable cannot silently fall back to the .env model.
        env = dict(os.environ, **({'DIRECT_MODEL': model, 'OPENCODE_MODEL': model} if model else {}))
        slot = slots.get()
        try:
            with lock:
                print(f'== start {name}' + (f' (slot {slot})' if jobs > 1 else ''), flush=True)
            if jobs == 1:
                code = runner(command, cwd=ROOT, env=env).returncode
            else:
                # Concurrent OpenCode processes must not share one state database.
                env['HARNESS_OPENCODE_STATE'] = str(ROOT / '.tools/state' / f'job{slot}')
                Path(cell['output']).mkdir(parents=True, exist_ok=True)
                with (Path(cell['output']) / 'sweep.log').open('w') as log:
                    code = runner(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        finally:
            slots.put(slot)
        results = Path(cell['output']) / 'results.json'
        with lock:
            cell['exit_code'] = code
            cell['status'] = (json.loads(results.read_text()).get('status', 'unknown')
                              if results.exists() else 'not_started')
            write_json(manifest_path, manifest)
            print(f'== done {name}: {cell["status"]}', flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(execute, pending))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Arguments after -- are passed to every run.py call, '
                                            'e.g. -- --budget-usd 2 --budget-guard --iterations 1000')
    parser.add_argument('--mode', default='both', help='run.py --mode for every cell')
    parser.add_argument('--traces', type=Path, nargs='+', default=DEFAULT_TRACES)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--model', default=None,
                        help='set DIRECT_MODEL and OPENCODE_MODEL for every run, overriding .env '
                             '(e.g. openai/gpt-6-luna)')
    parser.add_argument('--jobs', type=int, default=1,
                        help='cells to run at once; mind API rate limits and CPU cores')
    parser.add_argument('--output', type=Path, default=None, help='default: runs/sweep-<timestamp>')
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    run_args = argv[split + 1:]
    if args.repeats < 1 or args.jobs < 1:
        parser.error('--repeats and --jobs must be at least 1')
    for flag in ('--mode', '--trace', '--output'):
        if flag in run_args:
            parser.error(f'{flag} is set per cell by the sweep; do not pass it after --')
    missing = [str(t) for t in args.traces if not t.is_file()]
    if missing:
        parser.error(f'Trace files do not exist: {", ".join(missing)}')
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.output = ROOT / 'runs' / f'sweep-{stamp}'
    manifest = run_sweep(args.output, args.traces, args.repeats, args.mode, run_args, jobs=args.jobs,
                         model=args.model)
    from .analysis import write_sweep_analysis
    write_sweep_analysis(args.output)
    print(f'Sweep: {Path(args.output).resolve() / "report.md"}')
    return int(any(cell.get('status') != 'complete' for cell in manifest['cells']))


if __name__ == '__main__':
    sys.exit(main())
