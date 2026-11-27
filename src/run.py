"""Independent best-so-far hill climbing with a shared evaluator."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from .accounting import Accounting
from .config import ROOT, EUDOXIA_REV, SCALES, TRACE, RESOURCE_TOLERANCE, default_agent_timeout, get_canonical_base_params
from .evaluator import evaluate_scheduler
from .generators import build_prompt, detect_opencode, generate_direct, generate_opencode, validate_code, write_json
from .selection import SELECTOR_MAX_TOKENS, select_parent


def best_candidate(pool):
    valid = [c for c in pool if c['evaluation'].get('ok') and c['evaluation'].get('score') is not None]
    return min(valid, key=lambda c: c['evaluation']['score']) if valid else pool[0]


def restore_pool(arm_summary, arm_dir):
    """Resume only fully recorded attempts; never overwrite an unfinished call."""
    if not arm_summary['cost_complete']:
        raise ValueError('Cannot resume with unknown accumulated cost')
    rows = arm_summary['trajectory']
    if not rows or [r['iteration'] for r in rows] != list(range(len(rows))):
        raise ValueError('Cannot resume a missing or noncontiguous trajectory')
    pool = []
    for row in rows:
        directory = Path(row['path'])
        if directory.parent.resolve() != arm_dir.resolve():
            raise ValueError('Candidate path does not belong to this arm')
        code_hash = hashlib.sha256((directory / 'scheduler.py').read_bytes()).hexdigest()
        if code_hash != row['scheduler_sha256'] or json.loads((directory / 'meta.json').read_text()) != row:
            raise ValueError('Recorded candidate changed; refusing resume')
        evaluation = json.loads((directory / 'eval.json').read_text())
        if evaluation['score'] != row['score']:
            raise ValueError('Recorded evaluation changed; refusing resume')
        if not row.get('duplicate_source'):
            pool.append({'path': row['path'], 'iteration': row['iteration'], 'evaluation': evaluation,
                         'scheduler_sha256': code_hash})
    if {p.resolve() for p in arm_dir.iterdir() if p.is_dir()} != {Path(r['path']).resolve() for r in rows}:
        raise ValueError('Unrecorded candidate exists; inspect it before resume')
    total = sum(r['observed_cost_usd'] for r in rows)
    if abs(total - arm_summary['known_cost_usd']) > 1e-9:
        raise ValueError('Accumulated cost does not match recorded attempts')
    return pool


def run_arm(arm, args, summary, save, *, generator=None, selector='best', accountant=None):
    generator = generator or arm
    pool = []
    arm_dir = args.output / 'candidates' / arm
    model = os.getenv('DIRECT_MODEL' if generator == 'direct' else 'OPENCODE_MODEL', '')
    reasoning = os.getenv('DIRECT_REASONING_EFFORT' if generator == 'direct' else 'OPENCODE_VARIANT')
    arm_summary = {'model': model, 'generator': generator, 'selector': selector,
                   'trajectory': [], 'stop_reason': 'iterations',
                   'known_cost_usd': 0.0, 'cost_complete': True, 'failed_attempts': 0}
    start = 0
    if getattr(args, 'resume', False):
        arm_summary = summary['arms'][arm]
        if (arm_summary['model'], arm_summary['generator'], arm_summary['selector']) != (model, generator, selector):
            raise ValueError('Model, generator or selector changed; refusing resume')
        pool = restore_pool(arm_summary, arm_dir)
        start = len(arm_summary['trajectory'])
        arm_summary['stop_reason'] = 'iterations'
    summary['arms'][arm] = arm_summary
    for iteration in range(start, args.iterations + 1):
        if iteration and args.budget_usd is not None and arm_summary['known_cost_usd'] >= args.budget_usd:
            arm_summary['stop_reason'] = 'budget'
            break
        started = time.monotonic()
        directory = arm_dir / ('000_seed' if iteration == 0 else f'{iteration:03d}')
        directory.mkdir(parents=True)
        parent = best_candidate(pool) if pool else None
        source = Path(parent['path']) / 'scheduler.py' if parent else ROOT / 'src/seed.py'
        shutil.copyfile(source, directory / 'scheduler.py')
        generation_usage = {'input_tokens': None, 'output_tokens': None, 'estimated_cost_usd': 0.0,
                 'observed_cost_usd': 0.0, 'model_invocations': 0}
        selection_usage = {'input_tokens': 0, 'output_tokens': 0, 'estimated_cost_usd': 0.0,
                           'observed_cost_usd': 0.0, 'model_invocations': 0}
        error = None
        error_fatal = False
        selection_only = False
        stage = 'selection'
        if iteration:
            if accountant is not None:
                accountant.set_context(arm, directory)
            try:
                if not model:
                    raise RuntimeError(f'Set {"DIRECT_MODEL" if generator == "direct" else "OPENCODE_MODEL"} in .env')
                remaining = None if args.budget_usd is None else args.budget_usd - arm_summary['known_cost_usd']
                parent = select_parent(pool, arm_summary['trajectory'], directory, strategy=selector,
                                       generator=generator, model=model, reasoning_effort=reasoning,
                                       timeout=args.timeout, remaining_budget=remaining, usage=selection_usage)
                source = Path(parent['path']) / 'scheduler.py'
                shutil.copyfile(source, directory / 'scheduler.py')
                if selection_usage['estimated_cost_usd'] is None:
                    raise RuntimeError('Selector cost unavailable; stopped before starting generation')
                selection_usage['observed_cost_usd'] = max(selection_usage['observed_cost_usd'],
                                                            selection_usage['estimated_cost_usd'])
                selection_only = remaining is not None and selection_usage['observed_cost_usd'] >= remaining
                if not selection_only:
                    stage = 'generation'
                    prompt = build_prompt(source.read_text(), parent['evaluation'])
                    (directory / 'prompt.txt').write_text(prompt)
                    generation_usage['estimated_cost_usd'] = None
                    if generator == 'direct':
                        generate_direct(directory, prompt, model, args.timeout, generation_usage)
                    else:
                        generate_opencode(directory, prompt, model, args.agent_timeout, generation_usage,
                                          prototype=args.prototype, trace=args.trace)
            except (SyntaxError, ValueError, TimeoutError) as exc:
                # Invalid/missing candidate code or a deadline consumes an attempt.
                error = f'{type(exc).__name__}: {exc}'
                error_fatal = stage == 'selection'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                # Provider/CLI/configuration failures and unexpected harness bugs
                # require attention; do not repeat paid requests blindly.
                error_fatal = True
        else:
            (directory / 'prompt.txt').write_text('Identical seed; no model call.\n')
        stop_path = directory / 'request-stop.json'
        # The proxy refused a request because this arm reached its target: a budget stop, not a failure.
        budget_stop = stop_path.exists() and json.loads(stop_path.read_text())['reason'] == 'target'
        if budget_stop and error:
            error_fatal = False
        if error or selection_only:
            evaluation = {'ok': False, 'score': None, 'error': error or 'Budget exhausted by selection'}
        else:
            stage = 'evaluation'
            try:
                evaluation = evaluate_scheduler(directory / 'scheduler.py', prototype=args.prototype, trace=args.trace)
            except Exception as exc:
                # Record the paid attempt before stopping on a harness failure.
                error = f'{type(exc).__name__}: {exc}'
                error_fatal = True
                evaluation = {'ok': False, 'score': None, 'error': error}
        write_json(directory / 'eval.json', evaluation)
        candidate = {'path': str(directory.resolve()), 'iteration': iteration, 'evaluation': evaluation}
        candidate_hash = hashlib.sha256((directory / 'scheduler.py').read_bytes()).hexdigest()
        # Byte-identical source is not added to the pool again; the attempt is still recorded.
        duplicate = any(c['scheduler_sha256'] == candidate_hash for c in pool)
        candidate['scheduler_sha256'] = candidate_hash
        if not duplicate:
            pool.append(candidate)
        best = best_candidate(pool)
        if not iteration:
            status = 'seed'
        elif selection_only:
            status = 'selection_only'
        elif error:
            status = f'{stage}_failed'
        elif not evaluation['ok']:
            status = 'evaluation_failed'
        elif (directory / 'scheduler.py').read_bytes() == source.read_bytes():
            status = 'unchanged'
        elif best is candidate:
            status = 'improved'
        else:
            status = 'not_improved'
        if iteration and not evaluation['ok'] and not selection_only:
            arm_summary['failed_attempts'] += 1
        usage = dict(generation_usage)
        parts = [selection_usage, generation_usage]
        # Count complete estimates and partial observations once per component.
        for part in parts:
            part['observed_cost_usd'] = max(part['observed_cost_usd'], part['estimated_cost_usd'] or 0)
        usage['estimated_cost_usd'] = (sum(p['estimated_cost_usd'] for p in parts)
                                       if all(p['estimated_cost_usd'] is not None for p in parts) else None)
        usage['observed_cost_usd'] = sum(p['observed_cost_usd'] for p in parts)
        for key in ('model_invocations', 'input_tokens', 'output_tokens'):
            usage[key] = sum(p[key] for p in parts) if all(p.get(key) is not None for p in parts) else None
        if accountant is not None:
            # Every request of this attempt, billed from the provider's usage by the proxy.
            usage.update(accountant.attempt_usage(directory))
        cost = usage['estimated_cost_usd']
        if cost is None:
            arm_summary['cost_complete'] = False
        arm_summary['known_cost_usd'] += usage['observed_cost_usd']
        decision_path = directory / 'selection.json'
        decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
        row = {'arm': arm, 'iteration': iteration, 'parent': parent['path'] if parent else None,
               'path': candidate['path'], 'model': model, 'score': evaluation['score'],
               'best_score': best['evaluation']['score'], 'best_path': best['path'],
               'elapsed_seconds': time.monotonic() - started, 'error': error,
               'status': status, 'error_fatal': error_fatal,
               'generator': generator, 'selector': selector,
               'selection_usage': selection_usage, 'generation_usage': generation_usage,
               'selected_parent_rank': decision.get('parent_rank'),
               'selection_fallback': decision.get('fallback', False),
               'scheduler_sha256': candidate_hash, 'duplicate_source': duplicate,
               **usage}
        row['cumulative_cost_usd'] = arm_summary['known_cost_usd']
        row['cumulative_elapsed_seconds'] = sum(r['elapsed_seconds'] for r in arm_summary['trajectory']) + row['elapsed_seconds']
        write_json(directory / 'meta.json', row)
        arm_summary['trajectory'].append(row)
        arm_summary['best_path'] = best['path']
        arm_summary['estimated_cost_usd'] = arm_summary['known_cost_usd'] if arm_summary['cost_complete'] else None
        save()
        def fmt(v):
            return f'{v:.4f}' if v is not None else 'n/a'
        print(f'{arm:<10} {iteration:>4} {fmt(row["score"]):>10} {fmt(row["best_score"]):>12} '
              f'{row["elapsed_seconds"]:>8.1f}s {fmt(cost):>14}', flush=True)
        if error:
            print(f'  {error}', file=sys.stderr)
        if selection_only or budget_stop:
            arm_summary['stop_reason'] = 'budget'
            break
        if error_fatal:
            arm_summary['stop_reason'] = 'harness_error' if stage == 'evaluation' else 'generation_error'
            break
        if not arm_summary['cost_complete']:
            arm_summary['stop_reason'] = 'cost_unknown'
            break
        if not iteration and not evaluation['ok']:
            arm_summary['stop_reason'] = 'seed_evaluation_failed'
            break
    best = best_candidate(pool) if pool else None
    test_trace = getattr(args, 'test_trace', None)
    if test_trace is not None and best is not None and best['evaluation'].get('ok') and arm_summary['stop_reason'] in ('budget', 'iterations'):
        # The training endpoint is fixed first; the held-out workload is evaluated once, after search.
        test = evaluate_scheduler(Path(best['path']) / 'scheduler.py', prototype=args.prototype, trace=test_trace)
        write_json(arm_dir / 'test_eval.json', test)
        arm_summary['test'] = {'path': best['path'], 'trace': test_trace.name, 'score': test['score']}
    save()


# Explicit allowlist: capture research inputs, never .env or credentials.
SNAPSHOT_FILES = ['src/' + name for name in [
    '__init__.py', 'run.py', 'config.py', 'evaluator.py', 'worker.py', 'generators.py',
    'seed.py', 'analysis.py', 'selection.py', 'accounting.py', 'opencode.json',
    'prompts/system.md', 'prompts/context.md', 'prompts/selection.md']] + ['pyproject.toml', 'uv.lock']


def snapshot_hashes(trace=TRACE, test_trace=None):
    """Hashes of the live inputs a snapshot would capture, for checking a resume."""
    paths = {name: ROOT / name for name in SNAPSHOT_FILES}
    for kind, path in (('train', trace), ('test', test_trace)):
        if path is not None:
            paths[f'scenarios/{kind}/{path.name}'] = path
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def snapshot_experiment(output, trace=TRACE, test_trace=None):
    """Copy the research inputs next to the results and record their hashes."""
    hashes = {}
    for name in SNAPSHOT_FILES:
        source = ROOT / name
        target = output / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for kind, path in (('train', trace), ('test', test_trace)):
        if path is None:
            continue
        name = f'scenarios/{kind}/{path.name}'
        target = output / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'packages': {name: version(name) for name in ['eudoxia', 'litellm', 'python-dotenv']},
        'sha256': hashes,
    }
    return manifest


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['direct', 'opencode', 'both'], default=os.getenv('RUN_MODE', 'both'))
    parser.add_argument('--selector', choices=['best', 'llm', 'compare'], default='best',
                        help='compare runs best and llm selectors with the same generator')
    parser.add_argument('--iterations', type=int, default=os.getenv('MAX_ITERATIONS', '3'), help='generated children per arm; 0 evaluates seeds only')
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--trace', type=Path, default=TRACE, help='CSV workload shared by both arms and agent self-tests')
    parser.add_argument('--test-trace', type=Path, default=None,
                        help='held-out CSV evaluated once on each arm\'s final training best; default: scenarios/test/<trace name>')
    parser.add_argument('--budget-usd', type=float, default=os.getenv('BUDGET_USD'),
                        help='target per arm; requests stop once recorded cost reaches it; the last request may overshoot')
    parser.add_argument('--timeout', type=float, default=300, help='direct model-call timeout in seconds')
    parser.add_argument('--agent-timeout', type=float, default=None,
                        help='OpenCode session seconds; default: --timeout plus two full evaluator timeout budgets')
    parser.add_argument('--output', type=Path, default=None, help='default: a fresh timestamped directory under runs/')
    parser.add_argument('--resume', action='store_true',
                        help='continue the interrupted run in --output from its saved state, with unchanged settings and source')
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error('--resume requires --output')
    if args.selector == 'compare' and args.mode == 'both':
        parser.error('--selector compare requires one generator: --mode direct or opencode')
    args.trace = args.trace.resolve()
    if not args.trace.is_file():
        parser.error(f'Trace file does not exist: {args.trace}')
    if args.test_trace is None and (ROOT / 'scenarios/test' / args.trace.name).is_file():
        args.test_trace = ROOT / 'scenarios/test' / args.trace.name
    if args.test_trace is not None:
        args.test_trace = args.test_trace.resolve()
        if not args.test_trace.is_file() or args.test_trace == args.trace:
            parser.error(f'Test trace must be an existing file different from the training trace: {args.test_trace}')
    import math
    if args.iterations < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or (
        args.agent_timeout is not None and (not math.isfinite(args.agent_timeout) or args.agent_timeout <= 0)) or (
        args.budget_usd is not None and (not math.isfinite(args.budget_usd) or args.budget_usd < 0)):
        parser.error('iterations must be nonnegative; timeouts positive and finite; budget nonnegative and finite')
    if args.agent_timeout is None:
        args.agent_timeout = default_agent_timeout(args.prototype, args.timeout)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.output = ROOT / 'runs' / f'experiment-{stamp}'
    args.output = args.output.resolve()
    generators = ['direct', 'opencode'] if args.mode == 'both' else [args.mode]
    selectors = ['best', 'llm'] if args.selector == 'compare' else [args.selector]
    conditions = [(f'{g}_{s}' if args.selector == 'compare' else g, g, s)
                  for g in generators for s in selectors]
    arms = [name for name, _, _ in conditions]
    if args.iterations > 0:
        for arm in generators:
            model = os.getenv('DIRECT_MODEL' if arm == 'direct' else 'OPENCODE_MODEL', '')
            if not model:
                parser.error(f'Set the model for {arm} in {ROOT / ".env"}')
            if model.startswith('openai/') and not os.getenv('OPENAI_API_KEY'):
                parser.error(f'Fill OPENAI_API_KEY in {ROOT / ".env"} before running')
            if arm == 'opencode':
                try:
                    detect_opencode()
                except (RuntimeError, OSError, TimeoutError) as exc:
                    parser.error(str(exc))
    if args.resume:
        saved = json.loads((args.output / 'run_config.json').read_text())
        # A resumed run must execute the same source, data and settings it started with.
        current = snapshot_hashes(args.trace, args.test_trace)
        changed = [name for name in saved['sha256'] if current.get(name) != saved['sha256'][name]]
        settings = {'mode': args.mode, 'selector': args.selector, 'iterations': args.iterations,
                    'budget_usd_per_arm': args.budget_usd, 'prototype': args.prototype,
                    'direct_timeout_seconds': args.timeout, 'agent_timeout_seconds': args.agent_timeout}
        changed += [key for key, value in settings.items() if saved[key] != value]
        if changed:
            parser.error(f'Cannot resume; changed since the run started: {", ".join(changed)}')
        summary = json.loads((args.output / 'results.json').read_text())
        summary['status'] = 'running'
    elif (args.output / 'run_config.json').exists() or (args.output / 'results.json').exists() or any(
        (args.output / 'candidates' / arm).exists() and any((args.output / 'candidates' / arm).iterdir()) for arm in arms):
        parser.error('Output already contains results. Use --output runs/NEW_NAME for a fresh experiment, or --resume.')
    validate_code((ROOT / 'src/seed.py').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        run_config = {'schema_version': 4, **snapshot_experiment(args.output, args.trace, args.test_trace),
                   'mode': args.mode, 'eudoxia_revision': EUDOXIA_REV, 'prototype': args.prototype,
                   'selector': args.selector, 'selector_max_completion_tokens': SELECTOR_MAX_TOKENS,
                   'conditions': {name: {'generator': g, 'selector': s} for name, g, s in conditions},
                   'objective': 'geometric mean priority-weighted latency, lower is better; all scales required',
                   'iterations': args.iterations, 'budget_usd_per_arm': args.budget_usd,
                   'budget_rule': 'every request is billed by the local proxy from provider usage; once an arm reaches the target, '
                                  'further requests are refused (also mid-session); the last request may overshoot; unknown billing stops',
                   'pool_admission': 'every attempt is recorded; byte-identical source is not added to the pool again',
                   'test_trace': args.test_trace.name if args.test_trace else None,
                   'test_trace_sha256': hashlib.sha256(args.test_trace.read_bytes()).hexdigest() if args.test_trace else None,
                   'direct_timeout_seconds': args.timeout, 'agent_timeout_seconds': args.agent_timeout,
                   'direct_reasoning_effort': os.getenv('DIRECT_REASONING_EFFORT') or None,
                   'opencode_variant': os.getenv('OPENCODE_VARIANT') or None,
                   'models': {name: os.getenv('DIRECT_MODEL' if g == 'direct' else 'OPENCODE_MODEL', '')
                              for name, g, _ in conditions},
                   'simulation_params': get_canonical_base_params(args.prototype),
                   'scales': SCALES, 'trace': args.trace.name, 'trace_source': str(args.trace),
                   'trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                   'resource_tolerance': RESOURCE_TOLERANCE}
    # Evaluate the saved workloads so an edit to a source CSV cannot split the arms.
    args.trace = args.output / 'snapshot/scenarios/train' / args.trace.name
    if args.test_trace is not None:
        args.test_trace = args.output / 'snapshot/scenarios/test' / args.test_trace.name
    if not args.resume:
        write_json(args.output / 'run_config.json', run_config)
        summary = {'status': 'running', 'run_config': 'run_config.json', 'arms': {}}
    def save():
        temp = args.output / 'results.json.tmp'
        write_json(temp, summary)
        temp.replace(args.output / 'results.json')
    print('arm        iter      score   best_score   elapsed estimated_cost')
    save()
    with ExitStack() as stack:
        accountant = None
        if args.iterations > 0:
            # Route every model request (litellm and OpenCode) through the billing proxy.
            models = json.loads((ROOT / 'src/opencode.json').read_text())['provider']['openai']['models']
            backend = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
            accountant = Accounting(args.output, backend, os.environ['OPENAI_API_KEY'],
                                    {name: m['cost'] for name, m in models.items()}, target=args.budget_usd)
            proxy = stack.enter_context(accountant.serve())
            os.environ.update(OPENAI_API_KEY='harness-proxy', OPENAI_BASE_URL=proxy, OPENAI_API_BASE=proxy)
        resume = args.resume
        for arm, generator, selector in conditions:
            # Resume only arms that already started; an arm with its held-out result is finished.
            args.resume = resume and arm in summary['arms']
            if args.resume and 'test' in summary['arms'][arm]:
                continue
            run_arm(arm, args, summary, save, generator=generator, selector=selector, accountant=accountant)
            if summary['arms'][arm]['stop_reason'] in ('harness_error', 'cost_unknown'):
                break
    failed = any(a['stop_reason'] in ('generation_error', 'harness_error', 'seed_evaluation_failed', 'cost_unknown') for a in summary['arms'].values())
    summary['status'] = 'needs_attention' if failed else 'complete'
    summary['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    from .analysis import write_analysis, plot_analysis
    report = write_analysis(args.output, summary)
    plot_analysis(args.output, report)
    print(f'Results: {args.output / "results.json"}')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
