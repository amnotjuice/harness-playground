"""OpenEvolve arm: OpenEvolve's own search loop on the harness's evaluator, model calls and records.

OpenEvolve (MAP-Elites islands, diff-based edits, top and diverse programs in the prompt) chooses
parents and builds prompts. The harness supplies three hooks so the arm is comparable with the others:

- every model call goes through `call_llm` (same model, reasoning effort, pricing, retries);
- every candidate is scored by `evaluate_scheduler` on the run's trace;
- every iteration is written to candidates/openevolve/NNN/ with the same eval.json and meta.json
  rows as the other arms, so analysis, holdout and signals read it unchanged.

OpenEvolve runs iterations in worker processes. Each worker iteration is wrapped (recorded_iteration);
the wrapper, the model client and the evaluator all run in that worker and share the CURRENT context.
One worker is used, so iterations, costs and budget checks happen strictly one after another.
"""
import asyncio
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

from openevolve import process_parallel
from openevolve.evaluation_result import EvaluationResult

from .config import ROOT
from .evaluator import evaluate_scheduler
from .generators import call_llm, evaluation_diagnostics, evaluation_rules, write_json
from .worker import validate_code

SETTINGS_ENV = 'HARNESS_OPENEVOLVE_SETTINGS'
STOP_FILE = 'stop.json'
# Score shown to OpenEvolve for an ineligible candidate: below any valid one (a valid score can
# reach at most 2 * max_job_seconds = 720 s, i.e. combined_score -720).
INVALID_COMBINED_SCORE = -1440.0

_ORIGINAL_ITERATION = process_parallel._run_iteration_worker
CURRENT = None  # the iteration running in this worker process
_SEED_EVALUATIONS = {}  # parent process only: scheduler sha256 -> the seed's harness evaluation


class BudgetStop(RuntimeError):
    """Raised instead of a model call once the arm's budget rule says stop."""


def load_settings():
    return json.loads(Path(os.environ[SETTINGS_ENV]).read_text())


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def recorded_rows(arm_dir):
    rows = [json.loads(p.read_text()) for p in Path(arm_dir).glob('*/meta.json')]
    return sorted(rows, key=lambda r: r['iteration'])


def request_stop(settings, reason, detail=None):
    path = Path(settings['work_dir']) / STOP_FILE
    if not path.exists():
        write_json(path, {'reason': reason, 'detail': detail})


def stop_requested(settings):
    path = Path(settings['work_dir']) / STOP_FILE
    return json.loads(path.read_text()) if path.exists() else None


def budget_refusal(settings, rows):
    """The run.py stopping rule, checked before each model call instead of before each round."""
    from .run import next_round_reserve
    budget = settings['budget_usd']
    if budget is None:
        return None
    known = sum(r['observed_cost_usd'] for r in rows)
    if any(r['iteration'] for r in rows) and known >= budget:
        return {'reason': 'budget'}
    if settings['budget_guard'] and any(r['iteration'] for r in rows):
        slack = settings['budget_slack_usd']
        reserve = next_round_reserve(rows, 'direct', slack)
        if budget + slack - known < reserve:
            return {'reason': 'budget_reserve', 'next_round_reserve_usd': reserve}
    return None


def to_openevolve(evaluation):
    """Harness evaluation -> OpenEvolve metrics (higher is better) plus the diagnostics as an artifact."""
    if evaluation.get('ok') and evaluation.get('score') is not None:
        metrics = {'combined_score': -evaluation['score'], 'weighted_latency_s': evaluation['score'],
                   'eligible': 1.0}
        for scale, result in evaluation['scale_results'].items():
            metrics[f'latency_{scale}x_s'] = result['latency']
    else:
        metrics = {'combined_score': INVALID_COMBINED_SCORE, 'eligible': 0.0}
    if 'scale_results' in evaluation:
        artifacts = {'harness_evaluation': evaluation_diagnostics(evaluation)}
    else:
        artifacts = {'harness_evaluation': f'Not evaluated: {evaluation.get("error")}'}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path):
    """OpenEvolve's evaluate(): the harness evaluator, recorded into the current iteration's directory."""
    settings = load_settings()
    code = Path(program_path).read_text()
    context = CURRENT
    if context is None:
        # The controller scores the starting program in the parent process; the harness already did.
        cached = _SEED_EVALUATIONS.get(sha256(code))
        if cached is None:
            raise RuntimeError('OpenEvolve evaluated a program outside an iteration that is not the seed')
        return to_openevolve(cached)
    directory = context['directory']
    (directory / 'scheduler.py').write_text(code)
    context['evaluated_code'] = code
    try:
        validate_code(code)
    except (SyntaxError, ValueError) as exc:
        # As in run.py: unusable code from the model is a failed generation, not an evaluation.
        context['generation_error'] = f'{type(exc).__name__}: {exc}'
        return to_openevolve({'ok': False, 'score': None, 'error': context['generation_error']})
    try:
        evaluation = evaluate_scheduler(directory / 'scheduler.py', prototype=settings['prototype'],
                                        trace=Path(settings['trace']))
    except Exception as exc:
        context['harness_error'] = f'{type(exc).__name__}: {exc}'
        evaluation = {'ok': False, 'score': None, 'error': context['harness_error']}
    context['evaluation'] = evaluation
    return to_openevolve(evaluation)


class HarnessLLM:
    """OpenEvolve model client (LLMModelConfig.init_client) that spends through call_llm."""

    def __init__(self, model_config):
        self.settings = load_settings()
        self.last_usage = None

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context(None, [{'role': 'user', 'content': prompt}], **kwargs)

    async def generate_with_context(self, system_message, messages, **kwargs):
        context = CURRENT
        if context is None:
            raise RuntimeError('Model call outside a recorded iteration')
        refusal = budget_refusal(self.settings, recorded_rows(self.settings['arm_dir']))
        if refusal:
            context['refused'] = refusal
            request_stop(self.settings, refusal['reason'], refusal)
            raise BudgetStop(refusal['reason'])
        if context['usage']['model_invocations']:
            raise RuntimeError('A second model call in one iteration is not supported')
        prompt = '\n\n'.join(m['content'] for m in messages)
        usage = context['usage']
        usage['estimated_cost_usd'] = None
        try:
            raw = call_llm(context['directory'], prompt, self.settings['model'], self.settings['timeout'], usage,
                           reasoning_effort=self.settings['reasoning_effort'], system=system_message)
        except TimeoutError as exc:
            context['generation_error'] = f'TimeoutError: {exc}'
            raise
        except Exception as exc:
            # Provider/configuration failures stop the arm, as in run.py: never repeat paid calls blindly.
            context['generation_error'] = f'{type(exc).__name__}: {exc}'
            context['fatal'] = True
            raise
        # OpenEvolve's parent adds these up and drops the child if any is not a number.
        prompt_tokens, completion_tokens = usage.get('input_tokens') or 0, usage.get('output_tokens') or 0
        self.last_usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                           'total_tokens': prompt_tokens + completion_tokens, 'model': self.settings['model']}
        return raw


def init_client(model_config):
    return HarnessLLM(model_config)


def recorded_iteration(iteration, db_snapshot, parent_id, inspiration_ids):
    """Run one OpenEvolve iteration and record it as candidates/openevolve/NNN/."""
    global CURRENT
    settings = load_settings()
    if stop_requested(settings):
        # Already-queued iterations after a stop make no model call and leave no record.
        return process_parallel.SerializableResult(error='harness stop requested', iteration=iteration)
    arm_dir = Path(settings['arm_dir'])
    directory = arm_dir / f'{iteration:03d}'
    directory.mkdir(parents=True)
    parent_code = db_snapshot['programs'][parent_id]['code']
    # Until a child exists, the candidate is the parent (as run.py copies the parent first).
    (directory / 'scheduler.py').write_text(parent_code)
    usage = {'input_tokens': None, 'output_tokens': None, 'estimated_cost_usd': 0.0,
             'observed_cost_usd': 0.0, 'model_invocations': 0}
    CURRENT = context = {'directory': directory, 'usage': usage}
    started = time.monotonic()
    try:
        result = _ORIGINAL_ITERATION(iteration, db_snapshot, parent_id, inspiration_ids)
    finally:
        CURRENT = None
    if context.get('refused'):
        shutil.rmtree(directory)
        return result
    rows = recorded_rows(arm_dir)
    evaluation = context.get('evaluation')
    error = context.get('generation_error') or context.get('harness_error')
    if evaluation is None:
        # OpenEvolve rejected the response (no diff, unmatched SEARCH, identical code) or code was unusable.
        error = error or result.error or 'OpenEvolve produced no candidate'
        evaluation = {'ok': False, 'score': None, 'error': error}
    write_json(directory / 'eval.json', evaluation)
    if 'prompt.txt' not in {p.name for p in directory.iterdir()}:
        (directory / 'prompt.txt').write_text('No model call.\n')
    code = (directory / 'scheduler.py').read_text()
    valid = [r for r in rows if r['score'] is not None]
    previous_best = min(valid, key=lambda r: r['score']) if valid else None
    if context.get('harness_error'):
        status = 'evaluation_failed'
    elif context.get('evaluation') is None:
        status = 'generation_failed'
    elif not evaluation['ok']:
        status = 'evaluation_failed'
    elif code == parent_code:
        status = 'unchanged'
    elif previous_best is None or evaluation['score'] < previous_best['score']:
        status = 'improved'
    else:
        status = 'not_improved'
    if status == 'improved':
        best_score, best_path = evaluation['score'], str(directory.resolve())
    elif previous_best:
        best_score, best_path = previous_best['score'], previous_best['path']
    else:
        best_score, best_path = None, str(directory.resolve())
    parent_sha = sha256(parent_code)
    parent_path = next((r['path'] for r in rows if r['scheduler_sha256'] == parent_sha), None)
    usage['observed_cost_usd'] = max(usage['observed_cost_usd'], usage['estimated_cost_usd'] or 0)
    selection_usage = {'input_tokens': 0, 'output_tokens': 0, 'estimated_cost_usd': 0.0,
                       'observed_cost_usd': 0.0, 'model_invocations': 0}
    child = result.child_program_dict or {}
    elapsed = time.monotonic() - started
    row = {'arm': settings['arm'], 'iteration': iteration, 'parent': parent_path,
           'path': str(directory.resolve()), 'model': settings['model'], 'score': evaluation['score'],
           'best_score': best_score, 'best_path': best_path, 'elapsed_seconds': elapsed, 'error': error,
           'status': status, 'error_fatal': bool(context.get('fatal') or context.get('harness_error')),
           'generator': 'openevolve', 'selector': 'openevolve',
           'selection_usage': selection_usage, 'generation_usage': dict(usage),
           'selected_parent_rank': None, 'selection_fallback': False,
           'scheduler_sha256': hashlib.sha256((directory / 'scheduler.py').read_bytes()).hexdigest(),
           **usage,
           'openevolve': {'parent_id': parent_id, 'inspiration_ids': list(inspiration_ids),
                          'program_id': child.get('id'), 'island': result.target_island,
                          'evolution_mode': result.evolution_mode,
                          'changes': (child.get('metadata') or {}).get('changes'),
                          'rejected': result.error}}
    row['cumulative_cost_usd'] = sum(r['observed_cost_usd'] for r in rows) + row['observed_cost_usd']
    row['cumulative_elapsed_seconds'] = sum(r['elapsed_seconds'] for r in rows) + elapsed
    write_json(directory / 'meta.json', row)
    if context.get('fatal'):
        request_stop(settings, 'generation_error', error)
    elif context.get('harness_error'):
        request_stop(settings, 'harness_error', error)
    elif usage['estimated_cost_usd'] is None and settings['budget_guard']:
        request_stop(settings, 'cost_unknown', usage.get('cost_error'))
    return result


def system_message(seed_evaluation):
    """The task text every arm gets (system.md with context.md), plus the objective and settings."""
    system = (ROOT / 'src/prompts/system.md').read_text(encoding='utf-8-sig').replace(
        '{context}', (ROOT / 'src/prompts/context.md').read_text(encoding='utf-8-sig'))
    return (system.rstrip() + '\n\n' + evaluation_rules(seed_evaluation)
            + '\n## Delivery\nFollow the output format requested in the user message.\n')


def openevolve_config(args, settings, seed_evaluation):
    from openevolve.config import Config, LLMModelConfig
    config = Config()
    config.max_iterations = args.iterations
    config.random_seed = settings['random_seed']
    config.language = 'python'
    config.log_dir = str(Path(settings['work_dir']) / 'logs')
    config.checkpoint_interval = 10
    # Long enough for the schedulers the other arms write; OpenEvolve's default (10,000 chars) is not.
    config.max_code_length = 100_000
    config.llm.models = [LLMModelConfig(name=settings['model'], init_client=init_client, weight=1.0,
                                        random_seed=settings['random_seed'])]
    config.llm.evaluator_models = []
    config.prompt.system_message = system_message(seed_evaluation)
    config.database.random_seed = settings['random_seed']
    config.evaluator.cascade_evaluation = False
    config.evaluator.use_llm_feedback = False
    config.evaluator.max_retries = 0
    # One worker keeps evaluation load and spending sequential, like the direct arm.
    config.evaluator.parallel_evaluations = 1
    # Above the harness's own bound (every scale at its subprocess timeout), so it never fires first.
    params = seed_evaluation['params']
    config.evaluator.timeout = int(len(seed_evaluation['scales']) * params['subprocess_timeout'] + 300)
    return config


def config_record(config):
    """What run_config.json records about OpenEvolve's settings."""
    from importlib.metadata import version
    return {'version': version('openevolve'), 'random_seed': config.random_seed,
            'diff_based_evolution': config.diff_based_evolution,
            'full_rewrite_probability': config.full_rewrite_probability,
            'max_code_length': config.max_code_length,
            'num_islands': config.database.num_islands,
            'feature_dimensions': list(config.database.feature_dimensions),
            'migration_interval': config.database.migration_interval,
            'exploration_ratio': config.database.exploration_ratio,
            'exploitation_ratio': config.database.exploitation_ratio,
            'num_top_programs': config.prompt.num_top_programs,
            'num_diverse_programs': config.prompt.num_diverse_programs,
            'parallel_evaluations': config.evaluator.parallel_evaluations,
            'evaluator_timeout_seconds': config.evaluator.timeout,
            'combined_score': '-weighted latency score (s); ineligible = ' + str(INVALID_COMBINED_SCORE)}


def random_seed_for(output):
    """Fixed per output directory, so repeats differ but a rerun of one directory does not."""
    return int(hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:8], 16)


def run_openevolve_arm(arm, args, summary, save, *, model, reasoning):
    """Evaluate the start as 000_seed, then let OpenEvolve run args.iterations iterations."""
    from openevolve import OpenEvolve
    from openevolve.controller import EvolutionFailedError
    if getattr(args, 'resume', False):
        raise ValueError('Resuming an OpenEvolve arm is not supported')
    work_dir = args.output / 'openevolve'
    arm_dir = args.output / 'candidates' / arm
    seed_dir = arm_dir / '000_seed'
    seed_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    arm_summary = {'model': model, 'generator': 'openevolve', 'selector': 'openevolve',
                   'trajectory': [], 'stop_reason': 'iterations',
                   'known_cost_usd': 0.0, 'cost_complete': True, 'failed_attempts': 0}
    summary['arms'][arm] = arm_summary

    def refresh():
        rows = recorded_rows(arm_dir)
        arm_summary['trajectory'] = rows
        arm_summary['known_cost_usd'] = sum(r['observed_cost_usd'] for r in rows)
        arm_summary['cost_complete'] = all(r['estimated_cost_usd'] is not None for r in rows)
        arm_summary['failed_attempts'] = sum(1 for r in rows if r['iteration'] and r['score'] is None)
        valid = [r for r in rows if r['score'] is not None]
        arm_summary['best_path'] = (min(valid, key=lambda r: r['score']) if valid else rows[0])['path']
        arm_summary['estimated_cost_usd'] = arm_summary['known_cost_usd'] if arm_summary['cost_complete'] else None

    start = getattr(args, 'start_scheduler', None) or ROOT / 'src/seed.py'
    shutil.copyfile(start, seed_dir / 'scheduler.py')
    (seed_dir / 'prompt.txt').write_text('Identical seed; no model call.\n')
    started = time.monotonic()
    seed_evaluation = evaluate_scheduler(seed_dir / 'scheduler.py', prototype=args.prototype, trace=args.trace)
    write_json(seed_dir / 'eval.json', seed_evaluation)
    usage = {'input_tokens': None, 'output_tokens': None, 'estimated_cost_usd': 0.0,
             'observed_cost_usd': 0.0, 'model_invocations': 0}
    elapsed = time.monotonic() - started
    seed_sha = hashlib.sha256((seed_dir / 'scheduler.py').read_bytes()).hexdigest()
    seed_row = {'arm': arm, 'iteration': 0, 'parent': None, 'path': str(seed_dir.resolve()), 'model': model,
                'score': seed_evaluation['score'], 'best_score': seed_evaluation['score'],
                'best_path': str(seed_dir.resolve()), 'elapsed_seconds': elapsed, 'error': None,
                'status': 'seed', 'error_fatal': False, 'generator': 'openevolve', 'selector': 'openevolve',
                'selection_usage': {'input_tokens': 0, 'output_tokens': 0, 'estimated_cost_usd': 0.0,
                                    'observed_cost_usd': 0.0, 'model_invocations': 0},
                'generation_usage': dict(usage), 'selected_parent_rank': None, 'selection_fallback': False,
                'scheduler_sha256': seed_sha, **usage,
                'cumulative_cost_usd': 0.0, 'cumulative_elapsed_seconds': elapsed}
    write_json(seed_dir / 'meta.json', seed_row)
    refresh()
    save()
    print(f'{arm:<10} {0:>4} {seed_row["score"] or float("nan"):>10.4f}', flush=True)
    if not seed_evaluation['ok']:
        arm_summary['stop_reason'] = 'seed_evaluation_failed'
        save()
        return
    if args.iterations == 0:
        save()
        return
    if not model:
        raise RuntimeError('Set DIRECT_MODEL in .env')

    settings = {'arm': arm, 'arm_dir': str(arm_dir.resolve()), 'work_dir': str(work_dir.resolve()),
                'trace': str(Path(args.trace).resolve()), 'prototype': args.prototype, 'model': model,
                'reasoning_effort': reasoning, 'timeout': args.timeout, 'budget_usd': args.budget_usd,
                'budget_guard': bool(getattr(args, 'budget_guard', False)),
                'budget_slack_usd': getattr(args, 'budget_slack_usd', 0),
                'random_seed': random_seed_for(args.output)}
    settings_path = work_dir / 'settings.json'
    write_json(settings_path, settings)
    os.environ[SETTINGS_ENV] = str(settings_path)
    _SEED_EVALUATIONS[seed_sha] = seed_evaluation
    config = openevolve_config(args, settings, seed_evaluation)
    arm_summary['openevolve'] = config_record(config)
    process_parallel._run_iteration_worker = recorded_iteration
    controller = OpenEvolve(initial_program_path=str(seed_dir / 'scheduler.py'),
                            evaluation_file=str(ROOT / 'src/openevolve_eval.py'),
                            config=config, output_dir=str(work_dir / 'output'))
    finished = threading.Event()
    printed = set()

    def watch():
        # A stop file from the worker (budget, provider failure) ends OpenEvolve's loop.
        while not finished.wait(2):
            refresh()
            for row in arm_summary['trajectory']:
                if row['iteration'] and row['iteration'] not in printed:
                    printed.add(row['iteration'])
                    fmt = lambda v: f'{v:.4f}' if v is not None else 'n/a'
                    print(f'{arm:<10} {row["iteration"]:>4} {fmt(row["score"]):>10} {fmt(row["best_score"]):>12} '
                          f'{row["elapsed_seconds"]:>8.1f}s {fmt(row["estimated_cost_usd"]):>14} {row["status"]}',
                          flush=True)
            save()
            if stop_requested(settings) and controller.parallel_controller:
                controller.parallel_controller.request_shutdown()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        asyncio.run(controller.run(iterations=args.iterations))
    except EvolutionFailedError as exc:
        # Every iteration failed before producing a candidate; the rows say why.
        arm_summary['openevolve_error'] = str(exc)
    finally:
        finished.set()
        watcher.join()
        process_parallel._run_iteration_worker = _ORIGINAL_ITERATION
        os.environ.pop(SETTINGS_ENV, None)
    refresh()
    stop = stop_requested(settings)
    if stop:
        arm_summary['stop_reason'] = stop['reason']
        if stop['reason'] == 'budget_reserve':
            arm_summary['next_round_reserve_usd'] = stop['detail']['next_round_reserve_usd']
    save()
