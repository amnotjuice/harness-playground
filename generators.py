"""Two generators; the experiment driver owns all evaluation and selection."""
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from config import ROOT, TRACE, default_agent_timeout
from feedback import get_iteration_feedback_prompt
from worker import validate_code


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def build_prompt(code, evaluation):
    system = (ROOT / 'prompts/system.md').read_text(encoding='utf-8-sig').replace(
        '{context}', (ROOT / 'prompts/context.md').read_text(encoding='utf-8-sig'))
    return system.rstrip() + '\n\n' + get_iteration_feedback_prompt(code, evaluation)


def render_direct_prompt(prompt):
    return prompt + '\n## Delivery\nReturn ONLY the complete Python scheduler code. No Markdown fences or prose outside the code.\n'


def render_opencode_prompt(prompt, *, prototype=False, timeout=None):
    if timeout is None:
        timeout = default_agent_timeout(prototype)
    return prompt + f'''\n## Delivery
Edit scheduler.py. You may test and revise it within this session's
{timeout:g}-second limit by running:

```sh
./evaluate
```

This uses the evaluation settings above and saves each tested version and result.
Leave the final policy in scheduler.py for independent evaluation.
'''


def write_evaluation_launcher(work, *, prototype=False, trace=TRACE):
    """Resolve this machine's paths at setup; expose only ./evaluate to the agent."""
    work = work.resolve()
    command = [sys.executable, str(ROOT / 'evaluator.py'),
               '--scheduler', str(work / 'scheduler.py'),
               '--output-dir', str(work.parent / 'internal_evals'), '--trace', str(trace.resolve())]
    if prototype:
        command.append('--prototype')
    launcher = work / 'evaluate'
    launcher.write_text('#!/bin/sh\nexec ' + shlex.join(command) + '\n')
    launcher.chmod(0o700)
    return launcher


def call_llm(directory, prompt, model, timeout, accounting, *, reasoning_effort=None, max_tokens=None):
    """One paid call; save the exact prompt, response and usage before parsing."""
    import litellm
    # Official standard short-context rates checked 2026-10-08. Older LiteLLM
    # catalogs lack this new model. Long-context/other-tier pricing stays unknown.
    if model == 'openai/gpt-6.1-sol':
        try:
            prices = json.loads((ROOT / 'opencode.json').read_text())['provider']['openai']['models']['gpt-6.1-sol']['cost']
            if any(not math.isfinite(prices[key]) or prices[key] < 0
                   for key in ('input', 'output', 'cache_read', 'cache_write')):
                raise ValueError('Prices must be finite and nonnegative')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f'Invalid model pricing in opencode.json: {exc}') from exc
        # OpenCode uses dollars per million tokens; LiteLLM uses dollars per token.
        litellm.register_model({'gpt-6.1-sol': {
            'litellm_provider': 'openai', 'mode': 'chat',
            'input_cost_per_token': prices['input'] / 1e6,
            'output_cost_per_token': prices['output'] / 1e6,
            'cache_read_input_token_cost': prices['cache_read'] / 1e6,
            'cache_creation_input_token_cost': prices['cache_write'] / 1e6}})
    (directory / 'prompt.txt').write_text(prompt)
    kwargs = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
              'num_retries': 0, 'timeout': timeout}
    if reasoning_effort:
        kwargs['reasoning_effort'] = reasoning_effort
    if not reasoning_effort or reasoning_effort == 'none':
        kwargs['temperature'] = 1.0
    if max_tokens is not None:
        kwargs['max_completion_tokens'] = max_tokens
    accounting['model_invocations'] = 1  # attempted; provider failure can leave usage unknown
    try:
        response = litellm.completion(**kwargs)
    except litellm.Timeout as exc:
        raise TimeoutError('Model call timed out') from exc
    except Exception as exc:
        raise RuntimeError(f'Provider call failed ({type(exc).__name__}): {exc}') from exc
    write_json(directory / 'response.json', response.model_dump(mode='json'))
    raw = response.choices[0].message.content or ''
    (directory / 'response.txt').write_text(raw)
    usage = response.usage
    accounting.update(input_tokens=getattr(usage, 'prompt_tokens', None),
                      output_tokens=getattr(usage, 'completion_tokens', None))
    if model == 'openai/gpt-6.1-sol' and (
        (accounting['input_tokens'] or 0) > 272000
        or getattr(response, 'service_tier', None) not in (None, 'default', 'standard')
    ):
        accounting['cost_error'] = 'Pricing outside verified standard short-context rates'
        return raw
    try:
        cost = litellm.completion_cost(completion_response=response)
        if cost is None or not math.isfinite(cost) or cost < 0:
            raise ValueError(f'Invalid cost estimate: {cost!r}')
    except Exception as exc:
        raise RuntimeError(f'Cost estimation failed ({type(exc).__name__}): {exc}') from exc
    accounting.update(estimated_cost_usd=cost, observed_cost_usd=cost)
    return raw


def generate_direct(directory, prompt, model, timeout, accounting):
    raw = call_llm(directory, render_direct_prompt(prompt), model, timeout, accounting,
                   reasoning_effort=os.getenv('DIRECT_REASONING_EFFORT'))
    match = re.search(r'```(?:python)?\s*\n(.*?)```', raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()
    (directory / 'scheduler.py').write_text(code + '\n')
    validate_code(code)


def opencode_environment():
    env = os.environ.copy()
    for kind in ('DATA', 'CONFIG', 'CACHE', 'STATE'):
        env[f'XDG_{kind}_HOME'] = str(ROOT / '.tools/state' / kind.lower())
    env['OPENCODE_CONFIG'] = str(ROOT / 'opencode.json')
    return env


def detect_opencode():
    local = ROOT / '.tools/opencode'
    executable = str(local) if local.is_file() else shutil.which('opencode')
    if not executable:
        raise RuntimeError('opencode is not installed/on PATH; install it manually, then run opencode run --help')
    help_result = subprocess.run([executable, 'run', '--help'], capture_output=True, text=True, timeout=20, env=opencode_environment())
    help_text = help_result.stdout + help_result.stderr
    if help_result.returncode or any(flag not in help_text for flag in ('--model', '--format', '--dir')):
        raise RuntimeError('Installed opencode must support run --model, --format json and --dir')
    return executable, help_text


def parse_opencode_usage(stdout):
    steps = {}
    errors = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'error':
            errors.append(event.get('error'))
        if event.get('type') == 'step_finish':
            part = event.get('part', {})
            steps[part.get('id', str(len(steps)))] = part
    def total(field):
        vals = [p.get('tokens', {}).get(field) for p in steps.values()]
        return sum(vals) if vals and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals) else None
    costs = [p.get('cost') for p in steps.values()]
    observed_cost = sum(v for v in costs if isinstance(v, (int, float)) and math.isfinite(v) and v > 0)
    # A zero can mean missing model pricing/subscription billing. Do not call it free.
    cost = observed_cost if costs and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in costs) and not errors else None
    return {'input_tokens': total('input'), 'output_tokens': total('output'),
            'reasoning_tokens': total('reasoning'),
            'estimated_cost_usd': cost, 'observed_cost_usd': observed_cost,
            'model_invocations': len(steps) or None,
            'usage_note': 'Observed step_finish events; hidden retries/subagents may be unreported.',
            'errors': errors, 'step_usage': list(steps.values())}


def generate_opencode(directory, prompt, model, timeout, accounting, *, prototype=False, trace=TRACE):
    rendered = render_opencode_prompt(prompt, prototype=prototype, timeout=timeout)
    (directory / 'prompt.txt').write_text(rendered)
    executable, help_text = detect_opencode()
    (directory / 'opencode-help.txt').write_text(help_text)
    # Separate the agent's mutable workspace from the immutable evaluation artifacts.
    work = directory / 'workspace'
    work.mkdir()
    write_evaluation_launcher(work, prototype=prototype, trace=trace)
    original = (directory / 'scheduler.py').read_text()
    (work / 'scheduler.py').write_text(original)
    (work / 'task.md').write_text(rendered)
    cmd = [executable, 'run', '--dir', str(work.resolve()), '--model', model, '--format', 'json']
    if os.getenv('OPENCODE_VARIANT'):
        cmd += ['--variant', os.environ['OPENCODE_VARIANT']]
    cmd += ['Read task.md and edit scheduler.py as instructed.']
    write_json(directory / 'command.json', {'argv': cmd, 'cwd': str(work.resolve())})
    env = opencode_environment()
    # OpenCode resolves its root from PWD before falling back to process.cwd().
    # subprocess cwd alone leaves the parent's PWD pointing at the project root.
    env['PWD'] = str(work.resolve())
    with (directory / 'response.txt').open('w') as out, (directory / 'stderr.txt').open('w') as err:
        proc = subprocess.Popen(cmd, cwd=work, stdout=out, stderr=err, start_new_session=True, env=env)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    accounting.update(parse_opencode_usage((directory / 'response.txt').read_text()))
    result = work / 'scheduler.py'
    if result.exists():
        shutil.copyfile(result, directory / 'scheduler.py')
    if timed_out or proc.returncode or accounting['errors']:
        # Keep the observed lower bound even though incomplete events cannot establish a total.
        accounting['estimated_cost_usd'] = None
        if timed_out:
            raise TimeoutError('OpenCode session timed out')
        raise RuntimeError(f'OpenCode failed: timeout={timed_out}, exit={proc.returncode}, errors={accounting["errors"]}')
    if not result.exists():
        raise ValueError('OpenCode did not produce scheduler.py')
    # An unchanged valid policy is an unsuccessful improvement, not a broken run.
    validate_code(result.read_text())
