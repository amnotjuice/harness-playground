"""Select one valid parent; the driver owns generation, accounting and stopping."""
import json
import time
from pathlib import Path

from config import ROOT
from generators import call_llm, write_json

SELECTOR_MAX_TOKENS = 4096


def select_parent(pool, trajectory, directory, *, strategy, generator, model,
                  reasoning_effort, timeout, remaining_budget, usage):
    valid = [c for c in pool if c['evaluation'].get('ok') and c['evaluation'].get('score') is not None]
    ranked = sorted(valid, key=lambda c: c['evaluation']['score'])
    parent = ranked[0]
    decision = {'strategy': strategy, 'eligible_ids': [Path(c['path']).name for c in valid],
                'reason': 'Lowest score' if strategy == 'best' else 'Only one eligible candidate',
                'fallback': False, 'remaining_budget_usd': remaining_budget}
    started = time.monotonic()
    try:
        if strategy == 'llm' and len(valid) > 1:
            records = []
            for candidate in valid:
                origin = next((row.get('parent') for row in trajectory if row['path'] == candidate['path']), None)
                children = [row for row in trajectory if row.get('parent') == candidate['path']
                            and row.get('status') not in ('selection_failed', 'selection_only')]
                evaluation = candidate['evaluation']
                records.append({
                    'id': Path(candidate['path']).name,
                    'parent_id': Path(origin).name if origin else None,
                    'score': evaluation['score'], 'scale_results': evaluation.get('scale_results', {}),
                    'children_attempted': len(children),
                    'children_valid': sum(row.get('score') is not None for row in children),
                    'children_better_than_parent': sum(row.get('score') is not None and
                        row['score'] < evaluation['score'] for row in children),
                    'code': (Path(candidate['path']) / 'scheduler.py').read_text(),
                })
            payload = {'generator': generator, 'remaining_budget_usd': remaining_budget,
                       'evaluation_params': valid[0]['evaluation'].get('params', {}),
                       'candidates': records}
            prompt = (ROOT / 'prompts/selection.md').read_text() + '\n\n' + json.dumps(payload, indent=2)
            folder = directory / 'selection'
            folder.mkdir()
            usage['estimated_cost_usd'] = None
            raw = call_llm(folder, prompt, model, timeout, usage,
                           reasoning_effort=reasoning_effort, max_tokens=SELECTOR_MAX_TOKENS)
            try:
                choice = json.loads(raw)
                if not isinstance(choice, dict) or set(choice) != {'parent_id', 'reason'}:
                    raise ValueError('Expected parent_id and reason')
                if not isinstance(choice['reason'], str) or not choice['reason'].strip():
                    raise ValueError('Missing selection rationale')
                parent = next(c for c in valid if Path(c['path']).name == choice['parent_id'])
                decision['reason'] = choice['reason']
            except (ValueError, StopIteration, TypeError) as exc:
                decision.update(fallback=True, reason='Invalid selection; used lowest score',
                                error=f'{type(exc).__name__}: {exc}')
        return parent
    except Exception as exc:
        decision['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        decision.update(parent_id=Path(parent['path']).name, parent_path=parent['path'],
                        parent_score=parent['evaluation']['score'],
                        best_available_score=ranked[0]['evaluation']['score'],
                        parent_rank=next(i + 1 for i, c in enumerate(ranked) if c is parent),
                        elapsed_seconds=time.monotonic() - started, usage=dict(usage))
        write_json(directory / 'selection.json', decision)
