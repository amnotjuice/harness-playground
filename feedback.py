"""Best-parent feedback: retain summer2026 metrics in a Bauplan-based task.

The evaluation artifact is authoritative for the score and run settings.
Render each diagnostic once; keep archival metadata out of model input.
"""


def get_iteration_feedback_prompt(scheduler_code, evaluation):
    params = evaluation['params']
    results = evaluation['scale_results']
    scales = evaluation['scales']
    penalty = 2 * params['max_job_seconds']

    def number(value, suffix=''):
        return f'{value:.4f}{suffix}' if isinstance(value, (int, float)) else 'n/a'

    rows = []
    for scale in scales:
        r = results[str(scale)]
        cpus = params['cpus_per_pool'] * scale
        ram = params['ram_gb_per_pool'] * scale
        rows.append(f'### {scale}x: {cpus} CPUs, {ram} GB RAM per pool')
        if not r.get('ok'):
            rows.append(f'Simulation FAILED: {r.get("error", "unknown error")}')
            continue
        rows.extend([
            f'Weighted latency: {number(r["latency"], " s")}',
            f'Failures: {r.get("failures", "n/a")}; OOMs: {r.get("oom_count", "n/a")}; '
            f'error counts: {r.get("failure_error_counts", {})}',
            f'Mean memory allocated: {number(r.get("mean_memory_allocated_percent"), "%")}; '
            f'consumed: {number(r.get("mean_memory_consumed_percent"), "%")}',
            f'Assignments: {r.get("assignments", "n/a")}; '
            f'containers completed: {r.get("containers_completed", "n/a")}; '
            f'suspensions: {r.get("suspensions", "n/a")}',
            '| Priority | Completed / arrived | Mean completed latency (s) | p99 completed latency (s) |',
            '| --- | ---: | ---: | ---: |',
        ])
        for priority in ('query', 'interactive', 'batch'):
            p = r.get(priority)
            if p:
                rows.append(f'| {priority} | {p["completed"]}/{p["arrived"]} | '
                            f'{number(p.get("mean_s"))} | {number(p.get("p99_s"))} |')
        rows.append('')

    settings = (
        f'Trace: {evaluation["trace"]}; the same workload is used at every scale.\n'
        f'Scales: {", ".join(str(s) + "x" for s in scales)}; CPU and RAM scale together.\n'
        f'Pools: {params["num_pools"]}; base CPU/RAM per pool: '
        f'{params["cpus_per_pool"]} CPUs / {params["ram_gb_per_pool"]} GB.\n'
        f'Simulation duration: {params["duration"]} s; ticks per second: {params["ticks_per_second"]}.\n'
        f'Max job duration: {params["max_job_seconds"]} s; '
        f'wall-clock timeout: {params["subprocess_timeout"]} s per scale.\n'
        f'Multiple operators per container: {params["multi_operator_containers"]}.'
    )
    diagnostics = '\n'.join(rows)
    return f'''## Improvement request

Improve the supplied selected valid parent scheduler. Use its code and
feedback to choose changes; both incremental improvements and redesigns are allowed.

## Evaluation objective

Lower is better. For every arrived pipeline, latency is its actual end-to-end
latency if completed, or {penalty:g} seconds if unfinished/failed
(2 * max_job_seconds, with max_job_seconds={params['max_job_seconds']:g}).

For each scale:
weighted_latency = sum(priority_weight * pipeline_latency) / sum(priority_weight)
The sums cover all arrived pipelines. Weights: query=10, interactive=5, batch=1.

The candidate score is the geometric mean of weighted_latency across ALL
configured scales, with each scale weighted equally. Implementation:
exp(mean(log(max(weighted_latency, 1e-12)))).
Every scale must finish simulation without a structural error or timeout and
return a finite, nonnegative latency. Otherwise the candidate is ineligible
(score=null); failed scales are never dropped from the aggregate.

Unfinished pipelines incur the penalty above. Recoverable OOM events do not
independently invalidate a candidate. Completion, memory utilization, failure
counts, and p99 are diagnostic information, not additional score terms.

## Evaluation settings

{settings}

## Parent scheduler

```python
{scheduler_code.rstrip()}
```

## Parent evaluation

Eligible: {evaluation['ok']}; aggregate score: {number(evaluation['score'], ' s')}.
Per-priority mean/p99 below cover completed pipelines only; they exclude the
unfinished penalty. n/a means unavailable, not zero.

{diagnostics}
'''
