{context}

## Instructions

Based on the Eudoxia/Bauplan context above, produce an improved scheduling
policy from the selected parent and its evaluation feedback. Implement the
policy using the interface below, with comments and a short docstring explaining
its logic. Follow the arm-specific delivery instructions at the end.

## Scheduler interface

Implement initialization and scheduling functions with these registrations:

```python
@register_scheduler_init(key="experiment")
def init(s):
    ...

@register_scheduler(key="experiment")
def schedule(s, results: List[ExecutionResult], pipelines: List[Pipeline]) -> Tuple[List[Suspend], List[Assignment]]:
    ...
    return suspensions, assignments
```

Use exactly `experiment` for both keys. Initialize persistent policy state on
`s`; `s.executor` and `s.params` are provided. Use comments and a short function
docstring to explain the policy. Return or write the complete policy, including
any helper functions, according to the delivery instructions at the end.

Available classes and access patterns:

- `Priority.QUERY`, `Priority.INTERACTIVE`, `Priority.BATCH_PIPELINE`.
- `Assignment(ops=op_list, cpu=cpu_amount, ram=ram_amount, priority=priority, pool_id=pool_id, pipeline_id=pipeline.pipeline_id)`.
- `Suspend(container_id, pool_id)`.
- `s.executor.num_pools` and `s.executor.pools[i]`.
- Pool fields: `avail_cpu_pool`, `avail_ram_pool`, `max_cpu_pool`, `max_ram_pool`.
- Pipeline fields: `priority`, `pipeline_id`, `values` (DAG of operators).
- `pipeline.runtime_status().get_ops(states, require_parents_complete=True)`.
- `pipeline.runtime_status().is_pipeline_successful()`.
- ExecutionResult fields: `priority`, `ops`, `container_id`, `pool_id`, `ram`, `cpu`, `error`; method: `failed()`.
- `OperatorState`: PENDING, ASSIGNED, RUNNING, SUSPENDING, COMPLETED, FAILED.
- `ASSIGNABLE_STATES`: PENDING and FAILED.

The worker supplies these globals: List, Tuple, Pipeline, OperatorState,
ASSIGNABLE_STATES, Assignment, ExecutionResult, Suspend,
register_scheduler_init, register_scheduler, Priority. No imports are needed
for them. Any additional imports must be standard-library only.

## Valid execution and information boundaries

- Only assign operators currently in ASSIGNABLE_STATES with completed parents.
- Available CPU/RAM are snapshots at the start of each scheduling call. Track
  new allocations locally; assignments do not decrement those snapshots.
  Do not allocate more than the available resources.
- True operator resource requirements are hidden; infer them from observable
  execution feedback, including OOMs. Do not read resource curves from the
  workload trace or simulator internals. No memory estimator is configured.
- Modify only the policy and its own state. Do not alter scoring, simulator
  internals, workload state, or another candidate. Use the documented API.
- Keep per-call work and retry handling bounded enough to finish within the
  evaluator's timeout. The task does not prescribe a particular queue policy,
  resource split, retry count, or number of assignments per call.
