"""Weighted fair, concurrent scheduling with feedback-driven RAM backoff.

Small containers replace whole-pool allocations. Queries receive more scheduling
turns, batch retains a share, and failed operators retry with larger RAM budgets.
Running work is never preempted, avoiding lost work and suspension churn.
"""

from collections import deque


@register_scheduler_init(key="experiment")
def init(s):
    s.policy_queues = [deque(), deque(), deque()]
    s.policy_live = set()
    s.policy_ops = {}
    s.policy_turn = 0
    s.policy_wheel = [0, 1, 0, 1, 0, 2, 0, 1, 0, 1,
                      0, 2, 0, 1, 0, 0, 0]
    s.policy_max_ram = max(
        (s.executor.pools[i].max_ram_pool
         for i in range(s.executor.num_pools)),
        default=0,
    )


def _experiment_class(priority):
    if priority == Priority.QUERY:
        return 0
    if priority == Priority.INTERACTIVE:
        return 1
    return 2


def _experiment_record(s, op):
    key = id(op)
    record = s.policy_ops.get(key)
    if record is None:
        # Retain the operator reference so object IDs cannot be recycled.
        record = {
            "op": op,
            "ram": min(8.0, s.policy_max_ram),
            "failures": 0,
            "blocked": False,
        }
        s.policy_ops[key] = record
    return record


@register_scheduler(key="experiment")
def schedule(
    s, results: List[ExecutionResult], pipelines: List[Pipeline]
) -> Tuple[List[Suspend], List[Assignment]]:
    """Pack ready operators using weighted turns and bounded memory retries."""
    # Allocations and DAG readiness change only on arrivals or executor events.
    if not pipelines and not results:
        return [], []

    for result in results:
        if not result.failed():
            continue

        error = str(result.error).lower()
        terminal = any(
            word in error
            for word in ("timeout", "timed out", "cancelled", "canceled")
        )
        for op in result.ops:
            record = _experiment_record(s, op)
            record["failures"] += 1
            if terminal or record["failures"] >= 16:
                record["blocked"] = True
                continue

            # OOMs are recoverable. Unknown failures also receive a bounded
            # memory retry rather than silently abandoning the entire DAG.
            previous = max(float(result.ram), record["ram"])
            if previous >= s.policy_max_ram:
                record["blocked"] = True
            else:
                record["ram"] = min(
                    s.policy_max_ram, max(previous * 2.0, previous + 1.0)
                )

    for pipeline in pipelines:
        pid = pipeline.pipeline_id
        if pid not in s.policy_live:
            s.policy_live.add(pid)
            s.policy_queues[_experiment_class(pipeline.priority)].append(pipeline)

    free_cpu = [
        max(0.0, s.executor.pools[i].avail_cpu_pool)
        for i in range(s.executor.num_pools)
    ]
    free_ram = [
        max(0.0, s.executor.pools[i].avail_ram_pool)
        for i in range(s.executor.num_pools)
    ]
    if not any(cpu >= 1.0 and ram > 0 for cpu, ram in zip(free_cpu, free_ram)):
        return [], []

    ready = [deque(), deque(), deque()]

    # Inspect each live pipeline at most once. Keep blocked/running pipelines
    # in the rotation, but do not repeatedly poll them within this call.
    for cls, queue in enumerate(s.policy_queues):
        for _ in range(len(queue)):
            pipeline = queue.popleft()
            status = pipeline.runtime_status()
            if status.is_pipeline_successful() or status.has_timed_out():
                s.policy_live.discard(pipeline.pipeline_id)
                continue

            queue.append(pipeline)
            ops = status.get_ops(
                ASSIGNABLE_STATES, require_parents_complete=True
            )
            candidates = deque()
            for op in ops:
                record = _experiment_record(s, op)
                if not record["blocked"]:
                    candidates.append((op, record))
            if candidates:
                ready[cls].append((pipeline, candidates))

    assignments = []
    wheel = s.policy_wheel
    idle_turns = 0

    # Every unsuccessful placement removes a candidate for this call; every
    # successful placement consumes at least one CPU. Thus this loop is bounded.
    while any(ready) and idle_turns < len(wheel):
        cls = wheel[s.policy_turn]
        s.policy_turn = (s.policy_turn + 1) % len(wheel)
        if not ready[cls]:
            idle_turns += 1
            continue
        idle_turns = 0

        pipeline, candidates = ready[cls].popleft()
        op, record = candidates.popleft()
        ram = record["ram"]
        target_cpu = (4.0, 2.0, 1.0)[cls]

        best_pool = None
        best_cpu = 0.0
        best_key = None
        for pool_id in range(s.executor.num_pools):
            if free_ram[pool_id] < ram or free_cpu[pool_id] < 1.0:
                continue
            cpu = min(target_cpu, free_cpu[pool_id])
            # Prefer the desired CPU size, then pack RAM tightly to preserve
            # room for larger retries elsewhere.
            key = (
                cpu < target_cpu,
                free_ram[pool_id] - ram,
                free_cpu[pool_id] - cpu,
                pool_id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pool = pool_id
                best_cpu = cpu

        if best_pool is not None:
            assignments.append(
                Assignment(
                    ops=[op],
                    cpu=best_cpu,
                    ram=ram,
                    priority=pipeline.priority,
                    pool_id=best_pool,
                    pipeline_id=pipeline.pipeline_id,
                )
            )
            # Executor availability is a snapshot, not a mutable reservation.
            free_cpu[best_pool] -= best_cpu
            free_ram[best_pool] -= ram

        # An operator that cannot fit waits until the next executor event.
        # Other operators, including smaller siblings, can still make progress.
        if candidates:
            ready[cls].append((pipeline, candidates))

        if not any(
            cpu >= 1.0 and available_ram > 0
            for cpu, available_ram in zip(free_cpu, free_ram)
        ):
            break

    return [], assignments
