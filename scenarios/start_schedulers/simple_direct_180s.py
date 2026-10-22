"""Priority-biased, memory-balanced admission without preemption.

The parent underutilized CPU when RAM limited concurrency. Size CPU shares
from the number of containers that actually fit in memory, rather than half
the CPU/RAM ratio. Prefer valuable query completions, with periodic admission
opportunities for interactive and batch work. Learn RAM only from OOMs.
"""

from collections import deque


def _exp_op_key(op):
    if isinstance(op, (str, int, bytes)):
        return (type(op), op)
    return ("object", id(op))


def _exp_class(priority):
    if priority == Priority.QUERY:
        return 0
    if priority == Priority.INTERACTIVE:
        return 1
    return 2


@register_scheduler_init(key="experiment")
def init(s):
    """Initialize persistent queues, admission credits, and OOM feedback."""
    s.exp_queues = [deque(), deque(), deque()]
    s.exp_known = set()
    s.exp_ram = {}
    s.exp_retries = {}
    s.exp_terminal = set()
    s.exp_admissions = 0
    s.exp_max_ram = max(
        (
            float(s.executor.pools[i].max_ram_pool)
            for i in range(s.executor.num_pools)
        ),
        default=0.0,
    )
    s.exp_initial_ram = min(64.0, s.exp_max_ram)


@register_scheduler(key="experiment")
def schedule(
    s, results: List[ExecutionResult], pipelines: List[Pipeline]
) -> Tuple[List[Suspend], List[Assignment]]:
    """Admit priority-biased work and balance CPU against feasible RAM slots."""
    for result in results:
        if not result.failed():
            continue

        error = str(result.error).lower()
        oom = "oom" in error or "memory" in error
        for op in result.ops:
            key = _exp_op_key(op)
            retries = s.exp_retries.get(key, 0) + 1
            s.exp_retries[key] = retries
            previous = max(
                float(result.ram),
                s.exp_ram.get(key, s.exp_initial_ram),
            )
            if oom and previous < s.exp_max_ram and retries <= 12:
                s.exp_ram[key] = min(
                    s.exp_max_ram, max(64.0, previous * 2.0)
                )
            else:
                # Do not repeatedly spend capacity on expired or impossible work.
                s.exp_terminal.add(key)

    for pipeline in pipelines:
        pid = pipeline.pipeline_id
        if pid not in s.exp_known:
            s.exp_known.add(pid)
            s.exp_queues[_exp_class(pipeline.priority)].append(pipeline)

    if not pipelines and not results:
        return [], []

    available = [
        [
            max(0.0, float(s.executor.pools[i].avail_cpu_pool)),
            max(0.0, float(s.executor.pools[i].avail_ram_pool)),
        ]
        for i in range(s.executor.num_pools)
    ]
    if not any(cpu >= 1.0 and ram > 0.0 for cpu, ram in available):
        return [], []

    ready = [deque(), deque(), deque()]
    discovered = set()
    ready_count = 0

    # Rotate retained pipelines so inspection remains bounded and fair.
    for cls in range(3):
        queue = s.exp_queues[cls]
        for _ in range(min(len(queue), 2048)):
            pipeline = queue.popleft()
            status = pipeline.runtime_status()
            if status.is_pipeline_successful():
                continue
            queue.append(pipeline)

            candidates = deque()
            for op in status.get_ops(
                ASSIGNABLE_STATES, require_parents_complete=True
            ):
                key = _exp_op_key(op)
                if key in discovered or key in s.exp_terminal:
                    continue
                ram = s.exp_ram.get(key, s.exp_initial_ram)
                if ram <= 0.0 or ram > s.exp_max_ram:
                    continue
                discovered.add(key)
                candidates.append((op, key, ram))
                ready_count += 1
                if len(candidates) >= 32:
                    break
            if candidates:
                ready[cls].append((pipeline, candidates))

    assignments = []
    attempts = 0

    while any(ready) and attempts < 8192:
        attempts += 1

        # Credits count successful admissions, not scheduler invocations.
        # Lower classes retain service even during a sustained query backlog.
        turn = s.exp_admissions + 1
        if turn % 64 == 0:
            preference = (2, 1, 0)
        elif turn % 8 == 0:
            preference = (1, 0, 2)
        else:
            preference = (0, 1, 2)
        cls = next(c for c in preference if ready[c])

        pipeline, candidates = ready[cls].popleft()
        op, key, ram = candidates.popleft()
        if candidates:
            ready[cls].append((pipeline, candidates))

        best_pool = None
        best_cpu = 0.0
        best_score = None

        for pool_id, (free_cpu, free_ram) in enumerate(available):
            if free_cpu < 1.0 or free_ram < ram:
                continue

            # Balance the remaining CPU over containers that can actually fit.
            # At 64 CPUs / 500 RAM and 64 RAM per operator, seven containers
            # receive roughly nine CPUs each instead of leaving half CPU idle.
            memory_slots = max(1, int((free_ram + 1e-9) / ram))
            slots = max(1, min(ready_count, memory_slots))
            balanced_cpu = free_cpu / slots

            # A modest floor avoids very slow individual functions. The cap
            # prevents a single admission from monopolizing a large machine.
            target_cpu = min(64.0, max(4.0, balanced_cpu))
            cpu = min(free_cpu, target_cpu)

            pool = s.executor.pools[pool_id]
            score = (
                cpu / max(target_cpu, 1.0),
                -(
                    (free_ram - ram)
                    / max(float(pool.max_ram_pool), 1.0)
                ),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pool = pool_id
                best_cpu = cpu

        ready_count -= 1
        if best_pool is None:
            # Keep the operator in its pipeline for a later scheduling event;
            # other, smaller ready operators may still fit this snapshot.
            continue

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
        s.exp_admissions += 1

        # New assignments do not update executor snapshots themselves.
        available[best_pool][0] = max(
            0.0, available[best_pool][0] - best_cpu
        )
        available[best_pool][1] = max(
            0.0, available[best_pool][1] - ram
        )
        if not any(cpu >= 1.0 and mem > 0.0 for cpu, mem in available):
            break

    return [], assignments
