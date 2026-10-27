"""Weighted fair, concurrent scheduling with bounded, feedback-driven RAM retries.

Unlike the serial baseline, this policy packs small containers concurrently.
Queries receive preferential admission, while interactive and batch work retain
regular scheduling opportunities. OOM feedback doubles an operator's RAM request;
running work is never preempted.
"""

from collections import deque


def _op_key(op):
    # Operator objects need not be hashable. Retain identity-based state without
    # inspecting hidden resource requirements.
    if isinstance(op, (str, int, bytes)):
        return (type(op), op)
    return ("object", id(op))


def _priority_class(priority):
    if priority == Priority.QUERY:
        return 0
    if priority == Priority.INTERACTIVE:
        return 1
    return 2


@register_scheduler_init(key="experiment")
def init(s):
    """Initialize rotating pipeline queues and observable resource history."""
    s.exp_queues = [deque(), deque(), deque()]
    s.exp_known = set()
    s.exp_ram = {}
    s.exp_failed = set()
    s.exp_retries = {}
    s.exp_cursor = 0
    s.exp_order = (0, 1, 0, 1, 0, 2, 0)
    s.exp_max_ram = max(
        (s.executor.pools[i].max_ram_pool
         for i in range(s.executor.num_pools)),
        default=0,
    )


@register_scheduler(key="experiment")
def schedule(
    s, results: List[ExecutionResult], pipelines: List[Pipeline]
) -> Tuple[List[Suspend], List[Assignment]]:
    """Pack ready operators fairly, increasing memory only after observed OOMs."""
    for result in results:
        if not result.failed():
            continue

        error = str(result.error).lower()
        oom = (
            "oom" in error
            or "memory" in error
            or "outofmemory" in error
        )

        for op in result.ops:
            key = _op_key(op)
            retries = s.exp_retries.get(key, 0) + 1
            s.exp_retries[key] = retries

            if oom and result.ram < s.exp_max_ram and retries <= 12:
                previous = max(float(result.ram), s.exp_ram.get(key, 32.0))
                s.exp_ram[key] = min(s.exp_max_ram, previous * 2.0)
            else:
                # Do not spin on terminal errors or an OOM at maximum capacity.
                s.exp_failed.add(key)

    for pipeline in pipelines:
        pid = pipeline.pipeline_id
        if pid not in s.exp_known:
            s.exp_known.add(pid)
            s.exp_queues[_priority_class(pipeline.priority)].append(pipeline)

    # Between arrivals and completions, neither readiness nor capacity changes.
    if not pipelines and not results:
        return [], []

    available = [
        [float(s.executor.pools[i].avail_cpu_pool),
         float(s.executor.pools[i].avail_ram_pool)]
        for i in range(s.executor.num_pools)
    ]
    if not any(cpu >= 1 and ram > 0 for cpu, ram in available):
        return [], []

    ready = [deque(), deque(), deque()]

    # Bound status inspection, rotating queues so a large backlog cannot hide
    # the same pipelines forever. Successful and timed-out pipelines are retired.
    for cls in range(3):
        queue = s.exp_queues[cls]
        for _ in range(min(len(queue), 2048)):
            pipeline = queue.popleft()
            status = pipeline.runtime_status()

            if status.is_pipeline_successful():
                continue
            timed_out = getattr(status, "has_timed_out", None)
            if callable(timed_out) and timed_out():
                continue

            queue.append(pipeline)
            ops = status.get_ops(
                ASSIGNABLE_STATES, require_parents_complete=True
            )
            candidates = deque()
            for op in ops:
                key = _op_key(op)
                if key not in s.exp_failed:
                    candidates.append(op)
                    if len(candidates) >= 32:
                        break
            if candidates:
                ready[cls].append((pipeline, candidates))

    assignments = []
    selected = set()
    attempts = 0

    # A weighted rotation gives batch work a service opportunity even when
    # foreground queues remain nonempty. Empty classes lend their share.
    while any(ready) and attempts < 4096:
        cls = s.exp_order[s.exp_cursor]
        s.exp_cursor = (s.exp_cursor + 1) % len(s.exp_order)
        attempts += 1

        if not ready[cls]:
            continue

        pipeline, ops = ready[cls].popleft()
        op = ops.popleft()
        key = _op_key(op)

        if ops:
            ready[cls].append((pipeline, ops))
        if key in selected:
            continue

        ram = s.exp_ram.get(key, min(32.0, s.exp_max_ram))
        if ram <= 0:
            continue

        # Keep containers modest: parallelism avoids the baseline's severe
        # head-of-line blocking and leaves resources for later arrivals.
        target_cpu = 4.0

        best_pool = None
        best_score = None
        for pool_id, (free_cpu, free_ram) in enumerate(available):
            if free_cpu < 1 or free_ram < ram:
                continue

            cpu = min(target_cpu, free_cpu)
            pool = s.executor.pools[pool_id]
            # Prefer a full CPU allocation, then pack into the tighter pool.
            score = (
                cpu,
                -((free_ram - ram) / max(float(pool.max_ram_pool), 1.0)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pool = pool_id

        if best_pool is None:
            # Leave this operator pending for the next resource-release event.
            continue

        cpu = min(target_cpu, available[best_pool][0])
        assignments.append(
            Assignment(
                ops=[op],
                cpu=cpu,
                ram=ram,
                priority=pipeline.priority,
                pool_id=best_pool,
                pipeline_id=pipeline.pipeline_id,
            )
        )
        selected.add(key)
        available[best_pool][0] -= cpu
        available[best_pool][1] -= ram

        if not any(c >= 1 and r > 0 for c, r in available):
            break

    return [], assignments
