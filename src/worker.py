#!/usr/bin/env python3
"""Run one scheduler/trace/scale in a process owned by evaluator.py.

Runs one (scheduler, trace, params) simulation and prints the result as:
    __RESULT__{"ok": true, "latency": 1.234, ...}
or
    __RESULT__{"ok": false, "error": "..."}

The successful payload includes diagnostic fields beyond `latency` so the LLM
sees concrete failure modes (OOM/timeout) rather than just an aggregate score.

The dependency is pinned; an incompatible simulator must fail, not change scores.
"""
import ast
import json
import math
import sys
from pathlib import Path

from .config import RESOURCE_TOLERANCE


def validate_code(code):
    """Check the same registration contract before generation and evaluation."""
    registrations = {'register_scheduler': [], 'register_scheduler_init': []}
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id in registrations:
                    value = dec.args[0] if dec.args else next((k.value for k in dec.keywords if k.arg == 'key'), None)
                    registrations[dec.func.id].append(ast.literal_eval(value))
    if any(keys != ['experiment'] for keys in registrations.values()):
        raise ValueError('Exactly one of each scheduler decorator must use key="experiment"')
    return code


def verify_valid_assignment(pool, assignments):
    """Local fix for the pinned simulator's exact floating-point capacity checks.

    Use accurate sums and a tiny tolerance; do not clip or rewrite allocations.
    Installed dependency files remain untouched. Both arms use this worker.
    """
    for resource, available in [('cpu', pool.avail_cpu_pool), ('ram', pool.avail_ram_pool)]:
        values = [getattr(a, resource) for a in assignments]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f'Assignments require finite positive {resource.upper()}')
        allocated = math.fsum(values)
        if resource == 'ram' and pool.allow_memory_overcommit:
            continue
        if allocated > available and not math.isclose(allocated, available, **RESOURCE_TOLERANCE):
            raise ValueError(f'Overallocated {resource.upper()} in assignment: {allocated} > {available}')


def main() -> None:
    if len(sys.argv) != 4:
        print("__RESULT__" + json.dumps({"ok": False, "error": "expected 3 args"}))
        return

    _, scheduler_file, trace_file, params_json = sys.argv
    params = json.loads(params_json)

    src = Path(scheduler_file).read_text()
    try:
        validate_code(src)
        from typing import List, Tuple
        from eudoxia.executor.assignment import Assignment, ExecutionResult, Suspend
        from eudoxia.scheduler.decorators import register_scheduler, register_scheduler_init
        from eudoxia.utils import Priority
        from eudoxia.workload import OperatorState, Pipeline
        from eudoxia.workload.runtime_status import ASSIGNABLE_STATES

        exec(src, {
            "__builtins__": __builtins__,
            "__name__": "__worker__",
            "List": List, "Tuple": Tuple,
            "Pipeline": Pipeline, "OperatorState": OperatorState,
            "ASSIGNABLE_STATES": ASSIGNABLE_STATES,
            "Assignment": Assignment, "ExecutionResult": ExecutionResult,
            "Suspend": Suspend,
            "register_scheduler_init": register_scheduler_init,
            "register_scheduler": register_scheduler,
            "Priority": Priority,
        })
    except Exception as exc:
        print("__RESULT__" + json.dumps({"ok": False, "error": f"exec error: {exc}"}))
        return

    try:
        from eudoxia.simulator import run_simulator
        from eudoxia.workload.csv_io import CSVWorkloadReader
        from eudoxia.executor.resource_pool import ResourcePool

        ResourcePool.verify_valid_assignment = verify_valid_assignment
        params["scheduler_algo"] = 'experiment'
        with open(trace_file) as f:
            reader = CSVWorkloadReader(f)
            workload = reader.get_workload(params["ticks_per_second"])
            stats = run_simulator(params, workload=workload)
        weights = {Priority.QUERY: 10, Priority.INTERACTIVE: 5, Priority.BATCH_PIPELINE: 1}
        latency = stats.adjusted_latency(
            weights=weights, divide_by_completion_rate=False,
            unfinished_penalty_seconds=2 * params['max_job_seconds'],
        )

        def _f(v):
            return round(v, 2) if (isinstance(v, (int, float)) and math.isfinite(v)) else None

        def _ps(ps):
            return {
                "arrived": ps.arrival_count,
                "completed": ps.completion_count,
                "mean_s": _f(ps.mean_latency_seconds),
                "p99_s": _f(ps.p99_latency_seconds),
            }

        print("__RESULT__" + json.dumps({
            "ok": True,
            "latency": latency,
            "failures": stats.failures,
            "suspensions": stats.suspensions,
            "assignments": stats.assignments,
            "failure_error_counts": dict(stats.failure_error_counts),
            "oom_count": stats.failure_error_counts.get("OOM", 0),
            "mean_memory_consumed_percent": _f(stats.mean_memory_consumed_percent),
            "mean_memory_allocated_percent": _f(stats.mean_memory_allocated_percent),
            "containers_completed": stats.containers_completed,
            "query":       _ps(stats.pipelines_query),
            "interactive": _ps(stats.pipelines_interactive),
            "batch":       _ps(stats.pipelines_batch),
        }))
    except Exception as exc:
        print("__RESULT__" + json.dumps({"ok": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
