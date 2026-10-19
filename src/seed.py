"""Eudoxia's built-in naive baseline, registered as experiment.

Adapted from scheduler/naive.py at c62122cd15eca36d8872f7f9204d2f9c30774b28.
The worker supplies the imports. Queue rotation and failure handling are retained.
"""
@register_scheduler_init(key="experiment")
def init(s):
    s.waiting_queue = []
    s.multi_operator_containers = s.params["multi_operator_containers"]


@register_scheduler(key="experiment")
def schedule(s, results, pipelines):
    # No decision can change without arrivals or execution results.
    if not pipelines and not results:
        return [], []
    s.waiting_queue.extend(pipelines)
    assignments = []
    requeue_pipelines = []
    for pool_id in range(s.executor.num_pools):
        pool = s.executor.pools[pool_id]
        avail_cpu = pool.avail_cpu_pool
        avail_ram = pool.avail_ram_pool
        if avail_cpu <= 0 or avail_ram <= 0:
            continue
        while s.waiting_queue:
            pipeline = s.waiting_queue.pop(0)
            status = pipeline.runtime_status()
            has_failures = status.state_counts[OperatorState.FAILED] > 0
            if status.is_pipeline_successful() or has_failures or status.has_timed_out():
                continue
            requeue_pipelines.append(pipeline)
            if s.multi_operator_containers:
                ops = status.get_ops(ASSIGNABLE_STATES, require_parents_complete=False)
            else:
                ops = status.get_ops(ASSIGNABLE_STATES, require_parents_complete=True)[:1]
            if not ops:
                continue
            assignments.append(Assignment(ops=ops, cpu=avail_cpu, ram=avail_ram,
                                          priority=pipeline.priority, pool_id=pool_id,
                                          pipeline_id=pipeline.pipeline_id))
            break
    s.waiting_queue.extend(requeue_pipelines)
    return [], assignments
