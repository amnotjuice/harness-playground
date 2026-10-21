from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EUDOXIA_REV = "c62122cd15eca36d8872f7f9204d2f9c30774b28"
SCALES = [1, 2, 4, 8, 16]
TRACE = ROOT / "scenarios/train/hard.csv"
# Applies to total CPU/RAM allocations; this is rounding tolerance, not overcommit.
RESOURCE_TOLERANCE = {'rel_tol': 1e-12, 'abs_tol': 1e-9}

CANONICAL_SIM_PARAMS = {
    "duration": 3600,           # 60 minutes
    "ticks_per_second": 100,    # coarser than old 1000
    "num_pools": 1,
    "cpus_per_pool": 64,
    "ram_gb_per_pool": 500,
    "num_pipelines": 10,
    "num_operators": 10,
    "waiting_seconds_mean": 10.0,
    "multi_operator_containers": False,
    "interactive_prob": 0.3,
    "query_prob": 0.1,
    "batch_prob": 0.6,
    "random_seed": 42,
    "per_trace_timeout": None,
    "subprocess_timeout": 180,   # 3-min wall clock per single sim (per scale).
                                  # Historical max for functional schedulers in 01/02 was 119s/sim,
                                  # so 180s gives ~1.5x headroom while killing infinite loops fast.
    # Max job time = 6 minutes (eudoxia param name: max_job_seconds)
    "max_job_seconds": 360,
}

PROTOTYPE_OVERRIDES = {
    "duration": 60,             # 1 minute
    "ticks_per_second": 10,
    "subprocess_timeout": 120,
}


def default_agent_timeout(prototype=False, direct_timeout=300):
    """Allow two full evaluator passes plus time for model calls and editing."""
    params = dict(CANONICAL_SIM_PARAMS)
    if prototype:
        params.update(PROTOTYPE_OVERRIDES)
    return direct_timeout + 2 * len(SCALES) * params['subprocess_timeout']


def get_canonical_base_params(prototype: bool = False) -> dict:
    from eudoxia.simulator import get_param_defaults

    params = get_param_defaults()
    params.update(CANONICAL_SIM_PARAMS)
    if prototype:
        params.update(PROTOTYPE_OVERRIDES)
    return params
