# Workloads

Fixed train/test traces derived from the hash-pinned summer2026 v1 canonical inputs.

| Condition | Tasks per job | Arrivals |
| --- | --- | --- |
| Simple | First root task | All at zero |
| Medium | First root task | Original staggered arrivals |
| Hard | Complete original DAG | Original staggered arrivals |

Simple and Medium differ only in arrival time. Hard preserves canonical bytes and has more total work; this is not an equal-work topology comparison. Single-task jobs and chains remain valid DAGs.

Train: 4,110 jobs; Hard: 28,083 tasks. Test: 4,030 jobs; Hard: 27,443 tasks.
The arrival window and simulation horizon are both 3,600 seconds; late jobs have less time before the horizon.
Source URLs and hashes are recorded in `workload_config.json`. Test traces are for evaluation after search.

Validate from the project root:

```sh
uv run python -m scenarios.make_workloads --validate-only
```

To regenerate, use `--output` with a new empty directory.
