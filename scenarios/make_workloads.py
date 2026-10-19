"""Derive and validate full-window conditions from frozen canonical traces.

Hard is the complete canonical CSV. Medium is its first root task per job, without
resource rewriting. Simple changes only Medium's arrival times to zero.
Hard therefore contains more total work; this is not a work-matched ablation.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eudoxia.workload.csv_io import CSVOperatorRow, CSVWorkloadReader
from eudoxia.workload.pipeline import Segment

from src.config import EUDOXIA_REV, ROOT, SCALES, get_canonical_base_params

FIELDS = list(CSVOperatorRow._fields)
CONDITIONS = ('simple', 'medium', 'hard')
REFERENCE_REV = 'aeac04a9360e95a2beff3c30af025f307248d05c'
SOURCE_HASHES = {
    'train': '832d4427aa98385a50205ea1c81b387f8ce49e27430772ede4ec1b3febe694f9',
    'test': 'af5e5f27ba0065b78c1f98075afaaee4f9ffdd784dbd7c7ddcc7d0ae8fb269d4',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_family(source):
    """Retain every canonical job; change only the explicitly defined factors."""
    with Path(source).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, 'Unexpected source CSV columns')
        hard = list(reader)
    medium = [dict(row) for row in hard if row['arrival_seconds'] != '']
    require(medium and all(row['parents'] == '' for row in medium),
            'Each canonical job must begin with a root task')
    simple = [dict(row, arrival_seconds='0.0') for row in medium]
    return {'simple': simple, 'medium': medium, 'hard': hard}


def inspect_trace(path, jobs, cutoff):
    """Inspect the serialized CSV, including actual native-loader compatibility."""
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, f'{path}: unexpected CSV columns')
        rows = list(reader)
    groups, current = {}, None
    for row in rows:
        identifier = row['pipeline_id']
        require(identifier and None not in row and None not in row.values(), f'{path}: malformed row')
        if identifier != current:
            require(identifier not in groups, f'{path}: noncontiguous/repeated job ID')
            groups[identifier] = []
            current = identifier
        groups[identifier].append(row)
    require(len(groups) == jobs, f'{path}: expected {jobs} jobs')
    arrivals, priorities, shapes = [], Counter(), Counter()
    cpu = io = 0.0
    for identifier, operators in groups.items():
        first = operators[0]
        arrival = float(first['arrival_seconds'])
        require(math.isfinite(arrival) and 0 <= arrival < cutoff, f'{path}: invalid arrival')
        require(first['priority'] in ('QUERY', 'INTERACTIVE', 'BATCH_PIPELINE'), f'{path}: invalid priority')
        arrivals.append(arrival)
        priorities[first['priority']] += 1
        seen, children, joins = set(), Counter(), 0
        for index, row in enumerate(operators):
            if index:
                require(row['arrival_seconds'] == row['priority'] == '', f'{path}: repeated job header')
            op = row['operator_id']
            parents = row['parents'].split(';') if row['parents'] else []
            require(op and op not in seen, f'{path}: duplicate/empty task ID')
            require(len(parents) == len(set(parents)) and set(parents) <= seen,
                    f'{path}: cyclic, forward or missing dependency')
            seen.add(op)
            children.update(parents)
            joins += len(parents) > 1
            for field in ('baseline_cpu_seconds', 'storage_read_gb', 'memory_gb'):
                if field == 'memory_gb' and row[field] == '':
                    continue
                value = float(row[field])
                require(math.isfinite(value) and value >= 0, f'{path}: invalid {field}')
            require(row['cpu_scaling'] in Segment.SCALING_FUNCS, f'{path}: unsupported CPU scaling')
            cpu += float(row['baseline_cpu_seconds'])
            io += float(row['storage_read_gb'])
        shape = ('single_task' if len(operators) == 1 else 'fan_in' if joins else
                 'fan_out' if any(n > 1 for n in children.values()) else 'chain')
        shapes[shape] += 1
    require(arrivals == sorted(arrivals), f'{path}: arrivals not sorted')
    with path.open() as stream:
        loaded = list(CSVWorkloadReader(stream).batch_by_pipeline())
    require(len(loaded) == jobs, f'{path}: native loader job count mismatch')
    require(sum(len(item.pipeline.values) for item in loaded) == len(rows), f'{path}: native task count mismatch')
    return groups, {'jobs': jobs, 'tasks': len(rows), 'shapes': dict(shapes),
                    'priorities': dict(priorities), 'distinct_arrivals': len(set(arrivals)),
                    'first_arrival_seconds': min(arrivals), 'last_arrival_seconds': max(arrivals),
                    'total_baseline_cpu_seconds': cpu, 'total_storage_read_gb': io,
                    'sha256': sha256(path), 'native_loader_ok': True}


def validate_family(directory, jobs, cutoff):
    groups, summaries = {}, {}
    for condition in CONDITIONS:
        groups[condition], summaries[condition] = inspect_trace(directory / f'{condition}.csv', jobs, cutoff)
    simple, medium, hard = (groups[key] for key in CONDITIONS)
    require(list(simple) == list(medium) == list(hard), 'Job IDs/order differ between conditions')
    require(summaries['medium']['distinct_arrivals'] > 1, 'Medium must have staggered arrivals')
    require(summaries['hard']['tasks'] > jobs, 'Hard must contain multi-task jobs')
    require(summaries['hard']['shapes'].get('fan_in', 0) > 0 and
            summaries['hard']['shapes'].get('fan_out', 0) > 0, 'Hard must include both branch shapes')
    for identifier in simple:
        s, m, h = simple[identifier], medium[identifier], hard[identifier]
        require(len(s) == len(m) == 1, 'Simple/Medium must have exactly one task per job')
        require(float(s[0]['arrival_seconds']) == 0, 'Simple must have all jobs arrive at zero')
        require(s[0]['parents'] == m[0]['parents'] == '', 'Single tasks must have no dependencies')
        require({k: v for k, v in s[0].items() if k != 'arrival_seconds'} ==
                {k: v for k, v in m[0].items() if k != 'arrival_seconds'},
                'Simple/Medium differ outside arrival time')
        require(m[0] == h[0], 'Medium must preserve the exact first canonical root task and job header')
    return {'passed': True, 'traces': summaries,
            'checks': ['exactly one task in every Simple/Medium job', 'all Simple arrivals zero',
                       'staggered Medium arrivals', 'identical Simple/Medium except arrival time',
                       'same job IDs, priorities and arrival times in Medium/Hard',
                       'Medium task equals Hard first root task',
                       'Hard has multiple tasks, fan-in and fan-out',
                       'finite resources, unique IDs, topological dependencies, sorted arrivals',
                       'all CSVs load with Eudoxia']}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    logging.getLogger('eudoxia').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'scenarios/complexity-full')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--check-seed', action='store_true', help='also full-evaluate naive on all 6 traces and 5 scales; no API')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.validate_only:
        config = json.loads((output / 'workload_config.json').read_text())
    else:
        require(not output.exists() or not any(output.iterdir()), 'Output is not empty; choose a fresh directory')
        config = {'schema_version': 2, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'eudoxia_revision': EUDOXIA_REV, 'generator_sha256': sha256(Path(__file__)),
                  'seeds': {'train': 42, 'test': 137}, 'jobs_per_split': {},
                  'simulation_params': get_canonical_base_params(),
                  'arrival_window_seconds': 3600, 'source_reference_revision': REFERENCE_REV,
                  'construction': 'Hard = full canonical CSV; Medium = first root per job; Simple = Medium at t=0',
                  'work_matching': False,
                  'limitation': 'Hard has more total CPU/I/O work and possible concurrent RAM demand. '
                                'No work or memory equivalence is claimed. Late arrivals receive less than '
                                'a full job deadline before the 3600-second horizon, as in the canonical protocol.',
                  'sources': {}, 'trace_sha256': {}}
        for split in config['seeds']:
            source = ROOT / 'scenarios' / f'bench_canonical_{split}.csv'
            require(sha256(source) == SOURCE_HASHES[split], f'Canonical {split} source hash changed')
            traces = build_family(source)
            config['jobs_per_split'][split] = len(traces['medium'])
            config['sources'][split] = {'file': source.name, 'sha256': sha256(source),
                'url': f'https://github.com/BauplanLabs/wiscurds-playground/blob/{REFERENCE_REV}/'
                       f'summer2026/scenarios/traces/v1/{source.name}'}
            for condition, rows in traces.items():
                path = output / split / f'{condition}.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                if condition == 'hard':
                    shutil.copyfile(source, path)
                else:
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                config['trace_sha256'][f'{split}/{condition}.csv'] = sha256(path)
        write_json(output / 'workload_config.json', config)
    report = {'structural_validation_passed': True, 'families': {}}
    for split in config['seeds']:
        if config.get('schema_version', 1) == 1:
            params = config['native_generator_params'][split]
            jobs, cutoff = config['jobs_per_trace'], params['duration'] - params['max_job_seconds']
        else:
            jobs, cutoff = config['jobs_per_split'][split], config['arrival_window_seconds']
            require(sha256(output / split / 'hard.csv') == config['sources'][split]['sha256'],
                    'Hard is not the complete, unchanged canonical trace')
        report['families'][split] = validate_family(output / split, jobs, cutoff)
        if config.get('schema_version', 1) >= 2:
            last = report['families'][split]['traces']['hard']['last_arrival_seconds']
            require(last >= .95 * cutoff, 'Canonical arrivals do not cover the full window')
    require(all(sha256(output / name) == digest for name, digest in config['trace_sha256'].items()),
            'Trace hash differs from frozen configuration')
    # Rechecking file constraints must not erase earlier full simulator evidence.
    report_path = output / 'validation.json'
    if args.validate_only and report_path.exists():
        previous = json.loads(report_path.read_text())
        report.update({key: value for key, value in previous.items() if key.startswith('seed_evaluation')})
    write_json(output / 'validation.json', report)
    if args.check_seed:
        from src.evaluator import evaluate_scheduler
        report['seed_evaluations'] = {}
        for name in config['trace_sha256']:
            print(f'Full seed evaluation: {name}', flush=True)
            evaluation = evaluate_scheduler(ROOT / 'src/seed.py', trace=output / name)
            report['seed_evaluations'][name] = evaluation
            write_json(output / 'validation.json', report)
        report['seed_evaluations_passed'] = all(e['ok'] for e in report['seed_evaluations'].values())
        report['seed_evaluation_scales'] = SCALES
        write_json(output / 'validation.json', report)
        require(report['seed_evaluations_passed'], 'A full seed evaluation failed; inspect validation.json')
    print(f'Validated workload family: {output}', flush=True)


if __name__ == '__main__':
    main()
