"""Check serialized constraints and pairing, not just generator parameters."""
import copy
import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scenarios.make_workloads import FIELDS, SOURCE_HASHES, build_family, main, sha256, validate_family
from src.config import ROOT, get_canonical_base_params


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_family = build_family(ROOT / 'scenarios/train/hard.csv')
        ids = {row['pipeline_id'] for row in cls.full_family['medium'][:40]}
        cls.family = {name: [row for row in rows if row['pipeline_id'] in ids]
                      for name, rows in cls.full_family.items()}
        cls.params = get_canonical_base_params()
        cls.cutoff = cls.params['duration']

    def write_family(self, directory, family):
        for name, rows in family.items():
            with (directory / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    def test_deterministic_derivation_preserves_every_canonical_job(self):
        same = build_family(ROOT / 'scenarios/train/hard.csv')
        other = build_family(ROOT / 'scenarios/test/hard.csv')
        self.assertEqual(self.full_family, same)
        self.assertNotEqual(self.full_family, other)
        self.assertEqual(len(same['medium']), 4110)
        self.assertEqual(len(same['hard']), 28083)
        self.assertGreater(float(same['medium'][-1]['arrival_seconds']), 3590)
        for split in ('train', 'test'):
            self.assertEqual(sha256(ROOT / 'scenarios' / split / 'hard.csv'), SOURCE_HASHES[split])

    def test_full_cli_reproduction_and_complete_hard_source_identity(self):
        hashes = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with patch('sys.argv', ['make_workloads.py', '--output', str(root)]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    main()
                config = json.loads((root / 'workload_config.json').read_text())
                self.assertEqual(config['schema_version'], 2)
                self.assertEqual(config['jobs_per_split']['train'], 4110)
                for split in ('train', 'test'):
                    self.assertEqual((root / split / 'hard.csv').read_bytes(),
                                     (ROOT / 'scenarios' / split / 'hard.csv').read_bytes())
                hashes.append(config['trace_sha256'])
        self.assertEqual(hashes[0], hashes[1])

    def test_csv_pairing_constraints_and_native_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_family(directory, self.family)
            report = validate_family(directory, 40, self.cutoff)
        self.assertTrue(report['passed'])
        traces = report['traces']
        self.assertEqual(traces['simple']['tasks'], 40)
        self.assertEqual(traces['medium']['tasks'], 40)
        self.assertGreater(traces['hard']['tasks'], 40)
        self.assertEqual(traces['simple']['priorities'], traces['hard']['priorities'])
        self.assertEqual(traces['simple']['total_baseline_cpu_seconds'], traces['medium']['total_baseline_cpu_seconds'])
        self.assertGreater(traces['hard']['total_baseline_cpu_seconds'], traces['medium']['total_baseline_cpu_seconds'])
        self.assertEqual(traces['medium']['last_arrival_seconds'], traces['hard']['last_arrival_seconds'])

    def test_invalid_serialized_conditions_are_rejected(self):
        for fault in ['simple_time', 'medium_resource', 'hard_cycle', 'extra_task', 'missing_job']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                family = copy.deepcopy(self.family)
                if fault == 'simple_time':
                    family['simple'][-1]['arrival_seconds'] = '1'
                elif fault == 'medium_resource':
                    family['medium'][0]['baseline_cpu_seconds'] = '999'
                elif fault == 'hard_cycle':
                    row = family['hard'][0]
                    row['parents'] = row['operator_id']
                elif fault == 'extra_task':
                    row = dict(family['medium'][0], operator_id='extra', arrival_seconds='', priority='')
                    family['medium'].insert(1, row)
                else:
                    family['simple'].pop()
                directory = Path(temp)
                self.write_family(directory, family)
                with self.assertRaises(ValueError):
                    validate_family(directory, 40, self.cutoff)

    def test_forward_references_and_nonfinite_resources_are_rejected(self):
        for field, value in [('parents', 'missing'), ('storage_read_gb', 'nan'),
                             ('baseline_cpu_seconds', '-1'), ('memory_gb', 'inf')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                family = copy.deepcopy(self.family)
                family['hard'][0][field] = value
                directory = Path(temp)
                self.write_family(directory, family)
                with self.assertRaises(ValueError):
                    validate_family(directory, 40, self.cutoff)

    def test_validation_only_preserves_existing_simulation_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train').mkdir()
            self.write_family(root / 'train', self.family)
            config = {'jobs_per_trace': 40, 'seeds': {'train': 42},
                      'native_generator_params': {'train': self.params},
                      'trace_sha256': {f'train/{name}.csv': sha256(root / 'train' / f'{name}.csv')
                                       for name in self.family}}
            (root / 'workload_config.json').write_text(json.dumps(config))
            prior = {'seed_evaluations': {'train/simple.csv': {'ok': True, 'score': 12.3}},
                     'seed_evaluations_passed': True}
            (root / 'validation.json').write_text(json.dumps(prior))
            with patch('sys.argv', ['make_workloads.py', '--validate-only', '--output', str(root)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((root / 'validation.json').read_text())
            self.assertTrue(report['structural_validation_passed'])
            self.assertEqual(report['seed_evaluations'], prior['seed_evaluations'])


if __name__ == '__main__':
    unittest.main()
