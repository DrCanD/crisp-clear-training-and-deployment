"""Full-set power reproduction and checks against corrupted measurement records."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HARDWARE = Path(__file__).resolve().parents[1]
ROOT = HARDWARE.parent
sys.path.insert(0, str(HARDWARE))
import analyze_power


class PowerAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampled = analyze_power.load_measurement(
            ROOT / 'data/reference/hardware/power_sampled.json', expected_variant='sampled')
        cls.mean_field = analyze_power.load_measurement(
            ROOT / 'data/reference/hardware/power_mean_field.json', expected_variant='mf')

    def load_changed_batch(self, change, expected_variant='mf'):
        data = copy.deepcopy(self.mean_field)
        data['repeats'] = data['repeats'][:1]
        count = len(data['repeats'][0]['inputs'])
        change(data)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'measurement.json'
            path.write_text(json.dumps(data))
            return analyze_power.load_measurement(path, expected_inputs=count,
                                                  expected_variant=expected_variant)

    def test_default_cli_reproduces_fullset_mean_field_and_ratio(self):
        output = ROOT / 'outputs' / 'hardware'
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            destination = Path(temporary)
            args = ['analyze_power.py', '--output', destination.relative_to(ROOT).as_posix()]
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                analyze_power.main()
            summary = json.loads((destination / 'summary.json').read_text())
            mf = summary['modes']['MF']
            self.assertEqual((mf['batches'], mf['inputs']), (28, 2264))
            self.assertAlmostEqual(mf['energy_uJ_per_input'], 563.939422448099, places=9)
            self.assertAlmostEqual(mf['energy_standard_error_uJ'], 1.5964321159206956, places=10)
            self.assertAlmostEqual(mf['energy_ci95_uJ'][0], 560.657909237029, places=9)
            self.assertAlmostEqual(mf['energy_ci95_uJ'][1], 567.220935659169, places=9)
            self.assertEqual(mf['residual_degrees_of_freedom'], 26)
            self.assertAlmostEqual(mf['processing_time_ms_per_input'], 3.3089013112757217, places=10)
            self.assertAlmostEqual(summary['sequential_over_mean_field'], 4.61716364833731, places=10)
            with (destination / 'batches.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 28 * (5 + 3))
            for mode in ('D0', 'PREFIX'):
                self.assertEqual({r['variant'] for r in rows if r['mode'] == mode}, {'sampled', 'mf'})
            for source in summary['sources'].values():
                self.assertFalse(Path(source['path']).is_absolute())
                self.assertEqual(source['sha256'], hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest())

    def test_sampled_only_cli_remains_available(self):
        output = ROOT / 'outputs' / 'hardware'
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            destination = Path(temporary)
            args = ['analyze_power.py', '--mean-field', '', '--output', destination.relative_to(ROOT).as_posix()]
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                analyze_power.main()
            summary = json.loads((destination / 'summary.json').read_text())
            self.assertNotIn('MF', summary['modes'])
            self.assertNotIn('sequential_over_mean_field', summary)
            self.assertEqual(set(summary['sources']), {'sampled'})

    def test_archived_record_hashes_and_build_identities(self):
        manifest = json.loads((ROOT / 'data/manifests/power_measurements.json').read_text())
        self.assertEqual({record['variant'] for record in manifest['records']}, {'sampled', 'mf'})
        for record in manifest['records']:
            raw = (ROOT / record['path']).read_bytes()
            data = json.loads(raw)
            receipt = json.loads((ROOT / record['build_receipt_path']).read_text())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])
            self.assertEqual(data['source_sha256'], record['source_measurement_sha256'])
            self.assertEqual(data['settings']['variant'], record['variant'])
            self.assertEqual(receipt['variant'], record['variant'])
            self.assertEqual(data['package'], record['package'])
            self.assertEqual(receipt['package'], record['package'])
            self.assertTrue(receipt['completed'])
            self.assertEqual(data['environment']['bitstream_sha256'], record['bitstream_sha256'])
            self.assertEqual(receipt['bitstream_sha256'], record['bitstream_sha256'])
            self.assertEqual(data['environment']['clock_hz'], record['clock_hz'])
            self.assertEqual(len(data['repeats']), record['batches'])
            self.assertEqual(sum(len(batch['inputs']) for batch in data['repeats']), record['unique_inputs'])
            blocks = [block for batch in data['repeats']
                      for block in list(batch['runs'].values()) + batch['idles']]
            self.assertEqual(sum(len(block['samples']) for block in blocks), record['raw_sensor_samples'])

    def test_duplicate_input_rejected(self):
        def change(data):
            data['repeats'][0]['inputs'][-1] = 0
        with self.assertRaisesRegex(ValueError, 'disjoint batches'):
            self.load_changed_batch(change)

    def test_incomplete_batch_rejected(self):
        def change(data):
            data['repeats'][0]['complete'] = False
        with self.assertRaisesRegex(ValueError, 'input batch is incomplete'):
            self.load_changed_batch(change)

    def test_wrong_variant_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected the sampled measurement variant'):
            self.load_changed_batch(lambda data: None, expected_variant='sampled')

    def test_missing_mode_rejected(self):
        def change(data):
            del data['repeats'][0]['runs']['D0']
        with self.assertRaisesRegex(ValueError, 'required variant mode'):
            self.load_changed_batch(change)

    def test_changed_sensor_sample_rejected(self):
        def change(data):
            data['repeats'][0]['runs']['MF']['samples'][0]['power_mW'] += 100
        with self.assertRaisesRegex(ValueError, 'raw sensor samples'):
            self.load_changed_batch(change)

    def test_changed_idle_bracket_rejected(self):
        def change(data):
            data['repeats'][0]['runs']['MF']['idle_bracket_mW'] += 1
        with self.assertRaisesRegex(ValueError, 'neighboring idle blocks'):
            self.load_changed_batch(change)

    def test_counter_mismatch_rejected(self):
        def change(data):
            data['repeats'][0]['runs']['MF']['items'] += 1
        with self.assertRaisesRegex(ValueError, 'item counter differs'):
            self.load_changed_batch(change)

    def test_window_throughput_is_not_whole_run_throughput(self):
        data = self.load_changed_batch(lambda data: None)
        row = data['repeats'][0]['runs']['MF']
        self.assertNotAlmostEqual(row['decisions_per_s'], row['items'] / row['elapsed_s'], places=6)

    def test_noninteger_window_counter_delta_rejected(self):
        def change(data):
            row = data['repeats'][0]['runs']['MF']
            row['decisions_per_s'] += 0.25 / row['duration_s']
            row['s_per_decision'] = 1 / row['decisions_per_s']
        with self.assertRaisesRegex(ValueError, 'integer item-counter delta'):
            self.load_changed_batch(change)

    def test_different_package_rejected_before_comparison(self):
        data = copy.deepcopy(self.mean_field)
        data['package'] = 'different'
        with self.assertRaisesRegex(ValueError, 'input packages differ'):
            analyze_power.validate_comparison(self.sampled, data)

    def test_different_clock_rejected_before_comparison(self):
        data = copy.deepcopy(self.mean_field)
        data['environment']['clock_hz'] += 1
        with self.assertRaisesRegex(ValueError, 'clocks differ'):
            analyze_power.validate_comparison(self.sampled, data)


if __name__ == '__main__':
    unittest.main()
