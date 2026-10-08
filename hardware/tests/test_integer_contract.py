"""Independent checks for the checkpoint, table export and board event encoding."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HARDWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARDWARE))
import integer_reference as reference
import quantize
from board_interface import Package
from verify_integer import load_model


class IntegerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        quantize.device = 'cpu'
        cls.fixture = HARDWARE.parent / 'data/reference/hardware/fixture'
        cls.q, cls.params, cls.read = load_model(cls.fixture)

    def test_checkpoint_recreates_original_tables(self):
        model, beta, _ = quantize.load_fixed_checkpoint()
        exported = reference.quantize(reference.fold_params(model.state_dict(), beta), self.q['fmt'])
        for actual, expected in zip(exported['layers'] + [exported['ro']], self.q['layers'] + [self.q['ro']]):
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        for key in ('NOISE', 'SIG', 'THR_SEQ', 'THR_FIX'):
            np.testing.assert_array_equal(exported[key], self.q[key])

    def test_board_packing_retains_input_counts(self):
        package = Package(self.fixture)
        for index in (0, 1, 127):
            dense = np.zeros((package.T, package.NIN), dtype=np.int64)
            timestep = 0
            for value in package.stream(index):
                value = int(value)
                if value == 0xffff:
                    timestep += 1
                else:
                    dense[timestep, value & 0x3ff] += value >> 10
            self.assertEqual(timestep, package.T)
            expected = reference.events_to_dense(package.events[package.offsets[index]:package.offsets[index + 1]], package.T, package.NIN)
            np.testing.assert_array_equal(dense, expected)

    def test_export_can_be_loaded_by_the_board_driver(self):
        model, beta, checkpoint_hash = quantize.load_fixed_checkpoint()
        exported = reference.quantize(reference.fold_params(model.state_dict(), beta), self.q['fmt'])
        read = type(self).read
        offsets, events = read('test_offsets.bin'), read('test_events.bin')
        inputs = np.stack([reference.events_to_dense(events[offsets[i]:offsets[i + 1]], 100, 700) for i in range(2)])
        votes = read('expected_certificate_votes.bin')[:2]
        reference_votes = read('expected_reference_votes.bin')[:2]
        labels = read('test_labels.bin')[:2]
        certificate = quantize.certificate_summary(exported, votes, reference_votes, labels)
        parent = HARDWARE.parent / 'outputs/hardware'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            destination = Path(tmp)
            quantize.write_package(exported, inputs, labels, votes, reference_votes,
                                  read('expected_mean_field_decisions.bin')[:2], read('expected_mean_field_logits.bin')[:2],
                                  certificate, read('expected_layer1_logits.bin')[:2], {}, self.params['widths'],
                                  destination, 2, {'key': self.params['key'], 'ckpt_sha256': checkpoint_hash, 'data_fp': 'test'})
            result = Package(destination)
            np.testing.assert_array_equal(result.exp['votes_c'], votes)
            np.testing.assert_array_equal(result.exp['mf_logits'], read('expected_mean_field_logits.bin')[:2])
            self.assertEqual(result.n, 2)

    def test_empty_events_and_zero_weights(self):
        self.assertEqual(reference.dense_to_events(np.zeros((100, 700))).size, 0)
        weights, scale = reference.q_weights(np.zeros((2, 3)), 8)
        self.assertTrue(np.all(weights == 0))
        self.assertEqual(scale, 1.0)


if __name__ == '__main__':
    unittest.main()
