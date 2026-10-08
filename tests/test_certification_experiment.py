"""Small numerical checks for certificate sampling and audit reductions."""
import unittest

import numpy as np
import torch

from reproduce.certification_experiment import (
    analyse_decisions, group_decisions, group_budgets, sample_logits,
    sampling_checks, sequential_looks,
)
from reproduce.models.crisp import Config, build_model
from reproduce.training import set_all_seeds


class CertificateExperimentTests(unittest.TestCase):
    def test_grouped_class_uses_mean_logits_not_vote_majority(self):
        logits = np.array([[[100., 0.]], [[-1., 0.]], [[-1., 0.]]])
        self.assertEqual(group_decisions(logits, 1).tolist(), [[0, 1, 1]])
        self.assertEqual(group_decisions(logits, 3).tolist(), [[0]])

    def test_two_sided_feasible_looks_and_all_group_budgets(self):
        self.assertEqual(sequential_looks(1, 256, .001), [16, 32, 64, 128, 256])
        self.assertEqual(sum(len(group_budgets(g, 256)) for g in (1, 2, 4, 8, 16)), 35)
        self.assertEqual(sequential_looks(16, 256, .001), [16])
        self.assertEqual(sequential_looks(1, 4, .001), [])

    def test_abstention_denominators_and_sequential_draw_cost(self):
        votes = np.stack([np.zeros(256, dtype=np.int16), np.tile(np.array([0, 1], dtype=np.int16), 128)])
        arrays = {"labels": np.array([0, 1]), "mean_field_predictions": np.array([0, 0]),
                  "certificate_group_1": votes, "reference_group_1": votes.copy()}
        result = analyse_decisions(arrays, 2, alphas=[.001])
        full = result["fixed"][-1]
        self.assertEqual(full["correct_and_accepted_fraction"], .5)
        self.assertEqual(full["abstention_fraction"], .5)
        self.assertEqual(full["wrong_among_accepted_fraction"], 0.)
        self.assertEqual(full["resolved_reference_inputs"], 1)
        self.assertEqual(result["sequential"][0]["mean_draws_all_inputs"], 136.)
        self.assertEqual(result["sequential"][0]["mean_draws_accepted"], 16.)
        self.assertEqual(result["sampling_noise"][0]["mean_group_accuracy"], .75)

    def test_crisp_cached_forward_and_continued_reference_stream(self):
        set_all_seeds(31)
        model = build_model(Config(n_in=3, n_classes=2, n_layers=2, n_hidden=4,
                                   k_modes=2, hidden_dropout=0), device="cpu").eval()
        inputs = torch.randn(2, 4, 3)
        checked = sampling_checks(model, inputs, "crisp", beta=1., chunk=2)
        self.assertTrue(checked["cached_first_layer_matches_full_forward"])
        set_all_seeds(45)
        certificate = sample_logits(model, inputs, "crisp", beta=1., count=8, chunk=2)
        reference = sample_logits(model, inputs, "crisp", beta=1., count=8, chunk=2)
        set_all_seeds(45)
        combined = sample_logits(model, inputs, "crisp", beta=1., count=16, chunk=2)
        self.assertTrue(torch.equal(torch.cat([certificate, reference]), combined))
        model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            sample_logits(model, inputs, "crisp", beta=1., count=1)


if __name__ == "__main__":
    unittest.main()
