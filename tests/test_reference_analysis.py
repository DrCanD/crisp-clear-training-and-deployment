"""Checks for reductions of archived evidence, not new experimental results."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from reproduce.analysis import analyse, paired_differences, read_table, summarise


class StatisticalReductionTests(unittest.TestCase):
    def test_sample_sd_and_missing_observations(self):
        records = [{"group": "x", "accuracy": x} for x in ("0.10", "0.20", "0.30", "")]
        result = summarise(records, ["group"], {"accuracy": (100, "percent")})[0]
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 20)
        self.assertAlmostEqual(result["sample_sd"], 10)
        singleton = summarise(records[:1], ["group"], {"accuracy": (100, "percent")})[0]
        self.assertEqual(singleton["sample_sd"], "")

    def test_pairing_uses_seed_identity_and_intervention_minus_baseline(self):
        records = [
            {"seed": "2", "arm": "new", "accuracy": "0.70"},
            {"seed": "1", "arm": "old", "accuracy": "0.80"},
            {"seed": "2", "arm": "old", "accuracy": "0.60"},
            {"seed": "1", "arm": "new", "accuracy": "0.75"},
        ]
        result = paired_differences(records, group_by=["seed"], condition="arm", baseline="old", intervention="new", metrics=["accuracy"])
        by_seed = {r["seed"]: r for r in result}
        self.assertAlmostEqual(by_seed["1"]["accuracy_change_pp"], -5)
        self.assertAlmostEqual(by_seed["2"]["accuracy_change_pp"], 10)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired_differences(records[:2], group_by=["seed"], condition="arm", baseline="old", intervention="new", metrics=["accuracy"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_differences(records + records[:1], group_by=["seed"], condition="arm", baseline="old", intervention="new", metrics=["accuracy"])


class ReferenceConcordanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name)
        cls.metadata = analyse(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def table(self, name):
        with (self.output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_matched_schedule_uses_completed_twelve_run_control(self):
        records = read_table("matched_slope_schedule.csv")
        self.assertEqual(len(records), 36)
        self.assertEqual(len({r["checkpoint_sha256"] for r in records}), 12)
        for arch in ("crisp", "pspikessm"):
            for arm in ("fixed_five", "annealed_one_to_five"):
                self.assertEqual({int(r["seed"]) for r in records if r["architecture"] == arch and r["schedule"] == arm}, {42, 123, 999})
        summaries = self.table("matched_slope_paired_summary.csv")
        expected = {"crisp": (0.6919905771495927, 2.3223024631718463), "pspikessm": (1.0895170789163762, 0.6405862755538003)}
        for arch, (mean, sample_sd) in expected.items():
            row = next(r for r in summaries if r["architecture"] == arch and r["inference"] == "single_draw")
            self.assertEqual(int(row["n"]), 3)
            self.assertAlmostEqual(float(row["mean"]), mean, places=10)
            self.assertAlmostEqual(float(row["sample_sd"]), sample_sd, places=10)

    def test_main_paired_sharpening_reductions(self):
        rows = self.table("sampler_sharpening_paired_summary.csv")
        for dataset, reduction in [("shd", 20.52414605418139), ("ssc", 30.836031792758313)]:
            row = next(r for r in rows if r["dataset"] == dataset and r["sampling_locations"] == "both" and r["metric"] == "deployment_gap_change_pp")
            self.assertAlmostEqual(float(row["mean"]), reduction, places=10)
            self.assertEqual(int(row["n"]), 3)

    def test_audit_reuses_draws_and_retains_reference_disagreements(self):
        rows = self.table("certification_implementation_audit.csv")
        self.assertEqual(len(rows), 1)
        for row in rows:
            self.assertEqual(int(row["audit_units"]), 78)
            self.assertEqual(int(row["reference_comparisons"]), 19_921_282)
            self.assertEqual(int(row["reference_disagreements"]), 9)
            self.assertTrue(math.isclose(float(row["maximum_cell_disagreement_rate"]), 0.00048146364949446316))
        self.assertIn("reuse cached draws", self.metadata["certificate_scope"])

    def test_sampler_sites_are_aggregated_before_seeds(self):
        raw = read_table("sampling_variance.csv")
        models = self.table("sampling_variance_by_seed.csv")
        self.assertEqual(len(raw), 114)
        self.assertEqual(len(models), 39)
        summary = self.table("sampling_variance_summary.csv")
        self.assertTrue(all(int(row["n"]) == 3 for row in summary))

    def test_provenance_and_relative_product_paths(self):
        self.assertEqual(self.metadata["result_kind"], "reference_summary")
        self.assertFalse(self.metadata["fresh_training_or_inference"])
        self.assertFalse(self.metadata["fresh_hardware_measurement"])
        for item in self.metadata["outputs"] + self.metadata["input_files"]:
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertNotIn("..", Path(item["path"]).parts)
        saved = json.loads((self.output / "analysis_metadata.json").read_text())
        self.assertEqual(saved, self.metadata)


if __name__ == "__main__":
    unittest.main()
