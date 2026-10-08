"""Checks for reductions of archived evidence, not new experimental results."""
import csv
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import mean, median

from reproduce.analysis import (
    ACCELERATOR_MANIFEST_PATH, accelerator_summary, analyse,
    coarse_training_summary, paired_differences, read_table, summarise,
)


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

    def test_epoch_speedup_is_ratio_of_seed_means(self):
        records = []
        for seed, fine, coarse, epochs in [("1", 10, 2, 100), ("2", 20, 10, 40), ("3", 30, 20, 20)]:
            for factor, duration in [(1, fine), (2, coarse)]:
                records.append(dict(dataset="example", seed=seed, coarsening_factor=str(factor),
                    temporal_bins=str(100 // factor), mean_epoch_s=str(duration), epochs_run=str(epochs),
                    augmentation_domain="fine" if factor == 1 else "fine_then_pool",
                    paired_epoch_speedup=str(fine / duration)))
        result = next(r for r in coarse_training_summary(records) if r["coarsening_factor"] == 2)
        self.assertAlmostEqual(result["mean_epoch_s"], 32 / 3)
        self.assertAlmostEqual(result["speedup_ratio_of_mean_epoch_times"], 1.875)
        self.assertNotAlmostEqual(result["speedup_ratio_of_mean_epoch_times"], mean([5, 2, 1.5]))
        wrong_protocol = copy.deepcopy(records)
        wrong_protocol[1]["augmentation_domain"] = "coarse"
        with self.assertRaisesRegex(ValueError, "augmentation protocol"):
            coarse_training_summary(wrong_protocol)
        with self.assertRaisesRegex(ValueError, "Unpaired training seeds"):
            coarse_training_summary(records[:-1])

    def test_accelerator_rejects_missing_repetitions_and_failed_gates(self):
        configurations = read_table("accelerator_benchmarks.csv")
        repetitions = read_table("accelerator_timing_repetitions.csv")
        provenance = json.loads(ACCELERATOR_MANIFEST_PATH.read_text())
        with self.assertRaisesRegex(ValueError, "Missing timing repetitions"):
            accelerator_summary(configurations, repetitions[1:], provenance)
        with self.assertRaisesRegex(ValueError, "Duplicate timing repetition"):
            accelerator_summary(configurations, repetitions + repetitions[:1], provenance)
        failed = copy.deepcopy(provenance)
        failed["gate_checks"][0]["checks"]["pass"] = False
        with self.assertRaisesRegex(ValueError, "failed correctness gate"):
            accelerator_summary(configurations, repetitions, failed)
        failed_configurations = copy.deepcopy(configurations)
        failed_configurations[0]["status"] = "oom"
        with self.assertRaisesRegex(ValueError, "Unsuccessful configuration contains timings"):
            accelerator_summary(failed_configurations, repetitions, provenance)


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

    def test_accelerator_medians_use_all_original_repetitions(self):
        configurations = self.table("accelerator_benchmark_summary.csv")
        repetitions = read_table("accelerator_timing_repetitions.csv")
        self.assertEqual(len(configurations), 70)
        self.assertEqual(len(repetitions), 951)
        successful = [r for r in configurations if r["status"] == "ok"]
        self.assertEqual(len(successful), 65)
        for row in successful:
            values = [float(r["step_ms"]) for r in repetitions
                      if all(r[k] == row[k] for k in ("device", "method", "sequence_length", "batch_size"))]
            self.assertEqual(len(values), int(row["timed_repetitions"]))
            self.assertAlmostEqual(float(row["median_step_ms"]), median(values), places=10)
        spsn = next(r for r in successful if r["method"] == "SPSN official code" and r["sequence_length"] == "2000")
        chunked = next(r for r in successful if r["device"] == "TPU" and r["method"] == "CRISP chunked matrix" and r["sequence_length"] == "500")
        self.assertEqual(round(float(spsn["median_step_ms"]), 1), 24.0)
        self.assertEqual(round(float(chunked["median_step_ms"]), 1), 18.3)
        for row in configurations:
            if row["status"] != "ok":
                self.assertEqual(row["median_step_ms"], "")
        ratios = self.table("runtime_ratios.csv")
        failed = {(r["device"], r["method"], r["sequence_length"]) for r in configurations if r["status"] != "ok"}
        self.assertTrue(all((r["device"], r["method"], r["time_steps"]) not in failed for r in ratios))

    def test_coarse_speedups_match_recovered_augmentation_protocol(self):
        source = read_table("coarse_training_timing.csv")
        self.assertEqual(len(source), 24)
        expected = {"shd": [2.637349938745, 6.639500947584366, 9.844992061292908],
                    "ssc": [2.648680015753316, 6.7136952356829545, 10.177479459242928]}
        results = self.table("coarse_training_timing_summary.csv")
        for dataset, ratios in expected.items():
            for factor, ratio in zip((2, 4, 10), ratios):
                row = next(r for r in results if r["dataset"] == dataset and int(r["coarsening_factor"]) == factor)
                self.assertEqual(int(row["training_seeds"]), 3)
                self.assertAlmostEqual(float(row["speedup_ratio_of_mean_epoch_times"]), ratio, places=12)
        provenance = json.loads(ACCELERATOR_MANIFEST_PATH.read_text())
        transfer = read_table("time_step_transfer.csv")
        for protocol in provenance["coarse_protocol_validation"]:
            records = [r for r in transfer if r["architecture"] == "crisp" and r["dataset"] == protocol["dataset"]
                       and int(r["transfer_factor"]) == protocol["coarsening_factor"]]
            self.assertEqual(len(records), 3)
            self.assertAlmostEqual(mean(float(r["native_accuracy"]) for r in records), protocol["native_mean_field_accuracy"], places=12)
            self.assertAlmostEqual(mean(float(r["transferred_accuracy"]) for r in records), protocol["retimed_fine_accuracy"], places=12)

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
