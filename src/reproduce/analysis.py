"""Recompute numerical summaries from archived reference observations.

This module performs no training, checkpoint inference, sensor acquisition or
plotting. Outputs are explicitly labelled ``reference_summary``. Accuracies in
input tables are fractions; reported means use percent and differences use
percentage points. Standard deviations use the sample denominator n-1.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifests" / "reference_files.json"


def read_table(name: str, reference_dir: Path | None = None) -> list[dict[str, str]]:
    """Read one reference table by its basename, independent of the working directory."""
    if Path(name).name != name:
        raise ValueError("A reference table must be a basename.")
    with ((reference_dir or REFERENCE_DIR) / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarise(
    rows: Iterable[Mapping],
    group_by: Sequence[str],
    metrics: Mapping[str, tuple[float, str]],
) -> list[dict]:
    """Return mean, sample SD and observation count for each available metric.

    ``metrics`` maps a numeric column to its (scale, unit). Missing cells are
    omitted for that metric. A singleton has no estimable sample SD (blank).
    Callers must supply one independent seed record per group, not sampler sites
    or repeated measurements from the same seed.
    """
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_by)].append(row)
    result = []
    for keys, records in groups.items():
        for metric, (scale, unit) in metrics.items():
            values = [float(r[metric]) * scale for r in records if r.get(metric) not in (None, "")]
            if not values:
                continue
            result.append({**dict(zip(group_by, keys)), "metric": metric,
                           "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else "",
                           "n": len(values), "unit": unit})
    return result


def paired_differences(
    rows: Iterable[Mapping],
    *, group_by: Sequence[str], condition: str, baseline: str,
    intervention: str, metrics: Sequence[str], scale: float = 100.0,
) -> list[dict]:
    """Subtract baseline from intervention within an explicitly matched seed.

    Duplicate or unpaired conditions raise rather than silently pool seeds.
    Missing metrics remain blank; missing observations are never replaced by 0.
    """
    pairs: dict[tuple, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        arm = row[condition]
        if arm not in (baseline, intervention):
            continue
        key = tuple(row[k] for k in group_by)
        if arm in pairs[key]:
            raise ValueError(f"Duplicate paired observation for {key}: {arm}")
        pairs[key][arm] = row
    result = []
    for key, pair in pairs.items():
        if set(pair) != {baseline, intervention}:
            raise ValueError(f"Unpaired observation for {key}")
        record = {**dict(zip(group_by, key)), "comparison": f"{intervention} minus {baseline}"}
        for metric in metrics:
            left, right = pair[baseline].get(metric), pair[intervention].get(metric)
            record[metric + "_change_pp"] = (float(right) - float(left)) * scale if left not in (None, "") and right not in (None, "") else ""
        result.append(record)
    return result


def _with_gap(rows: list[dict]) -> list[dict]:
    return [{**r, "deployment_gap_pp": (float(r["single_draw_accuracy"]) - float(r["mean_field_accuracy"])) * 100
             if r.get("mean_field_accuracy") not in (None, "") else ""} for r in rows]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No observations for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(output_dir: str | Path) -> dict:
    """Write reproducible CSV/JSON summaries of stored reference evidence.

    ``output_dir`` is interpreted as supplied by the caller; reference inputs
    are resolved relative to this package, never relative to a personal path.
    The returned metadata names only output basenames and repo-relative inputs.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = REFERENCE_DIR / Path(entry["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path.name}")
    products: dict[str, list[dict]] = {}
    accuracy_metrics = {"mean_field_accuracy": (100, "percent"), "single_draw_accuracy": (100, "percent"),
                        "five_draw_accuracy": (100, "percent"), "deployment_gap_pp": (1, "percentage_points")}
    architecture = _with_gap(read_table("architecture_training.csv"))
    products["architecture_training_summary.csv"] = summarise(architecture, ["dataset", "architecture", "training_rule"], accuracy_metrics)

    locations = read_table("sampling_locations.csv")
    for row in architecture:
        if row["architecture"] == "pspikessm" and row["training_rule"] == "mean_field":
            locations.append({**row, "sampler_schedule": "fixed_one", "sampling_locations": "both"})
    locations = _with_gap(locations)
    products["sampling_locations_summary.csv"] = summarise(locations, ["dataset", "architecture", "sampling_locations", "sampler_schedule"], accuracy_metrics)
    pairs = paired_differences(locations, group_by=["dataset", "architecture", "sampling_locations", "seed"],
                               condition="sampler_schedule", baseline="fixed_one", intervention="annealed_one_to_five",
                               metrics=["mean_field_accuracy", "single_draw_accuracy"])
    for row in pairs:
        row["deployment_gap_change_pp"] = row["single_draw_accuracy_change_pp"] - row["mean_field_accuracy_change_pp"]
    products["sampler_sharpening_paired_seeds.csv"] = pairs
    products["sampler_sharpening_paired_summary.csv"] = summarise(pairs, ["dataset", "architecture", "sampling_locations", "comparison"],
        {k: (1, "percentage_points") for k in ["mean_field_accuracy_change_pp", "single_draw_accuracy_change_pp", "deployment_gap_change_pp"]})
    products["sampler_slope_summary.csv"] = summarise(_with_gap(read_table("sampler_sharpening.csv")),
        ["dataset", "architecture", "beta_start", "beta_end"], {**accuracy_metrics, "threshold_accuracy": (100, "percent")})

    matched = read_table("matched_slope_schedule.csv")
    products["matched_slope_schedule_summary.csv"] = summarise(matched, ["dataset", "architecture", "schedule", "inference"],
        {"accuracy": (100, "percent"), "macro_f1": (1, "fraction"), "firing_rate": (1, "fraction"), "graded_fraction": (1, "fraction")})
    pairs = paired_differences(matched, group_by=["dataset", "architecture", "inference", "seed"], condition="schedule",
                               baseline="fixed_five", intervention="annealed_one_to_five", metrics=["accuracy"])
    products["matched_slope_paired_seeds.csv"] = pairs
    products["matched_slope_paired_summary.csv"] = summarise(pairs, ["dataset", "architecture", "inference", "comparison"],
        {"accuracy_change_pp": (1, "percentage_points")})
    loss_rows = []
    by_run: dict[tuple, dict] = defaultdict(dict)
    for row in matched:
        by_run[(row["dataset"], row["architecture"], row["schedule"], row["seed"])][row["inference"]] = float(row["accuracy"])
    for (dataset, arch, schedule, seed), metrics in by_run.items():
        loss_rows.append(dict(dataset=dataset, architecture=arch, schedule=schedule, seed=seed,
                              deployment_loss=metrics["mean_field"] - metrics["single_draw"]))
    loss_pairs = paired_differences(loss_rows, group_by=["dataset", "architecture", "seed"], condition="schedule",
        baseline="fixed_five", intervention="annealed_one_to_five", metrics=["deployment_loss"])
    products["matched_slope_deployment_loss_paired.csv"] = loss_pairs
    products["matched_slope_deployment_loss_summary.csv"] = summarise(loss_pairs, ["dataset", "architecture", "comparison"],
        {"deployment_loss_change_pp": (1, "percentage_points")})

    sampler_groups: dict[tuple, list[dict]] = defaultdict(list)
    variance_keys = ["dataset", "architecture", "training_rule", "sampler_schedule", "seed"]
    for row in read_table("sampling_variance.csv"):
        sampler_groups[tuple(row[k] for k in variance_keys)].append(row)
    variance = []
    for key, records in sampler_groups.items():
        expected = int(records[0]["sampler_count"])
        if len(records) != expected or len({r["sampler"] for r in records}) != expected:
            raise ValueError(f"Incomplete or duplicate sampler sites for {key}")
        row = {**dict(zip(variance_keys, key)), "sampler_count": expected}
        for metric in ["mean_probability", "mean_bernoulli_variance", "saturated_fraction"]:
            row[metric] = mean(float(r[metric]) for r in records)
        row["deployment_gap_pp"] = (float(records[0]["single_draw_accuracy"]) - float(records[0]["mean_field_accuracy"])) * 100 if records[0]["mean_field_accuracy"] else ""
        variance.append(row)
    products["sampling_variance_by_seed.csv"] = variance
    products["sampling_variance_summary.csv"] = summarise(variance, variance_keys[:-1],
        {"mean_probability": (1, "fraction"), "mean_bernoulli_variance": (1, "variance"), "saturated_fraction": (100, "percent"), "deployment_gap_pp": (1, "percentage_points")})

    certificate = read_table("certification_audit.csv")
    primary = [r for r in certificate if r["primary_comparison"] == "true"]
    products["decision_certification_summary.csv"] = summarise(primary, ["dataset", "architecture", "training_rule", "alpha"],
        {**{k: (100, "percent") for k in ["mean_field_accuracy", "single_draw_accuracy", "cert_acc", "cert_abstain", "cert_wrong_among_certified", "seq_acc", "seq_abstain"]},
         "seq_mean_draws": (1, "draws")})
    # The source producer summed audit cells over all alpha values and then
    # repeated those totals on each alpha-specific result row. Count each unit once.
    audit_keys = ["dataset", "architecture", "training_rule", "audit_group", "seed",
                  "transfer_factor", "regularization_coefficient", "rate_target"]
    units = {}
    audit_columns = ["reference_comparisons", "reference_disagreements", "maximum_cell_disagreement_rate", "audit_cells"]
    for row in certificate:
        key = tuple(row[k] for k in audit_keys)
        if key in units and any(row[c] != units[key][c] for c in audit_columns):
            raise ValueError(f"Repeated source audit totals disagree for {key}")
        units[key] = row
    audit = [dict(alpha_levels=";".join(sorted({r["alpha"] for r in certificate})),
        audit_units=len(units), reference_comparisons=sum(int(r["reference_comparisons"]) for r in units.values()),
        reference_disagreements=sum(int(r["reference_disagreements"]) for r in units.values()),
        maximum_cell_disagreement_rate=max(float(r["maximum_cell_disagreement_rate"]) for r in units.values()))]
    products["certification_implementation_audit.csv"] = audit

    transfer = read_table("time_step_transfer.csv")
    for row in transfer:
        row["transfer_change_pp"] = 100 * (float(row["transferred_accuracy"]) - float(row["native_accuracy"]))
    products["time_step_transfer_summary.csv"] = summarise(transfer,
        ["dataset", "binning", "architecture", "transfer_factor", "inference", "transfer_rule"],
        {**{k: (100, "percent") for k in ["native_accuracy", "transferred_accuracy", "naive_accuracy", "calibrated_accuracy"]}, "transfer_change_pp": (1, "percentage_points")})
    products["gesture_binning_summary.csv"] = summarise(read_table("gesture_binning.csv"),
        ["dataset", "binning", "architecture", "transfer_factor", "transfer_rule"],
        {k: (100, "percent") for k in ["headline_accuracy", "native_accuracy", "transferred_accuracy", "calibrated_accuracy", "naive_accuracy"]})
    products["local_learning_summary.csv"] = summarise(read_table("local_learning.csv"), ["dataset", "rule"],
        {**{k: (100, "percent") for k in ["last_graded", "last_sampled_n1", "last_sampled_n5", "bestval_graded", "bestval_sampled_n1", "patience20_graded"]},
         "last_epoch": (1, "epochs"), "best_epoch": (1, "epochs"), "patience20_stop": (1, "epochs")})
    products["sparsity_summary.csv"] = summarise(read_table("sparsity.csv"), ["dataset", "architecture", "regularization_coefficient", "rate_target"],
        {**{k: (100, "percent") for k in ["rate_layer1", "rate_layer2", "rate_mean", "acc_meanfield", "acc_one_draw", "acc_threshold", "flip_one_draw_vs_MP", "cert_seq_acc", "cert_seq_abstain", "acc"]},
         "cert_seq_mean_draws": (1, "draws"), **{k: (1, "million_operations") for k in ["input_AC_M", "input_denseMAC_M", "spike_AC_one_pass_M", "state_MAC_one_pass_M", "threshold_ops_one_pass_M"]}})
    memory = read_table("local_learning_memory_reported.csv")
    products["local_learning_memory_ratios.csv"] = [dict(time_steps=int(r["T"]),
        bptt_to_clear_memory_ratio=float(r["bptt_mb"]) / float(r["clear_mb"]),
        clear_to_inference_memory_ratio=float(r["clear_mb"]) / float(r["inference_mb"]), evidence_level="reported_summary") for r in memory]
    runtime = read_table("runtime_reported.csv")
    timing = []
    for row in runtime:
        if not row["method"].startswith("CRISP"):
            continue
        matches = [r for r in runtime if r["device"] == row["device"] and r["T"] == row["T"] and r["method"].startswith("LIF")]
        for baseline in matches:
            timing.append(dict(device=row["device"], time_steps=int(row["T"]), method=row["method"], baseline=baseline["method"],
                baseline_time_over_method_time=float(baseline["step_ms"]) / float(row["step_ms"]), evidence_level="reported_summary"))
    products["runtime_ratios.csv"] = timing

    for name, records in products.items():
        _write_csv(output / name, records)
    metadata = {
        "result_kind": "reference_summary",
        "fresh_training_or_inference": False,
        "fresh_hardware_measurement": False,
        "input_manifest": "data/manifests/reference_files.json",
        "input_files": [{"path": r["path"], "sha256": r["sha256"], "evidence_level": r["evidence_level"]} for r in manifest["files"]],
        "outputs": [{"path": name, "rows": len(rows)} for name, rows in products.items()],
        "statistics": "Arithmetic means and sample SD (ddof=1) across seeds. Paired differences use matched seed identities. No significance or equivalence claim is inferred.",
        "certificate_scope": "Audit comparisons reuse cached draws across alpha levels and cells. Source aggregate counts cover both alpha levels and are deduplicated across their repeated rows. Counts are not independent draws. The guarantee concerns the limiting sampled decision, not true-label correctness or adversarial robustness.",
        "reported_summary_scope": "Memory, runtime, epoch speed and moment-diagnostic records are archived reported summaries; per-run traces are not supplied by those tables.",
        "hardware_scope": "No manuscript-transcribed full-test power values are used as raw measurements by this analysis.",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
