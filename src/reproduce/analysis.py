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
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Sequence

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifests" / "reference_files.json"
ACCELERATOR_MANIFEST_PATH = MANIFEST_PATH.with_name("accelerator_provenance.json")


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


def accelerator_summary(
    configurations: Sequence[Mapping], repetitions: Sequence[Mapping], provenance: Mapping,
) -> list[dict]:
    """Recompute synchronized timing summaries, retaining unsuccessful configurations.

    Timings are repeated measurements of one device/configuration, not training
    seeds. Failed or unreported configurations cannot contribute to ratios.
    Every non-reference implementation must have a passing archived correctness
    gate before its successful measurements are accepted.
    """
    keys = ("device", "method", "sequence_length", "batch_size")
    gates = {(r["device"], r["method"]): r["checks"].get("pass") is True
             for r in provenance["gate_checks"]}
    references = {(r["device"], r["method"]) for r in provenance["reference_implementations"]}
    samples: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in repetitions:
        key = tuple(row[k] for k in keys)
        repeat = int(row["repetition"])
        value = float(row["step_ms"])
        if repeat in samples[key]:
            raise ValueError(f"Duplicate timing repetition for {key}: {repeat}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid timing repetition for {key}")
        samples[key][repeat] = value
    result, seen = [], set()
    for row in configurations:
        key = tuple(row[k] for k in keys)
        if key in seen:
            raise ValueError(f"Duplicate accelerator configuration: {key}")
        seen.add(key)
        values = samples.get(key, {})
        status = row["status"]
        if status not in {"ok", "oom", "not_reported"}:
            raise ValueError(f"Unknown accelerator status for {key}: {status}")
        record = dict(row)
        if status != "ok":
            if values or any(row.get(k) not in (None, "") for k in (
                "median_step_ms", "minimum_step_ms", "maximum_step_ms", "timed_repetitions")):
                raise ValueError(f"Unsuccessful configuration contains timings: {key}")
        else:
            implementation = (row["device"], row["method"])
            if implementation not in references and not gates.get(implementation, False):
                raise ValueError(f"Missing or failed correctness gate: {implementation}")
            count = int(row["timed_repetitions"])
            if count <= 0 or set(values) != set(range(1, count + 1)):
                raise ValueError(f"Missing timing repetitions for {key}")
            reductions = {"median_step_ms": median(values.values()),
                          "minimum_step_ms": min(values.values()),
                          "maximum_step_ms": max(values.values())}
            for name, value in reductions.items():
                if not math.isclose(value, float(row[name]), rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"Archived {name} disagrees with repetitions for {key}")
                record[name] = value
            if row.get("peak_memory_MiB") not in (None, ""):
                increment = float(row["peak_memory_MiB"]) - float(row["baseline_memory_MiB"])
                if not math.isclose(increment, float(row["above_baseline_memory_MiB"]), abs_tol=1e-8):
                    raise ValueError(f"Memory baseline mismatch for {key}")
        record["evidence_level"] = "timing_repetitions" if status == "ok" else "configuration_status"
        result.append(record)
    if set(samples) - seen:
        raise ValueError("Timing repetitions have no matching accelerator configuration")
    return result


def coarse_training_summary(rows: Sequence[Mapping]) -> list[dict]:
    """Reduce seed-level epoch durations; speedups are ratios of seed means.

    Coarse models must use fine-bin augmentation followed by pooling. Epoch
    duration is the recorded mean within each run; seeds receive equal weight
    irrespective of their early-stopping epoch counts.
    """
    groups: dict[tuple, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        factor = int(row["coarsening_factor"])
        key = (row["dataset"], factor)
        if factor < 1 or row["augmentation_domain"] != ("fine" if factor == 1 else "fine_then_pool"):
            raise ValueError(f"Unexpected augmentation protocol for {key}")
        seed = row["seed"]
        if seed in groups[key]:
            raise ValueError(f"Duplicate training seed for {key}: {seed}")
        duration = float(row["mean_epoch_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid epoch duration for {key}: {seed}")
        groups[key][seed] = row
    result = []
    for (dataset, factor), records in groups.items():
        baseline = groups.get((dataset, 1), {})
        if len(records) < 2 or set(records) != set(baseline):
            raise ValueError(f"Unpaired training seeds for {(dataset, factor)}")
        for seed, row in records.items():
            if int(row["temporal_bins"]) * factor != int(baseline[seed]["temporal_bins"]):
                raise ValueError(f"Inconsistent temporal bins for {(dataset, factor, seed)}")
            ratio = float(baseline[seed]["mean_epoch_s"]) / float(row["mean_epoch_s"])
            if not math.isclose(ratio, float(row["paired_epoch_speedup"]), rel_tol=1e-12):
                raise ValueError(f"Paired epoch speedup mismatch for {(dataset, factor, seed)}")
        durations = [float(row["mean_epoch_s"]) for row in records.values()]
        result.append(dict(dataset=dataset, coarsening_factor=factor,
            mean_epoch_s=mean(durations), sample_sd_epoch_s=stdev(durations), training_seeds=len(records),
            speedup_ratio_of_mean_epoch_times=mean(float(r["mean_epoch_s"]) for r in baseline.values()) / mean(durations)))
    return result


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
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "data":
            raise ValueError(f"Invalid reference path: {entry['path']}")
        path = REFERENCE_DIR.parents[1] / relative
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
    accelerator_provenance = json.loads(ACCELERATOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    runtime = accelerator_summary(read_table("accelerator_benchmarks.csv"),
        read_table("accelerator_timing_repetitions.csv"), accelerator_provenance)
    products["accelerator_benchmark_summary.csv"] = runtime
    timing = []
    for row in runtime:
        if row["status"] != "ok" or not row["method"].startswith("CRISP"):
            continue
        matches = [r for r in runtime if r["status"] == "ok" and r["device"] == row["device"]
                   and r["sequence_length"] == row["sequence_length"] and r["batch_size"] == row["batch_size"]
                   and r["method"].startswith("LIF")]
        for baseline in matches:
            timing.append(dict(device=row["device"], time_steps=int(row["sequence_length"]), method=row["method"], baseline=baseline["method"],
                baseline_time_over_method_time=float(baseline["median_step_ms"]) / float(row["median_step_ms"]), evidence_level="timing_repetitions"))
    products["runtime_ratios.csv"] = timing
    coarse = coarse_training_summary(read_table("coarse_training_timing.csv"))
    archived_rows = read_table("coarse_training_timing_summary.csv")
    archived_coarse = {(r["dataset"], int(r["coarsening_factor"])): r for r in archived_rows}
    if len(archived_coarse) != len(coarse) or len(archived_rows) != len(coarse):
        raise ValueError("Coarse-training summary has missing or duplicate conditions")
    for row in coarse:
        stored = archived_coarse.get((row["dataset"], row["coarsening_factor"]), {})
        for metric in ("mean_epoch_s", "sample_sd_epoch_s", "training_seeds", "speedup_ratio_of_mean_epoch_times"):
            if metric not in stored or not math.isclose(float(row[metric]), float(stored[metric]), rel_tol=1e-12):
                raise ValueError(f"Coarse-training summary mismatch: {row['dataset']}, {row['coarsening_factor']}, {metric}")
    products["coarse_training_timing_summary.csv"] = coarse

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
        "reported_summary_scope": "Local-learning memory/timing and moment-diagnostic tables remain archived reported summaries; their individual measurement traces are not supplied.",
        "accelerator_scope": "All successful accelerator medians, minima and maxima are recomputed from synchronized timed repetitions after the archived correctness gates. Repetitions measure one configuration and are not independent training seeds. Warmup and first-call initialization are excluded. Failed and unreported configurations retain blank values and do not enter runtime ratios. Memory is in MiB (2**20 bytes).",
        "coarse_training_scope": "Mean epoch durations and sample SD use equal weight for each training seed. Speedup is fine-resolution mean epoch duration divided by coarse-resolution mean epoch duration, using augmentation in fine bins followed by pooling. It is neither a mean of paired ratios nor a total training-time reduction under early stopping.",
        "hardware_scope": "No manuscript-transcribed full-test power values are used as raw measurements by this analysis.",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
