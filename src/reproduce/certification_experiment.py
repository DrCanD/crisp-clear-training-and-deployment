"""Sample trained models and reproduce fixed/sequential decision certificates.

The archived protocol uses 256 certificate draws and a separate 256-draw
reference stream. Consecutive groups of 1, 2, 4, 8 or 16 draws are classified
by the argmax of their mean logits. The two-sided PREDICT test is then applied
to these group decisions. Cached numerical arrays support CPU-only reanalysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .certification import counts_top2, predict_table, sequential, threshold_table
from .paths import resolve_path
from .training import set_all_seeds


@lru_cache(maxsize=128)
def _thresholds(alpha: str, maximum: int):
    return threshold_table(Fraction(alpha), maximum)


def _predict(votes, classes, alpha):
    return predict_table(votes, classes, _thresholds(str(alpha), votes.shape[1]))


def group_decisions(logits, group_size):
    """[draws, inputs, classes] -> [inputs, nonoverlapping groups]."""
    if logits.ndim != 3 or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("Expected 3D logits and a positive integer group size")
    groups = logits.shape[0] // group_size
    if groups == 0:
        raise ValueError("Group size exceeds the number of draws")
    grouped = logits[:groups * group_size].reshape(groups, group_size, *logits.shape[1:]).mean(1)
    if isinstance(grouped, torch.Tensor):
        return grouped.argmax(2).T.cpu().numpy().astype(np.int16)
    return grouped.argmax(2).T.astype(np.int16)


def group_budgets(group_size, maximum_draws):
    """Archived fixed budgets: powers of two groups plus the final full budget."""
    groups = maximum_draws // group_size
    result, count = [], 1
    while count <= groups:
        result.append(count * group_size)
        count *= 2
    if groups and groups * group_size not in result:
        result.append(groups * group_size)
    return result


def sequential_looks(group_size, maximum_draws, alpha):
    """Remove looks where even unanimous votes cannot pass alpha / number of looks."""
    candidates = [n // group_size for n in group_budgets(group_size, maximum_draws)]
    level = Fraction(str(alpha))
    while candidates:
        keep = [m for m in candidates if Fraction(2, 1 << m) <= level / len(candidates)]
        if keep == candidates:
            break
        candidates = keep
    return candidates


def _decision_metrics(decisions, labels, mean_field, reference, resolved):
    accepted = decisions >= 0
    checked = int(resolved.sum())
    disagreements = int((accepted & resolved & (decisions != reference)).sum())
    return {
        "correct_and_accepted_fraction": float((decisions == labels).mean()),
        "abstention_fraction": float((~accepted).mean()),
        "wrong_among_accepted_fraction": float((decisions[accepted] != labels[accepted]).mean()) if accepted.any() else None,
        "agreement_with_mean_field_among_accepted": float((decisions[accepted] == mean_field[accepted]).mean()) if accepted.any() else None,
        "resolved_reference_inputs": checked,
        "reference_disagreements": disagreements,
        "disagreement_fraction_resolved": disagreements / checked if checked else None,
        "disagreement_fraction_all_inputs": disagreements / len(labels),
    }


def analyse_decisions(arrays, classes, alphas=(0.01, 0.001), reference_alpha=0.001):
    """Reanalyse grouped decisions; no model execution is required.

    Reference errors and certificate errors are different from true-label
    errors. Reference-resolved input counts are repeated across budget cells;
    aggregate comparisons are therefore not independent trials.
    """
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    mean_field = np.asarray(arrays["mean_field_predictions"], dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or mean_field.shape != labels.shape:
        raise ValueError("Labels and mean-field predictions must be nonempty matching vectors")
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError("Labels fall outside the configured class range")
    groups = sorted(int(key.removeprefix("certificate_group_")) for key in arrays if key.startswith("certificate_group_"))
    if not groups or groups[0] != 1:
        raise ValueError("Single-draw decisions are required")
    maximum = arrays["certificate_group_1"].shape[1]
    reference_draws = arrays["reference_group_1"].shape[1]
    fixed_rows, sequential_rows, reference_rows, noise_rows = [], [], [], []
    for group in groups:
        certificate = np.asarray(arrays[f"certificate_group_{group}"])
        ref = np.asarray(arrays[f"reference_group_{group}"])
        if certificate.shape != (len(labels), maximum // group) or ref.shape != (len(labels), reference_draws // group):
            raise ValueError(f"Inconsistent group-decision dimensions for group size {group}")
        reference, _, _, _ = counts_top2(ref, classes)
        resolved = _predict(ref, classes, reference_alpha) >= 0
        reference_rows.append({"group_size": group, "reference_draws": reference_draws,
            "resolved_fraction": float(resolved.mean()), "modal_reference_accuracy": float((reference == labels).mean()),
            "resolved_reference_accuracy": float((reference[resolved] == labels[resolved]).mean()) if resolved.any() else None})
        noise_rows.append({"group_size": group, "groups_per_input": certificate.shape[1],
            "mean_group_accuracy": float((certificate == labels[:, None]).mean()),
            "flip_fraction_vs_mean_field": float((certificate != mean_field[:, None]).mean())})
        for alpha in alphas:
            for draws in group_budgets(group, maximum):
                decisions = _predict(certificate[:, :draws // group], classes, alpha)
                fixed_rows.append({"group_size": group, "alpha": float(alpha), "draws": draws,
                    **_decision_metrics(decisions, labels, mean_field, reference, resolved)})
            looks = sequential_looks(group, maximum, alpha)
            if looks:
                thresholds = _thresholds(str(Fraction(str(alpha)) / len(looks)), certificate.shape[1])
                decisions, used_groups = sequential(certificate, classes, looks, thresholds)
                used = used_groups * group
            else:
                decisions = np.full(len(labels), -1, dtype=np.int64)
                used = np.full(len(labels), (maximum // group) * group)
            accepted = decisions >= 0
            sequential_rows.append({"group_size": group, "alpha": float(alpha), "looks_groups": ";".join(map(str, looks)),
                "mean_draws_all_inputs": float(used.mean()),
                "mean_draws_accepted": float(used[accepted].mean()) if accepted.any() else None,
                **_decision_metrics(decisions, labels, mean_field, reference, resolved)})
    return {"inputs": len(labels), "certificate_draws": maximum, "reference_draws": reference_draws,
        "mean_field_accuracy": float((mean_field == labels).mean()),
        "mean_certificate_logits_accuracy": float((arrays["mean_certificate_logits"].argmax(1) == labels).mean()) if "mean_certificate_logits" in arrays else None,
        "fixed": fixed_rows, "sequential": sequential_rows, "reference": reference_rows, "sampling_noise": noise_rows,
        "implementation_audit": {"reference_comparisons": sum(r["resolved_reference_inputs"] for r in fixed_rows),
            "reference_disagreements": sum(r["reference_disagreements"] for r in fixed_rows),
            "maximum_cell_disagreement_fraction_resolved": max((r["disagreement_fraction_resolved"] or 0) for r in fixed_rows)},
        "scope": "Two-sided test under independent draws. Acceptance concerns the most probable grouped sampled decision, not true-label correctness or adversarial robustness. Reported aggregate comparisons reuse cached draws. The alpha + reference_alpha union bound is unconditional; resolved-only disagreement fractions are descriptive diagnostics, not conditional guarantees.",
        "reference_alpha": float(reference_alpha)}


@torch.no_grad()
def mean_field_logits(model, inputs, family, beta):
    if family == "crisp":
        return model(inputs, "graded", beta)[0]
    if family == "pspikessm":
        model.sampling_mode = "meanfield"
        model.sampler_slope = beta
        return model(inputs)
    raise ValueError("Certificates require CRISP or P-SpikeSSM sampled models")


@torch.no_grad()
def sample_logits(model, inputs, family, beta, count, chunk=8, rows_max=8192):
    """Draw independent sampled forwards in eval mode, preserving archived batching.

    CRISP caches its deterministic first-layer probability. The remaining
    layers are sampled for every draw. P-SpikeSSM repeats inputs along the batch
    axis. This changes no batch statistics because normalization is in eval mode.
    """
    if model.training:
        raise ValueError("Certificate draws require model.eval()")
    if min(count, chunk, rows_max) < 1:
        raise ValueError("Draw counts, chunk size and row limit must be positive")
    batch = inputs.shape[0]
    results = []
    if family == "crisp":
        features = model.stem(inputs) if hasattr(model, "stem") else inputs
        probabilities = model.layers[0](features, "graded", beta)[0].clamp(1e-6, 1 - 1e-6)
        repeat_count = chunk
        for first in range(0, count, repeat_count):
            repeats = min(repeat_count, count - first)
            spikes = torch.bernoulli(probabilities.repeat(repeats, 1, 1))
            for layer in model.layers[1:]:
                spikes = layer(spikes, "sample", beta)[0]
            results.append(model.pool(model.ro(model.readout(spikes))).reshape(repeats, batch, -1))
    elif family == "pspikessm":
        model.sampling_mode = "sample"
        model.sampler_slope = beta
        repeat_count = max(1, min(count, rows_max // batch))
        for first in range(0, count, repeat_count):
            repeats = min(repeat_count, count - first)
            results.append(model(inputs.repeat(repeats, 1, 1)).reshape(repeats, batch, -1))
    else:
        raise ValueError("Certificates require CRISP or P-SpikeSSM")
    return torch.cat(results)


def sampling_checks(model, inputs, family, beta, chunk=8, rows_max=8192):
    """Check batch isolation, repeatability and cached CRISP forward agreement."""
    with torch.no_grad():
        direct = mean_field_logits(model, inputs, family, beta)
        repeated = mean_field_logits(model, inputs.repeat((2,) + (1,) * (inputs.ndim - 1)), family, beta).reshape(2, len(inputs), -1)
        relative = float((repeated - direct[None]).abs().max() / direct.abs().max().clamp_min(1e-12))
        if relative > 1e-4:
            raise RuntimeError("Repeated mean-field batches interact or disagree")
        set_all_seeds(4242)
        first = sample_logits(model, inputs, family, beta, 8, chunk, rows_max)
        set_all_seeds(4242)
        second = sample_logits(model, inputs, family, beta, 8, chunk, rows_max)
        if not torch.equal(first, second):
            raise RuntimeError("Fixed-seed sampled forwards are not reproducible")
        checks = {"mean_field_repeated_batch_relative_difference": relative, "fixed_seed_bitwise_repeatability": True}
        if family == "crisp":
            set_all_seeds(12345)
            direct = model(inputs, "sample", beta)[0]
            set_all_seeds(12345)
            cached = sample_logits(model, inputs, family, beta, 1, 1, rows_max)[0]
            if not torch.equal(direct, cached):
                raise RuntimeError("Cached first-layer sampling disagrees with the full forward")
            checks["cached_first_layer_matches_full_forward"] = True
    return checks


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_arrays(path, arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    _save_json(path.with_suffix(".json"), {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def _load_arrays(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Stored certificate batch checksum mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _write_tables(directory, result):
    for name in ("fixed", "sequential", "reference", "sampling_noise"):
        rows = result[name]
        with (directory / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _save_json(directory / "results.json", result)


def run(config):
    """Evaluate configured checkpoints, saving resumable grouped-decision batches."""
    from .datasets import load_data, data_fingerprint
    from .experiments import _load_or_train, _validate_job
    dataset = config["dataset"]["name"]
    if dataset == "dvs_gesture":
        from .vision import load_dvs, load_model
        data = load_dvs(config["dataset"]["path"], config["dataset"].get("fingerprint"))
    else:
        data = load_data(config["dataset"]["path"], dataset=dataset, expected_fingerprint=config["dataset"].get("fingerprint"))
    classes = {"shd": 20, "ssc": 35, "dvs_gesture": 11}[dataset]
    options = dict(certificate_draws=256, reference_draws=256, group_sizes=[1, 2, 4, 8, 16],
                   alphas=[0.01, 0.001], reference_alpha=0.001, batch_size=64, draw_chunk=8,
                   maximum_forward_rows=8192, draw_seed=20260930, tf32=False)
    options.update(config.get("certification", {}))
    for key in ("certificate_draws", "reference_draws", "batch_size", "draw_chunk", "maximum_forward_rows"):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    groups = options["group_sizes"]
    if (not groups or groups[0] != 1 or groups != sorted(set(groups))
            or any(not isinstance(g, int) or g < 1 or g > min(options["certificate_draws"], options["reference_draws"]) for g in groups)):
        raise ValueError("group_sizes must be sorted, unique, positive and start at 1")
    device = config.get("device", "cpu")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(options["tf32"])
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(options["tf32"])
    directory = resolve_path(config.get("output", f"outputs/certification_{dataset}"))
    directory.mkdir(parents=True, exist_ok=True)
    jobs, seeds = config["jobs"], config.get("seeds", [42, 123, 999])
    if not jobs or len({j["name"] for j in jobs}) != len(jobs) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Jobs and seeds must be nonempty and unique")
    summary = []
    for job in jobs:
        _validate_job(job)
        if not job.get("checkpoint") and not job.get("train", False):
            raise ValueError("Supply a checkpoint from the training workflow or explicitly set train: true")
        for seed in seeds:
            family = job.get("family", "crisp")
            if family not in ("crisp", "pspikessm"):
                raise ValueError("Certification applies only to sampled model families")
            print(f'Certificate evaluation: {job["name"]}, seed {seed}', flush=True)
            if dataset == "dvs_gesture":
                if family != "crisp" or not job.get("checkpoint"):
                    raise ValueError("Gesture certification needs a trained CRISP gesture checkpoint")
                model, architecture, _ = load_model(job["checkpoint"].format(seed=seed), device=device)
                info = {"source": "checkpoint", "checkpoint": job["checkpoint"].format(seed=seed)}
            else:
                model, architecture, _, info = _load_or_train(config, job, seed, data, directory)
            model.eval()
            beta = float(job.get("beta_end", getattr(architecture, "beta_end", 1.0)))
            factor = int(job.get("factor", 1))
            if factor > 1:
                if family == "pspikessm":
                    raise ValueError("P-SpikeSSM certificate transfer needs explicit model-rate support; use factor 1")
                model.set_dt(1.0 / factor)
            checkpoint = info.get("checkpoint") or f'{config["output"]}/checkpoints/{job["name"]}/seed_{seed}/best_model.pt'
            checkpoint_path = resolve_path(checkpoint)
            unit = directory / job["name"] / f"seed_{seed}"
            unit.mkdir(parents=True, exist_ok=True)
            identity = {"dataset": dataset, "data_fingerprint": data_fingerprint(data), "job": job, "seed": seed,
                        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                        "options": options, "device": str(device), "torch_version": torch.__version__}
            identity = json.loads(json.dumps(identity, allow_nan=False))
            protocol = unit / "protocol.json"
            if protocol.exists() and json.loads(protocol.read_text()) != identity:
                raise ValueError("Existing certificate outputs belong to a different protocol or checkpoint")
            _save_json(protocol, identity)
            inputs, labels = data[2]
            batches = (len(labels) + options["batch_size"] - 1) // options["batch_size"]
            batch_dir = unit / "batches"
            batch_dir.mkdir(exist_ok=True)
            first = inputs[:min(options["batch_size"], len(labels))].to(device).float()
            checks = sampling_checks(model, first, family, beta, options["draw_chunk"], options["maximum_forward_rows"])
            _save_json(unit / "sampling_checks.json", checks)
            seed_map = job.get("draw_seeds", {})
            supplied_seed = seed_map.get(seed, seed_map.get(str(seed)))
            base = int(supplied_seed) if supplied_seed is not None else (int(hashlib.sha256(f'{dataset}|{job["name"]}|{seed}'.encode()).hexdigest()[:8], 16) + int(options["draw_seed"])) % (2 ** 31 - 1)
            records = []
            for batch_index in range(batches):
                start = batch_index * options["batch_size"]
                stop = min(start + options["batch_size"], len(labels))
                path = batch_dir / f"batch_{batch_index:05d}.npz"
                if path.exists():
                    arrays = _load_arrays(path)
                    if not np.array_equal(arrays["labels"], labels[start:stop].numpy()):
                        raise ValueError("Cached certificate labels differ from the configured input order")
                    print(f'  batch {batch_index + 1}/{batches}: cached, inputs {start + 1}-{stop}', flush=True)
                else:
                    x = inputs[start:stop].to(device).float()
                    mf = mean_field_logits(model, x, family, beta)
                    set_all_seeds((base + batch_index) % (2 ** 31 - 1))
                    certificate = sample_logits(model, x, family, beta, options["certificate_draws"], options["draw_chunk"], options["maximum_forward_rows"])
                    # Continue the PRNG stream; never reseed or reuse certificate
                    # draws to create the reference decision.
                    reference = sample_logits(model, x, family, beta, options["reference_draws"], options["draw_chunk"], options["maximum_forward_rows"])
                    arrays = {"labels": labels[start:stop].numpy().astype(np.int16),
                              "mean_field_predictions": mf.argmax(1).cpu().numpy().astype(np.int16),
                              "mean_field_logits": mf.cpu().numpy(),
                              "mean_certificate_logits": certificate.mean(0).cpu().numpy(),
                              "mean_reference_logits": reference.mean(0).cpu().numpy()}
                    for group in groups:
                        arrays[f"certificate_group_{group}"] = group_decisions(certificate, group)
                        arrays[f"reference_group_{group}"] = group_decisions(reference, group)
                    _save_arrays(path, arrays)
                    print(f'  batch {batch_index + 1}/{batches}: sampled, inputs {start + 1}-{stop}; {options["certificate_draws"]}+{options["reference_draws"]} draws', flush=True)
                    del certificate, reference
                records.append(arrays)
            combined = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
            result = analyse_decisions(combined, classes, options["alphas"], options["reference_alpha"])
            result.update({"job": job["name"], "seed": seed, "dataset": dataset, "result_kind": "checkpoint_evaluation",
                           "checkpoint_sha256": identity["checkpoint_sha256"],
                           "stream_identity": "Explicit numerical draw seed preserved from the archived protocol." if supplied_seed is not None else "Descriptive job, dataset and seed determine the PRNG stream. Statistical reproduction is intended; original private run-label hashes are not retained."})
            _write_tables(unit, result)
            summary.append({"job": job["name"], "seed": seed, "result": f'{job["name"]}/seed_{seed}/results.json',
                            "inputs": result["inputs"], "implementation_audit": result["implementation_audit"]})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output = {"experiment": "certification", "dataset": dataset, "result_kind": "checkpoint_evaluation", "units": summary}
    _save_json(directory / "results.json", output)
    return output
