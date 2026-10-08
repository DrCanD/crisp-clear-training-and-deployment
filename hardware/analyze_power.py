#!/usr/bin/env python3
"""Recompute incremental SOM energy from the original timed power blocks.

Each evaluated input has equal weight, including inputs receiving an abstention.
Processing time is inverse throughput during the power-sampling window.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import ROOT, repository_path

VARIANTS = {1: 'sampled', 2: 'mf'}
MODES = {'D0': 0, 'PREFIX': 1, 'N1': 2, 'SEQ': 3, 'FIX': 4, 'MF': 5}


def load_measurement(path, expected_inputs=2264, expected_variant=None):
    data = json.loads(path.read_text())
    if not data.get('complete') or not data.get('repeats'):
        raise ValueError('Measurement is incomplete')
    variant = VARIANTS.get(data.get('variant'))
    if variant is None or data.get('settings', {}).get('variant') != variant:
        raise ValueError('Measurement variant and settings disagree')
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f'Expected the {expected_variant} measurement variant')
    expected_modes = {'D0', 'PREFIX', 'N1', 'SEQ', 'FIX'} if variant == 'sampled' else {'D0', 'PREFIX', 'MF'}
    seen = []
    for repeat in data['repeats']:
        if not repeat.get('complete'):
            raise ValueError('An input batch is incomplete')
        seen.extend(repeat['inputs'])
        if (set(repeat['order']) != expected_modes or set(repeat['runs']) != expected_modes
                or len(repeat['order']) != len(expected_modes)):
            raise ValueError('Mode order and runs must cover each required variant mode once')
        if len(repeat['idles']) != len(repeat['order']) + 1:
            raise ValueError('Idle brackets do not cover the mode order')
        for index, mode in enumerate(repeat['order']):
            bracket = (repeat['idles'][index]['mean_mW'] + repeat['idles'][index + 1]['mean_mW']) / 2
            if not np.isclose(bracket, repeat['runs'][mode]['idle_bracket_mW'], rtol=0, atol=1e-8):
                raise ValueError('Stored idle bracket differs from the measured neighboring idle blocks')
        for row in list(repeat['runs'].values()) + repeat['idles']:
            samples = np.asarray([sample['power_mW'] for sample in row['samples']], dtype=float)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError('Power blocks require finite raw sensor samples')
            if not np.isfinite(row['duration_s']) or row['duration_s'] <= 0:
                raise ValueError('Power-sampling duration must be positive and finite')
            observed = np.mean(samples)
            if not np.isclose(observed, row['mean_mW'], rtol=0, atol=1e-8):
                raise ValueError('Stored mean power differs from the raw sensor samples')
        for mode, row in repeat['runs'].items():
            if row['mode'] != MODES[mode]:
                raise ValueError('Stored hardware mode differs from its label')
            items = row['items']
            if items <= 0 or items != row['passes'] * len(repeat['inputs']):
                raise ValueError('Whole-run item counter differs from passes times batch size')
            rate = row['decisions_per_s']
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError('Non-positive or non-finite item-counter throughput')
            if not np.isclose(row['s_per_decision'], 1 / rate, rtol=0, atol=1e-12):
                raise ValueError('Processing time differs from inverse window throughput')
            # The saved rate uses the counter delta during the power window.
            # Whole-run items / elapsed_s also includes warm-up and completion.
            window_items = rate * row['duration_s']
            if not np.isclose(window_items, round(window_items), rtol=0, atol=1e-6) or window_items > items + 1e-6:
                raise ValueError('Power-window throughput is inconsistent with an integer item-counter delta')
            for stored, observed in ((row['draws_per_item'], row['draws_total'] / items),
                                     (row['accuracy'], row['correct'] / items)):
                if not np.isclose(stored, observed, rtol=0, atol=1e-10):
                    raise ValueError('Per-input statistic differs from whole-run counters')
    if len(seen) != expected_inputs or sorted(seen) != list(range(expected_inputs)):
        raise ValueError(f'Expected all {expected_inputs} inputs in disjoint batches')
    return data


def validate_comparison(sampled, mean_field):
    """Require matching input packages and clocks before forming an energy ratio."""
    if sampled['variant'] != 1 or mean_field['variant'] != 2:
        raise ValueError('Comparison requires sampled and mean-field measurement variants')
    if not sampled.get('package') or sampled['package'] != mean_field.get('package'):
        raise ValueError('Sampled and mean-field input packages differ')
    clock = sampled.get('environment', {}).get('clock_hz')
    if not clock or clock != mean_field.get('environment', {}).get('clock_hz'):
        raise ValueError('Sampled and mean-field clocks differ')
    sampled_inputs = sorted(i for batch in sampled['repeats'] for i in batch['inputs'])
    mf_inputs = sorted(i for batch in mean_field['repeats'] for i in batch['inputs'])
    if sampled_inputs != mf_inputs:
        raise ValueError('Sampled and mean-field evaluated inputs differ')


def measurement_source(path, data):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'package': data['package'], 'variant': VARIANTS[data['variant']],
            'bitstream_sha256': data['environment']['bitstream_sha256']}


def batch_rows(data):
    rows = []
    for batch in data['repeats']:
        control = batch['runs']['D0']
        control_energy = 1000 * (control['mean_mW'] - control['idle_bracket_mW']) / control['decisions_per_s']
        for mode, values in batch['runs'].items():
            if values['decisions_per_s'] <= 0:
                raise ValueError('Non-positive item-counter throughput')
            energy = 1000 * (values['mean_mW'] - values['idle_bracket_mW']) / values['decisions_per_s']
            rows.append({'variant': VARIANTS[data['variant']], 'batch': batch['repeat'], 'mode': mode,
                         'input_count': len(batch['inputs']),
                         'events_per_input': values['events_per_item'], 'draws_per_input': values['draws_per_item'],
                         'incremental_energy_uJ': energy - control_energy,
                         'idle_adjusted_energy_uJ': energy,
                         'processing_time_ms': 1000 / values['decisions_per_s'],
                         'accuracy': values['accuracy']})
    return rows


def summarize_mode(rows):
    y = np.asarray([r['incremental_energy_uJ'] for r in rows])
    weights = np.asarray([r['input_count'] for r in rows], dtype=float)
    weights /= weights.sum()
    features = [np.ones(len(rows)), np.asarray([r['events_per_input'] for r in rows])]
    draws = np.asarray([r['draws_per_input'] for r in rows])
    if np.ptp(draws) > 1e-12:
        features.append(draws)
    x = np.column_stack(features)
    # Centre and scale nonconstant regressors to avoid condition-number effects.
    for i in range(1, x.shape[1]):
        scale = np.std(x[:, i])
        if scale <= 0:
            raise ValueError('Constant regression feature')
        x[:, i] = (x[:, i] - np.mean(x[:, i])) / scale
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    degrees = len(y) - rank
    if degrees <= 0 or rank != x.shape[1]:
        raise ValueError('Insufficient independent measurement batches')
    residuals = y - x @ coefficients
    residual_variance = float(residuals @ residuals / degrees)
    mean_design = weights @ x
    standard_error = float(np.sqrt(residual_variance * mean_design @ np.linalg.inv(x.T @ x) @ mean_design))
    mean = float(weights @ y)
    half = float(student_t.ppf(0.975, degrees) * standard_error)
    return {'batches': len(rows), 'inputs': int(sum(r['input_count'] for r in rows)),
            'energy_uJ_per_input': mean, 'energy_standard_error_uJ': standard_error,
            'energy_ci95_uJ': [mean - half, mean + half], 'residual_degrees_of_freedom': int(degrees),
            'processing_time_ms_per_input': float(weights @ np.asarray([r['processing_time_ms'] for r in rows])),
            'mean_draws': float(weights @ draws),
            'accuracy': float(weights @ np.asarray([r['accuracy'] for r in rows]))}


def fit_draw_cost(rows, field):
    rows = [row for row in rows if row['mode'] != 'D0']
    x = np.column_stack([np.ones(len(rows)), [r['draws_per_input'] for r in rows]])
    y = np.asarray([r[field] for r in rows])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    r_squared = 1 - float(residuals @ residuals / ((y - y.mean()) @ (y - y.mean())))
    return {'observations': len(rows), 'intercept': float(coefficients[0]),
            'per_draw': float(coefficients[1]), 'r_squared': r_squared,
            'method': 'Unweighted ordinary least squares over sampled-mode-by-batch observations, excluding D0.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampled', default='data/reference/hardware/power_sampled.json')
    parser.add_argument('--mean-field', default='data/reference/hardware/power_mean_field.json',
                        help='Full-test mean-field raw measurement; pass an empty string for sampled-only analysis')
    parser.add_argument('--output', default='outputs/hardware/power_analysis')
    args = parser.parse_args()
    sampled_path = repository_path(args.sampled)
    sampled = load_measurement(sampled_path, expected_variant='sampled')
    rows = batch_rows(sampled)
    modes = {mode: summarize_mode([r for r in rows if r['mode'] == mode])
             for mode in ('PREFIX', 'N1', 'SEQ', 'FIX')}
    result = {'scope': 'Incremental SOM input energy after bracketed-idle and D0 subtraction; all evaluated inputs, including abstentions.',
              'standard_error_method': 'OLS batch residual variance with events per input and, when varying, draws per input; covariance evaluated at the input-weighted mean design.',
              'sources': {'sampled': measurement_source(sampled_path, sampled)},
              'modes': modes, 'energy_draw_fit': fit_draw_cost(rows, 'incremental_energy_uJ'),
              'time_draw_fit': fit_draw_cost(rows, 'processing_time_ms')}
    if args.mean_field:
        mf_path = repository_path(args.mean_field)
        mean_field = load_measurement(mf_path, expected_variant='mf')
        validate_comparison(sampled, mean_field)
        result['sources']['mean_field'] = measurement_source(mf_path, mean_field)
        mf_rows = batch_rows(mean_field)
        result['modes']['MF'] = summarize_mode([r for r in mf_rows if r['mode'] == 'MF'])
        rows += mf_rows
        mf_energy = result['modes']['MF']['energy_uJ_per_input']
        if not np.isfinite(mf_energy) or mf_energy <= 0:
            raise ValueError('Mean-field incremental energy must be positive to form the ratio')
        result['sequential_over_mean_field'] = modes['SEQ']['energy_uJ_per_input'] / mf_energy
    result['sequential_saving_vs_fixed_fraction'] = 1 - modes['SEQ']['energy_uJ_per_input'] / modes['FIX']['energy_uJ_per_input']
    out = repository_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'batches.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
