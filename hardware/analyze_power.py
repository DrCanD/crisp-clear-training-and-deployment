#!/usr/bin/env python3
"""Recompute incremental SOM energy from the original timed power blocks.

Each evaluated input has equal weight, including inputs receiving an abstention.
Processing time is inverse throughput during the power-sampling window.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_integer import repository_path


def load_measurement(path, expected_inputs=2264):
    data = json.loads(path.read_text())
    if not data.get('complete') or not data.get('repeats'):
        raise ValueError('Measurement is incomplete')
    seen = []
    for repeat in data['repeats']:
        if not repeat.get('complete'):
            raise ValueError('An input batch is incomplete')
        seen.extend(repeat['inputs'])
        if len(repeat['idles']) != len(repeat['order']) + 1:
            raise ValueError('Idle brackets do not cover the mode order')
        for index, mode in enumerate(repeat['order']):
            bracket = (repeat['idles'][index]['mean_mW'] + repeat['idles'][index + 1]['mean_mW']) / 2
            if not np.isclose(bracket, repeat['runs'][mode]['idle_bracket_mW'], rtol=0, atol=1e-8):
                raise ValueError('Stored idle bracket differs from the measured neighboring idle blocks')
        for row in list(repeat['runs'].values()) + repeat['idles']:
            observed = np.mean([sample['power_mW'] for sample in row['samples']])
            if not np.isclose(observed, row['mean_mW'], rtol=0, atol=1e-8):
                raise ValueError('Stored mean power differs from the raw sensor samples')
    if len(seen) != expected_inputs or sorted(seen) != list(range(expected_inputs)):
        raise ValueError(f'Expected all {expected_inputs} inputs in disjoint batches')
    return data


def batch_rows(data):
    rows = []
    for batch in data['repeats']:
        control = batch['runs']['D0']
        control_energy = 1000 * (control['mean_mW'] - control['idle_bracket_mW']) / control['decisions_per_s']
        for mode, values in batch['runs'].items():
            if values['decisions_per_s'] <= 0:
                raise ValueError('Non-positive item-counter throughput')
            energy = 1000 * (values['mean_mW'] - values['idle_bracket_mW']) / values['decisions_per_s']
            rows.append({'batch': batch['repeat'], 'mode': mode, 'input_count': len(batch['inputs']),
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
    parser.add_argument('--mean-field', default='', help='Optional full-test mean-field raw measurement; no substitute is used when absent')
    parser.add_argument('--output', default='outputs/hardware/power_analysis')
    args = parser.parse_args()
    sampled = load_measurement(repository_path(args.sampled))
    rows = batch_rows(sampled)
    modes = {mode: summarize_mode([r for r in rows if r['mode'] == mode])
             for mode in ('PREFIX', 'N1', 'SEQ', 'FIX')}
    result = {'scope': 'Incremental SOM input energy after bracketed-idle and D0 subtraction; all evaluated inputs, including abstentions.',
              'standard_error_method': 'OLS batch residual variance with events per input and, when varying, draws per input; covariance evaluated at the input-weighted mean design.',
              'modes': modes, 'energy_draw_fit': fit_draw_cost(rows, 'incremental_energy_uJ'),
              'time_draw_fit': fit_draw_cost(rows, 'processing_time_ms')}
    if args.mean_field:
        mean_field = load_measurement(repository_path(args.mean_field))
        mf_rows = batch_rows(mean_field)
        result['modes']['MF'] = summarize_mode([r for r in mf_rows if r['mode'] == 'MF'])
        rows += mf_rows
        result['sequential_over_mean_field'] = modes['SEQ']['energy_uJ_per_input'] / result['modes']['MF']['energy_uJ_per_input']
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
