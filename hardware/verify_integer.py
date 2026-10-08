#!/usr/bin/env python3
"""Check integer inference against the archived fixture and board decisions.

Run from any working directory with this script's relative path. All arguments
that name files or directories are interpreted relative to the repository root.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import integer_reference as reference
from paths import repository_path, check_hashes


def load_model(directory):
    params = json.loads((directory / 'parameters.json').read_text())
    def read(name):
        spec = params['files'][name]
        return np.fromfile(directory / name, dtype=np.dtype(spec['dtype']).newbyteorder('<')).reshape(spec['shape']).astype(np.int64)
    model = {'fmt': params['fmt'], 'layers': []}
    for i in (1, 2):
        model['layers'].append({
            'W': read(f'layer{i}_weights.bin'), 'b': read(f'layer{i}_bias.bin'),
            'a': read(f'layer{i}_decay.bin'), 'c': read(f'layer{i}_input_coefficient.bin'),
            'h': read(f'layer{i}_offset.bin'), 'shc': params['shc'][f'l{i}'],
        })
    model['ro'] = {'W': read('readout_weights.bin'), 'b': read('readout_bias.bin'),
                   'a': read('readout_decay.bin'), 'c': read('readout_input_coefficient.bin'),
                   'shc': params['shc']['ro']}
    for key, name in [('NOISE', 'logistic_noise.bin'), ('SIG', 'sigmoid_probability.bin'),
                      ('THR_SEQ', 'sequential_threshold.bin'), ('THR_FIX', 'fixed_threshold.bin')]:
        model[key] = read(name)
    return model, params, read


def assert_equal(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise AssertionError(f'{label} differs from archived output')


def verify_fixture(directory, inputs, draws):
    checked = check_hashes(directory)
    model, params, read = load_model(directory)
    f = model['fmt']
    inputs = min(inputs, params['n_test'])
    if not (1 <= draws <= f['M']):
        raise ValueError(f'Draw count must be between 1 and {f["M"]}')
    expected_tables = reference.build_tables(f)
    for key in expected_tables:
        assert_equal(model[key], expected_tables[key], key)
    model = reference.to_dev(model, 'cpu')
    offsets, events = read('test_offsets.bin'), read('test_events.bin')
    x = np.stack([reference.events_to_dense(events[offsets[i]:offsets[i + 1]], f['T'], f['N_IN']) for i in range(inputs)])
    prefix, _, _ = reference.prefix_z1(model, x, 'cpu')
    assert_equal(prefix.numpy(), read('expected_layer1_logits.bin')[:inputs], 'Layer 1 integer logits')
    decisions, logits, _ = reference.mf_forward(model, prefix, 'cpu')
    assert_equal(decisions.numpy(), read('expected_mean_field_decisions.bin')[:inputs], 'Mean-field decisions')
    assert_equal(logits.numpy(), read('expected_mean_field_logits.bin')[:inputs], 'Mean-field integer logits')
    for stream, name in [(0, 'expected_certificate_votes.bin'), (1, 'expected_reference_votes.bin')]:
        expected = read(name)
        for start in range(0, inputs, 4):
            stop = min(start + 4, inputs)
            votes, _, _ = reference.sampled_draws(model, prefix[start:stop], np.arange(start, stop), stream, np.arange(draws), 'cpu')
            assert_equal(votes.numpy(), expected[start:stop, :draws], f'Sampled stream {stream}')
            print(f'[verify] stream {stream}: {stop}/{inputs} inputs x {draws} draws match', flush=True)
    votes = read('expected_certificate_votes.bin')
    decisions, used = reference.sequential(votes, f['C'], f['LOOKS'], model['THR_SEQ'])
    assert_equal(np.column_stack([decisions, used]), read('expected_sequential_decisions.bin'), 'Sequential decisions')
    assert_equal(reference.predict_table(votes, f['C'], model['THR_FIX']), read('expected_fixed_decisions.bin'), 'Fixed-budget decisions')
    return {'fixture_files_checked': checked, 'inference_inputs': inputs, 'draws_per_stream': draws,
            'integer_prefix_matches': True, 'integer_mean_field_logits_match': True,
            'sampled_streams_match': 2, 'certificate_inputs_checked': len(votes)}


def verify_full_test(directory):
    check_hashes(directory)
    params = json.loads((directory / 'parameters.json').read_text())
    f = params['fmt']
    def read(name):
        spec = params['files'][name]
        return np.fromfile(directory / name, dtype=np.dtype(spec['dtype']).newbyteorder('<')).reshape(spec['shape']).astype(np.int64)
    votes = read('expected_certificate_votes.bin')
    reference_votes = read('expected_reference_votes.bin')
    tables = reference.build_tables(f)
    seq, used = reference.sequential(votes, f['C'], f['LOOKS'], tables['THR_SEQ'])
    fix = reference.predict_table(votes, f['C'], tables['THR_FIX'])
    ref = reference.predict_table(reference_votes, f['C'], tables['THR_FIX'])
    assert_equal(np.column_stack([seq, used]), read('expected_sequential_decisions.bin'), 'Full-test sequential decisions')
    assert_equal(fix, read('expected_fixed_decisions.bin'), 'Full-test fixed decisions')
    assert_equal(ref, read('expected_reference_decisions.bin'), 'Full-test reference decisions')
    board_dir = directory.parent
    sampled = json.loads((board_dir / 'board_verification_sampled.json').read_text())
    mean_field = json.loads((board_dir / 'board_verification_mean_field.json').read_text())
    expected = {'seq': seq, 'used': used, 'fix': fix, 'ref': ref, 'n1': votes[:, 0]}
    for key, value in expected.items():
        assert_equal(sampled['decisions'][key], value, f'Board record {key}')
    assert_equal(mean_field['decisions']['mf'], read('expected_mean_field_decisions.bin'), 'Board mean-field decisions')
    for record in [sampled, mean_field]:
        if not (record['completed'] and len(record['bit_exact']) == len(votes) and all(record['bit_exact'])):
            raise AssertionError('Archived board bit-exact record is incomplete')
    labels = read('test_labels.bin')
    return {'inputs': len(votes), 'sequential_accuracy': float(np.mean(seq == labels)),
            'sequential_abstention': float(np.mean(seq < 0)), 'mean_draws': float(np.mean(used)),
            'fixed_accuracy': float(np.mean(fix == labels)),
            'mean_field_accuracy': float(np.mean(read('expected_mean_field_decisions.bin') == labels)),
            'accepted_disagreements_with_reference': int(np.sum((seq >= 0) & (ref >= 0) & (seq != ref))),
            'board_record_decisions_match': True,
            'scope': 'Stored vote streams and original board records checked; no new full-test inference or board run.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default='data/reference/hardware/fixture')
    parser.add_argument('--full-test', default='data/reference/hardware/full_test')
    parser.add_argument('--inputs', type=int, default=4)
    parser.add_argument('--draws', type=int, default=256)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='outputs/hardware/integer_verification.json')
    args = parser.parse_args()
    if args.inputs < 1 or args.threads < 1:
        parser.error('--inputs and --threads must be positive')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    print('[verify] checking fixture hashes and integer inference', flush=True)
    result = {'fixture': verify_fixture(repository_path(args.fixture), args.inputs, args.draws),
              'full_test_records': verify_full_test(repository_path(args.full_test))}
    out = repository_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
