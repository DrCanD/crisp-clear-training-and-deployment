#!/usr/bin/env python3
"""Rebuild the original full-test integer input package from the prepared SHD cache.

The event stream, offsets and labels must match their archived SHA-256 values.
No model training, quantization selection or test-dependent adjustment is done.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_integer import repository_path, check_hashes
from integer_reference import dense_to_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/shd.npz')
    parser.add_argument('--output', default='outputs/hardware/full_test')
    args = parser.parse_args()
    fixture = repository_path('data/reference/hardware/fixture')
    recorded = repository_path('data/reference/hardware/full_test')
    check_hashes(fixture)
    check_hashes(recorded)
    identity = json.loads((recorded / 'input_identity.json').read_text())
    with np.load(repository_path(args.dataset), allow_pickle=False) as cache:
        x = cache['Xte']
        labels = cache['yte']
    if x.shape != (2264, 100, 700):
        raise ValueError('Expected the original SHD test shape (2264, 100, 700)')
    if x.size and (x.min() < 0 or x.max() > 255 or not np.array_equal(x, np.rint(x))):
        raise ValueError('Input cache must contain integer spike counts in [0,255]')
    pieces, offsets = [], [0]
    for i, item in enumerate(x):
        events = dense_to_events(item)
        pieces.append(events)
        offsets.append(offsets[-1] + len(events))
        if (i + 1) % 256 == 0 or i + 1 == len(x):
            print(f'[events] {i + 1}/{len(x)} inputs packed', flush=True)
    arrays = {'test_events.bin': np.concatenate(pieces).astype('<u4'),
              'test_offsets.bin': np.asarray(offsets, dtype='<u4'),
              'test_labels.bin': np.asarray(labels, dtype='<i2')}
    for name, value in arrays.items():
        if hashlib.sha256(value.tobytes()).hexdigest() != identity['sha256'][name]:
            raise ValueError(f'{name} differs from the original test-input identity; no package written')
    destination = repository_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    for source in [fixture, recorded]:
        for item in source.glob('*.bin'):
            shutil.copyfile(item, destination / item.name)
    for name, value in arrays.items():
        value.tofile(destination / name)
    # The fixture prefix is not valid for the full test set.
    (destination / 'expected_layer1_logits.bin').unlink(missing_ok=True)
    params = json.loads((fixture / 'parameters.json').read_text())
    full = json.loads((recorded / 'parameters.json').read_text())
    params['files'].pop('expected_layer1_logits.bin', None)
    params['files'].update(full['files'])
    params['files']['test_events.bin'] = {'dtype': 'uint32', 'shape': [len(arrays['test_events.bin'])]}
    for key in ['n_test', 'n_events_total', 'n_events_max']:
        params[key] = full[key]
    params.pop('fixture_of', None)
    params['scope'] = 'Original 2264-input package; event bytes rebuilt from the SHA-256-matched SHD cache.'
    (destination / 'parameters.json').write_text(json.dumps(params, indent=2) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file()}
    (destination / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'[complete] {args.output}: all 2264 test inputs match archived hashes', flush=True)


if __name__ == '__main__':
    main()
