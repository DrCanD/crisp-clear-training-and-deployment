"""Check that file-backed audio binning preserves the registered arithmetic."""
import h5py
import numpy as np
import pytest
import torch

from reproduce.data_preparation import bin_h5
from reproduce.datasets import data_fingerprint, load_data


def test_file_backed_binning_matches_in_memory(tmp_path):
    source = tmp_path / "events.h5"
    times = [np.array([0.0, 0.125, 0.5, 1.0]), np.array([]),
             np.full(260, 0.5)]
    units = [np.array([0, 1, 2, 3]), np.array([], dtype=np.int64),
             np.full(260, 2)]
    with h5py.File(source, "w") as h5:
        group = h5.create_group("spikes")
        ts = group.create_dataset("times", (3,), dtype=h5py.vlen_dtype(np.float64))
        us = group.create_dataset("units", (3,), dtype=h5py.vlen_dtype(np.int64))
        for i in range(3):
            ts[i], us[i] = times[i], units[i]
        h5["labels"] = np.array([0, 1, 2])
    expected, labels, maximum = bin_h5(source, 8, 4)
    mapped, mapped_labels, mapped_maximum = bin_h5(
        source, 8, 4, max_time=maximum, output_path=tmp_path / "binned.npy")
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, expected)
    np.testing.assert_array_equal(mapped_labels, labels)
    assert mapped_maximum == maximum == 1.0
    assert mapped[0, -1, 3] == 1
    assert mapped[1].sum() == 0
    assert mapped[2, 4, 2] == 255
    del mapped


@pytest.mark.parametrize('modern', [False, True])
def test_audio_cache_loader_preserves_legacy_and_modern_arrays(tmp_path, monkeypatch, modern):
    from reproduce import paths
    monkeypatch.setattr(paths, 'ROOT', tmp_path)
    arrays = tuple(
        ((torch.arange(n * 2 * 700).reshape(n, 2, 700) % 13).to(torch.uint8),
         torch.arange(n) % 35)
        for n in (35, 14, 17)
    )
    payload = {key + suffix: value
               for suffix, pair in zip(('tr', 'va', 'te'), arrays)
               for key, value in zip(('X', 'y'), pair)}
    cache = tmp_path / 'ssc.pt'
    torch.save(payload, cache, _use_new_zipfile_serialization=modern)
    fingerprint = data_fingerprint(arrays)
    loaded = load_data(cache, 'ssc', fingerprint)
    for actual_pair, expected_pair in zip(loaded, arrays):
        for actual, expected in zip(actual_pair, expected_pair):
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)
