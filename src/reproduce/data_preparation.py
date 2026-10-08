"""Download official audio data and prepare the registered dense representation.

SSC binning is taken from the archived preprocessing code. SHD uses the same
count binning with its recorded maximum time, then requires the archived
array fingerprint. A mismatch aborts without publishing a cache.
"""
import gzip
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
import numpy as np
import torch
from .paths import resolve_path
from .datasets import _strat, _store, _labels, data_fingerprint

def bin_h5(h5_path, t_bins, n_channels, max_time=None):
    """
    Read zenkelab h5 (spikes/times, spikes/units, labels) and bin to
    dense [N, T, C] uint8 array. Event-conserving binning.

    If max_time is None, auto-detect from data.
    Returns X [N, T, C] uint8, y [N] int64, detected max_time.
    """
    import h5py
    h5_path = Path(h5_path)
    print(f'  [BIN] {h5_path.name} -> T={t_bins}, C={n_channels} ...', flush=True)
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        times = f['spikes']['times']
        units = f['spikes']['units']
        labels = f['labels'][:]
        N = len(labels)
        if max_time is None:
            global_max = 0.0
            for i in range(N):
                t = times[i]
                if len(t) > 0:
                    m = float(t.max()) if hasattr(t, 'max') else float(np.max(t))
                    if m > global_max:
                        global_max = m
            max_time = global_max
            print(f'    auto max_time = {max_time:.6f}s', flush=True)
        X = np.zeros((N, t_bins, n_channels), dtype=np.uint8)
        empty = 0
        for i in range(N):
            ts = np.asarray(times[i], dtype=np.float64)
            us = np.asarray(units[i], dtype=np.int64)
            if len(ts) == 0:
                empty += 1
                continue
            tb = np.clip((ts / max_time * t_bins).astype(np.int64), 0, t_bins - 1)
            us = np.clip(us, 0, n_channels - 1)
            flat = tb * n_channels + us
            counts = np.bincount(flat, minlength=t_bins * n_channels)
            X[i] = counts[:t_bins * n_channels].reshape(t_bins, n_channels).clip(0, 255).astype(np.uint8)
            if (i + 1) % 10000 == 0:
                print(f'    {i + 1}/{N} ...', flush=True)
    y = labels.astype(np.int64)
    dt = time.time() - t0
    print(f'    done: {N} samples, {empty} empty, range [{X.min()},{X.max()}], {dt:.1f}s', flush=True)
    return (X, y, max_time)


def _md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_and_decompress(record, raw_dir):
    archive = raw_dir / record["filename"]
    h5 = archive.with_suffix("")
    if not archive.exists() or _md5(archive) != record["md5"]:
        partial = archive.with_suffix(archive.suffix + ".partial")
        print(f"[DOWNLOAD] {record['filename']}", flush=True)
        with urllib.request.urlopen(record["url"]) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received, last_report = 0, time.monotonic()
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                stream.write(block)
                received += len(block)
                if time.monotonic() - last_report >= 5:
                    print(f"[DOWNLOAD] {received / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                    last_report = time.monotonic()
        if _md5(partial) != record["md5"]:
            raise ValueError(f"Official source checksum mismatch: {record['filename']}")
        os.replace(partial, archive)
    # Decompress the verified archive, replacing any incomplete HDF5 file.
    partial_h5 = h5.with_suffix(h5.suffix + ".partial")
    print(f"[DECOMPRESS] {archive.name}", flush=True)
    with gzip.open(archive, "rb") as source, partial_h5.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    os.replace(partial_h5, h5)
    return h5


def prepare_audio(dataset, download=True):
    """Prepare SHD or SSC using relative manifest paths and verify identity."""
    if dataset not in ("shd", "ssc"):
        raise ValueError("dataset must be shd or ssc")
    manifest = json.loads(resolve_path("data/manifests/datasets.json").read_text())
    entry = manifest["datasets"][dataset]
    cache = resolve_path(entry["cache_path"])
    raw_dir = resolve_path(entry["raw_directory"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        from .datasets import load_data
        load_data(entry["cache_path"], dataset, entry["array_fingerprint"])
        print(f"[VERIFIED] {entry['cache_path']}", flush=True)
        return entry["cache_path"]
    paths = {}
    for split, record in entry["sources"].items():
        if download:
            paths[split] = _fetch_and_decompress(record, raw_dir)
        else:
            paths[split] = raw_dir / record["filename"].removesuffix(".gz")
            if not paths[split].is_file():
                raise FileNotFoundError(f"Missing raw data: {record['filename'].removesuffix('.gz')}")
    maximum = None if dataset == "ssc" else entry["recorded_maximum_time_seconds"]
    Xtr, ytr, maximum = bin_h5(paths["train"], 100, 700, max_time=maximum)
    if dataset == "ssc":
        Xva, yva, _ = bin_h5(paths["valid"], 100, 700, max_time=maximum)
    Xte, yte, _ = bin_h5(paths["test"], 100, 700, max_time=maximum)
    if dataset == "shd":
        itr, iva = _strat(ytr, .10, 0)
        arrays = ((_store(Xtr[itr]), _labels(ytr[itr])),
                  (_store(Xtr[iva]), _labels(ytr[iva])), (_store(Xte), _labels(yte)))
        payload = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, t_max=maximum)
    else:
        arrays = ((_store(Xtr), _labels(ytr)), (_store(Xva), _labels(yva)),
                  (_store(Xte), _labels(yte)))
        payload = dict(Xtr=arrays[0][0], ytr=arrays[0][1], Xva=arrays[1][0], yva=arrays[1][1],
                       Xte=arrays[2][0], yte=arrays[2][1], t_max=maximum, T=100, C=700)
    actual = data_fingerprint(arrays)
    if actual != entry["array_fingerprint"]:
        raise ValueError(f"Prepared arrays differ from the archived fingerprint: {actual}. "
                         "No cache was published. The original prepared cache is required.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(cache.suffix + ".partial")
    if dataset == "shd":
        with partial.open("wb") as stream:
            np.savez_compressed(stream, **payload)
    else:
        torch.save(payload, partial)
    os.replace(partial, cache)
    print(f"[VERIFIED] {entry['cache_path']} fingerprint={actual}", flush=True)
    return entry["cache_path"]
