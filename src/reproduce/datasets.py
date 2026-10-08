"""Prepared audio caches, fixed validation splits and reproducible batches."""
import hashlib
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from .paths import resolve_path

def _strat(y, frac, seed):
    rng = np.random.RandomState(seed)
    tr, va = ([], [])
    for cc in np.unique(y):
        ix = np.where(y == cc)[0]
        rng.shuffle(ix)
        n = max(1, int(round(len(ix) * frac)))
        va += list(ix[:n])
        tr += list(ix[n:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return (np.array(tr), np.array(va))

def _store(x):
    """Integer-valued data (spike counts) kept as uint8, anything else as float32; cast per batch.
    Values are identical to v1's float32 arrays in both cases (uint8 -> float32 is exact)."""
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
    if not t.is_floating_point():
        assert int(t.max()) <= 255 and int(t.min()) >= 0, 'integer data outside uint8 range'
        return t.to(torch.uint8).contiguous()
    f = t.float()
    if torch.equal(f, f.round()) and float(f.max()) <= 255 and (float(f.min()) >= 0):
        return f.to(torch.uint8).contiguous()
    return f.contiguous()

def _labels(x):
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
    return t.long().contiguous()

def data_fingerprint(data):
    h = hashlib.sha1()
    for X, y in data:
        h.update(f'{tuple(X.shape)}|{X.dtype}|{tuple(y.shape)}'.encode())
        xs = X.reshape(-1)[::7919][:2000000].contiguous().numpy().tobytes()
        h.update(xs)
        h.update(y.numpy().tobytes())
    return h.hexdigest()[:12]

def temporal_pool_exact(X, f, chunk=2048):
    """Mean of f consecutive bins, same values as the reference temporal_pool (float32 mean).
    Stored as fp16 when every value is exact in fp16 (uint8 counts, f in {2,4}); float32 otherwise (x10)."""
    N, T, C = X.shape
    Tn = T // f * f
    try_half = X.dtype == torch.uint8 and f in (1, 2, 4, 8)
    out = torch.empty((N, T // f, C), dtype=torch.float16 if try_half else torch.float32)
    for i in range(0, N, chunk):
        o = X[i:i + chunk, :Tn].float().reshape(-1, T // f, f, C).mean(2)
        oc = o.to(out.dtype)
        if try_half:
            assert torch.equal(oc.float(), o), 'pooled bins not exact in fp16'
        out[i:i + chunk] = oc
    return out

def pool_batch(X, f):
    """Mean of f consecutive bins of a [B,T,C] batch, same arithmetic as temporal_pool_exact (float32 mean)."""
    B, T, C = X.shape
    Tn = T // f * f
    return X[:, :Tn].reshape(B, T // f, f, C).mean(2)

def load_data(path, dataset="shd", expected_fingerprint=None):
    key = dataset
    ds_name = dataset
    p = resolve_path(path)
    print(f"  [{time.strftime('%H:%M:%S')}] loading {key} <- {p.name} ({p.stat().st_size / 1000000.0:.0f}MB)...", flush=True)
    ext = p.suffix.lower()
    if ext == '.pt':
        # Modern tensor caches can be paged from disk, including the 7.4 GB
        # SSC cache. Legacy torch files retain their original eager loader.
        d = torch.load(p, map_location='cpu', weights_only=False,
                       mmap=zipfile.is_zipfile(p))
        keys = set(d.keys())
    elif ext == '.npz':
        dn = np.load(p, allow_pickle=True)
        keys = set(dn.keys())
        d = {k: dn[k] for k in keys}
    else:
        raise ValueError(ext)
    print(f'[DATA] keys={sorted(keys)}', flush=True)
    if {'Xtr', 'ytr', 'Xva', 'yva', 'Xte', 'yte'} <= keys:
        Xtr, ytr = (_store(d['Xtr']), _labels(d['ytr']))
        Xva, yva = (_store(d['Xva']), _labels(d['yva']))
        Xte, yte = (_store(d['Xte']), _labels(d['yte']))
    elif {'X_train', 'y_train', 'X_test', 'y_test'} <= keys:
        Xtr, ytr = (_store(d['X_train']), _labels(d['y_train']))
        Xte, yte = (_store(d['X_test']), _labels(d['y_test']))
        if 'X_valid' in keys:
            Xva, yva = (_store(d['X_valid']), _labels(d['y_valid']))
        else:
            itr, iva = _strat(ytr.numpy(), 0.10, 0)
            itr, iva = (torch.from_numpy(itr), torch.from_numpy(iva))
            Xva, yva = (Xtr[iva], ytr[iva])
            Xtr, ytr = (Xtr[itr], ytr[itr])
    elif {'Xtr', 'ytr', 'Xte', 'yte'} <= keys:
        Xtr, ytr = (_store(d['Xtr']), _labels(d['ytr']))
        Xte, yte = (_store(d['Xte']), _labels(d['yte']))
        itr, iva = _strat(ytr.numpy(), 0.10, 0)
        itr, iva = (torch.from_numpy(itr), torch.from_numpy(iva))
        Xva, yva = (Xtr[iva], ytr[iva])
        Xtr, ytr = (Xtr[itr], ytr[itr])
        print(f'[DATA] no validation split in file -> stratified {0.10:.0%} of train (seed 0)', flush=True)
    else:
        raise KeyError(f'layout: {sorted(keys)}')
    del d
    assert Xtr.shape[-1] == 700, "Expected 700 input channels"
    ncls = int(ytr.max()) + 1
    assert ncls == {"shd": 20, "ssc": 35}[ds_name], "Unexpected class count"
    print(f'[DATA] train {tuple(Xtr.shape)} val {tuple(Xva.shape)} test {tuple(Xte.shape)} classes={ncls} stored={Xtr.dtype} range[{float(Xtr.min()):.1f},{float(Xtr.max()):.1f}]', flush=True)
    data = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    if expected_fingerprint and data_fingerprint(data) != expected_fingerprint:
        raise ValueError("Dataset arrays differ from the configured fingerprint")
    return data

class ArrLoader:
    """In-order (or per-epoch shuffled) batches, cast to float32. Reference functions consume it directly."""

    def __init__(self, X, y, bs, shuffle=False, seed=0, device="cpu"):
        self.X, self.y, self.bs, self.shuffle, self.seed, self.epoch = (X, y, bs, shuffle, seed, 0)
        self.device = torch.device(device)

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed * 100003 + self.epoch)) if self.shuffle else None
        for i in range(0, n, self.bs):
            if idx is None:
                yield (self.X[i:i + self.bs].to(self.device, non_blocking=True).float(), self.y[i:i + self.bs])
            else:
                j = idx[i:i + self.bs]
                yield (self.X[j].to(self.device, non_blocking=True).float(), self.y[j])
