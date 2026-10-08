"""DVS128 Gesture preprocessing with the archived numerical conventions.

The event parser and frame integration are adapted from SpikingJelly.
The distributed adapter was compared with the verified PyPI release
0.0.0.0.14; source hashes, function mapping and local changes are recorded in
data/manifests/dvs_preprocessing_source.json. Upstream portions retain the
Open-Intelligence Open Source License V1.0 in licenses/. See
licenses/THIRD_PARTY_NOTICES.md for scope and attribution.

The historical notebook did not record its upstream revision. The verified
release is the current compatibility reference, not a claim about that
unrecorded revision. Archived bin boundaries and empty-interval behavior
are retained; AEDAT decoding uses explicit little-endian byte order.
"""
import json
import os
import struct
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .paths import resolve_path
from .datasets import _strat, data_fingerprint

def load_aedat_v3(file_name):
    xs, ys, ts, ps = ([], [], [], [])
    with open(file_name, 'rb') as f:
        line = f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            line = f.readline()
        while True:
            header = f.read(28)
            if not header or len(header) < 28:
                break
            e_type = struct.unpack('<H', header[0:2])[0]
            e_size = struct.unpack('<I', header[4:8])[0]
            e_tsoverflow = struct.unpack('<I', header[12:16])[0]
            e_capacity = struct.unpack('<I', header[16:20])[0]
            data = f.read(e_capacity * e_size)
            if e_type == 1 and e_size == 8:
                raw = np.frombuffer(data, dtype="<u4")
                aer = raw[0::2]
                tlow = raw[1::2]
                xs.append(aer >> 17 & 32767)
                ys.append(aer >> 2 & 32767)
                ps.append(aer >> 1 & 1)
                ts.append(tlow.astype(np.int64) | np.int64(e_tsoverflow) << 31)
    cat = lambda L: np.concatenate(L) if L else np.array([], np.int64)
    return {'x': cat(xs).astype(np.int64), 'y': cat(ys).astype(np.int64), 'p': cat(ps).astype(np.int64), 't': cat(ts).astype(np.int64)}

def _seg_to_frame(ev, H, W, j_l, j_r):
    frame = np.zeros([2, H * W], np.float32)
    x = ev['x'][j_l:j_r].astype(int)
    y = ev['y'][j_l:j_r].astype(int)
    p = ev['p'][j_l:j_r]
    m = [p == 0]
    m.append(np.logical_not(m[0]))
    for c in range(2):
        pos = y[m[c]] * W + x[m[c]]
        if pos.size:
            cnt = np.bincount(pos)
            frame[c][:cnt.size] += cnt
    return frame.reshape((2, H, W))

def _seg_index(t, split_by, M):
    """Archived boundaries, including the original empty-final-interval behavior.

    For time bins, a final interval with no selected event retains start 0;
    its end is then extended to N. This inherited edge case is preserved so
    regeneration remains subject to the original full-dataset fingerprint.
    """
    j_l = np.zeros(M, int)
    j_r = np.zeros(M, int)
    N = t.size
    if N == 0:
        return (j_l, j_r)
    if split_by == 'number':
        di = max(1, N // M)
        for i in range(M):
            j_l[i] = min(i * di, N)
            j_r[i] = min(j_l[i] + di, N)
        j_r[-1] = N
    else:
        dt = max(1, (t[-1] - t[0]) // M)
        idx = np.arange(N)
        for i in range(M):
            tl = dt * i + t[0]
            mm = idx[np.logical_and(t >= tl, t < tl + dt)]
            if mm.size:
                j_l[i] = mm[0]
                j_r[i] = mm[-1] + 1
        j_r[-1] = N
    return (j_l, j_r)

def integrate_to_frames(ev, split_by, M, H, W):
    j_l, j_r = _seg_index(ev['t'], split_by, M)
    fr = np.zeros([M, 2, H, W], np.float32)
    for i in range(M):
        fr[i] = _seg_to_frame(ev, H, W, j_l[i], j_r[i])
    return fr

def downsample(ev, src, dst):
    if src == dst:
        return ev
    return {'t': ev['t'], 'p': ev['p'], 'x': ev['x'] * dst // src, 'y': ev['y'] * dst // src}


def build_split(base, list_file, split_by="time"):
    names = [line.strip() for line in (base / list_file).read_text().splitlines() if line.strip()]
    inputs, labels = [], []
    for filename in tqdm(names, desc=list_file):
        events_path = base / filename
        labels_path = base / filename.replace(".aedat", "_labels.csv")
        if not events_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Missing event recording or labels: {filename}")
        events = downsample(load_aedat_v3(events_path), 128, 64)
        rows = np.loadtxt(labels_path, dtype=np.uint32, delimiter=",", skiprows=1)
        rows = rows[None, :] if rows.ndim == 1 else rows
        for label, begin, end in rows:
            selected = (events["t"] >= begin) & (events["t"] < end)
            segment = {name: values[selected] for name, values in events.items()}
            frames = integrate_to_frames(segment, split_by, 64, 64, 64)
            inputs.append(np.clip(frames, 0, 255).astype(np.uint8))
            labels.append(int(label) - 1)
    return np.stack(inputs), np.asarray(labels, np.int64)


def prepare_gesture(raw_directory="data/raw/dvs_gesture", split_by="time"):
    """Build from the extracted official recordings, requiring archived identity."""
    if split_by not in ("time", "number"):
        raise ValueError("split_by must be time or number")
    root = resolve_path(raw_directory)
    matches = sorted(root.rglob("trials_to_train.txt"))
    if len(matches) != 1:
        raise FileNotFoundError("Place one extracted DVS128 Gesture dataset under " + raw_directory)
    base = matches[0].parent
    kind = "time" if split_by == "time" else "event_count"
    output = "data/processed/gesture_" + kind + ".pt"
    expected = {"time": "3c87c257bb2d", "number": "56155d24937b"}[split_by]
    target = resolve_path(output)
    if target.exists():
        from .vision import load_dvs
        load_dvs(output, expected)
        return output
    xtrain, ytrain = build_split(base, "trials_to_train.txt", split_by)
    xtest, ytest = build_split(base, "trials_to_test.txt", split_by)
    itr, iva = _strat(ytrain, .10, 0)
    arrays = ((torch.from_numpy(xtrain[itr]), torch.from_numpy(ytrain[itr])),
              (torch.from_numpy(xtrain[iva]), torch.from_numpy(ytrain[iva])),
              (torch.from_numpy(xtest), torch.from_numpy(ytest)))
    observed = data_fingerprint(arrays)
    if observed != expected:
        raise ValueError(f"Gesture arrays differ from the archived fingerprint: {observed}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.partial")
    torch.save(dict(Xtr=torch.from_numpy(xtrain), ytr=torch.from_numpy(ytrain),
                    Xte=torch.from_numpy(xtest), yte=torch.from_numpy(ytest)), partial)
    os.replace(partial, target)
    print(f"[VERIFIED] {output} fingerprint={observed}", flush=True)
    return output
