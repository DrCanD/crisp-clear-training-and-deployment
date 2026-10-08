#!/usr/bin/env python3
"""Select fixed-point precision on the existing SHD validation split.

Calibration reads training arrays only. Evaluation requires an immutable,
content-hashed freeze and aborts on integer overflow instead of retuning.
The original hardware format was test-informed; a validation-only reevaluation
does not make that historical test set an untouched confirmatory set.
All input and output paths are relative to the repository root.
"""
import os, sys, json, math, time, hashlib, copy, shutil, io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SMOKE = False          # yerel duman testi: kucuk sahte veri/model, az draws (bu hucre gercek kipte teslim edilir)

from scipy.stats import binom
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from verify_integer import repository_path
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_VERSION = 5  # Reconstructed after the 2026-10-03 runtime replacement.
if hasattr(torch, 'set_float32_matmul_precision'): torch.set_float32_matmul_precision('highest')
if hasattr(torch.backends, 'cuda'): torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, 'cudnn'): torch.backends.cudnn.allow_tf32 = False
REPRO_TOL_SAMPLES = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class HWConfig:
    exp_name: str = 'hardware_quantization'
    ckpt_rel: str = 'data/checkpoints/hardware/shd_seed999.pt'
    expected_ckpt_sha256: str = '5927a1929f86d51db61e4ce6dbcc1bb420c35fc15fcc61a70bfcd6cd5ba9d609'
    dataset_key: str = 'SHD'
    dataset_confirmed_rel: str = 'data/processed/shd.npz'
    val_frac: float = 0.10           # Existing training validation split: stratified Xtr, seed 0
    # --- donanim bicimi (temel) ---
    fmt: dict = field(default_factory=lambda: dict(
        T=100, N_IN=700, H=256, C=20, K=8, P=8,      # P: RNG seridi = cevrim basina islenen noron sayisi
        WB=8, CB=18, FA=18, FS=16, FZ=10, NB=12, SB=11,
        M=256, DENOM_SEQ=5000, DENOM_FIX=1000, LOOKS=[16, 32, 64, 128, 256]))
    # Validation-only candidate grid; the noise-LUT width remains fixed.
    sweep_FA: list = field(default_factory=lambda: [16, 18])
    sweep_FS: list = field(default_factory=lambda: [12, 16])
    sweep_FZ: list = field(default_factory=lambda: [8, 10])
    sweep_NB: list = field(default_factory=lambda: [10, 12])   # duyarlilik kaydi; paket NB=fmt['NB'] ile yazilir
    mf_agree_min: float = 0.995      # nicemlenmis MF karari = float MF karari (oran)
    mf_acc_tol: float = 0.003        # |MF accuracy farki|
    samp_acc_tol: float = 0.010      # alt kumede |n=1 accuracy farki| (32 draws ortalamasi)
    subset_n: int = 256
    subset_draws: int = 32
    float_sample_passes: int = 16
    # --- kahin yurutme ---
    chunk_inputs: int = 128
    chunk_draws: int = 64
    n_fixture: int = 128
    test_limit: int = 0
    # --- calisma zamani (anahtara girmez) ---
    smoke: bool = SMOKE
    device: str = device

cfg = HWConfig()
OUTPUT_ROOT = PROJECT / 'outputs' / 'hardware' / 'quantization'
if cfg.smoke:
    cfg.fmt.update(M=16, LOOKS=[16]); cfg.subset_n = 8; cfg.subset_draws = 4; cfg.float_sample_passes = 2
    cfg.chunk_inputs = 4; cfg.chunk_draws = 8; cfg.n_fixture = 4; cfg.test_limit = 12
_KEY_EXCLUDE = ('device', 'smoke', 'exp_name', 'chunk_inputs', 'chunk_draws')

def get_dataset_path(name):
    if name != cfg.dataset_key:
        raise ValueError('Unexpected dataset key')
    path = repository_path(cfg.dataset_confirmed_rel)
    if not path.is_file():
        raise FileNotFoundError('Prepare the SHD cache first: ' + cfg.dataset_confirmed_rel)
    return path

def relative_metadata(obj):
    if isinstance(obj, dict):
        return {k: relative_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [relative_metadata(v) for v in obj]
    if isinstance(obj, Path):
        return obj.relative_to(PROJECT).as_posix()
    if isinstance(obj, str) and obj.startswith(str(PROJECT) + os.sep):
        return Path(obj).relative_to(PROJECT).as_posix()
    return obj

def write_json_atomic(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f: json.dump(relative_metadata(obj), f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def save_npz_atomic(path, **arrs):
    path = Path(path); tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrs); os.replace(tmp, path)
def save_npy_atomic(path, array):
    path=Path(path);tmp=path.with_suffix('.tmp.npy')
    np.save(tmp,array,allow_pickle=False);os.replace(tmp,path)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''): h.update(blk)
    return h.hexdigest()
def _u8(x):
    """Validate non-negative integer counts and return a uint8 tensor."""
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
    if t.dtype != torch.uint8:
        f = t.float()
        if not (torch.equal(f, f.round()) and float(f.min()) >= 0 and float(f.max()) <= 255):
            raise ValueError("expected non-negative integer counts")
        t = f.to(torch.uint8)
    return t.contiguous()
def set_all_seeds(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class RunLog:
    def __init__(self): self.path = None
    def open(self, path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def __call__(self, msg, screen=True):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        if screen: print(line, flush=True)
        if self.path:
            with open(self.path, 'a') as f: f.write(line + '\n')
log = RunLog()

# The freeze records the exact numerical source identities.
from float_model import Config, DendStochAudioNet
from integer_reference import *
REF_SRC = (HERE / 'float_model.py').read_text()
ORACLE_SRC = (HERE / 'integer_reference.py').read_text()
REF_MD5 = hashlib.md5(REF_SRC.encode()).hexdigest()
ORACLE_MD5 = hashlib.md5(ORACLE_SRC.encode()).hexdigest()

# ====================== PIPELINE ======================
def batches(X, bs):
    for i in range(0, len(X), bs): yield i, X[i:i + bs]

def run_key_of(ckpt_sha, data_fp):
    payload = {k: v for k, v in asdict(cfg).items() if k not in _KEY_EXCLUDE}
    payload.update(run_version=RUN_VERSION, oracle_version=ORACLE_VERSION, ckpt=ckpt_sha[:16], data_fp=data_fp,
                   ref_md5=REF_MD5, oracle_md5=ORACLE_MD5, exporter_sha256=sha256_file(Path(__file__)))
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

def fmt_dur(sec):
    sec = max(0, int(sec)); h, m = divmod(sec, 3600); m, s = divmod(m, 60)
    return f'{h} h {m:02d} min' if h else f'{m} min {s:02d} s'

@torch.no_grad()
def float_reference(model, Xte, yte, beta, bs=128):
    """reference (torch) graded logits/decisions and the sampled n=1 accuracy averaged over several passes."""
    model.eval(); zs = []
    for i, xb in batches(Xte, bs):
        lg, _, _ = model(xb.to(device).float(), 'graded', beta); zs.append(lg.cpu().numpy())
    zbar = np.concatenate(zs); cs = zbar.argmax(1)
    accs, flips = [], []
    for k in range(cfg.float_sample_passes):
        set_all_seeds(1000 + k); ys = []
        for i, xb in batches(Xte, bs):
            lg, _, _ = model(xb.to(device).float(), 'sample', beta); ys.append(lg.argmax(1).cpu().numpy())
        ys = np.concatenate(ys); accs.append(float((ys == yte).mean())); flips.append(float((ys != cs).mean()))
    return zbar, cs, float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, float(np.mean(flips))

def thresholds_selftest(Q):
    """The integer acceptance table must reproduce the seven Lean-checked boundary cases (the Lean threshold proof)."""
    ts, tf = Q['THR_SEQ'], Q['THR_FIX']; M = Q['fmt']['M']
    checks = []
    if M >= 16:
        checks += [tf[10] > 10, tf[11] <= 11, ts[13] > 13, ts[14] <= 14, ts[16] <= 16, not (15 >= ts[16]), 15 >= tf[16]]
    if not all(checks): raise RuntimeError(f'[GATE] threshold table disagrees with the Lean boundary cases: {checks}')
    log(f'[GATE] threshold tables match Lean boundary cases (1e-3: 11 unanimous votes; 2e-4: 14; 16 draws: (16,0) passes; (15,1) fails sequential and passes fixed-budget)')

def sweep_mf(F, Xte_np, cs_float, yte):
    """Mean-field candidate diagnostics on the supplied calibration split only; does not access any dataset file."""
    rows = []; base = dict(cfg.fmt)
    for FA in cfg.sweep_FA:
        for FS in cfg.sweep_FS:
            for FZ in cfg.sweep_FZ:
                fmt = dict(base, FA=FA, FS=FS, FZ=FZ); Q = to_dev(quantize(F, fmt), device)
                dec = []; mx = {'u1': 0, 'S1': 0, 'u2': 0, 'S2': 0, 'r': 0, 'R': 0, 'L': 0}
                for i, xb in batches(Xte_np, cfg.chunk_inputs):
                    z1, mu1, ms1 = prefix_z1(Q, xb, device); d, _, m2 = mf_forward(Q, z1, device)
                    dec.append(d.cpu().numpy()); mx['u1'] = max(mx['u1'], mu1); mx['S1'] = max(mx['S1'], ms1)
                    for k in m2: mx[k] = max(mx[k], m2[k])
                dec = np.concatenate(dec); agree = float((dec == cs_float).mean()); acc = float((dec == yte).mean())
                rows.append({'FA': FA, 'FS': FS, 'FZ': FZ, 'agree': agree, 'acc': acc, 'max': mx,
                             'state_bits': int(math.ceil(math.log2(max(mx['S1'], mx['S2'], 1) * 4)) + 1)})
                log(f'  [MF] FA={FA} FS={FS} FZ={FZ}: decision agreement {agree:.4f}, accuracy {acc:.4f}, state maximum 2^{math.log2(max(mx["S1"], mx["S2"], 1)):.1f}', screen=False)
    return rows

def choose_format(rows, acc_float_mf):
    """smallest total bits (FA+FS+FZ) meeting the agreement and accuracy criteria; prefer fewer state bits."""
    ok = [r for r in rows if r['agree'] >= cfg.mf_agree_min and abs(r['acc'] - acc_float_mf) <= cfg.mf_acc_tol]
    if not ok:
        raise RuntimeError('No candidate meets the validation-only accuracy/agreement gates. No format was frozen; report the failed grid and revise the calibration protocol before evaluation.')
    return min(ok, key=lambda r: (r['FA'] + r['FS'] + r['FZ'], r['state_bits']))

@torch.no_grad()
def sampled_subset_check(model, F, fmt_base, Xte_np, Xte_t, yte, beta, idx):
    """quantized sampled n=1 accuracy (subset, several draws) for each NB vs the float sampler on the same subset."""
    set_all_seeds(2026); accs = []
    xb = Xte_t[torch.as_tensor(idx)].to(device).float()
    for _ in range(cfg.subset_draws):
        lg, _, _ = model(xb, 'sample', beta); accs.append(float((lg.argmax(1).cpu().numpy() == yte[idx]).mean()))
    acc_float = float(np.mean(accs)); rows = []
    for NB in cfg.sweep_NB:
        fmt = dict(fmt_base, NB=NB); Q = to_dev(quantize(F, fmt), device)
        z1, _, _ = prefix_z1(Q, Xte_np[idx], device)
        votes, mx, rates = sampled_draws(Q, z1, idx, 2, np.arange(cfg.subset_draws), device)   # stream 2: sweep only
        acc = float((votes.cpu().numpy() == yte[idx][:, None]).mean())
        rows.append({'NB': NB, 'acc_q': acc, 'acc_float': acc_float, 'rates': rates})
        log(f'  [SAMPLE] NB={NB}: n=1 accuracy (subset x {cfg.subset_draws} draws) quantized {acc:.4f} vs float {acc_float:.4f}; spike rate {rates[0]:.3f}/{rates[1]:.3f}', screen=False)
    NB = fmt_base['NB']                      # the noise-LUT width is fixed (2^12 x 16 bit = 2 BRAM36 per lane); rows are a sensitivity record
    row = [r for r in rows if r['NB'] == NB][0]
    if abs(row['acc_q'] - row['acc_float']) > cfg.samp_acc_tol:
        log(f"[WARNING] NB={NB}: quantized n=1 accuracy differs from the float sampler by {abs(row['acc_q'] - row['acc_float']):.4f} (tolerance {cfg.samp_acc_tol}); the package is still written and the summary retains this difference")
    return rows, NB

def oracle_full(Q, Xte_np, yte, KEY_DIR):
    """prefix + mean-field + 256 certificate + 256 reference draws for every input, chunked and resumable."""
    fmt = Q['fmt']; N = len(Xte_np); M = fmt['M']
    part = KEY_DIR / 'oracle_partial.npz'
    mx_keys=('u1','S1','u2','S2','r','R','L','mf_u2','mf_S2','mf_r','mf_R','mf_L','spk1','spk2','n_spk')
    identity={'schema':1,'data_sha256':arrays_fingerprint(Xte_np,yte),
              'quantized_tables_sha256':quantized_tables_digest(Q),'fmt':fmt,
              'exporter_sha256':sha256_file(Path(__file__)),'oracle_md5':ORACLE_MD5,
              'shape':list(Xte_np.shape),'chunk_inputs':cfg.chunk_inputs,'chunk_draws':cfg.chunk_draws}
    if part.exists():
        try:
            with np.load(part,allow_pickle=False) as z:
                if json.loads(str(z['identity']))!=identity:
                    raise ValueError('Oracle cache data/arithmetic/source/chunk identity differs')
                votes_c,votes_r,mf_dec,mf_log,done=z['votes_c'],z['votes_r'],z['mf_dec'],z['mf_log'],z['done']
                mx=json.loads(str(z['mx']))
            schemas=[(votes_c,(N,M),np.int8),(votes_r,(N,M),np.int8),(mf_dec,(N,),np.int16),
                     (mf_log,(N,fmt['C']),np.int64),(done,(N,),np.bool_)]
            if any(a.shape!=shape or a.dtype!=dtype for a,shape,dtype in schemas):raise ValueError('Oracle cache schema differs')
            nd=int(done.sum())
            if not (done[:nd].all() and not done[nd:].any()) or (nd!=N and nd%cfg.chunk_inputs):
                raise ValueError('Oracle completion mask is not a complete chunk prefix')
            if set(mx)!=set(mx_keys) or any(not math.isfinite(v) or v<0 for v in mx.values()) or mx['n_spk']!=2*M*nd:
                raise ValueError('Oracle maxima/completed-draw counts invalid')
            for a in (votes_c[:nd],votes_r[:nd],mf_dec[:nd]):
                if a.size and (a.min()<0 or a.max()>=fmt['C']):raise ValueError('Oracle class outside range')
            log(f'[RESUME] verified oracle cache: {nd}/{N} inputs ready')
        except Exception as exc:
            raise RuntimeError('Oracle cache identity/schema failed; retained unchanged. Investigate or start a new output run: '+str(part)) from exc
    else: done = None
    if done is None:
        votes_c = np.zeros((N, M), np.int8); votes_r = np.zeros((N, M), np.int8); mf_dec = np.zeros(N, np.int16)
        mf_log = np.zeros((N, fmt['C']), np.int64); done = np.zeros(N, bool)
        mx = {k: 0 for k in ('u1', 'S1', 'u2', 'S2', 'r', 'R', 'L', 'mf_u2', 'mf_S2', 'mf_r', 'mf_R', 'mf_L', 'spk1', 'spk2', 'n_spk')}
    t0 = time.time(); last = 0.0; n_done0 = int(done.sum())
    for i0 in range(0, N, cfg.chunk_inputs):
        i1 = min(N, i0 + cfg.chunk_inputs)
        if done[i0:i1].all(): continue
        idx = np.arange(i0, i1)
        z1, mu1, ms1 = prefix_z1(Q, Xte_np[i0:i1], device); mx['u1'] = max(mx['u1'], mu1); mx['S1'] = max(mx['S1'], ms1)
        d, L, m2 = mf_forward(Q, z1, device); mf_dec[i0:i1] = d.cpu().numpy(); mf_log[i0:i1] = L.cpu().numpy()
        for k in m2: mx['mf_' + k] = max(mx['mf_' + k], m2[k])
        for stream, out in ((0, votes_c), (1, votes_r)):
            for d0 in range(0, M, cfg.chunk_draws):
                draws = np.arange(d0, min(M, d0 + cfg.chunk_draws))
                v, m3, rates = sampled_draws(Q, z1, idx, stream, draws, device)
                out[i0:i1, d0:d0 + len(draws)] = v.cpu().numpy()
                for k in m3: mx[k] = max(mx[k], m3[k])
                mx['spk1'] += rates[0] * len(idx) * len(draws); mx['spk2'] += rates[1] * len(idx) * len(draws); mx['n_spk'] += len(idx) * len(draws)
        done[i0:i1] = True
        save_npz_atomic(part, votes_c=votes_c, votes_r=votes_r, mf_dec=mf_dec, mf_log=mf_log, done=done,
                        mx=np.array(json.dumps(mx)),identity=np.array(json.dumps(identity,sort_keys=True)))
        now = time.time()
        if now - last >= 20 or i1 == N:
            last = now; nd = int(done.sum()); rate = (nd - n_done0) / max(now - t0, 1e-9)
            acc1 = float((votes_c[:nd, 0] == yte[:nd]).mean())
            log(f'[ORACLE] input {nd}/{N} | n=1 accuracy {acc1:.4f} | elapsed {fmt_dur(now - t0)}, remaining ~{fmt_dur((N - nd) / max(rate, 1e-9))}')
    return votes_c, votes_r, mf_dec, mf_log, mx

def certificate_summary(Q, votes_c, votes_r, yte):
    fmt = Q['fmt']; C = fmt['C']; looks = [m for m in fmt['LOOKS'] if m <= fmt['M']]
    seq_dec, used = sequential(votes_c, C, looks, Q['THR_SEQ'])
    fix_dec = predict_table(votes_c, C, Q['THR_FIX'])
    ref_dec = predict_table(votes_r, C, Q['THR_FIX'])
    ok = seq_dec >= 0; res = ref_dec >= 0; k = int((ok & res & (seq_dec != ref_dec)).sum())
    okf = fix_dec >= 0; kf = int((okf & res & (fix_dec != ref_dec)).sum())
    n = len(yte)
    return {'looks': looks, 'level_seq': 1 / fmt['DENOM_SEQ'], 'level_fix': 1 / fmt['DENOM_FIX'],
            'n1_acc': float((votes_c[:, 0] == yte).mean()),
            'seq': {'acc': float((seq_dec == yte).mean()), 'abstain': float((~ok).mean()), 'mean_draws': float(used.mean()),
                    'wrong_among_certified': float((seq_dec != yte)[ok].mean()) if ok.any() else None,
                    'disagree_k': k, 'n_resolved': int(res.sum()), 'rate_unconditional': k / n, 'rate_conditional': (k / int(res.sum())) if res.any() else None},
            'fix256': {'acc': float((fix_dec == yte).mean()), 'abstain': float((~okf).mean()), 'disagree_k': kf, 'rate_unconditional': kf / n},
            'bound_alpha_plus_ref': 1 / fmt['DENOM_SEQ'] * len(looks) + 1 / fmt['DENOM_FIX'],
            'seq_dec': seq_dec, 'used': used, 'fix_dec': fix_dec, 'ref_dec': ref_dec}

def bits_for(maxabs, floor_bits=16):
    return max(floor_bits, int(math.ceil(math.log2(max(1, maxabs) * 4)) + 1))

def write_package(Q, Xte_np, yte, votes_c, votes_r, mf_dec, mf_log, cert, z1_fix, mx, widths, out_dir, n_sel, meta):
    """binary package (little-endian) + parameters.h + integer_reference.py + manifest.json for n_sel inputs."""
    out_dir.mkdir(parents=True, exist_ok=True); fmt = Q['fmt']; files = {}
    def put(name, arr, dtype):
        raw=np.asarray(arr);limits=np.iinfo(np.dtype(dtype))
        if raw.size and (not np.isfinite(raw).all() or raw.min()<limits.min or raw.max()>limits.max or not np.equal(raw,np.rint(raw)).all()):
            raise OverflowError('Binary cast would truncate or overflow: '+name)
        a = np.ascontiguousarray(raw.astype(dtype)); p = out_dir / name
        tmp = p.with_suffix(p.suffix + '.tmp'); a.tofile(tmp); os.replace(tmp, p); files[name] = {'dtype': str(np.dtype(dtype)), 'shape': list(a.shape)}
    for l, L in enumerate(Q['layers'], 1):
        put(f'layer{l}_weights.bin', L['W'], '<i1'); put(f'layer{l}_bias.bin', L['b'], '<i4'); put(f'layer{l}_decay.bin', L['a'], '<u4')
        put(f'layer{l}_input_coefficient.bin', L['c'], '<i4'); put(f'layer{l}_offset.bin', L['h'], '<i4')
    put('readout_weights.bin', Q['ro']['W'], '<i1'); put('readout_bias.bin', Q['ro']['b'], '<i4'); put('readout_decay.bin', Q['ro']['a'], '<u4'); put('readout_input_coefficient.bin', Q['ro']['c'], '<i4')
    put('logistic_noise.bin', Q['NOISE'], '<i2'); put('sigmoid_probability.bin', Q['SIG'], '<u1'); put('sequential_threshold.bin', Q['THR_SEQ'], '<u2'); put('fixed_threshold.bin', Q['THR_FIX'], '<u2')
    # test inputs as event lists
    words, offs = [], [0]
    for i in range(n_sel):
        ev = dense_to_events(Xte_np[i]); words.append(ev); offs.append(offs[-1] + len(ev))
    put('test_events.bin', np.concatenate(words) if words else np.zeros(0, np.uint32), '<u4'); put('test_offsets.bin', np.array(offs), '<u4')
    put('test_labels.bin', yte[:n_sel], '<i2')
    put('expected_mean_field_decisions.bin', mf_dec[:n_sel], '<i2'); put('expected_mean_field_logits.bin', mf_log[:n_sel], '<i8')
    put('expected_certificate_votes.bin', votes_c[:n_sel], '<i1'); put('expected_reference_votes.bin', votes_r[:n_sel], '<i1')
    put('expected_sequential_decisions.bin', np.stack([cert['seq_dec'][:n_sel], cert['used'][:n_sel]], 1), '<i2')
    put('expected_fixed_decisions.bin', cert['fix_dec'][:n_sel], '<i2'); put('expected_reference_decisions.bin', cert['ref_dec'][:n_sel], '<i2')
    if z1_fix is not None: put('expected_layer1_logits.bin', z1_fix[:n_sel], '<i2')
    n_ev_max = int(max(np.diff(offs))) if n_sel else 0
    hdr = ['// parameters.h — generated by the CRISP KV260 quantization exporter; the HLS core and the oracle share these constants.',
           f'// package {meta["key"]}  ckpt {meta["ckpt_sha256"][:16]}  data {meta["data_fp"]}  oracle_md5 {ORACLE_MD5}', '#ifndef HW_PARAMS_H', '#define HW_PARAMS_H']
    for k in ('T', 'N_IN', 'H', 'C', 'K', 'P', 'WB', 'CB', 'FA', 'FS', 'FZ', 'NB', 'SB', 'M', 'DENOM_SEQ', 'DENOM_FIX'):
        hdr.append(f'#define HW_{k} {fmt[k]}')
    hdr += [f'#define HW_SHC1 ({Q["layers"][0]["shc"]})', f'#define HW_SHC2 ({Q["layers"][1]["shc"]})', f'#define HW_SHCRO ({Q["ro"]["shc"]})',
            f'#define HW_N_LOOKS {len(cert["looks"])}', '#define HW_LOOKS {' + ', '.join(str(m) for m in cert['looks']) + '}']
    for k, v in widths.items(): hdr.append(f'#define HW_{k} {v}')
    hdr += [f'#define HW_N_TEST {n_sel}', f'#define HW_NEV_MAX {n_ev_max}', f'#define HW_NEV_TOTAL {offs[-1]}',
            f'#define HW_PACKAGE_ID 0x{int(meta["key"], 16):08x}u', '#endif']
    (out_dir / 'parameters.h').write_text('\n'.join(hdr) + '\n'); (out_dir / 'integer_reference.py').write_text(ORACLE_SRC)
    params = {'fmt': fmt, 'shc': {'l1': Q['layers'][0]['shc'], 'l2': Q['layers'][1]['shc'], 'ro': Q['ro']['shc']}, 'scales': Q['scales'],
              'widths': widths, 'maxabs': mx, 'n_test': n_sel, 'n_events_max': n_ev_max, 'n_events_total': int(offs[-1]), 'files': files, **meta}
    write_json_atomic(params, out_dir / 'parameters.json')
    manifest = {'package': meta['key'], 'created': datetime.now().isoformat(), 'oracle_md5': ORACLE_MD5, 'ref_md5': REF_MD5,
                'files': {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    write_json_atomic(manifest, out_dir / 'manifest.json')
    write_json_atomic(manifest['files'], out_dir / 'sha256.json')
    return params

# ---------- Validation-only selection and immutable freeze ----------
HISTORICAL_TEST_EXPOSURE = (
    'The original SHD test set was used in the earlier hardware-format selection. '
    'These repaired results are a validation-selected engineering reevaluation; '
    'they do not restore a previously untouched test set.'
)

def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _strat_validation_indices(y, frac=0.10, seed=0):
    """Stratified validation indices with the original per-class rounding rule."""
    rng = np.random.RandomState(seed); tr, va = [], []
    for cc in np.unique(y):
        ix = np.where(y == cc)[0]; rng.shuffle(ix)
        n = max(1, int(round(len(ix) * frac)))
        va += list(ix[:n]); tr += list(ix[n:])
    rng.shuffle(tr); rng.shuffle(va)
    return np.asarray(va, dtype=np.int64)

def load_calibration_only(path, frac=0.10):
    # NPZ arrays are lazy: deliberately request Xtr/ytr only. Do not request Xte/yte.
    with np.load(path, allow_pickle=False) as z:
        Xtrain = _u8(z['Xtr']); ytrain = np.asarray(z['ytr'], dtype=np.int64)
    idx = _strat_validation_indices(ytrain, frac, 0)
    return Xtrain[idx].contiguous(), ytrain[idx], idx

def arrays_fingerprint(X, y):
    xa = np.ascontiguousarray(X.numpy() if isinstance(X, torch.Tensor) else X)
    ya = np.ascontiguousarray(y)
    h = hashlib.sha256()
    for a in (xa, ya):
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()

def widths_from_calibration(mx):
    """Engineering widths with factor-four headroom, frozen before evaluation.

    These are observed validation bounds, not a proof of bounds on all inputs.
    Evaluation aborts on a violation; it never widens these values automatically.
    """
    return {'WU1': bits_for(mx['u1']), 'WS': bits_for(max(mx['S1'], mx['S2'], mx['mf_S2']), 24),
            'WU2': bits_for(mx['u2']), 'WU2_MF': bits_for(mx['mf_u2']),
            'WR': bits_for(mx['r']), 'WR_MF': bits_for(mx['mf_r']),
            'WRS': bits_for(max(mx['R'], mx['mf_R']), 24),
            'WL': bits_for(max(mx['L'], mx['mf_L']), 32)}

def validate_frozen_widths(mx, widths):
    groups = {'WU1': ('u1',), 'WS': ('S1', 'S2', 'mf_S2'), 'WU2': ('u2',),
              'WU2_MF': ('mf_u2',), 'WR': ('r',), 'WR_MF': ('mf_r',),
              'WRS': ('R', 'mf_R'), 'WL': ('L', 'mf_L')}
    violations = {name: {'observed_absmax': max(mx[k] for k in keys),
                         'signed_max': (1 << (widths[name] - 1)) - 1}
                  for name, keys in groups.items()
                  if max(mx[k] for k in keys) > (1 << (widths[name] - 1)) - 1}
    if violations:
        raise RuntimeError('Frozen hardware width exceeded. No package produced and no automatic retuning: ' + json.dumps(violations))
    return True

def seal_freeze(payload):
    return {'payload': payload, 'sha256': canonical_sha(payload)}

def verify_freeze(record, ckpt_sha, oracle_md5, ref_md5):
    p = record['payload']
    if record['sha256'] != canonical_sha(p):
        raise ValueError('Freeze digest mismatch; the frozen record was modified.')
    if p.get('selection_split') != 'training_validation_seed0':
        raise ValueError('A test-selected or unknown calibration split cannot be used.')
    if (p['ckpt_sha256'], p['oracle_md5'], p['ref_md5']) != (ckpt_sha, oracle_md5, ref_md5):
        raise ValueError('Checkpoint or exact arithmetic differs from the frozen calibration.')
    if p['exporter_sha256'] != sha256_file(Path(__file__)):
        raise ValueError('Exporter code differs from calibration. Create a new validation freeze; do not modify an existing record.')
    return p

def load_fixed_checkpoint():
    ckpt = PROJECT / cfg.ckpt_rel
    identity=sha256_file(ckpt)
    if identity!=cfg.expected_ckpt_sha256:raise ValueError('Checkpoint identity differs from confirmed SHA256')
    blob = torch.load(ckpt, map_location='cpu', weights_only=True)
    if not isinstance(blob,dict) or not isinstance(blob.get('cfg'),dict) or not isinstance(blob.get('state'),dict):
        raise TypeError('Expected confirmed tensor state/plain config; unsafe pickle fallback disabled')
    arch = dict(blob['cfg'])
    model = DendStochAudioNet(Config(**{k: v for k, v in arch.items() if k in Config.__dataclass_fields__})).to(device)
    model.load_state_dict(blob['state'], strict=True); model.eval()
    if not (arch.get('ct_exact') and arch.get('n_layers')==2 and arch.get('norm_type')=='batchnorm' and arch.get('readout_pool')=='mean'):
        raise ValueError('Checkpoint outside confirmed two-layer real-ZOH BatchNorm mean-readout contract')
    for field,key in [('n_in','N_IN'),('n_hidden','H'),('n_classes','C'),('k_modes','K')]:
        if arch.get(field)!=cfg.fmt[key]:raise ValueError('Checkpoint dimension differs: '+field)
    return model, float(arch.get('beta_end', 5.0)), identity

def fold_gate(model, beta, X, y):
    zbar, cs, acc_s1, acc_s1_sd, flip = float_reference(model, X, y, beta)
    Fp = fold_params(model.state_dict(), beta); ncheck = min(len(y), 64)
    lg, _ = folded_float_forward(Fp, X[:ncheck].numpy().astype(np.int64))
    rel = float(np.abs(lg-zbar[:ncheck]).max()/(np.abs(zbar[:ncheck]).max()+1e-30))
    if rel > 1e-3 or int((lg.argmax(1) != cs[:ncheck]).sum()):
        raise RuntimeError('Folded model fidelity gate failed before hardware export.')
    return Fp, cs, {'mf_acc': float((cs == y).mean()), 'n1_acc': acc_s1,
                    'n1_acc_sd': acc_s1_sd, 'flip_vs_mf': flip, 'fold_max_relative_error': rel}

def environment_record():
    import platform,scipy
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'numpy':np.__version__,'scipy':scipy.__version__,'device':str(device),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32)}

def quantized_tables_digest(Q):
    """Hash exact integer arithmetic, excluding device caches and observed outputs."""
    h=hashlib.sha256();h.update(json.dumps(Q['fmt'],sort_keys=True).encode())
    for name,L in [(f'layer{i}',v) for i,v in enumerate(Q['layers'])]+[('ro',Q['ro'])]:
        for key in ('W','b','a','c','h','shc'):
            if key not in L:continue
            a=np.ascontiguousarray(L[key],dtype='<i8');h.update((name+'/'+key).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    for key in ('NOISE','SIG','THR_SEQ','THR_FIX'):
        a=np.ascontiguousarray(Q[key],dtype='<i8');h.update(key.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def check_int64_contract(Q):
    """Conservative finite-horizon integer-oracle bounds, not universal HLS-width bounds."""
    fmt=Q['fmt'];lim=(1<<63)-1;records={}
    if not (5<=fmt['FZ']<=fmt['FS'] and fmt['T']<=128 and fmt['N_IN']<=1024 and fmt['P']<=256 and fmt['M']<=65536):
        raise ValueError('Format violates event, shift or RNG-address contract')
    if fmt['H']%fmt['P']:raise ValueError('Hidden channels must be divisible by RNG lanes')
    def checked(v,label):
        v=int(v)
        if v<0 or v>lim:raise OverflowError('Conservative int64 oracle bound failed: '+label+'='+str(v))
        return v
    def ceil_shift(v,shift):
        return (v+(1<<shift)-1)//(1<<shift) if shift>=0 else checked(v<<(-shift),'left_shift')
    def layer(name,L,input_max,extra=0,readout=False):
        W=np.asarray(L['W']);b=np.asarray(L['b']);a=np.asarray(L['a']);c=np.asarray(L['c'])
        if not np.isfinite(W).all() or not np.isfinite(b).all():raise ValueError('Non-finite integer tables')
        if (a<0).any() or (a>=(1<<fmt['FA'])).any():raise ValueError('Decay table outside stable range')
        rows=max(sum(abs(int(v)) for v in row) for row in W)
        dot=checked(rows*int(input_max),name+'/linear_dot')
        if dot>=(1<<53):raise OverflowError('Dot product outside float64 exact-integer range: '+name)
        ub=checked(dot+max(abs(int(v)) for v in b)*(1<<extra),name+'/projection_bias')
        am=max(int(v) for v in a.flat);cm=max(abs(int(v)) for v in c.flat)
        prod_c=checked(cm*ub,name+'/c_times_u');state=0;acc=0
        for _ in range(fmt['T']):
            prod_a=checked(am*state,name+'/a_times_state')
            state=checked(ceil_shift(prod_a,fmt['FA'])+ceil_shift(prod_c,L['shc']+extra),name+'/state')
            if readout:acc=checked(acc+state,name+'/logit_accumulator')
        if not readout:
            summed=checked(state*fmt['K'],name+'/pole_sum')
            checked(ceil_shift(summed,fmt['FS']-fmt['FZ'])+max(abs(int(v)) for v in np.asarray(L['h']).flat),name+'/pre_saturation_z')
        records[name]={'projection_abs_bound':ub,'state_abs_bound':state,'logit_accumulator_abs_bound':acc,'dot_product_abs_bound':dot}
    layer('prefix',Q['layers'][0],255)
    layer('sampled_layer2',Q['layers'][1],1)
    layer('meanfield_layer2',Q['layers'][1],255,8)
    layer('sampled_readout',Q['ro'],1,readout=True)
    layer('meanfield_readout',Q['ro'],255,8,readout=True)
    return records

def calibration_stage():
    model, beta, ckpt_sha = load_fixed_checkpoint()
    X, y, idx = load_calibration_only(get_dataset_path(cfg.dataset_key), cfg.val_frac)
    fp = arrays_fingerprint(X, y); key = run_key_of(ckpt_sha, fp)
    out = OUTPUT_ROOT / ('validation_' + key)
    out.mkdir(parents=True, exist_ok=True); log.open(out / 'log.txt')
    if (out / 'format_freeze.json').exists():
        freeze=json.loads((out/'format_freeze.json').read_text())
        p=verify_freeze(freeze,ckpt_sha,ORACLE_MD5,REF_MD5)
        if p['validation_fingerprint']!=fp or p['validation_indices']!=idx.tolist() or p['ckpt_rel']!=cfg.ckpt_rel or p['beta']!=beta:
            raise ValueError('Existing freeze differs from source/split')
        Q=quantize(fold_params(model.state_dict(),beta),p['format'])
        if quantized_tables_digest(Q)!=p['quantized_tables_sha256']:raise ValueError('Existing freeze integer tables differ')
        check_int64_contract(Q);validate_frozen_widths(p['validation_maxabs'],p['widths'])
        indices_path=out/'validation_indices.npy'
        if indices_path.exists() and not np.array_equal(np.load(indices_path,allow_pickle=False),idx):raise ValueError('Validation indices differ')
        if not indices_path.exists():save_npy_atomic(indices_path,idx)
        write_json_atomic({'freeze_path':str(out/'format_freeze.json'),'freeze_sha256':freeze['sha256']},OUTPUT_ROOT/'calibration_completed.json')
        log('[RESUME] Verified immutable calibration freeze; no reselection or test access.')
        return
    log(f'[CALIBRATION] validation only: {len(y)} inputs; no test arrays loaded')
    Fp, cs, fl = fold_gate(model, beta, X, y)
    rows = sweep_mf(Fp, X.numpy().astype(np.int64), cs, y)
    write_json_atomic({'selection_split': 'training_validation_seed0', 'rows': rows}, out / 'candidate_grid.json')
    chosen = choose_format(rows, fl['mf_acc'])
    fmt = dict(cfg.fmt, FA=chosen['FA'], FS=chosen['FS'], FZ=chosen['FZ'])
    rng = np.random.RandomState(7); sub = np.sort(rng.choice(len(y), min(cfg.subset_n, len(y)), replace=False))
    nb_rows, NB = sampled_subset_check(model, Fp, fmt, X.numpy().astype(np.int64), X, y, beta, sub)
    if NB != cfg.fmt['NB']: raise RuntimeError('Noise LUT width was not the prespecified fixed width.')
    Q=quantize(Fp,fmt);arithmetic_safety=check_int64_contract(Q);table_digest=quantized_tables_digest(Q)
    Q=to_dev(Q,device);thresholds_selftest(Q)
    od = out / 'validation_oracle'; od.mkdir(exist_ok=True)
    vc, vr, md, ml, mx = oracle_full(Q, X.numpy().astype(np.int64), y, od)
    widths = widths_from_calibration(mx); validate_frozen_widths(mx, widths)
    payload = {'selection_split': 'training_validation_seed0', 'validation_fraction': cfg.val_frac,
               'validation_indices': idx.tolist(), 'validation_fingerprint': fp,
               'ckpt_rel': cfg.ckpt_rel, 'ckpt_sha256': ckpt_sha, 'beta': beta,
               'format': fmt, 'widths': widths, 'validation_maxabs': mx,
               'quantized_tables_sha256':table_digest,'int64_contract':arithmetic_safety,'environment':environment_record(),
               'float_validation': fl, 'candidate_grid': rows, 'chosen': chosen, 'noise_lut_sensitivity': nb_rows,
               'oracle_md5': ORACLE_MD5, 'ref_md5': REF_MD5, 'exporter_sha256': sha256_file(Path(__file__)),
               'created': datetime.now().isoformat(), 'historical_test_exposure': HISTORICAL_TEST_EXPOSURE,
               'checkpoint_selection': 'Fixed archived seed999 checkpoint retained for engineering repair; original representative-seed selection history is not claimed independent.',
               'width_scope': 'Validation observed maxima with fourfold headroom; no universal overflow proof.'}
    freeze = seal_freeze(payload); write_json_atomic(freeze, out / 'format_freeze.json')
    save_npy_atomic(out / 'validation_indices.npy', idx)
    log(f'[FROZEN] {(out / "format_freeze.json").relative_to(PROJECT).as_posix()}; SHA256={freeze["sha256"]}')
    write_json_atomic({'freeze_path':str(out/'format_freeze.json'),'freeze_sha256':freeze['sha256']},OUTPUT_ROOT/'calibration_completed.json')
    print(HISTORICAL_TEST_EXPOSURE, flush=True)


def evaluation_stage(freeze_path):
    # Freeze identity and arithmetic are validated BEFORE requesting test arrays.
    record = json.loads(Path(freeze_path).read_text())
    if record['payload']['ckpt_rel']!=cfg.ckpt_rel:raise ValueError('Freeze checkpoint path differs')
    model, beta, ckpt_sha = load_fixed_checkpoint()
    p = verify_freeze(record, ckpt_sha, ORACLE_MD5, REF_MD5)
    if beta != p['beta']: raise ValueError('Checkpoint soma slope differs from the freeze.')
    fmt = p['format']; widths = p['widths']
    Q=quantize(fold_params(model.state_dict(),beta),fmt)
    if quantized_tables_digest(Q)!=p['quantized_tables_sha256']:raise ValueError('Frozen integer tables differ; test not loaded')
    check_int64_contract(Q);Q=to_dev(Q,device);thresholds_selftest(Q)
    with np.load(get_dataset_path(cfg.dataset_key), allow_pickle=False) as z:
        X = _u8(z['Xte']); y = np.asarray(z['yte'], dtype=np.int64)
    fp = arrays_fingerprint(X, y)
    key = hashlib.sha256((record['sha256'] + fp).encode()).hexdigest()[:8]
    out = OUTPUT_ROOT / ('evaluation_' + key)
    out.mkdir(parents=True, exist_ok=True); log.open(out / 'log.txt')
    log('[EVALUATION] Frozen validation format; all test outcomes are descriptive reevaluation.')
    print(HISTORICAL_TEST_EXPOSURE, flush=True)
    Fp, cs, fl = fold_gate(model, beta, X, y)
    vc, vr, md, ml, mx = oracle_full(Q, X.numpy().astype(np.int64), y, out)
    # Any width failure is retained as a failure; test maxima never change the format.
    try: validate_frozen_widths(mx, widths)
    except RuntimeError as e:
        write_json_atomic({'status':'failed_frozen_width_gate','reason':str(e),'test_maxabs':mx,
                           'freeze_sha256':record['sha256']}, out/'evaluation_failure.json')
        raise
    cert = certificate_summary(Q, vc, vr, y); nf = min(cfg.n_fixture, len(y))
    z1f, _, _ = prefix_z1(Q, X[:nf].numpy().astype(np.int64), device)
    meta = {'key': key, 'ckpt': str(PROJECT/cfg.ckpt_rel), 'ckpt_sha256':ckpt_sha, 'data_fp':fp,
            'freeze_sha256':record['sha256'], 'selection_split':p['selection_split'],'quantized_tables_sha256':p['quantized_tables_sha256'],
            'historical_test_exposure':HISTORICAL_TEST_EXPOSURE, 'width_selection':'frozen_validation_only',
            'package_abi':'integer-events32-v1',
            'host_compatibility':'Requires the accelerator matching this integer model and register interface.'}
    Xnp = X.numpy().astype(np.int64)
    pk = write_package(Q,Xnp,y,vc,vr,md,ml,cert,None,mx,widths,out/'full',len(y),meta)
    write_package(Q,Xnp,y,vc,vr,md,ml,cert,z1f.cpu().numpy(),mx,widths,out/'fixture',nf,dict(meta,fixture_of=len(y)))
    summary={'run_id':RUN_ID,'key':key,'n_test':len(y),'float':fl,
             'quantized':{'mf_acc':float((md==y).mean()),'mf_agree_with_float':float((md==cs).mean()),'n1_acc':cert['n1_acc']},
             'certificate':{k:v for k,v in cert.items() if k not in ('seq_dec','used','fix_dec','ref_dec')},
             'format':fmt,'widths':widths,'test_maxabs':mx,'freeze_sha256':record['sha256'],
             'historical_test_exposure':HISTORICAL_TEST_EXPOSURE,
             'scope':'Post-freeze engineering evaluation; historical test exposure is disclosed.',
             'package_full':str(out/'full'),'package_fixture':str(out/'fixture')}
    write_json_atomic(summary,out/'summary.json')
    write_json_atomic({'status':'complete','evaluation_dir':str(out),'summary':str(out/'summary.json'),
                       'freeze':str(Path(freeze_path).resolve()),'freeze_sha256':record['sha256']},OUTPUT_ROOT/'run_completed.json')
    print(json.dumps({'status':'complete','output':out.relative_to(PROJECT).as_posix(),'frozen_format':fmt,
                      'mf_acc':summary['quantized']['mf_acc'],'n1_acc':cert['n1_acc']},indent=2),flush=True)


def main():
    global OUTPUT_ROOT, device
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('calibrate','evaluate'),default='calibrate')
    parser.add_argument('--freeze', help='Repository-relative validation freeze path')
    parser.add_argument('--dataset', default=cfg.dataset_confirmed_rel, help='Repository-relative SHD NPZ cache')
    parser.add_argument('--output', default='outputs/hardware/quantization', help='Repository-relative output directory')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=device)
    args=parser.parse_args()
    cfg.dataset_confirmed_rel = repository_path(args.dataset).relative_to(PROJECT).as_posix()
    OUTPUT_ROOT = repository_path(args.output)
    device = args.device
    cfg.device = device
    if args.stage=='calibrate':
        if args.freeze: parser.error('--freeze is used only for evaluation')
        calibration_stage()
    else:
        if not args.freeze: parser.error('Evaluation requires --freeze; no automatic selection is allowed')
        evaluation_stage(repository_path(args.freeze))

if __name__=='__main__':
    main()
