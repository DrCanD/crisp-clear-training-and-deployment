"""CRISP and LIF on DVS128 Gesture with the archived vision recipe."""
import copy
import math
import time
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from .paths import resolve_path
from .datasets import _strat, data_fingerprint
from .training import set_all_seeds, save_torch_atomic
from .models.crisp import DendriticSSM, DendStochLayer, DendStochAudioNet, anneal, evaluate
from .models.lif import LIFNet

class LinearConvStem128(nn.Module):
    """Fully LINEAR spatial front-end: Conv(no bias)+AvgPool only, no ReLU/GroupNorm/MaxPool.
    Conv(Σ_frames)=Σ Conv(frames) => the whole Conv∘SSM path is LTI and Δt-transferable.
    Batch normalization and dropout follow the archived training configuration."""

    def __init__(self, in_ch, c, out_hw, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1, bias=False), nn.AvgPool2d(2), nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.AvgPool2d(2), nn.Conv2d(32, c, 3, padding=1, bias=False), nn.AvgPool2d(2), nn.AdaptiveAvgPool2d((out_hw, out_hw)))
        self.drop = nn.Dropout(dropout)
        self.F = c * out_hw * out_hw

    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.drop(self.net(x.reshape(B * T, C, H, W)).reshape(B, T, -1))

class DendStochVisionNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        if not cfg.linear_stem:
            raise ValueError("The study uses the linear convolutional stem")
        Stem = LinearConvStem128
        self.stem = Stem(cfg.in_polarity, cfg.stem_ch, cfg.stem_out_hw, cfg.stem_dropout)
        self.readout_pool = getattr(cfg, 'readout_pool', 'max')
        zb = getattr(cfg, 'zoh_b', False)
        nt = getattr(cfg, 'norm_type', 'layernorm')
        ce = getattr(cfg, 'ct_exact', False)
        dims = [self.stem.F] + [cfg.n_hidden] * cfg.n_layers
        self.layers = nn.ModuleList([DendStochLayer(dims[i], dims[i + 1], cfg.k_modes, cfg.complex_poles, cfg.hidden_dropout, zb, nt, ce) for i in range(cfg.n_layers)])
        self.readout = nn.Linear(cfg.n_hidden, cfg.n_classes)
        self.ro = DendriticSSM(cfg.n_classes, 1, complex_poles=False, zoh_b=zb, ct_exact=ce)

    def pool(self, x):
        return x.mean(dim=1) if self.readout_pool == 'mean' else x.max(dim=1).values

    def forward(self, x, mode, beta):
        s = self.stem(x)
        rates = []
        gfs = []
        for layer in self.layers:
            s, r, gf = layer(s, mode, beta)
            rates.append(r)
            gfs.append(gf)
        logits = self.pool(self.ro(self.readout(s)))
        return (logits, torch.stack(rates).mean(), torch.stack(gfs).mean())

    def forward_masked(self, x, beta, sample_mask):
        s = self.stem(x)
        for i, layer in enumerate(self.layers):
            V = layer.norm_apply(layer.dend(layer.proj(s)))
            p = torch.sigmoid(beta * (V - layer.theta))
            s = torch.bernoulli(p.clamp(1e-06, 1 - 1e-06)) if sample_mask[i] else p
        return self.pool(self.ro(self.readout(s)))

    def set_scan(self, fn):
        for m in self.modules():
            if isinstance(m, DendriticSSM):
                m._scan = fn

    def set_dt(self, dt):
        for m in self.modules():
            if isinstance(m, DendriticSSM):
                m.dt = float(dt)

def augment_vision(xb, cfg):
    B, T, P, H, W = xb.shape
    d = xb.device
    tx = (torch.rand(B, device=d) * 2 - 1) * cfg.aug_translate
    ty = (torch.rand(B, device=d) * 2 - 1) * cfg.aug_translate
    sc = 1.0 + (torch.rand(B, device=d) * 2 - 1) * cfg.aug_scale
    theta = torch.zeros(B, 2, 3, device=d)
    theta[:, 0, 0] = sc
    theta[:, 1, 1] = sc
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, P, H, W), align_corners=False)[:, None].expand(B, T, H, W, 2).reshape(B * T, H, W, 2)
    xb = F.grid_sample(xb.reshape(B * T, P, H, W), grid, align_corners=False, mode='nearest').reshape(B, T, P, H, W)
    sh = torch.randint(-cfg.aug_tshift, cfg.aug_tshift + 1, (B,), device=d)
    idx = (torch.arange(T, device=d)[None, :] - sh[:, None]) % T
    xb = torch.gather(xb, 1, idx.view(B, T, 1, 1, 1).expand(B, T, P, H, W))
    return xb * (torch.rand_like(xb) > cfg.aug_event_drop).float()

@dataclass
class VisArch:
    """Everything DendStochVisionNet (reference) reads from its cfg; stored in every CRISP checkpoint."""
    n_classes: int = 11
    in_polarity: int = 2
    stem_ch: int = 64
    stem_out_hw: int = 4
    stem_dropout: float = 0.2
    linear_stem: bool = True
    complex_poles: bool = False
    zoh_b: bool = True
    norm_type: str = 'batchnorm'
    readout_pool: str = 'mean'
    ct_exact: bool = True
    n_layers: int = 2
    k_modes: int = 8
    n_hidden: int = 256
    hidden_dropout: float = 0.2
    beta_start: float = 1.0
    beta_end: float = 5.0
    anneal_frac: float = 0.7
    reg_coef: float = 0.02
    rate_target: float = 0.1

@dataclass
class LifArch:
    """Stem fields + everything LIFNet (LIF baseline) reads from its cfg (n_in is set to the stem's feature count)."""
    n_classes: int = 11
    in_polarity: int = 2
    stem_ch: int = 64
    stem_out_hw: int = 4
    stem_dropout: float = 0.2
    linear_stem: bool = True
    n_layers: int = 2
    n_hidden: int = 256
    hidden_dropout: float = 0.2
    norm_type: str = 'batchnorm'
    surrogate_beta: float = 5.0
    learnable_tau: bool = True
    tau_init_alpha: float = 0.7
    theta_init: float = 0.0
    reset: str = 'hard'
    readout_type: str = 'li'
    readout_pool: str = 'mean'
    readout_learnable_tau: bool = True
    reg_coef: float = 0.02
    rate_target: float = 0.1
    reg_scope: str = 'hidden'

class HeavisideSG(torch.autograd.Function):
    """Deterministic threshold 1[u > 0] (audio implementation's 'heaviside' deployment mode; only its forward is used here)."""

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        u, = ctx.saved_tensors
        s = torch.sigmoid(u)
        return g * s * (1 - s)

class RuleLayer(DendStochLayer):
    """Reference layer + the deterministic-threshold deployment mode of audio implementation. Every other mode goes to the
    reference forward unchanged (gated bitwise)."""

    def forward(self, s_in, mode, beta):
        if mode != 'heaviside':
            return super().forward(s_in, mode, beta)
        V = self.norm_apply(self.dend(self.proj(s_in)))
        p = torch.sigmoid(beta * (V - self.theta))
        s = self.drop(HeavisideSG.apply(beta * (V - self.theta)))
        gf = ((p > 0.2) & (p < 0.8)).float().mean()
        return (s, p.mean(), gf)

class CoreView(DendStochAudioNet):
    """The CRISP part of the DVS network (hidden layers + readout) seen as the reference network whose input is
    the stem's feature sequence. Shares the modules (no copy); forward = reference DendStochAudioNet.forward. The stem is
    linear and deterministic, so the certificate machinery of audio implementation applies to this view unchanged."""

    def __init__(self, net):
        nn.Module.__init__(self)
        self.readout_pool = net.readout_pool
        self.layers = net.layers
        self.readout = net.readout
        self.ro = net.ro

class LIFVisionNet(nn.Module):
    """reference linear stem + LIF baseline network (LIFNet, verbatim) on the stem's features. forward(x) -> (logits, rate) as
    LIFNet. `layers` / `ro` are views of the LIF network (the LIF baseline ZOH hooks address them)."""

    def __init__(self, larch):
        super().__init__()
        self.stem = LinearConvStem128(larch.in_polarity, larch.stem_ch, larch.stem_out_hw, larch.stem_dropout)
        core_cfg = SimpleNamespace(**asdict(larch))
        core_cfg.n_in = self.stem.F
        self.core = LIFNet(core_cfg, larch.n_classes)

    @property
    def layers(self):
        return self.core.layers

    @property
    def ro(self):
        return self.core.ro

    def forward(self, x):
        return self.core(self.stem(x))

    def set_dt(self, dt):
        self.core.set_dt(dt)

def pool5(X, f, chunk=128):
    """Coarse-step frames: mean of f consecutive frames of [N, T, P, H, W] (same values as reference's temporal_pool).
    uint8 counts -> k/f exact in fp16 for f in {1, 2, 4, 8}; float32 otherwise. Chunked over samples."""
    N, T = X.shape[:2]
    Tn = T // f * f
    odt = torch.float16 if X.dtype == torch.uint8 and f in (1, 2, 4, 8) else torch.float32
    out = torch.empty((N, T // f) + tuple(X.shape[2:]), dtype=odt)
    for i in range(0, N, chunk):
        o = X[i:i + chunk, :Tn].float().reshape(-1, T // f, f, *X.shape[2:]).mean(2)
        oc = o.to(odt)
        if odt == torch.float16:
            assert torch.equal(oc.float(), o), 'pooled frames not exact in fp16'
        out[i:i + chunk] = oc
    return out

def pool_batch5(xb, f):
    """Mean of f consecutive frames of a [B, T, P, H, W] batch (training path: augmented at T64, then pooled)."""
    B, T = xb.shape[:2]
    Tn = T // f * f
    return xb[:, :Tn].reshape(B, T // f, f, *xb.shape[2:]).mean(2)

def macro_f1(yp, yt):
    f1s = []
    for cc in np.unique(yt):
        tp = np.sum((yp == cc) & (yt == cc))
        fp = np.sum((yp == cc) & (yt != cc))
        fn = np.sum((yp != cc) & (yt == cc))
        pr = tp / (tp + fp + 1e-09)
        rc = tp / (tp + fn + 1e-09)
        f1s.append(2 * pr * rc / (pr + rc + 1e-09))
    return float(np.mean(f1s))

def build_crisp(varch, device="cpu"):
    m = DendStochVisionNet(varch).to(device)
    for layer in m.layers:
        layer.__class__ = RuleLayer
    return m

def build_lif(larch, device="cpu"):
    return LIFVisionNet(larch).to(device)

def load_model(ckpt, device="cpu"):
    blob = torch.load(resolve_path(ckpt), map_location='cpu', weights_only=False)
    if blob['family'] == 'crisp':
        a = VisArch(**blob['cfg'])
        m = build_crisp(a, device)
    else:
        a = LifArch(**blob['cfg'])
        m = build_lif(a, device)
    m.load_state_dict(blob['state'], strict=True)
    m.eval()
    return (m, a, blob)

def load_dvs(path, expected_fingerprint=None):
    p = resolve_path(path)
    print(f"  [{time.strftime('%H:%M:%S')}] loading {"DVS128 Gesture"} <- {p} ({p.stat().st_size / 1000000.0:.0f} MB)...", flush=True)
    d = torch.load(p, map_location='cpu', weights_only=False)
    print(f'[DATA] keys={sorted(d.keys())}', flush=True)
    Xtr, Xte = (torch.as_tensor(d['Xtr']), torch.as_tensor(d['Xte']))
    ytr, yte = (torch.as_tensor(d['ytr']).long().contiguous(), torch.as_tensor(d['yte']).long().contiguous())
    del d
    for nm, X in (('Xtr', Xtr), ('Xte', Xte)):
        if X.dtype != torch.uint8 or X.ndim != 5 or X.shape[2] != 2:
            raise ValueError(f"{nm}: expected uint8 [N, T, {2}, H, W], got {X.dtype} {tuple(X.shape)}")
    ncls = int(ytr.max()) + 1
    if ncls != 11 or int(yte.max()) + 1 > 11:
        raise ValueError(f'{ncls} classes in the data, cfg says {11}')
    itr, iva = _strat(ytr.numpy(), 0.10, 0)
    itr, iva = (torch.from_numpy(itr), torch.from_numpy(iva))
    Xva, yva = (Xtr[iva].contiguous(), ytr[iva])
    Xtr, ytr = (Xtr[itr].contiguous(), ytr[itr])
    u, c = torch.unique(ytr, return_counts=True)
    print(f'[DATA] train {tuple(Xtr.shape)} val {tuple(Xva.shape)} test {tuple(Xte.shape)} classes={ncls} stored={Xtr.dtype} range[{int(Xtr.min())},{int(Xtr.max())}] | train per class {int(c.min())}-{int(c.max())} | validation: stratified {0.10:.0%} of train (seed 0)', flush=True)
    data = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    if expected_fingerprint and data_fingerprint(data) != expected_fingerprint:
        raise ValueError("Gesture cache differs from the configured array fingerprint")
    return data

class ArrLoader:
    """In-order (or per-epoch shuffled) batches cast to float32 on the device; labels stay on the CPU.
    subset: fixed row indices (in that order) into X / y, read without copying the array."""

    def __init__(self, X, y, bs, shuffle=False, seed=0, subset=None, device="cpu"):
        self.X, self.y, self.bs, self.shuffle, self.seed, self.epoch = (X, y, bs, shuffle, seed, 0)
        self.subset = subset
        self.device = torch.device(device)

    def __len__(self):
        return ((len(self.y) if self.subset is None else len(self.subset)) + self.bs - 1) // self.bs

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        if self.subset is not None:
            for i in range(0, len(self.subset), self.bs):
                j = self.subset[i:i + self.bs]
                yield (self.X[j].to(self.device).float(), self.y[j])
            return
        n = len(self.y)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed * 100003 + self.epoch)) if self.shuffle else None
        for i in range(0, n, self.bs):
            if idx is None:
                yield (self.X[i:i + self.bs].to(self.device).float(), self.y[i:i + self.bs])
            else:
                j = idx[i:i + self.bs]
                yield (self.X[j].to(self.device).float(), self.y[j])

@torch.no_grad()
def eval_crisp(model, loader, mode, beta, n_samples=1):
    """'graded' / 'sample' -> reference evaluate (reference; softmax average for n > 1); 'heaviside' -> RuleLayer."""
    if mode in ('graded', 'sample'):
        return evaluate(model, loader, mode, beta, n_samples=n_samples)
    model.eval()
    yp, yt, rates = ([], [], [])
    for xb, yb in loader:
        lg, mr, _ = model(xb.to(next(model.parameters()).device), 'heaviside', beta)
        yp.append(lg.argmax(1).cpu().numpy())
        yt.append(yb.numpy())
        rates.append(float(mr))
    yp = np.concatenate(yp)
    yt = np.concatenate(yt)
    return {'acc': float((yp == yt).mean()), 'macro_f1': macro_f1(yp, yt), 'firing_rate': float(np.mean(rates))}

@torch.no_grad()
def eval_lif(model, loader):
    """Same arithmetic as LIF baseline `evaluate` (argmax of the time-mean readout membrane)."""
    model.eval()
    yp, yt, rates = ([], [], [])
    for xb, yb in loader:
        lg, mr = model(xb.to(next(model.parameters()).device).float())
        yp.append(lg.argmax(1).cpu().numpy())
        yt.append(yb.numpy())
        rates.append(float(mr))
    yp = np.concatenate(yp)
    yt = np.concatenate(yt)
    return {'acc': float((yp == yt).mean()), 'macro_f1': macro_f1(yp, yt), 'firing_rate': float(np.mean(rates))}

def eval_all(family, model, te, beta=None, eval_sample_ns=(1, 5), eval_seed=2026):
    if family == 'lif':
        return {'deterministic': eval_lif(model, te)}
    out = {'graded': eval_crisp(model, te, 'graded', beta)}
    for n in eval_sample_ns:
        set_all_seeds(eval_seed + n)
        out[f'sampled_n{n}'] = eval_crisp(model, te, 'sample', beta, n_samples=n)
    out['heaviside'] = eval_crisp(model, te, 'heaviside', beta)
    return out

@dataclass
class VisionTrainConfig:
    n_classes: int = 11
    val_frac: float = 0.1
    stem: dict = field(default_factory=lambda: dict(in_polarity=2, stem_ch=64, stem_out_hw=4, stem_dropout=0.2, linear_stem=True))
    crisp: dict = field(default_factory=lambda: dict(complex_poles=False, zoh_b=True, norm_type='batchnorm', readout_pool='mean', ct_exact=True, n_layers=2, k_modes=8, n_hidden=256, hidden_dropout=0.2, beta_start=1.0, beta_end=5.0, anneal_frac=0.7, reg_coef=0.02, rate_target=0.1))
    lif: dict = field(default_factory=lambda: dict(n_layers=2, n_hidden=256, hidden_dropout=0.2, norm_type='batchnorm', surrogate_beta=5.0, learnable_tau=True, tau_init_alpha=0.7, theta_init=0.0, reset='hard', readout_type='li', readout_pool='mean', readout_learnable_tau=True, reg_coef=0.02, rate_target=0.1, reg_scope='hidden'))
    lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    batch_size: int = 32
    n_epochs: int = 100
    patience: int = 20
    schedule: str = 'cosine'
    aug: dict = field(default_factory=lambda: dict(aug_translate=0.08, aug_scale=0.1, aug_tshift=8, aug_event_drop=0.1))
    coarse_aug_domain: str = 'fine_then_pool'
    bn_recal_batches: int = 30
    bn_shuffle_seed: int = 0
    eval_sample_ns: list = field(default_factory=lambda: [1, 5])
    eval_seed: int = 2026
    eval_batch: int = 32
    tf32: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one(family, seed, f, Xtr, ytr, val_ld, cfg, checkpoint_dir):
    """One run; resume per epoch. Xtr is the T64 training set: each batch is augmented at T64 and pooled xf (f > 1)."""
    device = torch.device(cfg.device)
    ck_dir = resolve_path(checkpoint_dir)
    desc = f"{family}_gesture_factor{f}_seed{seed}"
    if cfg.schedule != "cosine":
        raise ValueError("The archived recipe uses the cosine schedule")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    T_model = int(Xtr.shape[1]) // f
    set_all_seeds(seed)
    arch = VisArch(n_classes=cfg.n_classes, **cfg.stem, **cfg.crisp) if family == 'crisp' else LifArch(n_classes=cfg.n_classes, **cfg.stem, **cfg.lif)
    model = build_crisp(arch, device) if family == 'crisp' else build_lif(arch, device)
    tr = ArrLoader(Xtr, ytr, cfg.batch_size, shuffle=True, seed=seed, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    n_params = sum((p.numel() for p in model.parameters()))
    aug = SimpleNamespace(**cfg.aug)
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / 'latest.pt'
    start, best_val, best_state, wait, ep_times = (0, -1.0, copy.deepcopy(model.state_dict()), 0, [])
    if ck.exists():
        try:
            c = torch.load(ck, map_location=device, weights_only=False)
            if c.get('done', False):
                model.load_state_dict(c['best_state'])
                model.eval()
                print(f'      [done] {desc}: from checkpoint', flush=True)
                return (c['train_info'], model)
            model.load_state_dict(c['model'])
            opt.load_state_dict(c['opt'])
            sched.load_state_dict(c['sched'])
            start = c['epoch'] + 1
            best_val = c['best_val']
            best_state = c['best_state']
            wait = c.get('wait', 0)
            ep_times = c.get('ep_times', [])
            torch.set_rng_state(c['rng_cpu'].cpu())
            if torch.cuda.is_available() and c.get('rng_cuda') is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in c['rng_cuda']])
            print(f'      [RESUME] {desc} from epoch {start} (best val {best_val:.4f})', flush=True)
        except Exception as e:
            raise RuntimeError("Existing checkpoint could not be restored; preserving it") from e
    b_end = arch.beta_end if family == 'crisp' else None
    print(f'      [model] {family.upper()} {n_params / 1000.0:.0f}K params | T{T_model}' + (f' (augmented at T{int(Xtr.shape[1])}, pooled x{f})' if f > 1 else ''), flush=True)
    nb = len(tr)
    pbar = tqdm(total=(cfg.n_epochs - start) * nb, desc=desc, leave=False, dynamic_ncols=True)
    epoch = max(start - 1, 0)
    for epoch in range(start, cfg.n_epochs):
        beta = anneal(epoch, cfg.n_epochs, arch.beta_start, arch.beta_end, arch.anneal_frac) if family == 'crisp' else None
        t0 = time.time()
        model.train()
        tr.set_epoch(epoch)
        seen = corr = 0
        loss_sum = 0.0
        n_skip = 0
        for xb, yb in tr:
            xb = augment_vision(xb, aug)
            yb = yb.to(device)
            if f > 1:
                xb = pool_batch5(xb, f)
            opt.zero_grad(set_to_none=True)
            if family == 'crisp':
                logits, mr, _ = model(xb, 'meanfield', beta)
            else:
                logits, mr = model(xb)
            loss = F.cross_entropy(logits, yb) + arch.reg_coef * (mr - arch.rate_target) ** 2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; preserving the last checkpoint")
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            seen += yb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
            pbar.update(1)
        sched.step()
        set_all_seeds(cfg.eval_seed + epoch)
        va_acc = (eval_crisp(model, val_ld, 'graded', beta) if family == 'crisp' else eval_lif(model, val_ld))['acc']
        if va_acc > best_val:
            best_val = va_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        dt_ep = time.time() - t0
        ep_times.append(dt_ep)
        pbar.set_postfix_str(f'ep {epoch + 1}/{cfg.n_epochs} | train {corr / max(seen, 1):.3f} | val {va_acc:.4f} | best {best_val:.4f}')
        if (epoch + 1) % 10 == 0 or epoch == cfg.n_epochs - 1:
            pbar.write(f'      ep {epoch + 1:>3}/{cfg.n_epochs} | train {corr / max(seen, 1):.3f} | val {va_acc:.4f} | best {best_val:.4f}')
        save_torch_atomic({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'best_state': best_state, 'wait': wait, 'ep_times': ep_times, 'done': False, 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, ck)
        if wait >= cfg.patience:
            pbar.write(f'      [stop] no validation gain for {cfg.patience} epochs (epoch {epoch + 1})')
            break
    pbar.close()
    model.load_state_dict(best_state)
    model.eval()
    info = {'family': family, 'seed': seed, 'factor': f, 'T': T_model, 'best_val': best_val, 'stopped_ep': epoch + 1, 'n_params': n_params, 'mean_epoch_s': float(np.mean(ep_times)) if ep_times else None, 'n_epochs_run': len(ep_times)}
    save_torch_atomic({'state': best_state, 'cfg': asdict(arch), 'family': family, 'info': info, 'recipe': {k: getattr(cfg, k) for k in ('lr', 'weight_decay', 'grad_clip', 'batch_size', 'n_epochs', 'patience', 'schedule', 'aug', 'coarse_aug_domain')}}, ck_dir / 'best_model.pt')
    save_torch_atomic({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'best_state': best_state, 'wait': wait, 'ep_times': ep_times, 'done': True, 'train_info': info}, ck)
    return (info, model)


def transfer_crisp(model, factor, beta, coarse_loader, fine_loader, calibration_loader,
                   eval_seed=2026, n_batches=30):
    """Evaluate native, unchanged-step and retimed CRISP inference."""
    from .models.crisp import recalibrate_bn
    model.set_dt(1.)
    result = {'coarse_native': eval_crisp(model, coarse_loader, 'graded', beta)['acc'],
              'naive_same_dt': eval_crisp(model, fine_loader, 'graded', beta)['acc']}
    model.set_dt(1. / factor)
    result['zoh_fine'] = eval_crisp(model, fine_loader, 'graded', beta)['acc']
    set_all_seeds(eval_seed)
    result['zoh_fine_sampled_n1'] = eval_crisp(model, fine_loader, 'sample', beta, 1)['acc']
    calibrated = recalibrate_bn(model, calibration_loader, beta, n_batches)
    result['zoh_fine_bn_reestimated'] = eval_crisp(calibrated, fine_loader, 'graded', beta)['acc']
    model.set_dt(1.)
    return result


@torch.no_grad()
def _certificate_votes(model, loader, beta, seed, draws=256, chunk=8):
    """Preserve the archived eight-draw batching and uninterrupted reference stream."""
    if draws % chunk:
        raise ValueError('Draw count must be divisible by the archived draw chunk')
    model.eval()
    cert_parts, reference_parts = [], []
    for index, (xb, _) in enumerate(tqdm(loader, desc='certificate batches')):
        features = model.stem(xb)
        first = model.layers[0]
        voltage = first.norm_apply(first.dend(first.proj(features)))
        probability = torch.sigmoid(beta * (voltage - first.theta)).clamp(1e-6, 1 - 1e-6)
        set_all_seeds(seed + index)
        streams = []
        for _ in range(2):
            blocks = []
            for _ in range(draws // chunk):
                states = torch.bernoulli(probability.repeat(chunk, 1, 1))
                for layer in model.layers[1:]:
                    states, _, _ = layer(states, 'sample', beta)
                logits = model.pool(model.ro(model.readout(states))).view(chunk, len(xb), -1)
                blocks.append(logits.argmax(-1).cpu().numpy())
            streams.append(np.concatenate(blocks).T)
        cert_parts.append(streams[0])
        reference_parts.append(streams[1])
    return np.concatenate(cert_parts), np.concatenate(reference_parts)


def certificate_evaluation(model, loader, labels, beta, seed, draws=256,
                           alpha=.001, looks=(16, 32, 64, 128, 256)):
    from .certification import fixed_predict, sequential_predict
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    votes, reference_votes = _certificate_votes(model, loader, beta, seed, draws)
    reference = fixed_predict(reference_votes, 11, alpha)
    fixed = fixed_predict(votes, 11, alpha)
    sequential, used = sequential_predict(votes, 11, looks, alpha)
    labels = np.asarray(labels)
    result = {}
    for name, decisions in (('fixed', fixed), ('sequential', sequential)):
        reference_resolved = reference >= 0
        both = (decisions >= 0) & reference_resolved
        result[name] = {
            'certified_accuracy': float(np.mean(decisions == labels)),
            'abstention_rate': float(np.mean(decisions < 0)),
            'accepted': int(np.sum(decisions >= 0)),
            'reference_resolved': int(np.sum(reference_resolved)),
            'reference_comparisons_both_accepted': int(np.sum(both)),
            'reference_disagreements': int(np.sum(both & (decisions != reference))),
            'mean_draws': float(np.mean(used)) if name == 'sequential' else float(draws),
        }
    return result, {'votes': votes, 'reference_votes': reference_votes,
                    'labels': labels, 'fixed': fixed, 'sequential': sequential, 'draws_used': used}


def run(config):
    """Train and evaluate both gesture models at native and coarser time steps."""
    import gc
    import json
    import os
    from .models.lif import transfer_evaluation
    training_options = dict(config.get('training', {}), device=config.get('device', 'cpu'))
    cfg = VisionTrainConfig(**training_options)
    dataset = config['dataset']
    data = load_dvs(dataset['path'], dataset.get('fingerprint'))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    factors = [1] + list(config.get('factors', [2, 4]))
    families = config.get('families', ['crisp', 'lif'])
    seeds = config.get('seeds', [42, 123, 999])
    if any(name not in ('crisp', 'lif') for name in families):
        raise ValueError('Gesture models are crisp and lif')
    if len(set(factors)) != len(factors) or any(not isinstance(f, int) or f < 1 for f in factors):
        raise ValueError('Time-step factors must be unique positive integers')
    output = resolve_path(config.get('output', 'outputs/dvs_gesture'))
    output.mkdir(parents=True, exist_ok=True)
    identity = {'configuration': config, 'data_fingerprint': data_fingerprint(data)}
    frozen = output / 'configuration.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('Existing output belongs to a different configuration or dataset')
    if not frozen.exists():
        frozen.write_text(json.dumps(identity, indent=2) + '\n')
    loaders = {'validation': {}, 'test': {}}
    for factor in factors:
        xv = Xva if factor == 1 else pool5(Xva, factor)
        xt = Xte if factor == 1 else pool5(Xte, factor)
        loaders['validation'][factor] = ArrLoader(xv, yva, cfg.eval_batch, device=cfg.device)
        loaders['test'][factor] = ArrLoader(xt, yte, cfg.eval_batch, device=cfg.device)
    subset = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(cfg.bn_shuffle_seed))[:cfg.bn_recal_batches * cfg.batch_size]
    calibration = ArrLoader(Xtr, ytr, cfg.batch_size, subset=subset, device=cfg.device)
    certificate = config.get('certificate', {})
    all_results = []
    total = len(families) * len(factors) * len(seeds)
    for family in families:
        for factor in factors:
            for seed in seeds:
                label = f'{family}_factor{factor}_seed{seed}'
                result_file = output / f'{label}.json'
                print(f'[GESTURE {len(all_results) + 1}/{total}] {label}', flush=True)
                if result_file.exists():
                    all_results.append(json.loads(result_file.read_text()))
                    continue
                checkpoint_dir = output / 'checkpoints' / label
                info, model = train_one(family, seed, factor, Xtr, ytr,
                    loaders['validation'][factor], cfg, checkpoint_dir)
                beta = cfg.crisp['beta_end'] if family == 'crisp' else None
                result = dict(family=family, factor=factor, seed=seed, training=info,
                    checkpoint=(checkpoint_dir / 'best_model.pt').relative_to(resolve_path('.')).as_posix(),
                    test=eval_all(family, model, loaders['test'][factor], beta,
                                  cfg.eval_sample_ns, cfg.eval_seed))
                if factor > 1:
                    if family == 'crisp':
                        result['transfer'] = transfer_crisp(model, factor, beta,
                            loaders['test'][factor], loaders['test'][1], calibration,
                            cfg.eval_seed, cfg.bn_recal_batches)
                    else:
                        result['transfer'] = transfer_evaluation(model, factor,
                            loaders['test'][factor], loaders['test'][1], calibration, cfg.bn_recal_batches)
                if family == 'crisp' and certificate.get('enabled', True):
                    model.set_dt(1. / factor)
                    seed_key = f'factor{factor}_seed{seed}'
                    draw_seed = certificate['draw_seeds'][seed_key]
                    metrics, arrays = certificate_evaluation(model, loaders['test'][1], yte.numpy(),
                        beta, draw_seed, int(certificate.get('draws', 256)),
                        float(certificate.get('alpha', .001)), tuple(certificate.get('looks', [16, 32, 64, 128, 256])))
                    result['certificate'] = metrics
                    np.savez_compressed(output / f'{label}_votes.npz', **arrays)
                partial = result_file.with_suffix('.json.partial')
                partial.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                os.replace(partial, result_file)
                all_results.append(result)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summary = output / 'results.json'
    summary.write_text(json.dumps(all_results, indent=2, allow_nan=False) + '\n')
    return {'runs': len(all_results), 'results': summary.relative_to(resolve_path('.')).as_posix()}
