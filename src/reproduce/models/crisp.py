"""CRISP: continuous-time reset-free independent-sampling perceptron.

The scan, soma, readout, augmentation and inference arithmetic are extracted
from the archived implementation. State-dictionary keys are preserved.
"""
import copy
import math
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..paths import resolve_path

@dataclass
class Config:
    dataset: str = 'ssc_T100_700'
    n_in: int = 700
    n_classes: int = 35
    val_frac: float = 0.1
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
    eval_sample_ns: list = field(default_factory=lambda: [1, 5])
    lr: float = 0.0015
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    batch_size: int = 128
    n_epochs: int = 100
    patience: int = 20
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    dt_factor: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def _shift(X, d, fill):
    ps = list(X.shape)
    ps[1] = d
    block = torch.full(ps, fill, dtype=X.dtype, device=X.device)
    return torch.cat([block, X[:, :X.shape[1] - d]], dim=1)

def parallel_scan(a, x):
    T = x.shape[1]
    A = a.reshape((1, 1) + tuple(a.shape)).expand_as(x).contiguous()
    H = x.clone()
    d = 1
    while d < T:
        H = A * _shift(H, d, 0.0) + H
        A = A * _shift(A, d, 1.0)
        d *= 2
    return H

def sequential_scan(a, x):
    B, T = (x.shape[0], x.shape[1])
    h = torch.zeros((B,) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)
    outs = []
    for t in range(T):
        h = a * h + x[:, t]
        outs.append(h)
    return torch.stack(outs, 1)

class DendriticSSM(nn.Module):

    def __init__(self, C, K, complex_poles=False, zoh_b=False, ct_exact=False):
        super().__init__()
        self.C, self.K, self.cplx, self.zoh_b, self.ct_exact = (C, K, complex_poles, zoh_b, ct_exact)
        self._scan = parallel_scan
        self.dt = 1.0
        if ct_exact:
            self.raw_lam = nn.Parameter(torch.log(torch.linspace(0.05, 2.5, K)).repeat(C, 1) + 0.01 * torch.randn(C, K))
            self.mix = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
        elif complex_poles:
            self.r_raw = nn.Parameter(torch.linspace(2.0, 4.0, K).repeat(C, 1) + 0.05 * torch.randn(C, K))
            self.theta_p = nn.Parameter(torch.linspace(0.0, math.pi, K).repeat(C, 1) + 0.05 * torch.randn(C, K))
            self.mix_re = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
            self.mix_im = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
        else:
            self.a_raw = nn.Parameter(torch.linspace(-2.5, 2.5, K).repeat(C, 1) + 0.01 * torch.randn(C, K))
            self.mix = nn.Parameter(torch.randn(C, K) / math.sqrt(K))

    def _a(self):
        if self.cplx:
            r = torch.sigmoid(self.r_raw)
            return torch.complex(r * torch.cos(self.theta_p), r * torch.sin(self.theta_p))
        return torch.sigmoid(self.a_raw)

    def forward(self, x):
        B, T, C = x.shape
        xd = x.unsqueeze(-1).expand(B, T, C, self.K)
        if self.ct_exact:
            lam = -torch.exp(self.raw_lam)
            a = torch.exp(lam * self.dt)
            Bd = torch.expm1(lam * self.dt) / lam
            return (self._scan(a, xd * Bd) * self.mix).sum(-1)
        if self.cplx:
            a = self._a().to(torch.complex64)
            xin = xd.to(torch.complex64)
            if self.zoh_b:
                xin = xin * (1 - a)
            h = self._scan(a, xin)
            return (h * torch.complex(self.mix_re, self.mix_im)).sum(-1).real
        a = self._a()
        xin = xd * (1 - a) if self.zoh_b else xd
        return (self._scan(a, xin) * self.mix).sum(-1)

class DendStochLayer(nn.Module):

    def __init__(self, d_in, d_out, k, complex_poles, dropout=0.0, zoh_b=False, norm_type='batchnorm', ct_exact=False):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.dend = DendriticSSM(d_out, k, complex_poles, zoh_b, ct_exact)
        self.norm_type = norm_type
        self.norm = nn.LayerNorm(d_out) if norm_type == 'layernorm' else nn.BatchNorm1d(d_out) if norm_type == 'batchnorm' else nn.Identity()
        self.theta = nn.Parameter(torch.zeros(d_out))
        self.drop = nn.Dropout(dropout)

    def norm_apply(self, V):
        if self.norm_type == 'batchnorm':
            B, T, d = V.shape
            return self.norm(V.reshape(B * T, d)).reshape(B, T, d)
        return self.norm(V)

    def forward(self, s_in, mode, beta):
        V = self.norm_apply(self.dend(self.proj(s_in)))
        p = torch.sigmoid(beta * (V - self.theta))
        p = self.drop(p)
        s = torch.bernoulli(p.clamp(1e-06, 1 - 1e-06)) if mode == 'sample' else p
        gf = ((p > 0.2) & (p < 0.8)).float().mean()
        return (s, p.mean(), gf)

class DendStochAudioNet(nn.Module):
    """Audio variant: no conv-stem, linear projection 700 -> n_hidden."""

    def __init__(self, cfg):
        super().__init__()
        self.readout_pool = cfg.readout_pool
        zb = cfg.zoh_b
        nt = cfg.norm_type
        ce = cfg.ct_exact
        dims = [cfg.n_in] + [cfg.n_hidden] * cfg.n_layers
        self.layers = nn.ModuleList([DendStochLayer(dims[i], dims[i + 1], cfg.k_modes, cfg.complex_poles, cfg.hidden_dropout, zb, nt, ce) for i in range(cfg.n_layers)])
        self.readout = nn.Linear(cfg.n_hidden, cfg.n_classes)
        self.ro = DendriticSSM(cfg.n_classes, 1, complex_poles=False, zoh_b=zb, ct_exact=ce)

    def pool(self, x):
        return x.mean(dim=1) if self.readout_pool == 'mean' else x.max(dim=1).values

    def forward(self, x, mode, beta):
        s = x
        rates = []
        gfs = []
        for layer in self.layers:
            s, r, gf = layer(s, mode, beta)
            rates.append(r)
            gfs.append(gf)
        logits = self.pool(self.ro(self.readout(s)))
        return (logits, torch.stack(rates).mean(), torch.stack(gfs).mean())

    def set_scan(self, fn):
        for m in self.modules():
            if isinstance(m, DendriticSSM):
                m._scan = fn

    def set_dt(self, dt):
        for m in self.modules():
            if isinstance(m, DendriticSSM):
                m.dt = float(dt)

def anneal(ep, n, v0, v1, fr):
    return v0 + (v1 - v0) * min(1.0, ep / max(1.0, fr * (n - 1)))

class HeavisideSG(torch.autograd.Function):
    """Forward: spike = 1[u > 0]. Backward: d spike/du = sigmoid(u)(1 - sigmoid(u))  (u = beta (V - theta),
    i.e. the same slope the soma probability has, so the surrogate matches the mean-field derivative)."""

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
    """Reference layer + two training/deployment modes. Every other mode goes to the reference forward."""

    def forward(self, s_in, mode, beta):
        if mode not in ('sample_ste', 'heaviside_sg', 'heaviside'):
            return super().forward(s_in, mode, beta)
        V = self.norm_apply(self.dend(self.proj(s_in)))
        p = torch.sigmoid(beta * (V - self.theta))
        if mode == 'sample_ste':
            p = self.drop(p)
            pc = p.clamp(1e-06, 1 - 1e-06)
            s = torch.bernoulli(pc.detach()) + (pc - pc.detach())
        else:
            s = self.drop(HeavisideSG.apply(beta * (V - self.theta)))
        gf = ((p > 0.2) & (p < 0.8)).float().mean()
        return (s, p.mean(), gf)

def to_rule_model(m):
    for layer in m.layers:
        layer.__class__ = RuleLayer
    return m

def build_model(config, device="cpu"):
    """Construct CRISP with the archived checkpoint parameter names."""
    return to_rule_model(DendStochAudioNet(config).to(device))


def load_ckpt(path, device="cpu", cfg_override=None):
    """Load a trusted checkpoint and filter archived non-model metadata."""
    blob = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    fields = Config.__dataclass_fields__
    values = {k: v for k, v in blob["cfg"].items() if k in fields}
    if cfg_override:
        values.update(cfg_override)
    config = Config(**values)
    model = build_model(config, device)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, config


@torch.no_grad()
def evaluate(model, loader, mode, beta, n_samples=1):
    model.eval()
    yp, yt, rates, gfs = ([], [], [], [])
    for xb, yb in loader:
        xb = xb.to(next(model.parameters()).device)
        yt.append(yb.numpy())
        if mode == 'sample':
            acc = 0.0
            for _ in range(n_samples):
                lg, mr, gf = model(xb, 'sample', beta)
                acc = acc + F.softmax(lg, 1)
            probs = acc / n_samples
        else:
            lg, mr, gf = model(xb, 'graded', beta)
            probs = F.softmax(lg, 1)
        rates.append(float(mr))
        gfs.append(float(gf))
        yp.append(probs.argmax(1).cpu().numpy())
    yp = np.concatenate(yp)
    yt = np.concatenate(yt)
    acc = float((yp == yt).mean())
    f1s = []
    for cc in np.unique(yt):
        tp = np.sum((yp == cc) & (yt == cc))
        fp = np.sum((yp == cc) & (yt != cc))
        fn = np.sum((yp != cc) & (yt == cc))
        pr = tp / (tp + fp + 1e-09)
        rc = tp / (tp + fn + 1e-09)
        f1s.append(2 * pr * rc / (pr + rc + 1e-09))
    return {'acc': acc, 'macro_f1': float(np.mean(f1s)), 'firing_rate': float(np.mean(rates)), 'graded_frac': float(np.mean(gfs))}

def recalibrate_bn(model, loader, beta, n_batches=30):
    m = copy.deepcopy(model)
    m.eval()
    has = False
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.train()
            mod.reset_running_stats()
            mod.momentum = None
            has = True
    if not has:
        return m
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= n_batches:
                break
            m(xb.to(next(m.parameters()).device), 'graded', beta)
    m.eval()
    return m

def analytic_variance(model, loader, beta, subset=512, M=64):
    """Predicted last-layer logit variance (autograd Jacobian
    of linear readout × Bernoulli p(1-p)) vs measured variance over M samplings
    of the last hidden layer. Mean-pool readout is linear => prediction is exact
    in expectation. Returns R² and Pearson of measured~predicted."""
    model.eval()
    preds, meas = ([], [])
    seen = 0
    for xb, yb in loader:
        if seen >= subset:
            break
        xb = xb.to(next(model.parameters()).device)
        with torch.no_grad():
            s = xb
            for layer in model.layers:
                V = layer.norm_apply(layer.dend(layer.proj(s)))
                s = torch.sigmoid(beta * (V - layer.theta))
            p_last = s
        s2 = p_last.detach().clone().requires_grad_(True)
        logits = model.pool(model.ro(model.readout(s2)))
        Bn, C = logits.shape
        pq = p_last * (1 - p_last)
        var_pred = logits.new_zeros(Bn, C)
        for c in range(C):
            g = torch.autograd.grad(logits[:, c].sum(), s2, retain_graph=c < C - 1)[0]
            var_pred[:, c] = (g * g * pq).sum(dim=(1, 2))
        with torch.no_grad():
            lg = torch.stack([model.pool(model.ro(model.readout(torch.bernoulli(p_last.clamp(1e-06, 1 - 1e-06))))) for _ in range(M)], 0)
            var_meas = lg.var(0)
        preds.append(var_pred.detach().cpu().numpy().ravel())
        meas.append(var_meas.cpu().numpy().ravel())
        seen += xb.shape[0]
    preds = np.concatenate(preds)
    meas = np.concatenate(meas)
    ss_res = np.sum((meas - preds) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    corr = float(np.corrcoef(preds, meas)[0, 1]) if preds.size > 1 else 0.0
    return {'r2': r2, 'pearson': corr, 'n_points': int(preds.size), 'pred_sample': preds[:1500].tolist(), 'meas_sample': meas[:1500].tolist()}
