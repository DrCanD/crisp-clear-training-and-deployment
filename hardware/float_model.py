"""Floating-point network used for the hardware quantization contract."""
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    dataset: str = "ssc_T100_700"
    n_in: int = 700
    n_classes: int = 35
    val_frac: float = 0.10       # fallback if cache lacks val split
    # dendritic-stochastic core (linear-until-spike)
    complex_poles: bool = False
    zoh_b: bool = True
    norm_type: str = "batchnorm"
    readout_pool: str = "mean"
    ct_exact: bool = True
    n_layers: int = 2
    k_modes: int = 8
    n_hidden: int = 256
    hidden_dropout: float = 0.2
    # soma / mean-field
    beta_start: float = 1.0
    beta_end: float = 5.0
    anneal_frac: float = 0.7
    reg_coef: float = 0.02
    rate_target: float = 0.10
    eval_sample_ns: list = field(default_factory=lambda: [1, 5])
    # optim
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 128
    n_epochs: int = 100
    patience: int = 20
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    # Δt-transfer
    dt_factor: int = 2          # coarse T50 -> deploy T100 (T25 too lossy for 35-class)
    speed_T_sweep: list = field(default_factory=lambda: [100, 500, 1000, 2000])
    speed_B: int = 64
    speed_N: int = 256
    # runtime
    use_amp: bool = False       # float32: fp16 NaN at high β
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def _shift(X, d, fill):
    ps = list(X.shape); ps[1] = d
    block = torch.full(ps, fill, dtype=X.dtype, device=X.device)
    return torch.cat([block, X[:, :X.shape[1] - d]], dim=1)

def parallel_scan(a, x):
    T = x.shape[1]
    A = a.reshape((1, 1) + tuple(a.shape)).expand_as(x).contiguous()
    H = x.clone(); d = 1
    while d < T:
        H = A * _shift(H, d, 0.0) + H
        A = A * _shift(A, d, 1.0)
        d *= 2
    return H

def sequential_scan(a, x):
    B, T = x.shape[0], x.shape[1]
    h = torch.zeros((B,) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)
    outs = []
    for t in range(T):
        h = a * h + x[:, t]; outs.append(h)
    return torch.stack(outs, 1)

class DendriticSSM(nn.Module):
    def __init__(self, C, K, complex_poles=False, zoh_b=False, ct_exact=False):
        super().__init__()
        self.C, self.K, self.cplx, self.zoh_b, self.ct_exact = C, K, complex_poles, zoh_b, ct_exact
        self._scan = parallel_scan; self.dt = 1.0
        if ct_exact:
            self.raw_lam = nn.Parameter(
                torch.log(torch.linspace(0.05, 2.5, K)).repeat(C, 1)
                + 0.01 * torch.randn(C, K))
            self.mix = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
        elif complex_poles:
            self.r_raw   = nn.Parameter(torch.linspace(2.0, 4.0, K).repeat(C, 1) + 0.05 * torch.randn(C, K))
            self.theta_p = nn.Parameter(torch.linspace(0.0, math.pi, K).repeat(C, 1) + 0.05 * torch.randn(C, K))
            self.mix_re  = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
            self.mix_im  = nn.Parameter(torch.randn(C, K) / math.sqrt(K))
        else:
            self.a_raw = nn.Parameter(torch.linspace(-2.5, 2.5, K).repeat(C, 1) + 0.01 * torch.randn(C, K))
            self.mix   = nn.Parameter(torch.randn(C, K) / math.sqrt(K))

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
            a   = torch.exp(lam * self.dt)
            Bd  = torch.expm1(lam * self.dt) / lam
            return (self._scan(a, xd * Bd) * self.mix).sum(-1)
        if self.cplx:
            a = self._a().to(torch.complex64)
            xin = xd.to(torch.complex64)
            if self.zoh_b: xin = xin * (1 - a)
            h = self._scan(a, xin)
            return (h * torch.complex(self.mix_re, self.mix_im)).sum(-1).real
        a = self._a()
        xin = xd * (1 - a) if self.zoh_b else xd
        return (self._scan(a, xin) * self.mix).sum(-1)

class DendStochLayer(nn.Module):
    def __init__(self, d_in, d_out, k, complex_poles, dropout=0.0,
                 zoh_b=False, norm_type="batchnorm", ct_exact=False):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.dend = DendriticSSM(d_out, k, complex_poles, zoh_b, ct_exact)
        self.norm_type = norm_type
        self.norm = (nn.LayerNorm(d_out) if norm_type == "layernorm"
                     else nn.BatchNorm1d(d_out) if norm_type == "batchnorm"
                     else nn.Identity())
        self.theta = nn.Parameter(torch.zeros(d_out))
        self.drop = nn.Dropout(dropout)

    def norm_apply(self, V):
        if self.norm_type == "batchnorm":
            B, T, d = V.shape
            return self.norm(V.reshape(B * T, d)).reshape(B, T, d)
        return self.norm(V)

    def forward(self, s_in, mode, beta):
        V = self.norm_apply(self.dend(self.proj(s_in)))
        p = torch.sigmoid(beta * (V - self.theta))
        p = self.drop(p)
        s = torch.bernoulli(p.clamp(1e-6, 1 - 1e-6)) if mode == "sample" else p
        gf = ((p > 0.2) & (p < 0.8)).float().mean()
        return s, p.mean(), gf

class DendStochAudioNet(nn.Module):
    """Audio variant: no conv-stem, linear projection 700 -> n_hidden."""
    def __init__(self, cfg):
        super().__init__()
        self.readout_pool = cfg.readout_pool
        zb = cfg.zoh_b; nt = cfg.norm_type; ce = cfg.ct_exact
        dims = [cfg.n_in] + [cfg.n_hidden] * cfg.n_layers
        self.layers = nn.ModuleList([
            DendStochLayer(dims[i], dims[i+1], cfg.k_modes, cfg.complex_poles,
                           cfg.hidden_dropout, zb, nt, ce)
            for i in range(cfg.n_layers)])
        self.readout = nn.Linear(cfg.n_hidden, cfg.n_classes)
        self.ro = DendriticSSM(cfg.n_classes, 1, complex_poles=False, zoh_b=zb, ct_exact=ce)

    def pool(self, x):
        return x.mean(dim=1) if self.readout_pool == "mean" else x.max(dim=1).values

    def forward(self, x, mode, beta):
        s = x; rates = []; gfs = []
        for layer in self.layers:
            s, r, gf = layer(s, mode, beta)
            rates.append(r); gfs.append(gf)
        logits = self.pool(self.ro(self.readout(s)))
        return logits, torch.stack(rates).mean(), torch.stack(gfs).mean()

    def set_scan(self, fn):
        for m in self.modules():
            if isinstance(m, DendriticSSM): m._scan = fn

    def set_dt(self, dt):
        for m in self.modules():
            if isinstance(m, DendriticSSM): m.dt = float(dt)
